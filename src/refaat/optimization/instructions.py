from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from refaat.application.budget import budget_scope, prices_from
from refaat.core.guidance import sanitize_demonstration_answer
from refaat.core.text import QUERY_STOPWORDS, tokens
from refaat.evaluation.bundle import MetricBundle, default_bundle
from refaat.evaluation.dataset import Dataset, EvaluationCase
from refaat.evaluation.runner import ProgramEvaluator
from refaat.optimization.candidates import OptimizationResult, build_ranked_result_budgeted
from refaat.programs.program import Program, Tunables
from refaat.programs.registry import BoundProgram

if TYPE_CHECKING:
    from refaat.application.runtime import RefaatProject

# Composable MIPRO-style instruction fragments. A candidate is one task framing
# joined with one constraint framing plus a salient-terms focus clause, so the
# candidate set is a small, deterministic cartesian enumeration seeded by the
# dataset signal rather than a model call.
_TASK_TEMPLATES = (
    "Answer the question using only the retrieved evidence.",
    "Give a direct answer supported entirely by the retrieved evidence.",
)
_CONSTRAINT_TEMPLATES = (
    "Ground every claim in a cited evidence chunk and omit unsupported detail.",
    "Keep the answer to a single concise sentence.",
)

# Hard ceiling on the GEPA reflection depth so the evolutionary loop is finite
# and reproducible regardless of the caller's requested round count.
_MAX_REFLECTION_ROUNDS = 5


def _case_texts(cases: Sequence[EvaluationCase]) -> Iterable[str]:
    for case in cases:
        yield case.query
        yield from case.expected_terms
        if case.reference_answer:
            yield case.reference_answer


def _salient_terms(texts: Iterable[str]) -> list[str]:
    """Rank the dataset's content terms by frequency with a deterministic tie-break.

    Tokenization and the stopword filter predicate are the shipped
    ``core.text.tokens`` and ``core.text.QUERY_STOPWORDS``: a token qualifies
    when its length is at least three and it is not a query stopword. Ties are
    broken by the term itself, so the ordering is fully deterministic.
    """
    counts: dict[str, int] = {}
    for text in texts:
        for token in tokens(text):
            if len(token) < 3 or token in QUERY_STOPWORDS:
                continue
            counts[token] = counts.get(token, 0) + 1
    return [term for term, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def propose_instructions(
    dataset: Dataset,
    *,
    seed_instructions: tuple[str, ...] = (),
    max_candidates: int = 6,
) -> tuple[str | None, ...]:
    """Deterministically enumerate MIPRO-style instruction candidates.

    The first element is ``None``: the unguided baseline that carries no
    instructions. Every non-None member is a composed task-plus-constraint
    framing anchored to the dataset's salient terms, so distinct datasets yield
    distinct proposals. Caller-supplied ``seed_instructions`` are offered first,
    de-duplicated against the generated set. Only the non-None members are later
    sanitized before becoming ``Tunables.instructions``.
    """
    terms = _salient_terms(_case_texts(dataset.cases))
    focus = ", ".join(terms[:3])
    generated: list[str] = []
    for task in _TASK_TEMPLATES:
        for constraint in _CONSTRAINT_TEMPLATES:
            text = f"{task} {constraint}"
            if focus:
                text = f"{text} Focus on: {focus}."
            generated.append(text)
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in (*seed_instructions, *generated):
        stripped = candidate.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            ordered.append(stripped)
    return (None, *ordered[:max_candidates])


def reflect_instruction(instruction: str | None, failures: Sequence[EvaluationCase]) -> str:
    """Breed a GEPA-style mutant by appending a reminder derived from failures.

    Deterministic and idempotent: the reminder is built from the salient terms of
    the failing cases, and if it is already present in the instruction the
    instruction is returned unchanged. Idempotency lets the evolutionary loop
    detect a fixed point and stop, so the search is finite and reproducible.
    """
    base = (instruction or "").strip()
    terms = _salient_terms(_case_texts(failures))
    if not terms:
        return base
    reminder = (
        f"Prior answers were not grounded on {', '.join(terms[:3])}; cite explicit evidence for these before answering."
    )
    if reminder in base:
        return base
    return f"{base} {reminder}" if base else reminder


def _evolve_instructions(failing_cases: Sequence[EvaluationCase], rounds: int) -> tuple[str, ...]:
    """Chain reflective mutations over a growing window of failing cases.

    Each round reflects the previous mutant against one more failing case, so a
    larger failure signal yields a distinct mutant. The loop stops at the round
    cap or when reflection reaches a fixed point, so it always terminates.
    """
    bound = min(rounds, _MAX_REFLECTION_ROUNDS)
    evolved: list[str] = []
    current = ""
    for round_index in range(bound):
        window = failing_cases[: round_index + 1]
        mutant = reflect_instruction(current, window)
        if mutant == current or mutant in evolved:
            break
        evolved.append(mutant)
        current = mutant
    return tuple(evolved)


def _split_cases(dataset: Dataset) -> tuple[tuple[EvaluationCase, ...], tuple[EvaluationCase, ...]]:
    ordered = sorted(dataset.cases, key=lambda case: case.id)
    if len(ordered) < 2:
        raise ValueError("Instruction optimization needs at least two cases to hold out an evaluation split")
    cut = max(1, len(ordered) // 2)
    return tuple(ordered[:cut]), tuple(ordered[cut:])


def _variant_space(program: Program, proposed: Sequence[str | None]) -> list[Tunables]:
    """Build the tunables space with the unguided baseline held at index 0.

    The ``None`` member maps to the program's own (empty-instruction) tunables,
    already present as element 0, so the baseline is never duplicated and its
    artifact id stays byte-identical. Every non-None instruction is sanitized so
    a citation-shaped token can never be injected through an instruction.
    """
    space = [program.tunables]
    for instruction in proposed:
        if instruction is None:
            continue
        space.append(replace(program.tunables, instructions=sanitize_demonstration_answer(instruction)))
    return space


class InstructionOptimizer:
    """Native, model-free optimizer over a program's shipped INSTRUCTION tunable.

    ``optimize_mipro`` enumerates templated instruction candidates seeded by the
    dataset's salient terms; ``optimize_gepa`` reflectively breeds mutant
    instructions from the held-out grounding failures. Both hold out an
    evaluation split, keep the unguided variant at index 0, and rank through the
    shared build-and-rank helper, so the active program is never mutated and the
    winning instruction folds into both the artifact id and the answer-cache
    identity through the shipped guidance digest. On the offline extractive
    generator the instruction does not change the answer, so ranking measures that the
    search machinery ran, not that quality improved.
    """

    def __init__(self, project: RefaatProject) -> None:
        self._project = project

    def optimize_mipro(
        self,
        program: Program,
        dataset: Dataset,
        *,
        bundle: MetricBundle | None = None,
        seed_instructions: tuple[str, ...] = (),
        max_candidates: int = 6,
        budget: int | None = None,
    ) -> OptimizationResult:
        active_bundle = bundle or default_bundle()
        signal_cases, held_out = _split_cases(dataset)
        eval_dataset = Dataset(name=f"{dataset.name}:held-out", cases=held_out)
        signal_dataset = Dataset(name=f"{dataset.name}:signal", cases=signal_cases)
        proposed = propose_instructions(
            signal_dataset, seed_instructions=seed_instructions, max_candidates=max_candidates
        )
        tunables_space: list[Tunables] = _variant_space(program, proposed)
        if budget is not None:
            tunables_space = tunables_space[:budget]
        return build_ranked_result_budgeted(self._project, program, tunables_space, eval_dataset, active_bundle)

    def optimize_gepa(
        self,
        program: Program,
        dataset: Dataset,
        *,
        bundle: MetricBundle | None = None,
        rounds: int = 3,
        budget: int | None = None,
    ) -> OptimizationResult:
        active_bundle = bundle or default_bundle()
        settings = self._project.runtime.settings
        # One shared meter over the initial held-out eval and the candidate ranking,
        # so a configured run budget bounds the whole GEPA search, not just the
        # ranking that build_ranked_result_budgeted scopes. The inner scope shares
        # this meter, so spend accrues against one ceiling.
        with budget_scope(
            settings.run_token_budget,
            settings.run_cost_budget_usd,
            prices=prices_from(settings.model_prices),
        ):
            _, held_out = _split_cases(dataset)
            eval_dataset = Dataset(name=f"{dataset.name}:held-out", cases=held_out)
            report = ProgramEvaluator(BoundProgram(self._project, program)).evaluate(eval_dataset, active_bundle)
            failed_ids = {case.case_id for case in report.cases if not case.passed}
            failing = tuple(case for case in eval_dataset.cases if case.id in failed_ids)
            # Fall back to the full held-out set when nothing fails the hard
            # constraint, so the reflective loop always has a signal to evolve on the
            # offline stack, where the evidence-only generator grounds most cases.
            signal_cases = failing or eval_dataset.cases
            evolved = _evolve_instructions(signal_cases, rounds)
            tunables_space: list[Tunables] = _variant_space(program, (None, *evolved))
            if budget is not None:
                tunables_space = tunables_space[:budget]
            return build_ranked_result_budgeted(self._project, program, tunables_space, eval_dataset, active_bundle)

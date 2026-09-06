from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from refaat.domain.authz import Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import JsonObject, QueryRequest, QueryResponse
from refaat.errors import ValidationError
from refaat.infrastructure.reflection import (
    DeterministicNecessityGate,
    NecessityGate,
    ReflectionDecision,
    is_useful_answer,
)
from refaat.programs.policy import REFINE, BoundedInferencePolicy
from refaat.programs.program import (
    GroundedRagInput,
    GroundedRagOutput,
    Program,
    ProgramResult,
    Tunables,
    _has_pure_execute,
)

if TYPE_CHECKING:
    from refaat.application.runtime import RefaatProject


class _SelfRagAnswer(Program):
    """The inner grounded answer program Self-RAG reflects over and refines.

    Every answer flows through the shared ``_query_request`` into the runtime
    query pipeline, so retrieval, generation, caching, and the citation contract
    are the runtime's, unchanged, and the stream path folds the identical request.
    The necessity decision only raises retrieval breadth; it never reaches the
    generator. The program counts its executions and keeps the last response so
    Self-RAG can report the refine attempt count and rebuild the final result
    under its own identity.
    """

    name = "self-rag"
    input_signature = GroundedRagInput
    output_signature = GroundedRagOutput

    @property
    def shadow_safe(self) -> bool:
        return _has_pure_execute(self, _SelfRagAnswer)

    def declared_side_effect(self) -> SideEffectClass:
        if _has_pure_execute(self, _SelfRagAnswer):
            return SideEffectClass.READ_ONLY
        return super().declared_side_effect()

    def __init__(self, *, tunables: Tunables, query_expansion: bool, max_query_variants: int) -> None:
        super().__init__(tunables=tunables)
        self._query_expansion = query_expansion
        self._max_query_variants = max_query_variants
        self.calls = 0
        self.last_response: QueryResponse | None = None

    def steps(self) -> tuple[str, ...]:
        return ("retrieve", "predict", "verify_grounding")

    def _query_request(
        self,
        project: RefaatProject,
        question: str,
        *,
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> QueryRequest:
        return QueryRequest(
            workspace_id=project.workspace_id,
            corpus_id=project.corpus_id,
            query=question,
            user_id=user_id,
            session_id=session_id,
            top_k=self.tunables.top_k,
            token_budget=self.tunables.token_budget,
            fusion=self.tunables.fusion,
            rrf_k=self.tunables.rrf_k,
            autocut_enabled=self.tunables.autocut_enabled,
            autocut_threshold=self.tunables.autocut_threshold,
            query_expansion=self._query_expansion,
            max_query_variants=self._max_query_variants,
            instructions=self.tunables.instructions,
            demonstrations=self.tunables.demonstrations,
            response_schema=self.response_schema,
            principal=principal,
        )

    def execute(
        self,
        project: RefaatProject,
        *,
        inputs: Mapping[str, Any],
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> ProgramResult:
        question = _require_question(inputs)
        self.calls += 1
        response = project.runtime.query(
            self._query_request(project, question, user_id=user_id, session_id=session_id, principal=principal)
        )
        self.last_response = response
        return self._result(response)


class SelfRagProgram(Program):
    """Self-reflective RAG: retrieve-necessity, IsSupported, and a bounded refine.

    The three Self-RAG reflections map onto shipped machinery. A
    :class:`NecessityGate` decides whether retrieval is needed and, when it is,
    raises retrieval breadth; the answer always flows through the grounded
    runtime query pipeline either way, so no ungrounded, uncited answer is ever
    returned. IsSupported is the shipped :class:`GroundingValidator` over the
    result. When the answer is not supported the retry is the shipped
    :class:`BoundedInferencePolicy` refine (retry until grounded, hard-capped),
    not a reimplemented loop. IsUseful is the deterministic usefulness heuristic.
    The reflections are recorded in the outputs, never reach the generator, and
    are never citeable. The necessity gate identity and the reflection policy are
    folded into the content-addressed artifact.
    """

    name = "self-rag"
    input_signature = GroundedRagInput
    output_signature = GroundedRagOutput

    def __init__(
        self,
        *,
        necessity_gate: NecessityGate | None = None,
        refine_attempt_cap: int = 3,
        usefulness_min_answer_chars: int = 1,
        retrieve_max_query_variants: int = 4,
        tunables: Tunables | None = None,
    ) -> None:
        super().__init__(tunables=tunables)
        if refine_attempt_cap < 1:
            raise ValidationError("refine_attempt_cap must be at least 1")
        if retrieve_max_query_variants < 2:
            raise ValidationError("retrieve_max_query_variants must be at least 2 to broaden retrieval")
        if usefulness_min_answer_chars < 1:
            raise ValidationError("usefulness_min_answer_chars must be at least 1")
        self._necessity_gate = necessity_gate or DeterministicNecessityGate()
        self._refine_attempt_cap = refine_attempt_cap
        self._usefulness_min_answer_chars = usefulness_min_answer_chars
        self._retrieve_max_query_variants = retrieve_max_query_variants

    @property
    def shadow_safe(self) -> bool:
        return _has_pure_execute(self, SelfRagProgram)

    def declared_side_effect(self) -> SideEffectClass:
        if _has_pure_execute(self, SelfRagProgram):
            return SideEffectClass.READ_ONLY
        return super().declared_side_effect()

    def steps(self) -> tuple[str, ...]:
        return ("assess_necessity", "retrieve", "predict", "reflect_support", "refine", "reflect_useful")

    def _definition_extra(self) -> JsonObject:
        extra = dict(super()._definition_extra())
        extra["self_rag"] = {
            "necessity_gate": self._necessity_gate.cache_id,
            "refine_attempt_cap": self._refine_attempt_cap,
            "usefulness_min_answer_chars": self._usefulness_min_answer_chars,
            "retrieve_max_query_variants": self._retrieve_max_query_variants,
        }
        return extra

    def stream(
        self,
        project: RefaatProject,
        *,
        inputs: Mapping[str, Any],
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> Iterator[dict[str, Any]]:
        # execute runs an initial attempt then a bounded refine and selects among
        # them, and no single request reproduces that selection, so refuse streaming
        # rather than answer as one bare attempt with the wrong cache identity.
        raise ValidationError("SelfRagProgram does not support streaming; call run() for the selected result")

    def execute(
        self,
        project: RefaatProject,
        *,
        inputs: Mapping[str, Any],
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> ProgramResult:
        question = _require_question(inputs)
        retrieve = self._necessity_gate.decide(question)

        first_answerer = self._make_answerer(retrieve)
        first = first_answerer.execute(
            project, inputs=inputs, user_id=user_id, session_id=session_id, principal=principal
        )
        if first.grounded:
            winner = _selected_response(first_answerer.last_response)
            refine_attempts = 0
        else:
            refine_answerer = self._make_answerer(retrieve)
            BoundedInferencePolicy(refine_answerer, mode=REFINE, attempts=self._refine_attempt_cap).execute(
                project, inputs=inputs, user_id=user_id, session_id=session_id, principal=principal
            )
            winner = _selected_response(refine_answerer.last_response)
            refine_attempts = refine_answerer.calls

        result = self._result(winner)
        is_supported = result.grounded
        is_useful = is_useful_answer(result.answer, grounded=is_supported, min_length=self._usefulness_min_answer_chars)
        decision = ReflectionDecision(
            retrieve=retrieve,
            is_supported=is_supported,
            is_useful=is_useful,
            rationale=(
                f"necessity[{self._necessity_gate.cache_id}] retrieve={retrieve}; "
                f"supported={is_supported}; useful={is_useful}; refine_attempts={refine_attempts}"
            ),
        )
        return replace(
            result,
            outputs={**result.outputs, "self_rag": _decision_json(decision, refine_attempts)},
        )

    def _make_answerer(self, retrieve: bool) -> _SelfRagAnswer:
        query_expansion = retrieve
        max_query_variants = self._retrieve_max_query_variants if retrieve else 1
        return _SelfRagAnswer(
            tunables=self.tunables,
            query_expansion=query_expansion,
            max_query_variants=max_query_variants,
        )


def _require_question(inputs: Mapping[str, Any]) -> str:
    question = inputs.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValidationError("Program input 'question' is required and must be a non-empty string")
    return question


def _selected_response(response: QueryResponse | None) -> QueryResponse:
    if response is None:
        raise ValidationError("Self-RAG produced no answer to reflect over")
    return response


def _decision_json(decision: ReflectionDecision, refine_attempts: int) -> JsonObject:
    return {
        "retrieve": decision.retrieve,
        "is_supported": decision.is_supported,
        "is_useful": decision.is_useful,
        "rationale": decision.rationale,
        "refine_attempts": refine_attempts,
    }

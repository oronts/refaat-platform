from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from refaat.core.guidance import guidance_digest, sanitize_demonstration_answer
from refaat.core.math import cosine_similarity
from refaat.domain.authz import Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import Demonstration, JsonObject
from refaat.errors import ValidationError
from refaat.ports import Embedder
from refaat.programs.program import Program, ProgramResult, Validator

if TYPE_CHECKING:
    from refaat.application.runtime import RefaatProject


class KNNFewShotProgram(Program):
    """Inference-time few-shot selector around an inner program.

    Holds a labeled demonstration pool and, per query, embeds the query and each
    pool demonstration's question with an injected embedder (defaulting to the
    runtime embedder reachable from the project, so the selector stays
    provider-neutral), picks the ``k`` nearest by cosine similarity in a
    deterministic order, sanitizes the selected demonstrations through
    :mod:`refaat.core.guidance`, and runs the inner program with those
    demonstrations via :meth:`Program.with_tunables`. The selected demonstrations
    flow through the existing ``QueryRequest`` guidance-digest path, so each
    query's selection enters the answer-cache identity. The inner program is
    never mutated (``with_tunables`` returns a clone), and the pool digest, ``k``,
    and the inner artifact id are pinned into this program's content-addressed
    artifact, so distinct pools or ``k`` values are distinct, reproducible
    artifacts.

    Sanitizing before the demonstrations reach the inner program means a real
    corpus chunk id planted in a pool demonstration is rewritten to the reserved
    example sentinel and can never be echoed back as a resolved citation.
    """

    def __init__(
        self,
        inner: Program,
        pool: Sequence[Demonstration],
        *,
        k: int = 3,
        embedder: Embedder | None = None,
    ) -> None:
        if not pool:
            raise ValidationError("KNNFewShotProgram requires a non-empty demonstration pool")
        if k < 1:
            raise ValidationError("k must be at least 1")
        super().__init__(tunables=inner.tunables)
        self._inner = inner
        self._pool = tuple(pool)
        self._k = k
        self._embedder = embedder
        self.name = f"{inner.name}+knn"
        self.input_signature = inner.input_signature
        self.output_signature = inner.output_signature

    @property
    def shadow_safe(self) -> bool:
        # KNN folds demonstrations into the inner program; its side-effect profile is the inner's.
        return self._inner.shadow_safe

    def declared_side_effect(self) -> SideEffectClass:
        return self._inner.declared_side_effect()

    def steps(self) -> tuple[str, ...]:
        return (*self._inner.steps(), "knn")

    def validators(self) -> tuple[Validator, ...]:
        return (*self._inner.validators(), *self._extra_validators)

    def _definition_extra(self) -> JsonObject:
        return {
            "knn": {
                "k": self._k,
                "pool": guidance_digest(None, self._pool),
                "inner": self._inner.artifact().artifact_id,
            }
        }

    def _variant_for(self, project: RefaatProject, question: str) -> tuple[Program, tuple[Demonstration, ...]]:
        embedder = self._embedder or project.runtime.queries.embedder
        selected = self._select(embedder, question)
        # Fold this program's own tunables (which carry any outer selection when a KNN
        # wraps a KNN) into the variant, appending this level's selection, so a nested
        # inner receives the outer demonstrations instead of dropping them. A top-level
        # KNN inherits the inner's tunables with no demonstrations, so the variant is
        # byte-identical to the selection alone.
        demonstrations = (*self.tunables.demonstrations, *selected)
        variant = self._inner.with_tunables(replace(self.tunables, demonstrations=demonstrations))
        return variant, selected

    def stream(
        self,
        project: RefaatProject,
        *,
        inputs: Mapping[str, Any],
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> Iterator[dict[str, Any]]:
        # Build the same per-query variant execute uses, then delegate to its stream so
        # a parity inner streams with the selected demonstrations folded into the request
        # and a multi-result inner reaches its own refusal guard instead of being bypassed.
        question = inputs.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValidationError("Program input 'question' is required and must be a non-empty string")
        variant, _ = self._variant_for(project, question)
        yield from variant.stream(project, inputs=inputs, user_id=user_id, session_id=session_id, principal=principal)

    def execute(
        self,
        project: RefaatProject,
        *,
        inputs: Mapping[str, Any],
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> ProgramResult:
        question = inputs.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValidationError("Program input 'question' is required and must be a non-empty string")
        variant, selected = self._variant_for(project, question)
        result = variant.execute(project, inputs=inputs, user_id=user_id, session_id=session_id, principal=principal)
        return replace(
            result,
            program=self.name,
            artifact_id=self.artifact().artifact_id,
            outputs={
                **result.outputs,
                "knn": {
                    "k": self._k,
                    "pool_size": len(self._pool),
                    "selected": [demonstration.question for demonstration in selected],
                },
            },
        )

    def _select(self, embedder: Embedder, question: str) -> tuple[Demonstration, ...]:
        query_vector = embedder.embed([question])[0]
        pool_vectors = embedder.embed([demonstration.question for demonstration in self._pool])
        scores = [cosine_similarity(query_vector, vector) for vector in pool_vectors]
        order = sorted(range(len(self._pool)), key=lambda index: (-scores[index], index))
        chosen = order[: self._k]
        return tuple(
            Demonstration(
                question=sanitize_demonstration_answer(self._pool[index].question),
                answer=sanitize_demonstration_answer(self._pool[index].answer),
            )
            for index in chosen
        )

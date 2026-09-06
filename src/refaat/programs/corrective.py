from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

from refaat.domain.authz import Principal
from refaat.domain.context_transform import ContextTransform, redact_for_egress
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import ContextPacket, GenerationRequest, JsonObject, QueryRequest
from refaat.errors import DependencyUnavailableError, ValidationError
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
    from refaat.ports import Generator

CORRECT = "correct"
LOW = "low"

_JUDGE_SYSTEM_PROMPT = (
    "You are a retrieval sufficiency judge. Decide whether the retrieved context is sufficient to answer "
    "the question with grounded evidence. Reply with a single word: 'correct' if sufficient, 'low' if not. "
    "No commentary."
)


@dataclass(frozen=True, slots=True)
class ConfidenceAssessment:
    """The confidence gate's verdict over a retrieved context.

    ``score`` and ``top_semantic`` are always the measured per-hit semantic
    signal, never a fabricated value, so a degraded model evaluator can flip the
    label but can never invent a score.
    """

    label: str
    score: float
    top_semantic: float
    grounded_chunks: int
    hit_count: int
    rationale: str


class ConfidenceEvaluator(Protocol):
    @property
    def cache_id(self) -> str: ...

    def assess(
        self,
        query: str,
        context: ContextPacket,
        *,
        context_transform: ContextTransform | None = None,
    ) -> ConfidenceAssessment:
        """Assess retrieval sufficiency. ``context_transform``, when set, redacts the evidence a
        model-backed evaluator sends to its judge model; a model-free evaluator ignores it."""
        ...


@dataclass(frozen=True, slots=True)
class DeterministicConfidenceEvaluator:
    """Model-free confidence gate over the non-normalized per-hit semantic signal.

    It reads only ``SearchHit.semantic_score``, the raw per-hit semantic score,
    never the fused ``score`` (which RRF and linear fusion normalize so the top
    hit saturates to 1.0 regardless of true similarity). ``top_semantic`` is the
    strongest hit's semantic score and ``grounded_chunks`` counts hits at or above
    ``semantic_floor``. The context is labeled ``correct`` only when it has hits,
    its top semantic score meets ``top_semantic_threshold``, and at least
    ``min_grounded_chunks`` clear the floor; otherwise ``low``. Pure, so the gate
    is reproducible offline.
    """

    top_semantic_threshold: float = 0.35
    semantic_floor: float = 0.15
    min_grounded_chunks: int = 1

    @property
    def cache_id(self) -> str:
        return f"crag-deterministic-v1:{self.top_semantic_threshold}:{self.semantic_floor}:{self.min_grounded_chunks}"

    def assess(
        self,
        query: str,
        context: ContextPacket,
        *,
        context_transform: ContextTransform | None = None,
    ) -> ConfidenceAssessment:
        # The deterministic gate does no generation, so it never sends evidence to a model and the
        # egress transform is irrelevant to it.
        del query, context_transform
        hits = context.hits
        hit_count = len(hits)
        top_semantic = max((hit.semantic_score for hit in hits), default=0.0)
        grounded_chunks = sum(1 for hit in hits if hit.semantic_score >= self.semantic_floor)
        correct = (
            hit_count > 0
            and top_semantic >= self.top_semantic_threshold
            and grounded_chunks >= self.min_grounded_chunks
        )
        label = CORRECT if correct else LOW
        rationale = (
            f"top_semantic={top_semantic} vs threshold {self.top_semantic_threshold}, "
            f"grounded_chunks={grounded_chunks} (floor {self.semantic_floor}) vs min {self.min_grounded_chunks}, "
            f"hits={hit_count} -> {label}"
        )
        return ConfidenceAssessment(
            label=label,
            score=top_semantic,
            top_semantic=top_semantic,
            grounded_chunks=grounded_chunks,
            hit_count=hit_count,
            rationale=rationale,
        )


class ModelConfidenceEvaluator:
    """Model-backed confidence gate that degrades to the deterministic label.

    A generator judges whether the retrieved context is sufficient. The model may
    only flip the ``correct`` / ``low`` label; ``score`` and every measured field
    stay the deterministic evaluator's genuine values, so the model never
    fabricates a score. A missing dependency degrades to the deterministic label
    rather than guessing, keeping the offline path honest.
    """

    def __init__(
        self,
        generator: Generator,
        *,
        deterministic: DeterministicConfidenceEvaluator | None = None,
    ) -> None:
        self._generator = generator
        self._deterministic = deterministic or DeterministicConfidenceEvaluator()

    @property
    def cache_id(self) -> str:
        return f"crag-model:{self._generator.model_id}"

    def assess(
        self,
        query: str,
        context: ContextPacket,
        *,
        context_transform: ContextTransform | None = None,
    ) -> ConfidenceAssessment:
        baseline = self._deterministic.assess(query, context)
        request = GenerationRequest(query=query, context=context, system_prompt=_JUDGE_SYSTEM_PROMPT)
        egress = redact_for_egress(context_transform, request)
        try:
            result = self._generator.generate(egress.request)
        except DependencyUnavailableError:
            return replace(
                baseline,
                rationale=f"model judge unavailable, deterministic fallback: {baseline.rationale}",
            )
        verdict = result.answer.strip().casefold()
        label = CORRECT if verdict.startswith(CORRECT) else LOW
        return replace(
            baseline,
            label=label,
            rationale=f"model judge verdict={verdict!r}; measured {baseline.rationale}",
        )


def _assess_supports_transform(evaluator: ConfidenceEvaluator) -> bool:
    """Whether ``evaluator.assess`` accepts the optional ``context_transform`` keyword.

    An external evaluator predating this seam has the old two-argument signature, so the keyword is
    threaded only when the callable actually declares it (or accepts arbitrary keywords), keeping
    that evaluator working unchanged.
    """
    try:
        signature = inspect.signature(evaluator.assess)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters
    if "context_transform" in parameters:
        return True
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


class CorrectiveRagProgram(Program):
    """Corrective RAG: a confidence gate that broadens retrieval on low confidence.

    A narrow retrieval (single query variant, no expansion) feeds a
    :class:`ConfidenceEvaluator`. The gate decision selects the shared
    ``_query_request``: when it says ``correct`` the answer comes from a narrow
    request; when it says ``low`` a corrective request with query expansion
    activates the runtime-configured ``QueryTransform`` (step-back offline), whose
    extra variant is RRF-merged by chunk id. Both execute and the inherited stream
    build that same request, so retrieval breadth is the only lever and
    generation, caching, and the citation contract are untouched. The gate
    decision is recorded in the outputs but never reaches the generator and is
    never citeable. The evaluator identity and the corrective breadth are folded
    into the content-addressed artifact.
    """

    name = "corrective-rag"
    input_signature = GroundedRagInput
    output_signature = GroundedRagOutput

    @property
    def shadow_safe(self) -> bool:
        return _has_pure_execute(self, CorrectiveRagProgram)

    def declared_side_effect(self) -> SideEffectClass:
        if _has_pure_execute(self, CorrectiveRagProgram):
            return SideEffectClass.READ_ONLY
        return super().declared_side_effect()

    def __init__(
        self,
        *,
        evaluator: ConfidenceEvaluator | None = None,
        corrective_max_query_variants: int = 4,
        tunables: Tunables | None = None,
    ) -> None:
        super().__init__(tunables=tunables)
        if corrective_max_query_variants < 2:
            raise ValidationError("corrective_max_query_variants must be at least 2 to broaden retrieval")
        self._evaluator = evaluator or DeterministicConfidenceEvaluator()
        self._corrective_max_query_variants = corrective_max_query_variants

    def steps(self) -> tuple[str, ...]:
        return ("retrieve", "assess", "correct", "predict", "verify_grounding")

    def _definition_extra(self) -> JsonObject:
        extra = dict(super()._definition_extra())
        extra["corrective_rag"] = {
            "evaluator": self._evaluator.cache_id,
            "corrective_max_query_variants": self._corrective_max_query_variants,
        }
        return extra

    def _gate(
        self,
        project: RefaatProject,
        question: str,
        *,
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> ConfidenceAssessment:
        context = project.retrieve(
            question,
            user_id=user_id,
            session_id=session_id,
            top_k=self.tunables.top_k,
            token_budget=self.tunables.token_budget,
            query_expansion=False,
            max_query_variants=1,
            principal=principal,
        )
        # Thread the runtime's egress redactor into a model-backed evaluator so its judge model sees
        # redacted evidence too. Fetched defensively: an evaluator predating the keyword keeps its
        # two-argument call, and with no transform installed the two-argument call is byte-identical.
        transform = getattr(project, "context_transform", None)
        if transform is not None and _assess_supports_transform(self._evaluator):
            return self._evaluator.assess(question, context, context_transform=transform)
        return self._evaluator.assess(question, context)

    def _request_for(
        self,
        project: RefaatProject,
        question: str,
        *,
        corrective_applied: bool,
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
            query_expansion=corrective_applied,
            max_query_variants=self._corrective_max_query_variants if corrective_applied else 1,
            instructions=self.tunables.instructions,
            demonstrations=self.tunables.demonstrations,
            response_schema=self.response_schema,
            principal=principal,
        )

    def _query_request(
        self,
        project: RefaatProject,
        question: str,
        *,
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> QueryRequest:
        """Authoritative corrective answer request shared by execute and stream.

        Reproducing the confidence gate here is what makes the corrective breadth
        hold on the inherited ``Program.stream`` path: the gate decides whether the
        streamed answer expands retrieval, and the same ``query_expansion`` and
        ``max_query_variants`` enter the answer-cache identity as ``execute`` builds.
        """
        assessment = self._gate(project, question, user_id=user_id, session_id=session_id, principal=principal)
        return self._request_for(
            project,
            question,
            corrective_applied=assessment.label == LOW,
            user_id=user_id,
            session_id=session_id,
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
        question = inputs.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValidationError("Program input 'question' is required and must be a non-empty string")

        assessment = self._gate(project, question, user_id=user_id, session_id=session_id, principal=principal)
        corrective_applied = assessment.label == LOW
        response = project.runtime.query(
            self._request_for(
                project,
                question,
                corrective_applied=corrective_applied,
                user_id=user_id,
                session_id=session_id,
                principal=principal,
            )
        )
        result = self._result(response)
        return replace(
            result,
            outputs={**result.outputs, "corrective_rag": self._assessment_json(assessment, corrective_applied)},
        )

    @staticmethod
    def _assessment_json(assessment: ConfidenceAssessment, corrective_applied: bool) -> JsonObject:
        return {
            "label": assessment.label,
            "score": assessment.score,
            "top_semantic": assessment.top_semantic,
            "grounded_chunks": assessment.grounded_chunks,
            "hit_count": assessment.hit_count,
            "corrective_applied": corrective_applied,
            "rationale": assessment.rationale,
        }

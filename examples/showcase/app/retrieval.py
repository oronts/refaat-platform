"""Named retrieval modes over the Oronts knowledge base.

Each method exercises one refaat retrieval feature or a combination and returns the
grounded answer together with its real citations and the routes that fired. A route
is the runtime's own record of which retrieval stage ran, so asserting a route is
present proves the feature actually executed rather than being merely configured.

The query-transform mode reports the runtime's active transform provider. The
entrypoint builds one runtime per provider and calls the same mode on each, which
demonstrates the whole transform catalog through one code path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from refaat import RefaatProject
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import Citation, RlmLimits
from refaat.domain.sandbox import CodePolicy, CodeRequest, SandboxLimits
from refaat.programs import CorrectiveRagProgram, SelfRagProgram


def _code_policy() -> CodePolicy:
    return CodePolicy(
        granted_scopes=("compute",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        max_steps=8,
        limits=SandboxLimits(max_nodes=64, max_output_bytes=4096, max_wall_seconds=2.0),
    )


def _governed_requests() -> tuple[CodeRequest, ...]:
    """Four governed code steps: a computed step, a fed-forward step, an over-ceiling
    denial, and an escape attempt the safe evaluator refuses.
    """
    return (
        CodeRequest(code="a + b", inputs={"a": 2, "b": 3}, required_scopes=("compute",)),
        CodeRequest(code="result_0 * c", inputs={"c": 10}, required_scopes=("compute",)),
        CodeRequest(
            code="a + b",
            inputs={"a": 1, "b": 1},
            required_scopes=("compute",),
            side_effect=SideEffectClass.EXTERNAL,
        ),
        CodeRequest(code="__import__('os')", required_scopes=("compute",)),
    )


def stored_chunk_ids(project: RefaatProject) -> set[str]:
    """The set of real stored chunk ids, enumerated through the public retrieve surface.

    Retrieving each document's title with a large token budget returns every chunk of
    that document, so the union across all documents is the whole corpus. This keeps the
    citation-resolution universe on the public RefaatProject API.
    """
    identifiers: set[str] = set()
    for document in project.list_documents():
        context = project.retrieve(str(document["title"]), top_k=1000, token_budget=1_000_000)
        for hit in context.hits:
            identifiers.add(hit.chunk.id)
    return identifiers


def citations_resolve(citations: tuple[Citation, ...], identifiers: set[str]) -> bool:
    return bool(citations) and all(citation.chunk_id in identifiers for citation in citations)


@dataclass(frozen=True, slots=True)
class ModeResult:
    mode: str
    answer: str
    citations: tuple[str, ...]
    routes: tuple[str, ...]
    grounded: bool
    cached: bool = False
    extra: dict[str, object] = field(default_factory=dict)


class ShowcaseRetrieval:
    def __init__(self, project: RefaatProject) -> None:
        self.project = project
        self._chunk_ids = stored_chunk_ids(project)

    def _resolves(self, citations: tuple[Citation, ...]) -> bool:
        return citations_resolve(citations, self._chunk_ids)

    def plain_rag(self, question: str) -> ModeResult:
        answer = self.project.ask(question, fusion="rrf")
        return ModeResult(
            mode="plain_rag",
            answer=answer.answer,
            citations=tuple(c.chunk_id for c in answer.citations),
            routes=tuple(answer.routes),
            grounded=self._resolves(answer.citations),
            cached=answer.cached,
        )

    def cache_hit(self, question: str) -> ModeResult:
        first = self.project.ask(question, fusion="rrf")
        second = self.project.ask(question, fusion="rrf")
        return ModeResult(
            mode="cache_hit",
            answer=second.answer,
            citations=tuple(c.chunk_id for c in second.citations),
            routes=tuple(second.routes),
            grounded=self._resolves(second.citations),
            cached=second.cached,
            extra={"first_cached": first.cached, "exact": second.cached and second.routes[0] == "cag:exact"},
        )

    def graphrag(self, question: str) -> ModeResult:
        context = self.project.retrieve(question)
        return ModeResult(
            mode="graphrag",
            answer="",
            citations=(),
            routes=tuple(context.routes),
            grounded=bool(context.hits),
            extra={
                "graph_facts": list(context.graph_facts[:6]),
                "graph_route": "graph" in context.routes,
            },
        )

    def kag(self, question: str) -> ModeResult:
        answer = self.project.ask(question)
        return ModeResult(
            mode="kag",
            answer=answer.answer,
            citations=tuple(c.chunk_id for c in answer.citations),
            routes=tuple(answer.routes),
            grounded=self._resolves(answer.citations),
            extra={"graph_route": "graph" in answer.routes},
        )

    def feature_routes(self, question: str) -> ModeResult:
        context = self.project.retrieve(question)
        return ModeResult(
            mode="feature_routes",
            answer="",
            citations=(),
            routes=tuple(context.routes),
            grounded=bool(context.hits),
            extra={
                "hype_route": "hype" in context.routes,
                "raptor_route": "raptor" in context.routes,
                "rerank_route": "rerank" in context.routes,
            },
        )

    def reranked(self, question: str) -> ModeResult:
        context = self.project.retrieve(question)
        return ModeResult(
            mode="reranked",
            answer="",
            citations=(),
            routes=tuple(context.routes),
            grounded=bool(context.hits),
            extra={"rerank_route": "rerank" in context.routes},
        )

    def autocut(self, question: str) -> ModeResult:
        context = self.project.retrieve(question, autocut_enabled=True, autocut_threshold=0.05)
        return ModeResult(
            mode="autocut",
            answer="",
            citations=(),
            routes=tuple(context.routes),
            grounded=bool(context.hits),
            extra={"autocut_route": "autocut" in context.routes},
        )

    def transform(self, question: str, provider: str) -> ModeResult:
        context = self.project.retrieve(question, query_expansion=True, max_query_variants=4)
        return ModeResult(
            mode=f"transform:{provider}",
            answer="",
            citations=(),
            routes=tuple(context.routes),
            grounded=bool(context.hits),
            extra={"provider": provider, "expansion_route": "expansion" in context.routes},
        )

    def corrective_rag(self, question: str) -> ModeResult:
        result = self.project.programs.register(CorrectiveRagProgram()).run(question=question)
        assessment = result.outputs.get("corrective_rag", {})
        return ModeResult(
            mode="corrective_rag",
            answer=result.answer,
            citations=tuple(c.chunk_id for c in result.citations),
            routes=tuple(result.routes),
            grounded=result.grounded and self._resolves(result.citations),
            extra={"assessment_present": bool(assessment), "label": assessment.get("label")},
        )

    def self_rag(self, question: str) -> ModeResult:
        result = self.project.programs.register(SelfRagProgram()).run(question=question)
        reflection = result.outputs.get("self_rag", {})
        return ModeResult(
            mode="self_rag",
            answer=result.answer,
            citations=tuple(c.chunk_id for c in result.citations),
            routes=tuple(result.routes),
            grounded=result.grounded and self._resolves(result.citations),
            extra={
                "reflection_present": bool(reflection),
                "is_supported": reflection.get("is_supported"),
            },
        )

    def rlm(self, question: str) -> ModeResult:
        result = self.project.rlm(question, limits=RlmLimits(max_depth=1, max_nodes=6, branch_factor=3))
        return ModeResult(
            mode="rlm",
            answer=result.answer,
            citations=tuple(c.chunk_id for c in result.citations),
            routes=(),
            grounded=self._resolves(result.citations),
            extra={"nodes": len(result.nodes), "usage": dict(result.usage)},
        )

    def governed_code_rlm(
        self, question: str, code_policy: CodePolicy, code_requests: Sequence[CodeRequest]
    ) -> ModeResult:
        result = self.project.rlm(question, code_policy=code_policy, code_requests=code_requests)
        grounded = self._resolves(result.citations)
        return ModeResult(
            mode="governed_code_rlm",
            answer=result.answer,
            citations=tuple(c.chunk_id for c in result.citations),
            routes=(),
            grounded=grounded,
            extra={
                "steps": list(result.usage["code_steps"]),
                "sandbox": result.usage["code_sandbox"],
                "grounded": grounded,
            },
        )

    def metadata_filtered(self, question: str, category: str) -> ModeResult:
        context = self.project.retrieve(question)
        categories = sorted({str(hit.chunk.metadata.get("category")) for hit in context.hits})
        return ModeResult(
            mode="metadata_filtered",
            answer="",
            citations=(),
            routes=tuple(context.routes),
            grounded=bool(context.hits),
            extra={
                "filter_route": "filter" in context.routes,
                "categories": categories,
                "only_category": bool(context.hits) and categories == [category],
            },
        )


__all__ = [
    "ModeResult",
    "ShowcaseRetrieval",
    "_code_policy",
    "_governed_requests",
    "citations_resolve",
    "stored_chunk_ids",
]

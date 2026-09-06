from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from refaat.core.router import RouterConfig, route_query
from refaat.domain.authz import Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import JsonObject, QueryRequest
from refaat.errors import ValidationError
from refaat.programs.program import GroundedRagProgram, Program, ProgramResult, Tunables

if TYPE_CHECKING:
    from refaat.application.runtime import RefaatProject


class RouterProgram(Program):
    """Per-query router over an inner grounded program.

    Deterministic query features (length, clause count, an entity proxy) select a
    retrieval strategy through the model-free :func:`route_query` table, and the
    inner program answers through the runtime with that strategy layered on top of
    its own tunables. Routing is offline-deterministic, so there is no
    honesty-boundary caveat: the router only re-shapes retrieval, never grounding
    or citations.
    """

    name = "router"

    def __init__(self, inner: Program | None = None, *, tunables: Tunables | None = None) -> None:
        resolved_inner = inner or GroundedRagProgram()
        super().__init__(tunables=tunables if tunables is not None else resolved_inner.tunables)
        self._inner = resolved_inner
        self.input_signature = resolved_inner.input_signature
        self.output_signature = resolved_inner.output_signature

    @property
    def shadow_safe(self) -> bool:
        # The router layers a retrieval strategy over the inner program; its side-effect profile is the inner's.
        return self._inner.shadow_safe

    def declared_side_effect(self) -> SideEffectClass:
        return self._inner.declared_side_effect()

    def steps(self) -> tuple[str, ...]:
        return ("route", *self._inner.steps())

    def _config(self, project: RefaatProject) -> RouterConfig:
        if self.tunables.router is not None:
            return RouterConfig.from_json(self.tunables.router)
        settings = project.runtime.settings
        return RouterConfig(
            long_query_tokens=settings.router_long_query_tokens,
            multi_clause_threshold=settings.router_multi_clause_threshold,
            entity_threshold=settings.router_entity_threshold,
        )

    def _definition_extra(self) -> JsonObject:
        extra = dict(super()._definition_extra())
        config = RouterConfig.from_json(self.tunables.router)
        extra["router"] = {
            "kind": "router",
            "inner": self._inner.artifact().artifact_id,
            "config": config.cache_id,
        }
        return extra

    def _query_request(
        self,
        project: RefaatProject,
        question: str,
        *,
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> QueryRequest:
        """Authoritative routed answer request shared by execute and stream.

        Overriding the one request-builder seam is what makes routing hold on
        BOTH paths: the inherited ``Program.stream`` builds its request through
        this method too, so a streamed router run is routed and folds
        ``router_id`` into the answer-cache identity exactly as ``execute`` does,
        instead of the base builder's unrouted request that isolated nothing.
        """
        config = self._config(project)
        strategy = route_query(question, config)
        inner = self.tunables
        return QueryRequest(
            workspace_id=project.workspace_id,
            corpus_id=project.corpus_id,
            query=question,
            user_id=user_id,
            session_id=session_id,
            top_k=strategy.top_k if strategy.top_k is not None else inner.top_k,
            token_budget=inner.token_budget,
            fusion=strategy.fusion,
            rrf_k=inner.rrf_k,
            autocut_enabled=strategy.autocut_enabled,
            autocut_threshold=inner.autocut_threshold,
            query_expansion=strategy.query_expansion,
            instructions=inner.instructions,
            demonstrations=inner.demonstrations,
            response_schema=self.response_schema,
            router_id=config.cache_id,
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
        response = project.runtime.query(
            self._query_request(
                question=question, project=project, user_id=user_id, session_id=session_id, principal=principal
            )
        )
        return self._result(response)

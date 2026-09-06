"""Composition root for the showcase runtime.

``build_runtime`` is the single place that turns a :class:`ShowcaseConfig` into a
live :class:`Refaat`. Every retrieval feature the showcase demonstrates is wired
here through settings and builder injection points: the co-occurrence graph, the
lexical reranker, HyPE and RAPTOR at ingestion, contextual retrieval, the query
transform catalog, RRF fusion with autocut, compression, metadata filtering, the
answer cache, graph answers, and the bounded recursive method. Each degrades to
the deterministic default when the offline stack is active.

Later waves attach their own surfaces on top of this seam (a FastAPI factory, the
program and agent layers, an MCP server, a UI). They receive the same runtime.
"""

from __future__ import annotations

from refaat import Refaat, RuntimeBuilder
from refaat.ports import CodeSandbox, CrossEncoderScorer, Skill

from .config import ShowcaseConfig


def build_runtime(
    config: ShowcaseConfig,
    *,
    cross_encoder_scorer: CrossEncoderScorer | None = None,
    code_sandbox: CodeSandbox | None = None,
    extra_skills: tuple[Skill, ...] = (),
) -> Refaat:
    builder = RuntimeBuilder(config.settings)
    # A cross-encoder reranker activates only when both the provider is selected and a
    # scorer is injected; the lexical reranker remains the offline-safe default.
    if cross_encoder_scorer is not None:
        builder.with_cross_encoder_scorer(cross_encoder_scorer)
    # The bounded RLM executes caller-supplied code only when an operator installs a
    # sandbox and a policy admits the request. With no sandbox the deny-by-default
    # boundary stands, so every existing caller stays byte-identical.
    if code_sandbox is not None:
        builder.with_code_sandbox(code_sandbox)
    for skill in extra_skills:
        builder.with_skill(skill)
    return builder.build()


__all__ = ["build_runtime"]

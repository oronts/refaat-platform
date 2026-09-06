"""Coverage guard for generation-port egress classification.

This AST-enumerates every ``.generate`` / ``.stream`` / ``.stream_tool_call`` call site in
``src/refaat`` and asserts the set matches a pinned, annotated classification. It guards the
GENERATION-PORT egress surface only: which call sites send corpus evidence to a generation model,
and therefore which ones the pre-model redaction seam must cover. It is NOT a guard over every
provider HTTP egress. The embedding adapter and the fine-tuning provider reach their providers
through ``post_with_retry`` (not a ``.generate`` / ``.stream`` method call), so they are out of this
scan by construction, and the redaction control does not cover them (see docs/egress-redaction.md).

A new ``.generate`` / ``.stream`` / ``.stream_tool_call`` call site, or an extra occurrence of an
existing one, changes the collected set and fails this test, forcing a maintainer to classify it and
decide whether it is a generation-model evidence egress that must be redacted.
"""

from __future__ import annotations

import ast
import pathlib

_METHODS = frozenset({"generate", "stream", "stream_tool_call"})
_SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src"
_SCAN_ROOT = _SRC_ROOT / "refaat"

# The honest taxonomy. Each category states what the call site is and why the generation-evidence
# redaction seam does or does not apply to it.
_CATEGORIES = {
    # The generation-model corpus-evidence egress this control redacts. Each site sends retrieved
    # evidence to a real generator and is wired to redact_for_egress upstream of the call.
    "covered-generation",
    # The in-process ExtractiveGenerator fallback. Deliberately NOT redacted: it must ground in and
    # cite the real stored evidence, so it always receives the original request.
    "local-fallback",
    # A query-only model egress that carries no corpus evidence to redact (a pre-retrieval query
    # rewrite, a Self-RAG necessity gate over an empty context, the tool planner).
    "empty-context",
    # A model egress OUTSIDE this control's scope by design: an ingestion-time or evaluation-time
    # provider call. Documented in docs/egress-redaction.md as a separate provider egress to handle
    # with a trusted, self-hosted, or zero-retention provider.
    "separate-egress",
    # A caching wrapper delegating to the inner generator. The request it forwards was already
    # redacted upstream, so no redaction happens (or is needed) at the delegation call.
    "model-cache-passthrough",
    # A circuit-breaker wrapper delegating to the inner generator. It only brackets the call to fast-fail
    # a wedged endpoint; the request it forwards was already redacted upstream, so it is a passthrough.
    "circuit-breaker-passthrough",
    # A program stream delegating to an inner program's stream. The inner program's covered-generation
    # egress is where redaction runs; this call only forwards events.
    "program-orchestration",
    # The httpx transport stream inside a generator adapter. The request bytes were already built from
    # the redacted request, so this is transport, not a redaction seam.
    "adapter-transport",
    # An httpx client stream from the SDK to a Refaat server. Client-side HTTP, not a model egress.
    "sdk-client",
    # A non-model httpx stream fetching a URL for ingestion. Not a model egress.
    "connector",
    # A governed plugin's own outbound HTTP through the OutboundClient. Not a model egress; it carries
    # no corpus evidence, and it is guarded by the shared egress allowlist, SSRF pin, and audit seam.
    "plugin-outbound",
}

# The pinned classification: "relpath::receiver.method" -> (occurrence_count, category).
# Every current call site is enumerated so the guard passes on first run and any change is caught.
_PINNED: dict[str, tuple[int, str]] = {
    "refaat/api/routes.py::bound.stream": (1, "program-orchestration"),
    "refaat/application/rlm.py::self.generator.generate": (1, "covered-generation"),
    # Two local-fallback sites: the citation-miss fallback and the degradation serve-degraded path.
    "refaat/application/rlm.py::self.fallback.generate": (2, "local-fallback"),
    "refaat/application/services.py::self.generator.stream": (2, "covered-generation"),
    "refaat/application/services.py::self.generator.generate": (1, "covered-generation"),
    # Three local-fallback sites: the citation-miss fallback and the two degradation paths
    # (non-streaming outage handler and the streamed degraded events).
    "refaat/application/services.py::self.fallback.generate": (3, "local-fallback"),
    "refaat/infrastructure/connectors.py::self.client.stream": (1, "connector"),
    "refaat/infrastructure/generation.py::self.client.stream": (2, "adapter-transport"),
    "refaat/infrastructure/hype.py::self._generator.generate": (1, "separate-egress"),
    "refaat/infrastructure/judging.py::self._generator.generate": (1, "separate-egress"),
    "refaat/infrastructure/model_cache.py::self._inner.generate": (2, "model-cache-passthrough"),
    "refaat/infrastructure/model_cache.py::self._inner.stream": (1, "model-cache-passthrough"),
    "refaat/infrastructure/resilience.py::self._inner.generate": (1, "circuit-breaker-passthrough"),
    "refaat/infrastructure/resilience.py::self._inner.stream": (1, "circuit-breaker-passthrough"),
    "refaat/infrastructure/multimodal.py::self.client.stream": (1, "adapter-transport"),
    "refaat/infrastructure/outbound.py::self._client.stream": (1, "plugin-outbound"),
    "refaat/infrastructure/query_transform.py::generator.generate": (1, "empty-context"),
    "refaat/infrastructure/reflection.py::self._generator.generate": (1, "empty-context"),
    "refaat/infrastructure/summarization.py::self._generator.generate": (1, "separate-egress"),
    "refaat/programs/corrective.py::self._generator.generate": (1, "covered-generation"),
    "refaat/programs/knn.py::variant.stream": (1, "program-orchestration"),
    "refaat/programs/registry.py::self.program.stream": (1, "program-orchestration"),
    "refaat/programs/toolcall.py::self._generator.stream_tool_call": (1, "empty-context"),
    "refaat/sdk/async_client.py::self.client.stream": (2, "sdk-client"),
    "refaat/sdk/client.py::self.client.stream": (2, "sdk-client"),
}


def _collect() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(_SCAN_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(_SRC_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in _METHODS:
                receiver = ast.unparse(func.value)
                key = f"{relative}::{receiver}.{func.attr}"
                counts[key] = counts.get(key, 0) + 1
    return counts


def test_every_pinned_category_is_a_known_taxonomy_entry() -> None:
    for key, (_count, category) in _PINNED.items():
        assert category in _CATEGORIES, f"{key} has an unknown category {category!r}"


def test_generation_egress_call_sites_match_the_pinned_classification() -> None:
    collected = _collect()
    pinned_counts = {key: count for key, (count, _category) in _PINNED.items()}
    new_sites = sorted(set(collected) - set(pinned_counts))
    removed_sites = sorted(set(pinned_counts) - set(collected))
    assert not new_sites, (
        "New generation-port egress call site(s) are unclassified. Classify each in _PINNED and, "
        f"if it sends corpus evidence to a generation model, wire redact_for_egress: {new_sites}"
    )
    assert not removed_sites, f"Pinned egress call site(s) no longer exist; update _PINNED: {removed_sites}"
    assert collected == pinned_counts


def test_covered_generation_sites_are_exactly_the_redacted_egresses() -> None:
    # The generation-model corpus-evidence egresses this control redacts. If a site becomes
    # covered-generation, redact_for_egress must wrap it; this pins the closed set.
    covered = {key for key, (_count, category) in _PINNED.items() if category == "covered-generation"}
    assert covered == {
        "refaat/application/services.py::self.generator.generate",
        "refaat/application/services.py::self.generator.stream",
        "refaat/application/rlm.py::self.generator.generate",
        "refaat/programs/corrective.py::self._generator.generate",
    }

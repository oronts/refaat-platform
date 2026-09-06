"""Data-classification model routing gates generation before it reaches the model.

The router derives the data classification deterministically from the retrieved
evidence ACL tags (never from a model) and denies generation when the evidence
exceeds a ceiling. A denial returns a governed no-generation response on both the
non-streaming and streaming paths. With no router installed the pipeline is
byte-identical to the unrouted default, which the wider suite proves.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from refaat.application.runtime import RefaatProject, RuntimeBuilder
from refaat.application.services import ROUTING_REFUSAL_MESSAGE, ROUTING_REFUSAL_REASON, ROUTING_ROUTE
from refaat.config import Settings
from refaat.domain.authz import Classification
from refaat.domain.models import DocumentInput, QueryRequest
from refaat.domain.routing import (
    ClassificationCeilingRouter,
    ProviderResidency,
    ResidencyPolicy,
    ResidencyRouter,
    ResidencyRule,
    RetentionTerm,
)

pytestmark = pytest.mark.integration

QUERY = "checkout production release rollback plan"
EVIDENCE_TERM = "rollback"

DOCUMENTS: dict[str, dict[str, object]] = {
    "public.md": {"acl": {"classification": "public"}, "doc": "public"},
    "internal.md": {"acl": {"classification": "internal"}, "doc": "internal"},
    "secret.md": {"acl": {"classification": "confidential"}, "doc": "secret"},
    # Present-but-unparseable ACL blocks: an unknown classification name, a
    # non-mapping acl, and a malformed allow_groups (a bare string). Each must fail
    # closed to maximum sensitivity, never default to PUBLIC.
    "unknown-name.md": {"acl": {"classification": "top_secret"}, "doc": "codename bluejay"},
    "non-mapping.md": {"acl": "restricted", "doc": "codename bluejay"},
    "bad-groups.md": {"acl": {"classification": "restricted", "allow_groups": "admins"}, "doc": "codename bluejay"},
}


def _ingest(project: RefaatProject, source_ids: tuple[str, ...]) -> None:
    for source_id in source_ids:
        metadata = DOCUMENTS[source_id]
        content = f"The {metadata['doc']} checkout production release requires an approved rollback plan."
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=source_id,
                title=source_id,
                media_type="text/markdown",
                content=content.encode(),
                metadata=dict(metadata),
            ),
        )


def _routed_project(settings: Settings, corpus: str, source_ids: tuple[str, ...]) -> RefaatProject:
    runtime = (
        RuntimeBuilder(settings).with_model_router(ClassificationCeilingRouter(ceiling=Classification.INTERNAL)).build()
    )
    project = runtime.project(workspace="acme", corpus=corpus)
    _ingest(project, source_ids)
    return project


@pytest.fixture
def confidential_project(settings: Settings) -> Iterator[RefaatProject]:
    project = _routed_project(settings, "confidential", ("secret.md",))
    try:
        yield project
    finally:
        project.runtime.close()


@pytest.fixture
def internal_project(settings: Settings) -> Iterator[RefaatProject]:
    project = _routed_project(settings, "internal", ("public.md", "internal.md"))
    try:
        yield project
    finally:
        project.runtime.close()


def test_over_classified_evidence_is_refused_without_generation(confidential_project: RefaatProject) -> None:
    response = confidential_project.ask(QUERY)
    assert ROUTING_ROUTE in response.routes
    assert response.answer == ROUTING_REFUSAL_MESSAGE
    assert response.citations == ()
    assert EVIDENCE_TERM not in response.answer.casefold()
    assert "secret" not in response.answer.casefold()
    assert response.abstained is False


def test_within_ceiling_evidence_is_answered_with_citations(internal_project: RefaatProject) -> None:
    response = internal_project.ask(QUERY)
    assert ROUTING_ROUTE not in response.routes
    assert response.citations
    assert response.answer != ROUTING_REFUSAL_MESSAGE


def test_streaming_path_also_refuses_over_classified_evidence(confidential_project: RefaatProject) -> None:
    request = QueryRequest(
        workspace_id=confidential_project.workspace_id,
        corpus_id=confidential_project.corpus_id,
        query=QUERY,
    )
    events = list(confidential_project.runtime.stream_query(request))
    final = next(event for event in events if event["type"] == "final")
    assert ROUTING_ROUTE in final["routes"]
    assert final["answer"] == ROUTING_REFUSAL_MESSAGE
    assert final["citations"] == []
    deltas = "".join(event["text"] for event in events if event["type"] == "delta")
    assert deltas == ROUTING_REFUSAL_MESSAGE
    assert EVIDENCE_TERM not in deltas.casefold()


MALFORMED_ACL_SOURCES = ("unknown-name.md", "non-mapping.md", "bad-groups.md")


@pytest.mark.parametrize("source_id", MALFORMED_ACL_SOURCES)
def test_present_but_malformed_acl_fails_closed_on_query(settings: Settings, source_id: str) -> None:
    runtime = (
        RuntimeBuilder(settings).with_model_router(ClassificationCeilingRouter(ceiling=Classification.PUBLIC)).build()
    )
    try:
        project = runtime.project(workspace="acme", corpus=f"malformed-query-{source_id}")
        _ingest(project, (source_id,))
        response = project.ask(QUERY)
        assert ROUTING_ROUTE in response.routes
        assert response.answer == ROUTING_REFUSAL_MESSAGE
        assert response.citations == ()
        assert "bluejay" not in response.answer.casefold()
    finally:
        runtime.close()


@pytest.mark.parametrize("source_id", MALFORMED_ACL_SOURCES)
def test_present_but_malformed_acl_fails_closed_on_stream(settings: Settings, source_id: str) -> None:
    runtime = (
        RuntimeBuilder(settings).with_model_router(ClassificationCeilingRouter(ceiling=Classification.PUBLIC)).build()
    )
    try:
        project = runtime.project(workspace="acme", corpus=f"malformed-stream-{source_id}")
        _ingest(project, (source_id,))
        request = QueryRequest(
            workspace_id=project.workspace_id,
            corpus_id=project.corpus_id,
            query=QUERY,
        )
        events = list(project.runtime.stream_query(request))
        final = next(event for event in events if event["type"] == "final")
        assert ROUTING_ROUTE in final["routes"]
        assert final["answer"] == ROUTING_REFUSAL_MESSAGE
        assert final["citations"] == []
        deltas = "".join(event["text"] for event in events if event["type"] == "delta")
        assert "bluejay" not in deltas.casefold()
    finally:
        runtime.close()


def test_router_identity_scopes_the_answer_cache(settings: Settings) -> None:
    # A permissive runtime caches a confidential-derived answer; a stricter runtime
    # over the same database, workspace, corpus, and query must not serve that cached
    # answer but re-route and refuse, because the router identity scopes the cache.
    permissive = RuntimeBuilder(settings).build()
    try:
        project = permissive.project(workspace="acme", corpus="shared")
        _ingest(project, ("secret.md",))
        cached = project.ask(QUERY)
        assert cached.citations
        assert cached.answer != ROUTING_REFUSAL_MESSAGE
    finally:
        permissive.close()

    strict = (
        RuntimeBuilder(settings).with_model_router(ClassificationCeilingRouter(ceiling=Classification.PUBLIC)).build()
    )
    try:
        project = strict.project(workspace="acme", corpus="shared")
        response = project.ask(QUERY)
        assert ROUTING_ROUTE in response.routes
        assert response.answer == ROUTING_REFUSAL_MESSAGE
        assert response.citations == ()
        assert response.cached is False
    finally:
        strict.close()


def test_no_router_installed_answers_confidential_evidence_normally(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).build()
    try:
        project = runtime.project(workspace="acme", corpus="unrouted")
        _ingest(project, ("secret.md",))
        response = project.ask(QUERY)
        assert ROUTING_ROUTE not in response.routes
        assert response.citations
        assert response.answer != ROUTING_REFUSAL_MESSAGE
    finally:
        runtime.close()


def _deny_confidential_residency() -> ResidencyRouter:
    # CONFIDENTIAL requires an eu provider with at most zero retention; the active provider is
    # tagged us with retained retention, so CONFIDENTIAL evidence is refused.
    policy = ResidencyPolicy(
        {
            Classification.PUBLIC: ResidencyRule(
                allowed_regions=frozenset({"eu", "us"}), max_retention=RetentionTerm.RETAINED
            ),
            Classification.CONFIDENTIAL: ResidencyRule(
                allowed_regions=frozenset({"eu"}), max_retention=RetentionTerm.ZERO
            ),
        }
    )
    provider = ProviderResidency(provider="gen", region="us", retention=RetentionTerm.RETAINED)
    return ResidencyRouter(policy, provider)


def _no_detail_leak(text: str) -> None:
    lowered = text.casefold()
    assert "region_not_allowed" not in lowered
    assert "retention_exceeds_max" not in lowered
    assert "confidential" not in lowered


def test_residency_router_denies_query_over_confidential_evidence(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).with_model_router(_deny_confidential_residency()).build()
    try:
        project = runtime.project(workspace="acme", corpus="residency-query")
        _ingest(project, ("secret.md",))
        response = project.ask(QUERY)
        assert ROUTING_ROUTE in response.routes
        assert response.answer == ROUTING_REFUSAL_MESSAGE
        assert response.citations == ()
        assert response.usage.get("reason") == ROUTING_REFUSAL_REASON
        # The generic caller-facing reason is returned; no region or retention detail leaks.
        _no_detail_leak(response.answer + str(response.usage))
    finally:
        runtime.close()


def test_residency_router_refuses_rlm_over_confidential_evidence(settings: Settings) -> None:
    # The load-bearing fix: the model router gates the RLM generation egress, not only query.
    runtime = RuntimeBuilder(settings).with_model_router(_deny_confidential_residency()).build()
    try:
        project = runtime.project(workspace="acme", corpus="residency-rlm")
        _ingest(project, ("secret.md",))
        result = project.rlm(QUERY)
        assert result.answer == ROUTING_REFUSAL_MESSAGE
        assert result.citations == ()
        assert result.usage.get("reason") == ROUTING_REFUSAL_REASON
        assert EVIDENCE_TERM not in result.answer.casefold()
        _no_detail_leak(result.answer + str(result.usage))
    finally:
        runtime.close()


def _zone_gated_residency() -> ResidencyRouter:
    # Region and retention pass for CONFIDENTIAL, so the residency ZONE is the deciding factor.
    policy = ResidencyPolicy(
        {
            Classification.CONFIDENTIAL: ResidencyRule(
                allowed_regions=frozenset({"eu"}),
                max_retention=RetentionTerm.ZERO,
                allowed_zones=frozenset({"eu"}),
            )
        }
    )
    provider = ProviderResidency(provider="gen", region="eu", retention=RetentionTerm.ZERO)
    return ResidencyRouter(policy, provider)


def _ingest_zoned(project: RefaatProject, source_id: str, acl: dict[str, object]) -> None:
    content = f"The {source_id} checkout production release requires an approved rollback plan."
    project.runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id=source_id,
            title=source_id,
            media_type="text/markdown",
            content=content.encode(),
            metadata={"acl": acl},
        ),
    )


def test_residency_zone_gate_admits_in_zone_and_denies_out_of_zone_and_untagged(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).with_model_router(_zone_gated_residency()).build()
    try:
        eu = runtime.project(workspace="acme", corpus="zone-eu")
        _ingest_zoned(eu, "eu.md", {"classification": "confidential", "residency_zone": "eu"})
        assert eu.ask(QUERY).answer != ROUTING_REFUSAL_MESSAGE

        us = runtime.project(workspace="acme", corpus="zone-us")
        _ingest_zoned(us, "us.md", {"classification": "confidential", "residency_zone": "us"})
        us_resp = us.ask(QUERY)
        assert us_resp.answer == ROUTING_REFUSAL_MESSAGE
        assert us_resp.usage.get("reason") == ROUTING_REFUSAL_REASON
        assert "zone_not_allowed" not in (us_resp.answer + str(us_resp.usage))
        assert us.rlm(QUERY).answer == ROUTING_REFUSAL_MESSAGE

        untagged = runtime.project(workspace="acme", corpus="zone-untagged")
        _ingest_zoned(untagged, "plain.md", {"classification": "confidential"})
        assert untagged.ask(QUERY).answer == ROUTING_REFUSAL_MESSAGE
    finally:
        runtime.close()


def test_no_router_installed_rlm_answers_confidential_evidence_normally(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).build()
    try:
        project = runtime.project(workspace="acme", corpus="rlm-unrouted")
        _ingest(project, ("secret.md",))
        result = project.rlm(QUERY)
        assert result.answer != ROUTING_REFUSAL_MESSAGE
        assert result.citations
    finally:
        runtime.close()


def test_ceiling_router_now_gates_rlm(settings: Settings) -> None:
    # The pre-existing RLM egress hole is closed for the shipped ceiling router too.
    runtime = (
        RuntimeBuilder(settings).with_model_router(ClassificationCeilingRouter(ceiling=Classification.INTERNAL)).build()
    )
    try:
        project = runtime.project(workspace="acme", corpus="ceiling-rlm")
        _ingest(project, ("secret.md",))
        result = project.rlm(QUERY)
        assert result.answer == ROUTING_REFUSAL_MESSAGE
        assert result.citations == ()
        assert EVIDENCE_TERM not in result.answer.casefold()
    finally:
        runtime.close()


def test_residency_config_auto_installs_router(settings: Settings) -> None:
    configured = settings.model_copy(
        update={
            "residency_policy": {"confidential": {"allowed_regions": ["eu"], "max_retention": "zero"}},
            "residency_provider_region": "us",
            "residency_provider_retention": "retained",
        }
    )
    runtime = RuntimeBuilder(configured).build()
    try:
        project = runtime.project(workspace="acme", corpus="residency-config")
        _ingest(project, ("secret.md",))
        response = project.ask(QUERY)
        assert ROUTING_ROUTE in response.routes
        assert response.answer == ROUTING_REFUSAL_MESSAGE
        assert response.citations == ()
    finally:
        runtime.close()

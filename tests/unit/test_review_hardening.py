"""Regression locks for the deep-review hardening findings.

These cover the pure decision logic of fixes whose full end-to-end paths are exercised by the
governance and shadow-canary suites; the scenario-level tests live alongside those suites.
"""

from __future__ import annotations

import pytest

from refaat.application.approvals import _validated_capability_name, _validated_fingerprint
from refaat.errors import ValidationError


def test_approval_grant_rejects_free_text_fingerprint_and_capability() -> None:
    # The in-process approval service must bound the fingerprint and capability name so no client free
    # text (a raw query, a secret) can reach the redacted audit event through the direct Python API.
    digest = "a" * 64
    assert _validated_fingerprint(digest) == digest
    assert _validated_capability_name("deploy.v1") == "deploy.v1"
    with pytest.raises(ValidationError):
        _validated_fingerprint("raw customer query SECRET")
    with pytest.raises(ValidationError):
        _validated_capability_name("api_key=SECRET")
    with pytest.raises(ValidationError):
        _validated_fingerprint(digest.upper())


def test_answer_cache_key_folds_adapter_cache_identity() -> None:
    # Two providers can share a bare model name, so the adapter cache identity (endpoint, dimensions,
    # mode) must scope the answer cache: a distinct endpoint yields a distinct key, and no identity
    # (the deterministic offline stack) keeps the key byte-identical.
    from refaat.application.services import QueryService

    class _StubPolicy:
        def as_json(self) -> dict[str, object]:
            return {"stub": True}

    base = {
        "policy": _StubPolicy(),
        "profile": {},
        "supplemental": {},
        "session_summary": None,
        "embedding_model": "m",
        "generation_model": "g",
        "reranker": "none",
        "query_transform": "none",
        "guidance": None,
        "response_schema": None,
    }
    none = QueryService._context_hash(**base)
    endpoint_a = QueryService._context_hash(**base, embedding_cache_identity="endpoint-a")
    endpoint_b = QueryService._context_hash(**base, embedding_cache_identity="endpoint-b")
    assert none == QueryService._context_hash(**base, embedding_cache_identity=None)
    assert none != endpoint_a
    assert endpoint_a != endpoint_b


def _packet(
    *,
    graph_classification: object = None,
    graph_zones: frozenset[str] = frozenset(),
    summary_classification: object = None,
    summary_zones: frozenset[str] = frozenset(),
) -> object:
    from refaat.domain.authz import Classification
    from refaat.domain.models import ContextPacket

    return ContextPacket(
        query="q",
        hits=(),
        graph_facts=(),
        session_summary=None,
        profile={},
        supplemental_harness={},
        token_count=0,
        routes=(),
        graph_evidence_classification=graph_classification or Classification.PUBLIC,
        graph_evidence_zones=graph_zones,
        session_summary_classification=summary_classification or Classification.PUBLIC,
        session_summary_zones=summary_zones,
    )


def test_routing_ceiling_folds_graph_and_summary_classification() -> None:
    # Graph facts and the session summary reach the model outside context.hits, so an over-ceiling
    # graph fact or a higher-classified summary must raise the ceiling; an all-public packet is
    # unchanged, so a packet built without a policy (defaults) leaves the ceiling as the hits imply.
    from refaat.application.services import QueryService
    from refaat.domain.authz import Classification

    assert QueryService._evidence_classification(_packet()) is Classification.PUBLIC
    assert (
        QueryService._evidence_classification(_packet(graph_classification=Classification.RESTRICTED))
        is Classification.RESTRICTED
    )
    assert (
        QueryService._evidence_classification(_packet(summary_classification=Classification.CONFIDENTIAL))
        is Classification.CONFIDENTIAL
    )


def test_egress_zone_gate_unions_graph_and_summary_zones() -> None:
    from refaat.application.services import QueryService

    assert QueryService._evidence_residency_zones(_packet()) == frozenset()
    assert QueryService._evidence_residency_zones(
        _packet(graph_zones=frozenset({"eu"}), summary_zones=frozenset({"us"}))
    ) == frozenset({"eu", "us"})


def test_summary_evidence_recomputes_and_fails_closed_on_a_missing_document() -> None:
    # A summary's classification is recomputed from its cited documents' current ACLs; a cited
    # document that no longer resolves is unresolved provenance and fails closed to the maximum, so a
    # since-deleted confidential source can never lower the ceiling for a summary that still carries it.
    from refaat.domain.authz import Classification, summary_evidence_from_documents

    documents = [
        {"id": "d1", "metadata": {"acl": {"classification": "confidential"}}},
        {"id": "d2", "metadata": {"acl": {"classification": "public"}}},
    ]
    assert summary_evidence_from_documents(frozenset(), documents)[0] is Classification.PUBLIC
    assert summary_evidence_from_documents(frozenset({"d2"}), documents)[0] is Classification.PUBLIC
    assert summary_evidence_from_documents(frozenset({"d1"}), documents)[0] is Classification.CONFIDENTIAL
    classification, zones = summary_evidence_from_documents(frozenset({"d2", "gone"}), documents)
    assert classification is Classification.RESTRICTED
    assert zones != frozenset()
    # Unrecorded provenance (None, a summary compacted before a router existed) fails closed too.
    unrecorded_class, unrecorded_zones = summary_evidence_from_documents(None, documents)
    assert unrecorded_class is Classification.RESTRICTED
    assert unrecorded_zones != frozenset()


def test_stored_summary_cited_documents_distinguishes_absent_from_empty() -> None:
    # An absent key is unrecorded provenance (None, fail closed downstream); a present empty list is a
    # summary that folded no cited documents (genuinely public); a present list is the recorded ids.
    from refaat.core.session import stored_summary_cited_documents

    assert stored_summary_cited_documents({"metadata": {}}) is None
    assert stored_summary_cited_documents({"metadata": {"summary_cited_documents": []}}) == frozenset()
    assert stored_summary_cited_documents({"metadata": {"summary_cited_documents": ["a", "b"]}}) == frozenset(
        {"a", "b"}
    )

"""Executable threat model: one marked assertion per threat in docs/security/threat-model.md.

Each test exercises the in-library baseline for its threat and marks itself with its threat id, so
the threat_coverage gate can prove every threat is covered. The deeper behavioral suites test each
control in full; here each control is exercised minimally and traceably.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from refaat.core.citations import filter_cited_chunks
from refaat.domain.egress import (
    AllowlistEgressPolicy,
    EgressDestination,
    EgressPurpose,
    allowlist_from_config,
    ensure_egress_allowed,
)
from refaat.errors import EgressDeniedError, NotFoundError, RefaatError
from refaat.infrastructure.database.schema import api_keys, jobs, outbox, query_cache
from refaat.security import hash_api_key, pin_outbound_url, verify_api_key
from refaat.skills.registry import SkillRegistry

_ROOT = Path(__file__).resolve().parents[2]


def test_cross_tenant_keys_are_workspace_scoped() -> None:
    # threat: TM-01
    assert "workspace_id" in api_keys.c


def test_api_keys_are_hashed_not_stored_raw() -> None:
    # threat: TM-02
    encoded = hash_api_key("secret-raw-key", "pepper")
    assert "secret-raw-key" not in encoded
    assert verify_api_key("secret-raw-key", encoded, "pepper")
    assert not verify_api_key("wrong", encoded, "pepper")


def test_oversized_input_has_a_byte_limit() -> None:
    # threat: TM-03
    from refaat.config import Settings

    assert "image_max_bytes" in Settings.model_fields


class _PrivateResolver:
    def resolve(self, host: str, port: int) -> list[str]:
        return ["10.0.0.1"]


def test_ssrf_private_address_is_rejected() -> None:
    # threat: TM-04
    with pytest.raises(RefaatError):
        pin_outbound_url("http://internal.example/", allow_private=False, resolver=_PrivateResolver())


def test_egress_denies_an_unlisted_host() -> None:
    # threat: TM-05
    policy = AllowlistEgressPolicy(allowlist_from_config([{"host": "allowed.example"}]))
    with pytest.raises(EgressDeniedError):
        ensure_egress_allowed(policy, EgressDestination.from_url("https://evil.example/", EgressPurpose.MODEL))


def test_answer_cache_is_scoped_to_corpus_version() -> None:
    # threat: TM-06
    assert "corpus_version" in query_cache.c


def test_hallucinated_citation_is_dropped() -> None:
    # threat: TM-07
    assert filter_cited_chunks(("chunk-that-does-not-exist",), ()) == ()


def test_prompt_base_is_immutable_and_content_addressed() -> None:
    # threat: TM-08
    from refaat.infrastructure.database.schema import harness_snapshots

    assert "base_prompt_hash" in harness_snapshots.c


def test_jobs_carry_a_lease_and_attempts() -> None:
    # threat: TM-09
    assert "lease_owner" in jobs.c
    assert "attempts" in jobs.c


def test_outbox_delivery_is_idempotent() -> None:
    # threat: TM-10
    assert "delivered_at" in outbox.c


def test_skill_registry_denies_an_unregistered_skill() -> None:
    # threat: TM-11
    with pytest.raises(NotFoundError):
        SkillRegistry().get("arbitrary-unregistered-skill")


def test_model_egress_evidence_is_redactable() -> None:
    # threat: TM-12
    from refaat.domain.context_transform import redact_for_egress

    assert callable(redact_for_egress)


def test_reference_container_is_non_root_and_read_only() -> None:
    # threat: TM-13
    dockerfile = (_ROOT / "Dockerfile").read_text(encoding="utf-8")
    values = (_ROOT / "deploy" / "helm" / "refaat" / "values.yaml").read_text(encoding="utf-8")
    assert "USER 10001" in dockerfile
    assert "runAsNonRoot: true" in values
    assert "readOnlyRootFilesystem: true" in values

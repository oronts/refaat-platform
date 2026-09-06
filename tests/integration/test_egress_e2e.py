"""End-to-end egress governance: byte-identity when unconfigured, fail-closed at every seam.

Byte-identity: with no egress policy the runtime installs None, the pinned grounded-rag artifact id
and default context hash recompute unchanged, and the status plane reports the new egress_policy key
NOT_CONFIGURED while the existing egress (redaction) key is untouched.

Fail-closed: an allowlisted destination passes and an un-allowlisted one is denied at each outbound
seam (model generation, embedding, graph, fine-tuning, webhook, connector, S3 endpoint), the
connector deny fires before pin_outbound_url resolves (the resolver spy stays untouched), an injected
client cannot bypass the explicit seam check, and a policy-read failure denies.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings
from refaat.domain.egress import (
    AllowlistEgressPolicy,
    EgressDecision,
    EgressDestination,
    allowlist_from_config,
)
from refaat.domain.status import SubsystemState
from refaat.errors import EgressDeniedError
from refaat.infrastructure.blob import S3BlobStore
from refaat.infrastructure.connectors import HttpConnector
from refaat.infrastructure.embeddings import OpenAICompatibleEmbedder
from refaat.infrastructure.events import WebhookEventSink
from refaat.infrastructure.fine_tuning import OpenAIFineTuningProvider
from refaat.infrastructure.generation import OpenAICompatibleGenerator
from refaat.infrastructure.graph import OpenAICompatibleGraphExtractor
from refaat.programs.program import GroundedRagProgram

pytestmark = pytest.mark.integration

PINNED_GROUNDED_RAG_ID = "sha256:b76d73f8d956bcd38ccc9d4c24372b07b387647a189f3a9e300cded186febbb4"
PINNED_CONTEXT_HASH = "7937b529c3dadd9ada1e5483bac13fcef58f71fcee4aa31da688b7d137df034c"


def _policy(*entries: dict[str, object]) -> AllowlistEgressPolicy:
    return AllowlistEgressPolicy(allowlist_from_config(entries))


class _RaisingEgressPolicy:
    cache_id: str | None = "raising"

    def decide(self, dest: EgressDestination) -> EgressDecision:
        raise RuntimeError("policy store down")


class _SpyResolver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def resolve(self, host: str, port: int) -> tuple[str, ...]:
        self.calls.append((host, port))
        return ("93.184.216.34",)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "environment": "test",
        "database_url": f"sqlite:///{tmp_path / 'refaat.db'}",
        "auto_create_schema": True,
        "blob_root": tmp_path / "blobs",
        "connector_file_root": tmp_path / "connectors",
        "skill_root": tmp_path / "skills",
        "api_key_pepper": "test-pepper-with-enough-entropy",
        "bootstrap_token": "bootstrap-test-token",
        "trusted_hosts": ["testserver", "localhost", "127.0.0.1"],
        "rate_limit_requests": 10_000,
        "log_json": False,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# Byte-identity ------------------------------------------------------------------------------------


def test_unconfigured_is_byte_identical_and_status_additive(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).build()
    try:
        assert runtime.egress_policy is None
        ws = runtime.create_workspace("acme", "Acme")
        subs = {sub.name: sub for sub in runtime.status(workspace_id=ws).subsystems}
        assert subs["egress_policy"].state is SubsystemState.NOT_CONFIGURED
        # The pre-existing egress (redaction) subsystem is untouched.
        assert subs["egress"].state is SubsystemState.NOT_CONFIGURED
        assert GroundedRagProgram().artifact().artifact_id == PINNED_GROUNDED_RAG_ID
    finally:
        runtime.close()


def test_default_context_hash_unchanged() -> None:
    from tests.integration.test_admin_status_e2e import _default_context_hash

    assert _default_context_hash() == PINNED_CONTEXT_HASH


def test_installed_policy_reports_active(tmp_path: Path) -> None:
    settings = _settings(tmp_path, egress_policy="allowlist", egress_allowlist=[{"host": "api.openai.com"}])
    runtime = RuntimeBuilder(settings).build()
    try:
        assert isinstance(runtime.egress_policy, AllowlistEgressPolicy)
        ws = runtime.create_workspace("acme", "Acme")
        subs = {sub.name: sub for sub in runtime.status(workspace_id=ws, operator=True).subsystems}
        assert subs["egress_policy"].state is SubsystemState.ACTIVE
        assert subs["egress_policy"].detail["cache_id"] == runtime.egress_policy.cache_id
    finally:
        runtime.close()


# Model seams (fixed destination, checked at construction) -----------------------------------------


def test_model_generation_seam_allows_listed_denies_unlisted() -> None:
    ok = OpenAICompatibleGenerator(
        base_url="https://api.openai.com/v1", api_key="k", model="m", egress_policy=_policy({"host": "api.openai.com"})
    )
    ok.close()
    with pytest.raises(EgressDeniedError):
        OpenAICompatibleGenerator(
            base_url="https://evil.test/v1", api_key="k", model="m", egress_policy=_policy({"host": "api.openai.com"})
        )


def test_embedding_and_graph_and_fine_tuning_seams_deny_unlisted() -> None:
    policy = _policy({"host": "api.openai.com"})
    with pytest.raises(EgressDeniedError):
        OpenAICompatibleEmbedder(
            base_url="https://evil.test/v1", api_key="k", model="m", dimensions=8, egress_policy=policy
        )
    with pytest.raises(EgressDeniedError):
        OpenAICompatibleGraphExtractor(base_url="https://evil.test/v1", api_key="k", model="m", egress_policy=policy)
    with pytest.raises(EgressDeniedError):
        OpenAIFineTuningProvider(base_url="https://evil.test/v1", api_key="k", egress_policy=policy)


def test_a_policy_read_failure_denies_the_model_seam() -> None:
    with pytest.raises(EgressDeniedError) as excinfo:
        OpenAICompatibleGenerator(
            base_url="https://api.openai.com/v1", api_key="k", model="m", egress_policy=_RaisingEgressPolicy()
        )
    assert excinfo.value.reason_code == "policy_unavailable"


# Webhook seam -------------------------------------------------------------------------------------


def test_webhook_seam_allows_listed_denies_unlisted() -> None:
    ok = WebhookEventSink(
        "https://hooks.test/in",
        "secret",
        allow_private_hosts=True,
        egress_policy=_policy({"host": "hooks.test"}),
    )
    ok.close()
    with pytest.raises(EgressDeniedError):
        WebhookEventSink(
            "https://evil.test/in",
            "secret",
            allow_private_hosts=True,
            egress_policy=_policy({"host": "hooks.test"}),
        )


# Connector seam (dynamic destination, checked before the resolve) ---------------------------------


def test_connector_denies_before_the_resolver_runs() -> None:
    spy = _SpyResolver()
    connector = HttpConnector(
        max_bytes=100,
        allow_private_hosts=False,
        resolver=spy,
        egress_policy=_policy({"host": "allowed.test"}),
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x"))),
    )
    try:
        with pytest.raises(EgressDeniedError):
            connector.fetch("https://blocked.test/data")
        # The deny fired before pin_outbound_url could resolve, so the resolver never ran: no partial
        # egress and no resolver leak.
        assert spy.calls == []
        # An allowlisted host passes the check and then resolves and fetches.
        content, _, _ = connector.fetch("https://allowed.test/data")
        assert content == b"x"
        assert spy.calls == [("allowed.test", 443)]
    finally:
        connector.close()


def test_an_injected_client_cannot_bypass_the_connector_check() -> None:
    # Even with an injected client (no guard transport), the explicit seam check denies.
    connector = HttpConnector(
        max_bytes=100,
        allow_private_hosts=True,
        egress_policy=_policy({"host": "allowed.test"}),
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x"))),
    )
    try:
        with pytest.raises(EgressDeniedError):
            connector.fetch("https://blocked.test/data")
    finally:
        connector.close()


def test_connector_is_byte_identical_when_no_policy_is_installed() -> None:
    # No policy: the destination is never reconstructed and the connector behaves exactly as before.
    connector = HttpConnector(
        max_bytes=100,
        allow_private_hosts=True,
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x"))),
    )
    try:
        content, _, _ = connector.fetch("https://anything.test/data")
        assert content == b"x"
    finally:
        connector.close()


# S3 endpoint seam (boto3, endpoint host checked at construction) ----------------------------------


def test_s3_endpoint_seam_denies_an_unlisted_endpoint() -> None:
    pytest.importorskip("boto3")
    with pytest.raises(EgressDeniedError):
        S3BlobStore(
            bucket="b",
            endpoint_url="https://evil.test",
            region="eu-central-1",
            access_key_id=None,
            secret_access_key=None,
            egress_policy=_policy({"host": "s3.eu-central-1.amazonaws.com"}),
        )


# Config startup coherence -------------------------------------------------------------------------


def test_startup_fails_when_a_configured_model_host_is_off_the_allowlist(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not permit"):
        _settings(
            tmp_path,
            generation_provider="openai_compatible",
            generation_model="gpt-4o-mini",
            openai_api_key="k",
            egress_policy="allowlist",
            egress_allowlist=[{"host": "other.test"}],
        ).validate_for_startup()


def test_startup_allows_a_configured_model_host_on_the_allowlist(tmp_path: Path) -> None:
    _settings(
        tmp_path,
        generation_provider="openai_compatible",
        generation_model="gpt-4o-mini",
        openai_api_key="k",
        egress_policy="allowlist",
        egress_allowlist=[{"host": "api.openai.com"}],
    ).validate_for_startup()


def test_startup_requires_the_policy_selector_when_an_allowlist_is_configured(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="REFAAT_EGRESS_POLICY"):
        _settings(tmp_path, egress_allowlist=[{"host": "api.openai.com"}]).validate_for_startup()

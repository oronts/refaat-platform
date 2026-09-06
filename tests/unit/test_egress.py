"""Unit tests for the pure egress decision core.

Cover the fail-closed destination parse, the allowlist matching semantics (exact host, leading-dot
suffix with its boundary, default-port normalization, explicit-port sensitivity, scheme and purpose
scoping, lowercased rule hosts), the chokepoint (no-op when unconfigured, deny raises, a raising
policy converts to policy_unavailable), and startup validation.
"""

from __future__ import annotations

import pytest

from refaat.domain.egress import (
    AllowlistEgressPolicy,
    EgressDecision,
    EgressDestination,
    EgressPurpose,
    NullEgressPolicy,
    allowlist_from_config,
    ensure_egress_allowed,
)
from refaat.errors import ConfigurationError, EgressDeniedError


def _policy(*entries: dict[str, object]) -> AllowlistEgressPolicy:
    return AllowlistEgressPolicy(allowlist_from_config(entries))


def test_from_url_denies_a_malformed_destination() -> None:
    with pytest.raises(EgressDeniedError) as excinfo:
        EgressDestination.from_url("not-a-url", EgressPurpose.MODEL)
    assert excinfo.value.reason_code == "malformed_destination"
    with pytest.raises(EgressDeniedError):
        EgressDestination.from_url("ftp://host/x", EgressPurpose.CONNECTOR)


def test_from_url_denies_an_out_of_range_port() -> None:
    for url in ("https://allowed.test:99999/x", "https://allowed.test:-1/x"):
        with pytest.raises(EgressDeniedError) as excinfo:
            EgressDestination.from_url(url, EgressPurpose.MODEL)
        assert excinfo.value.reason_code == "malformed_destination"


def test_exact_host_allows_and_unlisted_denies() -> None:
    policy = _policy({"host": "api.openai.com"})
    allow = policy.decide(EgressDestination.from_url("https://api.openai.com/v1", EgressPurpose.MODEL))
    assert allow.allowed
    assert allow.reason_code == "ok"
    deny = policy.decide(EgressDestination.from_url("https://evil.test/v1", EgressPurpose.MODEL))
    assert not deny.allowed
    assert deny.reason_code == "not_allowlisted"


def test_rule_host_is_lowercased() -> None:
    policy = _policy({"host": "API.OpenAI.Com"})
    assert policy.decide(EgressDestination.from_url("https://api.openai.com/v1", EgressPurpose.MODEL)).allowed


def test_rule_host_normalization_lowercases_not_casefolds() -> None:
    # Case folding the eszett host would store "strasse.com" and admit that distinct registrable
    # domain while locking out the operator's real host; lowercasing keeps them separate, so the real
    # host is allowed and the unlisted ascii-fold host is denied. This locks the closed fail-open.
    policy = _policy({"host": "straße.com"})
    assert policy.decide(EgressDestination.from_url("https://straße.com/x", EgressPurpose.MODEL)).allowed
    assert not policy.decide(EgressDestination.from_url("https://strasse.com/x", EgressPurpose.MODEL)).allowed


def test_leading_dot_suffix_matches_subdomains_and_apex_but_not_the_boundary() -> None:
    policy = _policy({"host": ".example.com"})
    assert policy.decide(EgressDestination.from_url("https://example.com/x", EgressPurpose.MODEL)).allowed
    assert policy.decide(EgressDestination.from_url("https://a.example.com/x", EgressPurpose.MODEL)).allowed
    # The boundary case a naive endswith would wrongly admit.
    assert not policy.decide(EgressDestination.from_url("https://evilexample.com/x", EgressPurpose.MODEL)).allowed


def test_default_port_normalizes_and_explicit_port_is_enforced() -> None:
    policy = _policy({"host": "api.openai.com", "port": 443})
    assert policy.decide(EgressDestination.from_url("https://api.openai.com/v1", EgressPurpose.MODEL)).allowed
    assert not policy.decide(EgressDestination.from_url("https://api.openai.com:8443/v1", EgressPurpose.MODEL)).allowed


def test_scheme_and_purpose_scoping() -> None:
    policy = _policy({"host": "api.openai.com", "schemes": ["https"], "purposes": ["model"]})
    assert policy.decide(EgressDestination.from_url("https://api.openai.com/v1", EgressPurpose.MODEL)).allowed
    assert not policy.decide(EgressDestination.from_url("http://api.openai.com/v1", EgressPurpose.MODEL)).allowed
    assert not policy.decide(EgressDestination.from_url("https://api.openai.com/v1", EgressPurpose.CONNECTOR)).allowed


def test_allowlist_from_config_rejects_malformed_entries() -> None:
    with pytest.raises(ConfigurationError):
        allowlist_from_config([{"host": ""}])
    with pytest.raises(ConfigurationError):
        allowlist_from_config([{"host": "h", "port": "not-int"}])
    with pytest.raises(ConfigurationError):
        allowlist_from_config([{"host": "h", "purposes": ["nonsense"]}])
    with pytest.raises(ConfigurationError):
        allowlist_from_config([{"host": "h", "schemes": ["ftp"]}])
    with pytest.raises(ConfigurationError):
        allowlist_from_config([{"host": "h", "unknown_key": 1}])
    with pytest.raises(ConfigurationError):
        allowlist_from_config(["not-a-mapping"])


def test_cache_id_folds_only_when_active_and_is_stable_by_content() -> None:
    assert NullEgressPolicy().cache_id is None
    a = _policy({"host": "a.test"}, {"host": "b.test"})
    b = _policy({"host": "b.test"}, {"host": "a.test"})
    assert a.cache_id is not None
    assert a.cache_id == b.cache_id
    assert _policy({"host": "c.test"}).cache_id != a.cache_id


def test_ensure_egress_allowed_is_a_noop_when_unconfigured() -> None:
    ensure_egress_allowed(None, EgressDestination("https", "anything.test", None, EgressPurpose.MODEL))


def test_ensure_egress_allowed_raises_on_deny() -> None:
    policy = _policy({"host": "allowed.test"})
    with pytest.raises(EgressDeniedError) as excinfo:
        ensure_egress_allowed(policy, EgressDestination.from_url("https://blocked.test/x", EgressPurpose.MODEL))
    assert excinfo.value.reason_code == "not_allowlisted"
    assert excinfo.value.host == "blocked.test"
    # Redacted message: only the machine code, never a URL path.
    assert "not_allowlisted" in str(excinfo.value)
    assert "/x" not in str(excinfo.value)


def test_a_raising_policy_denies_as_policy_unavailable() -> None:
    class Raising:
        cache_id: str | None = "raising"

        def decide(self, dest: EgressDestination) -> EgressDecision:
            raise RuntimeError("policy store down")

    with pytest.raises(EgressDeniedError) as excinfo:
        ensure_egress_allowed(Raising(), EgressDestination("https", "h.test", None, EgressPurpose.MODEL))
    assert excinfo.value.reason_code == "policy_unavailable"

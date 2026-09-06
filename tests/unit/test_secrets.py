"""SecretValue is redacted by construction and the reference providers fail closed on a blank value.

Every serialization, log, and pickle surface masks or raises; only an explicit reveal returns the
value, and scrub zeroes it. A skill can only ever see the names its own capability declared.
"""

from __future__ import annotations

import json
import pickle
import traceback

import pytest

from refaat.application.secrets import SecretResolver, _SanitizedSkillError, sanitize_secret_exception
from refaat.canonical import canonical_json
from refaat.clock import SystemClock
from refaat.domain.enums import SideEffectClass
from refaat.domain.governance import ToolCapability
from refaat.domain.secrets import SecretBundle, SecretScope, SecretValue
from refaat.infrastructure.secrets import EnvSecretProvider, StaticSecretProvider
from refaat.observability import redact_field_value


def test_secret_value_masks_str_repr_and_format() -> None:
    secret = SecretValue("hunter2", name="api_key")
    assert str(secret) == "[secret]"
    assert f"{secret}" == "[secret]"
    assert f"{secret:>20}" == "[secret]"
    assert "hunter2" not in repr(secret)
    assert "[secret]" in repr(secret)
    assert "api_key" in repr(secret)


def test_secret_value_is_not_json_serializable() -> None:
    secret = SecretValue("hunter2", name="api_key")
    with pytest.raises(TypeError):
        json.dumps(secret)
    with pytest.raises(TypeError):
        canonical_json({"credential": secret})


def test_secret_value_collapses_through_the_observability_redactor() -> None:
    secret = SecretValue("hunter2", name="api_key")
    assert redact_field_value("anything", secret) == "[redacted:SecretValue]"


def test_secret_value_is_not_picklable() -> None:
    with pytest.raises(TypeError):
        pickle.dumps(SecretValue("hunter2", name="api_key"))


def test_reveal_returns_the_value_and_scrub_zeroes_it() -> None:
    secret = SecretValue("hunter2", name="api_key")
    assert secret.reveal() == "hunter2"
    secret.scrub()
    assert secret.reveal() == ""


def test_fingerprint_is_stable_and_value_sensitive() -> None:
    same = SecretValue("hunter2", name="api_key").fingerprint()
    assert same == SecretValue("hunter2", name="api_key").fingerprint()
    assert same != SecretValue("hunter3", name="api_key").fingerprint()
    assert same != SecretValue("hunter2", name="other").fingerprint()
    assert "hunter2" not in same


def _scope() -> SecretScope:
    return SecretScope(workspace_id="ws", capability_name="push", side_effect=SideEffectClass.WRITE)


def test_static_provider_available_and_resolve() -> None:
    provider = StaticSecretProvider({"api_key": "hunter2", "blank": ""})
    assert provider.provider_id == "static"
    assert provider.available(frozenset({"api_key", "missing", "blank"}), scope=_scope()) == frozenset({"api_key"})
    resolved = provider.resolve("api_key", scope=_scope())
    assert resolved is not None
    assert resolved.reveal() == "hunter2"
    assert provider.resolve("missing", scope=_scope()) is None
    # An empty value is unresolvable so a blank credential fails closed rather than injecting "".
    assert provider.resolve("blank", scope=_scope()) is None


def test_env_provider_reads_prefixed_uppercased_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REFAAT_TOOL_SECRET_DEPLOY_KEY", "s3cr3t")
    provider = EnvSecretProvider()
    names = frozenset({"deploy-key", "absent"})
    assert provider.available(names, scope=_scope()) == frozenset({"deploy-key"})
    resolved = provider.resolve("deploy-key", scope=_scope())
    assert resolved is not None
    assert resolved.reveal() == "s3cr3t"
    assert provider.resolve("absent", scope=_scope()) is None


def test_bundle_exposes_only_its_names_and_scrubs() -> None:
    bundle = SecretBundle({"api_key": SecretValue("hunter2", name="api_key")})
    assert bundle.names == frozenset({"api_key"})
    assert bundle.reveal("api_key") == "hunter2"
    assert bundle.get("missing") is None
    bundle.scrub()
    assert bundle.reveal("api_key") == ""


def test_bundle_is_not_picklable() -> None:
    with pytest.raises(TypeError):
        pickle.dumps(SecretBundle({"api_key": SecretValue("hunter2", name="api_key")}))


def _format(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def test_sanitize_returns_a_controlled_wrapper_carrying_no_skill_bytes() -> None:
    values = frozenset({"ARG-CANARY"})
    sanitized = sanitize_secret_exception(RuntimeError("boom ARG-CANARY"), values)
    assert isinstance(sanitized, _SanitizedSkillError)
    assert "ARG-CANARY" not in _format(sanitized)
    assert "boom" not in str(sanitized)


def test_sanitize_never_touches_a_hostile_exceptions_surfaces() -> None:
    class _Hell(RuntimeError):
        def __init__(self) -> None:
            super().__init__("HELL-CANARY")
            self.str_calls = 0

        def __str__(self) -> str:
            self.str_calls += 1
            raise KeyboardInterrupt("HELL-CANARY str")

        @property
        def __notes__(self) -> list[str]:
            raise KeyboardInterrupt("HELL-CANARY notes")

    exc = _Hell()
    sanitized = sanitize_secret_exception(exc, frozenset({"HELL-CANARY"}))
    assert isinstance(sanitized, _SanitizedSkillError)
    assert exc.str_calls == 0
    assert "HELL-CANARY" not in _format(sanitized)


def test_sanitize_masks_a_split_value() -> None:
    exc = RuntimeError("SPLIT-")
    exc.add_note("CANARY")
    sanitized = sanitize_secret_exception(exc, frozenset({"SPLIT-CANARY"}))
    text = _format(sanitized)
    assert "SPLIT-CANARY" not in text
    assert "SPLIT-" not in text
    assert "CANARY" not in text


def test_sanitize_scrubs_a_non_string_note_whose_str_reveals() -> None:
    class _EvilNote:
        def __str__(self) -> str:
            return "NON-STRING-CANARY"

    exc = RuntimeError("boom")
    exc.__notes__ = [_EvilNote()]  # type: ignore[list-item]
    sanitized = sanitize_secret_exception(exc, frozenset({"NON-STRING-CANARY"}))
    assert "NON-STRING-CANARY" not in _format(sanitized)


def test_sanitize_defeats_a_stateful_str() -> None:
    class _Flip(RuntimeError):
        def __init__(self) -> None:
            super().__init__()
            self._n = 0

        def __str__(self) -> str:
            self._n += 1
            return "FLIP-CANARY" if self._n % 2 else "[secret]"

    sanitized = sanitize_secret_exception(_Flip(), frozenset({"FLIP-CANARY"}))
    for _ in range(5):
        assert "FLIP-CANARY" not in _format(sanitized)
        assert "FLIP-CANARY" not in str(sanitized)


def test_sanitize_ignores_a_hostile_derive() -> None:
    calls = {"n": 0}

    class _Hostile(ExceptionGroup):  # type: ignore[type-arg]
        def derive(self, excs: object) -> _Hostile:
            calls["n"] += 1
            del excs
            return self

    group = _Hostile("boom", [RuntimeError("member DERIVE-CANARY")])
    sanitized = sanitize_secret_exception(group, frozenset({"DERIVE-CANARY"}))
    assert isinstance(sanitized, _SanitizedSkillError)
    assert calls["n"] == 0
    assert "DERIVE-CANARY" not in _format(sanitized)


def test_sanitize_scrubs_a_mixed_base_exception_group() -> None:
    group = BaseExceptionGroup("boom", [RuntimeError("member MIXED-CANARY"), KeyboardInterrupt()])
    sanitized = sanitize_secret_exception(group, frozenset({"MIXED-CANARY"}))
    assert isinstance(sanitized, _SanitizedSkillError)
    assert "MIXED-CANARY" not in _format(sanitized)


def test_sanitize_scrubs_a_group_deeper_than_a_bounded_walk() -> None:
    exc: BaseException = RuntimeError("deep DEEP-CANARY")
    for _ in range(20):
        exc = ExceptionGroup("layer", [exc])
    sanitized = sanitize_secret_exception(exc, frozenset({"DEEP-CANARY"}))
    assert "DEEP-CANARY" not in _format(sanitized)


def test_redacting_formatter_contains_a_hostile_str() -> None:
    from refaat.observability import _redacted_exception_text

    class _HostileStr(RuntimeError):
        def __str__(self) -> str:
            raise KeyboardInterrupt("HOSTILE-CANARY")

    text = _redacted_exception_text((_HostileStr, _HostileStr(), None))
    assert isinstance(text, str)
    assert "HOSTILE-CANARY" not in text


def test_redacting_formatter_does_not_leak_a_hostile_group() -> None:
    from refaat.observability import _redacted_exception_text

    class _WideGroup(BaseExceptionGroup):  # type: ignore[type-arg]
        @property
        def exceptions(self) -> object:
            return (RuntimeError("DOS-CANARY") for _ in range(3000))

    exc = _WideGroup("boom", [RuntimeError("seed")])
    text = _redacted_exception_text((type(exc), exc, None))
    assert "DOS-CANARY" not in text


def test_materialize_scrubs_a_value_whose_reveal_raises() -> None:
    class _ExplodingSecret(SecretValue):
        def __init__(self, value: str, *, name: str) -> None:
            super().__init__(value, name=name)
            object.__setattr__(self, "scrubbed", False)

        def reveal(self) -> str:
            raise RuntimeError("reveal boom")

        def scrub(self) -> None:
            object.__setattr__(self, "scrubbed", True)
            super().scrub()

    holder: dict[str, _ExplodingSecret] = {}

    class _ExplodingProvider:
        @property
        def provider_id(self) -> str:
            return "exploding"

        def available(self, names: frozenset[str], *, scope: SecretScope) -> frozenset[str]:
            return names

        def resolve(self, name: str, *, scope: SecretScope) -> SecretValue | None:
            value = _ExplodingSecret("REVEAL-CANARY", name=name)
            holder[name] = value
            return value

    resolver = SecretResolver(_ExplodingProvider(), "ws", clock=SystemClock())
    capability = ToolCapability(
        name="t",
        required_scopes=frozenset({"write:corpus"}),
        side_effect=SideEffectClass.EXTERNAL,
        required_secrets=frozenset({"api_key"}),
    )
    scope = SecretScope("ws", "t", SideEffectClass.EXTERNAL)
    with pytest.raises(RuntimeError):
        resolver.materialize(capability, scope)
    assert holder["api_key"].scrubbed is True

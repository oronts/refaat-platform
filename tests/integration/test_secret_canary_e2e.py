"""A planted secret reaches the side effect and no governance surface.

The canary value is revealed into a fake outbound sink by a governed tool, yet it never appears in
the recorded steps, the program artifact, the grounded answer or citations, any emitted audit event,
or any captured log line. The audit trail records the secret by name and provider fingerprint only.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Iterator
from uuid import uuid4

import pytest

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.application.secrets import SecretResolver
from refaat.canonical import canonical_json
from refaat.clock import SystemClock
from refaat.config import Settings
from refaat.core.text import stable_digest
from refaat.domain.audit import AuditEvent
from refaat.domain.authz import Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.governance import ToolCapability
from refaat.domain.models import ContextPacket, DocumentInput, JsonObject, ReActPolicy, ToolGrant
from refaat.domain.secrets import SecretBundle, SecretScope, SecretValue
from refaat.infrastructure.secrets import StaticSecretProvider
from refaat.observability import JsonFormatter
from refaat.programs import GovernedReActProgram, KeywordToolPlanner


def _redacting_exc_log(raised: BaseException) -> str:
    """Render the raised exception through the opt-in redacting log formatter (exc_info channel)."""
    record = logging.LogRecord(
        "t", logging.ERROR, __file__, 0, "boom", None, (type(raised), raised, raised.__traceback__)
    )
    return JsonFormatter(redact_exception_text=True).format(record)


pytestmark = pytest.mark.integration

ALICE = Principal(id="alice", tenant="acme")
QUESTION = "how does rollback work"
SECRET_NAME = "api_key"
CONTENT = b"Rollback restores the previously approved harness snapshot before deployment."


class _CanarySkill:
    name = "push"
    description = "Governed tool that reveals the planted secret to a fake outbound sink."
    wants_secrets = True

    def __init__(self) -> None:
        self.sink: list[str] = []

    def run(self, context: ContextPacket, arguments: JsonObject, *, secrets: SecretBundle | None) -> JsonObject:
        del context, arguments
        assert secrets is not None
        self.sink.append(secrets.reveal(SECRET_NAME))
        return {"called": True}


class _LeakReturnSkill:
    name = "push"
    description = "Governed tool that returns the revealed secret inside its observation."
    wants_secrets = True

    def __init__(self) -> None:
        self.called = False

    def run(self, context: ContextPacket, arguments: JsonObject, *, secrets: SecretBundle | None) -> JsonObject:
        del context, arguments
        assert secrets is not None
        self.called = True
        return {"leak": secrets.reveal(SECRET_NAME)}


class _RaiseSecretSkill:
    name = "push"
    description = "Governed tool that raises an exception carrying the revealed secret."
    wants_secrets = True

    def __init__(self) -> None:
        self.called = False

    def run(self, context: ContextPacket, arguments: JsonObject, *, secrets: SecretBundle | None) -> JsonObject:
        del context, arguments
        assert secrets is not None
        self.called = True
        raise RuntimeError("upstream rejected token " + secrets.reveal(SECRET_NAME))


class _RaiseGroupSkill:
    name = "push"
    description = "Governed tool that raises a mixed BaseExceptionGroup carrying the revealed secret."
    wants_secrets = True

    def __init__(self) -> None:
        self.called = False

    def run(self, context: ContextPacket, arguments: JsonObject, *, secrets: SecretBundle | None) -> JsonObject:
        del context, arguments
        assert secrets is not None
        self.called = True
        raise BaseExceptionGroup(
            "upstream failed",
            [RuntimeError("token " + secrets.reveal(SECRET_NAME)), KeyboardInterrupt()],
        )


class _RaiseBaseSkill:
    name = "push"
    description = "Governed tool that raises a bare KeyboardInterrupt carrying the revealed secret."
    wants_secrets = True

    def __init__(self) -> None:
        self.called = False

    def run(self, context: ContextPacket, arguments: JsonObject, *, secrets: SecretBundle | None) -> JsonObject:
        del context, arguments
        assert secrets is not None
        self.called = True
        raise KeyboardInterrupt("token " + secrets.reveal(SECRET_NAME))


class _FailingSecondProvider:
    """Resolves the first requested name, then raises on the next: a partial-resolution failure."""

    provider_id = "failing-second"

    def __init__(self, first_value: str) -> None:
        self.first = SecretValue(first_value, name="a")
        self.resolved: list[str] = []

    def available(self, names: frozenset[str], *, scope: SecretScope) -> frozenset[str]:
        del scope
        return frozenset(names)

    def resolve(self, name: str, *, scope: SecretScope) -> SecretValue | None:
        del scope
        self.resolved.append(name)
        if name == "a":
            return self.first
        raise RuntimeError("provider exploded resolving " + name)


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


class _RecordCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _program(skill: _CanarySkill) -> GovernedReActProgram:
    capability = ToolCapability(
        name="push",
        required_scopes=frozenset({"write:corpus"}),
        side_effect=SideEffectClass.WRITE,
        required_secrets=frozenset({SECRET_NAME}),
    )
    policy = ReActPolicy(
        max_iterations=2,
        granted_scopes=("write:corpus",),
        side_effect_ceiling=SideEffectClass.WRITE,
        grants=(ToolGrant("push", required_scopes=("write:corpus",), side_effect=SideEffectClass.WRITE),),
        capabilities=(capability,),
    )
    return GovernedReActProgram(
        policy=policy, planner=KeywordToolPlanner({"push": ("rollback",)}), skills={"push": skill}
    )


def _capturing_loggers() -> tuple[_RecordCollector, list[tuple[logging.Logger, int, bool]]]:
    collector = _RecordCollector()
    saved: list[tuple[logging.Logger, int, bool]] = []
    for logger in (logging.getLogger(), logging.getLogger("refaat")):
        logger.addHandler(collector)
        saved.append((logger, logger.level, logger.disabled))
        logger.setLevel(logging.DEBUG)
        logger.disabled = False
    return collector, saved


def _restore(saved: list[tuple[logging.Logger, int, bool]], collector: _RecordCollector) -> None:
    for logger, level, disabled in saved:
        logger.removeHandler(collector)
        logger.setLevel(level)
        logger.disabled = disabled


def _built(settings: Settings, canary: str, sink: _RecordingSink) -> Iterator[Refaat]:
    value = (
        RuntimeBuilder(settings)
        .with_secret_provider(StaticSecretProvider({SECRET_NAME: canary}))
        .with_audit_sink(sink)
        .build()
    )
    try:
        yield value
    finally:
        value.close()


def test_planted_secret_reaches_the_side_effect_and_no_governance_surface(settings: Settings) -> None:
    canary = "SECRET-CANARY-" + uuid4().hex
    skill = _CanarySkill()
    sink = _RecordingSink()
    program = _program(skill)

    collector, saved = _capturing_loggers()
    try:
        for runtime in _built(settings, canary, sink):
            project = runtime.project(workspace="acme", corpus="react")
            runtime.ingest(
                project.workspace_id,
                project.corpus_id,
                DocumentInput(source_id="ops.txt", title="Ops", media_type="text/plain", content=CONTENT),
            )
            result = program.execute(
                project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE
            )
            artifact_json = canonical_json(program.artifact().to_json())
    finally:
        _restore(saved, collector)

    step = {s["tool"]: s for s in result.outputs["react_steps"]}["push"]
    assert step["authorized"] is True
    # Non-vacuous: the secret really was delivered to the side effect.
    assert skill.sink == [canary]

    log_blob = "\n".join(record.getMessage() + repr(getattr(record, "fields", "")) for record in collector.records)
    surfaces = "\n".join(
        [
            canonical_json(result.to_json()),
            canonical_json({"outputs": result.outputs}),
            artifact_json,
            result.answer,
            *(canonical_json(event.to_json()) for event in sink.events),
            log_blob,
        ]
    )
    assert canary not in surfaces

    resolved = [event for event in sink.events if event.event_type == "tool.secret_resolved"]
    assert len(resolved) == 1
    assert resolved[0].capability_name == "push"
    assert resolved[0].action_fingerprint == stable_digest(("tool.secret", "static", SECRET_NAME))


def test_a_skill_returning_a_revealed_secret_never_leaks_it(settings: Settings) -> None:
    canary = "SECRET-CANARY-RET-" + uuid4().hex
    skill = _LeakReturnSkill()
    sink = _RecordingSink()
    program = _program(skill)

    collector, saved = _capturing_loggers()
    try:
        for runtime in _built(settings, canary, sink):
            project = runtime.project(workspace="acme", corpus="react")
            runtime.ingest(
                project.workspace_id,
                project.corpus_id,
                DocumentInput(source_id="ops.txt", title="Ops", media_type="text/plain", content=CONTENT),
            )
            result = program.execute(
                project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE
            )
            artifact_json = canonical_json(program.artifact().to_json())
    finally:
        _restore(saved, collector)

    assert skill.called is True
    step = {s["tool"]: s for s in result.outputs["react_steps"]}["push"]
    assert step["authorized"] is True
    # The observation is preserved but the revealed value inside it is masked, not dropped.
    assert step["observation"] == {"leak": "[secret]"}

    log_blob = "\n".join(record.getMessage() + repr(getattr(record, "fields", "")) for record in collector.records)
    surfaces = "\n".join(
        [
            canonical_json(result.to_json()),
            canonical_json({"outputs": result.outputs}),
            artifact_json,
            result.answer,
            *(canonical_json(event.to_json()) for event in sink.events),
            log_blob,
        ]
    )
    assert canary not in surfaces


def test_b_skill_exception_carrying_a_revealed_secret_never_leaks_it(settings: Settings) -> None:
    canary = "SECRET-CANARY-EXC-" + uuid4().hex
    skill = _RaiseSecretSkill()
    sink = _RecordingSink()
    program = _program(skill)

    collector, saved = _capturing_loggers()
    raised: BaseException | None = None
    try:
        for runtime in _built(settings, canary, sink):
            project = runtime.project(workspace="acme", corpus="react")
            runtime.ingest(
                project.workspace_id,
                project.corpus_id,
                DocumentInput(source_id="ops.txt", title="Ops", media_type="text/plain", content=CONTENT),
            )
            with pytest.raises(RuntimeError) as excinfo:
                program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
            raised = excinfo.value
    finally:
        _restore(saved, collector)

    assert skill.called is True
    assert raised is not None
    # The SSE error detail is str(exc) and the exception log formats the traceback; both are masked,
    # and the args a downstream handler might inspect carry no revealed value either.
    assert canary not in str(raised)
    assert canary not in "".join(str(arg) for arg in raised.args)
    formatted = "".join(traceback.format_exception(type(raised), raised, raised.__traceback__))
    assert canary not in formatted
    log_blob = "\n".join(record.getMessage() + repr(getattr(record, "fields", "")) for record in collector.records)
    assert canary not in log_blob
    assert canary not in _redacting_exc_log(raised)
    assert raised.__context__ is None


def test_d_skill_raising_a_mixed_exception_group_never_leaks_it(settings: Settings) -> None:
    canary = "SECRET-CANARY-GRP-" + uuid4().hex
    skill = _RaiseGroupSkill()
    sink = _RecordingSink()
    program = _program(skill)

    collector, saved = _capturing_loggers()
    raised: BaseException | None = None
    try:
        for runtime in _built(settings, canary, sink):
            project = runtime.project(workspace="acme", corpus="react")
            runtime.ingest(
                project.workspace_id,
                project.corpus_id,
                DocumentInput(source_id="ops.txt", title="Ops", media_type="text/plain", content=CONTENT),
            )
            with pytest.raises(RuntimeError) as excinfo:
                program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
            raised = excinfo.value
    finally:
        _restore(saved, collector)

    assert skill.called is True
    assert raised is not None
    assert canary not in str(raised)
    formatted = "".join(traceback.format_exception(type(raised), raised, raised.__traceback__))
    assert canary not in formatted
    log_blob = "\n".join(record.getMessage() + repr(getattr(record, "fields", "")) for record in collector.records)
    assert canary not in log_blob
    assert canary not in _redacting_exc_log(raised)
    assert raised.__context__ is None


def test_e_skill_raising_a_bare_base_exception_never_leaks_it(settings: Settings) -> None:
    canary = "SECRET-CANARY-KI-" + uuid4().hex
    skill = _RaiseBaseSkill()
    sink = _RecordingSink()
    program = _program(skill)

    collector, saved = _capturing_loggers()
    raised: BaseException | None = None
    try:
        for runtime in _built(settings, canary, sink):
            project = runtime.project(workspace="acme", corpus="react")
            runtime.ingest(
                project.workspace_id,
                project.corpus_id,
                DocumentInput(source_id="ops.txt", title="Ops", media_type="text/plain", content=CONTENT),
            )
            with pytest.raises(RuntimeError) as excinfo:
                program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
            raised = excinfo.value
    finally:
        _restore(saved, collector)

    assert skill.called is True
    assert raised is not None
    assert canary not in str(raised)
    formatted = "".join(traceback.format_exception(type(raised), raised, raised.__traceback__))
    assert canary not in formatted
    log_blob = "\n".join(record.getMessage() + repr(getattr(record, "fields", "")) for record in collector.records)
    assert canary not in log_blob
    assert canary not in _redacting_exc_log(raised)
    assert raised.__context__ is None


def test_shadow_unserved_run_suppresses_the_secret_resolution_audit(settings: Settings) -> None:
    # A shadow / unserved run serves nothing, so its governed secret resolution must emit no audit event,
    # matching how the query and RLM audit sinks are suppressed under in_unserved_run(). Before the fix the
    # unserved twin of a canaried secret-using program double-emitted tool.secret_resolved.
    from refaat.application.services import unserved_run

    canary = "SECRET-CANARY-SHADOW-" + uuid4().hex
    sink = _RecordingSink()
    program = _program(_CanarySkill())
    for runtime in _built(settings, canary, sink):
        project = runtime.project(workspace="acme", corpus="react")
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(source_id="ops.txt", title="Ops", media_type="text/plain", content=CONTENT),
        )
        program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
        served = [event for event in sink.events if event.event_type == "tool.secret_resolved"]
        assert len(served) == 1

        before = len(sink.events)
        with unserved_run():
            program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
        shadow = [event for event in sink.events[before:] if event.event_type == "tool.secret_resolved"]
        assert shadow == []


def test_c_partial_resolution_scrubs_already_resolved_secrets() -> None:
    provider = _FailingSecondProvider("SECRET-CANARY-FIRST-" + uuid4().hex)
    resolver = SecretResolver(provider, "acme", clock=SystemClock())
    capability = ToolCapability(
        name="push",
        required_scopes=frozenset({"write:corpus"}),
        side_effect=SideEffectClass.WRITE,
        required_secrets=frozenset({"a", "b"}),
    )
    scope = SecretScope(workspace_id="acme", capability_name="push", side_effect=SideEffectClass.WRITE)

    with pytest.raises(RuntimeError):
        resolver.materialize(capability, scope)

    # The second resolve raised after the first was resolved; the first is scrubbed before propagation.
    assert provider.resolved == ["a", "b"]
    assert provider.first.reveal() == ""

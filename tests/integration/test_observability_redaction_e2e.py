"""Canary end-to-end proof that non-audit observability never egresses raw content.

A unique marker is planted at every real leak vector (evidence text, a tool argument value, a tool
observation's content, and a governed-step reason) and driven through a plain query, a governed
STANDARD run, a governed ENHANCED run, and an RLM flow. No marker may appear in any span name, any
span attribute value, any log message, or any serialized log field. The positive control proves the
argument and observation markers ARE caller-visible in the returned react_steps, so the negative
assertions are live rather than vacuous.

This is regression prevention: no current span or log field leaks any of these; the seam locks it.
"""

from __future__ import annotations

import io
import json
import logging

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from refaat.application.runtime import Refaat
from refaat.domain.authz import Principal
from refaat.domain.enums import AuditLevel, SideEffectClass
from refaat.domain.governance import ToolCapability
from refaat.domain.models import (
    ContextPacket,
    DocumentInput,
    JsonObject,
    ReActPolicy,
    ToolGrant,
    ToolProposal,
    ToolSpec,
)
from refaat.observability import JsonFormatter
from refaat.programs import GovernedReActProgram
from refaat.programs.react import governance_reason_code

pytestmark = pytest.mark.integration

ALICE = Principal(id="alice", tenant="acme")
QUESTION = "how does rollback work"

ARG_MARKER = "CANARY-ARG-7a3"
OBS_MARKER = "CANARY-OBS-7a3"
EVIDENCE_MARKER = "CANARY-EVIDENCE-7a3"
REASON_MARKER = "CANARY-REASON-7a3"


class _CanaryPlanner:
    kind = "keyword"

    def __init__(self, arg_value: str) -> None:
        self._arg = arg_value

    def tools(self) -> tuple[str, ...]:
        return ("lookup",)

    def plan(self, query: str, used: frozenset[str]) -> ToolProposal | None:
        del query
        if "lookup" in used:
            return None
        return ToolProposal(tool="lookup", arguments={"q": self._arg})


class _CanarySkill:
    name = "lookup"
    description = "Read-only lookup that returns a marked observation."

    def __init__(self, obs_value: str) -> None:
        self._obs = obs_value

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del context, arguments
        return {"secret": self._obs}


def _ingest(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    runtime.ingest(
        workspace_id,
        corpus_id,
        DocumentInput(
            source_id="ops.txt",
            title="Ops",
            media_type="text/plain",
            content=f"Rollback restores the previously approved snapshot. Marker {EVIDENCE_MARKER} here.".encode(),
        ),
    )


def _program(audit_level: AuditLevel) -> GovernedReActProgram:
    policy = ReActPolicy(
        max_iterations=2,
        granted_scopes=("read:corpus",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
        capabilities=(
            ToolCapability(
                name="lookup",
                required_scopes=frozenset({"read:corpus"}),
                side_effect=SideEffectClass.READ_ONLY,
                requires_approval=False,
                audit_level=audit_level,
            ),
        ),
    )
    return GovernedReActProgram(
        policy=policy,
        planner=_CanaryPlanner(ARG_MARKER),
        skills={"lookup": _CanarySkill(OBS_MARKER)},
    )


def test_canary_never_egresses_to_spans_or_logs(runtime: Refaat, monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("refaat.observability._program_tracer", lambda: provider.get_tracer("test"))

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    # Attach to the refaat logger, not root: RuntimeBuilder.build() calls configure_logging, which
    # clears root handlers, and a build can run mid-flow, so a root handler would be dropped. The
    # refaat logger's handlers are never cleared, and every refaat.* record propagates up to it.
    capture_logger = logging.getLogger("refaat")
    capture_logger.addHandler(handler)
    previous_level = capture_logger.level
    capture_logger.setLevel(logging.DEBUG)
    # Alembic's fileConfig (run by an earlier migration test) disables every existing logger by
    # default, which would silently starve this capture; re-enable the whole refaat tree here so the
    # canary observes real emissions, and restore each logger's flag afterward.
    disabled_state = {
        name: obj.disabled
        for name, obj in list(logging.root.manager.loggerDict.items())
        if isinstance(obj, logging.Logger) and (name == "refaat" or name.startswith("refaat."))
    }
    for name in disabled_state:
        logging.getLogger(name).disabled = False

    project = runtime.project(workspace="acme", corpus="react")
    _ingest(runtime, project.workspace_id, project.corpus_id)

    try:
        project.ask(QUESTION, principal=ALICE)
        standard = _program(AuditLevel.STANDARD).execute(
            project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE
        )
        enhanced = _program(AuditLevel.ENHANCED).execute(
            project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE
        )
        project.rlm(QUESTION, principal=ALICE)
    finally:
        capture_logger.removeHandler(handler)
        capture_logger.setLevel(previous_level)
        for name, was in disabled_state.items():
            logging.getLogger(name).disabled = was

    log_output = stream.getvalue()
    spans = exporter.get_finished_spans()

    # Non-vacuity: the ENHANCED run really emitted a governed-step span and log, so the negative
    # assertions below run against a live emission.
    assert any(s.name == "refaat.tool.step" for s in spans)
    assert "governed_tool_step" in log_output

    for marker in (ARG_MARKER, OBS_MARKER, EVIDENCE_MARKER):
        assert marker not in log_output
        for s in spans:
            assert marker not in s.name
            for value in s.attributes.values():
                assert marker not in str(value)

    # The reason-code fix strips a canary carried in a raw governed-step reason.
    assert REASON_MARKER not in governance_reason_code(f"argument validation failed: unexpected field: {REASON_MARKER}")

    # Positive control: the argument and observation markers ARE caller-visible in react_steps, so
    # the markers really are present in the run yet never reach an egress field.
    for result in (standard, enhanced):
        step = {s["tool"]: s for s in result.outputs["react_steps"]}["lookup"]
        assert step["arguments"] == {"q": ARG_MARKER}
        assert step["observation"] == {"secret": OBS_MARKER}


_CLOSED_SCHEMA: JsonObject = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


class _ModelReasonPlanner:
    kind = "model"

    def __init__(self, field_name: str) -> None:
        self._field = field_name

    def tools(self) -> tuple[str, ...]:
        return ("lookup",)

    def plan(self, query: str, used: frozenset[str]) -> ToolProposal | None:
        del query
        if "lookup" in used:
            return None
        return ToolProposal(tool="lookup", arguments={self._field: "x"})


def _reason_canary_program() -> GovernedReActProgram:
    policy = ReActPolicy(
        max_iterations=1,
        granted_scopes=("read:corpus",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
        capabilities=(
            ToolCapability(
                name="lookup",
                required_scopes=frozenset({"read:corpus"}),
                side_effect=SideEffectClass.READ_ONLY,
                requires_approval=False,
                audit_level=AuditLevel.ENHANCED,
            ),
        ),
    )
    return GovernedReActProgram(
        policy=policy,
        planner=_ModelReasonPlanner(REASON_MARKER),
        skills={"lookup": _CanarySkill(OBS_MARKER)},
        manifest=(ToolSpec(name="lookup", description="Read-only lookup.", parameters=_CLOSED_SCHEMA),),
    )


def test_governed_step_reason_code_is_opaque_end_to_end(runtime: Refaat, monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive an ENHANCED governed step whose raw reason carries a canary, and prove the CONSUMER
    (_emit_step_observability) reduces it to the opaque reason_code before egress.

    A model planner proposes an argument named after the canary marker against a closed tool schema,
    so the step's reason is genuinely 'argument validation failed: unexpected field: <marker>'. The
    marker must reach neither the emitted span nor the serialized log, and the emitted reason_code
    must be the fixed opaque code. This is the end-to-end lock, not governance_reason_code in isolation.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("refaat.observability._program_tracer", lambda: provider.get_tracer("test"))

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    capture_logger = logging.getLogger("refaat")
    capture_logger.addHandler(handler)
    previous_level = capture_logger.level
    capture_logger.setLevel(logging.DEBUG)
    disabled_state = {
        name: obj.disabled
        for name, obj in list(logging.root.manager.loggerDict.items())
        if isinstance(obj, logging.Logger) and (name == "refaat" or name.startswith("refaat."))
    }
    for name in disabled_state:
        logging.getLogger(name).disabled = False

    project = runtime.project(workspace="acme", corpus="reason")
    _ingest(runtime, project.workspace_id, project.corpus_id)

    try:
        result = _reason_canary_program().execute(
            project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE
        )
    finally:
        capture_logger.removeHandler(handler)
        capture_logger.setLevel(previous_level)
        for name, was in disabled_state.items():
            logging.getLogger(name).disabled = was

    log_output = stream.getvalue()
    spans = exporter.get_finished_spans()
    step_spans = [s for s in spans if s.name == "refaat.tool.step"]

    # Non-vacuity: the ENHANCED validation-failure step really emitted a governed-step span and log.
    assert step_spans
    assert "governed_tool_step" in log_output

    # Positive control: the raw reason genuinely carries the canary in the run output, so the
    # negative assertions below run against a live emission.
    step = {s["tool"]: s for s in result.outputs["react_steps"]}["lookup"]
    assert step["authorized"] is False
    assert REASON_MARKER in step["reason"]

    # The consumer stripped the canary to the opaque code end to end: it reaches no egress field.
    assert REASON_MARKER not in log_output
    for s in spans:
        assert REASON_MARKER not in s.name
        for value in s.attributes.values():
            assert REASON_MARKER not in str(value)
    assert all(s.attributes["reason_code"] == "argument_validation_failed" for s in step_spans)

    log_records = [json.loads(line) for line in log_output.splitlines() if '"governed_tool_step"' in line]
    assert log_records
    assert all(record["reason_code"] == "argument_validation_failed" for record in log_records)

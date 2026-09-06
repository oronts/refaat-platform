from __future__ import annotations

from refaat.application.sandbox import CodeGovernor, run_code_phase
from refaat.domain.enums import SideEffectClass
from refaat.domain.sandbox import CodePolicy, CodeRequest, SandboxLimits
from refaat.infrastructure.sandbox import DenyingSandbox, SafeExpressionSandbox

LIMITS = SandboxLimits(max_nodes=64, max_output_bytes=4096, max_wall_seconds=2.0)


def _policy(**kwargs: object) -> CodePolicy:
    defaults: dict = {
        "granted_scopes": ("compute",),
        "side_effect_ceiling": SideEffectClass.READ_ONLY,
        "max_steps": 8,
        "limits": LIMITS,
    }
    defaults.update(kwargs)
    return CodePolicy(**defaults)  # type: ignore[arg-type]


def test_governor_denies_when_no_executing_sandbox() -> None:
    governor = CodeGovernor(_policy(), sandbox_executes=False)
    admitted, reason = governor.authorize(CodeRequest(code="1", required_scopes=("compute",)))
    assert admitted is False
    assert "deny by default" in reason


def test_governor_denies_missing_scope() -> None:
    governor = CodeGovernor(_policy(granted_scopes=()), sandbox_executes=True)
    admitted, reason = governor.authorize(CodeRequest(code="1", required_scopes=("compute",)))
    assert admitted is False
    assert "missing scopes" in reason


def test_governor_denies_over_ceiling() -> None:
    governor = CodeGovernor(_policy(side_effect_ceiling=SideEffectClass.READ_ONLY), sandbox_executes=True)
    request = CodeRequest(code="1", required_scopes=("compute",), side_effect=SideEffectClass.EXTERNAL)
    admitted, reason = governor.authorize(request)
    assert admitted is False
    assert "exceeds ceiling" in reason


def test_governor_admits_within_scope_and_ceiling() -> None:
    governor = CodeGovernor(_policy(), sandbox_executes=True)
    admitted, reason = governor.authorize(CodeRequest(code="1", required_scopes=("compute",)))
    assert admitted is True
    assert reason == "authorized"


def test_phase_denies_every_step_under_default_deny_sandbox() -> None:
    requests = [CodeRequest(code="a + b", inputs={"a": 1, "b": 2}, required_scopes=("compute",))]
    steps = run_code_phase(DenyingSandbox(), _policy(), requests)
    assert len(steps) == 1
    assert steps[0].admitted is False
    assert steps[0].ok is False
    assert steps[0].sandbox == "deny"


def test_phase_executes_and_records_audit() -> None:
    requests = [CodeRequest(code="a + b", inputs={"a": 2, "b": 3}, required_scopes=("compute",))]
    steps = run_code_phase(SafeExpressionSandbox(), _policy(), requests)
    assert steps[0].admitted is True
    assert steps[0].ok is True
    assert steps[0].value == 5
    assert steps[0].as_json()["value"] == 5


def test_phase_feeds_results_forward() -> None:
    requests = [
        CodeRequest(code="a + b", inputs={"a": 2, "b": 3}, required_scopes=("compute",)),
        CodeRequest(code="result_0 * c", inputs={"c": 10}, required_scopes=("compute",)),
    ]
    steps = run_code_phase(SafeExpressionSandbox(), _policy(), requests)
    assert steps[0].value == 5
    assert steps[1].value == 50


def test_phase_records_denied_and_admitted_together() -> None:
    requests = [
        CodeRequest(code="a + b", inputs={"a": 1, "b": 1}, required_scopes=("compute",)),
        CodeRequest(
            code="a + b",
            inputs={"a": 1, "b": 1},
            required_scopes=("compute",),
            side_effect=SideEffectClass.WRITE,
        ),
    ]
    steps = run_code_phase(SafeExpressionSandbox(), _policy(), requests)
    assert steps[0].admitted is True
    assert steps[1].admitted is False
    assert "exceeds ceiling" in steps[1].reason


def test_phase_stops_at_max_steps() -> None:
    requests = [CodeRequest(code="1", required_scopes=("compute",)) for _ in range(5)]
    steps = run_code_phase(SafeExpressionSandbox(), _policy(max_steps=2), requests)
    assert len(steps) == 2


def test_phase_records_execution_refusal_as_not_ok() -> None:
    requests = [CodeRequest(code="__import__('os')", required_scopes=("compute",))]
    steps = run_code_phase(SafeExpressionSandbox(), _policy(), requests)
    assert steps[0].admitted is True
    assert steps[0].ok is False
    assert steps[0].error

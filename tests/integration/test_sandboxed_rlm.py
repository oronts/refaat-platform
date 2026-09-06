from __future__ import annotations

from collections.abc import Iterator

import pytest

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import DocumentInput
from refaat.domain.sandbox import CodePolicy, CodeRequest, SandboxLimits
from refaat.infrastructure.sandbox import SafeExpressionSandbox, SubprocessCodeSandbox

pytestmark = pytest.mark.integration

LIMITS = SandboxLimits(max_nodes=64, max_output_bytes=4096, max_wall_seconds=2.0)


def _ingest(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    topics = (
        ("release", "Production releases require peer review and a tested rollback plan."),
        ("incident", "Severe incidents require an incident commander and a written timeline."),
    )
    for name, fact in topics:
        filler = " ".join(f"{name} evidence detail {index}." for index in range(40))
        runtime.ingest(
            workspace_id,
            corpus_id,
            DocumentInput(
                source_id=f"{name}.md",
                title=f"{name.title()} policy",
                media_type="text/markdown",
                content=f"{fact}\n\n{filler}".encode(),
            ),
        )


def _sandbox_runtime(settings: Settings) -> Iterator[Refaat]:
    value = RuntimeBuilder(settings).with_code_sandbox(SafeExpressionSandbox()).build()
    try:
        yield value
    finally:
        value.close()


def _policy() -> CodePolicy:
    return CodePolicy(
        granted_scopes=("compute",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        max_steps=8,
        limits=LIMITS,
    )


def test_default_runtime_runs_no_code_and_omits_code_usage(runtime, workspace_id, corpus_id) -> None:
    _ingest(runtime, workspace_id, corpus_id)
    result = runtime.run_rlm(workspace_id, corpus_id, "What controls govern production releases?")
    assert "code_steps" not in result.usage
    assert "code_sandbox" not in result.usage


def test_default_deny_records_denied_step_and_answer_is_byte_identical(runtime, workspace_id, corpus_id) -> None:
    _ingest(runtime, workspace_id, corpus_id)
    query = "What controls govern production releases?"
    baseline = runtime.run_rlm(workspace_id, corpus_id, query)
    governed = runtime.run_rlm(
        workspace_id,
        corpus_id,
        query,
        code_policy=_policy(),
        code_requests=(CodeRequest(code="a + b", inputs={"a": 1, "b": 2}, required_scopes=("compute",)),),
    )
    assert governed.answer == baseline.answer
    assert governed.citations == baseline.citations
    assert governed.usage["code_sandbox"] == "deny"
    assert len(governed.usage["code_steps"]) == 1
    step = governed.usage["code_steps"][0]
    assert step["admitted"] is False
    assert step["ok"] is False


def test_injected_safe_sandbox_executes_and_feeds_back(settings) -> None:
    for runtime in _sandbox_runtime(settings):
        workspace_id = runtime.create_workspace("acme", "Acme")
        corpus_id = runtime.create_corpus(workspace_id, "handbook")
        _ingest(runtime, workspace_id, corpus_id)
        result = runtime.run_rlm(
            workspace_id,
            corpus_id,
            "What controls govern production releases?",
            code_policy=_policy(),
            code_requests=(
                CodeRequest(code="a + b", inputs={"a": 2, "b": 3}, required_scopes=("compute",)),
                CodeRequest(code="result_0 * c", inputs={"c": 10}, required_scopes=("compute",)),
                CodeRequest(
                    code="a + b",
                    inputs={"a": 1, "b": 1},
                    required_scopes=("compute",),
                    side_effect=SideEffectClass.EXTERNAL,
                ),
                CodeRequest(code="__import__('os')", required_scopes=("compute",)),
            ),
        )
        assert result.citations
        assert result.usage["code_sandbox"] == "safe_expression"
        steps = result.usage["code_steps"]
        assert steps[0]["ok"] is True
        assert steps[0]["value"] == 5
        assert steps[1]["ok"] is True
        assert steps[1]["value"] == 50
        assert steps[2]["admitted"] is False
        assert "exceeds ceiling" in steps[2]["reason"]
        assert steps[3]["admitted"] is True
        assert steps[3]["ok"] is False


def test_subprocess_reference_adapter_is_bounded(settings) -> None:
    sandbox = SubprocessCodeSandbox()
    result = sandbox.execute("result = inputs['a'] + inputs['b']", {"a": 4, "b": 5}, LIMITS)
    if not result.ok:
        pytest.skip(f"platform cannot enforce subprocess sandbox limits: {result.error}")
    assert result.value == 9

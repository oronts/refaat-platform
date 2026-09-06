from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import copy_demo_documents, demo_settings, emit, seed_demo_project

from refaat.application.runtime import RuntimeBuilder
from refaat.domain.enums import SideEffectClass
from refaat.domain.sandbox import CodePolicy, CodeRequest, SandboxLimits
from refaat.infrastructure.sandbox import SafeExpressionSandbox

QUERY = "What controls govern production releases?"


def _policy() -> CodePolicy:
    return CodePolicy(
        granted_scopes=("compute",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        max_steps=8,
        limits=SandboxLimits(max_nodes=64, max_output_bytes=4096, max_wall_seconds=2.0),
    )


def _governed_requests() -> tuple[CodeRequest, ...]:
    return (
        CodeRequest(code="a + b", inputs={"a": 2, "b": 3}, required_scopes=("compute",)),
        CodeRequest(code="result_0 * c", inputs={"c": 10}, required_scopes=("compute",)),
        CodeRequest(
            code="a + b",
            inputs={"a": 1, "b": 1},
            required_scopes=("compute",),
            side_effect=SideEffectClass.EXTERNAL,
        ),
        CodeRequest(code="__import__('os')", required_scopes=("compute",)),
    )


def main() -> None:
    with TemporaryDirectory(prefix="refaat-sandboxed-rlm-") as directory:
        root = Path(directory)
        documents = copy_demo_documents(root / "documents")
        settings = demo_settings(root)

        with RuntimeBuilder(settings).build() as deny_runtime:
            project = seed_demo_project(deny_runtime, documents)
            baseline = project.rlm(QUERY)
            denied = project.rlm(
                QUERY,
                code_policy=_policy(),
                code_requests=(CodeRequest(code="a + b", inputs={"a": 1, "b": 2}, required_scopes=("compute",)),),
            )
            deny_step = denied.usage["code_steps"][0]
            default_deny = {
                "default_deny_answer_identical": denied.answer == baseline.answer,
                "default_deny_citations_identical": denied.citations == baseline.citations,
                "default_deny_sandbox": denied.usage["code_sandbox"],
                "default_deny_step_denied": deny_step["admitted"] is False and deny_step["ok"] is False,
                "default_deny_nothing_executed": deny_step["value"] is None,
            }

        sandbox_root = root / "sandbox"
        sandbox_root.mkdir()
        sandbox_settings = demo_settings(sandbox_root)
        with RuntimeBuilder(sandbox_settings).with_code_sandbox(SafeExpressionSandbox()).build() as runtime:
            project = seed_demo_project(runtime, documents)
            result = project.rlm(QUERY, code_policy=_policy(), code_requests=_governed_requests())
            steps = result.usage["code_steps"]

        emit(
            {
                **default_deny,
                "sandbox_name": result.usage["code_sandbox"],
                "citations_present": bool(result.citations),
                "governed_executed_value": steps[0]["value"],
                "governed_step_admitted": steps[0]["admitted"] is True and steps[0]["ok"] is True,
                "governed_fed_forward_value": steps[1]["value"],
                "over_ceiling_denied": steps[2]["admitted"] is False and "exceeds ceiling" in steps[2]["reason"],
                "escape_attempt_refused": steps[3]["admitted"] is True and steps[3]["ok"] is False,
                "escape_attempt_error": steps[3]["error"],
                "audit_step_count": len(steps),
            }
        )


if __name__ == "__main__":
    main()

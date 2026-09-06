from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from refaat.domain.sandbox import CodePolicy, CodeRequest, CodeStep
from refaat.ports import CodeSandbox

_CODE_EXCERPT_CHARS = 500


class CodeGovernor:
    """Pure authorization for one governed code request.

    Modeled on the tool governor: a request is admitted only when an executing
    sandbox is present, the caller holds every scope the request requires, and
    the request's side-effect class does not exceed the policy ceiling. No I/O,
    so the decision is reproducible. Deny is the default: any missing condition
    refuses the request.
    """

    def __init__(self, policy: CodePolicy, *, sandbox_executes: bool) -> None:
        self._policy = policy
        self._sandbox_executes = sandbox_executes

    def authorize(self, request: CodeRequest) -> tuple[bool, str]:
        if not self._sandbox_executes:
            return False, "no execution sandbox is configured (deny by default)"
        missing = tuple(scope for scope in request.required_scopes if scope not in self._policy.granted_scopes)
        if missing:
            return False, f"missing scopes: {', '.join(missing)}"
        if request.side_effect.rank > self._policy.side_effect_ceiling.rank:
            return False, f"side effect {request.side_effect} exceeds ceiling {self._policy.side_effect_ceiling}"
        return True, "authorized"


def run_code_phase(
    sandbox: CodeSandbox,
    policy: CodePolicy,
    requests: Sequence[CodeRequest],
) -> tuple[CodeStep, ...]:
    """Run a bounded, governed sequence of code requests and record every step.

    Each request is authorized by a pure :class:`CodeGovernor` and, only if
    admitted, executed by the sandbox. A successful result is fed forward as
    ``result_<index>`` into the inputs available to later requests, so the phase
    is a genuine bounded loop whose outputs chain. Both admitted and denied
    steps are returned for audit; the phase never runs more than
    ``policy.max_steps`` requests.
    """
    governor = CodeGovernor(policy, sandbox_executes=sandbox.executes)
    steps: list[CodeStep] = []
    results: dict[str, Any] = {}
    for index, request in enumerate(requests):
        if index >= policy.max_steps:
            break
        excerpt = request.code[:_CODE_EXCERPT_CHARS]
        admitted, reason = governor.authorize(request)
        if not admitted:
            steps.append(
                CodeStep(
                    code=excerpt,
                    admitted=False,
                    reason=reason,
                    ok=False,
                    sandbox=sandbox.name,
                )
            )
            continue
        merged = {**results, **request.inputs}
        outcome = sandbox.execute(request.code, merged, policy.limits)
        steps.append(
            CodeStep(
                code=excerpt,
                admitted=True,
                reason=reason,
                ok=outcome.ok,
                sandbox=sandbox.name,
                value=outcome.value,
                error=outcome.error,
            )
        )
        if outcome.ok:
            results[f"result_{index}"] = outcome.value
    return tuple(steps)

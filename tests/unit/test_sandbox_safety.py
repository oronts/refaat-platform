from __future__ import annotations

import pytest

from refaat.domain.sandbox import SandboxLimits
from refaat.infrastructure.sandbox import DenyingSandbox, SafeExpressionSandbox

LIMITS = SandboxLimits(max_nodes=256, max_output_bytes=4096, max_wall_seconds=2.0)


def _run(code: str, inputs: dict | None = None) -> tuple[bool, object, str | None]:
    result = SafeExpressionSandbox().execute(code, inputs or {}, LIMITS)
    return result.ok, result.value, result.error


def test_denying_sandbox_executes_nothing() -> None:
    sandbox = DenyingSandbox()
    assert sandbox.executes is False
    result = sandbox.execute("1 + 1", {}, LIMITS)
    assert result.ok is False
    assert result.value is None
    assert "deny by default" in (result.error or "")


def test_valid_arithmetic_and_inputs() -> None:
    ok, value, _ = _run("a * b + 2", {"a": 3, "b": 4})
    assert ok
    assert value == 14


def test_valid_boolean_comparison_and_indexing() -> None:
    ok, value, _ = _run("data[1] > 5 and flag", {"data": [1, 9, 3], "flag": True})
    assert ok
    assert value is True


def test_valid_string_and_container_literals() -> None:
    ok, value, _ = _run("[x, y]", {"x": "left", "y": "right"})
    assert ok
    assert value == ["left", "right"]


ESCAPE_VECTORS = [
    "__import__('os')",
    "__import__('os').system('id')",
    "open('/etc/passwd')",
    "().__class__",
    "().__class__.__bases__",
    "(1).__class__.__mro__",
    "data.__class__",
    "__builtins__",
    "__builtins__['eval']('1')",
    "eval('1')",
    "exec('x=1')",
    "getattr(data, 'x')",
    "data.append(1)",
    "[x for x in range(10)]",
    "{k: 1 for k in range(3)}",
    "(lambda: 1)()",
    "lambda x: x",
    "(x := 5)",
    "f'{data}'",
    "abs(-1)",
    "len(data)",
    "os",
    "_secret",
    "data[_secret]",
    "1 if data else 2",
    "data[0:2]",
    "{1, 2, 3}",
    "{'a': 1}",
]


@pytest.mark.parametrize("code", ESCAPE_VECTORS)
def test_escape_vectors_are_refused(code: str) -> None:
    ok, value, error = _run(code, {"data": [1, 2, 3], "flag": True})
    assert ok is False
    assert value is None
    assert error


def test_import_statement_is_a_syntax_refusal() -> None:
    ok, _, error = _run("import os")
    assert ok is False
    assert "single safe expression" in (error or "")


def test_node_bound_refuses_a_deep_expression() -> None:
    tiny = SandboxLimits(max_nodes=5, max_output_bytes=4096, max_wall_seconds=2.0)
    result = SafeExpressionSandbox().execute("a + a + a + a + a + a", {"a": 1}, tiny)
    assert result.ok is False
    assert "node bound" in (result.error or "")


def test_huge_exponent_is_refused_before_materializing() -> None:
    ok, _, error = _run("10 ** 10 ** 10")
    assert ok is False
    assert "exponent" in (error or "")


def test_large_integer_power_is_refused() -> None:
    ok, _, error = _run("base ** exp", {"base": 9, "exp": 5000})
    assert ok is False
    assert "exponent" in (error or "")


def test_string_repetition_blowup_is_refused_before_building() -> None:
    ok, _, error = _run("s * n", {"s": "abcd", "n": 10_000_000})
    assert ok is False
    assert "output bound" in (error or "")


def test_output_bound_refuses_a_long_string_literal() -> None:
    tiny = SandboxLimits(max_nodes=64, max_output_bytes=8, max_wall_seconds=2.0)
    result = SafeExpressionSandbox().execute("'this string is definitely too long'", {}, tiny)
    assert result.ok is False


def test_source_larger_than_output_bound_is_refused() -> None:
    tiny = SandboxLimits(max_nodes=64, max_output_bytes=4, max_wall_seconds=2.0)
    result = SafeExpressionSandbox().execute("1 + 2 + 3", {}, tiny)
    assert result.ok is False
    assert "source" in (result.error or "")


def test_division_by_zero_fails_closed_not_crash() -> None:
    ok, _, error = _run("a / b", {"a": 1, "b": 0})
    assert ok is False
    assert error


def test_deterministic_repeat_runs_are_identical() -> None:
    first = SafeExpressionSandbox().execute("a ** 2 + b", {"a": 6, "b": 1}, LIMITS)
    second = SafeExpressionSandbox().execute("a ** 2 + b", {"a": 6, "b": 1}, LIMITS)
    assert first == second
    assert first.value == 37


def test_nested_shared_reference_repetition_fails_closed() -> None:
    """A nested repetition whose cumulative value traversal exceeds the output
    bound is refused deterministically, even though each list is within bounds and
    the source has few AST nodes. This is the resource-exhaustion fail-closed
    guarantee: the guard traversal is budget-counted, not just the node count."""
    result = SafeExpressionSandbox().execute("[[a] * 100] * 100", {"a": 1}, LIMITS)
    assert result.ok is False
    assert "output bound" in (result.error or "")
    again = SafeExpressionSandbox().execute("[[a] * 100] * 100", {"a": 1}, LIMITS)
    assert again == result


def test_deeply_nested_source_fails_closed_not_crash() -> None:
    """A source deep enough to overflow the interpreter recursion returns a clean
    refusal, never an uncaught RecursionError or MemoryError out of execute."""
    roomy = SandboxLimits(max_nodes=100_000, max_output_bytes=1_000_000, max_wall_seconds=2.0)
    code = "1" + ("+1" * 20_000)
    result = SafeExpressionSandbox().execute(code, {}, roomy)
    assert result.ok is False
    assert result.error

from __future__ import annotations

import ast
import contextlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from typing import Any

from refaat.domain.models import JsonObject
from refaat.domain.sandbox import SandboxLimits, SandboxResult


class DenyingSandbox:
    """The shipped default: refuses every execution and runs no code.

    With this sandbox installed the runtime has no code-execution capability at
    all. A request that a policy would otherwise admit is still refused here, so
    the safe-by-default posture holds even under a permissive policy: code runs
    only when an operator explicitly installs a sandbox whose ``executes`` is
    true.
    """

    name = "deny"
    executes = False

    def execute(self, code: str, inputs: JsonObject, limits: SandboxLimits) -> SandboxResult:
        return SandboxResult(
            ok=False,
            error="code execution denied: no execution sandbox is configured (deny by default)",
        )


_ALLOWED_BINOPS: dict[type[ast.operator], Any] = {}
_ALLOWED_UNARYOPS: dict[type[ast.unaryop], Any] = {}
_ALLOWED_COMPARE: dict[type[ast.cmpop], Any] = {}


def _register_operators() -> None:
    import operator

    _ALLOWED_BINOPS.update(
        {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }
    )
    _ALLOWED_UNARYOPS.update(
        {
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
            ast.Not: operator.not_,
        }
    )
    _ALLOWED_COMPARE.update(
        {
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
        }
    )


_register_operators()

_MAX_INT_BITS = 4096
_MAX_POW_EXPONENT = 64


class _SandboxRefusal(Exception):
    """Raised when a program leaves the allowlist or exceeds a bound."""


class _SafeInterpreter:
    """A bounded tree-walking evaluator over an allowlisted expression grammar.

    It is not a restricted ``eval``: it never consults builtins, globals, or any
    name outside the supplied inputs, and it implements only arithmetic,
    comparison, boolean logic, container literals, and indexing. Every node is
    counted against a hard bound and every produced value is size-guarded, so it
    can neither escape (no imports, attributes, calls, or comprehensions exist in
    the grammar) nor exhaust resources.
    """

    def __init__(self, inputs: JsonObject, limits: SandboxLimits) -> None:
        self._inputs = inputs
        self._limits = limits
        self._ops = 0
        self._guard_visits = 0
        self._deadline = time.monotonic() + limits.max_wall_seconds

    def evaluate(self, tree: ast.Expression) -> Any:
        return self._eval(tree.body)

    def _tick(self) -> None:
        self._ops += 1
        if self._ops > self._limits.max_nodes:
            raise _SandboxRefusal(f"program exceeds the {self._limits.max_nodes}-node bound")
        if time.monotonic() > self._deadline:
            raise _SandboxRefusal("program exceeds the wall-time bound")

    def _eval(self, node: ast.AST) -> Any:
        self._tick()
        handler = _DISPATCH.get(type(node))
        if handler is None:
            raise _SandboxRefusal(f"{type(node).__name__} is not permitted")
        return handler(self, node)

    def _eval_constant(self, node: ast.Constant) -> Any:
        value = node.value
        if isinstance(value, (int, float, bool, str)) or value is None:
            return self._guard(value)
        raise _SandboxRefusal(f"constant of type {type(value).__name__} is not permitted")

    def _eval_name(self, node: ast.Name) -> Any:
        if node.id.startswith("_"):
            raise _SandboxRefusal(f"name {node.id!r} is not permitted")
        if node.id not in self._inputs:
            raise _SandboxRefusal(f"name {node.id!r} is not a provided input")
        return self._guard(self._inputs[node.id])

    def _eval_binop(self, node: ast.BinOp) -> Any:
        op = _ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise _SandboxRefusal(f"operator {type(node.op).__name__} is not permitted")
        left = self._eval(node.left)
        right = self._eval(node.right)
        self._guard_binop(node.op, left, right)
        return self._guard(op(left, right))

    def _eval_unaryop(self, node: ast.UnaryOp) -> Any:
        op = _ALLOWED_UNARYOPS.get(type(node.op))
        if op is None:
            raise _SandboxRefusal(f"operator {type(node.op).__name__} is not permitted")
        return self._guard(op(self._eval(node.operand)))

    def _eval_boolop(self, node: ast.BoolOp) -> Any:
        values = [self._eval(value) for value in node.values]
        if isinstance(node.op, ast.And):
            result: Any = True
            for value in values:
                result = value
                if not value:
                    break
            return result
        result = False
        for value in values:
            result = value
            if value:
                break
        return result

    def _eval_compare(self, node: ast.Compare) -> Any:
        left = self._eval(node.left)
        for operator_node, comparator in zip(node.ops, node.comparators, strict=True):
            op = _ALLOWED_COMPARE.get(type(operator_node))
            if op is None:
                raise _SandboxRefusal(f"comparison {type(operator_node).__name__} is not permitted")
            right = self._eval(comparator)
            if not op(left, right):
                return False
            left = right
        return True

    def _eval_subscript(self, node: ast.Subscript) -> Any:
        if isinstance(node.slice, ast.Slice):
            raise _SandboxRefusal("slice subscripts are not permitted")
        container = self._eval(node.value)
        key = self._eval(node.slice)
        if not isinstance(container, (list, tuple, str, dict)):
            raise _SandboxRefusal("subscript target must be a sequence or mapping")
        try:
            return self._guard(container[key])
        except (IndexError, KeyError, TypeError) as exc:
            raise _SandboxRefusal(f"invalid subscript: {exc}") from exc

    def _eval_sequence(self, node: ast.List | ast.Tuple) -> Any:
        items = [self._eval(element) for element in node.elts]
        return self._guard(items if isinstance(node, ast.List) else tuple(items))

    def _guard_binop(self, op: ast.operator, left: Any, right: Any) -> None:
        if isinstance(op, ast.Pow) and isinstance(right, int) and abs(right) > _MAX_POW_EXPONENT:
            raise _SandboxRefusal("exponent exceeds the sandbox bound")
        if isinstance(op, ast.Mult):
            sequence, scalar = (left, right) if isinstance(left, (str, list, tuple)) else (right, left)
            if (
                isinstance(sequence, (str, list, tuple))
                and isinstance(scalar, int)
                and len(sequence) * max(scalar, 0) > self._limits.max_output_bytes
            ):
                raise _SandboxRefusal("repetition exceeds the output bound")

    def _guard(self, value: Any) -> Any:
        self._guard_visits += 1
        if self._guard_visits > self._limits.max_output_bytes:
            raise _SandboxRefusal("value traversal exceeds the output bound")
        if time.monotonic() > self._deadline:
            raise _SandboxRefusal("program exceeds the wall-time bound")
        limit = self._limits.max_output_bytes
        if isinstance(value, bool) or value is None or isinstance(value, float):
            return value
        if isinstance(value, int):
            if value.bit_length() > _MAX_INT_BITS:
                raise _SandboxRefusal("integer exceeds the sandbox magnitude bound")
            return value
        if isinstance(value, str):
            if len(value) > limit:
                raise _SandboxRefusal("string exceeds the output bound")
            return value
        if isinstance(value, (list, tuple)):
            if len(value) > limit:
                raise _SandboxRefusal("sequence exceeds the output bound")
            for element in value:
                self._guard(element)
            return value
        if isinstance(value, dict):
            if len(value) > limit:
                raise _SandboxRefusal("mapping exceeds the output bound")
            return value
        raise _SandboxRefusal(f"value of type {type(value).__name__} is not permitted")


_DISPATCH: dict[type[ast.AST], Any] = {
    ast.Constant: _SafeInterpreter._eval_constant,
    ast.Name: _SafeInterpreter._eval_name,
    ast.BinOp: _SafeInterpreter._eval_binop,
    ast.UnaryOp: _SafeInterpreter._eval_unaryop,
    ast.BoolOp: _SafeInterpreter._eval_boolop,
    ast.Compare: _SafeInterpreter._eval_compare,
    ast.Subscript: _SafeInterpreter._eval_subscript,
    ast.List: _SafeInterpreter._eval_sequence,
    ast.Tuple: _SafeInterpreter._eval_sequence,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_safe(element) for element in value]
    if isinstance(value, list):
        return [_json_safe(element) for element in value]
    return value


class SafeExpressionSandbox:
    """A provably-safe, deterministic evaluator for the offline governance flow.

    This is not a general code runtime. It evaluates a single expression over the
    supplied inputs using an allowlisted grammar and a hand-written interpreter
    that never reaches builtins, imports, attributes, calls, comprehensions, or
    any name outside the inputs. It exists to exercise the governor and the RLM
    code phase deterministically in the gate; arbitrary code needs the isolated
    reference adapter below.
    """

    name = "safe_expression"
    executes = True

    def execute(self, code: str, inputs: JsonObject, limits: SandboxLimits) -> SandboxResult:
        if len(code) > limits.max_output_bytes:
            return SandboxResult(ok=False, error="program source exceeds the output bound")
        try:
            tree = ast.parse(code, mode="eval")
        except SyntaxError as exc:
            return SandboxResult(ok=False, error=f"not a single safe expression: {exc.msg}")
        except (RecursionError, MemoryError, ValueError) as exc:
            return SandboxResult(ok=False, error=f"source is too complex to parse safely: {type(exc).__name__}")
        interpreter = _SafeInterpreter(inputs, limits)
        try:
            value = interpreter.evaluate(tree)
        except _SandboxRefusal as exc:
            return SandboxResult(ok=False, error=str(exc), usage={"nodes": interpreter._ops})
        except (ArithmeticError, ValueError, TypeError) as exc:
            return SandboxResult(ok=False, error=f"evaluation error: {exc}", usage={"nodes": interpreter._ops})
        except (RecursionError, MemoryError) as exc:
            return SandboxResult(
                ok=False,
                error=f"evaluation exceeded interpreter limits: {type(exc).__name__}",
                usage={"nodes": interpreter._ops},
            )
        rendered = repr(value)
        if len(rendered) > limits.max_output_bytes:
            rendered = rendered[: limits.max_output_bytes]
        return SandboxResult(
            ok=True,
            value=_json_safe(value),
            stdout=rendered,
            usage={"nodes": interpreter._ops},
        )


_SUBPROCESS_HARNESS = r"""
import json
import sys

payload = json.loads(sys.stdin.read())
code = payload["code"]
inputs = dict(payload["inputs"])
allowed = (
    "abs", "min", "max", "round", "len", "sum", "sorted", "range",
    "enumerate", "map", "filter", "int", "float", "str", "bool",
    "list", "tuple", "dict", "set",
)
builtins_module = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
safe_builtins = {name: builtins_module[name] for name in allowed if name in builtins_module}
namespace = {"__builtins__": safe_builtins, "inputs": inputs}
exec(code, namespace)
sys.stdout.write(json.dumps({"result": namespace.get("result")}))
"""


class SubprocessCodeSandbox:
    """A reference resource-bounded adapter that runs arbitrary Python in a child process.

    This is an OPT-IN reference adapter, never spawned by the offline gate. It
    bounds CPU time, address space, file writes, open file descriptors, the
    process group, and wall time; it runs in a throwaway scratch directory with an
    empty PATH and no inherited file descriptors under the isolated interpreter
    mode, and it makes a best-effort attempt to unshare the network namespace.

    It is NOT a confidentiality boundary on a standard unprivileged host. The
    network unshare needs privileges and is silently skipped when unavailable, so
    the child may keep network access; the file-size limit blocks writes but the
    child can still READ any file the user can read; and the restricted builtins
    in the child harness are bypassable from arbitrary Python. Use this adapter
    only for code you already trust to a CPU and memory and time bound. For
    genuinely untrusted code, run it inside a real jail (namespaces plus seccomp
    plus a read-only mount and a network namespace, for example nsjail or
    bubblewrap) and inject that as the CodeSandbox instead.
    """

    name = "subprocess"
    executes = True

    def __init__(self, *, max_memory_bytes: int = 256 * 1024 * 1024) -> None:
        self._max_memory_bytes = max_memory_bytes

    def execute(self, code: str, inputs: JsonObject, limits: SandboxLimits) -> SandboxResult:
        try:
            import resource
        except ImportError:
            return SandboxResult(ok=False, error="platform cannot enforce subprocess resource limits")

        payload = json.dumps({"code": code, "inputs": inputs})
        with tempfile.TemporaryDirectory(prefix="refaat-sandbox-") as scratch:
            try:
                completed = subprocess.run(  # noqa: S603 - the child is bounded by OS resource limits
                    [sys.executable, "-I", "-c", _SUBPROCESS_HARNESS],
                    input=payload,
                    capture_output=True,
                    text=True,
                    cwd=scratch,
                    env={"PATH": "", "PYTHONHASHSEED": "0"},
                    timeout=limits.max_wall_seconds,
                    close_fds=True,
                    preexec_fn=self._apply_limits(resource, limits),
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return SandboxResult(ok=False, error="execution exceeded the wall-time bound")
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "non-zero exit"
            return SandboxResult(ok=False, error=f"execution failed: {detail}")
        stdout = completed.stdout[: limits.max_output_bytes]
        try:
            value = json.loads(stdout).get("result")
        except json.JSONDecodeError:
            return SandboxResult(ok=False, error="child produced no serializable result")
        return SandboxResult(ok=True, value=value, stdout=stdout)

    def _apply_limits(self, resource: Any, limits: SandboxLimits) -> Any:
        max_cpu = max(1, math.ceil(limits.max_wall_seconds))
        max_memory = self._max_memory_bytes

        def _preexec() -> None:
            os.setsid()
            resource.setrlimit(resource.RLIMIT_CPU, (max_cpu, max_cpu + 1))
            resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
            resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
            unshare = getattr(os, "unshare", None)
            clone_newnet = getattr(os, "CLONE_NEWNET", None)
            if unshare is not None and clone_newnet is not None:
                with contextlib.suppress(OSError):
                    unshare(clone_newnet)

        return _preexec

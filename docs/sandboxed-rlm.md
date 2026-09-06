# Sandboxed code in the bounded RLM

## Safe by default

The bounded RLM keeps the corpus outside the model context and answers only from canonical evidence. It has never executed code, and the shipped default still executes none. This page describes an opt-in capability that lets an operator add governed code execution to an RLM run. It is off unless an operator turns it on, and turning it on requires two independent choices: installing a sandbox that can execute, and passing a policy that admits a request.

This is an honest change to the runtime's capabilities. Before this feature the runtime had no code-execution path at all. It now has one, gated so that the default remains no execution. The default `code_sandbox` is `deny`, which installs a sandbox that refuses every call and runs nothing. With it, every RLM run is byte-identical to before.

## The two gates

Code runs only when both gates open:

1. An executing sandbox is installed. The default `DenyingSandbox` never executes. An operator must inject a real sandbox through `RuntimeBuilder.with_code_sandbox(...)` or select one with `REFAAT_CODE_SANDBOX`.
2. A `CodePolicy` admits the request. The policy is passed on an in-process RLM call. With no policy the RLM never enters a code phase.

The `CodeGovernor` is modeled on the tool governor used in governed ReAct. It admits a request only when an executing sandbox is present, the caller holds every scope the request requires, and the request's side-effect class is within the policy ceiling. Any missing condition denies the request. Deny is the default in every ambiguous case.

Every decision is recorded. Both admitted-and-executed steps and denied steps appear in the RLM result usage under `code_steps`, alongside the sandbox name, so an operator can audit exactly what ran and what was refused.

## The governed code phase

When a policy is passed, the RLM runs a bounded phase over the caller-supplied requests. Each request carries its code, its inputs, the scopes it needs, and its side-effect class. A successful result is fed forward as `result_<index>` into the inputs available to later requests, so the phase is a genuine bounded loop whose outputs chain. The phase never runs more than `max_steps` requests.

The grounded answer and its citations are never fabricated from code output. Computation is recorded for audit; the RLM answer stays evidence-cited. This keeps the citation contract intact and keeps the answer byte-identical under the default deny sandbox.

The durable and HTTP RLM paths do not execute code. Code execution is available only on the in-process Python API, where an operator can supply a policy and requests deliberately.

## The offline evaluator

`SafeExpressionSandbox` is the sandbox the test gate uses to exercise governance and the RLM flow deterministically. It is not a general code runtime. It evaluates a single expression over the supplied inputs using an allowlisted grammar and a hand-written interpreter that never consults builtins, globals, imports, attributes, calls, or any name outside the inputs.

It permits numbers, strings, booleans, arithmetic, comparison and boolean operators, container literals, and indexing into the provided inputs. It refuses everything else: imports, attribute access, calls, comprehensions, lambdas, generator expressions, f-strings, walrus assignments, slices, and any dunder or private name. It bounds the evaluated node count, the exponent magnitude, sequence repetition, integer size, string length, and wall time, and it fails closed on any breach rather than truncating.

Because the grammar contains no construct that can reach the host, the evaluator cannot escape. Because every value is size-guarded and every program is node-bounded, it cannot exhaust resources. It exists to demonstrate the governance and RLM integration, not to run arbitrary logic.

## The isolated reference adapter

`SubprocessCodeSandbox` is the reference adapter for real arbitrary-code execution. It is opt-in and is never spawned by the offline gate. It runs code in a child process under POSIX resource limits (CPU time, address space, no file writes, few open files) in a throwaway scratch directory with a minimal environment and no inherited file descriptors, and it attempts to unshare the network namespace. These operating system bounds cap CPU, memory, file descriptors, the process group, and wall time, but they are NOT a confidentiality boundary on a standard unprivileged host: the network unshare needs privileges and is silently skipped when unavailable, the child can still read any file the user can read, and the restricted builtins in the child harness are bypassable from arbitrary Python.

Hard network isolation and hard filesystem isolation depend on the platform. The network unshare succeeds only where user namespaces permit it; for a strong guarantee, run the adapter inside a container or a network-namespaced context. The adapter refuses only when the platform provides no resource-limit facility at all; it does not refuse merely because network isolation could not be established, so treat network and filesystem confinement as absent unless you supply them. Any production deployment that admits untrusted code should run this adapter, or an equivalent WASM or container runtime, inside a real jail (namespaces plus seccomp plus a read-only mount and a network namespace, for example nsjail or bubblewrap) behind a strict policy.

## Configuration

| Setting | Default | Meaning |
| --- | --- | --- |
| `REFAAT_CODE_SANDBOX` | `deny` | `deny`, `safe_expression`, or `subprocess`. |
| `REFAAT_CODE_SANDBOX_MAX_NODES` | `256` | Node and operation bound per execution. |
| `REFAAT_CODE_SANDBOX_MAX_OUTPUT_BYTES` | `65536` | Value and output size bound. |
| `REFAAT_CODE_SANDBOX_MAX_WALL_SECONDS` | `2.0` | Wall-time bound per execution. |

## Example

```python
from refaat.application.runtime import RuntimeBuilder
from refaat.domain.enums import SideEffectClass
from refaat.domain.sandbox import CodePolicy, CodeRequest, SandboxLimits
from refaat.infrastructure.sandbox import SafeExpressionSandbox

runtime = RuntimeBuilder(settings).with_code_sandbox(SafeExpressionSandbox()).build()
policy = CodePolicy(
    granted_scopes=("compute",),
    side_effect_ceiling=SideEffectClass.READ_ONLY,
    max_steps=8,
    limits=SandboxLimits(),
)
result = runtime.run_rlm(
    workspace_id,
    corpus_id,
    "Summarize the release controls",
    code_policy=policy,
    code_requests=(
        CodeRequest(code="a + b", inputs={"a": 2, "b": 3}, required_scopes=("compute",)),
        CodeRequest(code="result_0 * c", inputs={"c": 10}, required_scopes=("compute",)),
    ),
)
for step in result.usage["code_steps"]:
    print(step["admitted"], step["ok"], step["value"], step["reason"])
```

See `examples/21_sandboxed_code_rlm.py` for a full offline run that shows the default deny, a governed computation, forward-fed results, and a refused escape attempt.

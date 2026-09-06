from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence

from refaat.domain.models import JsonObject, ToolCallDelta, ToolGrant, ToolProposal, ToolSpec
from refaat.errors import DependencyUnavailableError
from refaat.ports import Skill, StreamingToolCallingGenerator, ToolCallingGenerator
from refaat.programs.schema import validate_object

# A no-required-field object schema. Skills that take no arguments advertise
# this, so a keyword-routed tool is never rejected by validation on the model
# path and a model that proposes no arguments is always accepted.
_EMPTY_SCHEMA: JsonObject = {"type": "object", "properties": {}, "required": []}


def validate_tool_arguments(arguments: JsonObject, schema: Mapping[str, object]) -> tuple[bool, str]:
    """Validate model-proposed tool arguments against a tool's JSON schema.

    Delegates to the single shared pure-Python validator, so there is one
    validation body and no jsonschema dependency on the runtime path.
    """

    return validate_object(arguments, schema)


def build_tool_manifest(skills: Mapping[str, Skill], grants: Sequence[ToolGrant]) -> tuple[ToolSpec, ...]:
    """Build the tool manifest advertised to a model planner.

    One :class:`ToolSpec` per granted skill that is actually registered, in
    grant order. A skill may expose a ``parameters`` JSON-Schema object; a skill
    that does not falls back to a no-required-field schema.
    """

    specs: list[ToolSpec] = []
    for grant in grants:
        skill = skills.get(grant.skill_name)
        if skill is None:
            continue
        parameters = getattr(skill, "parameters", None)
        schema = dict(parameters) if isinstance(parameters, dict) else dict(_EMPTY_SCHEMA)
        specs.append(ToolSpec(name=skill.name, description=skill.description, parameters=schema))
    return tuple(specs)


class ModelToolPlanner:
    """Model-backed planner over a :class:`ToolCallingGenerator` and a manifest.

    Advertises only not-yet-used tools to the generator and returns its proposal
    (tool plus arguments) or ``None`` to stop. Argument validation is deferred to
    the governed loop so a violation is recorded in the trajectory rather than
    silently dropped.
    """

    kind = "model"

    def __init__(self, generator: ToolCallingGenerator, manifest: Sequence[ToolSpec]) -> None:
        self._generator = generator
        self._manifest = tuple(manifest)

    @property
    def model_id(self) -> str:
        return self._generator.model_id

    def manifest(self) -> tuple[ToolSpec, ...]:
        return self._manifest

    def tools(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._manifest)

    def plan(self, query: str, used: frozenset[str]) -> ToolProposal | None:
        available = tuple(spec for spec in self._manifest if spec.name not in used)
        if not available:
            return None
        return self._generator.propose_tool_call(query, available, used)


class _ToolCallAccumulator:
    """Pure reducer of streamed :class:`ToolCallDelta` fragments to a proposal.

    Concatenates each ``arguments_fragment`` and remembers the first non-None
    tool name. The joined argument buffer is parsed exactly once in
    :meth:`result` (never mid-stream), so a partial argument is never parsed,
    validated, authorized, or executed until the terminating delta.

    A terminal ``done`` delta closes the call: fragments arriving after it are
    ignored, and :meth:`result` returns a completed proposal only when a terminal
    ``done`` was seen. A stream that ends without ``done`` is incomplete and
    yields ``None``, so a truncated tool call never executes.
    """

    def __init__(self) -> None:
        self._tool: str | None = None
        self._buffer: list[str] = []
        self._done = False

    def feed(self, delta: ToolCallDelta) -> None:
        if self._done:
            return
        if self._tool is None and delta.tool is not None:
            self._tool = delta.tool
        if delta.arguments_fragment:
            self._buffer.append(delta.arguments_fragment)
        if delta.done:
            self._done = True

    def result(self) -> ToolProposal | None:
        if not self._done or self._tool is None:
            return None
        raw = "".join(self._buffer)
        if not raw:
            return ToolProposal(tool=self._tool, arguments={})
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DependencyUnavailableError("Tool-calling provider request failed") from exc
        arguments = parsed if isinstance(parsed, dict) else {}
        return ToolProposal(tool=self._tool, arguments=arguments)


class StreamingModelToolPlanner:
    """Streaming model planner over a :class:`StreamingToolCallingGenerator`.

    A sibling of :class:`ModelToolPlanner` with the same ``kind``, ``model_id``,
    and manifest, so a governed program built over either yields the SAME
    content-addressed artifact id. It exposes ``stream_plan`` for the streamed
    governed loop and a ``plan`` that drains ``stream_plan`` through the neutral
    reducer, so the SAME planner also satisfies the non-streaming
    :class:`ToolPlanner` contract used by ``execute`` and ``_definition_extra``.
    """

    kind = "model"

    def __init__(self, generator: StreamingToolCallingGenerator, manifest: Sequence[ToolSpec]) -> None:
        self._generator = generator
        self._manifest = tuple(manifest)

    @property
    def model_id(self) -> str:
        return self._generator.model_id

    def manifest(self) -> tuple[ToolSpec, ...]:
        return self._manifest

    def tools(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._manifest)

    def stream_plan(self, query: str, used: frozenset[str]) -> Iterator[ToolCallDelta]:
        available = tuple(spec for spec in self._manifest if spec.name not in used)
        if not available:
            return
        yield from self._generator.stream_tool_call(query, available, used)

    def plan(self, query: str, used: frozenset[str]) -> ToolProposal | None:
        accumulator = _ToolCallAccumulator()
        for delta in self.stream_plan(query, used):
            accumulator.feed(delta)
        return accumulator.result()

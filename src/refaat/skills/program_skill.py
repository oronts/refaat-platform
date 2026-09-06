from __future__ import annotations

from refaat.domain.authz import Principal
from refaat.domain.models import ContextPacket, JsonObject
from refaat.errors import AuthorizationError
from refaat.programs.registry import BoundProgram


class ProgramSkill:
    """Expose a promoted program as a single governed, registry-resolvable tool.

    A :class:`ProgramSkill` wraps a :class:`BoundProgram`, so a
    ``GovernedReActProgram`` can call an entire program (its full retrieval,
    generation, caching, and citation pipeline) as one tool. ``run`` re-enters
    that pipeline with the retrieved query and returns the sub-program's grounded
    answer, citations, and pinned inner artifact id as the observation.
    """

    is_program_skill = True

    def __init__(self, bound: BoundProgram, *, name: str | None = None, description: str | None = None) -> None:
        self._bound = bound
        # Capture the inner artifact id once, at construction, so the tool identity is
        # bound immutably to the program as promoted rather than recomputed from a
        # bound program that could be mutated later.
        self._artifact_id = bound.program.artifact().artifact_id
        self.name = name or f"program_{bound.name.replace('-', '_')}"
        self.description = description or f"Run the {bound.name} program as a governed tool."

    @property
    def artifact_id(self) -> str:
        # The construction-captured inner id (a property, never a to_json key:
        # artifact_id is the digest OF to_json, so to_json["artifact_id"] KeyErrors).
        return self._artifact_id

    def run(self, context: ContextPacket, arguments: JsonObject, principal: Principal | None = None) -> JsonObject:
        del arguments
        # Fail closed rather than let the inner run re-retrieve unfiltered under an active policy.
        if self._bound.project.runtime.access_policy is not None and principal is None:
            raise AuthorizationError("A program skill requires a principal while an access policy is active")
        result = self._bound.run(question=context.query, principal=principal)
        return result.to_json()

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from refaat.domain.authz import Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import JsonObject, QueryResponse
from refaat.errors import ValidationError
from refaat.programs.policy import REFINE, BoundedInferencePolicy
from refaat.programs.program import Program, ProgramResult, Tunables, ValidationOutcome, Validator
from refaat.programs.schema import validate_structured
from refaat.programs.signature import Signature

if TYPE_CHECKING:
    from refaat.application.runtime import RefaatProject


@dataclass(frozen=True, slots=True)
class StructuredValidator:
    """A :class:`Validator` that enforces a typed output Signature.

    Projects the runtime response into the structured object the program layer
    already exposes (the grounded answer plus its citation chunk ids) and checks
    it against the compiled output schema through the single shared
    :func:`validate_structured` body. It only adds structural typing: grounding
    and the citation contract are enforced by the grounding validator and the
    runtime, unchanged.
    """

    output_signature: type[Signature]
    name: str = "structured-output"

    def check(self, response: QueryResponse) -> ValidationOutcome:
        projected: JsonObject = {
            "answer": response.answer,
            "citations": [citation.chunk_id for citation in response.citations],
        }
        ok, detail = validate_structured(projected, self.output_signature.json_schema())
        return ValidationOutcome(validator=self.name, passed=ok, detail=detail)


class StructuredOutputProgram(Program):
    """Wrap an inner program with typed-output schema enforcement.

    The inner program is cloned with an extra :class:`StructuredValidator` and
    the compiled output schema, then driven through the shipped
    :class:`BoundedInferencePolicy` in ``refine`` mode up to a hard attempt cap:
    a schema violation flips the result ungrounded and the refine loop retries,
    reusing the shipped retry and selection verbatim. The schema digest, the
    attempt cap, and the inner artifact id fold into the content-addressed
    artifact, so a schema-enforced program is a distinct, reproducible artifact
    while every schema-free program id is unchanged.

    On the offline extractive stack the deterministic output is validated so the
    machinery is genuinely exercised; a real schema-capable provider additionally
    receives a schema-constrained generation request through the runtime.
    """

    name = "structured-output"

    def __init__(self, inner: Program, *, max_attempts: int = 3, tunables: Tunables | None = None) -> None:
        super().__init__(tunables=tunables or inner.tunables)
        self._inner = inner
        self._max_attempts = max_attempts
        self._validator = StructuredValidator(inner.output_signature)
        schema = inner.output_signature.json_schema()
        # Thread this program's tunables into the inner that execute actually runs, so
        # the executed request matches the artifact identity. A default StructuredOutput
        # inherits the inner's own tunables, so with_tunables is a no-op-equivalent and
        # the executed request and artifact id stay byte-identical.
        tuned_inner = inner.with_tunables(self.tunables)
        self._enforced = tuned_inner.with_extra_validators((self._validator,)).with_response_schema(schema)
        self._policy = BoundedInferencePolicy(self._enforced, mode=REFINE, attempts=max_attempts)
        self.name = f"{inner.name}+structured"
        self.input_signature = inner.input_signature
        self.output_signature = inner.output_signature

    @property
    def shadow_safe(self) -> bool:
        # A structure wrapper adds only validation; its side-effect profile is the wrapped program's.
        return self._inner.shadow_safe

    def declared_side_effect(self) -> SideEffectClass:
        return self._inner.declared_side_effect()

    def steps(self) -> tuple[str, ...]:
        return (*self._inner.steps(), "structured")

    def validators(self) -> tuple[Validator, ...]:
        return self._enforced.validators()

    def _definition_extra(self) -> JsonObject:
        return {
            "structured": {
                "schema_digest": self.output_signature.output_schema_digest(),
                "max_attempts": self._max_attempts,
                "inner": self._inner.artifact().artifact_id,
            }
        }

    def execute(
        self,
        project: RefaatProject,
        *,
        inputs: Mapping[str, Any],
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> ProgramResult:
        return self._policy.execute(project, inputs=inputs, user_id=user_id, session_id=session_id, principal=principal)

    def stream(
        self,
        project: RefaatProject,
        *,
        inputs: Mapping[str, Any],
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> Iterator[dict[str, Any]]:
        # execute drives a refine loop that selects among attempts, and no single
        # request reproduces that selection, so refuse streaming rather than answer
        # as one bare attempt with the wrong cache identity.
        raise ValidationError("StructuredOutputProgram does not support streaming; call run() for the selected result")

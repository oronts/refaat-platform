from __future__ import annotations

import abc
import copy
import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Protocol

from refaat.core.citations import CITATION_PATTERN
from refaat.core.harness import BASE_SYSTEM_PROMPT_HASH
from refaat.core.text import sentence_spans
from refaat.domain.authz import Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import Citation, Demonstration, JsonObject, QueryRequest, QueryResponse
from refaat.errors import ValidationError
from refaat.programs.artifact import ProgramArtifact
from refaat.programs.signature import InputField, OutputField, Signature
from refaat.version import __version__

if TYPE_CHECKING:
    from refaat.application.runtime import RefaatProject


@dataclass(frozen=True, slots=True)
class Tunables:
    """Parameters a program exposes for later optimization.

    These are the only knobs an optimizer may propose changing; the immutable
    grounding policy and citation contract are never tunable.
    """

    top_k: int | None = None
    token_budget: int | None = None
    minimum_citation_coverage: float = 0.0
    fusion: str | None = None
    rrf_k: int | None = None
    autocut_enabled: bool | None = None
    autocut_threshold: float | None = None
    instructions: str | None = None
    demonstrations: tuple[Demonstration, ...] = ()
    router: JsonObject | None = None

    def __post_init__(self) -> None:
        # An empty instructions string carries no instruction and as_json omits it,
        # so normalize it to None at construction. Without this, "" would round-trip
        # through as_json/from_json to None and break Tunables equality (and the
        # durable resume's winner-identity check) for a caller that passed "".
        if self.instructions == "":
            object.__setattr__(self, "instructions", None)

    def as_json(self) -> JsonObject:
        payload: JsonObject = {
            "top_k": self.top_k,
            "token_budget": self.token_budget,
            "minimum_citation_coverage": self.minimum_citation_coverage,
        }
        if self.fusion is not None:
            payload["fusion"] = self.fusion
        if self.rrf_k is not None:
            payload["rrf_k"] = self.rrf_k
        if self.autocut_enabled is not None:
            payload["autocut_enabled"] = self.autocut_enabled
        if self.autocut_threshold is not None:
            payload["autocut_threshold"] = self.autocut_threshold
        if self.instructions:
            payload["instructions"] = self.instructions
        if self.demonstrations:
            payload["demonstrations"] = [demonstration.as_json() for demonstration in self.demonstrations]
        if self.router is not None:
            payload["router"] = self.router
        return payload

    @classmethod
    def from_json(cls, payload: JsonObject) -> Tunables:
        """Reconstruct a :class:`Tunables` from its :meth:`as_json` form.

        A faithful inverse of :meth:`as_json`: ``.get`` defaults round-trip the
        conditional emission, demonstrations rebuild from their ``[{question,
        answer}]`` records, and the router config passes through unchanged. This
        is what lets a durable optimization run replay the exact materialized
        candidate set it persisted.
        """
        raw_demonstrations = payload.get("demonstrations") or ()
        demonstrations = tuple(
            Demonstration(question=str(item["question"]), answer=str(item["answer"])) for item in raw_demonstrations
        )
        router = payload.get("router")
        return cls(
            top_k=payload.get("top_k"),
            token_budget=payload.get("token_budget"),
            minimum_citation_coverage=payload.get("minimum_citation_coverage", 0.0),
            fusion=payload.get("fusion"),
            rrf_k=payload.get("rrf_k"),
            autocut_enabled=payload.get("autocut_enabled"),
            autocut_threshold=payload.get("autocut_threshold"),
            instructions=payload.get("instructions"),
            demonstrations=demonstrations,
            router=router,
        )


def _schema_digest(schema: JsonObject) -> str:
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    validator: str
    passed: bool
    detail: str


class Validator(Protocol):
    @property
    def name(self) -> str: ...

    def check(self, response: QueryResponse) -> ValidationOutcome: ...


def _cited_sentence_coverage(answer: str, resolved_ids: set[str]) -> tuple[int, int]:
    """Return (covered, total) answer sentences for a real citation-coverage fraction.

    Markers become a space before splitting so a boundary marker cannot merge two sentences, only a
    resolved id counts so a dangling marker cannot inflate coverage, and a marker attributes to the
    sentence whose content begins before it. Granularity is the shared Latin/ASCII sentence splitter's;
    non-Latin punctuation may under-segment, which only makes the metric more lenient.
    """
    segments: list[str] = []
    resolved_positions: list[int] = []
    cursor = 0
    stripped_length = 0
    for match in CITATION_PATTERN.finditer(answer):
        segment = answer[cursor : match.start()]
        segments.append(segment)
        stripped_length += len(segment)
        if match.group(1) in resolved_ids:
            resolved_positions.append(stripped_length)
        segments.append(" ")
        stripped_length += 1
        cursor = match.end()
    segments.append(answer[cursor:])
    stripped = "".join(segments)
    content_starts: list[int] = []
    for start, end in sentence_spans(stripped):
        offset = next((i for i, char in enumerate(stripped[start:end]) if char.isalnum()), None)
        if offset is not None:
            content_starts.append(start + offset)
    if not content_starts:
        return 0, 0
    covered: set[int] = set()
    for position in resolved_positions:
        index: int | None = None
        # Strict <: a marker at a sentence's content start is a prefix of it, so it cites the preceding one.
        for i, content_start in enumerate(content_starts):
            if content_start < position:
                index = i
            else:
                break
        if index is not None:
            covered.add(index)
    return len(covered), len(content_starts)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if is_dataclass(value):
        return _validator_config(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        # Coerce keys too (int 1 and str "1" must stay distinct) and emit sorted [key, value] pairs, so a
        # non-string or unstringifiable key can neither collide, embed an address, nor raise.
        pairs = [[_json_safe(key), _json_safe(item)] for key, item in value.items()]
        return sorted(pairs, key=lambda pair: json.dumps(pair, sort_keys=True, ensure_ascii=False))
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _validator_config(validator: Any) -> JsonObject:
    # A non-dataclass validator exposes no structured config, so it folds by class identity alone.
    if not is_dataclass(validator) or isinstance(validator, type):
        return {}
    return {f.name: _json_safe(getattr(validator, f.name)) for f in fields(validator)}


@dataclass(frozen=True, slots=True)
class GroundingValidator:
    minimum_citation_coverage: float = 0.0
    name: str = "grounding"

    def check(self, response: QueryResponse) -> ValidationOutcome:
        count = len(response.citations)
        if self.minimum_citation_coverage <= 0.0:
            # Grounding is satisfied by any resolved citation, or a deliberately empty (abstaining) answer.
            # Byte-identical to the pre-coverage behavior, so a program that leaves the floor at 0.0 is
            # unchanged, keeping every existing artifact id stable.
            passed = count > 0 or not response.answer.strip()
            return ValidationOutcome(validator=self.name, passed=passed, detail=f"{count} citation(s)")
        # A positive floor is a real fraction, not a toggle: the share of answer sentences that carry a
        # resolved citation must meet it. Only a citation whose id is in the resolved set counts (so a
        # dangling marker in a rewritten answer can never inflate coverage), and markers are stripped
        # before sentences are split (so a marker at a sentence boundary cannot collapse two into one).
        resolved_ids = {citation.chunk_id for citation in response.citations}
        covered, total = _cited_sentence_coverage(response.answer, resolved_ids)
        if total == 0:
            passed = not response.answer.strip()
            return ValidationOutcome(validator=self.name, passed=passed, detail="empty answer")
        coverage = covered / total
        passed = coverage >= self.minimum_citation_coverage
        return ValidationOutcome(
            validator=self.name, passed=passed, detail=f"{covered}/{total} sentences cited, coverage {coverage:.2f}"
        )


# react_steps carries ungoverned raw tool observations, so it is not serialized to a caller.
_UNSERIALIZED_OUTPUT_KEYS = frozenset({"react_steps"})


@dataclass(frozen=True, slots=True)
class ProgramResult:
    program: str
    artifact_id: str
    answer: str
    citations: tuple[Citation, ...]
    routes: tuple[str, ...]
    cached: bool
    corpus_version: int
    trace_id: str
    outputs: JsonObject
    validations: tuple[ValidationOutcome, ...]
    usage: JsonObject = field(default_factory=dict)

    @property
    def grounded(self) -> bool:
        return all(outcome.passed for outcome in self.validations)

    def to_json(self) -> JsonObject:
        return {
            "program": self.program,
            "artifact_id": self.artifact_id,
            "answer": self.answer,
            "citations": [
                {"chunk_id": citation.chunk_id, "document_id": citation.document_id, "label": citation.label}
                for citation in self.citations
            ],
            "routes": list(self.routes),
            "cached": self.cached,
            "corpus_version": self.corpus_version,
            "trace_id": self.trace_id,
            "grounded": self.grounded,
            "validations": [
                {"validator": outcome.validator, "passed": outcome.passed, "detail": outcome.detail}
                for outcome in self.validations
            ],
            "outputs": {key: value for key, value in self.outputs.items() if key not in _UNSERIALIZED_OUTPUT_KEYS},
            "usage": self.usage,
        }


def _has_pure_execute(program: Program, pure_class: type[Program]) -> bool:
    # A subclass that overrides execute() may add side effects the pure assertion does not cover.
    return type(program).execute is pure_class.execute


class Program(abc.ABC):
    """A typed, content-addressed AI program executed by the Refaat runtime.

    Subclasses declare an input and output :class:`Signature` and implement
    :meth:`execute`, which must reuse the runtime rather than reimplement
    retrieval, generation, caching, or citation logic.
    """

    name: str
    input_signature: type[Signature]
    output_signature: type[Signature]
    backend: str = "native"

    def __init__(self, *, tunables: Tunables | None = None) -> None:
        self.tunables = tunables or Tunables()
        self._extra_validators: tuple[Validator, ...] = ()
        self.response_schema: JsonObject | None = None

    def steps(self) -> tuple[str, ...]:
        return ()

    @property
    def shadow_safe(self) -> bool:
        """Whether an unserved shadow run of this program has no durable side effect.

        A shadow or canary run executes the unserved artifact under a scope that
        suppresses the session, answer-cache, and model-call-cache writes. That
        scope reaches the runtime's own writes, so a pure retrieval-and-answer
        program is safe to shadow. The base default is False (safe-by-default): a
        program must assert that it produces no durable row or external effect the
        scope cannot neutralize, so a custom program that forgot to declare its
        side-effect profile is never silently canaried. A pure program overrides
        this to True; a wrapper delegates it to whatever it wraps, so it is
        shadow-safe only when every wrapped program is.
        """
        return False

    def declared_side_effect(self) -> SideEffectClass:
        # Safe-by-default: an undeclared program is treated as maximally side-effectful, so a pure program
        # must assert read-only and a governed one derives it from its grants and capabilities.
        return SideEffectClass.EXTERNAL

    def validators(self) -> tuple[Validator, ...]:
        return (GroundingValidator(self.tunables.minimum_citation_coverage), *self._extra_validators)

    def artifact(self) -> ProgramArtifact:
        return ProgramArtifact(
            program=self.name,
            backend=self.backend,
            definition_hash=self._definition_hash(),
            immutable_policy_hash=BASE_SYSTEM_PROMPT_HASH,
            signatures={
                "input": self.input_signature.signature_hash(),
                "output": self.output_signature.signature_hash(),
            },
            tunables=self.tunables.as_json(),
            validators=tuple(validator.name for validator in self.validators()),
            dependency_versions={"refaat": __version__},
        )

    def _definition_extra(self) -> JsonObject:
        """Subclass policy folded into the content-addressed definition hash.

        Override to pin a subclass's own policy (an inference policy, a tool
        governance policy) into the artifact id, so two otherwise identical
        programs with different policies are distinct, reproducible artifacts.
        The base program pins nothing; the response schema and extra validators
        are folded separately by :meth:`_validators_fold` so a subclass override
        can never drop them.
        """
        return {}

    def _validators_fold(self) -> JsonObject:
        # The response schema and extra validators change the grounded verdict, so they are folded into the
        # definition hash independently of _definition_extra, which a subclass may replace without calling super.
        fold: JsonObject = {}
        if self.response_schema is not None:
            fold["response_schema"] = _schema_digest(self.response_schema)
        if self._extra_validators:
            fold["extra_validators"] = [
                {
                    "class": f"{type(validator).__module__}.{type(validator).__qualname__}",
                    "config": _validator_config(validator),
                }
                for validator in self._extra_validators
            ]
        return fold

    def _definition_hash(self) -> str:
        payload = {
            "program": self.name,
            "backend": self.backend,
            "steps": list(self.steps()),
            "input": self.input_signature.signature_hash(),
            "output": self.output_signature.signature_hash(),
            "tunables": self.tunables.as_json(),
        }
        extra = {**self._definition_extra(), **self._validators_fold()}
        if extra:
            payload["extra"] = extra
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _result(self, response: QueryResponse) -> ProgramResult:
        validations = tuple(validator.check(response) for validator in self.validators())
        return ProgramResult(
            program=self.name,
            artifact_id=self.artifact().artifact_id,
            answer=response.answer,
            citations=response.citations,
            routes=response.routes,
            cached=response.cached,
            corpus_version=response.corpus_version,
            trace_id=response.trace_id,
            outputs={
                "answer": response.answer,
                "citations": [citation.chunk_id for citation in response.citations],
            },
            validations=validations,
            usage=response.usage,
        )

    def with_tunables(self, tunables: Tunables) -> Program:
        clone = copy.copy(self)
        clone.tunables = tunables
        return clone

    def with_extra_validators(self, validators: tuple[Validator, ...]) -> Program:
        """Return a clone carrying additional validators on top of grounding.

        The default extra tuple is empty, so an unused seam leaves ``validators``
        and the artifact id byte-identical. Delegating wrappers surface their own
        extra validators too, so enforcing on a wrapped inner is never silently
        dropped.
        """
        clone = copy.copy(self)
        clone._extra_validators = tuple(validators)
        return clone

    def with_response_schema(self, schema: JsonObject) -> Program:
        """Return a clone that carries a structured output schema.

        The schema threads into the generation request through the runtime and
        folds into both the content-addressed artifact and the answer-cache
        identity, conditional on presence.
        """
        clone = copy.copy(self)
        clone.response_schema = schema
        return clone

    def _query_request(
        self,
        project: RefaatProject,
        question: str,
        *,
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> QueryRequest:
        """Build the answer request shared by the run and stream paths.

        Both ``execute`` and ``stream`` route the final grounded answer through
        this one request, so the streamed answer folds into the identical
        answer-cache key as the run path. It mirrors the fields
        ``RefaatProject.ask`` forwards, so the cache identity is byte-identical.

        The ``principal`` is a runtime authorization input carried on the request,
        so a program answers under the same ACL as a direct query. It never enters
        the program artifact identity; it reaches the answer-cache identity only
        through the runtime's existing auth scope, which folds in solely when a
        policy and principal are both active. With no policy the request is byte-identical to
        before; under an active policy a principal-less run is denied by default under strict
        authorization.
        """
        return QueryRequest(
            workspace_id=project.workspace_id,
            corpus_id=project.corpus_id,
            query=question,
            user_id=user_id,
            session_id=session_id,
            top_k=self.tunables.top_k,
            token_budget=self.tunables.token_budget,
            fusion=self.tunables.fusion,
            rrf_k=self.tunables.rrf_k,
            autocut_enabled=self.tunables.autocut_enabled,
            autocut_threshold=self.tunables.autocut_threshold,
            instructions=self.tunables.instructions,
            demonstrations=self.tunables.demonstrations,
            response_schema=self.response_schema,
            principal=principal,
        )

    @abc.abstractmethod
    def execute(
        self,
        project: RefaatProject,
        *,
        inputs: Mapping[str, Any],
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> ProgramResult: ...

    def stream(
        self,
        project: RefaatProject,
        *,
        inputs: Mapping[str, Any],
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream the program's grounded answer as SSE-shaped events.

        The default streams the answer only, through the single unchanged
        ``QueryService.stream_query`` pipeline, so retrieval, generation,
        caching, and citations are identical to :meth:`execute`. A program with
        a governed trajectory overrides this to emit its steps first.
        """
        question = inputs.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValidationError("Program input 'question' is required and must be a non-empty string")
        request = self._query_request(project, question, user_id=user_id, session_id=session_id, principal=principal)
        yield from project.runtime.stream_query(request)


class GroundedRagInput(Signature):
    question: str = InputField(description="The user question to answer from the corpus.")


class GroundedRagOutput(Signature):
    answer: str = OutputField(description="Answer grounded in retrieved evidence.", grounded=True)
    citations: list[str] = OutputField(description="Canonical citations supporting the answer.", required=True)


class GroundedRagProgram(Program):
    """Built-in grounded retrieval-augmented answer program.

    Delegates to the runtime's full query pipeline, so RAG, GraphRAG, CAG,
    version-safe caching, and citation validation all apply unchanged; the
    program layer adds a typed contract, tunables, validators, and a
    content-addressed artifact on top.
    """

    name = "grounded-rag"
    input_signature = GroundedRagInput
    output_signature = GroundedRagOutput

    @property
    def shadow_safe(self) -> bool:
        return _has_pure_execute(self, GroundedRagProgram)

    def declared_side_effect(self) -> SideEffectClass:
        if _has_pure_execute(self, GroundedRagProgram):
            return SideEffectClass.READ_ONLY
        return super().declared_side_effect()

    def steps(self) -> tuple[str, ...]:
        return ("retrieve", "predict", "verify_grounding")

    def execute(
        self,
        project: RefaatProject,
        *,
        inputs: Mapping[str, Any],
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> ProgramResult:
        question = inputs.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValidationError("Program input 'question' is required and must be a non-empty string")
        response = project.runtime.query(
            self._query_request(project, question, user_id=user_id, session_id=session_id, principal=principal)
        )
        return self._result(response)

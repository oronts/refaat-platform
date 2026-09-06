from __future__ import annotations

from dataclasses import dataclass

from refaat.canonical import canonical_digest
from refaat.domain.models import JsonObject

ARTIFACT_SCHEMA_VERSION = "1"


@dataclass(frozen=True, slots=True)
class ProgramArtifact:
    """Immutable, content-addressed program definition.

    Artifacts are canonical JSON only: no pickle, credentials, or executable
    payload. The digest is derived from the serialized contents so identical
    definitions always resolve to the same ``artifact_id``.
    """

    program: str
    backend: str
    definition_hash: str
    immutable_policy_hash: str
    signatures: JsonObject
    tunables: JsonObject
    validators: tuple[str, ...]
    dependency_versions: JsonObject

    @property
    def artifact_id(self) -> str:
        return f"sha256:{self._digest()}"

    @property
    def behavior_hash(self) -> str:
        # Answer-affecting identity (definition + immutable policy) only; excludes version, never enters _digest.
        return canonical_digest(
            {"definition_hash": self.definition_hash, "immutable_policy_hash": self.immutable_policy_hash}
        )

    def to_json(self) -> JsonObject:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "program": self.program,
            "backend": self.backend,
            "definition_hash": self.definition_hash,
            "immutable_policy_hash": self.immutable_policy_hash,
            "signatures": self.signatures,
            "tunables": self.tunables,
            "validators": list(self.validators),
            "dependency_versions": self.dependency_versions,
        }

    def _digest(self) -> str:
        return canonical_digest(self.to_json())

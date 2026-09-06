from __future__ import annotations

from typing import TYPE_CHECKING

from refaat.domain.models import JsonObject
from refaat.errors import NotFoundError, ValidationError
from refaat.infrastructure.database.store import SQLStore

if TYPE_CHECKING:
    from refaat.programs.artifact import ProgramArtifact

ENVIRONMENTS = frozenset({"development", "staging", "production"})
PRODUCTION = "production"


class RegistryService:
    """Durable, workspace-scoped store of immutable program artifacts and aliases.

    Promotion appends a deployment pointing an environment alias at an artifact;
    it never mutates the artifact. Rollback appends a deployment back to the most
    recent previously deployed artifact, so both operations are atomic and
    auditable through the append-only deployment history.
    """

    def __init__(self, store: SQLStore) -> None:
        self.store = store

    def save(self, workspace_id: str, artifact: ProgramArtifact) -> str:
        self.store.save_program_artifact(
            workspace_id, artifact.program, artifact.artifact_id, artifact.definition_hash, artifact.to_json()
        )
        return artifact.artifact_id

    def get(self, workspace_id: str, artifact_id: str) -> JsonObject:
        document = self.store.get_program_artifact(workspace_id, artifact_id)
        if document is None:
            raise NotFoundError(f"Artifact is not registered: {artifact_id}")
        return document

    def _require_program_artifact(self, workspace_id: str, program: str, artifact_id: str) -> JsonObject:
        document = self.store.get_program_artifact(workspace_id, artifact_id)
        if document is None:
            raise NotFoundError(f"Artifact is not registered: {artifact_id}")
        if str(document.get("program")) != program:
            raise ValidationError("Artifact does not belong to this program")
        return document

    def approve(
        self, workspace_id: str, program: str, environment: str, artifact_id: str, approver: str = "system"
    ) -> JsonObject:
        self._require_environment(environment)
        self._require_program_artifact(workspace_id, program, artifact_id)
        return self.store.record_approval(workspace_id, program, environment, artifact_id, approver)

    def promote(self, workspace_id: str, program: str, environment: str, artifact_id: str) -> JsonObject:
        self._require_environment(environment)
        self._require_program_artifact(workspace_id, program, artifact_id)
        if environment == PRODUCTION and not self.store.is_approved(workspace_id, program, PRODUCTION, artifact_id):
            raise ValidationError("Artifact must be approved for production before promotion")
        return self.store.record_deployment(workspace_id, program, environment, artifact_id, "promote")

    def resolve(self, workspace_id: str, program: str, environment: str) -> JsonObject:
        self._require_environment(environment)
        current = self.store.current_deployment(workspace_id, program, environment)
        if current is None:
            raise NotFoundError(f"No deployment for {program} in {environment}")
        return self.get(workspace_id, str(current["artifact_id"]))

    def rollback(self, workspace_id: str, program: str, environment: str) -> JsonObject:
        self._require_environment(environment)
        history = self.store.deployment_history(workspace_id, program, environment)
        if not history:
            raise NotFoundError(f"No deployment for {program} in {environment}")
        current_artifact = str(history[0]["artifact_id"])
        prior = next((item for item in history[1:] if str(item["artifact_id"]) != current_artifact), None)
        if prior is None:
            raise ValidationError("No previous artifact to roll back to")
        return self.store.record_deployment(workspace_id, program, environment, str(prior["artifact_id"]), "rollback")

    def history(self, workspace_id: str, program: str, environment: str) -> list[JsonObject]:
        self._require_environment(environment)
        return self.store.deployment_history(workspace_id, program, environment)

    def resolve_deployment(self, workspace_id: str, program: str, environment: str) -> tuple[str, JsonObject]:
        self._require_environment(environment)
        current = self.store.current_deployment(workspace_id, program, environment)
        if current is None:
            raise NotFoundError(f"No deployment for {program} in {environment}")
        artifact_id = str(current["artifact_id"])
        return artifact_id, self.get(workspace_id, artifact_id)

    def set_canary(
        self,
        workspace_id: str,
        program: str,
        environment: str,
        candidate_artifact_id: str,
        percent: int,
        compare: bool = True,
    ) -> JsonObject:
        self._require_environment(environment)
        if not 0 <= percent <= 100:
            raise ValidationError("Canary percent must be between 0 and 100")
        self._require_program_artifact(workspace_id, program, candidate_artifact_id)
        current = self.store.current_deployment(workspace_id, program, environment)
        if current is None:
            raise ValidationError("An active deployment is required before attaching a canary")
        if candidate_artifact_id == str(current["artifact_id"]):
            raise ValidationError("Canary candidate must differ from the active artifact")
        if environment == PRODUCTION and not self.store.is_approved(
            workspace_id, program, PRODUCTION, candidate_artifact_id
        ):
            raise ValidationError("Candidate must be approved for production before attaching a canary")
        return self.store.set_canary(workspace_id, program, environment, candidate_artifact_id, percent, compare)

    def clear_canary(self, workspace_id: str, program: str, environment: str) -> JsonObject:
        self._require_environment(environment)
        return self.store.clear_canary(workspace_id, program, environment)

    def current_canary(self, workspace_id: str, program: str, environment: str) -> JsonObject | None:
        self._require_environment(environment)
        return self.store.current_canary(workspace_id, program, environment)

    def canary_history(self, workspace_id: str, program: str, environment: str) -> list[JsonObject]:
        self._require_environment(environment)
        return self.store.canary_history(workspace_id, program, environment)

    @staticmethod
    def _require_environment(environment: str) -> None:
        if environment not in ENVIRONMENTS:
            raise ValidationError(f"Unknown environment: {environment}")


class BoundRegistry:
    """A :class:`RegistryService` bound to a single workspace."""

    def __init__(self, service: RegistryService, workspace_id: str) -> None:
        self._service = service
        self._workspace_id = workspace_id

    def save(self, artifact: ProgramArtifact) -> str:
        return self._service.save(self._workspace_id, artifact)

    def get(self, artifact_id: str) -> JsonObject:
        return self._service.get(self._workspace_id, artifact_id)

    def approve(self, program: str, environment: str, artifact_id: str, approver: str = "system") -> JsonObject:
        return self._service.approve(self._workspace_id, program, environment, artifact_id, approver)

    def promote(self, program: str, environment: str, artifact_id: str) -> JsonObject:
        return self._service.promote(self._workspace_id, program, environment, artifact_id)

    def resolve(self, program: str, environment: str) -> JsonObject:
        return self._service.resolve(self._workspace_id, program, environment)

    def rollback(self, program: str, environment: str) -> JsonObject:
        return self._service.rollback(self._workspace_id, program, environment)

    def history(self, program: str, environment: str) -> list[JsonObject]:
        return self._service.history(self._workspace_id, program, environment)

    def set_canary(
        self, program: str, environment: str, candidate_artifact_id: str, percent: int, compare: bool = True
    ) -> JsonObject:
        return self._service.set_canary(
            self._workspace_id, program, environment, candidate_artifact_id, percent, compare
        )

    def clear_canary(self, program: str, environment: str) -> JsonObject:
        return self._service.clear_canary(self._workspace_id, program, environment)

    def current_canary(self, program: str, environment: str) -> JsonObject | None:
        return self._service.current_canary(self._workspace_id, program, environment)

    def canary_history(self, program: str, environment: str) -> list[JsonObject]:
        return self._service.canary_history(self._workspace_id, program, environment)

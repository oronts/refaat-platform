from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat
from refaat.errors import NotFoundError, ValidationError
from refaat.programs import GroundedRagProgram, Tunables

pytestmark = pytest.mark.integration


def test_registry_promote_resolve_rollback(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    base = GroundedRagProgram().artifact()
    tuned = GroundedRagProgram(tunables=Tunables(top_k=4)).artifact()
    registry = project.registry

    base_id = registry.save(base)
    tuned_id = registry.save(tuned)
    assert base_id != tuned_id
    assert registry.save(base) == base_id

    registry.approve("grounded-rag", "production", base_id)
    registry.promote("grounded-rag", "production", base_id)
    assert registry.resolve("grounded-rag", "production")["definition_hash"] == base.definition_hash

    registry.approve("grounded-rag", "production", tuned_id)
    registry.promote("grounded-rag", "production", tuned_id)
    assert registry.resolve("grounded-rag", "production")["definition_hash"] == tuned.definition_hash

    rolled = registry.rollback("grounded-rag", "production")
    assert rolled["artifact_id"] == base_id
    assert rolled["action"] == "rollback"
    assert registry.resolve("grounded-rag", "production")["definition_hash"] == base.definition_hash
    assert len(registry.history("grounded-rag", "production")) == 3

    with pytest.raises(ValidationError):
        registry.promote("grounded-rag", "unknown-env", base_id)
    with pytest.raises(NotFoundError):
        registry.resolve("grounded-rag", "staging")


def test_registry_workspace_isolation(runtime: Refaat) -> None:
    tenant_a = runtime.project(workspace="tenant-a", corpus="c")
    tenant_b = runtime.project(workspace="tenant-b", corpus="c")
    artifact = GroundedRagProgram().artifact()

    tenant_a.registry.save(artifact)
    tenant_a.registry.approve("grounded-rag", "production", artifact.artifact_id)
    tenant_a.registry.promote("grounded-rag", "production", artifact.artifact_id)
    assert tenant_a.registry.resolve("grounded-rag", "production")["program"] == "grounded-rag"

    with pytest.raises(NotFoundError):
        tenant_b.registry.promote("grounded-rag", "production", artifact.artifact_id)
    with pytest.raises(NotFoundError):
        tenant_b.registry.resolve("grounded-rag", "production")


def test_production_promotion_requires_approval(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    artifact = GroundedRagProgram().artifact()
    registry = project.registry
    registry.save(artifact)

    registry.promote("grounded-rag", "staging", artifact.artifact_id)
    with pytest.raises(ValidationError):
        registry.promote("grounded-rag", "production", artifact.artifact_id)

    approval = registry.approve("grounded-rag", "production", artifact.artifact_id, "reviewer")
    assert approval["approver"] == "reviewer"
    registry.promote("grounded-rag", "production", artifact.artifact_id)
    assert registry.resolve("grounded-rag", "production")["program"] == "grounded-rag"

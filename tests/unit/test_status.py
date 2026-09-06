"""The admin status plane value objects: scalar-only detail, stable serialization, constructors."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from refaat.domain.status import (
    AdminStatus,
    BuildInfo,
    SchemaInfo,
    SubsystemState,
    SubsystemStatus,
)


def test_subsystem_state_values() -> None:
    assert SubsystemState.NOT_CONFIGURED.value == "not_configured"
    assert SubsystemState.ACTIVE.value == "active"
    assert SubsystemState.ENGAGED.value == "engaged"
    assert SubsystemState.DEGRADED.value == "degraded"
    assert SubsystemState.UNKNOWN.value == "unknown"
    assert SubsystemState.UNMANAGED.value == "unmanaged"


def test_not_configured_and_unknown_shapes() -> None:
    nc = SubsystemStatus.not_configured("audit")
    assert nc.state is SubsystemState.NOT_CONFIGURED
    assert nc.configured is False
    assert nc.detail == {}
    unk = SubsystemStatus.unknown("audit")
    assert unk.state is SubsystemState.UNKNOWN
    assert unk.configured is False
    assert unk.detail == {}


def test_detail_accepts_scalars_and_scalar_tuples() -> None:
    status = SubsystemStatus(
        name="quotas",
        state=SubsystemState.ACTIVE,
        configured=True,
        detail={"rule_count": 3, "enabled": True, "weight": 0.5, "note": "x", "empty": None, "rules": ("a", "b")},
    )
    assert status.to_json()["detail"]["rules"] == ["a", "b"]


def test_detail_rejects_non_scalar_value() -> None:
    with pytest.raises(TypeError):
        SubsystemStatus(
            name="bad",
            state=SubsystemState.ACTIVE,
            configured=True,
            detail={"nested": {"k": "v"}},  # type: ignore[dict-item]
        )


def test_detail_rejects_non_scalar_tuple_element() -> None:
    with pytest.raises(TypeError):
        SubsystemStatus(
            name="bad",
            state=SubsystemState.ACTIVE,
            configured=True,
            detail={"rules": ({"k": "v"},)},  # type: ignore[dict-item]
        )


def test_admin_status_to_json_is_stable() -> None:
    generated = datetime(2026, 8, 28, 12, 0, 0, tzinfo=UTC)
    plane = AdminStatus(
        generated_at=generated,
        scope="workspace",
        workspace_id="ws-1",
        build=BuildInfo(version="0.1.0", environment="test", features={"quotas": False}, state=SubsystemState.ACTIVE),
        schema=SchemaInfo(
            code_head="0009", db_revision="0009", at_head=True, managed=True, state=SubsystemState.ACTIVE
        ),
        subsystems=(SubsystemStatus.not_configured("audit"),),
    )
    payload = plane.to_json()
    assert payload["generated_at"] == generated.isoformat()
    assert payload["scope"] == "workspace"
    assert payload["workspace_id"] == "ws-1"
    assert payload["build"]["state"] == "active"
    assert payload["schema"]["at_head"] is True
    assert payload["subsystems"][0] == {
        "name": "audit",
        "state": "not_configured",
        "configured": False,
        "detail": {},
    }

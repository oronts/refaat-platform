"""Approver-loses-permission re-resolution for durable approvals.

With a PrincipalDirectory installed, the governed consume intersects each approval's grant-time role
snapshot with the approver's current roles, so a revoked role no longer satisfies (and a role gained
after grant cannot retroactively validate it), while an unknown or revoked approver, an id mismatch, a
non-Principal return, or an unavailable directory drops the approval (fail-closed). Byte-identical when
no directory is installed: the grant-time snapshot is used exactly as before.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from refaat.application.approvals import ApprovalService, StoreApprovalBackend
from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.authz import Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.governance import ApprovalRecord, ToolCapability, approval_satisfied

pytestmark = pytest.mark.integration

FINGERPRINT = "a" * 64
CAPABILITY = ToolCapability(
    name="lookup",
    required_scopes=frozenset({"read:corpus"}),
    side_effect=SideEffectClass.READ_ONLY,
    requires_approval=True,
    approver_role="reviewer",
)


class _Directory:
    def __init__(self, principal: Principal | None = None, *, unavailable: bool = False) -> None:
        self._principal = principal
        self._unavailable = unavailable

    def refresh(self, principal_id: str) -> Principal | None:
        del principal_id
        if self._unavailable:
            raise RuntimeError("directory unavailable")
        return self._principal


def _satisfied(records: tuple[ApprovalRecord, ...]) -> bool:
    return approval_satisfied(
        CAPABILITY, fingerprint=FINGERPRINT, approvals=records, initiator_id="alice", now=datetime.now(UTC)
    ).satisfied


@pytest.fixture
def granted(settings: Settings) -> Iterator[tuple[Refaat, str]]:
    runtime = RuntimeBuilder(settings).build()
    try:
        workspace_id = runtime.create_workspace("acme", "Acme")
        ApprovalService(runtime.store, workspace_id).grant(
            fingerprint=FINGERPRINT, capability_name="lookup", approver_id="bob", approver_roles=["reviewer"]
        )
        yield runtime, workspace_id
    finally:
        runtime.close()


def _active(runtime: Refaat, workspace_id: str, directory: _Directory | None) -> tuple[ApprovalRecord, ...]:
    return StoreApprovalBackend(runtime.store, workspace_id, directory=directory).active(FINGERPRINT)


def test_no_directory_uses_the_grant_time_roles(granted: tuple[Refaat, str]) -> None:
    runtime, workspace_id = granted
    assert _satisfied(_active(runtime, workspace_id, None))


def test_directory_with_the_role_still_satisfies(granted: tuple[Refaat, str]) -> None:
    runtime, workspace_id = granted
    directory = _Directory(Principal(id="bob", tenant="acme", roles=frozenset({"reviewer"})))
    assert _satisfied(_active(runtime, workspace_id, directory))


def test_directory_without_the_role_no_longer_satisfies(granted: tuple[Refaat, str]) -> None:
    runtime, workspace_id = granted
    directory = _Directory(Principal(id="bob", tenant="acme", roles=frozenset()))
    assert not _satisfied(_active(runtime, workspace_id, directory))


def test_a_role_gained_after_grant_does_not_retroactively_satisfy(settings: Settings) -> None:
    # No-widening: re-resolution intersects with the grant-time roles, so a role the approver did not
    # hold at grant cannot make the approval valid even when the directory now reports it.
    runtime = RuntimeBuilder(settings).build()
    try:
        workspace_id = runtime.create_workspace("acme", "Acme")
        ApprovalService(runtime.store, workspace_id).grant(
            fingerprint=FINGERPRINT, capability_name="lookup", approver_id="bob", approver_roles=[]
        )
        directory = _Directory(Principal(id="bob", tenant="acme", roles=frozenset({"reviewer"})))
        records = StoreApprovalBackend(runtime.store, workspace_id, directory=directory).active(FINGERPRINT)
        assert not _satisfied(records)
    finally:
        runtime.close()


def test_a_revoked_approver_is_dropped(granted: tuple[Refaat, str]) -> None:
    runtime, workspace_id = granted
    records = _active(runtime, workspace_id, _Directory(None))
    assert records == ()
    assert not _satisfied(records)


def test_an_unavailable_directory_fails_closed(granted: tuple[Refaat, str]) -> None:
    runtime, workspace_id = granted
    records = _active(runtime, workspace_id, _Directory(unavailable=True))
    assert records == ()
    assert not _satisfied(records)


def test_a_non_principal_directory_return_is_dropped(granted: tuple[Refaat, str]) -> None:
    # Fail-closed: a directory that violates the Principal | None contract (a roles-bearing duck) is
    # not trusted; the record is dropped rather than its roles being used in the decision.
    class _RolesDuck:
        roles = frozenset({"reviewer"})

    class _BadDirectory:
        def refresh(self, principal_id: str) -> object:
            del principal_id
            return _RolesDuck()

    runtime, workspace_id = granted
    backend = StoreApprovalBackend(runtime.store, workspace_id, directory=_BadDirectory())  # type: ignore[arg-type]
    records = backend.active(FINGERPRINT)
    assert records == ()
    assert not _satisfied(records)


def test_a_directory_returning_a_different_principal_is_dropped(granted: tuple[Refaat, str]) -> None:
    # Fail-closed: refresh must return the requested approver; a principal with a different id is not
    # trusted for this approval, so a directory that returns the wrong record cannot re-authorize it.
    runtime, workspace_id = granted
    directory = _Directory(Principal(id="mallory", tenant="acme", roles=frozenset({"reviewer"})))
    records = _active(runtime, workspace_id, directory)
    assert records == ()
    assert not _satisfied(records)

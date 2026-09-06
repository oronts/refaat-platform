"""Agent mailboxes are bound to their creating principal under an active policy, byte-identical off."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.authz import CanonicalAccessPolicy, Principal
from refaat.errors import AuthorizationError

pytestmark = pytest.mark.integration

ALICE = Principal(id="alice", tenant="merck", groups=frozenset({"research-a"}))
BOB = Principal(id="bob", tenant="merck", groups=frozenset({"research-b"}))


@pytest.fixture
def policy_runtime(settings: Settings) -> Iterator[tuple[Refaat, str]]:
    runtime = RuntimeBuilder(settings).with_access_policy(CanonicalAccessPolicy()).build()
    try:
        ws = runtime.create_workspace("merck", "Merck")
        yield runtime, ws
    finally:
        runtime.close()


def _pair(runtime: Refaat, ws: str, principal: Principal | None) -> tuple[str, str]:
    a = runtime.agents.create(ws, "orchestrator", "orchestrator", principal=principal)
    b = runtime.agents.create(ws, "specialist", "specialist", principal=principal)
    runtime.agents.send(ws, a, b, "handoff", {"task": "secret finance plan"}, principal=principal)
    return a, b


def test_owner_scope_is_enforced_across_principals(policy_runtime: tuple[Refaat, str]) -> None:
    runtime, ws = policy_runtime
    _sender, recipient = _pair(runtime, ws, ALICE)
    assert runtime.agents.receive(ws, recipient, principal=ALICE)
    with pytest.raises(AuthorizationError):
        runtime.agents.receive(ws, recipient, principal=BOB)
    with pytest.raises(AuthorizationError):
        runtime.agents.send(ws, recipient, recipient, "handoff", {"task": "x"}, principal=BOB)


def test_acknowledge_is_bound_to_the_mailbox_owner(policy_runtime: tuple[Refaat, str]) -> None:
    runtime, ws = policy_runtime
    _sender, recipient = _pair(runtime, ws, ALICE)
    message_id = runtime.agents.receive(ws, recipient, principal=ALICE)[0].id
    with pytest.raises(AuthorizationError):
        runtime.agents.acknowledge(ws, message_id, principal=BOB)
    runtime.agents.acknowledge(ws, message_id, principal=ALICE)


def test_null_owner_scope_agent_stays_readable_under_a_policy(policy_runtime: tuple[Refaat, str]) -> None:
    runtime, ws = policy_runtime
    _sender, recipient = _pair(runtime, ws, None)
    assert runtime.agents.receive(ws, recipient, principal=BOB) is not None


def test_no_policy_is_byte_identical(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).build()
    try:
        ws = runtime.create_workspace("merck", "Merck")
        _sender, recipient = _pair(runtime, ws, None)
        assert runtime.agents.receive(ws, recipient) is not None
        assert runtime.store.get_agent(ws, recipient)["owner_scope"] is None
    finally:
        runtime.close()

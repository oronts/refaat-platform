from __future__ import annotations

from refaat.domain.authz import AccessPolicy, Principal, auth_scope_hash
from refaat.domain.models import AgentMessage, JsonObject
from refaat.errors import AuthorizationError
from refaat.infrastructure.database.store import SQLStore


class AgentService:
    def __init__(self, store: SQLStore, access_policy: AccessPolicy | None = None) -> None:
        self.store = store
        self._access_policy = access_policy

    def _owner_scope(self, principal: Principal | None) -> str | None:
        if self._access_policy is None or principal is None:
            return None
        return auth_scope_hash(principal, self._access_policy)

    def _enforce(self, owner_scope: str | None, principal: Principal | None) -> None:
        # A NULL owner scope (agent created before a policy) stays workspace-readable, so the off path is identical.
        if self._access_policy is None or principal is None or owner_scope is None:
            return
        if owner_scope != auth_scope_hash(principal, self._access_policy):
            raise AuthorizationError("Not authorized for this agent")

    def create(
        self,
        workspace_id: str,
        name: str,
        kind: str,
        *,
        goal: str | None = None,
        state: JsonObject | None = None,
        retained: bool = True,
        parent_agent_id: str | None = None,
        principal: Principal | None = None,
    ) -> str:
        return self.store.create_agent(
            workspace_id,
            name,
            kind,
            goal=goal,
            state=state,
            retained=retained,
            parent_agent_id=parent_agent_id,
            owner_scope=self._owner_scope(principal),
        )

    def heartbeat(self, workspace_id: str, agent_id: str, state: JsonObject | None = None) -> None:
        self.store.heartbeat_agent(workspace_id, agent_id, state)

    def send(
        self,
        workspace_id: str,
        sender_agent_id: str,
        recipient_agent_id: str,
        kind: str,
        payload: JsonObject,
        correlation_id: str | None = None,
        *,
        principal: Principal | None = None,
    ) -> AgentMessage:
        if self._access_policy is not None and principal is not None:
            self._enforce(self.store.get_agent(workspace_id, sender_agent_id).get("owner_scope"), principal)
        return self.store.send_agent_message(
            workspace_id,
            sender_agent_id,
            recipient_agent_id,
            kind,
            payload,
            correlation_id,
        )

    def receive(
        self, workspace_id: str, agent_id: str, limit: int = 50, *, principal: Principal | None = None
    ) -> list[AgentMessage]:
        if self._access_policy is not None and principal is not None:
            self._enforce(self.store.get_agent(workspace_id, agent_id).get("owner_scope"), principal)
        return self.store.receive_agent_messages(workspace_id, agent_id, limit=limit)

    def acknowledge(self, workspace_id: str, message_id: str, *, principal: Principal | None = None) -> None:
        if self._access_policy is not None and principal is not None:
            recipient = self.store.get_agent_message(workspace_id, message_id)["recipient_agent_id"]
            self._enforce(self.store.get_agent(workspace_id, recipient).get("owner_scope"), principal)
        self.store.acknowledge_agent_message(workspace_id, message_id)

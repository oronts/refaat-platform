from __future__ import annotations

import pytest

from refaat.domain.enums import SideEffectClass
from refaat.domain.models import ReActPolicy, ToolGrant
from refaat.programs import KeywordToolPlanner, ToolGovernor


def test_side_effect_classes_are_ordered() -> None:
    assert SideEffectClass.READ_ONLY.rank < SideEffectClass.WRITE.rank < SideEffectClass.EXTERNAL.rank


def test_authorize_admits_a_granted_read_only_tool() -> None:
    policy = ReActPolicy(
        granted_scopes=("read:corpus",),
        grants=(ToolGrant("lookup", required_scopes=("read:corpus",)),),
    )
    ok, reason = ToolGovernor(policy).authorize("lookup")
    assert ok
    assert reason == "authorized"


def test_authorize_denies_an_ungranted_tool() -> None:
    policy = ReActPolicy(grants=(ToolGrant("lookup"),))
    ok, reason = ToolGovernor(policy).authorize("unknown")
    assert not ok
    assert "no grant" in reason


def test_authorize_denies_a_missing_scope() -> None:
    policy = ReActPolicy(
        granted_scopes=(),
        grants=(ToolGrant("lookup", required_scopes=("read:corpus",)),),
    )
    ok, reason = ToolGovernor(policy).authorize("lookup")
    assert not ok
    assert "missing scopes" in reason


def test_authorize_denies_a_side_effect_above_the_ceiling() -> None:
    policy = ReActPolicy(
        granted_scopes=("write:crm",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("notify", required_scopes=("write:crm",), side_effect=SideEffectClass.EXTERNAL),),
    )
    ok, reason = ToolGovernor(policy).authorize("notify")
    assert not ok
    assert "exceeds ceiling" in reason


def test_authorize_admits_a_side_effect_at_the_ceiling() -> None:
    policy = ReActPolicy(
        granted_scopes=("write:crm",),
        side_effect_ceiling=SideEffectClass.WRITE,
        grants=(ToolGrant("update", required_scopes=("write:crm",), side_effect=SideEffectClass.WRITE),),
    )
    ok, _ = ToolGovernor(policy).authorize("update")
    assert ok


def test_policy_rejects_out_of_bounds_iterations() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        ReActPolicy(max_iterations=0)
    with pytest.raises(ValueError, match="max_iterations"):
        ReActPolicy(max_iterations=100)


def test_planner_is_deterministic_and_skips_used_tools() -> None:
    planner = KeywordToolPlanner({"lookup": ("policy", "rollback"), "notify": ("email",)})
    first = planner.plan("how does rollback work", frozenset())
    assert first is not None
    assert first.tool == "lookup"
    assert first.arguments == {}
    assert planner.plan("how does rollback work", frozenset({"lookup"})) is None
    second = planner.plan("please send an email", frozenset())
    assert second is not None
    assert second.tool == "notify"
    assert planner.plan("nothing matches here", frozenset()) is None

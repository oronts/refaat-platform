from refaat.application.budget import active_meter, budget_scope, isolated_budget_scope
from refaat.core.retrieval import ContextBudgeter
from refaat.domain.models import Chunk, SearchHit


def make_hit(index: int, document: str, tokens: int, score: float) -> SearchHit:
    chunk = Chunk(
        id=f"chunk_{index}",
        corpus_id="corpus",
        document_id=document,
        ordinal=index,
        text="word " * tokens,
        token_count=tokens,
        checksum=str(index),
        metadata={},
        embedding=(1.0, 0.0),
    )
    return SearchHit(chunk=chunk, score=score)


def test_budgeter_enforces_budget_and_document_diversity() -> None:
    hits = [
        make_hit(1, "doc_a", 40, 1.0),
        make_hit(2, "doc_a", 40, 0.98),
        make_hit(3, "doc_b", 30, 0.95),
    ]
    selected = ContextBudgeter().select(hits, token_budget=95)

    assert selected
    assert sum(hit.chunk.token_count + 12 for hit in selected) <= 95
    assert selected[0].chunk.id == "chunk_1"


def test_charge_is_charge_only_and_never_raises_on_overrun() -> None:
    with budget_scope(10) as meter:
        assert meter is not None
        meter.charge({"prompt_tokens": 100, "completion_tokens": 0})
        assert meter.tokens_used == 100
        assert meter.exhausted() is True


def test_nested_budget_scopes_share_one_meter() -> None:
    with budget_scope(1000) as outer:
        assert outer is not None
        outer.charge({"prompt_tokens": 10})
        with budget_scope(5) as inner:
            # A nested scope shares the outer meter instead of installing its own,
            # so an inner budget cannot reset accounting and only the outermost
            # ceiling is in force.
            assert inner is outer
            inner.charge({"prompt_tokens": 10})
        assert outer.tokens_used == 20
        assert active_meter() is outer


def test_budget_scope_is_a_no_op_when_no_ceiling_is_configured() -> None:
    with budget_scope(None, None) as meter:
        assert meter is None
        assert active_meter() is None


def test_isolated_budget_scope_never_shares_an_outer_meter() -> None:
    with budget_scope(1000) as outer:
        assert outer is not None
        with isolated_budget_scope(500) as inner:
            # Unlike a nested budget_scope, an isolated scope always installs its own meter,
            # so an unserved run's spend can never draw down the served run's budget.
            assert inner is not None
            assert inner is not outer
            assert active_meter() is inner
        assert active_meter() is outer


def test_isolated_budget_scope_with_no_ceiling_detaches_the_outer_meter() -> None:
    # With no ceiling and an outer meter active, the isolated scope must DETACH (not share) the outer
    # meter, so an unserved run's spend can never accrue to the served run's meter.
    with budget_scope(1000) as outer:
        with isolated_budget_scope(None, None) as detached:
            assert detached is None
            assert active_meter() is None
        assert active_meter() is outer


def test_chargeable_meter_routes_the_unserved_run_to_its_own_meter() -> None:
    from refaat.application.services import shadow_execution, unserved_run
    from refaat.infrastructure.model_cache import _chargeable_meter

    with budget_scope(1000) as served:
        assert _chargeable_meter() is served
        with shadow_execution():
            # The served degraded path (a bare shadow scope) spends nothing.
            assert _chargeable_meter() is None
        with unserved_run(), isolated_budget_scope(500) as isolated:
            # An unserved deployment run charges its OWN isolated meter, not the served run's.
            assert _chargeable_meter() is isolated
        # No ceiling configured: the unserved run charges NOTHING and never draws down the served meter.
        with unserved_run(), isolated_budget_scope(None, None):
            assert _chargeable_meter() is None
        assert _chargeable_meter() is served
        assert served.tokens_used == 0


def test_cost_accrues_only_with_a_price_table() -> None:
    prices = {"gpt": {"input_per_1k": 2.0, "output_per_1k": 4.0}}
    with budget_scope(cost_ceiling_usd=1.0, prices=prices) as meter:
        assert meter is not None
        meter.charge({"prompt_tokens": 1000, "completion_tokens": 0}, model_id="gpt")
        assert meter.cost_used_usd == 2.0
        assert meter.exhausted() is True
    with budget_scope(cost_ceiling_usd=1.0) as unpriced:
        assert unpriced is not None
        unpriced.charge({"prompt_tokens": 1000, "completion_tokens": 0}, model_id="gpt")
        assert unpriced.cost_used_usd == 0.0

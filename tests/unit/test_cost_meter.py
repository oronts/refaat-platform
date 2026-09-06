from __future__ import annotations

import pytest

from refaat.application.budget import CostMeter, active_meter, budget_scope, prices_from


def test_charge_sums_prompt_and_completion_tokens() -> None:
    meter = CostMeter(token_ceiling=100)
    charged = meter.charge({"prompt_tokens": 30, "completion_tokens": 12})
    assert charged == 42
    assert meter.tokens_used == 42
    assert meter.last_charge == 42
    assert meter.remaining_tokens() == 58


def test_charge_ignores_non_numeric_usage_fields() -> None:
    meter = CostMeter()
    assert meter.charge({"provider": "extractive", "sentences": 3}) == 0
    assert meter.tokens_used == 0


def test_exhausted_reports_once_the_ceiling_is_reached() -> None:
    meter = CostMeter(token_ceiling=50)
    meter.charge({"prompt_tokens": 40, "completion_tokens": 0})
    assert not meter.exhausted()
    meter.charge({"prompt_tokens": 10, "completion_tokens": 0})
    assert meter.exhausted()


def test_charge_is_charge_only_and_overrun_never_raises() -> None:
    meter = CostMeter(token_ceiling=10)
    meter.charge({"prompt_tokens": 5, "completion_tokens": 0})
    # Token and cost are known only after a call, so a mid-call charge never raises;
    # it accrues past the ceiling and exhausted() then reports the overrun.
    meter.charge({"prompt_tokens": 8, "completion_tokens": 0})
    assert meter.tokens_used == 13
    assert meter.exhausted()


def test_exactly_reaching_the_ceiling_does_not_raise_but_exhausts() -> None:
    meter = CostMeter(token_ceiling=10)
    meter.charge({"prompt_tokens": 6, "completion_tokens": 4})
    assert meter.tokens_used == 10
    assert meter.exhausted()
    assert meter.remaining_tokens() == 0


def test_usd_accrual_is_exact_given_a_price_map() -> None:
    prices = {"gpt-x": {"input_per_1k": 2.0, "output_per_1k": 6.0}}
    meter = CostMeter(cost_ceiling_usd=1.0, prices=prices)
    meter.charge({"prompt_tokens": 100, "completion_tokens": 50}, model_id="gpt-x")
    assert meter.cost_used_usd == pytest.approx(100 / 1000 * 2.0 + 50 / 1000 * 6.0)
    assert meter.remaining_cost_usd() == pytest.approx(1.0 - 0.5)


def test_crossing_the_cost_ceiling_charges_without_raising() -> None:
    prices = {"gpt-x": {"input_per_1k": 10.0, "output_per_1k": 10.0}}
    meter = CostMeter(cost_ceiling_usd=0.5, prices=prices)
    meter.charge({"prompt_tokens": 100, "completion_tokens": 0}, model_id="gpt-x")
    assert meter.cost_used_usd == pytest.approx(1.0)
    assert meter.exhausted()


def test_unpriced_model_accrues_no_cost() -> None:
    meter = CostMeter(cost_ceiling_usd=1.0, prices={"gpt-x": {"input_per_1k": 5.0}})
    meter.charge({"prompt_tokens": 100, "completion_tokens": 100}, model_id="other")
    assert meter.cost_used_usd == 0.0


def test_budget_scope_installs_and_clears_the_active_meter() -> None:
    assert active_meter() is None
    with budget_scope(100) as meter:
        assert meter is not None
        assert active_meter() is meter
    assert active_meter() is None


def test_budget_scope_is_a_noop_when_no_ceiling_is_set() -> None:
    with budget_scope(None, None) as meter:
        assert meter is None
        assert active_meter() is None


def test_prices_from_coerces_to_floats() -> None:
    assert prices_from({"m": {"input_per_1k": 1}}) == {"m": {"input_per_1k": 1.0}}
    assert prices_from(None) == {}

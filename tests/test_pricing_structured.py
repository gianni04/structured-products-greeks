"""Tests du module pricing_structured : bornes de non-arbitrage du prix d'un Autocall Phoenix."""

from __future__ import annotations

import numpy as np

from structrisk.blackscholes import bs_price
from structrisk.payoffs import CapitalProtectedNoteSpec, PhoenixAutocallSpec
from structrisk.pricing_structured import (
    price_capital_protected_note_decomposition,
    price_phoenix_autocall,
)

SPOT = 100.0
VOL = 0.25
RATE = 0.03
DIVIDEND = 0.02
CORR = np.array([[1.0]])


def _spec() -> PhoenixAutocallSpec:
    return PhoenixAutocallSpec(
        notional=100.0,
        coupon_rate=0.02,
        autocall_barrier=1.00,
        coupon_barrier=0.70,
        protection_barrier=0.60,
        observation_times=np.array([0.25, 0.5, 0.75, 1.0]),
    )


def test_autocall_price_within_no_arbitrage_bounds():
    spec = _spec()
    result = price_phoenix_autocall(SPOT, VOL, RATE, DIVIDEND, CORR, spec,
                                     n_paths=100_000, seed=7)
    n_obs = len(spec.observation_times)
    max_possible = spec.notional + spec.coupon_rate * spec.notional * n_obs
    min_possible = 0.0
    assert min_possible <= result.price <= max_possible


def test_autocall_call_probabilities_sum_below_one():
    spec = _spec()
    result = price_phoenix_autocall(SPOT, VOL, RATE, DIVIDEND, CORR, spec,
                                     n_paths=100_000, seed=7)
    assert result.call_probability_by_date.sum() + result.prob_never_called <= 1.0 + 1e-9
    assert 0.0 <= result.prob_capital_loss <= 1.0


def test_capital_protected_note_delta_nonzero_with_fixed_strike():
    spec = CapitalProtectedNoteSpec(notional=1000.0, protection_level=1.0, participation_rate=0.8,
                                    strike_level=1.0, cap_level=1.3)
    maturity, h = 1.0, 1.0

    def price(s: float) -> float:
        return price_capital_protected_note_decomposition(s, VOL, RATE, DIVIDEND, maturity, spec,
                                                           spot_ref=SPOT)

    delta = (price(SPOT + h) - price(SPOT - h)) / (2.0 * h)

    def reference(s: float) -> float:
        # Reference independante : strikes fixes a 100 et 130, call-spread pondere.
        spread = (bs_price(s, SPOT, maturity, RATE, DIVIDEND, VOL, "call")
                  - bs_price(s, 1.3 * SPOT, maturity, RATE, DIVIDEND, VOL, "call"))
        return 1000.0 * np.exp(-RATE * maturity) + 0.8 * (1000.0 / SPOT) * spread

    expected = (reference(SPOT + h) - reference(SPOT - h)) / (2.0 * h)
    assert abs(delta) > 1.0
    assert abs(delta - expected) < 1e-8

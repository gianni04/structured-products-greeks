"""Tests du module montecarlo : le prix Monte Carlo d'un call vanille doit contenir le prix analytique Black-Scholes dans son intervalle de confiance."""

from __future__ import annotations

from structrisk.blackscholes import bs_price
from structrisk.montecarlo import mc_vanilla_call

SPOT = 100.0
STRIKE = 100.0
MATURITY = 1.0
RATE = 0.03
DIVIDEND = 0.015
VOL = 0.22


def test_mc_vanilla_call_ci_contains_analytic_price():
    analytic = bs_price(SPOT, STRIKE, MATURITY, RATE, DIVIDEND, VOL, "call")
    result = mc_vanilla_call(SPOT, STRIKE, MATURITY, RATE, DIVIDEND, VOL,
                              n_paths=200_000, seed=42, antithetic=True, use_control_variate=True)
    assert result.ci_low <= analytic <= result.ci_high

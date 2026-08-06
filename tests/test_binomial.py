"""Tests du module binomial : convergence CRR vers Black-Scholes, prime d'exercice anticipe du put americain."""

from __future__ import annotations

import numpy as np

from structrisk.binomial import convergence_to_bs, crr_price
from structrisk.blackscholes import bs_price

SPOT = 100.0
STRIKE = 100.0
MATURITY = 1.0
RATE = 0.03
DIVIDEND = 0.015
VOL = 0.22


def test_crr_convergence_to_black_scholes():
    steps_grid = np.array([50, 200, 800])
    errors = convergence_to_bs(SPOT, STRIKE, MATURITY, RATE, DIVIDEND, VOL, "call", steps_grid)
    assert errors[-1] < errors[0]
    assert errors[-1] < 0.05


def test_american_put_at_least_european_put():
    american = crr_price(SPOT, STRIKE, MATURITY, RATE, DIVIDEND, VOL, 300, "put", "american")
    european = crr_price(SPOT, STRIKE, MATURITY, RATE, DIVIDEND, VOL, 300, "put", "european")
    assert american >= european - 1e-9


def test_european_crr_close_to_bs_price():
    bs_ref = bs_price(SPOT, STRIKE, MATURITY, RATE, DIVIDEND, VOL, "call")
    crr = crr_price(SPOT, STRIKE, MATURITY, RATE, DIVIDEND, VOL, 500, "call", "european")
    assert abs(crr - bs_ref) < 0.1

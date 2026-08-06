"""Tests du module blackscholes : parite call-put, delta analytique vs difference finie, positivite du gamma."""

from __future__ import annotations

from structrisk.blackscholes import bs_price, delta, gamma

SPOT = 100.0
STRIKE = 100.0
MATURITY = 1.0
RATE = 0.03
DIVIDEND = 0.015
VOL = 0.22


def test_call_put_parity():
    call = bs_price(SPOT, STRIKE, MATURITY, RATE, DIVIDEND, VOL, "call")
    put = bs_price(SPOT, STRIKE, MATURITY, RATE, DIVIDEND, VOL, "put")
    forward_value = SPOT * (2.718281828459045 ** (-DIVIDEND * MATURITY)) - \
        STRIKE * (2.718281828459045 ** (-RATE * MATURITY))
    assert abs((call - put) - forward_value) < 1e-8


def test_delta_analytic_vs_finite_difference_call():
    h = 1e-4 * SPOT
    price_up = bs_price(SPOT + h, STRIKE, MATURITY, RATE, DIVIDEND, VOL, "call")
    price_dn = bs_price(SPOT - h, STRIKE, MATURITY, RATE, DIVIDEND, VOL, "call")
    fd_delta = (price_up - price_dn) / (2.0 * h)
    analytic_delta = delta(SPOT, STRIKE, MATURITY, RATE, DIVIDEND, VOL, "call")
    assert abs(fd_delta - analytic_delta) < 1e-6


def test_delta_analytic_vs_finite_difference_put():
    h = 1e-4 * SPOT
    price_up = bs_price(SPOT + h, STRIKE, MATURITY, RATE, DIVIDEND, VOL, "put")
    price_dn = bs_price(SPOT - h, STRIKE, MATURITY, RATE, DIVIDEND, VOL, "put")
    fd_delta = (price_up - price_dn) / (2.0 * h)
    analytic_delta = delta(SPOT, STRIKE, MATURITY, RATE, DIVIDEND, VOL, "put")
    assert abs(fd_delta - analytic_delta) < 1e-6


def test_gamma_positive_for_long_option():
    g = gamma(SPOT, STRIKE, MATURITY, RATE, DIVIDEND, VOL)
    assert g > 0.0

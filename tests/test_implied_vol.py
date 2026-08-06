"""Tests du module implied_vol : aller-retour prix -> vol implicite -> prix."""

from __future__ import annotations

from structrisk.blackscholes import bs_price
from structrisk.implied_vol import implied_vol

SPOT = 100.0
STRIKE = 105.0
MATURITY = 0.75
RATE = 0.03
DIVIDEND = 0.015
VOL = 0.25


def test_implied_vol_round_trip_call():
    price = bs_price(SPOT, STRIKE, MATURITY, RATE, DIVIDEND, VOL, "call")
    recovered_vol = implied_vol(price, SPOT, STRIKE, MATURITY, RATE, DIVIDEND, "call")
    assert abs(recovered_vol - VOL) < 1e-6


def test_implied_vol_round_trip_put():
    price = bs_price(SPOT, STRIKE, MATURITY, RATE, DIVIDEND, VOL, "put")
    recovered_vol = implied_vol(price, SPOT, STRIKE, MATURITY, RATE, DIVIDEND, "put")
    assert abs(recovered_vol - VOL) < 1e-6


def test_implied_vol_round_trip_deep_itm_uses_brent_fallback():
    deep_strike = 40.0
    price = bs_price(SPOT, deep_strike, MATURITY, RATE, DIVIDEND, VOL, "call")
    recovered_vol = implied_vol(price, SPOT, deep_strike, MATURITY, RATE, DIVIDEND, "call")
    assert abs(recovered_vol - VOL) < 1e-4

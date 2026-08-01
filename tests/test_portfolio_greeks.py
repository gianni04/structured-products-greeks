"""Tests du module portfolio_greeks : additivite de l'agregation des Grecques."""

from __future__ import annotations

from structrisk.portfolio_greeks import Position, aggregate_greeks_by_underlying


def test_portfolio_greeks_aggregate_additively():
    pos1 = Position(
        name="Call A", underlying="SX5E", quantity=10.0, multiplier=1.0, spot=4500.0,
        greeks={"delta": 0.55, "gamma": 0.002, "vega": 12.0}, price=180.0,
    )
    pos2 = Position(
        name="Put A", underlying="SX5E", quantity=-5.0, multiplier=1.0, spot=4500.0,
        greeks={"delta": -0.40, "gamma": 0.0015, "vega": 10.0}, price=90.0,
    )
    frame = aggregate_greeks_by_underlying([pos1, pos2])

    expected_delta = 10.0 * 0.55 + (-5.0) * (-0.40)
    expected_gamma = 10.0 * 0.002 + (-5.0) * 0.0015
    expected_vega = 10.0 * 12.0 + (-5.0) * 10.0

    assert frame.loc["SX5E", "delta"] == expected_delta
    assert frame.loc["SX5E", "gamma"] == expected_gamma
    assert frame.loc["SX5E", "vega"] == expected_vega

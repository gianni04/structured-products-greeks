"""Agregation des Grecques d'un portefeuille d'options et de produits structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Position:
    """Une ligne de portefeuille (option vanille ou produit structure)."""

    name: str
    underlying: str
    quantity: float
    multiplier: float
    spot: float
    greeks: dict[str, float]
    price: float
    reval_fn: Callable[[float, float], float] | None = field(default=None, compare=False)


def aggregate_greeks_by_underlying(positions: list[Position]) -> pd.DataFrame:
    """Agrege les Grecques nettes par sous-jacent, en unités "produit" (non converties en euros)."""
    rows: dict[str, dict[str, float]] = {}
    for pos in positions:
        bucket = rows.setdefault(pos.underlying, {})
        weight = pos.quantity * pos.multiplier
        for greek_name, greek_value in pos.greeks.items():
            bucket[greek_name] = bucket.get(greek_name, 0.0) + weight * greek_value
    frame = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
    return frame.sort_index()


def delta_equivalent_exposure(positions: list[Position]) -> pd.Series:
    """Exposition delta-équivalente en euros, par sous-jacent."""
    totals: dict[str, float] = {}
    for pos in positions:
        totals[pos.underlying] = totals.get(pos.underlying, 0.0) + (
            pos.quantity * pos.multiplier * pos.greeks.get("delta", 0.0) * pos.spot
        )
    return pd.Series(totals, name="delta_eur").sort_index()


def gamma_eur_per_1pct_move(positions: list[Position]) -> pd.Series:
    """Gamma exprime en P&L euros pour un mouvement de spot de +1 %, par sous-jacent."""
    totals: dict[str, float] = {}
    for pos in positions:
        shock = 0.01 * pos.spot
        contribution = 0.5 * pos.quantity * pos.multiplier * pos.greeks.get("gamma", 0.0) * shock ** 2
        totals[pos.underlying] = totals.get(pos.underlying, 0.0) + contribution
    return pd.Series(totals, name="gamma_eur_1pct").sort_index()


def vega_eur_per_vol_point(positions: list[Position]) -> pd.Series:
    """Vega exprime en P&L euros pour un point de volatilité implicite (+1 pt = +0.01), par sous-jacent."""
    totals: dict[str, float] = {}
    for pos in positions:
        totals[pos.underlying] = totals.get(pos.underlying, 0.0) + (
            pos.quantity * pos.multiplier * pos.greeks.get("vega", 0.0) * 0.01
        )
    return pd.Series(totals, name="vega_eur_1pt").sort_index()


def stress_grid(positions: list[Position], spot_shocks: np.ndarray, vol_shocks: np.ndarray) -> pd.DataFrame:
    """Grille de P&L du portefeuille (en euros) croisant chocs de spot et de vol."""
    grid = np.zeros((len(spot_shocks), len(vol_shocks)))
    for pos in positions:
        weight = pos.quantity * pos.multiplier
        for i, ds in enumerate(spot_shocks):
            for j, dv in enumerate(vol_shocks):
                if pos.reval_fn is not None:
                    new_price = pos.reval_fn(ds, dv)
                    pnl_unit = new_price - pos.price
                else:
                    d_spot = ds * pos.spot
                    delta = pos.greeks.get("delta", 0.0)
                    gamma = pos.greeks.get("gamma", 0.0)
                    vega = pos.greeks.get("vega", 0.0)
                    vanna = pos.greeks.get("vanna", 0.0)
                    pnl_unit = (delta * d_spot + 0.5 * gamma * d_spot ** 2
                                + vega * dv + vanna * d_spot * dv)
                grid[i, j] += weight * pnl_unit
    index = pd.Index(np.round(spot_shocks * 100, 4), name="spot_shock_pct")
    columns = pd.Index(np.round(vol_shocks * 100, 4), name="vol_shock_pt")
    return pd.DataFrame(grid, index=index, columns=columns)


@dataclass(frozen=True)
class PnLAttribution:
    """Decomposition du P&L realise d'une position par les Grecques."""

    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    explained_pnl: float
    residual: float


def pnl_attribution(position: Position, d_spot: float, d_vol: float, dt: float,
                     realized_pnl: float) -> PnLAttribution:
    """Attribution du P&L d'une position entre delta, gamma, vega, theta et residu."""
    delta_pnl = position.greeks.get("delta", 0.0) * d_spot
    gamma_pnl = 0.5 * position.greeks.get("gamma", 0.0) * d_spot ** 2
    vega_pnl = position.greeks.get("vega", 0.0) * d_vol
    theta_pnl = position.greeks.get("theta", 0.0) * dt
    explained = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
    residual = realized_pnl - explained
    return PnLAttribution(delta_pnl=delta_pnl, gamma_pnl=gamma_pnl, vega_pnl=vega_pnl,
                           theta_pnl=theta_pnl, explained_pnl=explained, residual=residual)

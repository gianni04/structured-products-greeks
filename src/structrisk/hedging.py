"""Simulation de couverture en delta (et delta-gamma) discrète."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .blackscholes import bs_price, delta as bs_delta, gamma as bs_gamma
from .montecarlo import simulate_gbm_paths


@dataclass(frozen=True)
class HedgeSimulationResult:
    """Resultat d'une simulation de couverture dynamique."""

    hedging_error: np.ndarray
    mean_error: float
    std_error: float


def simulate_delta_hedge_pnl(spot0: float, strike: float, maturity: float, rate: float,
                              dividend: float, vol: float, n_paths: int, n_rehedge: int,
                              seed: int | None = None, transaction_cost_bps: float = 0.0,
                              option_type: str = "call",
                              realized_vol: float | None = None) -> HedgeSimulationResult:
    """Simule la replication delta-neutre discrète d'un call/put vendu."""
    sim_vol = vol if realized_vol is None else realized_vol
    paths = simulate_gbm_paths(np.array([spot0]), rate, np.array([dividend]), np.array([sim_vol]),
                                np.array([[1.0]]), maturity, n_rehedge, n_paths, seed, antithetic=True)
    spot_path = paths[:, :, 0]

    dt = maturity / n_rehedge
    premium = float(bs_price(spot0, strike, maturity, rate, dividend, vol, option_type))
    cash = np.full(n_paths, premium)
    shares = np.zeros(n_paths)

    for i in range(n_rehedge):
        tau = maturity - i * dt
        s_i = spot_path[:, i]
        delta_i = bs_delta(s_i, strike, tau, rate, dividend, vol, option_type)
        trade = delta_i - shares
        cost = transaction_cost_bps * 1e-4 * np.abs(trade) * s_i
        cash = cash - trade * s_i - cost
        shares = delta_i
        cash = cash * np.exp(rate * dt) + shares * s_i * (np.exp(dividend * dt) - 1.0)

    s_final = spot_path[:, -1]
    if option_type == "call":
        payoff = np.maximum(s_final - strike, 0.0)
    else:
        payoff = np.maximum(strike - s_final, 0.0)

    portfolio_value = cash + shares * s_final
    hedging_error = portfolio_value - payoff
    return HedgeSimulationResult(hedging_error=hedging_error, mean_error=float(np.mean(hedging_error)),
                                  std_error=float(np.std(hedging_error, ddof=1)))


def simulate_delta_gamma_hedge_pnl(spot0: float, strike: float, maturity: float, rate: float,
                                    dividend: float, vol: float, hedge_option_strike: float,
                                    n_paths: int, n_rehedge: int, seed: int | None = None,
                                    transaction_cost_bps: float = 0.0,
                                    option_type: str = "call") -> HedgeSimulationResult:
    """Simule la replication delta-gamma-neutre discrète d'un call/put vendu."""
    paths = simulate_gbm_paths(np.array([spot0]), rate, np.array([dividend]), np.array([vol]),
                                np.array([[1.0]]), maturity, n_rehedge, n_paths, seed, antithetic=True)
    spot_path = paths[:, :, 0]

    dt = maturity / n_rehedge
    premium = float(bs_price(spot0, strike, maturity, rate, dividend, vol, option_type))
    cash = np.full(n_paths, premium)
    shares = np.zeros(n_paths)
    n_hedge_opt = np.zeros(n_paths)

    for i in range(n_rehedge):
        tau = maturity - i * dt
        s_i = spot_path[:, i]
        delta_t = bs_delta(s_i, strike, tau, rate, dividend, vol, option_type)
        gamma_t = bs_gamma(s_i, strike, tau, rate, dividend, vol)
        delta_h = bs_delta(s_i, hedge_option_strike, tau, rate, dividend, vol, option_type)
        gamma_h = bs_gamma(s_i, hedge_option_strike, tau, rate, dividend, vol)
        price_h = bs_price(s_i, hedge_option_strike, tau, rate, dividend, vol, option_type)

        new_n_hedge = gamma_t / gamma_h
        new_shares = delta_t - new_n_hedge * delta_h

        trade_shares = new_shares - shares
        trade_hedge = new_n_hedge - n_hedge_opt
        cost = transaction_cost_bps * 1e-4 * (np.abs(trade_shares) * s_i + np.abs(trade_hedge) * price_h)
        cash = cash - trade_shares * s_i - trade_hedge * price_h - cost
        shares = new_shares
        n_hedge_opt = new_n_hedge
        cash = cash * np.exp(rate * dt) + shares * s_i * (np.exp(dividend * dt) - 1.0)

    s_final = spot_path[:, -1]
    if option_type == "call":
        payoff = np.maximum(s_final - strike, 0.0)
        payoff_hedge = np.maximum(s_final - hedge_option_strike, 0.0)
    else:
        payoff = np.maximum(strike - s_final, 0.0)
        payoff_hedge = np.maximum(hedge_option_strike - s_final, 0.0)

    portfolio_value = cash + shares * s_final + n_hedge_opt * payoff_hedge
    hedging_error = portfolio_value - payoff
    return HedgeSimulationResult(hedging_error=hedging_error, mean_error=float(np.mean(hedging_error)),
                                  std_error=float(np.std(hedging_error, ddof=1)))


def rehedge_frequency_sweep(spot0: float, strike: float, maturity: float, rate: float, dividend: float,
                             vol: float, n_paths: int, rehedge_grid: np.ndarray, seed: int | None = None,
                             transaction_cost_bps: float = 0.0, option_type: str = "call") -> dict[str, np.ndarray]:
    """Balaie la fréquence de rehedge et rapporte moyenne/ecart-type de l'erreur de replication."""
    means = np.empty(len(rehedge_grid))
    stds = np.empty(len(rehedge_grid))
    for i, n_r in enumerate(rehedge_grid):
        res = simulate_delta_hedge_pnl(spot0, strike, maturity, rate, dividend, vol, n_paths, int(n_r),
                                        seed, transaction_cost_bps, option_type)
        means[i] = res.mean_error
        stds[i] = res.std_error
    return {"n_rehedge": np.asarray(rehedge_grid), "mean_error": means, "std_error": stds}

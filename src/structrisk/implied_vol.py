"""Volatilite implicite : inversion Newton-Raphson / Brent et surface de vol."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .blackscholes import bs_price, vega


def _intrinsic_value(spot: float, strike: float, maturity: float, rate: float,
                      dividend: float, option_type: str) -> float:
    """Valeur intrinsèque actualisee (borne inférieure de non-arbitrage)."""
    forward = spot * np.exp(-dividend * maturity)
    disc_strike = strike * np.exp(-rate * maturity)
    if option_type == "call":
        return max(forward - disc_strike, 0.0)
    return max(disc_strike - forward, 0.0)


def implied_vol(price: float, spot: float, strike: float, maturity: float, rate: float,
                 dividend: float, option_type: str = "call", initial_guess: float = 0.20,
                 tol: float = 1e-8, max_iter: int = 100,
                 bounds: tuple[float, float] = (1e-6, 5.0)) -> float:
    """Volatilite implicite par Newton-Raphson avec repli sur Brent."""
    if maturity <= 0.0:
        raise ValueError("La maturité doit être strictement positive")

    intrinsic = _intrinsic_value(spot, strike, maturity, rate, dividend, option_type)
    upper_bound_price = spot * np.exp(-dividend * maturity) if option_type == "call" \
        else strike * np.exp(-rate * maturity)
    eps = 1e-10
    if price < intrinsic - eps:
        raise ValueError(
            f"Prix {price:.6f} inférieur à la valeur intrinsèque actualisee {intrinsic:.6f} : "
            "incompatible avec toute volatilité positive."
        )
    if price > upper_bound_price + eps:
        raise ValueError(
            f"Prix {price:.6f} supérieur à la borne supérieure de non-arbitrage {upper_bound_price:.6f}."
        )

    near_intrinsic = (price - intrinsic) < 1e-8

    def objective(sigma: float) -> float:
        return bs_price(spot, strike, maturity, rate, dividend, sigma, option_type) - price

    if not near_intrinsic:
        sigma = initial_guess
        for _ in range(max_iter):
            diff = objective(sigma)
            if abs(diff) < tol:
                return float(sigma)
            v = float(vega(spot, strike, maturity, rate, dividend, sigma))
            if v < 1e-10:
                break
            step = diff / v
            sigma_next = sigma - step
            if sigma_next <= bounds[0] or sigma_next >= bounds[1] or not np.isfinite(sigma_next):
                break
            sigma = sigma_next
        else:
            sigma = None
        if sigma is not None and abs(objective(sigma)) < tol:
            return float(sigma)

    lo, hi = bounds
    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo > 0:
        lo = 1e-9
        f_lo = objective(lo)
    if f_hi < 0:
        hi = 10.0
        f_hi = objective(hi)
    if f_lo * f_hi > 0:
        raise ValueError("Impossible d'encadrer une racine pour la volatilité implicite (Brent).")
    return float(brentq(objective, lo, hi, xtol=tol, maxiter=200))


@dataclass(frozen=True)
class SmileParams:
    """Parametres d'une surface de volatilité synthetique."""

    atm_level: float = 0.20
    atm_term_slope: float = 0.02
    skew_level: float = -0.10
    skew_decay: float = 0.50
    curvature_level: float = 0.15
    curvature_decay: float = 0.50


def parametric_vol(strike: np.ndarray | float, maturity: np.ndarray | float, spot: float,
                    rate: float, dividend: float, params: SmileParams) -> np.ndarray | float:
    """Vol implicite parametree en un point (K, T) ou sur des tableaux broadcastables."""
    strike = np.asarray(strike, dtype=float)
    maturity = np.asarray(maturity, dtype=float)
    forward = spot * np.exp((rate - dividend) * maturity)
    log_moneyness = np.log(strike / forward)
    atm = params.atm_level + params.atm_term_slope * np.sqrt(maturity)
    skew = params.skew_level * np.exp(-params.skew_decay * maturity)
    curvature = params.curvature_level * np.exp(-params.curvature_decay * maturity)
    sigma = atm + skew * log_moneyness + curvature * log_moneyness ** 2
    return np.maximum(sigma, 0.01)


class VolSurfaceGrid:
    """Grille de volatilité implicite avec interpolation bilineaire."""

    def __init__(self, strikes: np.ndarray, maturities: np.ndarray, spot: float,
                 rate: float, dividend: float, params: SmileParams) -> None:
        self.strikes = np.asarray(strikes, dtype=float)
        self.maturities = np.asarray(maturities, dtype=float)
        if np.any(np.diff(self.strikes) <= 0) or np.any(np.diff(self.maturities) <= 0):
            raise ValueError("strikes et maturities doivent être strictement croissants")
        self.spot = spot
        self.rate = rate
        self.dividend = dividend
        self.params = params
        kk, tt = np.meshgrid(self.strikes, self.maturities, indexing="ij")
        self.grid = np.asarray(parametric_vol(kk, tt, spot, rate, dividend, params))

    def interpolate(self, strike: float, maturity: float) -> float:
        """Interpolation bilineaire de la vol au point (strike, maturity)."""
        x = float(np.clip(strike, self.strikes[0], self.strikes[-1]))
        y = float(np.clip(maturity, self.maturities[0], self.maturities[-1]))

        i1 = int(np.searchsorted(self.strikes, x))
        i1 = min(max(i1, 1), len(self.strikes) - 1)
        i0 = i1 - 1
        j1 = int(np.searchsorted(self.maturities, y))
        j1 = min(max(j1, 1), len(self.maturities) - 1)
        j0 = j1 - 1

        x0, x1 = self.strikes[i0], self.strikes[i1]
        y0, y1 = self.maturities[j0], self.maturities[j1]
        tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

        f00 = self.grid[i0, j0]
        f10 = self.grid[i1, j0]
        f01 = self.grid[i0, j1]
        f11 = self.grid[i1, j1]

        return float(
            (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10
            + (1 - tx) * ty * f01 + tx * ty * f11
        )

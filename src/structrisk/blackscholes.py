"""Modèle de Black-Scholes-Merton (dividende continu) et jeu complet de Grecques."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

ArrayLike = np.ndarray | float


@dataclass(frozen=True)
class BSMParams:
    """Parametres de marché et de contrat pour un pricing Black-Scholes-Merton."""

    spot: float
    strike: float
    maturity: float
    rate: float
    dividend: float
    vol: float


def _d1_d2(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike,
           rate: ArrayLike, dividend: ArrayLike, vol: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """Calcule d1 et d2, vectorisé sur tous les arguments (numpy broadcasting)."""
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    maturity = np.asarray(maturity, dtype=float)
    vol = np.asarray(vol, dtype=float)
    sqrt_t = np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate - dividend + 0.5 * vol ** 2) * maturity) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return d1, d2


def _dd1_dt(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike,
            rate: ArrayLike, dividend: ArrayLike, vol: ArrayLike) -> ArrayLike:
    """Dérivée partielle ddelta(d1)/dT = (r-q)/(sigma*sqrt(T)) - d2/(2T)."""
    _, d2 = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    sqrt_t = np.sqrt(maturity)
    return (rate - dividend) / (vol * sqrt_t) - d2 / (2.0 * maturity)


def bs_price(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
             dividend: ArrayLike, vol: ArrayLike, option_type: str = "call") -> ArrayLike:
    """Prix Black-Scholes-Merton d'un call ou put européen."""
    d1, d2 = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    disc_r = np.exp(-rate * maturity)
    disc_q = np.exp(-dividend * maturity)
    if option_type == "call":
        return spot * disc_q * norm.cdf(d1) - strike * disc_r * norm.cdf(d2)
    if option_type == "put":
        return strike * disc_r * norm.cdf(-d2) - spot * disc_q * norm.cdf(-d1)
    raise ValueError("option_type doit valoir 'call' ou 'put'")


def delta(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
          dividend: ArrayLike, vol: ArrayLike, option_type: str = "call") -> ArrayLike:
    """Delta : sensibilité du prix à une variation du spot, ``dV/dS``."""
    d1, _ = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    disc_q = np.exp(-dividend * maturity)
    if option_type == "call":
        return disc_q * norm.cdf(d1)
    if option_type == "put":
        return disc_q * (norm.cdf(d1) - 1.0)
    raise ValueError("option_type doit valoir 'call' ou 'put'")


def gamma(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
          dividend: ArrayLike, vol: ArrayLike) -> ArrayLike:
    """Gamma : convexité du prix par rapport au spot, ``d^2V/dS^2`` (identique call/put)."""
    d1, _ = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    disc_q = np.exp(-dividend * maturity)
    return disc_q * norm.pdf(d1) / (spot * vol * np.sqrt(maturity))


def vega(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
         dividend: ArrayLike, vol: ArrayLike) -> ArrayLike:
    """Vega : sensibilité du prix à la volatilité, ``dV/dsigma`` (identique call/put)."""
    d1, _ = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    disc_q = np.exp(-dividend * maturity)
    return spot * disc_q * norm.pdf(d1) * np.sqrt(maturity)


def theta(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
          dividend: ArrayLike, vol: ArrayLike, option_type: str = "call") -> ArrayLike:
    """Theta : sensibilité du prix au passage du temps, ``dV/dt = -dV/dT``."""
    d1, d2 = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    disc_r = np.exp(-rate * maturity)
    disc_q = np.exp(-dividend * maturity)
    common = -spot * disc_q * norm.pdf(d1) * vol / (2.0 * np.sqrt(maturity))
    if option_type == "call":
        return common - rate * strike * disc_r * norm.cdf(d2) + dividend * spot * disc_q * norm.cdf(d1)
    if option_type == "put":
        return common + rate * strike * disc_r * norm.cdf(-d2) - dividend * spot * disc_q * norm.cdf(-d1)
    raise ValueError("option_type doit valoir 'call' ou 'put'")


def rho(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
        dividend: ArrayLike, vol: ArrayLike, option_type: str = "call") -> ArrayLike:
    """Rho : sensibilité du prix au taux sans risque, ``dV/dr``."""
    _, d2 = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    disc_r = np.exp(-rate * maturity)
    if option_type == "call":
        return strike * maturity * disc_r * norm.cdf(d2)
    if option_type == "put":
        return -strike * maturity * disc_r * norm.cdf(-d2)
    raise ValueError("option_type doit valoir 'call' ou 'put'")


def vanna(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
          dividend: ArrayLike, vol: ArrayLike) -> ArrayLike:
    """Vanna : ``d(delta)/d(sigma) = d(vega)/dS`` (identique call/put)."""
    d1, d2 = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    disc_q = np.exp(-dividend * maturity)
    return -disc_q * norm.pdf(d1) * d2 / vol


def volga(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
          dividend: ArrayLike, vol: ArrayLike) -> ArrayLike:
    """Volga (vomma) : ``d(vega)/d(sigma)`` (identique call/put)."""
    v = vega(spot, strike, maturity, rate, dividend, vol)
    d1, d2 = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    return v * d1 * d2 / vol


def charm(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
          dividend: ArrayLike, vol: ArrayLike, option_type: str = "call") -> ArrayLike:
    """Charm (delta decay) : ``d(delta)/dT``, T etant la maturité résiduelle."""
    d1, _ = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    disc_q = np.exp(-dividend * maturity)
    dd1 = _dd1_dt(spot, strike, maturity, rate, dividend, vol)
    base = disc_q * norm.pdf(d1) * dd1
    if option_type == "call":
        return -dividend * disc_q * norm.cdf(d1) + base
    if option_type == "put":
        return dividend * disc_q * norm.cdf(-d1) + base
    raise ValueError("option_type doit valoir 'call' ou 'put'")


def speed(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
          dividend: ArrayLike, vol: ArrayLike) -> ArrayLike:
    """Speed : ``d(gamma)/dS``, sensibilité du gamma au spot (identique call/put)."""
    d1, _ = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    g = gamma(spot, strike, maturity, rate, dividend, vol)
    return -g / spot * (d1 / (vol * np.sqrt(maturity)) + 1.0)


def zomma(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
          dividend: ArrayLike, vol: ArrayLike) -> ArrayLike:
    """Zomma : ``d(gamma)/d(sigma)`` (identique call/put)."""
    g = gamma(spot, strike, maturity, rate, dividend, vol)
    d1, d2 = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    return g * (d1 * d2 - 1.0) / vol


def color(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
          dividend: ArrayLike, vol: ArrayLike) -> ArrayLike:
    """Color (gamma decay) : ``d(gamma)/dT`` (identique call/put)."""
    g = gamma(spot, strike, maturity, rate, dividend, vol)
    d1, _ = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    dd1 = _dd1_dt(spot, strike, maturity, rate, dividend, vol)
    return -g * (dividend + d1 * dd1 + 1.0 / (2.0 * maturity))


def veta(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
         dividend: ArrayLike, vol: ArrayLike) -> ArrayLike:
    """Veta : ``d(vega)/dT`` (identique call/put)."""
    v = vega(spot, strike, maturity, rate, dividend, vol)
    d1, _ = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    dd1 = _dd1_dt(spot, strike, maturity, rate, dividend, vol)
    return v * (-dividend - d1 * dd1 + 1.0 / (2.0 * maturity))


def all_greeks(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
               dividend: ArrayLike, vol: ArrayLike, option_type: str = "call") -> dict[str, ArrayLike]:
    """Calcule prix + jeu complet de Grecques en un seul appel."""
    return {
        "price": bs_price(spot, strike, maturity, rate, dividend, vol, option_type),
        "delta": delta(spot, strike, maturity, rate, dividend, vol, option_type),
        "gamma": gamma(spot, strike, maturity, rate, dividend, vol),
        "vega": vega(spot, strike, maturity, rate, dividend, vol),
        "theta": theta(spot, strike, maturity, rate, dividend, vol, option_type),
        "rho": rho(spot, strike, maturity, rate, dividend, vol, option_type),
        "vanna": vanna(spot, strike, maturity, rate, dividend, vol),
        "volga": volga(spot, strike, maturity, rate, dividend, vol),
        "charm": charm(spot, strike, maturity, rate, dividend, vol, option_type),
        "speed": speed(spot, strike, maturity, rate, dividend, vol),
        "zomma": zomma(spot, strike, maturity, rate, dividend, vol),
        "color": color(spot, strike, maturity, rate, dividend, vol),
        "veta": veta(spot, strike, maturity, rate, dividend, vol),
    }

"""Arbre binomial de Cox-Ross-Rubinstein (CRR) : options europeennes et américaines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .blackscholes import bs_price


@dataclass(frozen=True)
class CRRResult:
    """Resultat d'un pricing par arbre CRR."""

    price: float
    delta: float
    gamma: float
    theta: float


def _crr_params(maturity: float, rate: float, dividend: float, vol: float,
                 n_steps: int) -> tuple[float, float, float, float, float]:
    dt = maturity / n_steps
    u = np.exp(vol * np.sqrt(dt))
    d = 1.0 / u
    growth = np.exp((rate - dividend) * dt)
    p = (growth - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError(
            f"Probabilite risque-neutre p={p:.4f} hors de ]0,1[ : reduire le pas de temps "
            "(augmenter n_steps) ou vérifier les parametres de marché."
        )
    discount = np.exp(-rate * dt)
    return dt, u, d, p, discount


def crr_tree_price(spot: float, strike: float, maturity: float, rate: float, dividend: float,
                    vol: float, n_steps: int, option_type: str = "call",
                    exercise: str = "european") -> CRRResult:
    """Prix et Grecques (delta, gamma, theta) par arbre binomial CRR."""
    if n_steps < 2:
        raise ValueError("n_steps doit être >= 2 pour estimer delta/gamma/theta sur l'arbre")
    if option_type not in ("call", "put"):
        raise ValueError("option_type doit valoir 'call' ou 'put'")
    if exercise not in ("european", "american"):
        raise ValueError("exercise doit valoir 'european' ou 'american'")

    dt, u, d, p, discount = _crr_params(maturity, rate, dividend, vol, n_steps)

    j = np.arange(n_steps + 1)
    spot_at_maturity = spot * (u ** j) * (d ** (n_steps - j))
    if option_type == "call":
        values = np.maximum(spot_at_maturity - strike, 0.0)
    else:
        values = np.maximum(strike - spot_at_maturity, 0.0)

    values_at_step2 = None
    for step in range(n_steps - 1, -1, -1):
        continuation = discount * (p * values[1:len(values)] + (1.0 - p) * values[0:len(values) - 1])
        if exercise == "american":
            j_step = np.arange(step + 1)
            spot_at_step = spot * (u ** j_step) * (d ** (step - j_step))
            if option_type == "call":
                intrinsic = np.maximum(spot_at_step - strike, 0.0)
            else:
                intrinsic = np.maximum(strike - spot_at_step, 0.0)
            values = np.maximum(continuation, intrinsic)
        else:
            values = continuation
        if step == 2:
            values_at_step2 = values.copy()

    price = float(values[0])

    s_dd = spot * d ** 2
    s_ud = spot * u * d
    s_uu = spot * u ** 2
    v_dd, v_ud, v_uu = values_at_step2[0], values_at_step2[1], values_at_step2[2]

    delta_up = (v_uu - v_ud) / (s_uu - s_ud)
    delta_down = (v_ud - v_dd) / (s_ud - s_dd)
    delta = (delta_up + delta_down) / 2.0
    gamma = (delta_up - delta_down) / (0.5 * (s_uu - s_dd))
    theta = (v_ud - price) / (2.0 * dt)

    return CRRResult(price=price, delta=float(delta), gamma=float(gamma), theta=float(theta))


def crr_price(spot: float, strike: float, maturity: float, rate: float, dividend: float,
              vol: float, n_steps: int, option_type: str = "call",
              exercise: str = "european") -> float:
    """Raccourci : prix seul par arbre CRR (sans les Grecques à deux niveaux)."""
    return crr_tree_price(spot, strike, maturity, rate, dividend, vol, n_steps,
                           option_type, exercise).price


def crr_greeks_bump(spot: float, strike: float, maturity: float, rate: float, dividend: float,
                     vol: float, n_steps: int, option_type: str = "call",
                     exercise: str = "european", bump_vol: float = 1e-4,
                     bump_rate: float = 1e-4) -> dict[str, float]:
    """Jeu complet de Grecques par bump-and-revalue sur l'arbre CRR."""
    base = crr_tree_price(spot, strike, maturity, rate, dividend, vol, n_steps, option_type, exercise)
    price_vol_up = crr_price(spot, strike, maturity, rate, dividend, vol + bump_vol, n_steps,
                              option_type, exercise)
    price_vol_dn = crr_price(spot, strike, maturity, rate, dividend, vol - bump_vol, n_steps,
                              option_type, exercise)
    vega = (price_vol_up - price_vol_dn) / (2.0 * bump_vol)

    price_r_up = crr_price(spot, strike, maturity, rate + bump_rate, dividend, vol, n_steps,
                            option_type, exercise)
    price_r_dn = crr_price(spot, strike, maturity, rate - bump_rate, dividend, vol, n_steps,
                            option_type, exercise)
    rho = (price_r_up - price_r_dn) / (2.0 * bump_rate)

    return {
        "price": base.price, "delta": base.delta, "gamma": base.gamma,
        "theta": base.theta, "vega": vega, "rho": rho,
    }


def early_exercise_premium(spot: float, strike: float, maturity: float, rate: float,
                            dividend: float, vol: float, n_steps: int,
                            option_type: str = "put") -> float:
    """Prime d'exercice anticipe = prix américain - prix européen (même arbre CRR)."""
    american = crr_price(spot, strike, maturity, rate, dividend, vol, n_steps, option_type, "american")
    european = crr_price(spot, strike, maturity, rate, dividend, vol, n_steps, option_type, "european")
    return american - european


def convergence_to_bs(spot: float, strike: float, maturity: float, rate: float, dividend: float,
                       vol: float, option_type: str, steps_grid: np.ndarray) -> np.ndarray:
    """Ecart |prix CRR européen - prix Black-Scholes| pour une grille de ``n_steps``."""
    bs_ref = bs_price(spot, strike, maturity, rate, dividend, vol, option_type)
    errors = np.array([
        abs(crr_price(spot, strike, maturity, rate, dividend, vol, int(n), option_type, "european") - bs_ref)
        for n in steps_grid
    ])
    return errors

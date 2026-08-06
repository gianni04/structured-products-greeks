"""Moteur Monte Carlo vectorisé : simulation GBM multi-actifs, réduction de variance et Grecques."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class MCResult:
    """Resultat d'une estimation Monte Carlo avec intervalle de confiance."""

    price: float
    stderr: float
    ci_low: float
    ci_high: float
    n_paths: int


def confidence_interval(discounted_payoffs: np.ndarray, confidence: float = 0.95) -> MCResult:
    """Construit un intervalle de confiance gaussien sur la moyenne d'un echantillon."""
    n = discounted_payoffs.shape[0]
    price = float(np.mean(discounted_payoffs))
    stderr = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(n))
    z = float(norm.ppf(0.5 + confidence / 2.0))
    return MCResult(price=price, stderr=stderr, ci_low=price - z * stderr,
                     ci_high=price + z * stderr, n_paths=n)


def simulate_gbm_paths(spot: np.ndarray, rate: float, dividend: np.ndarray, vol: np.ndarray,
                        corr_matrix: np.ndarray, maturity: float, n_steps: int, n_paths: int,
                        seed: int | None = None, antithetic: bool = True) -> np.ndarray:
    """Simule des trajectoires GBM multi-actifs corrélées, entièrement vectorisé."""
    spot = np.atleast_1d(np.asarray(spot, dtype=float))
    dividend = np.broadcast_to(np.atleast_1d(np.asarray(dividend, dtype=float)), spot.shape)
    vol = np.broadcast_to(np.atleast_1d(np.asarray(vol, dtype=float)), spot.shape)
    corr_matrix = np.asarray(corr_matrix, dtype=float)
    n_assets = spot.shape[0]

    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(corr_matrix)

    if antithetic:
        n_half = -(-n_paths // 2)
        eps_half = rng.standard_normal(size=(n_half, n_steps, n_assets))
        eps = np.concatenate([eps_half, -eps_half], axis=0)[:n_paths]
    else:
        eps = rng.standard_normal(size=(n_paths, n_steps, n_assets))

    correlated = eps @ chol.T

    dt = maturity / n_steps
    drift = (rate - dividend - 0.5 * vol ** 2) * dt
    diffusion = vol * np.sqrt(dt)
    log_increments = drift + diffusion * correlated
    cum_log = np.cumsum(log_increments, axis=1)

    paths = np.empty((n_paths, n_steps + 1, n_assets))
    paths[:, 0, :] = spot
    paths[:, 1:, :] = spot * np.exp(cum_log)
    return paths


def simulate_gbm_at_observation_times(spot: np.ndarray, rate: float, dividend: np.ndarray,
                                       vol: np.ndarray, corr_matrix: np.ndarray,
                                       observation_times: np.ndarray, n_paths: int,
                                       seed: int | None = None, antithetic: bool = True) -> np.ndarray:
    """Simule un GBM multi-actifs sur une grille de dates non nécessairement équidistantes."""
    spot = np.atleast_1d(np.asarray(spot, dtype=float))
    dividend = np.broadcast_to(np.atleast_1d(np.asarray(dividend, dtype=float)), spot.shape)
    vol = np.broadcast_to(np.atleast_1d(np.asarray(vol, dtype=float)), spot.shape)
    corr_matrix = np.asarray(corr_matrix, dtype=float)
    observation_times = np.asarray(observation_times, dtype=float)
    n_assets = spot.shape[0]
    n_obs = observation_times.shape[0]

    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(corr_matrix)

    if antithetic:
        n_half = -(-n_paths // 2)
        eps_half = rng.standard_normal(size=(n_half, n_obs, n_assets))
        eps = np.concatenate([eps_half, -eps_half], axis=0)[:n_paths]
    else:
        eps = rng.standard_normal(size=(n_paths, n_obs, n_assets))

    correlated = eps @ chol.T

    dt = np.diff(np.concatenate([[0.0], observation_times]))
    drift = (rate - dividend[None, :] - 0.5 * vol[None, :] ** 2) * dt[:, None]
    diffusion = vol[None, :] * np.sqrt(dt)[:, None]
    log_increments = drift[None, :, :] + diffusion[None, :, :] * correlated
    cum_log = np.cumsum(log_increments, axis=1)

    return spot * np.exp(cum_log)


def control_variate_adjustment(payoffs: np.ndarray, control: np.ndarray,
                                control_mean_analytic: float) -> np.ndarray:
    """Ajuste des payoffs simules par variable de contrôle a esperance connue."""
    cov = float(np.cov(payoffs, control, ddof=1)[0, 1])
    var_c = float(np.var(control, ddof=1))
    beta = 0.0 if var_c < 1e-14 else cov / var_c
    return payoffs - beta * (control - control_mean_analytic)


def mc_vanilla_call(spot: float, strike: float, maturity: float, rate: float, dividend: float,
                     vol: float, n_paths: int, seed: int | None = None, antithetic: bool = True,
                     use_control_variate: bool = True) -> MCResult:
    """Prix Monte Carlo d'un call vanille européen, avec réduction de variance."""
    paths = simulate_gbm_paths(np.array([spot]), rate, np.array([dividend]), np.array([vol]),
                                np.array([[1.0]]), maturity, 1, n_paths, seed, antithetic)
    s_t = paths[:, -1, 0]
    discount = np.exp(-rate * maturity)
    payoff = discount * np.maximum(s_t - strike, 0.0)
    if use_control_variate:
        control_mean = spot * np.exp((rate - dividend) * maturity)
        adjusted = control_variate_adjustment(payoff, s_t, control_mean)
        return confidence_interval(adjusted)
    return confidence_interval(payoff)


def bump_delta_crn(pricer: "callable[[float], float]", spot: float, bump: float = 0.01) -> float:
    """Delta par différence centrée bump-and-revalue, nombres aléatoires communs."""
    h = bump * spot
    return (pricer(spot + h) - pricer(spot - h)) / (2.0 * h)


def bump_gamma_crn(pricer: "callable[[float], float]", spot: float, bump: float = 0.01) -> float:
    """Gamma par différence centrée a 3 points, nombres aléatoires communs (voir :func:`bump_delta_crn`)."""
    h = bump * spot
    return (pricer(spot + h) - 2.0 * pricer(spot) + pricer(spot - h)) / (h ** 2)


def pathwise_delta_call(spot: float, strike: float, maturity: float, rate: float, dividend: float,
                         vol: float, n_paths: int, seed: int | None = None,
                         antithetic: bool = True) -> float:
    """Delta d'un call vanille par la methode pathwise (derivation trajectoire par trajectoire)."""
    paths = simulate_gbm_paths(np.array([spot]), rate, np.array([dividend]), np.array([vol]),
                                np.array([[1.0]]), maturity, 1, n_paths, seed, antithetic)
    s_t = paths[:, -1, 0]
    discount = np.exp(-rate * maturity)
    indicator = (s_t > strike).astype(float)
    estimator = discount * indicator * s_t / spot
    return float(np.mean(estimator))


def likelihood_ratio_delta_call(spot: float, strike: float, maturity: float, rate: float,
                                 dividend: float, vol: float, n_paths: int,
                                 seed: int | None = None, antithetic: bool = True) -> float:
    """Delta d'un call vanille par la methode du score / likelihood ratio."""
    rng = np.random.default_rng(seed)
    if antithetic:
        n_half = -(-n_paths // 2)
        z_half = rng.standard_normal(n_half)
        z = np.concatenate([z_half, -z_half])[:n_paths]
    else:
        z = rng.standard_normal(n_paths)
    s_t = spot * np.exp((rate - dividend - 0.5 * vol ** 2) * maturity + vol * np.sqrt(maturity) * z)
    discount = np.exp(-rate * maturity)
    payoff = np.maximum(s_t - strike, 0.0)
    score = z / (spot * vol * np.sqrt(maturity))
    estimator = discount * payoff * score
    return float(np.mean(estimator))

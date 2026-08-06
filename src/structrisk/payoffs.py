"""Payoffs de produits structures usuels en banque privee / gestion d'actifs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def worst_of_performance(paths: np.ndarray, spot0: np.ndarray) -> np.ndarray:
    """Performance worst-of d'un panier à chaque date simulee."""
    performance = paths / spot0
    return np.min(performance, axis=2)


@dataclass(frozen=True)
class PhoenixAutocallSpec:
    """Fiche produit : Autocall Phoenix (coupon conditionnel a effet memoire)."""

    notional: float
    coupon_rate: float
    autocall_barrier: float
    coupon_barrier: float
    protection_barrier: float
    observation_times: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass(frozen=True)
class PhoenixAutocallCashflows:
    """Flux générés par un Autocall Phoenix, avant actualisation."""

    coupon_cashflows: np.ndarray
    redemption_amount: np.ndarray
    redemption_time: np.ndarray
    called: np.ndarray
    called_at_index: np.ndarray


def phoenix_autocall_cashflows(performance_paths: np.ndarray,
                                spec: PhoenixAutocallSpec) -> PhoenixAutocallCashflows:
    """Genere les flux (coupons + remboursement) d'un Autocall Phoenix, vectorisé sur les chemins."""
    n_paths, n_obs = performance_paths.shape
    if n_obs != len(spec.observation_times):
        raise ValueError("performance_paths et observation_times doivent avoir la même longueur")

    call_trigger = performance_paths >= spec.autocall_barrier
    coupon_trigger = performance_paths >= spec.coupon_barrier

    alive = np.ones(n_paths, dtype=bool)
    periods_since_paid = np.zeros(n_paths, dtype=float)
    coupon_cashflows = np.zeros((n_paths, n_obs))
    called = np.zeros(n_paths, dtype=bool)
    called_at_index = -np.ones(n_paths, dtype=int)

    for j in range(n_obs):
        periods_since_paid[alive] += 1.0

        pay_coupon = alive & coupon_trigger[:, j]
        coupon_cashflows[pay_coupon, j] = spec.coupon_rate * spec.notional * periods_since_paid[pay_coupon]
        periods_since_paid[pay_coupon] = 0.0

        call_now = alive & call_trigger[:, j] & (j < n_obs - 1)
        called[call_now] = True
        called_at_index[call_now] = j
        alive = alive & ~call_now

    redemption_amount = np.empty(n_paths)
    redemption_time = np.empty(n_paths)

    redemption_time[called] = spec.observation_times[called_at_index[called]]
    redemption_amount[called] = spec.notional

    matured = ~called
    final_perf = performance_paths[:, -1]
    protected = matured & (final_perf >= spec.protection_barrier)
    at_risk = matured & (final_perf < spec.protection_barrier)
    redemption_time[matured] = spec.observation_times[-1]
    redemption_amount[protected] = spec.notional
    redemption_amount[at_risk] = spec.notional * final_perf[at_risk]

    return PhoenixAutocallCashflows(
        coupon_cashflows=coupon_cashflows, redemption_amount=redemption_amount,
        redemption_time=redemption_time, called=called, called_at_index=called_at_index,
    )


@dataclass(frozen=True)
class ReverseConvertibleSpec:
    """Fiche produit : Reverse Convertible avec barrière de protection."""

    notional: float
    coupon_rate: float
    barrier: float
    barrier_type: str = "continuous"


def reverse_convertible_payoff(performance_paths: np.ndarray, spec: ReverseConvertibleSpec) -> np.ndarray:
    """Payoff (nominal + coupon) à l'échéance d'une Reverse Convertible."""
    final_perf = performance_paths[:, -1]
    if spec.barrier_type == "continuous":
        breached = np.any(performance_paths < spec.barrier, axis=1)
    elif spec.barrier_type == "european":
        breached = final_perf < spec.barrier
    else:
        raise ValueError("barrier_type doit valoir 'continuous' ou 'european'")

    capital = np.where(breached, spec.notional * final_perf, spec.notional)
    coupon = spec.coupon_rate * spec.notional
    return capital + coupon


@dataclass(frozen=True)
class BonusCapSpec:
    """Fiche produit : Certificat Bonus Cappe."""

    notional: float
    barrier: float
    bonus_level: float
    cap_level: float


def bonus_cap_payoff(performance_paths: np.ndarray, spec: BonusCapSpec) -> np.ndarray:
    """Payoff à l'échéance d'un Certificat Bonus Cappe."""
    final_perf = performance_paths[:, -1]
    breached = np.any(performance_paths < spec.barrier, axis=1)

    payoff_not_breached = np.minimum(np.maximum(final_perf, spec.bonus_level), spec.cap_level)
    payoff_breached = np.minimum(final_perf, spec.cap_level)

    performance = np.where(breached, payoff_breached, payoff_not_breached)
    return spec.notional * performance


@dataclass(frozen=True)
class CapitalProtectedNoteSpec:
    """Fiche produit : Note à capital protégé avec participation et cap."""

    notional: float
    protection_level: float
    participation_rate: float
    strike_level: float
    cap_level: float


def capital_protected_note_payoff(final_performance: np.ndarray, spec: CapitalProtectedNoteSpec) -> np.ndarray:
    """Payoff à l'échéance d'une note à capital protégé avec participation et cap."""
    participation = np.clip(final_performance - spec.strike_level, 0.0, spec.cap_level - spec.strike_level)
    return spec.notional * (spec.protection_level + spec.participation_rate * participation)


def worst_of_basket_payoff(final_performance_worst_of: np.ndarray, notional: float,
                            option_type: str = "call", strike_level: float = 1.0) -> np.ndarray:
    """Payoff d'une option vanille worst-of sur panier (brique de base multi-actifs)."""
    if option_type == "call":
        return notional * np.maximum(final_performance_worst_of - strike_level, 0.0)
    if option_type == "put":
        return notional * np.maximum(strike_level - final_performance_worst_of, 0.0)
    raise ValueError("option_type doit valoir 'call' ou 'put'")

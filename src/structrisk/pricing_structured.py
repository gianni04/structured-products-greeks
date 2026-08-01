"""Valorisation Monte Carlo des produits structures et Grecques associées.

Chaque fonction de pricing s'appuie sur le moteur vectorisé de
``montecarlo.py`` et les fiches produit de ``payoffs.py``. Les Grecques
sont calculées par bump-and-revalue en réutilisant systématiquement les
mêmes nombres aléatoires communs (CRN, même ``seed``) entre le prix de
base et le prix bumpe : cela annule une grande partie du bruit Monte
Carlo dans la différence, condition nécessaire pour obtenir des Grecques
stables sur des produits à barrière (payoff discontinu).

Quand une décomposition analytique en briques existe (note à capital
protégé = zero-coupon + call-spread), elle est fournie en complement du
prix Monte Carlo pour validation croisée.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .blackscholes import bs_price
from .montecarlo import (
    MCResult,
    confidence_interval,
    simulate_gbm_at_observation_times,
    simulate_gbm_paths,
)
from .payoffs import (
    BonusCapSpec,
    CapitalProtectedNoteSpec,
    PhoenixAutocallSpec,
    ReverseConvertibleSpec,
    bonus_cap_payoff,
    capital_protected_note_payoff,
    phoenix_autocall_cashflows,
    reverse_convertible_payoff,
    worst_of_basket_payoff,
)


def _central_diff(pricer: Callable[[float], float], x0: float, bump: float) -> float:
    """Difference centrée ``(f(x0+h)-f(x0-h))/(2h)`` avec h = bump (absolu)."""
    return (pricer(x0 + bump) - pricer(x0 - bump)) / (2.0 * bump)


def _central_diff2(pricer: Callable[[float], float], x0: float, bump: float) -> float:
    """Difference seconde centrée ``(f(x0+h)-2f(x0)+f(x0-h))/h^2`` avec h = bump (absolu)."""
    return (pricer(x0 + bump) - 2.0 * pricer(x0) + pricer(x0 - bump)) / (bump ** 2)


@dataclass(frozen=True)
class PhoenixPricingResult:
    """Resultat de valorisation Monte Carlo d'un Autocall Phoenix.

    Attributes
    ----------
    price, stderr, ci_low, ci_high : float
        Prix Monte Carlo (nominal actualise) et intervalle de confiance.
    call_probability_by_date : np.ndarray, shape (n_obs - 1,)
        Probabilite de rappel à chaque date d'observation anterieure a
        l'échéance (estimee par fréquence empirique sur les trajectoires).
    prob_never_called : float
        Probabilite que le produit survive jusqu'à l'échéance sans rappel.
    prob_capital_loss : float
        Probabilite (conditionnelle à la non-annulation par rappel) que
        la barrière de protection ne soit pas respectee à l'échéance,
        entrainant une perte en capital.
    """

    price: float
    stderr: float
    ci_low: float
    ci_high: float
    call_probability_by_date: np.ndarray
    prob_never_called: float
    prob_capital_loss: float


def price_phoenix_autocall(spot: np.ndarray | float, vol: np.ndarray | float, rate: float,
                            dividend: np.ndarray | float, corr_matrix: np.ndarray,
                            spec: PhoenixAutocallSpec, n_paths: int, seed: int | None = None,
                            antithetic: bool = True, spot_ref: np.ndarray | float | None = None
                            ) -> PhoenixPricingResult:
    """Prix Monte Carlo d'un Autocall Phoenix (mono-sous-jacent ou worst-of panier).

    Parameters
    ----------
    spot, vol, dividend : ArrayLike
        Parametres par actif (scalaires pour un sous-jacent unique,
        tableaux de taille ``n_assets`` pour un panier worst-of). ``spot``
        est le niveau de depart de la simulation (le spot "aujourd'hui").
    corr_matrix : np.ndarray
        Matrice de corrélation du panier (``[[1.0]]`` pour un seul actif).
    spec : PhoenixAutocallSpec
        Fiche produit (cf. ``payoffs.py``) ; les barrières y sont
        exprimées en fraction du niveau de référence initial.
    n_paths : int
        Nombre de trajectoires Monte Carlo.
    seed : int, optional
        Graine (reproductibilite + CRN pour les Grecques, cf.
        :func:`phoenix_delta_vega_gamma`).
    spot_ref : ArrayLike, optional
        Niveau de référence (fixing initial) servant a calculer la
        performance ``S_t / spot_ref`` par rapport auquel les barrières
        sont definies. Par defaut égal à ``spot`` (cas d'une valorisation
        à l'emission). **Doit rester fixe** (le fixing contractuel ne
        bouge pas) quand on calcule un delta par bump-and-revalue sur
        ``spot`` : sans ce decouplage, la performance ``S_t/S_0`` d'un
        GBM est independante du niveau de depart et le delta calcule
        serait artificiellement nul.

    Returns
    -------
    PhoenixPricingResult
    """
    spot_arr = np.atleast_1d(np.asarray(spot, dtype=float))
    ref_arr = spot_arr if spot_ref is None else np.atleast_1d(np.asarray(spot_ref, dtype=float))
    paths = simulate_gbm_at_observation_times(spot_arr, rate, dividend, vol, corr_matrix,
                                               spec.observation_times, n_paths, seed, antithetic)
    performance = np.min(paths / ref_arr, axis=2)  # worst-of (ou perf simple si 1 actif)

    flows = phoenix_autocall_cashflows(performance, spec)

    disc_coupons = flows.coupon_cashflows * np.exp(-rate * spec.observation_times)[None, :]
    disc_redemption = flows.redemption_amount * np.exp(-rate * flows.redemption_time)
    total = disc_coupons.sum(axis=1) + disc_redemption

    ci = confidence_interval(total)

    n_obs = len(spec.observation_times)
    call_prob = np.array([np.mean(flows.called_at_index == j) for j in range(n_obs - 1)])
    prob_never_called = float(np.mean(~flows.called))
    matured = ~flows.called
    final_perf = performance[:, -1]
    prob_capital_loss = float(np.mean(matured & (final_perf < spec.protection_barrier))) / max(prob_never_called, 1e-12)

    return PhoenixPricingResult(
        price=ci.price, stderr=ci.stderr, ci_low=ci.ci_low, ci_high=ci.ci_high,
        call_probability_by_date=call_prob, prob_never_called=prob_never_called,
        prob_capital_loss=prob_capital_loss,
    )


def phoenix_delta_vega_gamma(spot: float, vol: float, rate: float, dividend: float,
                              spec: PhoenixAutocallSpec, n_paths: int, seed: int,
                              bump_spot_frac: float = 0.01, bump_vol_abs: float = 0.01) -> dict[str, float]:
    """Delta, gamma et vega d'un Autocall Phoenix mono-sous-jacent par bump-and-revalue (CRN).

    Le même ``seed`` est reutilise pour le prix de base et les prix
    bumpes (nombres aléatoires communs) : les trois simulations partagent
    exactement les mêmes chocs, ce qui rend la différence finie
    beaucoup moins bruitee qu'avec des tirages independants -- essentiel
    ici car le payoff est discontinu aux barrières.

    Returns
    -------
    dict[str, float]
        Cles : ``price, delta, gamma, vega``.
    """
    corr = np.array([[1.0]])

    def price_spot(s: float) -> float:
        # spot_ref reste fixe au niveau de référence initial : seul le
        # niveau de depart de la simulation bouge (cf. docstring de
        # price_phoenix_autocall pour la justification).
        return price_phoenix_autocall(s, vol, rate, dividend, corr, spec, n_paths, seed,
                                       spot_ref=spot).price

    def price_vol(v: float) -> float:
        return price_phoenix_autocall(spot, v, rate, dividend, corr, spec, n_paths, seed,
                                       spot_ref=spot).price

    base_price = price_phoenix_autocall(spot, vol, rate, dividend, corr, spec, n_paths, seed,
                                         spot_ref=spot).price
    h_spot = bump_spot_frac * spot
    delta = (price_spot(spot + h_spot) - price_spot(spot - h_spot)) / (2.0 * h_spot)
    gamma = (price_spot(spot + h_spot) - 2.0 * base_price + price_spot(spot - h_spot)) / (h_spot ** 2)
    vega = (price_vol(vol + bump_vol_abs) - price_vol(vol - bump_vol_abs)) / (2.0 * bump_vol_abs)

    return {"price": base_price, "delta": delta, "gamma": gamma, "vega": vega}


def phoenix_barrier_sensitivity(spot: float, vol: float, rate: float, dividend: float,
                                 spec: PhoenixAutocallSpec, barrier_name: str,
                                 barrier_values: np.ndarray, n_paths: int, seed: int) -> np.ndarray:
    """Prix de l'Autocall Phoenix pour une grille de niveaux d'une barrière donnee.

    Le même seed est utilise pour tous les niveaux (CRN) : la courbe de
    sensibilité obtenue est lisse (même jeu de trajectoires sous-jacentes
    pour chaque niveau de barrière testee), pas polluee par du bruit
    Monte Carlo independant d'un point à l'autre.

    Parameters
    ----------
    barrier_name : str
        ``"autocall_barrier"``, ``"coupon_barrier"`` ou ``"protection_barrier"``.
    barrier_values : np.ndarray
        Niveaux de barrière a tester (fraction du spot initial).

    Returns
    -------
    np.ndarray
        Prix pour chaque niveau de ``barrier_values``.
    """
    corr = np.array([[1.0]])
    prices = np.empty(len(barrier_values))
    for i, level in enumerate(barrier_values):
        kwargs = {
            "notional": spec.notional, "coupon_rate": spec.coupon_rate,
            "autocall_barrier": spec.autocall_barrier, "coupon_barrier": spec.coupon_barrier,
            "protection_barrier": spec.protection_barrier, "observation_times": spec.observation_times,
        }
        kwargs[barrier_name] = level
        bumped_spec = PhoenixAutocallSpec(**kwargs)
        prices[i] = price_phoenix_autocall(spot, vol, rate, dividend, corr, bumped_spec, n_paths, seed).price
    return prices


def phoenix_dividend_sensitivity(spot: float, vol: float, rate: float,
                                  spec: PhoenixAutocallSpec, dividend_values: np.ndarray,
                                  n_paths: int, seed: int) -> np.ndarray:
    """Prix de l'Autocall Phoenix pour une grille de taux de dividende (CRN, même seed partout).

    Usage en gestion des risques : un dividende plus élevé réduit la
    dérive risque-neutre du sous-jacent, donc réduit la probabilité de
    rappel et augmente le risque de toucher la barrière de protection --
    sensibilité clé pour les sous-jacents a fort rendement de dividende.
    """
    corr = np.array([[1.0]])
    prices = np.empty(len(dividend_values))
    for i, q in enumerate(dividend_values):
        prices[i] = price_phoenix_autocall(spot, vol, rate, q, corr, spec, n_paths, seed).price
    return prices


def price_reverse_convertible(spot: float, vol: float, rate: float, dividend: float,
                               maturity: float, spec: ReverseConvertibleSpec, n_paths: int,
                               n_steps: int, seed: int | None = None, antithetic: bool = True,
                               spot_ref: float | None = None) -> MCResult:
    """Prix Monte Carlo d'une Reverse Convertible avec barrière (cf. :mod:`payoffs`).

    ``n_steps`` doit être suffisamment fin (ex : quotidien) si
    ``spec.barrier_type == "continuous"`` pour limiter le biais de
    discretisation de la barrière (une barrière surveillee sur une grille
    discrète est structurellement moins souvent touchee qu'une vraie
    barrière continue -- limite documentee dans le README).

    ``spot_ref`` (niveau de référence du fixing initial, par defaut
    ``spot``) doit rester fixe lors d'un bump-and-revalue sur ``spot``
    pour obtenir un delta non trivial (cf. :func:`price_phoenix_autocall`).
    """
    ref = spot if spot_ref is None else spot_ref
    paths = simulate_gbm_paths(np.array([spot]), rate, np.array([dividend]), np.array([vol]),
                                np.array([[1.0]]), maturity, n_steps, n_paths, seed, antithetic)
    performance = paths[:, :, 0] / ref
    payoff = reverse_convertible_payoff(performance, spec)
    discounted = payoff * np.exp(-rate * maturity)
    return confidence_interval(discounted)


def price_bonus_cap(spot: float, vol: float, rate: float, dividend: float, maturity: float,
                     spec: BonusCapSpec, n_paths: int, n_steps: int, seed: int | None = None,
                     antithetic: bool = True, spot_ref: float | None = None) -> MCResult:
    """Prix Monte Carlo d'un Certificat Bonus Cappe (cf. :mod:`payoffs`).

    ``spot_ref`` : voir :func:`price_reverse_convertible`.
    """
    ref = spot if spot_ref is None else spot_ref
    paths = simulate_gbm_paths(np.array([spot]), rate, np.array([dividend]), np.array([vol]),
                                np.array([[1.0]]), maturity, n_steps, n_paths, seed, antithetic)
    performance = paths[:, :, 0] / ref
    payoff = bonus_cap_payoff(performance, spec)
    discounted = payoff * np.exp(-rate * maturity)
    return confidence_interval(discounted)


def price_capital_protected_note_mc(spot: float, vol: float, rate: float, dividend: float,
                                     maturity: float, spec: CapitalProtectedNoteSpec, n_paths: int,
                                     seed: int | None = None, antithetic: bool = True,
                                     spot_ref: float | None = None) -> MCResult:
    """Prix Monte Carlo d'une note à capital protégé avec participation et cap.

    ``spot_ref`` : voir :func:`price_reverse_convertible`.
    """
    ref = spot if spot_ref is None else spot_ref
    paths = simulate_gbm_paths(np.array([spot]), rate, np.array([dividend]), np.array([vol]),
                                np.array([[1.0]]), maturity, 1, n_paths, seed, antithetic)
    final_performance = paths[:, -1, 0] / ref
    payoff = capital_protected_note_payoff(final_performance, spec)
    discounted = payoff * np.exp(-rate * maturity)
    return confidence_interval(discounted)


def price_capital_protected_note_decomposition(spot: float, vol: float, rate: float, dividend: float,
                                                maturity: float, spec: CapitalProtectedNoteSpec) -> float:
    """Prix par décomposition en briques (zero-coupon + call-spread) d'une note à capital protégé.

    Formule
    -------
    ``Prix = protection_level * notional * e^{-rT}``
    ``     + participation_rate * (notional / spot) * [ Call(K=strike_level*spot) - Call(K=cap_level*spot) ]``

    ou ``Call(K)`` est le prix Black-Scholes-Merton d'un call vanille de
    strike ``K``. Cette décomposition est exacte (pas d'approximation) et
    sert de référence analytique pour valider le prix Monte Carlo.
    """
    zero_coupon = spec.protection_level * spec.notional * np.exp(-rate * maturity)
    strike_low = spec.strike_level * spot
    strike_high = spec.cap_level * spot
    call_low = bs_price(spot, strike_low, maturity, rate, dividend, vol, "call")
    call_high = bs_price(spot, strike_high, maturity, rate, dividend, vol, "call")
    call_spread_value = call_low - call_high
    option_leg = spec.participation_rate * (spec.notional / spot) * call_spread_value
    return float(zero_coupon + option_leg)


def price_worst_of_basket(spot: np.ndarray, vol: np.ndarray, rate: float, dividend: np.ndarray,
                           corr_matrix: np.ndarray, maturity: float, notional: float,
                           n_paths: int, seed: int | None = None, antithetic: bool = True,
                           option_type: str = "call", strike_level: float = 1.0,
                           spot_ref: np.ndarray | None = None) -> MCResult:
    """Prix Monte Carlo d'une option vanille worst-of sur panier corrélé (cf. :mod:`payoffs`).

    ``spot_ref`` : voir :func:`price_reverse_convertible`.
    """
    spot = np.asarray(spot, dtype=float)
    ref = spot if spot_ref is None else np.asarray(spot_ref, dtype=float)
    paths = simulate_gbm_paths(spot, rate, dividend, vol, corr_matrix, maturity, 1, n_paths, seed, antithetic)
    final_perf_worst_of = np.min(paths[:, -1, :] / ref, axis=1)
    payoff = worst_of_basket_payoff(final_perf_worst_of, notional, option_type, strike_level)
    discounted = payoff * np.exp(-rate * maturity)
    return confidence_interval(discounted)

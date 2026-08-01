"""Payoffs de produits structures usuels en banque privee / gestion d'actifs.

Chaque fonction est vectorisée sur les trajectoires (le seul axe qui peut
compter des centaines de milliers d'elements) : les boucles Python
restantes, quand il y en a, portent uniquement sur le petit nombre de
dates d'observation du produit (typiquement 4 a 20), jamais sur les
trajectoires. Toutes les fonctions travaillent en "performance" (ratio
``S_t / S_0``), ce qui permet de reutiliser le même code pour un
sous-jacent unique ou pour un panier déjà réduit en performance
worst-of via :func:`worst_of_performance`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def worst_of_performance(paths: np.ndarray, spot0: np.ndarray) -> np.ndarray:
    """Performance worst-of d'un panier à chaque date simulee.

    Fiche produit
    -------------
    Le "worst-of" est la performance du plus mauvais actif du panier :
    ``perf(t) = min_i( S_i(t) / S_i(0) )``. C'est la brique de base de la
    plupart des autocalls et notes sur panier, car elle capture le risque
    de corrélation (une baisse de corrélation dégrade fortement le prix
    du worst-of).

    Parameters
    ----------
    paths : np.ndarray, shape (n_paths, n_dates, n_assets)
        Trajectoires simulees (cf. :func:`structrisk.montecarlo.simulate_gbm_paths`).
    spot0 : np.ndarray, shape (n_assets,)
        Spots initiaux de chaque actif du panier.

    Returns
    -------
    np.ndarray, shape (n_paths, n_dates)
        Performance worst-of à chaque date.
    """
    performance = paths / spot0
    return np.min(performance, axis=2)


@dataclass(frozen=True)
class PhoenixAutocallSpec:
    """Fiche produit : Autocall Phoenix (coupon conditionnel a effet memoire).

    Mecanisme
    ---------
    A chaque date d'observation ``t_j`` :

    1. **Rappel automatique (autocall)** : si la performance du
       sous-jacent (ou du worst-of panier) est supérieure ou égale à la
       barrière de rappel ``autocall_barrier``, le produit est rembourse
       au nominal a cette date, augmente du/des coupon(s) dus.
    2. **Coupon conditionnel a memoire** : si la performance est
       supérieure ou égale à la barrière de coupon ``coupon_barrier``
       (généralement inférieure ou égale à la barrière de rappel), un
       coupon est versé. Grâce à "l'effet memoire", si des coupons
       n'avaient pas été verses lors d'observations precedentes (barrière
       de coupon non atteinte), ils sont rattrapes des que la barrière
       est de nouveau franchie : le coupon versé à la date ``j`` vaut
       ``coupon_rate * notional * (nombre de périodes depuis le dernier
       coupon versé, incluses)``.
    3. **Protection du capital à l'échéance** : si le produit n'a jamais
       été rappelé, à la dernière date d'observation (l'échéance) :
       si la performance finale est supérieure ou égale à la barrière de
       protection ``protection_barrier``, le nominal est rembourse
       intégralement ; sinon le capital est rembourse au prorata de la
       performance finale (perte en capital, produit "capital a risque").

    Attributes
    ----------
    notional : float
        Nominal investi.
    coupon_rate : float
        Taux de coupon conditionnel par période d'observation (ex : 0.02
        pour un coupon trimestriel de 2 % du nominal).
    autocall_barrier : float
        Barrière de rappel automatique, en fraction du spot initial (ex :
        1.00 pour 100 %).
    coupon_barrier : float
        Barrière de coupon, en fraction du spot initial (ex : 0.70).
    protection_barrier : float
        Barrière de protection du capital à l'échéance, en fraction du
        spot initial (ex : 0.60).
    observation_times : np.ndarray
        Dates d'observation en années, strictement croissantes, la
        dernière etant l'échéance du produit.
    """

    notional: float
    coupon_rate: float
    autocall_barrier: float
    coupon_barrier: float
    protection_barrier: float
    observation_times: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass(frozen=True)
class PhoenixAutocallCashflows:
    """Flux générés par un Autocall Phoenix, avant actualisation.

    Attributes
    ----------
    coupon_cashflows : np.ndarray, shape (n_paths, n_obs)
        Montant du coupon versé à chaque date d'observation (0 si aucun).
    redemption_amount : np.ndarray, shape (n_paths,)
        Montant du remboursement du nominal (au rappel ou à l'échéance).
    redemption_time : np.ndarray, shape (n_paths,)
        Date (en années) à laquelle le remboursement du nominal intervient.
    called : np.ndarray, shape (n_paths,), dtype bool
        True si le produit a été rappele avant l'échéance.
    called_at_index : np.ndarray, shape (n_paths,), dtype int
        Indice de la date d'observation de rappel (-1 si jamais rappele).
    """

    coupon_cashflows: np.ndarray
    redemption_amount: np.ndarray
    redemption_time: np.ndarray
    called: np.ndarray
    called_at_index: np.ndarray


def phoenix_autocall_cashflows(performance_paths: np.ndarray,
                                spec: PhoenixAutocallSpec) -> PhoenixAutocallCashflows:
    """Genere les flux (coupons + remboursement) d'un Autocall Phoenix, vectorisé sur les chemins.

    La seule boucle explicite porte sur ``n_obs`` dates d'observation
    (petit nombre, typiquement < 20), toutes les operations à l'intérieur
    de la boucle sont vectorisées sur l'ensemble des trajectoires.

    Parameters
    ----------
    performance_paths : np.ndarray, shape (n_paths, n_obs)
        Performance (``S_t/S_0`` ou worst-of panier) à chaque date
        d'observation de :attr:`PhoenixAutocallSpec.observation_times`.
    spec : PhoenixAutocallSpec
        Fiche produit.

    Returns
    -------
    PhoenixAutocallCashflows
        Flux générés, a actualiser ensuite date par date.
    """
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

        call_now = alive & call_trigger[:, j] & (j < n_obs - 1)  # rappel possible avant l'échéance
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
    """Fiche produit : Reverse Convertible avec barrière de protection.

    Mecanisme
    ---------
    Le produit versé un coupon fixe garanti ``coupon_rate * notional`` a
    l'échéance (independant de la performance du sous-jacent, c'est la
    contrepartie du risque actions pris par l'investisseur). A
    l'échéance :

    - si la barrière de protection n'a **jamais** été franchie à la
      baisse (si ``barrier_type="continuous"``, vérifiée à chaque date
      simulee ; si ``barrier_type="european"``, vérifiée seulement à
      l'échéance) : remboursement intégral du nominal ;
    - sinon : remboursement au prorata de la performance finale du
      sous-jacent (perte en capital, l'investisseur est "converti" en
      exposition actions négative).

    Attributes
    ----------
    notional : float
        Nominal investi.
    coupon_rate : float
        Coupon fixe garanti (fraction du nominal), versé à l'échéance.
    barrier : float
        Niveau de barrière de protection, en fraction du spot initial.
    barrier_type : str
        ``"continuous"`` (barrière américaine, surveillee sur toute la
        trajectoire simulee) ou ``"european"`` (surveillee seulement à
        l'échéance).
    """

    notional: float
    coupon_rate: float
    barrier: float
    barrier_type: str = "continuous"


def reverse_convertible_payoff(performance_paths: np.ndarray, spec: ReverseConvertibleSpec) -> np.ndarray:
    """Payoff (nominal + coupon) à l'échéance d'une Reverse Convertible.

    Parameters
    ----------
    performance_paths : np.ndarray, shape (n_paths, n_steps)
        Performance du sous-jacent à chaque pas de temps simule (utilisée
        entièrement si barrière continue, ou seulement la dernière
        colonne si barrière européenne).
    spec : ReverseConvertibleSpec
        Fiche produit.

    Returns
    -------
    np.ndarray, shape (n_paths,)
        Payoff total (capital + coupon) à l'échéance, non actualisé.
    """
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
    """Fiche produit : Certificat Bonus Cappe.

    Mecanisme
    ---------
    Aucun coupon. A l'échéance :

    - si la barrière de protection ``barrier`` n'a jamais été franchie à
      la baisse sur la trajectoire (barrière continue) : l'investisseur
      recoit le meilleur des deux montants -- le niveau bonus garanti
      ``bonus_level`` ou la performance reelle du sous-jacent -- plafonne
      au niveau ``cap_level`` (le "bonus" protégé la performance minimale
      même si le sous-jacent est reste plat ou legerement baissier, sans
      toucher la barrière) ;
    - si la barrière a été franchie : la protection bonus disparait,
      l'investisseur recoit la performance reelle du sous-jacent
      (plafonnee au cap), intégralement exposée à la baisse.

    Attributes
    ----------
    notional : float
        Nominal investi.
    barrier : float
        Niveau de barrière de protection, en fraction du spot initial.
    bonus_level : float
        Niveau de performance garanti si la barrière n'est pas touchee
        (ex : 1.10 pour +10 %), en fraction du spot initial.
    cap_level : float
        Plafond de performance (ex : 1.30 pour +30 %), en fraction du
        spot initial ; doit être supérieur ou égal à ``bonus_level``.
    """

    notional: float
    barrier: float
    bonus_level: float
    cap_level: float


def bonus_cap_payoff(performance_paths: np.ndarray, spec: BonusCapSpec) -> np.ndarray:
    """Payoff à l'échéance d'un Certificat Bonus Cappe.

    Parameters
    ----------
    performance_paths : np.ndarray, shape (n_paths, n_steps)
        Performance du sous-jacent à chaque pas de temps simule (barrière
        surveillee en continu sur toute la trajectoire).
    spec : BonusCapSpec
        Fiche produit.

    Returns
    -------
    np.ndarray, shape (n_paths,)
        Payoff à l'échéance, non actualisé.
    """
    final_perf = performance_paths[:, -1]
    breached = np.any(performance_paths < spec.barrier, axis=1)

    payoff_not_breached = np.minimum(np.maximum(final_perf, spec.bonus_level), spec.cap_level)
    payoff_breached = np.minimum(final_perf, spec.cap_level)

    performance = np.where(breached, payoff_breached, payoff_not_breached)
    return spec.notional * performance


@dataclass(frozen=True)
class CapitalProtectedNoteSpec:
    """Fiche produit : Note à capital protégé avec participation et cap.

    Mecanisme
    ---------
    A l'échéance, l'investisseur recoit :

    ``notional * ( protection_level + participation_rate *
    clip(perf_final - strike_level, 0, cap_level - strike_level) )``

    Decomposition en briques (utilisée dans ``pricing_structured.py``) :
    ce payoff se réplique exactement par un **zero-coupon** de nominal
    ``protection_level * notional`` versant à l'échéance, plus
    ``participation_rate * notional / spot_0`` **call-spreads** (achat
    d'un call de strike ``strike_level * spot_0``, vente d'un call de
    strike ``cap_level * spot_0``, même maturité).

    Attributes
    ----------
    notional : float
        Nominal investi.
    protection_level : float
        Niveau de capital garanti à l'échéance (ex : 1.00 pour 100 %).
    participation_rate : float
        Taux de participation à la hausse au-delà de ``strike_level``.
    strike_level : float
        Niveau (en performance) a partir duquel la participation
        commence (ex : 1.00 = à la monnaie).
    cap_level : float
        Niveau (en performance) au-delà duquel la participation s'arrête
        (plafond de gain).
    """

    notional: float
    protection_level: float
    participation_rate: float
    strike_level: float
    cap_level: float


def capital_protected_note_payoff(final_performance: np.ndarray, spec: CapitalProtectedNoteSpec) -> np.ndarray:
    """Payoff à l'échéance d'une note à capital protégé avec participation et cap.

    Parameters
    ----------
    final_performance : np.ndarray, shape (n_paths,)
        Performance du sous-jacent à l'échéance.
    spec : CapitalProtectedNoteSpec
        Fiche produit.

    Returns
    -------
    np.ndarray, shape (n_paths,)
        Payoff à l'échéance, non actualisé.
    """
    participation = np.clip(final_performance - spec.strike_level, 0.0, spec.cap_level - spec.strike_level)
    return spec.notional * (spec.protection_level + spec.participation_rate * participation)


def worst_of_basket_payoff(final_performance_worst_of: np.ndarray, notional: float,
                            option_type: str = "call", strike_level: float = 1.0) -> np.ndarray:
    """Payoff d'une option vanille worst-of sur panier (brique de base multi-actifs).

    Fiche produit
    -------------
    Option (call ou put) dont le sous-jacent est la performance du plus
    mauvais actif du panier à l'échéance : ``payoff = notional *
    max(w * (perf_worst_of - strike_level), 0)`` avec ``w=+1`` pour un
    call, ``w=-1`` pour un put. Sensible non seulement aux volatilités
    individuelles mais aussi -- fortement -- à la corrélation entre les
    actifs du panier (une baisse de corrélation augmente la dispersion du
    minimum, donc baisse le prix d'un call worst-of).

    Parameters
    ----------
    final_performance_worst_of : np.ndarray, shape (n_paths,)
        Performance worst-of à l'échéance (cf. :func:`worst_of_performance`).
    notional : float
        Nominal / taille de la position.
    option_type : str
        ``"call"`` ou ``"put"``.
    strike_level : float
        Strike, en performance (ex : 1.00 = à la monnaie).

    Returns
    -------
    np.ndarray, shape (n_paths,)
        Payoff à l'échéance, non actualisé.
    """
    if option_type == "call":
        return notional * np.maximum(final_performance_worst_of - strike_level, 0.0)
    if option_type == "put":
        return notional * np.maximum(strike_level - final_performance_worst_of, 0.0)
    raise ValueError("option_type doit valoir 'call' ou 'put'")

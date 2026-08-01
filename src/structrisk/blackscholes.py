"""Modèle de Black-Scholes-Merton (dividende continu) et jeu complet de Grecques.

Convention notée dans tout le module :

    d1 = [ln(S/K) + (r - q + 0.5 * sigma**2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

avec S le spot, K le strike, r le taux sans risque continu, q le taux de
dividende continu, sigmà la volatilité annualisee et T la maturité en années.

Toutes les Grecques d'ordre supérieur (vanna, volga, charm, speed, zomma,
color, veta) sont dérivées analytiquement par derivation directe des
formules de d1/d2 (et non recopiees d'une table), ce qui garantit la
cohérence interne : chaque formule a été vérifiée par différence finie
lors du developpement (voir tests/test_blackscholes.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

ArrayLike = np.ndarray | float


@dataclass(frozen=True)
class BSMParams:
    """Parametres de marché et de contrat pour un pricing Black-Scholes-Merton.

    Parameters
    ----------
    spot : float
        Prix du sous-jacent aujourd'hui (S).
    strike : float
        Prix d'exercice de l'option (K).
    maturity : float
        Maturité résiduelle en années (T > 0).
    rate : float
        Taux sans risque continu (r).
    dividend : float
        Taux de dividende continu (q).
    vol : float
        Volatilite annualisee (sigma > 0).
    """

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
    """Dérivée partielle ddelta(d1)/dT = (r-q)/(sigma*sqrt(T)) - d2/(2T).

    Utilisee comme brique commune pour charm, color et veta.
    """
    _, d2 = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    sqrt_t = np.sqrt(maturity)
    return (rate - dividend) / (vol * sqrt_t) - d2 / (2.0 * maturity)


def bs_price(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
             dividend: ArrayLike, vol: ArrayLike, option_type: str = "call") -> ArrayLike:
    """Prix Black-Scholes-Merton d'un call ou put européen.

    Formules
    --------
    Call : ``C = S e^{-qT} N(d1) - K e^{-rT} N(d2)``
    Put  : ``P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)``

    Parameters
    ----------
    spot, strike, maturity, rate, dividend, vol : ArrayLike
        Parametres de marché (voir :class:`BSMParams`). Peuvent être des
        scalaires ou des tableaux numpy broadcastables entre eux.
    option_type : str
        ``"call"`` ou ``"put"``.

    Returns
    -------
    ArrayLike
        Prix de l'option (même forme que les entrées broadcastées).
    """
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
    """Delta : sensibilité du prix à une variation du spot, ``dV/dS``.

    Formule
    -------
    Call : ``e^{-qT} N(d1)`` ; Put : ``e^{-qT} (N(d1) - 1) = -e^{-qT} N(-d1)``.

    Usage en gestion des risques
    -----------------------------
    Quantité de sous-jacent à détenir pour être couvert au premier ordre
    (delta-hedge). Également utilise comme "exposition delta-équivalente"
    d'une position optionnelle en risque de portefeuille.
    """
    d1, _ = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    disc_q = np.exp(-dividend * maturity)
    if option_type == "call":
        return disc_q * norm.cdf(d1)
    if option_type == "put":
        return disc_q * (norm.cdf(d1) - 1.0)
    raise ValueError("option_type doit valoir 'call' ou 'put'")


def gamma(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
          dividend: ArrayLike, vol: ArrayLike) -> ArrayLike:
    """Gamma : convexité du prix par rapport au spot, ``d^2V/dS^2`` (identique call/put).

    Formule
    -------
    ``Gamma = e^{-qT} * phi(d1) / (S * sigma * sqrt(T))``

    Usage en gestion des risques
    -----------------------------
    Mesure la vitesse à laquelle le delta se dégrade quand le spot bouge :
    pilote le coût de re-hedging (plus le gamma est élevé, plus le
    rebalancement delta doit être fréquent). Toujours positif pour une
    position longue en option vanille.
    """
    d1, _ = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    disc_q = np.exp(-dividend * maturity)
    return disc_q * norm.pdf(d1) / (spot * vol * np.sqrt(maturity))


def vega(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
         dividend: ArrayLike, vol: ArrayLike) -> ArrayLike:
    """Vega : sensibilité du prix à la volatilité, ``dV/dsigma`` (identique call/put).

    Formule
    -------
    ``Vega = S * e^{-qT} * phi(d1) * sqrt(T)``

    Usage en gestion des risques
    -----------------------------
    Exposition au risque de volatilité implicite (souvent exprimée "par
    point de vol", i.e. Vega / 100). Clé pour le risque de smile / de
    surface de vol sur un book d'options ou de structures.
    """
    d1, _ = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    disc_q = np.exp(-dividend * maturity)
    return spot * disc_q * norm.pdf(d1) * np.sqrt(maturity)


def theta(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
          dividend: ArrayLike, vol: ArrayLike, option_type: str = "call") -> ArrayLike:
    """Theta : sensibilité du prix au passage du temps, ``dV/dt = -dV/dT``.

    Formule (convention "par an", diviser par 365 pour un theta journalier)
    -------------------------------------------------------------------
    Call : ``-S e^{-qT} phi(d1) sigma / (2 sqrt(T)) - r K e^{-rT} N(d2) + q S e^{-qT} N(d1)``

    Put  : ``-S e^{-qT} phi(d1) sigma / (2 sqrt(T)) + r K e^{-rT} N(-d2) - q S e^{-qT} N(-d1)``

    Usage en gestion des risques
    -----------------------------
    P&L attendu du seul écoulement du temps, a spot et vol inchangés (le
    "carry" de la position optionnelle). Une position longue en options a
    typiquement un theta négatif : c'est le coût de portage de la convexité
    (gamma).
    """
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
    """Rho : sensibilité du prix au taux sans risque, ``dV/dr``.

    Formule
    -------
    Call : ``K T e^{-rT} N(d2)`` ; Put : ``-K T e^{-rT} N(-d2)``.

    Usage en gestion des risques
    -----------------------------
    Risque de taux d'un book d'options, généralement secondaire pour des
    options courtes mais significatif pour les structures longues (notes
    à capital protégé dont la brique obligataire domine le rho).
    """
    _, d2 = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    disc_r = np.exp(-rate * maturity)
    if option_type == "call":
        return strike * maturity * disc_r * norm.cdf(d2)
    if option_type == "put":
        return -strike * maturity * disc_r * norm.cdf(-d2)
    raise ValueError("option_type doit valoir 'call' ou 'put'")


def vanna(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
          dividend: ArrayLike, vol: ArrayLike) -> ArrayLike:
    """Vanna : ``d(delta)/d(sigma) = d(vega)/dS`` (identique call/put).

    Formule
    -------
    ``Vanna = -e^{-qT} * phi(d1) * d2 / sigma``

    Usage en gestion des risques
    -----------------------------
    Mesure comment le delta d'une position dérive quand la vol bouge (ou,
    de maniere équivalente, comment le vega dérive quand le spot bouge).
    Clé pour les stratégies de risk reversal et pour le risque de "vol de
    vol / spot-vol corrélation" sur un book delta-hedge.
    """
    d1, d2 = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    disc_q = np.exp(-dividend * maturity)
    return -disc_q * norm.pdf(d1) * d2 / vol


def volga(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
          dividend: ArrayLike, vol: ArrayLike) -> ArrayLike:
    """Volga (vomma) : ``d(vega)/d(sigma)`` (identique call/put).

    Formule
    -------
    ``Volga = Vega * d1 * d2 / sigma``

    Usage en gestion des risques
    -----------------------------
    Convexite de la position par rapport à la vol : détermine la
    sensibilité au smile / à la convexité de la surface de vol, utile pour
    arbitrer straddles vs strangles ou juger le coût d'un book long
    volatilité-de-la-volatilité.
    """
    v = vega(spot, strike, maturity, rate, dividend, vol)
    d1, d2 = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    return v * d1 * d2 / vol


def charm(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
          dividend: ArrayLike, vol: ArrayLike, option_type: str = "call") -> ArrayLike:
    """Charm (delta decay) : ``d(delta)/dT``, T etant la maturité résiduelle.

    Formule
    -------
    Soit ``Dd1 = (r-q)/(sigma*sqrt(T)) - d2/(2T)``.

    Call : ``-q e^{-qT} N(d1) + e^{-qT} phi(d1) * Dd1``
    Put  : ``q e^{-qT} N(-d1) + e^{-qT} phi(d1) * Dd1``

    Convention de signe
    --------------------
    Défini ici comme la dérivée par rapport à la maturité résiduelle T (et
    non par rapport au temps calendaire écoulé). Comme T diminue d'une
    unité par an quand le temps passe, la variation de delta observée au
    fil du temps, à spot et vol fixes, vaut approximativement
    ``-charm / 365`` par jour.

    Usage en gestion des risques
    -----------------------------
    Anticipe le rehedge delta nécessaire au seul passage du temps (utile
    la veille d'un week-end ou d'une longue coupure de marché, quand on ne
    veut pas être exposé sans pouvoir rehedger).
    """
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
    """Speed : ``d(gamma)/dS``, sensibilité du gamma au spot (identique call/put).

    Formule
    -------
    ``Speed = -Gamma/S * (d1/(sigma*sqrt(T)) + 1)``

    Usage en gestion des risques
    -----------------------------
    Anticipe l'évolution du gamma lorsque le spot bouge fortement : utile
    pour juger le risque de "gamma qui explose" près d'une barrière ou
    d'un strike proche de la monnaie à l'approche de l'échéance.
    """
    d1, _ = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    g = gamma(spot, strike, maturity, rate, dividend, vol)
    return -g / spot * (d1 / (vol * np.sqrt(maturity)) + 1.0)


def zomma(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
          dividend: ArrayLike, vol: ArrayLike) -> ArrayLike:
    """Zomma : ``d(gamma)/d(sigma)`` (identique call/put).

    Formule
    -------
    ``Zomma = Gamma * (d1*d2 - 1) / sigma``

    Usage en gestion des risques
    -----------------------------
    Mesure comment le gamma (donc le coût de rehedge) se déforme quand la
    vol implicite bouge ; utile pour les livres de gamma-scalping en
    régime de vol instable.
    """
    g = gamma(spot, strike, maturity, rate, dividend, vol)
    d1, d2 = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    return g * (d1 * d2 - 1.0) / vol


def color(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
          dividend: ArrayLike, vol: ArrayLike) -> ArrayLike:
    """Color (gamma decay) : ``d(gamma)/dT`` (identique call/put).

    Formule
    -------
    Soit ``Dd1 = (r-q)/(sigma*sqrt(T)) - d2/(2T)``.

    ``Color = -Gamma * (q + d1*Dd1 + 1/(2T))``

    Convention de signe : idem charm, dérivée par rapport à la maturité
    résiduelle T (variation calendaire quotidienne du gamma
    approximativement ``-color / 365``).

    Usage en gestion des risques
    -----------------------------
    Anticipe l'usure du gamma au fil du temps : un gamma qui s'effondre
    (ou explose) à l'approche de l'échéance change le budget de rehedge
    nécessaire, notamment près de la monnaie.
    """
    g = gamma(spot, strike, maturity, rate, dividend, vol)
    d1, _ = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    dd1 = _dd1_dt(spot, strike, maturity, rate, dividend, vol)
    return -g * (dividend + d1 * dd1 + 1.0 / (2.0 * maturity))


def veta(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
         dividend: ArrayLike, vol: ArrayLike) -> ArrayLike:
    """Veta : ``d(vega)/dT`` (identique call/put).

    Formule
    -------
    Soit ``Dd1 = (r-q)/(sigma*sqrt(T)) - d2/(2T)``.

    ``Veta = Vega * (-q - d1*Dd1 + 1/(2T))``

    Convention de signe : dérivée par rapport à la maturité résiduelle T
    (variation calendaire quotidienne du vega approximativement
    ``-veta / 365``).

    Usage en gestion des risques
    -----------------------------
    Anticipe comment l'exposition vega d'une position se dégrade avec le
    temps, indépendamment de tout mouvement de marché ; utile pour juger
    quand "rouler" une couverture de vol qui perd sa sensibilité.
    """
    v = vega(spot, strike, maturity, rate, dividend, vol)
    d1, _ = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    dd1 = _dd1_dt(spot, strike, maturity, rate, dividend, vol)
    return v * (-dividend - d1 * dd1 + 1.0 / (2.0 * maturity))


def all_greeks(spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike,
               dividend: ArrayLike, vol: ArrayLike, option_type: str = "call") -> dict[str, ArrayLike]:
    """Calcule prix + jeu complet de Grecques en un seul appel.

    Returns
    -------
    dict[str, ArrayLike]
        Cles : ``price, delta, gamma, vega, theta, rho, vanna, volga,
        charm, speed, zomma, color, veta``.
    """
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

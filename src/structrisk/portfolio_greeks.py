"""Agregation des Grecques d'un portefeuille d'options et de produits structures.

Le portefeuille est represente comme une liste de :class:`Position`,
chacune portant ses Grecques "par unité" (déjà calculées en amont par
``blackscholes.py``, ``binomial.py`` ou ``pricing_structured.py``, quelle
que soit la methode de pricing sous-jacente) ainsi qu'une quantité et un
multiplicateur de contrat. L'agregation elle-même est une simple somme
ponderee -- les Grecques de premier ordre (delta, gamma, vega, theta,
rho) sont additives par construction (linearite de la derivation), ce qui
permet de netter des positions sur un même sous-jacent indépendamment de
l'instrument (option vanille, autocall, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Position:
    """Une ligne de portefeuille (option vanille ou produit structure).

    Attributes
    ----------
    name : str
        Identifiant de la position (ex : "Call SX5E Dec26 K=4500").
    underlying : str
        Nom du sous-jacent de référence, utilise pour le netting par
        sous-jacent (ex : "SX5E").
    quantity : float
        Nombre de contrats/parts détenus (négatif si position courte).
    multiplier : float
        Taille de contrat / nominal unitaire (ex : 1.0 pour une action,
        1000.0 pour un certificat de nominal 1000).
    spot : float
        Niveau de spot courant du sous-jacent, utilise pour convertir les
        Grecques "par unité de spot" en exposition en euros.
    greeks : dict[str, float]
        Grecques par unité (une part, un contrat) : clés typiques
        ``delta, gamma, vega, theta, rho`` (et ``vanna`` si disponible,
        utilisée pour la grille de stress spot x vol).
    price : float
        Prix (valorisation) par unité de la position, utilise pour le
        calcul de P&L.
    reval_fn : Callable[[float, float], float], optional
        Fonction de revalorisation complète ``(spot_shock_rel, vol_shock_abs)
        -> nouveau_prix_par_unite``, utilisée par :func:`stress_grid` pour
        un choc important (au-delà de l'approximation locale de Taylor).
        Si absente, la grille de stress utilise l'approximation delta
        gamma vega (+ vanna si fournie).
    """

    name: str
    underlying: str
    quantity: float
    multiplier: float
    spot: float
    greeks: dict[str, float]
    price: float
    reval_fn: Callable[[float, float], float] | None = field(default=None, compare=False)


def aggregate_greeks_by_underlying(positions: list[Position]) -> pd.DataFrame:
    """Agrege les Grecques nettes par sous-jacent, en unités "produit" (non converties en euros).

    Pour chaque sous-jacent, chaque Grecque est sommee sur les positions
    ``sum_i( quantity_i * multiplier_i * greek_i )`` -- additivite directe
    des Grecques de premier ordre.

    Returns
    -------
    pd.DataFrame
        Indexe par sous-jacent, colonnes = Grecques présentées dans au
        moins une position (``delta, gamma, vega, theta, rho, ...``),
        valeurs manquantes traitees comme 0.
    """
    rows: dict[str, dict[str, float]] = {}
    for pos in positions:
        bucket = rows.setdefault(pos.underlying, {})
        weight = pos.quantity * pos.multiplier
        for greek_name, greek_value in pos.greeks.items():
            bucket[greek_name] = bucket.get(greek_name, 0.0) + weight * greek_value
    frame = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
    return frame.sort_index()


def delta_equivalent_exposure(positions: list[Position]) -> pd.Series:
    """Exposition delta-équivalente en euros, par sous-jacent.

    Formule
    -------
    ``DeltaEUR = sum_i( quantity_i * multiplier_i * delta_i * spot_i )``

    Usage en gestion des risques
    -----------------------------
    Montant (en euros) de sous-jacent qu'il faudrait acheter/vendre pour
    neutraliser au premier ordre l'exposition directionnelle du
    portefeuille sur ce sous-jacent -- la mesure de risque directionnel
    la plus lue en salle de marché.
    """
    totals: dict[str, float] = {}
    for pos in positions:
        totals[pos.underlying] = totals.get(pos.underlying, 0.0) + (
            pos.quantity * pos.multiplier * pos.greeks.get("delta", 0.0) * pos.spot
        )
    return pd.Series(totals, name="delta_eur").sort_index()


def gamma_eur_per_1pct_move(positions: list[Position]) -> pd.Series:
    """Gamma exprime en P&L euros pour un mouvement de spot de +1 %, par sous-jacent.

    Formule
    -------
    ``GammaEUR_1% = sum_i( 0.5 * quantity_i * multiplier_i * gamma_i * (0.01 * spot_i)^2 )``

    C'est le terme de convexité du developpement de Taylor du P&L
    (``0.5 * Gamma * dS^2``) evalue pour un choc de spot de 1 %.

    Usage en gestion des risques
    -----------------------------
    Estime le P&L de convexité (gain pour une position longue gamma, perte
    pour une position courte) génère par un mouvement de marché "typique"
    d'1 %, complementaire du delta pour juger le risque non-lineaire d'un
    book d'options/structures.
    """
    totals: dict[str, float] = {}
    for pos in positions:
        shock = 0.01 * pos.spot
        contribution = 0.5 * pos.quantity * pos.multiplier * pos.greeks.get("gamma", 0.0) * shock ** 2
        totals[pos.underlying] = totals.get(pos.underlying, 0.0) + contribution
    return pd.Series(totals, name="gamma_eur_1pct").sort_index()


def vega_eur_per_vol_point(positions: list[Position]) -> pd.Series:
    """Vega exprime en P&L euros pour un point de volatilité implicite (+1 pt = +0.01), par sous-jacent.

    Formule
    -------
    ``VegaEUR_1pt = sum_i( quantity_i * multiplier_i * vega_i * 0.01 )``

    Usage en gestion des risques
    -----------------------------
    Exposition au risque de niveau de vol implicite, la mesure standard
    de risque de "vega book" en salle de marché (les vega des options
    sont exprimes par convention en prix par 1 point de vol, i.e. par
    variation de sigma de 0.01).
    """
    totals: dict[str, float] = {}
    for pos in positions:
        totals[pos.underlying] = totals.get(pos.underlying, 0.0) + (
            pos.quantity * pos.multiplier * pos.greeks.get("vega", 0.0) * 0.01
        )
    return pd.Series(totals, name="vega_eur_1pt").sort_index()


def stress_grid(positions: list[Position], spot_shocks: np.ndarray, vol_shocks: np.ndarray) -> pd.DataFrame:
    """Grille de P&L du portefeuille (en euros) croisant chocs de spot et de vol.

    Pour chaque position, le P&L au point de grille ``(ds, dv)`` (``ds``
    relatif, ``dv`` absolu en vol) est estime par :

    - :attr:`Position.reval_fn` s'il est fourni (revalorisation complète,
      valable pour des chocs de grande amplitude) ;
    - sinon par un developpement de Taylor local à l'ordre 2 avec terme
      croise :

      ``dV = delta*dS + 0.5*gamma*dS^2 + vega*dvol + vanna*dS*dvol``

      avec ``dS = ds * spot`` (variation absolue de spot). Le terme
      ``vanna`` n'est utilise que si present dans ``position.greeks``
      (sinon suppose nul).

    Les contributions de toutes les positions sont sommees pour obtenir
    le P&L total du portefeuille en chaque point de la grille.

    Parameters
    ----------
    spot_shocks : np.ndarray
        Chocs de spot relatifs (ex : ``np.linspace(-0.20, 0.20, 9)``).
    vol_shocks : np.ndarray
        Chocs de volatilité absolus (ex : ``np.linspace(-0.10, 0.10, 9)``).

    Returns
    -------
    pd.DataFrame
        Indexe par choc de spot (en %), colonnes = choc de vol (en pts),
        valeurs = P&L total du portefeuille en euros.
    """
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
    """Decomposition du P&L realise d'une position par les Grecques.

    Attributes
    ----------
    delta_pnl, gamma_pnl, vega_pnl, theta_pnl : float
        Contributions estimees par chaque Grecque de premier/second ordre.
    explained_pnl : float
        Somme des quatre contributions.
    residual : float
        P&L reellement observe moins ``explained_pnl`` -- capture les
        termes d'ordre supérieur (vanna, volga, ...), les chocs de taux
        non modelises ici, et toute non-linearite non capturee par un
        developpement de Taylor local à l'ordre 2.
    """

    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    explained_pnl: float
    residual: float


def pnl_attribution(position: Position, d_spot: float, d_vol: float, dt: float,
                     realized_pnl: float) -> PnLAttribution:
    """Attribution du P&L d'une position entre delta, gamma, vega, theta et residu.

    Formule
    -------
    ``P&L_explique = delta*dS + 0.5*gamma*dS^2 + vega*dvol + theta*dt``
    ``Residu = P&L_realise - P&L_explique``

    Parameters
    ----------
    position : Position
        Position au début de la période (Grecques évaluées en t0).
    d_spot : float
        Variation absolue du spot sur la période.
    d_vol : float
        Variation absolue de la vol implicite sur la période.
    dt : float
        Duree écoulée en années (theta exprime "par an", cf. ``blackscholes.py``).
    realized_pnl : float
        P&L reellement observe sur la position (marque a marché, par
        unité -- a multiplier par quantité*multiplicateur en amont si besoin).

    Returns
    -------
    PnLAttribution
    """
    delta_pnl = position.greeks.get("delta", 0.0) * d_spot
    gamma_pnl = 0.5 * position.greeks.get("gamma", 0.0) * d_spot ** 2
    vega_pnl = position.greeks.get("vega", 0.0) * d_vol
    theta_pnl = position.greeks.get("theta", 0.0) * dt
    explained = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
    residual = realized_pnl - explained
    return PnLAttribution(delta_pnl=delta_pnl, gamma_pnl=gamma_pnl, vega_pnl=vega_pnl,
                           theta_pnl=theta_pnl, explained_pnl=explained, residual=residual)

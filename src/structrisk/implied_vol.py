"""Volatilite implicite : inversion Newton-Raphson / Brent et surface de vol.

Le module fournit :

1. ``implied_vol`` : inversion du prix Black-Scholes-Merton pour retrouver
   la volatilité implicite, par Newton-Raphson (rapide, utilise le vega
   analytique) avec repli automatique sur la methode de Brent
   (``scipy.optimize.brentq``, robuste mais plus lente) quand Newton ne
   converge pas ou sort du domaine admissible -- ce qui arrive typiquement
   pour les options très in/out-of-the-money dont le vega est proche de 0.

2. ``SmileParams`` / ``parametric_vol`` : parametrisation d'un smile de vol
   en "sourire" quadratique en log-moneyness avec une structure par terme,
   couramment utilisée pour générer une surface de vol synthetique
   realiste sans donnees de marché.

3. ``VolSurfaceGrid`` : grille (strikes x maturités) construite a partir de
   la parametrisation, avec interpolation bilineaire pour retrouver la vol
   à un point (K, T) quelconque à l'intérieur de la grille.
"""

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
    """Volatilite implicite par Newton-Raphson avec repli sur Brent.

    Algorithme
    ----------
    1. Vérification des bornes de non-arbitrage : si ``price`` est en
       dessous de la valeur intrinsèque actualisee (a epsilon numérique
       près) ou au-dessus du prix du sous-jacent, on leve une
       ``ValueError`` (prix incompatible avec toute volatilité positive).
    2. Newton-Raphson sur ``f(sigma) = BS(sigma) - price`` avec
       ``f'(sigma) = vega(sigma)``. Arret des que ``|f(sigma)| < tol``.
    3. Si Newton diverge (vega quasi nul -- option très loin de la monnaie
       ou très courte maturité --, iteration hors bornes, ou non
       convergence en ``max_iter`` iterations), repli sur
       ``scipy.optimize.brentq`` borne par ``bounds``, qui converge
       toujours pourvu que le prix soit bien dans la fourchette
       [intrinsèque, prix a vol infinie[.

    Parameters
    ----------
    price : float
        Prix de marché (ou théorique) observe de l'option.
    spot, strike, maturity, rate, dividend : float
        Parametres de marché et de contrat.
    option_type : str
        ``"call"`` ou ``"put"``.
    initial_guess : float
        Point de depart de Newton-Raphson.
    tol : float
        Tolerance sur l'ecart de prix pour la convergence.
    max_iter : int
        Nombre maximal d'iterations de Newton avant repli sur Brent.
    bounds : tuple[float, float]
        Bornes de recherche pour Brent.

    Returns
    -------
    float
        Volatilite implicite annualisee.

    Raises
    ------
    ValueError
        Si le prix viole les bornes de non-arbitrage (borne inférieure =
        valeur intrinsèque actualisee, borne supérieure = spot actualise
        des dividendes pour un call / strike actualise pour un put).
    """
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

    # Cas limite : prix quasi-intrinsèque -> vega quasi nul, Newton est instable.
    # On part directement en Brent avec une borne inférieure très petite.
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
                break  # vega trop faible : Newton n'est pas fiable, on bascule sur Brent
            step = diff / v
            sigma_next = sigma - step
            if sigma_next <= bounds[0] or sigma_next >= bounds[1] or not np.isfinite(sigma_next):
                break  # sortie de domaine : on bascule sur Brent
            sigma = sigma_next
        else:
            sigma = None  # non convergence en max_iter : repli
        if sigma is not None and abs(objective(sigma)) < tol:
            return float(sigma)

    # Repli robuste : Brent, garanti convergent car objective(bounds[0]) <= 0 <= objective(bounds[1])
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
    """Parametres d'une surface de volatilité synthetique.

    La vol au point (log-moneyness ``k = ln(K/F)``, maturité ``T``) est
    modelisee par un smile quadratique en ``k`` dont les trois
    coefficients (niveau ATM, skew, courbure) evoluent avec la maturité
    (structure par terme) :

    ``sigma(k, T) = atm(T) + skew(T) * k + curvature(T) * k^2``

    avec

    ``atm(T)       = atm_level + atm_term_slope * sqrt(T)``
    ``skew(T)      = skew_level * exp(-skew_decay * T)``
    ``curvature(T) = curvature_level * exp(-curvature_decay * T)``

    Le skew et la courbure décroissent avec la maturité (smile qui
    s'aplatit sur les échéances longues), comportement usuel des marchés
    actions.
    """

    atm_level: float = 0.20
    atm_term_slope: float = 0.02
    skew_level: float = -0.10
    skew_decay: float = 0.50
    curvature_level: float = 0.15
    curvature_decay: float = 0.50


def parametric_vol(strike: np.ndarray | float, maturity: np.ndarray | float, spot: float,
                    rate: float, dividend: float, params: SmileParams) -> np.ndarray | float:
    """Vol implicite parametree en un point (K, T) ou sur des tableaux broadcastables.

    Formule
    -------
    ``F = S * e^{(r-q)T}`` (forward), ``k = ln(K/F)`` (log-moneyness),
    puis ``sigma(k, T)`` selon :class:`SmileParams`.

    Returns
    -------
    ArrayLike
        Volatilite(s) implicite(s), plancher a 1% pour eviter des valeurs
        non economiques aux bornes extremes.
    """
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
    """Grille de volatilité implicite avec interpolation bilineaire.

    La grille est evaluee une fois sur un produit cartesien
    ``strikes x maturities`` a partir de :func:`parametric_vol`, puis toute
    requête à un point (K, T) intermédiaire est obtenue par interpolation
    bilineaire classique sur les 4 noeuds encadrants (les points hors
    grille sont clampes sur le bord le plus proche).
    """

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
        """Interpolation bilineaire de la vol au point (strike, maturity).

        Formule (bilineaire standard)
        ------------------------------
        Pour un point (x, y) encadre par ``x0 <= x <= x1`` et
        ``y0 <= y <= y1`` sur la grille, avec ``tx = (x-x0)/(x1-x0)`` et
        ``ty = (y-y0)/(y1-y0)`` :

        ``f(x,y) = (1-tx)(1-ty) f00 + tx(1-ty) f10 + (1-tx)ty f01 + tx*ty*f11``
        """
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

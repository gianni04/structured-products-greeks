"""Moteur Monte Carlo vectorisé : simulation GBM multi-actifs, réduction de
variance et Grecques.

Toute la simulation est vectorisée numpy : un seul appel produit un
tenseur de trajectoires ``(n_paths, n_steps + 1, n_assets)`` sans boucle
Python sur les chemins (seule la récursion temporelle est gérée par un
``cumsum`` vectorisé, pas par une boucle sur les trajectoires).

Modèle : mouvement brownien géométrique (GBM) multi-actifs sous la mesure
risque-neutre, corrélation instantanee constante entre les browniens :

    dS_i / S_i = (r - q_i) dt + sigma_i dW_i,   corr(dW_i, dW_j) = rho_ij

Discrétisation exacte en log (pas de biais de schéma, contrairement à
Euler) :

    S_i(t+dt) = S_i(t) * exp[(r - q_i - 0.5*sigma_i^2)*dt + sigma_i*sqrt(dt)*Z_i]

avec ``Z = eps @ L^T`` ou ``L`` est le facteur de Cholesky de la matrice de
corrélation et ``eps`` des normales centrees réduites indépendantes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class MCResult:
    """Resultat d'une estimation Monte Carlo avec intervalle de confiance.

    Attributes
    ----------
    price : float
        Estimateur du prix (moyenne des payoffs actualises).
    stderr : float
        Erreur-type de l'estimateur (``std / sqrt(n_paths)``).
    ci_low, ci_high : float
        Bornes de l'intervalle de confiance a ``confidence`` (par defaut 95 %).
    n_paths : int
        Nombre de trajectoires effectivement utilisées.
    """

    price: float
    stderr: float
    ci_low: float
    ci_high: float
    n_paths: int


def confidence_interval(discounted_payoffs: np.ndarray, confidence: float = 0.95) -> MCResult:
    """Construit un intervalle de confiance gaussien sur la moyenne d'un echantillon.

    Formule
    -------
    ``price = mean(X)``, ``stderr = std(X, ddof=1) / sqrt(n)``,
    ``CI = price +/- z_{(1+confidence)/2} * stderr`` avec ``z`` le quantile
    de la loi normale standard (approximation valable par le TCL pour
    ``n`` grand, standard en pricing Monte Carlo).
    """
    n = discounted_payoffs.shape[0]
    price = float(np.mean(discounted_payoffs))
    stderr = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(n))
    z = float(norm.ppf(0.5 + confidence / 2.0))
    return MCResult(price=price, stderr=stderr, ci_low=price - z * stderr,
                     ci_high=price + z * stderr, n_paths=n)


def simulate_gbm_paths(spot: np.ndarray, rate: float, dividend: np.ndarray, vol: np.ndarray,
                        corr_matrix: np.ndarray, maturity: float, n_steps: int, n_paths: int,
                        seed: int | None = None, antithetic: bool = True) -> np.ndarray:
    """Simule des trajectoires GBM multi-actifs corrélées, entièrement vectorisé.

    Parameters
    ----------
    spot : np.ndarray, shape (n_assets,)
        Spots initiaux.
    rate : float
        Taux sans risque continu (commun a tous les actifs).
    dividend : np.ndarray, shape (n_assets,)
        Taux de dividende continu par actif.
    vol : np.ndarray, shape (n_assets,)
        Volatilite annualisee par actif.
    corr_matrix : np.ndarray, shape (n_assets, n_assets)
        Matrice de corrélation (definie positive), decomposee par
        Cholesky pour correler les chocs.
    maturity : float
        Horizon de simulation en années.
    n_steps : int
        Nombre de pas de temps (observations intermédiaires incluses,
        utile pour les produits à barrière/autocall).
    n_paths : int
        Nombre de trajectoires simulees.
    seed : int, optional
        Graine du générateur numpy (reproductibilite, et clé pour les
        Grecques par nombres aléatoires communs -- même seed => mêmes
        chocs pour deux jeux de parametres differents).
    antithetic : bool
        Si True, génère ``n_paths`` par paires antithetiques
        ``(eps, -eps)`` : réduit la variance de l'estimateur sans biais
        car la loi normale est symetrique.

    Returns
    -------
    np.ndarray, shape (n_paths, n_steps + 1, n_assets)
        Trajectoires simulees, ``paths[:, 0, :] == spot``.
    """
    spot = np.atleast_1d(np.asarray(spot, dtype=float))
    dividend = np.broadcast_to(np.atleast_1d(np.asarray(dividend, dtype=float)), spot.shape)
    vol = np.broadcast_to(np.atleast_1d(np.asarray(vol, dtype=float)), spot.shape)
    corr_matrix = np.asarray(corr_matrix, dtype=float)
    n_assets = spot.shape[0]

    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(corr_matrix)

    if antithetic:
        n_half = -(-n_paths // 2)  # ceil division
        eps_half = rng.standard_normal(size=(n_half, n_steps, n_assets))
        eps = np.concatenate([eps_half, -eps_half], axis=0)[:n_paths]
    else:
        eps = rng.standard_normal(size=(n_paths, n_steps, n_assets))

    correlated = eps @ chol.T  # (n_paths, n_steps, n_assets)

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
    """Simule un GBM multi-actifs sur une grille de dates non nécessairement équidistantes.

    Utile pour les produits structures dont les dates d'observation ne
    sont pas equireparties (autocalls semestriels/annuels sur maturité
    non entière, etc.). Chaque pas ``dt_j = t_j - t_{j-1}`` a sa propre
    dérive/diffusion ; la boucle porte uniquement sur le nombre de dates
    d'observation (petit), toutes les operations internes restant
    vectorisées sur les trajectoires.

    Parameters
    ----------
    observation_times : np.ndarray
        Dates d'observation strictement croissantes, en années (``t_0``
        implicite = 0, non inclus dans le tableau).

    Returns
    -------
    np.ndarray, shape (n_paths, n_obs, n_assets)
        Valeurs du sous-jacent (panier) à chaque date d'observation.
    """
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

    correlated = eps @ chol.T  # (n_paths, n_obs, n_assets)

    dt = np.diff(np.concatenate([[0.0], observation_times]))  # (n_obs,)
    drift = (rate - dividend[None, :] - 0.5 * vol[None, :] ** 2) * dt[:, None]  # (n_obs, n_assets)
    diffusion = vol[None, :] * np.sqrt(dt)[:, None]  # (n_obs, n_assets)
    log_increments = drift[None, :, :] + diffusion[None, :, :] * correlated
    cum_log = np.cumsum(log_increments, axis=1)

    return spot * np.exp(cum_log)


def control_variate_adjustment(payoffs: np.ndarray, control: np.ndarray,
                                control_mean_analytic: float) -> np.ndarray:
    """Ajuste des payoffs simules par variable de contrôle a esperance connue.

    Formule
    -------
    ``Y_cv = Y - beta * (C - E[C])``, avec ``beta = Cov(Y, C) / Var(C)``
    estime sur l'echantillon (choix optimal minimisant ``Var(Y_cv)``).
    ``E[Y_cv] = E[Y]`` (estimateur non biaise) mais ``Var(Y_cv) < Var(Y)``
    des lors que ``Y`` et ``C`` sont corrélés.

    Parameters
    ----------
    payoffs : np.ndarray
        Echantillon de la quantité d'interet (ex : payoff actualise).
    control : np.ndarray
        Echantillon de la variable de contrôle (même trajectoires).
    control_mean_analytic : float
        Esperance exacte, connue en forme fermee, de ``control``.

    Returns
    -------
    np.ndarray
        Echantillon ajuste, a utiliser ensuite avec :func:`confidence_interval`.
    """
    cov = float(np.cov(payoffs, control, ddof=1)[0, 1])
    var_c = float(np.var(control, ddof=1))
    beta = 0.0 if var_c < 1e-14 else cov / var_c
    return payoffs - beta * (control - control_mean_analytic)


def mc_vanilla_call(spot: float, strike: float, maturity: float, rate: float, dividend: float,
                     vol: float, n_paths: int, seed: int | None = None, antithetic: bool = True,
                     use_control_variate: bool = True) -> MCResult:
    """Prix Monte Carlo d'un call vanille européen, avec réduction de variance.

    Combine variables antithetiques (dans :func:`simulate_gbm_paths`) et
    variable de contrôle ``S_T`` (esperance connue :
    ``E[S_T] = S_0 * exp((r-q)*T)`` sous la mesure risque-neutre).
    """
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
    """Delta par différence centrée bump-and-revalue, nombres aléatoires communs.

    Le principe des nombres aléatoires communs (CRN) est de reutiliser
    exactement les mêmes tirages ``eps`` pour le prix de base et le prix
    bumpe : ``pricer`` doit être une fonction fermee sur un ``seed`` fixe
    (donc sur les mêmes chocs) qui ne varie qu'avec le spot passe en
    argument. Cela annule une grande partie de la variance de
    l'estimateur de différence (les deux prix sont fortement corrélés),
    contrairement à deux simulations indépendantes.

    Parameters
    ----------
    pricer : Callable[[float], float]
        Fonction qui, à un spot donné, renvoie un prix Monte Carlo simule
        avec un seed fixe (donc les mêmes chocs aléatoires à chaque appel).
    spot : float
        Spot central.
    bump : float
        Taille du bump relatif appliqué de part et d'autre.

    Returns
    -------
    float
        Estimateur du delta ``(V(S+h) - V(S-h)) / (2h)``.
    """
    h = bump * spot
    return (pricer(spot + h) - pricer(spot - h)) / (2.0 * h)


def bump_gamma_crn(pricer: "callable[[float], float]", spot: float, bump: float = 0.01) -> float:
    """Gamma par différence centrée a 3 points, nombres aléatoires communs (voir :func:`bump_delta_crn`)."""
    h = bump * spot
    return (pricer(spot + h) - 2.0 * pricer(spot) + pricer(spot - h)) / (h ** 2)


def pathwise_delta_call(spot: float, strike: float, maturity: float, rate: float, dividend: float,
                         vol: float, n_paths: int, seed: int | None = None,
                         antithetic: bool = True) -> float:
    """Delta d'un call vanille par la methode pathwise (derivation trajectoire par trajectoire).

    Formule
    -------
    Pour un GBM, ``dS_T/dS_0 = S_T/S_0`` (proportionnalite des trajectoires
    en spot initial). Le payoff actualise ``e^{-rT} max(S_T-K, 0)`` est
    presque partout derivable en ``S_0`` (indicatrice de discontinuite de
    mesure nulle), d'ou l'estimateur sans biais :

    ``Delta = e^{-rT} * E[ 1_{S_T > K} * S_T / S_0 ]``
    """
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
    """Delta d'un call vanille par la methode du score / likelihood ratio.

    Formule
    -------
    ``S_T = S_0 * exp[(r-q-0.5*sigma^2)T + sigma*sqrt(T)*Z]``, ``Z ~ N(0,1)``.
    En derivant le logarithme de la densite de ``Z`` par rapport a
    ``S_0`` (``S_0`` intervient comme un decalage de la loi de ``log S_T``),
    on obtient le score ``Z / (S_0 * sigma * sqrt(T))``, d'ou l'estimateur
    sans biais (valable même si le payoff est discontinu) :

    ``Delta = e^{-rT} * E[ payoff(S_T) * Z / (S_0 * sigma * sqrt(T)) ]``
    """
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

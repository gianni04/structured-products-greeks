"""Simulation de couverture en delta (et delta-gamma) discrète.

On simule la replication dynamique d'une option vanille vendue : le
vendeur encaisse la prime, achète/vend le sous-jacent (et eventuellement
une option de couverture additionnelle) a intervalles reguliers pour
maintenir un portefeuille delta-neutre (ou delta-gamma-neutre), en tenant
compte de coûts de transaction proportionnels. L'erreur de replication a
maturité -- l'ecart entre la valeur du portefeuille de couverture et le
payoff reellement du -- est nulle en moyenne seulement à la limite d'un
rehedge continu et sans frais ; elle est etudiee ici en fonction de la
fréquence de rehedge et des coûts de transaction.

Toute la simulation est vectorisée sur les trajectoires ; la seule boucle
explicite porte sur le nombre de dates de rehedge (quelques dizaines a
quelques centaines), jamais sur les trajectoires elles-mêmes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .blackscholes import bs_price, delta as bs_delta, gamma as bs_gamma
from .montecarlo import simulate_gbm_paths


@dataclass(frozen=True)
class HedgeSimulationResult:
    """Resultat d'une simulation de couverture dynamique.

    Attributes
    ----------
    hedging_error : np.ndarray, shape (n_paths,)
        Erreur de replication à maturité, par trajectoire :
        ``valeur du portefeuille de couverture - payoff dû à l'acheteur``.
    mean_error : float
        Moyenne de l'erreur de replication (biais, essentiellement dû aux
        coûts de transaction si non nul).
    std_error : float
        Ecart-type de l'erreur de replication (dispersion résiduelle du
        risque de gamma non couvert entre deux rehedges).
    """

    hedging_error: np.ndarray
    mean_error: float
    std_error: float


def simulate_delta_hedge_pnl(spot0: float, strike: float, maturity: float, rate: float,
                              dividend: float, vol: float, n_paths: int, n_rehedge: int,
                              seed: int | None = None, transaction_cost_bps: float = 0.0,
                              option_type: str = "call",
                              realized_vol: float | None = None) -> HedgeSimulationResult:
    """Simule la replication delta-neutre discrète d'un call/put vendu.

    Algorithme
    ----------
    A ``t=0``, le vendeur encaisse la prime Black-Scholes (calculée avec
    ``vol``, la vol implicite au moment de la vente) et achète
    ``delta_0`` unités de sous-jacent. A chaque date de rehedge
    ``t_i = i * T / n_rehedge`` (``i = 0, ..., n_rehedge - 1``), la
    position en sous-jacent est ajustee au nouveau delta Black-Scholes
    (calcule avec la même vol de pricing, recalculée à la maturité
    résiduelle ``T - t_i``), moyennant un coût de transaction
    proportionnel au notionnel echange. Le compte de cash capitalise au
    taux sans risque entre deux dates. A maturité, le portefeuille
    (cash + position en sous-jacent value au spot terminal) est compare
    au payoff réellement dû à l'acheteur de l'option.

    Parameters
    ----------
    spot0, strike, maturity, rate, dividend, vol : float
        Parametres de marché et de pricing de l'option vendue.
    n_paths : int
        Nombre de trajectoires simulees.
    n_rehedge : int
        Nombre de dates de rehedge (fréquence de couverture). Plus il est
        élevé, plus l'erreur de replication résiduelle (risque de gamma
        non couvert intra-période) diminue.
    transaction_cost_bps : float
        Cout de transaction, en points de base du notionnel echange a
        chaque rehedge (achat ET vente payent le coût).
    realized_vol : float, optional
        Si fourni et different de ``vol``, la trajectoire du sous-jacent
        est simulee avec cette volatilité reelle alors que le hedge est
        calcule avec ``vol`` (vol de pricing) -- illustre le risque de
        mauvaise estimation de la vol future en plus du seul risque de
        discretisation du rehedge. Par defaut, ``realized_vol = vol``.

    Returns
    -------
    HedgeSimulationResult
    """
    sim_vol = vol if realized_vol is None else realized_vol
    paths = simulate_gbm_paths(np.array([spot0]), rate, np.array([dividend]), np.array([sim_vol]),
                                np.array([[1.0]]), maturity, n_rehedge, n_paths, seed, antithetic=True)
    spot_path = paths[:, :, 0]  # (n_paths, n_rehedge + 1)

    dt = maturity / n_rehedge
    premium = float(bs_price(spot0, strike, maturity, rate, dividend, vol, option_type))
    cash = np.full(n_paths, premium)
    shares = np.zeros(n_paths)

    for i in range(n_rehedge):
        tau = maturity - i * dt
        s_i = spot_path[:, i]
        delta_i = bs_delta(s_i, strike, tau, rate, dividend, vol, option_type)
        trade = delta_i - shares
        cost = transaction_cost_bps * 1e-4 * np.abs(trade) * s_i
        cash = cash - trade * s_i - cost
        shares = delta_i
        # Le compte de cash capitalise au taux sans risque ; la position en
        # sous-jacent détenue entre deux rehedges encaisse le dividende
        # continu (sinon la stratégie de replication est structurellement
        # biaisée : elle "oublie" le rendement du dividende recu par le
        # détenteur physique du sous-jacent).
        cash = cash * np.exp(rate * dt) + shares * s_i * (np.exp(dividend * dt) - 1.0)

    s_final = spot_path[:, -1]
    if option_type == "call":
        payoff = np.maximum(s_final - strike, 0.0)
    else:
        payoff = np.maximum(strike - s_final, 0.0)

    portfolio_value = cash + shares * s_final
    hedging_error = portfolio_value - payoff
    return HedgeSimulationResult(hedging_error=hedging_error, mean_error=float(np.mean(hedging_error)),
                                  std_error=float(np.std(hedging_error, ddof=1)))


def simulate_delta_gamma_hedge_pnl(spot0: float, strike: float, maturity: float, rate: float,
                                    dividend: float, vol: float, hedge_option_strike: float,
                                    n_paths: int, n_rehedge: int, seed: int | None = None,
                                    transaction_cost_bps: float = 0.0,
                                    option_type: str = "call") -> HedgeSimulationResult:
    """Simule la replication delta-gamma-neutre discrète d'un call/put vendu.

    Meme principe que :func:`simulate_delta_hedge_pnl`, mais en utilisant
    à chaque rehedge une seconde option vanille (strike
    ``hedge_option_strike``, même maturité) en plus du sous-jacent, pour
    neutraliser simultanement le delta ET le gamma de la position :

    ``n_hedge = Gamma_cible / Gamma_hedge``
    ``n_shares = Delta_cible - n_hedge * Delta_hedge``

    (la position cible est courte, d'ou le signe : le gamma court de la
    position cible est compense par ``n_hedge`` unités de l'option de
    couverture, puis le delta residuel est neutralise par le sous-jacent
    lui-même, qui n'a pas de gamma).

    Returns
    -------
    HedgeSimulationResult
    """
    paths = simulate_gbm_paths(np.array([spot0]), rate, np.array([dividend]), np.array([vol]),
                                np.array([[1.0]]), maturity, n_rehedge, n_paths, seed, antithetic=True)
    spot_path = paths[:, :, 0]

    dt = maturity / n_rehedge
    premium = float(bs_price(spot0, strike, maturity, rate, dividend, vol, option_type))
    cash = np.full(n_paths, premium)
    shares = np.zeros(n_paths)
    n_hedge_opt = np.zeros(n_paths)

    for i in range(n_rehedge):
        tau = maturity - i * dt
        s_i = spot_path[:, i]
        delta_t = bs_delta(s_i, strike, tau, rate, dividend, vol, option_type)
        gamma_t = bs_gamma(s_i, strike, tau, rate, dividend, vol)
        delta_h = bs_delta(s_i, hedge_option_strike, tau, rate, dividend, vol, option_type)
        gamma_h = bs_gamma(s_i, hedge_option_strike, tau, rate, dividend, vol)
        price_h = bs_price(s_i, hedge_option_strike, tau, rate, dividend, vol, option_type)

        new_n_hedge = gamma_t / gamma_h
        new_shares = delta_t - new_n_hedge * delta_h

        trade_shares = new_shares - shares
        trade_hedge = new_n_hedge - n_hedge_opt
        cost = transaction_cost_bps * 1e-4 * (np.abs(trade_shares) * s_i + np.abs(trade_hedge) * price_h)
        cash = cash - trade_shares * s_i - trade_hedge * price_h - cost
        shares = new_shares
        n_hedge_opt = new_n_hedge
        # Cf. simulate_delta_hedge_pnl : la position en sous-jacent (mais
        # pas la position en option de couverture, qui ne versé pas de
        # dividende) encaisse le dividende continu entre deux rehedges.
        cash = cash * np.exp(rate * dt) + shares * s_i * (np.exp(dividend * dt) - 1.0)

    s_final = spot_path[:, -1]
    if option_type == "call":
        payoff = np.maximum(s_final - strike, 0.0)
        payoff_hedge = np.maximum(s_final - hedge_option_strike, 0.0)
    else:
        payoff = np.maximum(strike - s_final, 0.0)
        payoff_hedge = np.maximum(hedge_option_strike - s_final, 0.0)

    portfolio_value = cash + shares * s_final + n_hedge_opt * payoff_hedge
    hedging_error = portfolio_value - payoff
    return HedgeSimulationResult(hedging_error=hedging_error, mean_error=float(np.mean(hedging_error)),
                                  std_error=float(np.std(hedging_error, ddof=1)))


def rehedge_frequency_sweep(spot0: float, strike: float, maturity: float, rate: float, dividend: float,
                             vol: float, n_paths: int, rehedge_grid: np.ndarray, seed: int | None = None,
                             transaction_cost_bps: float = 0.0, option_type: str = "call") -> dict[str, np.ndarray]:
    """Balaie la fréquence de rehedge et rapporte moyenne/ecart-type de l'erreur de replication.

    Illustration du compromis classique : plus le rehedge est fréquent,
    plus l'erreur de discretisation (risque de gamma) diminue (en
    ``O(1/sqrt(n_rehedge))`` typiquement, sans frais), mais plus les
    coûts de transaction cumules augmentent -- d'ou un optimum de
    fréquence en presence de coûts.

    Returns
    -------
    dict[str, np.ndarray]
        Cles : ``n_rehedge, mean_error, std_error``.
    """
    means = np.empty(len(rehedge_grid))
    stds = np.empty(len(rehedge_grid))
    for i, n_r in enumerate(rehedge_grid):
        res = simulate_delta_hedge_pnl(spot0, strike, maturity, rate, dividend, vol, n_paths, int(n_r),
                                        seed, transaction_cost_bps, option_type)
        means[i] = res.mean_error
        stds[i] = res.std_error
    return {"n_rehedge": np.asarray(rehedge_grid), "mean_error": means, "std_error": stds}

"""Agrégation des Grecques d'un portefeuille mixte options vanille + produits structurés, et heatmap de la grille de stress spot x vol."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np

from structrisk.blackscholes import all_greeks, bs_price
from structrisk.montecarlo import simulate_gbm_paths
from structrisk.payoffs import BonusCapSpec, CapitalProtectedNoteSpec, bonus_cap_payoff
from structrisk.portfolio_greeks import (
    Position,
    aggregate_greeks_by_underlying,
    delta_equivalent_exposure,
    gamma_eur_per_1pct_move,
    stress_grid,
    vega_eur_per_vol_point,
)
from structrisk.pricing_structured import price_capital_protected_note_decomposition

SPOT = 100.0
RATE = 0.03
DIVIDEND = 0.02
VOL = 0.22
MATURITY = 1.0
SEED = 11

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)

call_strike = 105.0
call_greeks = all_greeks(SPOT, call_strike, MATURITY, RATE, DIVIDEND, VOL, "call")
pos_call = Position(
    name=f"Call K={call_strike:.0f}", underlying="ACTION_X", quantity=1000.0, multiplier=1.0,
    spot=SPOT, greeks=call_greeks, price=call_greeks["price"],
)

put_strike = 95.0
put_greeks = all_greeks(SPOT, put_strike, MATURITY, RATE, DIVIDEND, VOL, "put")
pos_put = Position(
    name=f"Put K={put_strike:.0f} (vendu)", underlying="ACTION_X", quantity=-600.0, multiplier=1.0,
    spot=SPOT, greeks=put_greeks, price=put_greeks["price"],
)

note_spec = CapitalProtectedNoteSpec(
    notional=1000.0, protection_level=1.00, participation_rate=0.80, strike_level=1.00, cap_level=1.30,
)
note_price = price_capital_protected_note_decomposition(SPOT, VOL, RATE, DIVIDEND, MATURITY, note_spec)
h = 0.01 * SPOT
note_price_up = price_capital_protected_note_decomposition(SPOT + h, VOL, RATE, DIVIDEND, MATURITY, note_spec)
note_price_dn = price_capital_protected_note_decomposition(SPOT - h, VOL, RATE, DIVIDEND, MATURITY, note_spec)
note_delta_unit = (note_price_up - note_price_dn) / (2.0 * h)
note_gamma_unit = (note_price_up - 2.0 * note_price + note_price_dn) / (h ** 2)
h_vol = 0.01
note_vega_unit = (
    price_capital_protected_note_decomposition(SPOT, VOL + h_vol, RATE, DIVIDEND, MATURITY, note_spec)
    - price_capital_protected_note_decomposition(SPOT, VOL - h_vol, RATE, DIVIDEND, MATURITY, note_spec)
) / (2.0 * h_vol)
pos_note = Position(
    name="Note capital protégé 80% cap 130%", underlying="ACTION_X", quantity=50.0, multiplier=1.0,
    spot=SPOT, greeks={"delta": note_delta_unit, "gamma": note_gamma_unit, "vega": note_vega_unit},
    price=note_price,
)

bonus_spec = BonusCapSpec(notional=1000.0, barrier=0.70, bonus_level=1.05, cap_level=1.25)
n_paths_bonus = 100_000
n_steps_bonus = 52


def _price_bonus_cap(spot: float) -> float:
    paths = simulate_gbm_paths(np.array([spot]), RATE, np.array([DIVIDEND]), np.array([VOL]),
                                np.array([[1.0]]), MATURITY, n_steps_bonus, n_paths_bonus, SEED, True)
    performance = paths[:, :, 0] / spot
    payoff = bonus_cap_payoff(performance, bonus_spec)
    return float(np.mean(payoff * np.exp(-RATE * MATURITY)))


bonus_price = _price_bonus_cap(SPOT)
bonus_price_up = _price_bonus_cap(SPOT + h)
bonus_price_dn = _price_bonus_cap(SPOT - h)
bonus_delta_unit = (bonus_price_up - bonus_price_dn) / (2.0 * h)
bonus_gamma_unit = (bonus_price_up - 2.0 * bonus_price + bonus_price_dn) / (h ** 2)
pos_bonus = Position(
    name="Bonus cappé 105%/125% barrière 70%", underlying="ACTION_X", quantity=30.0, multiplier=1.0,
    spot=SPOT, greeks={"delta": bonus_delta_unit, "gamma": bonus_gamma_unit, "vega": 0.0},
    price=bonus_price,
)

positions = [pos_call, pos_put, pos_note, pos_bonus]

frame = aggregate_greeks_by_underlying(positions)
delta_eur = delta_equivalent_exposure(positions)
gamma_eur = gamma_eur_per_1pct_move(positions)
vega_eur = vega_eur_per_vol_point(positions)

print("Grecques agrégées par sous-jacent (unités 'produit') :")
print(frame.to_string())
print(f"\nExposition delta-équivalente : {delta_eur['ACTION_X']:,.2f} EUR")
print(f"Gamma P&L pour +1% de spot : {gamma_eur['ACTION_X']:,.2f} EUR")
print(f"Vega P&L pour +1 pt de vol : {vega_eur['ACTION_X']:,.2f} EUR")

spot_shocks = np.linspace(-0.20, 0.20, 9)
vol_shocks = np.linspace(-0.10, 0.10, 9)
grid = stress_grid(positions, spot_shocks, vol_shocks)

fig, ax = plt.subplots(figsize=(8, 6), dpi=130)
im = ax.imshow(grid.values, cmap="RdYlGn", aspect="auto", origin="lower")
ax.set_xticks(range(len(grid.columns)))
ax.set_xticklabels([f"{v:+.0f}" for v in grid.columns])
ax.set_yticks(range(len(grid.index)))
ax.set_yticklabels([f"{v:+.0f}" for v in grid.index])
ax.set_xlabel("Choc de volatilité (pts)")
ax.set_ylabel("Choc de spot (%)")
ax.set_title("Grille de stress P&L portefeuille (EUR) : spot x vol")
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        ax.text(j, i, f"{grid.values[i, j]:,.0f}", ha="center", va="center", fontsize=6.5)
fig.colorbar(im, ax=ax, label="P&L (EUR)")
fig.tight_layout()

out_path = OUT_DIR / "04_portfolio_greeks_heatmap.png"
fig.savefig(out_path)
plt.close(fig)

print(f"\nFigure enregistrée : {out_path}")
worst_cell = grid.values.min()
best_cell = grid.values.max()
print(f"P&L de stress le plus défavorable : {worst_cell:,.0f} EUR ; le plus favorable : {best_cell:,.0f} EUR")

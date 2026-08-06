"""Erreur de réplication (delta-hedging discret) en fonction de la fréquence de rehedge et des coûts de transaction."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np

from structrisk.hedging import rehedge_frequency_sweep, simulate_delta_hedge_pnl

SPOT = 100.0
STRIKE = 100.0
MATURITY = 0.5
RATE = 0.03
DIVIDEND = 0.02
VOL = 0.25
N_PATHS = 40_000
SEED = 5

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rehedge_grid = np.array([1, 2, 4, 6, 12, 26, 52, 104, 252])

sweep_no_cost = rehedge_frequency_sweep(SPOT, STRIKE, MATURITY, RATE, DIVIDEND, VOL, N_PATHS,
                                         rehedge_grid, seed=SEED, transaction_cost_bps=0.0)
sweep_with_cost = rehedge_frequency_sweep(SPOT, STRIKE, MATURITY, RATE, DIVIDEND, VOL, N_PATHS,
                                           rehedge_grid, seed=SEED, transaction_cost_bps=10.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=130)

ax_std = axes[0]
ax_std.plot(rehedge_grid, sweep_no_cost["std_error"], marker="o", label="Sans coût de transaction")
ax_std.plot(rehedge_grid, sweep_with_cost["std_error"], marker="o", label="Coût 10 bps par rehedge")
ax_std.set_xscale("log")
ax_std.set_xlabel("Nombre de rehedges sur la vie de l'option")
ax_std.set_ylabel("Écart-type de l'erreur de réplication")
ax_std.set_title("Dispersion de l'erreur de réplication")
ax_std.legend(fontsize=8)

ax_mean = axes[1]
ax_mean.plot(rehedge_grid, sweep_no_cost["mean_error"], marker="o", label="Sans coût de transaction")
ax_mean.plot(rehedge_grid, sweep_with_cost["mean_error"], marker="o", label="Coût 10 bps par rehedge")
ax_mean.axhline(0.0, color="grey", linestyle=":", linewidth=1)
ax_mean.set_xscale("log")
ax_mean.set_xlabel("Nombre de rehedges sur la vie de l'option")
ax_mean.set_ylabel("Erreur de réplication moyenne (biais)")
ax_mean.set_title("Biais de l'erreur de réplication (coût cumulé)")
ax_mean.legend(fontsize=8)

fig.suptitle("Delta-hedging discret d'un call vendu : fréquence de rehedge vs coûts de transaction",
             fontsize=11)
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

out_path = OUT_DIR / "05_delta_hedging.png"
fig.savefig(out_path)
plt.close(fig)

print(f"Figure enregistrée : {out_path}")
idx_low, idx_high = 0, len(rehedge_grid) - 1
print(f"Écart-type sans coût : {sweep_no_cost['std_error'][idx_low]:.4f} à "
      f"{rehedge_grid[idx_low]} rehedge(s) -> {sweep_no_cost['std_error'][idx_high]:.4f} à "
      f"{rehedge_grid[idx_high]} rehedges")
print(f"Biais avec coût 10 bps : {sweep_with_cost['mean_error'][idx_low]:.4f} à "
      f"{rehedge_grid[idx_low]} rehedge(s) -> {sweep_with_cost['mean_error'][idx_high]:.4f} à "
      f"{rehedge_grid[idx_high]} rehedges")

single_result = simulate_delta_hedge_pnl(SPOT, STRIKE, MATURITY, RATE, DIVIDEND, VOL, N_PATHS, 52,
                                          seed=SEED, transaction_cost_bps=10.0)
print(f"Rehedge hebdomadaire (52x/an), coût 10 bps -- erreur moyenne={single_result.mean_error:.4f}, "
      f"écart-type={single_result.std_error:.4f}")

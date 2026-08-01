"""Exemple 3 : valorisation Monte Carlo d'un Autocall Phoenix, probabilité de
rappel par date d'observation, et sensibilité du prix à la barrière de rappel.

Produit type banque privée : 4 observations semestrielles sur 2 ans,
coupon conditionnel à effet mémoire, barrière de rappel à 100%, barrière de
protection à 60%.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np

from structrisk.payoffs import PhoenixAutocallSpec
from structrisk.pricing_structured import (
    phoenix_barrier_sensitivity,
    price_phoenix_autocall,
)

SPOT = 100.0
VOL = 0.25
RATE = 0.03
DIVIDEND = 0.02
CORR = np.array([[1.0]])
N_PATHS = 200_000
SEED = 2026

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)

spec = PhoenixAutocallSpec(
    notional=100.0,
    coupon_rate=0.04,
    autocall_barrier=1.00,
    coupon_barrier=0.70,
    protection_barrier=0.60,
    observation_times=np.array([0.5, 1.0, 1.5, 2.0]),
)

result = price_phoenix_autocall(SPOT, VOL, RATE, DIVIDEND, CORR, spec, N_PATHS, seed=SEED)

print(f"Prix Autocall Phoenix : {result.price:.4f} (IC95% [{result.ci_low:.4f}, {result.ci_high:.4f}], "
      f"stderr={result.stderr:.4f})")
print(f"Probabilité de survie sans rappel : {result.prob_never_called:.2%}")
print(f"Probabilité de perte en capital (si survie) : {result.prob_capital_loss:.2%}")
for t, p in zip(spec.observation_times[:-1], result.call_probability_by_date):
    print(f"  Probabilité de rappel à t={t:.2f} an(s) : {p:.2%}")

barrier_grid = np.linspace(0.90, 1.15, 21)
barrier_sensitivity = phoenix_barrier_sensitivity(
    SPOT, VOL, RATE, DIVIDEND, spec, "autocall_barrier", barrier_grid, N_PATHS, SEED
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=130)

ax_prob = axes[0]
obs_labels = [f"{t:.2f}" for t in spec.observation_times[:-1]]
bars = list(result.call_probability_by_date) + [result.prob_never_called]
bar_labels = obs_labels + ["Jamais rappelé"]
ax_prob.bar(bar_labels, np.array(bars) * 100.0, color="steelblue")
ax_prob.set_title("Probabilité de rappel par date d'observation")
ax_prob.set_xlabel("Date d'observation (années) / issue")
ax_prob.set_ylabel("Probabilité (%)")
ax_prob.tick_params(axis="x", rotation=20)

ax_barrier = axes[1]
ax_barrier.plot(barrier_grid * 100.0, barrier_sensitivity, color="darkorange")
ax_barrier.axvline(spec.autocall_barrier * 100.0, color="grey", linestyle=":", linewidth=1)
ax_barrier.set_title("Sensibilité du prix à la barrière de rappel")
ax_barrier.set_xlabel("Barrière de rappel (% du spot initial)")
ax_barrier.set_ylabel("Prix (% du nominal)")

ax_coupon = axes[2]
coupon_barrier_grid = np.linspace(0.50, 0.90, 21)
coupon_sensitivity = phoenix_barrier_sensitivity(
    SPOT, VOL, RATE, DIVIDEND, spec, "coupon_barrier", coupon_barrier_grid, N_PATHS, SEED
)
ax_coupon.plot(coupon_barrier_grid * 100.0, coupon_sensitivity, color="seagreen")
ax_coupon.axvline(spec.coupon_barrier * 100.0, color="grey", linestyle=":", linewidth=1)
ax_coupon.set_title("Sensibilité du prix à la barrière de coupon")
ax_coupon.set_xlabel("Barrière de coupon (% du spot initial)")
ax_coupon.set_ylabel("Prix (% du nominal)")

fig.suptitle("Autocall Phoenix : valorisation, probabilités de rappel et sensibilité aux barrières",
             fontsize=11)
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

out_path = OUT_DIR / "03_autocall_pricing.png"
fig.savefig(out_path)
plt.close(fig)

print(f"Figure enregistrée : {out_path}")
price_at_105 = barrier_sensitivity[np.argmin(np.abs(barrier_grid - 1.05))]
price_at_95 = barrier_sensitivity[np.argmin(np.abs(barrier_grid - 0.95))]
print(f"Prix si barrière de rappel à 105% = {price_at_105:.4f} vs à 95% = {price_at_95:.4f}")

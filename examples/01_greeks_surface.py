"""Grecques (delta, gamma, vega, theta) en fonction du spot et de la maturité pour un call vanille."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np

from structrisk.blackscholes import delta, gamma, theta, vega

STRIKE = 100.0
RATE = 0.03
DIVIDEND = 0.02
VOL = 0.22

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)

spots = np.linspace(60.0, 140.0, 161)
maturities = [0.08, 0.25, 1.0, 2.0]
labels = ["1 mois", "3 mois", "1 an", "2 ans"]

fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=130)
ax_delta, ax_gamma, ax_vega, ax_theta = axes.ravel()

for maturity, label in zip(maturities, labels):
    ax_delta.plot(spots, delta(spots, STRIKE, maturity, RATE, DIVIDEND, VOL, "call"), label=label)
    ax_gamma.plot(spots, gamma(spots, STRIKE, maturity, RATE, DIVIDEND, VOL), label=label)
    ax_vega.plot(spots, vega(spots, STRIKE, maturity, RATE, DIVIDEND, VOL), label=label)
    ax_theta.plot(spots, theta(spots, STRIKE, maturity, RATE, DIVIDEND, VOL, "call") / 365.0, label=label)

ax_delta.set_title("Delta")
ax_delta.set_xlabel("Spot")
ax_delta.set_ylabel("Delta")
ax_delta.axvline(STRIKE, color="grey", linestyle=":", linewidth=1)
ax_delta.legend(fontsize=8, title="Maturité")

ax_gamma.set_title("Gamma")
ax_gamma.set_xlabel("Spot")
ax_gamma.set_ylabel("Gamma")
ax_gamma.axvline(STRIKE, color="grey", linestyle=":", linewidth=1)

ax_vega.set_title("Vega")
ax_vega.set_xlabel("Spot")
ax_vega.set_ylabel("Vega (prix par unité de vol)")
ax_vega.axvline(STRIKE, color="grey", linestyle=":", linewidth=1)

ax_theta.set_title("Theta (par jour)")
ax_theta.set_xlabel("Spot")
ax_theta.set_ylabel("Theta journalier")
ax_theta.axvline(STRIKE, color="grey", linestyle=":", linewidth=1)

fig.suptitle(f"Grecques d'un call vanille (K={STRIKE:.0f}, r={RATE:.1%}, q={DIVIDEND:.1%}, "
             f"sigma={VOL:.0%}) en fonction du spot et de la maturité", fontsize=11)
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

out_path = OUT_DIR / "01_greeks_surface.png"
fig.savefig(out_path)
plt.close(fig)

atm_1y = {
    "delta": float(delta(STRIKE, STRIKE, 1.0, RATE, DIVIDEND, VOL, "call")),
    "gamma": float(gamma(STRIKE, STRIKE, 1.0, RATE, DIVIDEND, VOL)),
    "vega": float(vega(STRIKE, STRIKE, 1.0, RATE, DIVIDEND, VOL)),
    "theta_jour": float(theta(STRIKE, STRIKE, 1.0, RATE, DIVIDEND, VOL, "call") / 365.0),
}
print(f"Figure enregistrée : {out_path}")
print(f"Call ATM 1 an -- delta={atm_1y['delta']:.4f}, gamma={atm_1y['gamma']:.5f}, "
      f"vega={atm_1y['vega']:.3f}, theta/jour={atm_1y['theta_jour']:.4f}")

gamma_1m = float(gamma(STRIKE, STRIKE, 0.08, RATE, DIVIDEND, VOL))
gamma_2y = float(gamma(STRIKE, STRIKE, 2.0, RATE, DIVIDEND, VOL))
print(f"Gamma ATM 1 mois={gamma_1m:.5f} vs Gamma ATM 2 ans={gamma_2y:.5f} "
      f"(ratio={gamma_1m / gamma_2y:.2f}x)")

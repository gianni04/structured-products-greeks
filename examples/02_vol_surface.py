"""Exemple 2 : surface de volatilité implicite synthétique (smile + structure
par terme) et coupes du smile par maturité.

La surface est générée par la paramétrisation quadratique en log-moneyness
de ``implied_vol.SmileParams`` (pas de données de marché téléchargées --
projet 100% hors ligne), avec un skew négatif typique actions qui s'aplatit
sur les maturités longues.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np

from structrisk.implied_vol import SmileParams, VolSurfaceGrid, parametric_vol

SPOT = 100.0
RATE = 0.03
DIVIDEND = 0.02

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)

params = SmileParams()
strikes = np.linspace(60.0, 140.0, 41)
maturities = np.linspace(0.1, 3.0, 30)

kk, tt = np.meshgrid(strikes, maturities, indexing="ij")
vol_grid = np.asarray(parametric_vol(kk, tt, SPOT, RATE, DIVIDEND, params))

fig = plt.figure(figsize=(12, 5), dpi=130)

ax3d = fig.add_subplot(1, 2, 1, projection="3d")
ax3d.plot_surface(kk, tt, vol_grid * 100.0, cmap="viridis", linewidth=0, antialiased=True)
ax3d.set_xlabel("Strike")
ax3d.set_ylabel("Maturité (années)")
ax3d.set_zlabel("Vol implicite (%)")
ax3d.set_title("Surface de volatilité implicite synthétique")

ax_smile = fig.add_subplot(1, 2, 2)
maturity_cuts = [0.25, 0.5, 1.0, 2.0]
for maturity in maturity_cuts:
    smile = np.asarray(parametric_vol(strikes, maturity, SPOT, RATE, DIVIDEND, params))
    ax_smile.plot(strikes, smile * 100.0, label=f"T={maturity:.2f} an(s)")
ax_smile.axvline(SPOT, color="grey", linestyle=":", linewidth=1)
ax_smile.set_xlabel("Strike")
ax_smile.set_ylabel("Vol implicite (%)")
ax_smile.set_title("Coupes du smile par maturité")
ax_smile.legend(fontsize=8)

fig.suptitle("Surface de volatilité implicite (skew actions, structure par terme)", fontsize=11)
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

out_path = OUT_DIR / "02_vol_surface.png"
fig.savefig(out_path)
plt.close(fig)

# Vérification de la grille interpolée + chiffres clés
grid = VolSurfaceGrid(strikes, maturities, SPOT, RATE, DIVIDEND, params)
vol_atm_1y = grid.interpolate(SPOT, 1.0)
vol_90_1y = grid.interpolate(90.0, 1.0)
vol_110_1y = grid.interpolate(110.0, 1.0)

print(f"Figure enregistrée : {out_path}")
print(f"Vol ATM interpolée (K=100, T=1 an) = {vol_atm_1y:.4%}")
print(f"Skew 1 an : vol(K=90)={vol_90_1y:.4%} vs vol(K=110)={vol_110_1y:.4%} "
      f"(écart={(vol_90_1y - vol_110_1y) * 100:.2f} pts)")

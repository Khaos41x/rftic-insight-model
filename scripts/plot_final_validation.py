"""Regenerate the final validation figure with post-truncation confidence intervals."""
from pathlib import Path
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "data" / "final_sweep_data_CORRECTED_CI.csv").open(newline="", encoding="utf-8") as stream:
    DATA = list(csv.DictReader(stream))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=ROOT / "build" / "figure1_final_validation.png")
OUTPUT = parser.parse_args().output
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

def values(column):
    return [float(row[column]) for row in DATA]

r = values("r")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

ax = axes[0, 0]
ax.plot(r, values("kappa_asymptotic"), "-", color="black", lw=2,
        label="Kramers asymptotic")
ax.plot(r, values("kappa_exact_mfpt"), "--", color="#27ae60", lw=1.5,
        label="Exact 1D MFPT (numerical evaluation)")
ax.errorbar(
    r,
    values("kappa_sim_stabilized"),
    yerr=[
        [a - b for a, b in zip(values("kappa_sim_stabilized"), values("kappa_sim_ci_lo"))],
        [a - b for a, b in zip(values("kappa_sim_ci_hi"), values("kappa_sim_stabilized"))],
    ],
    fmt="o", ms=4, color="#2980b9",
    label="Simulation (stabilized, 95% CI)", capsize=2,
)
ax.set_yscale("log")
ax.set_xlabel(r"$r=\Delta E/T$")
ax.set_ylabel(r"$\kappa_{\mathrm{commit}}=k/\mu$")
ax.set_title("(a) Simulation ↔ exact 1D MFPT ↔ Kramers asymptotic")
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.plot(r, values("rel_err_sim_vs_exact"), "o-", color="#27ae60", ms=4,
        label="Simulation vs exact MFPT")
ax.plot(r, values("rel_err_sim_vs_asymptotic"), "o-", color="#c0392b", ms=4,
        label="Simulation vs Kramers asymptotic")
ax.set_xlabel(r"$r=\Delta E/T$")
ax.set_ylabel("Relative error")
ax.set_title("(b) Relative error vs r")
ax.legend(fontsize=8)

ax = axes[1, 0]
ax.plot(r, values("log_ratio_sim_vs_exact"), "o-", color="#27ae60", ms=4,
        label="log(simulation/exact MFPT)")
ax.plot(r, values("log_ratio_sim_vs_asymptotic"), "o-", color="#c0392b", ms=4,
        label="log(simulation/Kramers asymptotic)")
ax.plot(r, values("log_ratio_exact_vs_asymptotic"), "s--", color="#8e44ad", ms=4,
        label="log(exact MFPT/Kramers asymptotic)")
ax.axhline(0, color="gray", ls=":", lw=1)
ax.set_xlabel(r"$r=\Delta E/T$")
ax.set_ylabel("Log-ratio")
ax.set_title("(c) Log-ratios — error decomposition")
ax.legend(fontsize=7)

ax = axes[1, 1]
ax.plot(r, values("transient_bias_pct"), "o-", color="#d35400", ms=4)
ax.set_xlabel(r"$r=\Delta E/T$")
ax.set_ylabel("Relative difference (%)")
ax.set_title("(d) Stabilized vs untruncated hazard estimates")

fig.suptitle("Final validation of the committed transition rate", fontsize=12)
fig.tight_layout()
fig.savefig(OUTPUT, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT}")

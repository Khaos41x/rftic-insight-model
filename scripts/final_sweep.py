import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, time, json
from pathlib import Path

from kramers_validation import mle_rate_censored, kaplan_meier, Y_PLUS
from safeguards import adaptive_sample_commit, ci95_lognormal, hazard_stabilization_scan, T_RELAX
from exact_mfpt import exact_mfpt, kappa_commit_asymptotic

ROOT = Path(__file__).resolve().parents[1] / "build" / "full-simulation"
ROOT.mkdir(parents=True, exist_ok=False)
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RAW = DATA / "raw_data"
DATA.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
                      "axes.spines.top": False, "axes.spines.right": False})

r_grid = np.geomspace(0.15, 7.0, 16)
X0 = -Y_PLUS
TARGET_EVENTS = 100
N_TRAJ_INIT = 6000
N_TRAJ_MAX = 100000
MAX_S = 150.0
DS = 0.01
SEED_BASE_GLOBAL = 77000

t0 = time.time()
records = []
seeds_used = {}

for i, r in enumerate(r_grid):
    T_tilde = 0.75 / r
    seed_base = SEED_BASE_GLOBAL + i * 10000

    # --- Safeguard 1: mecanismo adaptativo disponível; na execução publicada,
    # a primeira batelada já atingiu a meta em todos os pontos. ---
    esc_t, escaped, cens, n_total, n_batches = adaptive_sample_commit(
        T_tilde, ds=DS, max_s=MAX_S, target_events=TARGET_EVENTS,
        n_traj_init=N_TRAJ_INIT, batch_size=N_TRAJ_INIT, n_traj_max=N_TRAJ_MAX,
        seed_base=seed_base, threshold=Y_PLUS)

    # salva dados brutos (tempos de escape + flag) para reprodutibilidade
    np.savez(RAW / f"r_{i:02d}_raw.npz", r=r, T_tilde=T_tilde, escape_times=esc_t,
             escaped=escaped, censoring_time=cens, n_traj=n_total, seed_base=seed_base)
    seeds_used[f"r_{i:02d}"] = {"r": float(r), "seed_base": int(seed_base), "n_traj": int(n_total)}

    # --- Safeguard 2: estimativa ingênua (t*=0) vs estabilizada (hazard scan) ---
    kappa_naive, se_naive, d_naive = mle_rate_censored(esc_t, escaped, cens)
    t_star, kappa_stab, hazard_table = hazard_stabilization_scan(esc_t, escaped, cens)

    d_post = next(row["n_events"] for row in hazard_table
                  if np.isclose(row["t_star"], t_star))
    ci_lo, ci_hi = ci95_lognormal(kappa_stab, d_post)

    # --- Triangulação: MFPT exato (numérico, sem aproximação de Laplace) ---
    tau_exact = exact_mfpt(X0, b=Y_PLUS, T_tilde=T_tilde)
    kappa_exact_mfpt = 1.0 / tau_exact

    # --- Kramers assintótico ---
    kappa_asymptotic = kappa_commit_asymptotic(r)

    rel_err_vs_exact = abs(kappa_stab - kappa_exact_mfpt) / kappa_exact_mfpt
    rel_err_vs_asymptotic = abs(kappa_stab - kappa_asymptotic) / kappa_asymptotic
    log_ratio_vs_exact = np.log(kappa_stab / kappa_exact_mfpt)
    log_ratio_vs_asymptotic = np.log(kappa_stab / kappa_asymptotic)
    exact_vs_asymptotic_log_ratio = np.log(kappa_exact_mfpt / kappa_asymptotic)

    records.append({
        "r": r, "T_tilde": T_tilde, "n_traj_total": n_total, "n_events": d_naive,
        "events_post_truncation": d_post,
        "kappa_sim_naive": kappa_naive, "t_star_stabilized": t_star,
        "kappa_sim_stabilized": kappa_stab, "kappa_sim_ci_lo": ci_lo, "kappa_sim_ci_hi": ci_hi,
        "kappa_exact_mfpt": kappa_exact_mfpt, "kappa_asymptotic": kappa_asymptotic,
        "rel_err_sim_vs_exact": rel_err_vs_exact, "rel_err_sim_vs_asymptotic": rel_err_vs_asymptotic,
        "log_ratio_sim_vs_exact": log_ratio_vs_exact, "log_ratio_sim_vs_asymptotic": log_ratio_vs_asymptotic,
        "log_ratio_exact_vs_asymptotic": exact_vs_asymptotic_log_ratio,
        "transient_bias_pct": abs(kappa_stab - kappa_naive) / kappa_naive * 100,
    })
    print(f"[{i+1}/{len(r_grid)}] r={r:.3f}  n_traj={n_total}  d={d_naive}  "
          f"kappa_stab={kappa_stab:.6f}  kappa_exact={kappa_exact_mfpt:.6f}  "
          f"kappa_asym={kappa_asymptotic:.6f}  t*={t_star:.2f}")

sweep_df = pd.DataFrame(records)
sweep_df.to_csv(DATA / "final_sweep_data.csv", index=False)
with open(DATA / "final_sweep_seeds.json", "w") as f:
    json.dump(seeds_used, f, indent=2)

print(f"\nSweep completo ({len(r_grid)} pontos) em {time.time()-t0:.1f}s")
print(sweep_df[["r", "n_events", "kappa_sim_stabilized", "kappa_exact_mfpt", "kappa_asymptotic",
                "rel_err_sim_vs_exact", "rel_err_sim_vs_asymptotic", "transient_bias_pct"]].to_string(index=False))

# ===========================================================================
# FIGURA FINAL: triangulação completa
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

ax = axes[0, 0]
ax.plot(r_grid, kappa_commit_asymptotic(r_grid), "-", color="black", lw=2, label="Kramers assintótico")
ax.plot(sweep_df["r"], sweep_df["kappa_exact_mfpt"], "--", color="#27ae60", lw=1.5, label="MFPT exato (numérico)")
ax.errorbar(sweep_df["r"], sweep_df["kappa_sim_stabilized"],
            yerr=[sweep_df["kappa_sim_stabilized"]-sweep_df["kappa_sim_ci_lo"],
                  sweep_df["kappa_sim_ci_hi"]-sweep_df["kappa_sim_stabilized"]],
            fmt="o", ms=4, color="#2980b9", label="Simulação (estabilizada, IC95%)", capsize=2)
ax.set_yscale("log")
ax.set_xlabel("r = ΔE/T")
ax.set_ylabel(r"$\kappa_{commit} = k/\mu$")
ax.set_title("(a) Simulação ↔ MFPT exato ↔ Kramers assintótico")
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.plot(sweep_df["r"], sweep_df["rel_err_sim_vs_exact"], "o-", color="#27ae60", ms=4, label="sim vs MFPT exato")
ax.plot(sweep_df["r"], sweep_df["rel_err_sim_vs_asymptotic"], "o-", color="#c0392b", ms=4, label="sim vs Kramers assintótico")
ax.set_xlabel("r = ΔE/T")
ax.set_ylabel("Erro relativo")
ax.set_title("(b) Erro relativo vs r")
ax.legend(fontsize=8)

ax = axes[1, 0]
ax.plot(sweep_df["r"], sweep_df["log_ratio_sim_vs_exact"], "o-", color="#27ae60", ms=4, label="log(sim/MFPT exato)")
ax.plot(sweep_df["r"], sweep_df["log_ratio_sim_vs_asymptotic"], "o-", color="#c0392b", ms=4, label="log(sim/Kramers assint.)")
ax.plot(sweep_df["r"], sweep_df["log_ratio_exact_vs_asymptotic"], "s--", color="#8e44ad", ms=4,
        label="log(MFPT exato/Kramers assint.)")
ax.axhline(0, color="gray", ls=":", lw=1)
ax.set_xlabel("r = ΔE/T")
ax.set_ylabel("log-razão")
ax.set_title("(c) Log-razões — decomposição do erro")
ax.legend(fontsize=7)

ax = axes[1, 1]
ax.plot(sweep_df["r"], sweep_df["transient_bias_pct"], "o-", color="#d35400", ms=4)
ax.set_xlabel("r = ΔE/T")
ax.set_ylabel("Viés do transiente (%)")
ax.set_title("(d) Viés de ajustar exponencial desde t=0 (sem truncar)")

fig.suptitle("Validação final: taxa de comprometimento (commit) com todos os safeguards aplicados", fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES / "figure5_final_validation.png", bbox_inches="tight")
plt.close(fig)

print("\nFigura final e dados salvos em", ROOT)

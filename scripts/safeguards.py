"""
Três safeguards antes do sweep final:
  1. CIs de 95% + mecanismo de amostragem adaptativa quando necessario
  2. Analise de transiente: estabilizacao do hazard via left-truncation scan,
     ao inves de ajustar exponencial cegamente desde t=0
  3. Teste de convergencia de Euler-Maruyama (Delta_s, Delta_s/2, Delta_s/4)
"""
import numpy as np
from kramers_validation import simulate_dimensionless_batch, mle_rate_censored, Y_PLUS

TARGET_EVENTS = 50
T_RELAX = 0.5  # 1/U''(well) in canonical units, where U''(y_-) = 2


# ---------------------------------------------------------------------------
# Safeguard 1: amostragem adaptativa + CI de 95% (log-normal, baseado no erro
# padrao assintotico da MLE censurada: SE(log kappa_hat) ~ 1/sqrt(d))
# ---------------------------------------------------------------------------

def adaptive_sample_commit(T_tilde, ds=0.01, max_s=80.0, target_events=TARGET_EVENTS,
                            n_traj_init=3000, batch_size=3000, n_traj_max=60000,
                            seed_base=0, threshold=Y_PLUS):
    all_esc_t, all_escaped = [], []
    n_total = 0
    batch_idx = 0
    censoring_time = None
    while True:
        n_batch = n_traj_init if batch_idx == 0 else batch_size
        esc_t, escaped, cens = simulate_dimensionless_batch(
            T_tilde, n_batch, ds=ds, max_s=max_s, seed=seed_base + batch_idx,
            escape_threshold=threshold)
        censoring_time = cens
        all_esc_t.append(esc_t)
        all_escaped.append(escaped)
        n_total += n_batch
        batch_idx += 1

        escaped_concat = np.concatenate(all_escaped)
        d = escaped_concat.sum()
        if d >= target_events or n_total >= n_traj_max:
            break

    esc_t_all = np.concatenate(all_esc_t)
    escaped_all = np.concatenate(all_escaped)
    return esc_t_all, escaped_all, censoring_time, n_total, batch_idx


def ci95_lognormal(kappa_hat, d):
    """CI log-normal aproximado: kappa_hat * exp(+-1.96/sqrt(d))."""
    if d == 0:
        return (0.0, np.inf)
    factor = np.exp(1.96 / np.sqrt(d))
    return (kappa_hat / factor, kappa_hat * factor)


# ---------------------------------------------------------------------------
# Safeguard 2: scan de left-truncation para achar onde o hazard estabiliza
# ---------------------------------------------------------------------------

def left_truncated_mle(escape_times, escaped, censoring_time, t_star):
    """
    Re-estima kappa condicionando em sobrevivencia ate t_star (left-truncation):
    descarta quem já tinha escapado ou sido censurado antes de t_star, re-baseia
    o tempo dos sobreviventes em (t - t_star).
    """
    times = np.where(escaped, escape_times, censoring_time)
    at_risk = times >= t_star
    if at_risk.sum() == 0:
        return np.nan, np.nan, 0
    esc_sub = escape_times[at_risk] - t_star
    escaped_sub = escaped[at_risk]
    cens_sub = censoring_time - t_star
    kappa_hat, se, d = mle_rate_censored(esc_sub, escaped_sub, cens_sub)
    return kappa_hat, se, d


def hazard_stabilization_scan(escape_times, escaped, censoring_time,
                                candidates=(0.0, 0.5*T_RELAX, T_RELAX, 2*T_RELAX, 4*T_RELAX,
                                            8*T_RELAX, 16*T_RELAX, 32*T_RELAX),
                                stability_tol=0.05):
    """
    Roda left_truncated_mle para uma sequencia crescente de t_star. Em vez de
    comparar só vizinhos consecutivos (que para cedo demais se há deriva lenta
    e sistemática acumulada), compara cada candidato contra a ESTIMATIVA MAIS
    TRUNCADA disponível (referência = maior t_star com eventos suficientes) e
    escolhe o menor t_star cuja estimativa já está dentro de stability_tol
    dessa referência -- isso captura deriva lenta, não só ruído entre vizinhos.
    Retorna (t_star_escolhido, kappa_hat_estabilizado, tabela_completa).
    """
    rows = []
    for t_star in candidates:
        k, se, d = left_truncated_mle(escape_times, escaped, censoring_time, t_star)
        rows.append({"t_star": t_star, "kappa_hat": k, "se": se, "n_events": d})

    # referencia: maior t_star que ainda tem numero razoavel de eventos (>=20)
    valid = [r for r in rows if r["n_events"] >= 20 and not np.isnan(r["kappa_hat"])]
    if not valid:
        return candidates[0], rows[0]["kappa_hat"], rows
    reference_kappa = valid[-1]["kappa_hat"]

    chosen_t_star, chosen_kappa = candidates[-1], rows[-1]["kappa_hat"]
    for row in rows:
        k = row["kappa_hat"]
        if k and not np.isnan(k) and reference_kappa:
            rel_change = abs(k - reference_kappa) / reference_kappa
            if rel_change < stability_tol:
                chosen_t_star, chosen_kappa = row["t_star"], k
                break
    return chosen_t_star, chosen_kappa, rows


# ---------------------------------------------------------------------------
# Safeguard 3: convergencia de Euler-Maruyama
# ---------------------------------------------------------------------------

def convergence_test(r_list, ds_list, n_traj=6000, max_s=80.0, seed_base=9000):
    from exact_mfpt import kappa_commit_asymptotic
    rows = []
    for r in r_list:
        T_tilde = 0.75 / r
        for j, ds in enumerate(ds_list):
            esc_t, escaped, cens = simulate_dimensionless_batch(
                T_tilde, n_traj, ds=ds, max_s=max_s, seed=seed_base + int(r*100) + j,
                escape_threshold=Y_PLUS)
            k, se, d = mle_rate_censored(esc_t, escaped, cens)
            rows.append({"r": r, "ds": ds, "kappa_hat": k, "se": se, "n_events": d})
    return rows

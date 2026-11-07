"""
Validação quantitativa do erro de Kramers em função de r = Delta_E/T (adimensional)
=====================================================================================
Implementa os 4 ajustes:
  1. (Lema já corrigido em texto -- ver resposta)
  2. (Correção textual -- ver resposta)
  3. Sistema canônico: y = sqrt(g/mu)*x, s = mu*t
     U_tilde(y) = y^4/12 - y^2/2
     theta = g*T/mu^2 (nome interno T_tilde), r = 3/(4*theta) = 0.75/T_tilde
     kappa_K(r) = k/mu = (1/(sqrt(2)*pi)) * exp(-r)
  4. MLE com censura à direita (Type I) + verificação de hazard aproximadamente
     constante via estimador de tabela de vida (life-table), reportando tanto
     erro relativo quanto log(kappa_emp/kappa_K).
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
Y_PLUS = np.sqrt(3.0)  # poço do sistema canônico

# ---------------------------------------------------------------------------
# Sistema canônico: U(y) = y^4/12 - y^2/2
# ---------------------------------------------------------------------------

def dU_unit(y):
    return (1.0 / 3.0) * y**3 - y


def kappa_K(r):
    """Previsão teórica de Kramers, adimensional: kappa = k/mu = f(r) só."""
    return (1.0 / (np.sqrt(2) * np.pi)) * np.exp(-r)


def simulate_dimensionless_batch(T_tilde, n_traj, ds=0.01, max_s=50.0, seed=0,
                                  escape_threshold=0.0):
    """
    Simula o sistema adimensional dy/ds = -dU_tilde/dy(y) + sqrt(2*T_tilde)*eta(s),
    partindo de y0 = -y_plus = -sqrt(3) (poço do sistema canônico).
    Retorna tempos de escape (s) e flag de censura à direita.
    """
    rng = np.random.default_rng(seed)
    max_steps = int(max_s / ds)
    y0 = -np.sqrt(3.0)  # y_plus do sistema canônico
    y = np.full(n_traj, y0, dtype=float)
    escaped = np.zeros(n_traj, dtype=bool)
    escape_times = np.full(n_traj, np.nan, dtype=float)
    noise_scale = np.sqrt(2.0 * T_tilde * ds)

    for t in range(1, max_steps + 1):
        active = ~escaped
        if not active.any():
            break
        noise = rng.standard_normal(active.sum())
        y[active] = y[active] - dU_unit(y[active]) * ds + noise_scale * noise
        newly = active & (y >= escape_threshold)
        escape_times[newly] = t * ds
        escaped |= newly

    censoring_time = max_steps * ds
    return escape_times, escaped, censoring_time


# ---------------------------------------------------------------------------
# Ajuste 4a: MLE com censura à direita (Type I) para taxa exponencial
# ---------------------------------------------------------------------------

def mle_rate_censored(escape_times, escaped, censoring_time):
    """
    kappa_hat = n_eventos / (soma dos tempos observados + n_censurados * tau)
    (estimador padrão de "tempo total em risco" para taxa exponencial com
    censura Tipo I -- equivalente à razão evento/pessoa-tempo em sobrevivência.)
    """
    n = len(escaped)
    d = escaped.sum()  # numero de eventos (escapes observados)
    total_time_at_risk = np.nansum(np.where(escaped, escape_times, censoring_time))
    if d == 0:
        return 0.0, np.inf, 0
    kappa_hat = d / total_time_at_risk
    # Variancia assintotica aproximada (Fisher info de exponencial censurada): Var ~ kappa^2/d
    se_approx = kappa_hat / np.sqrt(d)
    return kappa_hat, se_approx, d


# ---------------------------------------------------------------------------
# Ajuste 4b: estimador de sobrevivência (Kaplan-Meier) e hazard por tabela de vida
# ---------------------------------------------------------------------------

def kaplan_meier(escape_times, escaped, censoring_time):
    """Estimador KM padrão. Retorna tempos ordenados e S(t)."""
    times = np.where(escaped, escape_times, censoring_time)
    order = np.argsort(times)
    times_sorted = times[order]
    escaped_sorted = escaped[order]
    n = len(times)
    at_risk = n
    S = 1.0
    t_list, S_list = [0.0], [1.0]
    i = 0
    while i < n:
        t_cur = times_sorted[i]
        # agrupa eventos no mesmo instante (empates)
        j = i
        d_events = 0
        n_at_t = 0
        while j < n and times_sorted[j] == t_cur:
            n_at_t += 1
            if escaped_sorted[j]:
                d_events += 1
            j += 1
        if d_events > 0:
            S *= (1 - d_events / at_risk)
            t_list.append(t_cur)
            S_list.append(S)
        at_risk -= n_at_t
        i = j
    return np.array(t_list), np.array(S_list)


def life_table_hazard(escape_times, escaped, censoring_time, n_bins=15):
    """Hazard binned (atuarial): h_k = eventos_no_intervalo / (pessoa-tempo_no_intervalo)."""
    times = np.where(escaped, escape_times, censoring_time)
    max_t = censoring_time
    edges = np.linspace(0, max_t, n_bins + 1)
    hazards = []
    centers = []
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        width = hi - lo
        in_or_past = times >= lo  # ainda em risco no inicio do intervalo
        events_in_bin = escaped & (escape_times >= lo) & (escape_times < hi)
        # pessoa-tempo: quem sobreviveu o intervalo inteiro contribui 'width';
        # quem teve evento ou censura dentro do intervalo contribui tempo parcial
        person_time = 0.0
        at_risk_mask = times >= lo
        t_in_bin = np.clip(times[at_risk_mask], lo, hi) - lo
        person_time = t_in_bin.sum()
        n_events = events_in_bin.sum()
        h = n_events / person_time if person_time > 0 else np.nan
        hazards.append(h)
        centers.append((lo + hi) / 2)
    return np.array(centers), np.array(hazards)

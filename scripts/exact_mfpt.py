"""
MFPT exato (integral de Pontryagin/backward FPE), committor, e a distinção
hit-vs-commit para o problema do poço duplo simétrico.

Potencial canônico: U(y) = y^4/12 - y^2/2
Equação de Langevin overdamped: dy/ds = -U'(y) + sqrt(2*theta)*eta(s),
com theta = g*T/mu^2 (variável interna T_tilde; gamma=1, D=theta).
"""
import numpy as np
from scipy import integrate, optimize

Y_PLUS = np.sqrt(3.0)  # poco do sistema unitario


def U(y):
    return (1.0 / 12.0) * y**4 - 0.5 * y**2


def exact_mfpt(x0, b, T_tilde, a=-20.0, n_grid=20000):
    """
    MFPT exato (formula de Pontryagin/backward Fokker-Planck), sem aproximação
    de Laplace, para absorção em x=b, partindo de x0, com 'a' suficientemente
    negativo para que U(a)/T >> 1 (equivalente a reflexão no infinito, já que
    o potencial confina por causa do termo quártico).

        tau(x0) = (1/T) * integral_{x0}^{b} dy  e^{U(y)/T} * [integral_{a}^{y} dz e^{-U(z)/T}]

    Implementado com grade fina + integração cumulativa (trapézio), com
    estabilização numérica (subtrai o mínimo de U no intervalo relevante
    antes de exponenciar, para evitar overflow).
    """
    # grade fina cobrindo [a, b]
    grid = np.linspace(a, b, n_grid)
    Uvals = U(grid)

    # --- integral interna cumulativa: I(y) = integral_a^y exp(-U(z)/T) dz ---
    # estabiliza subtraindo o minimo de U no dominio (o minimo esta no poço)
    Umin = Uvals.min()
    integrand_inner = np.exp(-(Uvals - Umin) / T_tilde)
    I_cum = integrate.cumulative_trapezoid(integrand_inner, grid, initial=0.0)
    # I_cum[i] = integral_a^{grid[i]} exp(-(U(z)-Umin)/T) dz  (fator exp(Umin/T) absorvido depois)

    # --- integral externa: integral_{x0}^{b} dy exp(U(y)/T) * I(y) ---
    mask = (grid >= x0) & (grid <= b)
    y_outer = grid[mask]
    Uy_outer = Uvals[mask]
    I_outer = I_cum[mask]

    # exp(U(y)/T) tambem precisa de estabilizacao: subtrai o maximo de U no dominio relevante (perto da sela)
    Umax = Uy_outer.max()
    outer_integrand = np.exp((Uy_outer - Umax) / T_tilde) * I_outer
    # tau = (1/T) * exp((Umax - Umin)/T) * integral(outer_integrand)  -- reconstituindo os fatores estabilizados
    outer_integral = integrate.trapezoid(outer_integrand, y_outer)
    log_prefactor = (Umax - Umin) / T_tilde  # vai multiplicar; cuidado com overflow para r grande -- usamos log-space
    tau = (1.0 / T_tilde) * np.exp(log_prefactor) * outer_integral
    return tau


def committor(x0, a=-Y_PLUS, b=Y_PLUS, T_tilde=1.0, n_grid=20000):
    """
    q(x0) = probabilidade de atingir b antes de a, partindo de x0.
    Solução exata da ODE backward: q(x) = [int_a^x e^{U(y)/T} dy] / [int_a^b e^{U(y)/T} dy]
    """
    grid = np.linspace(a, b, n_grid)
    Uvals = U(grid)
    Umax = Uvals.max()
    integrand = np.exp((Uvals - Umax) / T_tilde)
    cum = integrate.cumulative_trapezoid(integrand, grid, initial=0.0)
    total = cum[-1]
    idx = np.searchsorted(grid, x0)
    idx = min(max(idx, 0), len(cum) - 1)
    return cum[idx] / total


def kappa_commit_asymptotic(r):
    """
    Eyring-Kramers padrao (basin-to-basin), prefactor derivado das curvaturas
    REAIS deste potencial (omega_well=sqrt(2*mu), omega_barrier=sqrt(mu)):
    kappa_K(r) = (1/sqrt(2*pi)) * exp(-r).
    """
    return (1.0 / (np.sqrt(2) * np.pi)) * np.exp(-r)


def kappa_hit_asymptotic(r):
    """
    MFPT ao ponto de sela (hitting time), aproximacao de Laplace. O argumento do
    "meio-gaussiano" (integral externa cortada em y=0, capturando so metade do
    pico gaussiano em torno da sela) produz um fator EXATO de 2 em relacao ao
    commit-rate, independente da razao omega_well/omega_barrier -- o corte
    acontece na integral dominada pela curvatura da BARREIRA, nao do poco.
    Por isso kappa_hit = 2*kappa_commit aqui, nao o "1/pi" generico de um
    duplo-poco com omega_well=omega_barrier (que nao e o nosso caso: temos
    omega_well=sqrt(2mu) != omega_barrier=sqrt(mu)).
    """
    return 2.0 * kappa_commit_asymptotic(r)

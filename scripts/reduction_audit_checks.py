"""
Audit checks for the chain: continuous Hopfield energy -> 1D reduction -> quartic-quadratic
normal form -> barrier law.  All checks are sanity/consistency checks, NOT proofs.

Energy (unit gain and unit leak, as in the manuscript):
    E(S) = -1/2 S^T W S + sum_i F(S_i),   F(z) = int_0^z atanh(s) ds = z*atanh(z) + 1/2*ln(1-z^2)
Gradient: -W S + atanh(S).   Hessian: H(S) = -W + diag(1/(1-S_i^2)).
"""
import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar

print("=" * 78)
print("CHECK 1 (symbolic): series of the saturation integral and reduced coefficients")
print("=" * 78)
z = sp.symbols('z', real=True)
Fz = z * sp.atanh(z) + sp.log(1 - z**2) / 2
print("dF/dz - atanh(z) =", sp.simplify(sp.diff(Fz, z) - sp.atanh(z)))
print("series F(z):", sp.series(Fz, z, 0, 9))
x = sp.symbols('x', real=True)
v = sp.symbols('v1 v2 v3', real=True)
Fser = sp.series(Fz, z, 0, 8).removeO()
sat = sp.expand(sum(Fser.subs(z, x * vi) for vi in v))
print("coef x^2:", sp.factor(sat.coeff(x, 2)), "| coef x^4:", sp.factor(sat.coeff(x, 4)),
      "| coef x^6:", sp.factor(sat.coeff(x, 6)))

print("\n" + "=" * 78)
print("CHECK 2 (symbolic): normal form U = g x^4/12 - mu x^2/2: wells, barrier, curvatures")
print("=" * 78)
mu, g, T = sp.symbols('mu g T', positive=True)
U = g * x**4 / 12 - mu * x**2 / 2
xp = sp.sqrt(3 * mu / g)
dE = sp.simplify(-U.subs(x, xp))
print("critical points:", sp.solve(sp.diff(U, x), x))
print("Delta E =", dE, "| U''(0) =", sp.diff(U, x, 2).subs(x, 0),
      "| U''(x+) =", sp.simplify(sp.diff(U, x, 2).subs(x, xp)))
k = sp.sqrt(sp.diff(U, x, 2).subs(x, xp) * (-sp.diff(U, x, 2).subs(x, 0))) / (2 * sp.pi) * sp.exp(-dE / T)
print("Kramers k =", sp.simplify(k))
print("M = 2 x+ =", sp.simplify(2 * xp))

print("\n" + "=" * 78)
print("CHECK 3 (symbolic): rescaling. y = sqrt(g/mu) x, s = mu t  ->  dy = -(y^3/3 - y) ds + sqrt(3/(2r)) dW")
print("=" * 78)
y, r = sp.symbols('y r', positive=True)
xx = sp.sqrt(mu / g) * y
drift = sp.simplify(-sp.diff(U, x).subs(x, xx) / sp.sqrt(mu / g) / mu)
noise = sp.sqrt(2 * T) / sp.sqrt(mu / g) / sp.sqrt(mu)
noise_r = sp.simplify(noise.subs(T, 3 * mu**2 / (4 * g * r)))
print("drift per ds:", drift, "| noise coefficient:", noise_r, "  (= sqrt(3/(2r)))")
print("theta = g T / mu^2 =", sp.simplify((g * T / mu**2).subs(T, 3 * mu**2 / (4 * g * r))), "  (= 3/(4r) = 0.75/r)")
print("kappa = k/mu =", sp.simplify((k / mu).subs(T, 3 * mu**2 / (4 * g * r))))

rng = np.random.default_rng(5)
F = lambda zz: zz * np.arctanh(zz) + 0.5 * np.log(1 - zz**2)

print("\n" + "=" * 78)
print("CHECK 4 (numeric): action of the Hebbian term on v* and on u (random orthonormal a,b)")
print("=" * 78)
N0 = 40
a = rng.standard_normal(N0); a /= np.linalg.norm(a)
b = rng.standard_normal(N0); b -= (a @ b) * a; b /= np.linalg.norm(b)
vv = (a - b) / np.sqrt(2); uu = (a + b) / np.sqrt(2)
for eps in [0.1, 0.7]:
    Wt = eps * (np.outer(a, b) + np.outer(b, a))
    print(f"eps={eps}: ||Wt v + eps v|| = {np.linalg.norm(Wt @ vv + eps * vv):.1e} ; "
          f"||Wt u - eps u|| = {np.linalg.norm(Wt @ uu - eps * uu):.1e}   (v: -eps, u: +eps)")

print("\n" + "=" * 78)
print("CHECK 5 (numeric): partial minimization over transverse directions (A2 holds), N=60")
print("=" * 78)
N = 60
vec = rng.standard_normal(N); vec /= np.linalg.norm(vec)
gg = (vec**4).sum(); hh = (vec**6).sum()
Q, _ = np.linalg.qr(np.column_stack([vec, rng.standard_normal((N, N - 1))]))
Qc = Q[:, 1:]
Rspec = rng.uniform(-0.4, 0.4, N - 1)           # transverse spectrum of W: <= 0.4 => gamma >= 0.6
Rm = Qc @ np.diag(Rspec) @ Qc.T
A = Qc.T @ (np.eye(N) - Rm) @ Qc                 # (I - W)|_perp (W acts as Rm on the complement)
q = Qc.T @ (vec**3)
c6 = hh / 30 - (q @ np.linalg.solve(A, q)) / 18
print(f"g = {gg:.6f}, h = {hh:.6e}, |P_perp v^3|^2 = {q@q:.6e} (= h - g^2 = {hh-gg**2:.6e})")
print(f"transverse spectral gap gamma = {1-Rspec.max():.3f};  c6 = h/30 - q^T A^-1 q/18 = {c6:.6e}  (h/30 = {hh/30:.6e})")

def U_partial(xv, lamW):
    """U(x) = min_y E(x v + y), y in v-perp, by Newton in transverse coordinates."""
    W = lamW * np.outer(vec, vec) + Rm
    zc = np.zeros(N - 1)
    for _ in range(60):
        S = xv * vec + Qc @ zc
        grad = Qc.T @ (-W @ S + np.arctanh(S))
        H = Qc.T @ (-W + np.diag(1.0 / (1 - S**2))) @ Qc
        step = np.linalg.solve(H, grad)
        zc = zc - step
        if np.linalg.norm(step) < 1e-15:
            break
    S = xv * vec + Qc @ zc
    return -0.5 * S @ W @ S + F(S).sum(), np.linalg.norm(Qc @ zc)

mu_t = 0.03
lamW = 1 + mu_t
print(f"\nExpansion check at mu = {mu_t}: R(x) = U(x) - (-mu x^2/2 + g x^4/12);  R/x^6 should approach c6")
for xv in [0.1, 0.15, 0.2, 0.3]:
    Ux, ynorm = U_partial(xv, lamW)
    Rx = Ux - (-mu_t * xv**2 / 2 + gg * xv**4 / 12)
    print(f"  x={xv:4.2f}: R/x^6 = {Rx / xv**6: .4e}   (c6 = {c6:.4e}) ; |y*| = {ynorm:.2e} (O(x^3): |y*|/x^3 = {ynorm/xv**3:.3e})")

print("\nBarrier: exact (partial minimization) vs normal form vs first-order correction -36 c6 mu/g^2")
print(f"{'mu':>7} | {'DE exact':>11} {'3mu^2/(4g)':>11} {'rel.err':>8} | {'pred. -36 c6 mu/g^2':>20} | {'line-only (S=xv) DE':>20}")
for mu_i in [0.005, 0.01, 0.02, 0.05]:
    lw = 1 + mu_i
    res = minimize_scalar(lambda xv: U_partial(xv, lw)[0], bounds=(1e-6, 0.95 / np.abs(vec).max()),
                          method='bounded', options={'xatol': 1e-12})
    dE_ex = -res.fun
    dE_nf = 3 * mu_i**2 / (4 * gg)
    Wl = lw * np.outer(vec, vec) + Rm
    resl = minimize_scalar(lambda xv: -0.5 * lw * xv**2 + F(xv * vec).sum(), bounds=(1e-6, 0.95 / np.abs(vec).max()),
                           method='bounded', options={'xatol': 1e-12})
    print(f"{mu_i:7.3f} | {dE_ex:11.6f} {dE_nf:11.6f} {(dE_nf-dE_ex)/dE_ex:8.2%} | "
          f"{-36 * c6 * mu_i / gg**2:20.2%} | {-resl.fun:20.6f}")
print("(rel.err = (normal form - exact)/exact; the first-order prediction is the fractional change of the exact barrier)")

print("\n" + "=" * 78)
print("CHECK 6 (numeric): transverse strong convexity of E on the cube (consequence of A2)")
print("=" * 78)
W_full = (1 + mu_t) * np.outer(vec, vec) + Rm
gamma = 1 - Rspec.max()
worst = np.inf
for _ in range(2000):
    S = rng.uniform(-0.999, 0.999, N)
    Hs = -W_full + np.diag(1.0 / (1 - S**2))
    worst = min(worst, np.linalg.eigvalsh(Qc.T @ Hs @ Qc).min())
print(f"min over 2000 random S in the cube of lambda_min(H_perp,perp) = {worst:.4f}  >= gamma = {gamma:.4f}")

print("\n" + "=" * 78)
print("CHECK 7 (numeric): A2 is NOT satisfied by symmetric two-pattern Hebbian storage W0 = c(aa^T + bb^T)")
print("=" * 78)
Nn = 40
aa = np.zeros(Nn); bb = np.zeros(Nn); aa[:20] = 1 / np.sqrt(20); bb[20:] = 1 / np.sqrt(20)
c = 1.2
vs = (aa - bb) / np.sqrt(2); us = (aa + bb) / np.sqrt(2)
for eps in [0.0, 0.05, 0.1]:
    Wf = c * (np.outer(aa, aa) + np.outer(bb, bb)) + eps * (np.outer(aa, bb) + np.outer(bb, aa))
    print(f"eps={eps:4.2f}: lambda_W(eps) = v^T W v = {vs @ Wf @ vs:.3f} (= c - eps) ; u^T W u = {us @ Wf @ us:.3f} (= c + eps > 1)")
print("-> transverse eigenvalue c+eps exceeds 1: no spectral gap; the 1-D reduction is not justified in this example.")
print("Necessary condition for A2 on [0, eps_max]:  u^T W0 u + eps_max <= 1 - gamma.")

print("\n" + "=" * 78)
print("CHECK 8 (numeric): when A2 FAILS (symmetric two-pattern storage, c = 1.2), do the wells along v* stay minima of the full energy?")
print("=" * 78)
from scipy.optimize import brentq
c8 = 1.2
print(f"W0 = c(aa^T + bb^T), c = {c8} (mu0 = {c8-1:.2f}); S = x v* exact (flat patterns, |S_i| = s); H = -W + diag(1/(1-S^2))")
print(f"{'eps':>6} | {'s':>7} | {'H_vv':>8} {'H_uu':>8} | verdict")
for eps in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.19]:
    s = brentq(lambda s: np.arctanh(s) / s - (c8 - eps), 1e-6, 0.9999)   # critical-point condition along v*
    Hvv = 1 / (1 - s**2) - (c8 - eps)      # curvature along v* (positive => minimum along the line)
    Huu = 1 / (1 - s**2) - (c8 + eps)      # curvature along the symmetric mode u
    print(f"{eps:6.2f} | {s:7.4f} | {Hvv:8.4f} {Huu:8.4f} | {'local minimum of full energy' if Huu > 0 else 'SADDLE of full energy'}")
print("-> A2 is sufficient for the clean reduction, not necessary for the wells to be minima; but in this example the wells along v* "
      "stop being minima of the full energy for eps between 0.10 and 0.12 (about mu0/2), so the 1-D picture breaks down.")

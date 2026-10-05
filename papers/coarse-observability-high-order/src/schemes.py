"""
Linear circulant scheme family for periodic scalar advection.

This module is part of the reproducibility package for the manuscript

    Coarse Observability Across High-Order Finite-Volume
    Reconstructions: Universal Rank, Conditioning, and
    Dissipative Memory

(article_4, target: Advances in Computational Mathematics).

Conventions
-----------
The PDE is u_t + a u_x = 0 with a > 0 and Courant number
lambda = a*dt/h in (0, 1].

A one-step linear circulant scheme is written as

    u_i^{n+1} = sum_s c_s(lambda) u_{i-s}^n        (periodic indexing),

i.e. the update matrix is A = sum_s c_s S^s, where S is the downstream
cyclic shift (S u)_i = u_{i-1}.  Coefficients are stored as a dict
{s: c_s}.  Positive s samples upstream cells (i-s to the left), negative
s samples downstream cells.  The upstream reach is m_L = max(s), the
downstream reach is m_R = max(-s), and the total stencil reach is
w = m_L + m_R.

Scheme construction
-------------------
All interpolation-based (semi-Lagrangian) schemes are generated from one
formula: interpolate the time-n solution at the integer offsets `nodes`
with the unique polynomial of degree len(nodes)-1, and evaluate at the
departure point x_i - lambda*h.  This reproduces the classical schemes:

    nodes = [-1, 0]           -> first-order upwind      (UW1)
    nodes = [-1, 0, 1]        -> Lax-Wendroff            (LW)
    nodes = [-2, -1, 0]       -> Beam-Warming            (BW)
    nodes = [-2, -1, 0, 1]    -> third-order upwind-biased (UB3)
    nodes = [-3, ..., 2]      -> fifth-order upwind-biased (UB5)

Fromm's scheme is the average of LW and BW.  The linear-weights WENO5
operator combined with forward Euler ("WENO5L-FE") is built from the
standard fifth-order upwind-biased numerical flux

    fhat_{i+1/2} = (2u_{i-2} - 13u_{i-1} + 47u_i + 27u_{i+1} - 3u_{i+2})/60.

Note: WENO5L-FE is included as an *observation operator* (rank and
conditioning experiments).  It is not linearly stable as a time-stepping
method with forward Euler; erasure experiments must pair the spatial
operator with SSP-RK time integration (later experiment, not E1).

Exact rank prediction
---------------------
The universality theorem of the manuscript is proved by block-
diagonalizing the observation matrix O_L = [R; RA; ...; RA^L] in the
discrete Fourier basis.  Fine mode k has A-eigenvalue

    g_k = sum_s c_s exp(-2*pi*i*k*s/N),

and the restriction R maps mode k to coarse mode (k mod P) with a
Dirichlet weight D_k that vanishes exactly for k in {P, 2P, ..., (r-1)P}.
Grouping fine modes by residue class m = k mod P, the class-m block of
O_L is a weighted Vandermonde matrix in the nodes {g_{m+jP}}.  Hence

    rank O_L = sum_m min(L+1, d_m),

where d_0 = 1 and, for m != 0, d_m is the number of *distinct* symbol
values in class m.  When all class nodes are distinct this collapses to
the scheme-independent ladder P + (P-1)*min(L, r-1).
`predicted_rank` implements the node-counting formula, and
`find_collisions` reports any coincident nodes (the exceptional set).
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

Coeffs = dict[int, float]
#: Polynomial in lambda, low degree first, exact rational coefficients.
Poly = list[Fraction]


# ----------------------------------------------------------------------
# Scheme constructors
# ----------------------------------------------------------------------

def semi_lagrangian(nodes: list[int], lam: float) -> Coeffs:
    """Interpolation scheme on integer offsets `nodes`, evaluated at -lambda.

    Returns {s: c_s} with c_s multiplying u_{i-s}; a node at offset d
    contributes to s = -d.
    """
    coeffs: Coeffs = {}
    for d in nodes:
        num = 1.0
        den = 1.0
        for dp in nodes:
            if dp != d:
                num *= (-lam - dp)
                den *= float(d - dp)
        coeffs[-d] = num / den
    return coeffs


def uw1(lam: float) -> Coeffs:
    return semi_lagrangian([-1, 0], lam)


def lax_wendroff(lam: float) -> Coeffs:
    return semi_lagrangian([-1, 0, 1], lam)


def beam_warming(lam: float) -> Coeffs:
    return semi_lagrangian([-2, -1, 0], lam)


def fromm(lam: float) -> Coeffs:
    a = lax_wendroff(lam)
    b = beam_warming(lam)
    out: Coeffs = {}
    for s in set(a) | set(b):
        out[s] = 0.5 * a.get(s, 0.0) + 0.5 * b.get(s, 0.0)
    return out


def ub3(lam: float) -> Coeffs:
    return semi_lagrangian([-2, -1, 0, 1], lam)


def ub5(lam: float) -> Coeffs:
    return semi_lagrangian([-3, -2, -1, 0, 1, 2], lam)


def weno5_linear_euler(lam: float) -> Coeffs:
    """Linear-weights WENO5 flux difference advanced with forward Euler.

    fhat_{i+1/2} = sum_d flux[d] * u_{i+d};
    u_i^{n+1} = u_i - lam * (fhat_{i+1/2} - fhat_{i-1/2}).
    """
    flux = {-2: 2.0 / 60.0, -1: -13.0 / 60.0, 0: 47.0 / 60.0,
            1: 27.0 / 60.0, 2: -3.0 / 60.0}
    coeffs: Coeffs = {0: 1.0}
    for d, wgt in flux.items():
        # fhat_{i+1/2} uses u_{i+d}  -> s = -d
        coeffs[-d] = coeffs.get(-d, 0.0) - lam * wgt
        # fhat_{i-1/2} uses u_{i-1+d} -> s = 1-d
        coeffs[1 - d] = coeffs.get(1 - d, 0.0) + lam * wgt
    return coeffs


#: Registry: name -> (constructor, formal spatial/temporal order, label)
SCHEMES: dict[str, tuple] = {
    "uw1":     (uw1,               1, "first-order upwind"),
    "lw":      (lax_wendroff,      2, "Lax-Wendroff"),
    "bw":      (beam_warming,      2, "Beam-Warming"),
    "fromm":   (fromm,             2, "Fromm"),
    "ub3":     (ub3,               3, "third-order upwind-biased"),
    "ub5":     (ub5,               5, "fifth-order upwind-biased"),
    "weno5l":  (weno5_linear_euler, 1, "WENO5 linear weights + Euler"),
}


def reach(coeffs: Coeffs) -> tuple[int, int]:
    """(m_L, m_R): upstream and downstream stencil reach."""
    nonzero = [s for s, c in coeffs.items() if abs(c) > 0.0]
    return max(max(nonzero), 0), max(-min(nonzero), 0)


# ----------------------------------------------------------------------
# Matrices
# ----------------------------------------------------------------------

def circulant_matrix(coeffs: Coeffs, N: int) -> np.ndarray:
    """Dense update matrix A = sum_s c_s S^s on N periodic cells."""
    A = np.zeros((N, N), dtype=float)
    for s, c in coeffs.items():
        for i in range(N):
            A[i, (i - s) % N] += c
    return A


def build_R(P: int, r: int) -> np.ndarray:
    """Parent-average restriction matrix, shape (P, P*r)."""
    if P < 2 or r < 2:
        raise ValueError("Require P >= 2 and r >= 2.")
    R = np.zeros((P, P * r), dtype=float)
    for K in range(P):
        R[K, K * r:(K + 1) * r] = 1.0 / r
    return R


def build_O(coeffs: Coeffs, P: int, r: int, L: int) -> np.ndarray:
    """Observation matrix [R; RA; ...; RA^L], shape ((L+1)P, P*r)."""
    if L < 0:
        raise ValueError("Require L >= 0.")
    N = P * r
    R = build_R(P, r)
    A = circulant_matrix(coeffs, N)
    rows = []
    Apow = np.eye(N)
    for _ in range(L + 1):
        rows.append(R @ Apow)
        Apow = A @ Apow
    return np.vstack(rows)


# ----------------------------------------------------------------------
# Symbols and exact rank prediction
# ----------------------------------------------------------------------

def symbol(coeffs: Coeffs, k: int, N: int) -> complex:
    """Fourier symbol g_k = sum_s c_s exp(-2*pi*i*k*s/N)."""
    return sum(c * np.exp(-2j * np.pi * k * s / N)
               for s, c in coeffs.items())


def class_nodes(coeffs: Coeffs, P: int, r: int, m: int) -> list[complex]:
    """Symbol values of residue class m, dropping zero-weight modes.

    The Dirichlet weight D_k vanishes exactly for k = P, 2P, ..., (r-1)P,
    i.e. for every mode of class m = 0 except k = 0.
    """
    N = P * r
    nodes = []
    for j in range(r):
        k = m + j * P
        if m == 0 and k != 0:
            continue
        nodes.append(symbol(coeffs, k, N))
    return nodes


def count_distinct(nodes: list[complex], tol: float = 1.0e-10) -> int:
    distinct: list[complex] = []
    for g in nodes:
        if all(abs(g - h) > tol for h in distinct):
            distinct.append(g)
    return len(distinct)


def predicted_rank(coeffs: Coeffs, P: int, r: int, L: int,
                   tol: float = 1.0e-10) -> int:
    """Exact rank of O_L from the Fourier node-counting formula."""
    total = 0
    for m in range(P):
        total += min(L + 1, count_distinct(class_nodes(coeffs, P, r, m), tol))
    return total


def universal_formula(P: int, r: int, L: int) -> int:
    """Scheme-independent ladder P + (P-1)*min(L, r-1)."""
    return P + (P - 1) * min(L, r - 1)


def find_collisions(coeffs: Coeffs, P: int, r: int,
                    tol: float = 1.0e-10) -> list[tuple[int, int, int]]:
    """Coincident symbol nodes (m, j, j') within nonzero residue classes."""
    N = P * r
    hits = []
    for m in range(1, P):
        gs = [symbol(coeffs, m + j * P, N) for j in range(r)]
        for j in range(r):
            for jp in range(j + 1, r):
                if abs(gs[j] - gs[jp]) <= tol:
                    hits.append((m, j, jp))
    return hits


def contraction_factor(coeffs: Coeffs, N: int) -> float:
    """rho_N = max_{k != 0} |g_k|: contraction on the zero-mean subspace."""
    return max(abs(symbol(coeffs, k, N)) for k in range(1, N))


def flux_weights(coeffs: Coeffs, lam: float) -> Coeffs:
    """Physical-flux weights of the unique conservation form.

    Writes the update u_i^{n+1} = sum_s c_s u_{i-s} as
    u_i - lam*(F_{i+1/2} - F_{i-1/2}) with F_{i+1/2} = sum_s w_s u_{i-s}.
    Partial summation gives w_s = -(1/lam) * sum_{t<=s} d_t with
    d = c - delta_0; consistency (sum_s c_s = 1) makes the support
    compact.  Normalization: the manuscript's compact flux
    (prop:flux-form) is phi = lam * F, i.e. w_s = H_s(lam)/lam
    (rem:flux-normalization); the rescaling is why lam = 0 is excluded
    here although phi itself is defined for every lam.
    """
    if lam == 0.0:
        raise ValueError("Flux weights are defined for lam != 0.")
    d = dict(coeffs)
    d[0] = d.get(0, 0.0) - 1.0
    smin, smax = min(d), max(d)
    w: Coeffs = {}
    acc = 0.0
    for s_ in range(smin, smax):
        acc += d.get(s_, 0.0)
        val = -acc / lam
        if val != 0.0:
            w[s_] = val
    total = sum(d.values())
    if abs(total) > 1.0e-12:
        raise ValueError("Scheme is not consistent: flux form does not close.")
    return w


def apply_flux(w: Coeffs, u: np.ndarray) -> np.ndarray:
    """Interface fluxes F_{i+1/2} = sum_s w_s u_{i-s} (periodic, axis 0)."""
    F = np.zeros_like(u)
    for s_, wv in w.items():
        F += wv * np.roll(u, s_, axis=0)
    return F


# ----------------------------------------------------------------------
# Exact coefficient polynomials in lambda (for the collision certificate)
# ----------------------------------------------------------------------

def _poly_mul(p: Poly, q: Poly) -> Poly:
    out = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return out


def _poly_add(p: Poly, q: Poly) -> Poly:
    n = max(len(p), len(q))
    return [(p[i] if i < len(p) else Fraction(0)) +
            (q[i] if i < len(q) else Fraction(0)) for i in range(n)]


def _sl_polys(nodes: list[int]) -> dict[int, Poly]:
    """Exact lambda-polynomials of the semi-Lagrangian coefficients.

    c_{-d}(lambda) = prod_{d' != d} (-lambda - d') / (d - d'), a polynomial
    of degree len(nodes)-1 with rational coefficients.
    """
    out: dict[int, Poly] = {}
    for d in nodes:
        num: Poly = [Fraction(1)]
        den = Fraction(1)
        for dp in nodes:
            if dp != d:
                num = _poly_mul(num, [Fraction(-dp), Fraction(-1)])
                den *= Fraction(d - dp)
        out[-d] = [a / den for a in num]
    return out


def coeff_polys(name: str) -> dict[int, Poly]:
    """Exact polynomial representation c_s(lambda) per discrete operator.

    Accepts the seven spatial-scheme keys of SCHEMES plus "weno5l_rk3",
    the composed WENO5L + SSP-RK3 one-step operator (REVISION_2 M2), so
    both collision-certificate layers can treat all eight fully discrete
    operators uniformly.
    """
    if name == "weno5l_rk3":
        return weno5l_rk3_polys()
    sl_nodes = {"uw1": [-1, 0], "lw": [-1, 0, 1], "bw": [-2, -1, 0],
                "ub3": [-2, -1, 0, 1], "ub5": [-3, -2, -1, 0, 1, 2]}
    if name in sl_nodes:
        return _sl_polys(sl_nodes[name])
    if name == "fromm":
        a = coeff_polys("lw")
        b = coeff_polys("bw")
        half = Fraction(1, 2)
        out: dict[int, Poly] = {}
        for s in set(a) | set(b):
            pa = [half * c for c in a.get(s, [Fraction(0)])]
            pb = [half * c for c in b.get(s, [Fraction(0)])]
            out[s] = _poly_add(pa, pb)
        return out
    if name == "weno5l":
        flux = {-2: Fraction(2, 60), -1: Fraction(-13, 60),
                0: Fraction(47, 60), 1: Fraction(27, 60),
                2: Fraction(-3, 60)}
        out = {0: [Fraction(1)]}
        for d, f in flux.items():
            out[-d] = _poly_add(out.get(-d, [Fraction(0)]),
                                [Fraction(0), -f])
            out[1 - d] = _poly_add(out.get(1 - d, [Fraction(0)]),
                                   [Fraction(0), f])
        return out
    raise ValueError(f"unknown scheme {name!r}")


def eval_polys(polys: dict[int, Poly], lam: float) -> Coeffs:
    return {s: float(sum(float(a) * lam ** i for i, a in enumerate(p)))
            for s, p in polys.items()}


def collision_certificate(name: str, P: int, r: int,
                          real_tol: float = 1.0e-8) -> list[tuple]:
    """Complete characterization of the spectral-collision set on (0, 1].

    For every residue class m != 0 and every mode pair (k, k') in the class,
    the symbol difference

        delta_{k,k'}(lambda) = g_k(lambda) - g_{k'}(lambda)

    is a polynomial in lambda (degree <= 5 for the present family) with
    complex coefficients.  A collision at Courant number lambda is a real
    root of delta in (0, 1].  This routine computes all roots of every pair
    polynomial and returns those falling in the admissible interval (up to
    floating-point root-finding tolerance), plus a flag for identically
    vanishing pairs (which would mean a collision at *every* lambda).

    Unlike the sampled check `find_collisions`, this certifies emptiness of
    the collision set on the whole interval, not just on a lambda grid.
    """
    polys = coeff_polys(name)
    N = P * r
    findings: list[tuple] = []
    for m in range(1, P):
        ks = [m + j * P for j in range(r)]
        for a_idx in range(r):
            for b_idx in range(a_idx + 1, r):
                k, kp = ks[a_idx], ks[b_idx]
                deg = max(len(p) for p in polys.values())
                coefs = np.zeros(deg, dtype=complex)
                for s, p in polys.items():
                    fac = (np.exp(-2j * np.pi * k * s / N)
                           - np.exp(-2j * np.pi * kp * s / N))
                    for i, c in enumerate(p):
                        coefs[i] += float(c) * fac
                if np.max(np.abs(coefs)) < 1.0e-14:
                    findings.append((m, k, kp, "identically_zero"))
                    continue
                # np.roots expects highest degree first; trim leading zeros.
                hi_first = coefs[::-1]
                nz = np.flatnonzero(np.abs(hi_first) > 1.0e-14)
                hi_first = hi_first[nz[0]:]
                if len(hi_first) < 2:
                    continue  # constant nonzero polynomial: no roots
                for rt in np.roots(hi_first):
                    if abs(rt.imag) < real_tol and \
                            real_tol < rt.real <= 1.0 + real_tol:
                        findings.append((m, k, kp, complex(rt)))
    return findings


# ----------------------------------------------------------------------
# Exact dissipation analysis (autocorrelation moments)
# ----------------------------------------------------------------------

def _poly_is_zero(p: Poly) -> bool:
    return all(a == 0 for a in p)


def _poly_scale(p: Poly, k: Fraction) -> Poly:
    return [a * k for a in p]


def conv_coeff_polys(X: dict[int, Poly], Y: dict[int, Poly]) -> dict[int, Poly]:
    """Coefficient dict of the product of two circulant operators (exact)."""
    out: dict[int, Poly] = {}
    for sa, pa in X.items():
        for sb, pb in Y.items():
            out[sa + sb] = _poly_add(out.get(sa + sb, [Fraction(0)]),
                                     _poly_mul(pa, pb))
    return {s: p for s, p in out.items() if not _poly_is_zero(p)}


def dissipation_expansion(polys: dict[int, Poly],
                          max_j: int = 5) -> tuple[int, Poly, list[Poly]]:
    """Exact small-theta expansion of 1 - |g(theta)|^2.

    Writing |g|^2 = sum_m ghat_m cos(m*theta) with
    ghat_0 = sum_s c_s^2 and ghat_m = 2 sum_s c_s c_{s+m} (m >= 1),
    consistency gives 1 - |g|^2 = sum_{m>=1} ghat_m (1 - cos m*theta),
    hence, with moment polynomials M_{2j}(lambda) = sum_m ghat_m m^{2j},

        1 - |g|^2 = M_2 th^2/2! - M_4 th^4/4! + M_6 th^6/6! - ...

    The fully discrete dissipative order is 2s = first 2j with M_{2j} != 0,
    and the leading coefficient A(lambda) = (-1)^{s+1} M_{2s}/(2s)! gives
    1 - |g|^2 = A(lambda) theta^{2s} + O(theta^{2s+2}), all exact in
    rational arithmetic.

    Returns (two_s, A_poly, [M_2, M_4, ...]).  two_s = 0 signals that every
    moment through 2*max_j vanishes (pure shift for all lambda).
    A negative leading value of A at some lambda means antidissipation
    (von Neumann instability), as for WENO5L with forward Euler.
    """
    shifts = sorted(polys)
    span = max(shifts) - min(shifts)
    moments: list[Poly] = []
    two_s, A_poly = 0, [Fraction(0)]
    for j in range(1, max_j + 1):
        M: Poly = [Fraction(0)]
        for m in range(1, span + 1):
            corr: Poly = [Fraction(0)]
            for s in shifts:
                if s + m in polys:
                    corr = _poly_add(corr, _poly_mul(polys[s], polys[s + m]))
            M = _poly_add(M, _poly_scale(corr, Fraction(2 * m ** (2 * j))))
        moments.append(M)
        if two_s == 0 and not _poly_is_zero(M):
            two_s = 2 * j
            fact = Fraction(math.factorial(2 * j))
            sign = Fraction((-1) ** (j + 1))
            A_poly = _poly_scale(M, sign / fact)
    return two_s, A_poly, moments


def amplitude_quotient(polys: dict[int, Poly], s: int) -> list[Poly]:
    """Exact Chebyshev quotient: 1 - |g(theta)|^2 = (1-x)^s Q(x, lam).

    Here x = cos(theta) and the return value is [q_0, ..., q_d] with
    Q(x, lam) = sum_k q_k(lam) x^k, each q_k an exact lambda-polynomial.
    Uses the autocorrelation form |g|^2 = rho_0 + 2 sum_m rho_m T_m(x)
    (Chebyshev polynomials T_m), then divides by (1-x)^s in exact
    rational arithmetic.  Raises if the division leaves a remainder
    (i.e., if 2s is not the dissipative order).
    """
    keys = sorted(polys)
    maxlag = keys[-1] - keys[0]
    rho: list[Poly] = []
    for m in range(maxlag + 1):
        acc: Poly = [Fraction(0)]
        for s_ in keys:
            if s_ + m in polys:
                acc = _poly_add(acc, _poly_mul(polys[s_], polys[s_ + m]))
        rho.append(acc)

    # Chebyshev T_m in the power basis (integer coefficients).
    cheb: list[list[int]] = [[1], [0, 1]]
    for _m in range(2, maxlag + 1):
        prev, cur = cheb[-2], cheb[-1]
        nxt = [0] + [2 * v for v in cur]
        nxt = [ci - (prev[i] if i < len(prev) else 0)
               for i, ci in enumerate(nxt)]
        cheb.append(nxt)

    # 1 - |g|^2 as a polynomial in x with lambda-polynomial coefficients.
    amp: list[Poly] = [[Fraction(0)] for _ in range(maxlag + 1)]
    amp[0] = _poly_add(amp[0], [Fraction(1)])
    amp[0] = _poly_add(amp[0], [-c for c in rho[0]])
    for m in range(1, maxlag + 1):
        for i, ti in enumerate(cheb[m]):
            if ti:
                amp[i] = _poly_add(amp[i], [-2 * ti * c for c in rho[m]])

    cur = amp
    for _ in range(s):
        # Divisibility by (1-x): the value at x = 1 must vanish identically.
        val1: Poly = [Fraction(0)]
        for p in cur:
            val1 = _poly_add(val1, p)
        if not _poly_is_zero(val1):
            raise ValueError("amplitude is not divisible by (1-x)^s")
        n = len(cur) - 1
        q: list[Poly] = [[] for _ in range(n)]
        qk: Poly = [Fraction(0)]
        for k in range(n, 0, -1):
            qk = _poly_add(qk, [-c for c in cur[k]])
            q[k - 1] = qk
        cur = q
    while len(cur) > 1 and _poly_is_zero(cur[-1]):
        cur = cur[:-1]
    return cur


def poly_to_string(p: Poly, var: str = "lam") -> str:
    """Human-readable exact polynomial."""
    terms = []
    for i, a in enumerate(p):
        if a == 0:
            continue
        if i == 0:
            terms.append(f"{a}")
        elif i == 1:
            terms.append(f"{a}*{var}")
        else:
            terms.append(f"{a}*{var}^{i}")
    return " + ".join(terms) if terms else "0"


# ----------------------------------------------------------------------
# SSP-RK3 pairing for the WENO5L spatial operator
# ----------------------------------------------------------------------

def weno5l_rk3_polys() -> dict[int, Poly]:
    """Exact coefficient polynomials of WENO5L advanced with SSP-RK3.

    For linear autonomous problems the Shu-Osher SSP-RK3 update reduces to
    the classical third-order stability polynomial: with A_FE = I + Z,
    Z = -lambda*Dtilde (Dtilde the constant flux-difference operator),

        A_RK3 = I + Z + Z^2/2 + Z^3/6 .

    The forward-Euler pairing is antidissipative (linearly unstable); this
    is the stable pairing used in the erasure experiments (E4/E5).
    """
    fe = coeff_polys("weno5l")
    # Z = A_FE - I: the coefficients of weno5l are affine in lambda, so Z
    # collects exactly the degree-1 parts.
    Z: dict[int, Poly] = {}
    for s, p in fe.items():
        lin = p[1] if len(p) > 1 else Fraction(0)
        if lin != 0:
            Z[s] = [Fraction(0), lin]
    I: dict[int, Poly] = {0: [Fraction(1)]}
    Z2 = conv_coeff_polys(Z, Z)
    Z3 = conv_coeff_polys(Z2, Z)
    out: dict[int, Poly] = {}
    for term, fac in ((I, Fraction(1)), (Z, Fraction(1)),
                      (Z2, Fraction(1, 2)), (Z3, Fraction(1, 6))):
        for s, p in term.items():
            out[s] = _poly_add(out.get(s, [Fraction(0)]),
                               _poly_scale(p, fac))
    return {s: p for s, p in out.items() if not _poly_is_zero(p)}


def weno5l_ssprk3(lam: float) -> Coeffs:
    """Float coefficients of the WENO5L + SSP-RK3 fully discrete update."""
    return eval_polys(weno5l_rk3_polys(), lam)


# ----------------------------------------------------------------------
# First-order node velocities (class-node geometry as lambda -> 0)
# ----------------------------------------------------------------------

def node_velocities(name: str, N: int) -> np.ndarray:
    """mu_k = dg_k/dlambda at lambda=0, computed exactly from coeff_polys.

    As lambda -> 0 every symbol satisfies g_k = 1 + lambda*mu_k + O(lambda^2),
    so intra-class node separations (which control the conditioning of the
    Vandermonde class blocks) are lambda*|mu_k - mu_k'| to leading order.
    A scheme whose mu(theta) is degenerate on some mode pair suffers
    second-order near-collisions and early effective-rank loss.
    """
    polys = coeff_polys(name)
    mu = np.zeros(N, dtype=complex)
    for s, p in polys.items():
        c1 = float(p[1]) if len(p) > 1 else 0.0
        mu += c1 * np.exp(-2j * np.pi * np.arange(N) * s / N)
    return mu


# ----------------------------------------------------------------------
# Single-parent (local) observability
# ----------------------------------------------------------------------

def rho_formula(m_L: int, m_R: int, r: int, L: int) -> int:
    """Closed form for the single-parent rank rho(L) (conj:local-rank).

    One new scalar per step for each parent currently receiving mass:
    ceil(t*m_L/r) downstream and ceil(t*m_R/r) upstream parents at time t,
    capped at the saturated value r-1.  Verified against measured ranks for
    all schemes at r in {6, 8} (experiment E1c).
    """
    total = sum(-(-t * m_L // r) + -(-t * m_R // r) for t in range(1, L + 1))
    return min(r - 1, total)


def local_basis(P: int, r: int) -> np.ndarray:
    """Basis of the zero-mean subspace of parent 0, shape (P*r, r-1)."""
    V = np.zeros((P * r, r - 1), dtype=float)
    for j in range(1, r):
        V[0, j - 1] = -1.0
        V[j, j - 1] = 1.0
    return V


def local_observability_matrix(coeffs: Coeffs, P: int, r: int,
                               L: int) -> np.ndarray:
    """O_L restricted to zero-mean perturbations of a single parent."""
    return build_O(coeffs, P, r, L) @ local_basis(P, r)


# ----------------------------------------------------------------------
# Self-tests
# ----------------------------------------------------------------------

def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def self_test() -> None:
    """Validate coefficients, orders, and matrix builders."""
    lams = [1.0, 0.9, 0.5, 0.1, 0.01, 0.001]

    for name, (fn, _order, _label) in SCHEMES.items():
        for lam in lams:
            c = fn(lam)
            _check(abs(sum(c.values()) - 1.0) < 1.0e-13,
                   f"{name}: coefficients do not sum to 1 at lambda={lam}")

    # Semi-Lagrangian moment conditions: sum_s c_s s^m = lambda^m, m <= degree.
    sl_cases = {"uw1": 1, "lw": 2, "bw": 2, "fromm": 2, "ub3": 3, "ub5": 5}
    for name, degree in sl_cases.items():
        fn = SCHEMES[name][0]
        for lam in lams:
            c = fn(lam)
            for mdeg in range(degree + 1):
                mom = sum(cs * s ** mdeg for s, cs in c.items())
                _check(abs(mom - lam ** mdeg) < 1.0e-11,
                       f"{name}: moment {mdeg} fails at lambda={lam}")

    # Closed forms: UW1 and LW.
    lam = 0.37
    c = uw1(lam)
    _check(abs(c[0] - (1 - lam)) < 1e-14 and abs(c[1] - lam) < 1e-14,
           "uw1 closed form")
    c = lax_wendroff(lam)
    _check(abs(c[1] - 0.5 * lam * (1 + lam)) < 1e-14, "lw c_1 closed form")
    _check(abs(c[0] - (1 - lam ** 2)) < 1e-14, "lw c_0 closed form")
    _check(abs(c[-1] + 0.5 * lam * (1 - lam)) < 1e-14, "lw c_-1 closed form")

    # Pure-shift endpoints at lambda = 1 (semi-Lagrangian family).
    for name in ("uw1", "lw", "bw", "fromm", "ub3", "ub5"):
        c = SCHEMES[name][0](1.0)
        for s, cs in c.items():
            target = 1.0 if s == 1 else 0.0
            _check(abs(cs - target) < 1e-12, f"{name}: lambda=1 not a shift")

    # Circulant matrix consistency: symbol equals eigenvalue on a Fourier mode.
    N = 24
    c = ub3(0.6)
    A = circulant_matrix(c, N)
    k = 5
    v = np.exp(2j * np.pi * k * np.arange(N) / N)
    _check(np.allclose(A @ v, symbol(c, k, N) * v, atol=1e-12),
           "circulant matrix does not match symbol")

    # UW1 contraction factor matches Paper A's rho_N.
    lamr, Nr = 0.5, 64
    rho_paperA = math.sqrt(1 - 4 * lamr * (1 - lamr) * math.sin(math.pi / Nr) ** 2)
    _check(abs(contraction_factor(uw1(lamr), Nr) - rho_paperA) < 1e-13,
           "uw1 contraction factor mismatch")

    # Exact coefficient polynomials agree with the float constructors.
    for name, (fn, _order, _label) in SCHEMES.items():
        polys = coeff_polys(name)
        for lam in (0.37, 0.9, 0.005):
            cf = fn(lam)
            cp = eval_polys(polys, lam)
            _check(set(cf) == set(cp) and
                   all(abs(cf[s] - cp[s]) < 1e-13 for s in cf),
                   f"{name}: coeff_polys disagrees with constructor")

    # Local basis: columns are zero-mean and supported in parent 0.
    Vb = local_basis(4, 5)
    _check(np.allclose(Vb.sum(axis=0), 0.0), "local basis not zero-mean")
    _check(np.allclose(Vb[5:], 0.0), "local basis leaks outside parent 0")

    # Exact dissipation expansions against known closed forms.
    #   UW1: 1 - |g|^2 = 4 lam (1-lam) sin^2(th/2)      -> 2s = 2, A = lam(1-lam)
    #   LW : 1 - |g|^2 = 4 lam^2 (1-lam^2) sin^4(th/2)  -> 2s = 4, A = lam^2(1-lam^2)/4
    two_s, A, _ = dissipation_expansion(coeff_polys("uw1"))
    _check(two_s == 2 and A == [Fraction(0), Fraction(1), Fraction(-1)],
           "uw1 dissipation expansion")
    two_s, A, _ = dissipation_expansion(coeff_polys("lw"))
    _check(two_s == 4 and
           A == [Fraction(0), Fraction(0), Fraction(1, 4),
                 Fraction(0), Fraction(-1, 4)],
           "lw dissipation expansion")

    # Flux weights: closed forms and one-step reconstruction.
    lam = 0.37
    w = flux_weights(uw1(lam), lam)
    _check(set(w) == {0} and abs(w[0] - 1.0) < 1e-13, "uw1 flux weights")
    w = flux_weights(lax_wendroff(lam), lam)
    _check(abs(w[0] - (1 + lam) / 2) < 1e-13 and
           abs(w[-1] - (1 - lam) / 2) < 1e-13, "lw flux weights")
    w = flux_weights(beam_warming(lam), lam)
    _check(abs(w[0] - (3 - lam) / 2) < 1e-13 and
           abs(w[1] + (1 - lam) / 2) < 1e-13, "bw flux weights")
    rng_f = np.random.default_rng(7)
    uf = rng_f.standard_normal(48)
    for name, (fn, _order, _label) in SCHEMES.items():
        for lam in (0.9, 0.5, 0.01):
            cf = fn(lam)
            F = apply_flux(flux_weights(cf, lam), uf)
            recon = uf - lam * (F - np.roll(F, 1))
            _check(np.allclose(recon, circulant_matrix(cf, 48) @ uf,
                               atol=1e-12),
                   f"{name}: flux form does not reproduce the update")
    cf = weno5l_ssprk3(0.5)
    F = apply_flux(flux_weights(cf, 0.5), uf)
    recon = uf - 0.5 * (F - np.roll(F, 1))
    _check(np.allclose(recon, circulant_matrix(cf, 48) @ uf, atol=1e-12),
           "weno5l+rk3: flux form does not reproduce the update")

    # WENO5L + SSP-RK3: fully discrete order 4 with A = lam^4/12 exactly
    # (integrator-set: the spatial operator alone is 6th order dissipative).
    rk3 = weno5l_rk3_polys()
    for lam in (0.37, 0.9):
        cf = eval_polys(rk3, lam)
        _check(abs(sum(cf.values()) - 1.0) < 1e-12, "rk3 consistency")
        mom1 = sum(cs * s for s, cs in cf.items())
        _check(abs(mom1 - lam) < 1e-11, "rk3 first moment")
    two_s, A, _ = dissipation_expansion(rk3, max_j=4)
    _check(two_s == 4, "rk3 dissipative order")
    _check(len(A) >= 5 and A[4] == Fraction(1, 12) and
           all(a == 0 for a in A[:4]), "rk3 leading coefficient lam^4/12")

    # Exact interior-stability factorizations: 1-|g|^2 = (1-x)^s Q(x,lam),
    # x = cos(theta).  These certify |g(theta)| < 1 on theta in (0,2pi) for
    # all lam in (0,1) (see the strict-interior-stability proposition).
    def _prod(*factors):
        out: Poly = [Fraction(1)]
        for f in factors:
            out = _poly_mul(out, [Fraction(c) for c in f])
        return out

    def _q_at(Q, xv):
        acc: Poly = [Fraction(0)]
        for k in range(len(Q) - 1, -1, -1):
            acc = _poly_add([c * xv for c in acc], Q[k])
        return acc

    def _peq(p, q):
        return _poly_is_zero(_poly_add(p, [-c for c in q]))

    Q = amplitude_quotient(coeff_polys("uw1"), 1)
    _check(len(Q) == 1 and _peq(Q[0], _prod([0, 2], [1, -1])),
           "uw1 quotient 2 lam (1-lam)")
    Q = amplitude_quotient(coeff_polys("lw"), 2)
    _check(len(Q) == 1 and _peq(Q[0], _prod([0, 0, 1], [1, -1], [1, 1])),
           "lw quotient lam^2 (1-lam^2)")
    Q = amplitude_quotient(coeff_polys("bw"), 2)
    _check(len(Q) == 1 and
           _peq(Q[0], _prod([0, 1], [1, -1], [1, -1], [2, -1])),
           "bw quotient lam (1-lam)^2 (2-lam)")
    Q = amplitude_quotient(coeff_polys("fromm"), 2)
    _check(len(Q) == 2, "fromm quotient linear in x")
    _check(_peq(_q_at(Q, Fraction(1)), _prod([0, 1], [1, -1], [1, -1, 1])),
           "fromm Q(1) = lam(1-lam)(1-lam+lam^2)")
    _check(_peq(_q_at(Q, Fraction(-1)), _prod([0, 1], [1, -1])),
           "fromm Q(-1) = lam(1-lam)")
    Q = amplitude_quotient(coeff_polys("ub3"), 2)
    _check(len(Q) == 2, "ub3 quotient linear in x")
    _check(_peq(_q_at(Q, Fraction(1)),
                _poly_scale(_prod([0, 1], [1, -1], [1, 1], [2, -1]),
                            Fraction(1, 3))),
           "ub3 Q(1) = lam(1-lam)(1+lam)(2-lam)/3")
    _check(_peq(_q_at(Q, Fraction(-1)),
                _poly_scale(_prod([0, 1], [1, -1], [1, 1], [2, -1],
                                  [3, -2], [1, 2]), Fraction(1, 9))),
           "ub3 Q(-1) = lam(1-lam)(1+lam)(2-lam)(3-2lam)(1+2lam)/9")
    Q = amplitude_quotient(coeff_polys("ub5"), 3)
    _check(len(Q) == 3, "ub5 quotient quadratic in x")
    _check(_peq(Q[2],
                _poly_scale(_prod([0, 0, 1], [1, -1], [1, -1], [1, 1],
                                  [1, 1], [2, -1], [2, -1], [2, 1],
                                  [3, -1]), Fraction(1, 450))),
           "ub5 q2 = lam^2(1-lam)^2(1+lam)^2(2-lam)^2(2+lam)(3-lam)/450")
    _check(_peq(_poly_add(Q[1], _poly_scale(Q[2], Fraction(2))),
                _poly_scale(_prod([0, 0, 1], [1, -1], [1, -1], [1, 1],
                                  [2, -1], [2, 1], [3, -1]),
                            Fraction(-1, 60))),
           "ub5 Q_x(1) = -lam^2(1-lam)^2(1+lam)(2-lam)(2+lam)(3-lam)/60")
    _check(_peq(_q_at(Q, Fraction(1)),
                _poly_scale(_prod([0, 1], [1, -1], [1, 1], [2, -1],
                                  [2, 1], [3, -1]), Fraction(1, 45))),
           "ub5 Q(1) = 8A = lam(1-lam^2)(4-lam^2)(3-lam)/45")
    Qr = amplitude_quotient(weno5l_rk3_polys(), 2)
    _check(_peq(_q_at(Qr, Fraction(1)),
                [Fraction(0)] * 4 + [Fraction(1, 3)]),
           "rk3 pairing Q(1) = 4A = lam^4/3")

    # Exact coincidences at lambda = 1/2 (manuscript
    # rem:coefficient-structure (ii)-(iii)):
    # (a) FR - UB3 = [lam(1-lam)(1-2lam)/12] * (S^-1 - 3I + 3S - S^2),
    #     so the two operators are identical at lam in {0, 1/2, 1}.
    d3 = {-1: 1.0, 0: -3.0, 1: 3.0, 2: -1.0}
    for lam in [0.25, 0.5, 0.7]:
        fr, u3 = fromm(lam), ub3(lam)
        k = lam * (1.0 - lam) * (1.0 - 2.0 * lam) / 12.0
        for s, w in d3.items():
            _check(abs(fr.get(s, 0.0) - u3.get(s, 0.0) - k * w) < 1e-14,
                   f"FR - UB3 third-difference identity at lambda={lam}")
    _check(max(abs(fromm(0.5).get(s, 0.0) - ub3(0.5).get(s, 0.0))
               for s in d3) == 0.0,
           "FR and UB3 are the same operator at lambda=1/2")
    # (b) LW and BW share the full amplitude spectrum at lambda = 1/2
    #     (reflection duality), hence identical E2 energy histories.
    _check(max(abs(abs(symbol(lax_wendroff(0.5), k, 64))
                   - abs(symbol(beam_warming(0.5), k, 64)))
               for k in range(64)) < 1e-14,
           "LW and BW amplitude spectra coincide at lambda=1/2")

    print("schemes.py self-test: all checks passed.")


if __name__ == "__main__":
    self_test()

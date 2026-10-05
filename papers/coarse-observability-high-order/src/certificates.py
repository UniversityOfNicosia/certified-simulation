"""Exact rational-arithmetic certificates (REVISION_1 items M1 and M3).

Both certificates reduce a strict-positivity claim to finitely many exact
rational operations -- proof by exhaustive rational arithmetic.  No
floating-point number enters any accepted step; the only approximate
objects are *certified rational interval enclosures* (endpoints are
`Fraction`s, and the enclosed quantity provably lies between them).

1.  Collision-set emptiness (M1; experiment E1b, exact part).
    For a scheme with exact coefficient polynomials c_s(lambda) (rational
    coefficients, `schemes.coeff_polys`) and a mode pair k != k' in the
    same residue class mod P on the N = P*r grid, the symbol difference

        delta(lambda) = g_k(lambda) - g_{k'}(lambda)
                      = sum_s c_s(lambda) (w^{-ks} - w^{-k's}),
        w = exp(2*pi*i/N),

    is a polynomial in lambda with coefficients in the cyclotomic field
    Q(w), represented exactly in the group ring Q[Z_N] (dict: exponent
    mod N -> Fraction).  Consistency forces delta(0) = 0; the exact
    lambda-adic valuation a is computed with zero tests modulo the
    cyclotomic polynomial Phi_N, and delta_tilde = delta / lambda^a.
    The squared modulus

        q(lambda) = delta_tilde(lambda) * conj(delta_tilde)(lambda)

    (conj = the group-ring involution v -> -v, complex conjugation on
    Q(w)) is a lambda-polynomial whose coefficients are *real* elements
    of Q(w).  A collision at lambda in (0, 1] is exactly a root of q
    there.  We certify q > 0 on the whole closed interval [0, 1]:

      * each group-ring coefficient is enclosed in a rational interval
        using certified enclosures of cos(2*pi*v/N) (Taylor series with
        explicit remainder bounds; pi enclosed via Machin's formula with
        alternating-series tail bounds; dyadic rounding for speed, with
        the rounding error added to the enclosure width);
      * the Bernstein coefficients of q on [0, 1] are formed by interval
        arithmetic (all conversion weights are nonnegative, so interval
        propagation is tight);
      * all Bernstein lower bounds positive on a cell certifies q > 0
        there (convex-hull property); otherwise the cell is split by
        exact de Casteljau subdivision.

    The certified minimum (min over leaf cells of the smallest Bernstein
    lower bound) is a rigorous uniform lower bound for q on [0, 1].

2.  Strict interior stability of WENO5L+SSP-RK3 (M3; experiment E4e).
    The exact Chebyshev quotient Q(x, lambda) of the pairing,
    1 - |g|^2 = (1-x)^2 Q with x = cos(theta), satisfies the exact
    bivariate identity (verified in rational arithmetic here)

        Q(x, lambda) = lambda * [ (1-x) * S(x, lambda) + lambda^3 / 3 ],
        S(x, 0) = 4/15,   deg_x S = 12,   deg_lambda S = 5.

    Two exact tensor-Bernstein certificates finish the proof:

        S > 0        on [-1, 1] x [0, 1/2],
        Q/lambda > 0 on [-1, 1] x [1/2, 1].

    For lambda in (0, 1/2] the identity gives Q >= lambda^4/3 > 0
    directly (both summands nonnegative, the second positive); for
    lambda in [1/2, 1] positivity is certified directly.  Hence
    |g(theta)| < 1 for all theta in (0, 2*pi) at every fixed
    lambda in (0, 1].

Coefficient input is exact (`Fraction`) throughout; see
`schemes.coeff_polys`, `schemes.weno5l_rk3_polys`,
`schemes.amplitude_quotient`.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

import schemes as sch

F = Fraction

# ----------------------------------------------------------------------
# Rational interval arithmetic
# ----------------------------------------------------------------------

Interval = tuple[Fraction, Fraction]


def _iv_add(a: Interval, b: Interval) -> Interval:
    return (a[0] + b[0], a[1] + b[1])


def _iv_scale(a: Interval, c: Fraction) -> Interval:
    return (a[0] * c, a[1] * c) if c >= 0 else (a[1] * c, a[0] * c)


def _iv_avg(a: Interval, b: Interval) -> Interval:
    half = F(1, 2)
    return ((a[0] + b[0]) * half, (a[1] + b[1]) * half)


def _dyadic(x: Fraction, bits: int) -> Fraction:
    """Nearest fraction with denominator 2**bits (error <= 2**-(bits+1))."""
    scaled = x * (1 << bits)
    return F(round(scaled), 1 << bits)


def pi_interval(prec_bits: int) -> Interval:
    """Certified enclosure of pi via Machin's formula.

    pi = 16 arctan(1/5) - 4 arctan(1/239); each arctan series is
    alternating with decreasing terms, so the truncation error is
    bounded by the first omitted term, with known sign.
    """
    def scaled_atan_inv(q: int, scale: int) -> Interval:
        x2 = F(1, q * q)
        term = F(1, q)
        total = F(0)
        k = 0
        sign = 1
        tol = F(1, 1 << prec_bits)
        while True:
            total += F(sign) * term / (2 * k + 1)
            k += 1
            sign = -sign
            term *= x2
            nxt = term / (2 * k + 1)
            if nxt < tol:
                # next omitted term has sign `sign`
                lo = total if sign > 0 else total - nxt
                hi = total + nxt if sign > 0 else total
                return (F(scale) * lo, F(scale) * hi)

    a = scaled_atan_inv(5, 16)
    b = scaled_atan_inv(239, 4)
    return (a[0] - b[1], a[1] - b[0])


def cos_table(N: int, prec_bits: int = 120) -> list[Interval]:
    """Certified enclosures of cos(2*pi*v/N) for v = 0..N-1.

    Reduces to y = 2*pi*v'/N with v' <= N/2 (cos symmetry), rounds y to
    a dyadic rational (rounding error added to the enclosure width via
    the Lipschitz bound |cos y - cos y'| <= |y - y'|), and evaluates the
    cosine Taylor series with the explicit alternating remainder bound
    y^(2K)/(2K)!.
    """
    piv = pi_interval(prec_bits + 16)
    out: list[Interval] = []
    for v in range(N):
        vp = min(v % N, N - (v % N))            # cos(2 pi v/N) = cos(2 pi vp/N)
        y_lo = 2 * piv[0] * vp / N
        y_hi = 2 * piv[1] * vp / N              # y in [0, pi]
        y_mid = _dyadic((y_lo + y_hi) / 2, prec_bits + 8)
        # |y - y_mid| <= half enclosure width + dyadic rounding error
        slack = (y_hi - y_lo) / 2 + F(1, 1 << (prec_bits + 8))
        y2 = y_mid * y_mid
        K = 40
        total = F(0)
        term = F(1)                              # y_mid^(2k) / (2k)!
        for k in range(K):
            total += term if k % 2 == 0 else -term
            term = term * y2 / ((2 * k + 1) * (2 * k + 2))
            term = _dyadic(term, 4 * prec_bits) if term else term
        # dyadic rounding of `term` above only perturbs the *remainder
        # bound*; make it safe by doubling it.
        err = 2 * term + slack
        out.append((total - err, total + err))
    return out


# ----------------------------------------------------------------------
# Cyclotomic group-ring arithmetic  (elements of Q(w_N) as Q[Z_N] dicts)
# ----------------------------------------------------------------------

#: Cyclotomic polynomials Phi_N for the grids used in the paper,
#: low-to-high coefficient lists.  Verified in `self_test`.
CYCLOTOMIC: dict[int, list[int]] = {
    48: [1] + [0] * 7 + [-1] + [0] * 7 + [1],    # x^16 - x^8 + 1
    64: [1] + [0] * 31 + [1],                    # x^32 + 1
}

GRElem = dict[int, Fraction]


def gr_mul(a: GRElem, b: GRElem, N: int) -> GRElem:
    out: GRElem = {}
    for va, ca in a.items():
        for vb, cb in b.items():
            v = (va + vb) % N
            s = out.get(v, F(0)) + ca * cb
            if s:
                out[v] = s
            elif v in out:
                del out[v]
    return out


def gr_conj(a: GRElem, N: int) -> GRElem:
    return {(-v) % N: c for v, c in a.items()}


def gr_reduce(a: GRElem, N: int) -> list[Fraction]:
    """Reduce to the canonical basis 1, w, ..., w^(phi(N)-1) mod Phi_N."""
    phi = CYCLOTOMIC[N]
    deg = len(phi) - 1
    dense = [F(0)] * N
    for v, c in a.items():
        dense[v % N] += c
    # polynomial remainder mod Phi_N (monic)
    for i in range(N - 1, deg - 1, -1):
        c = dense[i]
        if c:
            dense[i] = F(0)
            for j, pj in enumerate(phi[:-1]):
                if pj:
                    dense[i - deg + j] -= c * pj
    return dense[:deg]


def gr_is_zero(a: GRElem, N: int) -> bool:
    return all(c == 0 for c in gr_reduce(a, N))


def gr_eval_real_iv(a: GRElem, tab: list[Interval]) -> Interval:
    """Enclosure of a *real* group-ring element: sum_v c_v cos(2 pi v/N)."""
    iv: Interval = (F(0), F(0))
    for v, c in a.items():
        iv = _iv_add(iv, _iv_scale(tab[v], c))
    return iv


# ----------------------------------------------------------------------
# Interval Bernstein positivity on [0, 1]  (univariate, M1)
# ----------------------------------------------------------------------

def _bernstein_from_power_iv(coeffs: list[Interval]) -> list[Interval]:
    """Power-basis interval coefficients -> Bernstein interval coefficients.

    b_j = sum_{i<=j} [C(j,i)/C(d,i)] c_i with nonnegative weights, so the
    interval combination is exact (no dependency widening).
    """
    d = len(coeffs) - 1
    out: list[Interval] = []
    for j in range(d + 1):
        acc: Interval = (F(0), F(0))
        for i in range(j + 1):
            w = F(comb(j, i), comb(d, i))
            acc = _iv_add(acc, _iv_scale(coeffs[i], w))
        out.append(acc)
    return out


def _decasteljau_split_iv(
        bern: list[Interval]) -> tuple[list[Interval], list[Interval]]:
    work = list(bern)
    left = [work[0]]
    right = [work[-1]]
    n = len(work)
    for _ in range(n - 1):
        work = [_iv_avg(work[i], work[i + 1]) for i in range(len(work) - 1)]
        left.append(work[0])
        right.append(work[-1])
    right.reverse()
    return left, right


def bernstein_positive_iv(coeffs: list[Interval],
                          max_depth: int = 24) -> tuple[bool, Fraction, int]:
    """Certify a real polynomial (interval power coefficients) > 0 on [0,1].

    Returns (certified, certified_lower_bound, leaf_cells).  The lower
    bound is rigorous: min over leaves of the smallest Bernstein lower
    endpoint (convex-hull property).
    """
    root = _bernstein_from_power_iv(coeffs)
    stack = [(root, 0)]
    lo_min: Fraction | None = None
    cells = 0
    while stack:
        bern, depth = stack.pop()
        lo = min(b[0] for b in bern)
        if lo > 0:
            cells += 1
            lo_min = lo if lo_min is None else min(lo_min, lo)
            continue
        if depth >= max_depth:
            return (False, F(0), cells)
        left, right = _decasteljau_split_iv(bern)
        stack.append((left, depth + 1))
        stack.append((right, depth + 1))
    assert lo_min is not None
    return (True, lo_min, cells)


# ----------------------------------------------------------------------
# M1: exact collision-set emptiness certificate
# ----------------------------------------------------------------------

def collision_certificate_exact(name: str, P: int, r: int,
                                tab: list[Interval] | None = None,
                                ) -> dict:
    """Certify q = |delta_tilde|^2 > 0 on [0, 1] for every class pair.

    Returns a summary dict with the number of pairs, the certified
    uniform lower bound over all pairs (rigorous rational, reported as
    float), total leaf cells, and a list of failures (empty on success).
    """
    N = P * r
    if tab is None:
        tab = cos_table(N)
    polys = sch.coeff_polys(name)
    deg = max(len(p) for p in polys.values())

    n_pairs = 0
    min_bound: Fraction | None = None
    max_cells = 0
    failures: list[tuple] = []

    for m in range(1, P):
        ks = [m + j * P for j in range(r)]
        for ai in range(r):
            for bi in range(ai + 1, r):
                k, kp = ks[ai], ks[bi]
                n_pairs += 1
                # delta(lambda) coefficients in the group ring
                delta: list[GRElem] = [{} for _ in range(deg)]
                for s, p in polys.items():
                    for i, c in enumerate(p):
                        if c:
                            d = delta[i]
                            for kk, sign in ((k, c), (kp, -c)):
                                v = (-kk * s) % N
                                t = d.get(v, F(0)) + sign
                                if t:
                                    d[v] = t
                                elif v in d:
                                    del d[v]
                # exact lambda-adic valuation
                a = 0
                while a < len(delta) and gr_is_zero(delta[a], N):
                    a += 1
                if a == len(delta):
                    failures.append((m, k, kp, "identically_zero"))
                    continue
                dt = delta[a:]
                dtc = [gr_conj(c, N) for c in dt]
                # q = dt * conj(dt): lambda-convolution
                q: list[GRElem] = [{} for _ in range(2 * len(dt) - 1)]
                for i, ci in enumerate(dt):
                    for j, cj in enumerate(dtc):
                        prod = gr_mul(ci, cj, N)
                        tgt = q[i + j]
                        for v, c in prod.items():
                            t = tgt.get(v, F(0)) + c
                            if t:
                                tgt[v] = t
                            elif v in tgt:
                                del tgt[v]
                # interval coefficients (real elements by construction)
                civ = [gr_eval_real_iv(c, tab) for c in q]
                ok, lo, cells = bernstein_positive_iv(civ)
                if not ok:
                    failures.append((m, k, kp, "not_certified"))
                    continue
                max_cells = max(max_cells, cells)
                min_bound = lo if min_bound is None else min(min_bound, lo)

    return {
        "scheme": name, "P": P, "r": r, "n_pairs": n_pairs,
        "certified": not failures,
        "min_bound": float(min_bound) if min_bound is not None else None,
        "max_cells": max_cells, "failures": failures,
    }


# ----------------------------------------------------------------------
# Exact bivariate Bernstein positivity  (rational coefficients, M3)
# ----------------------------------------------------------------------

Biv = list[list[Fraction]]        # [i][j] = coeff of lambda^i x^j


def _biv_subst_box(p: Biv, x_dom: tuple[Fraction, Fraction],
                   l_dom: tuple[Fraction, Fraction]) -> Biv:
    """Exact substitution x = x0 + (x1-x0) u, lambda = l0 + (l1-l0) v.

    Maps the box x_dom x l_dom to the unit square in (u, v).
    """
    x0, x1 = x_dom
    l0, l1 = l_dom
    dx, dl = x1 - x0, l1 - l0
    nl = len(p)
    nx = max(len(row) for row in p)
    out: Biv = [[F(0)] * nx for _ in range(nl)]
    # x^j = sum_t C(j,t) x0^(j-t) dx^t u^t ; lambda^i analogous
    xpow = [[F(comb(j, t)) * x0 ** (j - t) * dx ** t for t in range(j + 1)]
            for j in range(nx)]
    lpow = [[F(comb(i, t)) * l0 ** (i - t) * dl ** t for t in range(i + 1)]
            for i in range(nl)]
    for i, row in enumerate(p):
        for j, c in enumerate(row):
            if c:
                for ti, wi in enumerate(lpow[i]):
                    if wi:
                        for tj, wj in enumerate(xpow[j]):
                            if wj:
                                out[ti][tj] += c * wi * wj
    return out


def _bern2d_from_power(p: Biv) -> Biv:
    """Tensor Bernstein coefficients on the unit square."""
    m = len(p) - 1                                # lambda-degree
    n = max(len(row) for row in p) - 1            # x-degree
    b: Biv = [[F(0)] * (n + 1) for _ in range(m + 1)]
    for kl in range(m + 1):
        for kx in range(n + 1):
            acc = F(0)
            for i in range(kl + 1):
                wl = F(comb(kl, i), comb(m, i))
                for j in range(min(kx, len(p[i]) - 1) + 1):
                    c = p[i][j]
                    if c:
                        acc += wl * F(comb(kx, j), comb(n, j)) * c
            b[kl][kx] = acc
    return b


def _split_rows(b: Biv) -> tuple[Biv, Biv]:
    """de Casteljau split along the lambda axis (rows)."""
    cols = len(b[0])
    left: Biv = []
    right: Biv = []
    work = [row[:] for row in b]
    left.append(work[0][:])
    right.append(work[-1][:])
    half = F(1, 2)
    for _ in range(len(b) - 1):
        work = [[(work[i][j] + work[i + 1][j]) * half for j in range(cols)]
                for i in range(len(work) - 1)]
        left.append(work[0][:])
        right.append(work[-1][:])
    right.reverse()
    return left, right


def _transpose(b: Biv) -> Biv:
    return [list(col) for col in zip(*b)]


def bern2d_positive(power: Biv, x_dom: tuple[Fraction, Fraction],
                    l_dom: tuple[Fraction, Fraction],
                    max_depth: int = 24) -> tuple[bool, Fraction, int]:
    """Certify a rational bivariate polynomial > 0 on a box.

    Exact tensor-Bernstein with adaptive de Casteljau subdivision
    (alternating axes).  Returns (certified, lower_bound, leaf_cells).
    """
    root = _bern2d_from_power(_biv_subst_box(power, x_dom, l_dom))
    stack: list[tuple[Biv, int]] = [(root, 0)]
    lo_min: Fraction | None = None
    cells = 0
    while stack:
        b, depth = stack.pop()
        lo = min(min(row) for row in b)
        if lo > 0:
            cells += 1
            lo_min = lo if lo_min is None else min(lo_min, lo)
            continue
        if depth >= max_depth:
            return (False, F(0), cells)
        if depth % 2 == 0:
            l_, r_ = _split_rows(b)
        else:
            bt = _transpose(b)
            lt, rt = _split_rows(bt)
            l_, r_ = _transpose(lt), _transpose(rt)
        stack.append((l_, depth + 1))
        stack.append((r_, depth + 1))
    assert lo_min is not None
    return (True, lo_min, cells)


# ----------------------------------------------------------------------
# M3: certified strict interior stability of WENO5L+SSP-RK3
# ----------------------------------------------------------------------

def rk3_quotient_decomposition() -> tuple[Biv, Biv]:
    """Exact decomposition Q = lambda [ (1-x) S + lambda^3/3 ] of the pairing.

    Returns (Qt, S) as bivariate coefficient tables [i][j] ~ lambda^i x^j
    with Qt = Q/lambda.  The identity Qt = (1-x) S + lambda^3/3 and the
    boundary values Qt(1, lambda) = lambda^3/3, S(x, 0) = 4/15 are
    verified exactly; any failure raises.
    """
    Q = sch.amplitude_quotient(sch.weno5l_rk3_polys(), 2)
    deg_l = max(len(q) for q in Q) - 1
    # lambda-adic valuation 1: constant lambda-terms must vanish
    if any(q[0] != 0 for q in Q if q):
        raise AssertionError("pairing quotient: lambda-valuation != 1")
    Qt: Biv = [[(Q[j][i + 1] if i + 1 < len(Q[j]) else F(0))
                for j in range(len(Q))] for i in range(deg_l)]
    # T = Qt - lambda^3/3 must be divisible by (1 - x)
    T = [row[:] for row in Qt]
    T[3][0] -= F(1, 3)

    def div_1mx(p: list[Fraction]) -> list[Fraction]:
        n = len(p) - 1
        q = [F(0)] * max(n, 1)
        if n >= 1:
            q[n - 1] = -p[n]
            for t in range(n - 2, -1, -1):
                q[t] = q[t + 1] - p[t + 1]
        if p[0] != q[0]:
            raise AssertionError("pairing quotient: (1-x) division fails")
        return q

    S: Biv = [div_1mx(row) if any(c != 0 for c in row) else [F(0)]
              for row in T]
    if any(c != 0 for j, c in enumerate(S[0]) if j > 0) or S[0][0] != F(4, 15):
        raise AssertionError("pairing quotient: S(x,0) != 4/15")
    # verify the identity Qt == (1-x) S + lambda^3/3 exactly
    nx = max(len(row) for row in Qt)
    recon: Biv = [[F(0)] * nx for _ in range(len(Qt))]
    for i, row in enumerate(S):
        for j, c in enumerate(row):
            if c:
                recon[i][j] += c
                recon[i][j + 1] -= c
    recon[3][0] += F(1, 3)
    for i in range(len(Qt)):
        for j in range(nx):
            got = recon[i][j]
            want = Qt[i][j] if j < len(Qt[i]) else F(0)
            if got != want:
                raise AssertionError("pairing quotient: identity fails")
    return Qt, S


def stability_certificate() -> dict:
    """Run both Bernstein certificates for the pairing; return summary."""
    Qt, S = rk3_quotient_decomposition()
    one, half = F(1), F(1, 2)
    ok_s, lo_s, cells_s = bern2d_positive(S, (-one, one), (F(0), half))
    ok_q, lo_q, cells_q = bern2d_positive(Qt, (-one, one), (half, one))
    return {
        "S_positive_on_low_lambda": ok_s, "S_lower_bound": float(lo_s),
        "S_cells": cells_s,
        "Qt_positive_on_high_lambda": ok_q, "Qt_lower_bound": float(lo_q),
        "Qt_cells": cells_q,
        "certified": ok_s and ok_q,
    }


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------

def self_test() -> None:
    import math

    def _check(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(f"certificates self-test failed: {msg}")

    # pi enclosure (reference: 36-digit decimal expansion, error < 1e-36)
    pi_ref = F("3.141592653589793238462643383279502884")
    piv = pi_interval(80)
    _check(piv[0] - F(1, 10**30) < pi_ref < piv[1] + F(1, 10**30)
           and piv[1] - piv[0] < F(1, 1 << 70), "pi interval")

    # cyclotomic polynomials divide x^N - 1 and vanish at the primitive root
    for N, phi in CYCLOTOMIC.items():
        rem = [F(0)] * N
        rem[0] -= 1
        dense = rem[:]              # x^N - 1 as remainder workspace
        dense = [F(0)] * (N + 1)
        dense[0], dense[N] = F(-1), F(1)
        deg = len(phi) - 1
        for i in range(N, deg - 1, -1):
            c = dense[i]
            if c:
                dense[i] = F(0)
                for j, pj in enumerate(phi[:-1]):
                    if pj:
                        dense[i - deg + j] -= c * pj
        _check(all(c == 0 for c in dense), f"Phi_{N} divides x^{N}-1")
        w = complex(math.cos(2 * math.pi / N), math.sin(2 * math.pi / N))
        val = sum(pj * w ** j for j, pj in enumerate(phi))
        _check(abs(val) < 1e-9, f"Phi_{N}(w_{N}) = 0")

    # cos table against floating point
    for N in (48, 64):
        tab = cos_table(N, prec_bits=80)
        for v in range(N):
            c = math.cos(2 * math.pi * v / N)
            lo, hi = tab[v]
            _check(float(lo) - 1e-12 <= c <= float(hi) + 1e-12,
                   f"cos enclosure N={N} v={v}")
            _check(hi - lo < F(1, 1 << 60), f"cos width N={N} v={v}")

    # group ring: (w^1 + w^-1)^2 reduces consistently and evaluates to
    # (2 cos(2 pi/N))^2
    N = 48
    tab = cos_table(N, prec_bits=80)
    a: GRElem = {1: F(1), N - 1: F(1)}
    sq = gr_mul(a, a, N)
    iv = gr_eval_real_iv(sq, tab)
    truth = (2 * math.cos(2 * math.pi / N)) ** 2
    _check(float(iv[0]) <= truth <= float(iv[1]), "group-ring square")

    # zero test: 1 + w^(N/2) = 0
    z: GRElem = {0: F(1), N // 2: F(1)}
    _check(gr_is_zero(z, N), "1 + w^(N/2) = 0")
    _check(not gr_is_zero(a, N), "w + w^-1 != 0")

    # univariate Bernstein: (lambda - 2)^2 > 0 on [0,1]; margin >= 1
    coeffs = [(F(4), F(4)), (F(-4), F(-4)), (F(1), F(1))]
    ok, lo, _ = bernstein_positive_iv(coeffs)
    _check(ok and lo >= 1, "univariate Bernstein")
    # ... and (lambda - 1/2)^2 is nonnegative with a zero: must NOT certify
    coeffs = [(F(1, 4), F(1, 4)), (F(-1), F(-1)), (F(1), F(1))]
    ok, _, _ = bernstein_positive_iv(coeffs, max_depth=8)
    _check(not ok, "Bernstein rejects touching zero")

    # bivariate Bernstein: x^2 + lambda + 1/10 > 0 on [-1,1]x[0,1]
    p: Biv = [[F(1, 10), F(0), F(1)], [F(1), F(0), F(0)]]
    ok, lo, _ = bern2d_positive(p, (F(-1), F(1)), (F(0), F(1)))
    _check(ok and lo >= F(1, 10), "bivariate Bernstein")

    # the pairing decomposition identity itself
    rk3_quotient_decomposition()

    print("certificates.py self-test passed")


if __name__ == "__main__":
    self_test()

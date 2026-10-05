#!/usr/bin/env python3
"""Independent exact certificate: dissipation ladder, interior stability,
conditioning exponents, and erasure rates for the paper "Coarse Observability
Across High-Order Finite-Volume Reconstructions" (Polemitis, Kokkinakis,
Christakis, Drikakis).

Every scheme coefficient is derived from the manuscript text alone (Lagrange
nodes of Appendix A.1, Fromm as the LW/BW mean, the WENO5 linear-weights flux
of Appendix A.2 with forward Euler, and the Shu-Osher SSP-RK3 stages).  The
script replays, with exact rational arithmetic or rigorous outward-rounded
interval enclosures:

* def:diss-order, lem:amplitude, prop:dissipation: the theta-series of
  1-|g|^2 by two independent routes (autocorrelation moments and the direct
  product of the symbol's Taylor series), 2s and A(lambda) for all eight
  operators, against the closed forms and erasure_orders.csv;
* tab:scheme-zoo: fully discrete consistency orders and dissipative orders;
* rem:coefficient-structure (i)-(iii) and the modified-equation constants;
* prop:interior-stability: the (1-x)^s Q factorizations and positivity;
* prop:weno-fe: forward-Euler antidissipation, the interior maximum of |g|,
  the SSP-RK3 order 2s=4 with A = lam^4/12, and the fifth-order flux symbol;
* prop:rk3-stability: the decomposition and both tensor-Bernstein bounds,
  against stability_certificate.csv;
* conj:conditioning: exact lambda-adic Smith exponents of the polynomial
  matrix O_L iota_K (which fix the asymptotic exponent of sigma+_min) for all
  seven single-stage operators at (P,r)=(8,6),(8,8), the exclusion of
  WENO5-LIN+SSP-RK3, exact Sturm brackets of sigma+_min at small lambda,
  and comparisons with queue_conditioning*.csv and tab:queue-exponents;
* prop:contraction-general and E4c: rigorous 1-rho_N and the ratio
  (1-rho_N)/[A theta_1^{2s}/2] for N = 24..384, against erasure_rates.csv
  and erasure_rate_fits.csv, including the UB5 lambda=0.9, N=384 cell;
* tab:half-life: information half-lives at N=48, lambda=1/2.

The script is self-contained, deterministic, and uses only the Python
standard library.  It never reads the authors' code; their published outputs
are embedded as transcribed reference data, which the companion test confirms
against the published files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction as Q
from math import comb, factorial, gcd, isqrt
from pathlib import Path
from typing import Any

type Poly = tuple[Q, ...]
type Operator = dict[int, Poly]
type Interval = tuple[int, int]
type Bivariate = list[Poly]

# ----------------------------------------------------------------------------
# Univariate polynomials in lambda (coefficient tuples, low degree first).


def ptrim(p: Any) -> Poly:
    out = list(p)
    while out and out[-1] == 0:
        out.pop()
    return tuple(Q(x) for x in out)


def padd(a: Poly, b: Poly) -> Poly:
    n = max(len(a), len(b))
    return ptrim(
        (a[i] if i < len(a) else Q(0)) + (b[i] if i < len(b) else Q(0)) for i in range(n)
    )


def psub(a: Poly, b: Poly) -> Poly:
    return padd(a, tuple(-x for x in b))


def pmul(a: Poly, b: Poly) -> Poly:
    if not a or not b:
        return ()
    out = [Q(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if y:
                    out[i + j] += x * y
    return ptrim(out)


def pscale(a: Poly, c: Q) -> Poly:
    return ptrim(c * x for x in a)


def ppow(a: Poly, n: int) -> Poly:
    out: Poly = (Q(1),)
    for _ in range(n):
        out = pmul(out, a)
    return out


def peval(p: Poly, x: Q) -> Q:
    acc = Q(0)
    for coeff in reversed(p):
        acc = acc * x + coeff
    return acc


def pcompose(p: Poly, q: Poly) -> Poly:
    acc: Poly = ()
    for coeff in reversed(p):
        acc = padd(pmul(acc, q), (coeff,) if coeff else ())
    return acc


def pderiv(p: Poly) -> Poly:
    return ptrim(i * p[i] for i in range(1, len(p)))


def pdivmod(a: Poly, b: Poly) -> tuple[Poly, Poly]:
    a_list = list(a)
    if len(a_list) < len(b):
        return (), ptrim(a_list)
    quot = [Q(0)] * (len(a_list) - len(b) + 1)
    lead = b[-1]
    for i in range(len(a_list) - len(b), -1, -1):
        coeff = a_list[i + len(b) - 1] / lead
        quot[i] = coeff
        if coeff:
            for j, y in enumerate(b):
                a_list[i + j] -= coeff * y
    return ptrim(quot), ptrim(a_list[: len(b) - 1])


def sturm_sequence(p: Poly) -> list[Poly]:
    p = ptrim(p)
    seq = [p, pderiv(p)]
    while len(seq[-1]) > 1:
        _, rem = pdivmod(seq[-2], seq[-1])
        if not rem:
            break
        seq.append(pscale(rem, Q(-1)))
    return seq


def sign_changes(seq: list[Poly], x: Q) -> int:
    values = [peval(f, x) for f in seq]
    values = [v for v in values if v != 0]
    return sum(1 for a, b in zip(values, values[1:]) if (a > 0) != (b > 0))


def pgcd(a: Poly, b: Poly) -> Poly:
    a, b = ptrim(a), ptrim(b)
    while b:
        _, rem = pdivmod(a, b)
        a, b = b, rem
    return pscale(a, 1 / a[-1]) if a else ()


def squarefree(p: Poly) -> Poly:
    """p / gcd(p, p'): same distinct roots, all simple."""
    g = pgcd(p, pderiv(p))
    return pdivmod(p, g)[0] if len(g) > 1 else ptrim(p)


def sturm_roots_in(p: Poly, lo: Q, hi: Q) -> int:
    """Number of distinct real roots of p in (lo, hi] (Sturm's theorem on the
    square-free part, valid when lo or hi is itself a root)."""
    p = squarefree(ptrim(p))
    if len(p) <= 1:
        return 0
    seq = sturm_sequence(p)
    return sign_changes(seq, lo) - sign_changes(seq, hi)


def roots_in_open_unit_interval(p: Poly) -> int:
    p = ptrim(p)
    while p and p[0] == 0:
        p = p[1:]
    if not p:
        raise ValueError("zero polynomial")
    return sturm_roots_in(p, Q(0), Q(1)) - (1 if peval(p, Q(1)) == 0 else 0)


def positive_on_open_unit_interval(p: Poly) -> bool:
    return roots_in_open_unit_interval(p) == 0 and peval(p, Q(1, 2)) > 0


def poly_from_factors(scale: Q, factors: list[tuple[Poly, int]]) -> Poly:
    out: Poly = (scale,)
    for factor, power in factors:
        out = pmul(out, ppow(factor, power))
    return out


LAM: Poly = (Q(0), Q(1))
ONE_P: Poly = (Q(1),)


def lin(a: int, b: int) -> Poly:
    """a + b*lam."""
    return ptrim((Q(a), Q(b)))


def poly_text(p: Poly) -> str:
    if not p:
        return "0"
    return " + ".join(
        (f"{c}" if i == 0 else f"{c}*lam^{i}") for i, c in enumerate(p) if c
    )


# ----------------------------------------------------------------------------
# Operators (Laurent polynomials in S, (S^s u)_i = u_{i-s}), from the text.


def op_add(a: Operator, b: Operator) -> Operator:
    out = dict(a)
    for s, p in b.items():
        out[s] = padd(out.get(s, ()), p)
    return {s: p for s, p in sorted(out.items()) if p}


def op_scale(a: Operator, c: Q) -> Operator:
    return {s: pscale(p, c) for s, p in sorted(a.items()) if pscale(p, c)}


def op_mul(a: Operator, b: Operator) -> Operator:
    out: dict[int, Poly] = {}
    for s, p in a.items():
        for t, q in b.items():
            out[s + t] = padd(out.get(s + t, ()), pmul(p, q))
    return {s: p for s, p in sorted(out.items()) if p}


IDENTITY: Operator = {0: (Q(1),)}


def lagrange_operator(nodes: tuple[int, ...]) -> Operator:
    ops: Operator = {}
    for d in nodes:
        poly: Poly = (Q(1),)
        for dp in nodes:
            if dp != d:
                poly = pmul(poly, (Q(-dp, d - dp), Q(-1, d - dp)))
        ops[-d] = poly
    return dict(sorted(ops.items()))


WENO5_FLUX = {2: Q(2, 60), 1: Q(-13, 60), 0: Q(47, 60), -1: Q(27, 60), -2: Q(-3, 60)}


def weno5_flux_difference() -> dict[int, Q]:
    return {
        s: WENO5_FLUX.get(s, Q(0)) - WENO5_FLUX.get(s - 1, Q(0))
        for s in range(-3, 4)
        if WENO5_FLUX.get(s, Q(0)) - WENO5_FLUX.get(s - 1, Q(0))
    }


def weno5_forward_euler() -> Operator:
    diff = weno5_flux_difference()
    ops: Operator = {}
    for s in range(-3, 4):
        poly = ptrim(((Q(1) if s == 0 else Q(0)), -diff.get(s, Q(0))))
        if poly:
            ops[s] = poly
    return ops


def ssp_rk3(fe: Operator) -> Operator:
    stage2 = op_add(op_scale(IDENTITY, Q(3, 4)), op_scale(op_mul(fe, fe), Q(1, 4)))
    return op_add(op_scale(IDENTITY, Q(1, 3)), op_scale(op_mul(fe, stage2), Q(2, 3)))


SCHEME_ORDER = ("uw1", "lw", "bw", "fromm", "ub3", "ub5", "weno5l", "weno5l_rk3")
SINGLE_STAGE = ("uw1", "lw", "bw", "fromm", "ub3", "ub5", "weno5l")
STABLE = ("uw1", "lw", "bw", "fromm", "ub3", "ub5", "weno5l_rk3")


def build_operators() -> dict[str, Operator]:
    lw = lagrange_operator((-1, 0, 1))
    bw = lagrange_operator((-2, -1, 0))
    fe = weno5_forward_euler()
    return {
        "uw1": lagrange_operator((-1, 0)),
        "lw": lw,
        "bw": bw,
        "fromm": op_scale(op_add(lw, bw), Q(1, 2)),
        "ub3": lagrange_operator((-2, -1, 0, 1)),
        "ub5": lagrange_operator((-3, -2, -1, 0, 1, 2)),
        "weno5l": fe,
        "weno5l_rk3": ssp_rk3(fe),
    }


def at_lambda(op: Operator, lam: Q) -> dict[int, Q]:
    values = {s: peval(p, lam) for s, p in op.items()}
    return {s: v for s, v in sorted(values.items()) if v != 0}


# ----------------------------------------------------------------------------
# Dissipation: autocorrelations, moments, theta-series.


def autocorrelations(op: Operator) -> dict[int, Poly]:
    """g^_0 = sum c_s^2 and g^_m = 2 sum_s c_s c_{s+m} (lem:amplitude)."""
    width = max(op) - min(op)
    out: dict[int, Poly] = {}
    total: Poly = ()
    for s, p in op.items():
        total = padd(total, pmul(p, p))
    out[0] = total
    for m in range(1, width + 1):
        acc: Poly = ()
        for s, p in op.items():
            if s + m in op:
                acc = padd(acc, pmul(p, op[s + m]))
        out[m] = pscale(acc, Q(2))
    return out


def defect_series_by_moments(op: Operator, order: int) -> list[Poly]:
    """Coefficients of theta^{2j}, j=1..order, of 1-|g|^2 via the moments M_{2j}."""
    gm = autocorrelations(op)
    out = []
    for j in range(1, order + 1):
        moment: Poly = ()
        for m, p in gm.items():
            if m >= 1:
                moment = padd(moment, pscale(p, Q(m ** (2 * j))))
        out.append(pscale(moment, Q((-1) ** (j + 1), factorial(2 * j))))
    return out


def defect_series_direct(op: Operator, order: int) -> list[Poly]:
    """Same coefficients from g(theta) = sum_n (-i theta)^n mu_n/n!, |g|^2 = Re^2 + Im^2."""
    top = 2 * order
    mu = []
    for n in range(top + 1):
        acc: Poly = ()
        for s, p in op.items():
            acc = padd(acc, pscale(p, Q(s**n)))
        mu.append(acc)
    real = [()] * (top + 1)
    imag = [()] * (top + 1)
    for n in range(top + 1):
        coeff = pscale(mu[n], Q(1, factorial(n)))
        # (-i)^n: n%4 = 0 -> 1, 1 -> -i, 2 -> -1, 3 -> i
        phase = n % 4
        if phase == 0:
            real[n] = coeff
        elif phase == 1:
            imag[n] = pscale(coeff, Q(-1))
        elif phase == 2:
            real[n] = pscale(coeff, Q(-1))
        else:
            imag[n] = coeff
    square: list[Poly] = [()] * (top + 1)
    for i in range(top + 1):
        for j in range(top + 1 - i):
            square[i + j] = padd(square[i + j], padd(pmul(real[i], real[j]), pmul(imag[i], imag[j])))
    if psub(ONE_P, square[0]) != ():
        raise AssertionError("|g(0)|^2 != 1")
    for n in range(1, top + 1, 2):
        if square[n] != ():
            raise AssertionError("odd theta power in |g|^2")
    return [pscale(square[2 * j], Q(-1)) for j in range(1, order + 1)]


def dissipative_order(series: list[Poly]) -> tuple[int, Poly]:
    for j, coeff in enumerate(series, start=1):
        if coeff:
            return 2 * j, coeff
    raise AssertionError("no nonzero moment within the computed order")


def consistency_order(op: Operator, limit: int = 8) -> int:
    """Largest p with g(theta) - exp(-i lam theta) = O(theta^{p+1}) identically."""
    for n in range(limit + 1):
        mu: Poly = ()
        for s, p in op.items():
            mu = padd(mu, pscale(p, Q(s**n)))
        if mu != ppow(LAM, n):
            return n - 1
    return limit


# ----------------------------------------------------------------------------
# Amplitude as a polynomial in x = cos(theta), coefficients in Q[lambda].


def chebyshev(m: int) -> list[int]:
    t0, t1 = [1], [0, 1]
    if m == 0:
        return t0
    for _ in range(m - 1):
        nxt = [0] + [2 * c for c in t1]
        for i, c in enumerate(t0):
            nxt[i] -= c
        t0, t1 = t1, nxt
    return t1


def amplitude_defect_in_x(op: Operator) -> Bivariate:
    """F(x, lam) = 1 - |g|^2 = 1 - g^_0 - sum_m g^_m T_m(x)."""
    gm = autocorrelations(op)
    width = max(gm)
    out: list[Poly] = [()] * (width + 1)
    out[0] = ONE_P
    for m, p in gm.items():
        for i, c in enumerate(chebyshev(m)):
            if c:
                out[i] = psub(out[i], pscale(p, Q(c)))
    while out and not out[-1]:
        out.pop()
    return out


def divide_by_one_minus_x(f: Bivariate) -> tuple[Bivariate, Poly]:
    """f = (1 - x) g + remainder, remainder = f(1)."""
    n = len(f) - 1
    b: list[Poly] = [()] * n
    acc: Poly = ()
    for i in range(n, 0, -1):
        acc = padd(acc, f[i])
        b[i - 1] = acc
    remainder = padd(acc, f[0])
    return [pscale(p, Q(-1)) for p in b], remainder


def bivariate_eval_x(f: Bivariate, x: Q) -> Poly:
    acc: Poly = ()
    for p in reversed(f):
        acc = padd(pscale(acc, x), p)
    return acc


def bivariate_deriv_x(f: Bivariate) -> Bivariate:
    return [pscale(f[i], Q(i)) for i in range(1, len(f))]


def tensor_bernstein_min(f: Bivariate, x_box: tuple[Q, Q], l_box: tuple[Q, Q]) -> tuple[Q, int, int]:
    """Minimum Bernstein coefficient of f on the box, with its degrees (n in x, m in lam)."""
    x0, x1 = x_box
    l0, l1 = l_box
    n = len(f) - 1
    m = max(len(p) for p in f) - 1
    lam_sub = ptrim((l0, l1 - l0))
    columns = [pcompose(p, lam_sub) for p in f]  # in v
    x_sub = ptrim((x0, x1 - x0))
    coeff = [[Q(0)] * (m + 1) for _ in range(n + 1)]  # coeff[i][j] u^i v^j
    for i, col in enumerate(columns):
        power = ppow(x_sub, i)
        for a, cu in enumerate(power):
            if cu:
                for j, cv in enumerate(col):
                    if cv:
                        coeff[a][j] += cu * cv
    best = None
    for k in range(n + 1):
        for ell in range(m + 1):
            total = Q(0)
            for i in range(k + 1):
                wi = Q(comb(k, i), comb(n, i))
                for j in range(ell + 1):
                    if coeff[i][j]:
                        total += wi * Q(comb(ell, j), comb(m, j)) * coeff[i][j]
            if best is None or total < best:
                best = total
    assert best is not None
    return best, n, m


# ----------------------------------------------------------------------------
# Rigorous fixed-point intervals, pi, cos/sin, square roots, logarithms.

BITS = 320
ONE = 1 << BITS


def _ceil_div(a: int, b: int) -> int:
    return -((-a) // b)


def _pi_bounds() -> Interval:
    def atan_inverse(n: int, terms: int) -> tuple[Q, Q]:
        partial = Q(0)
        sums = []
        for k in range(terms):
            term = Q(1, (2 * k + 1) * n ** (2 * k + 1))
            partial = partial + term if k % 2 == 0 else partial - term
            sums.append(partial)
        return min(sums[-2], sums[-1]), max(sums[-2], sums[-1])

    lo5, hi5 = atan_inverse(5, 90)
    lo239, hi239 = atan_inverse(239, 30)
    lo = 16 * lo5 - 4 * hi239
    hi = 16 * hi5 - 4 * lo239
    return (lo.numerator << BITS) // lo.denominator, _ceil_div(hi.numerator << BITS, hi.denominator)


PI = _pi_bounds()


def iv_add(x: Interval, y: Interval) -> Interval:
    return (x[0] + y[0], x[1] + y[1])


def iv_scale(x: Interval, q: Q) -> Interval:
    n, d = q.numerator, q.denominator
    if n >= 0:
        return ((x[0] * n) // d, _ceil_div(x[1] * n, d))
    return ((x[1] * n) // d, _ceil_div(x[0] * n, d))


def iv_fraction(x: Interval) -> tuple[Q, Q]:
    return Q(x[0], ONE), Q(x[1], ONE)


_TRIG: dict[tuple[int, int], tuple[Interval, Interval]] = {}


def cos_sin_turn(v: int, n: int) -> tuple[Interval, Interval]:
    """Enclosures of cos(2 pi v/n), sin(2 pi v/n): Taylor series at an exact
    fixed-point angle with Lagrange remainder plus the Lipschitz margin of the
    angle enclosure."""
    key = (v % n, n)
    if key in _TRIG:
        return _TRIG[key]
    w = v % n
    mirrored = 2 * w > n
    if mirrored:
        w = n - w
    a_lo = (2 * w * PI[0]) // n
    a_hi = _ceil_div(2 * w * PI[1], n)
    x = a_lo
    term: Interval = (ONE, ONE)
    c_lo = c_hi = s_lo = s_hi = 0
    k = 0
    while True:
        phase = k % 4
        if phase == 0:
            c_lo += term[0]
            c_hi += term[1]
        elif phase == 1:
            s_lo += term[0]
            s_hi += term[1]
        elif phase == 2:
            c_lo -= term[1]
            c_hi -= term[0]
        else:
            s_lo -= term[1]
            s_hi -= term[0]
        k += 1
        term = ((term[0] * x) // (k * ONE), _ceil_div(term[1] * x, k * ONE))
        if k > 8 and term[1] <= 1:
            break
    margin = term[1] + 2 + (a_hi - a_lo)
    cos_iv = (max(c_lo - margin, -ONE), min(c_hi + margin, ONE))
    sin_iv = (max(s_lo - margin, -ONE), min(s_hi + margin, ONE))
    if mirrored:
        sin_iv = (-sin_iv[1], -sin_iv[0])
    _TRIG[key] = (cos_iv, sin_iv)
    return cos_iv, sin_iv


def theta_one(n: int) -> tuple[Q, Q]:
    return Q(2 * PI[0], n * ONE), Q(2 * PI[1], n * ONE)


def sqrt_bounds(lo: Q, hi: Q) -> tuple[Q, Q]:
    """Enclosure of sqrt on [lo, hi] (0 <= lo <= hi) on the 2^-BITS grid."""
    a = (lo.numerator << (2 * BITS)) // lo.denominator
    b = _ceil_div(hi.numerator << (2 * BITS), hi.denominator)
    s_lo = isqrt(a)
    s_hi = isqrt(b)
    if s_hi * s_hi < b:
        s_hi += 1
    return Q(s_lo, ONE), Q(s_hi, ONE)


LN_BITS = 220
LN_ONE = 1 << LN_BITS
LN_TERMS = 72


def _atanh_fixed(t_lo: int, t_hi: int) -> tuple[int, int]:
    """atanh(t) = sum_k t^{2k+1}/(2k+1) on the 2^-LN_BITS grid for 0 <= t <= 1/3,
    with floor/ceil rounding and the tail bound t^{2K+1}/((2K+1)(1-t^2)) <=
    (9/8) t^{2K+1}/(2K+1)."""
    total_lo, term = 0, t_lo
    square = (t_lo * t_lo) >> LN_BITS
    for k in range(LN_TERMS):
        total_lo += term // (2 * k + 1)
        term = (term * square) >> LN_BITS
    total_hi, term = 0, t_hi
    square = _ceil_div(t_hi * t_hi, LN_ONE)
    for k in range(LN_TERMS):
        total_hi += _ceil_div(term, 2 * k + 1)
        term = _ceil_div(term * square, LN_ONE)
    total_hi += _ceil_div(9 * term, 8 * (2 * LN_TERMS + 1)) + 1
    return total_lo, total_hi


_LN2_HALF = _atanh_fixed(LN_ONE // 3, _ceil_div(LN_ONE, 3))
LN2 = (Q(2 * _LN2_HALF[0], LN_ONE), Q(2 * _LN2_HALF[1], LN_ONE))
_LN_CACHE: dict[Q, tuple[Q, Q]] = {}


def ln_bounds(q: Q) -> tuple[Q, Q]:
    """Rigorous enclosure of ln(q), q > 0, via ln q = e ln 2 + 2 atanh((y-1)/(y+1))
    with y = q 2^-e in [1, 2)."""
    if q <= 0:
        raise ValueError("ln of a nonpositive number")
    if q in _LN_CACHE:
        return _LN_CACHE[q]
    num, den = q.numerator, q.denominator
    e = num.bit_length() - den.bit_length()
    if e >= 0:
        den <<= e
    else:
        num <<= -e
    if num < den:
        num <<= 1
        e -= 1
    # y = num/den in [1, 2); t = (y-1)/(y+1) = (num-den)/(num+den) in [0, 1/3)
    t_num, t_den = num - den, num + den
    t_lo = (t_num * LN_ONE) // t_den
    t_hi = _ceil_div(t_num * LN_ONE, t_den)
    a_lo, a_hi = _atanh_fixed(t_lo, t_hi)
    base_lo, base_hi = (e * LN2[0], e * LN2[1]) if e >= 0 else (e * LN2[1], e * LN2[0])
    result = (base_lo + Q(2 * a_lo, LN_ONE), base_hi + Q(2 * a_hi, LN_ONE))
    _LN_CACHE[q] = result
    return result


def ln_mid(q: Q) -> Q:
    lo, hi = ln_bounds(q)
    return (lo + hi) / 2


def decimal_text(q: Q, digits: int = 15) -> str:
    """Deterministic decimal rendering with `digits` significant digits (round half up)."""
    if q == 0:
        return "0"
    sign = "-" if q < 0 else ""
    num, den = abs(q.numerator), q.denominator

    def at_least_power(e: int) -> bool:  # num/den >= 10^e
        return num >= den * 10**e if e >= 0 else num * 10 ** (-e) >= den

    exponent = ((num.bit_length() - den.bit_length()) * 30103) // 100000
    while not at_least_power(exponent):
        exponent -= 1
    while at_least_power(exponent + 1):
        exponent += 1
    shift = digits - 1 - exponent
    if shift >= 0:
        mantissa = (2 * num * 10**shift + den) // (2 * den)
    else:
        mantissa = (2 * num + den * 10 ** (-shift)) // (2 * den * 10 ** (-shift))
    if mantissa >= 10**digits:
        mantissa //= 10
        exponent += 1
    text = str(mantissa)
    return f"{sign}{text[0]}.{text[1:]}e{exponent:+d}"


# ----------------------------------------------------------------------------
# Conditioning: polynomial single-parent observation matrices.


def laurent_powers(op: Operator, top: int) -> list[Operator]:
    powers = [IDENTITY]
    for _ in range(top):
        powers.append(op_mul(powers[-1], op))
    return powers


def local_polynomial_blocks(
    powers: list[Operator], parents: int, children: int, horizon: int, host: int = 0
) -> list[list[list[Poly]]]:
    """blocks[t][K'][j-1] = (R A^t iota_K (e_j - e_0))_{K'} as a polynomial in lambda."""
    size = parents * children
    blocks = []
    for t in range(horizon + 1):
        arrive = [[() for _ in range(children)] for _ in range(parents)]
        for j in range(children):
            for k, p in powers[t].items():
                dest = ((host * children + j + k) % size) // children
                arrive[dest][j] = padd(arrive[dest][j], p)
        rows = [
            [pscale(psub(arrive[kk][j], arrive[kk][0]), Q(1, children)) for j in range(1, children)]
            for kk in range(parents)
        ]
        blocks.append(rows)
    return blocks


def _trunc(p: Poly, order: int) -> list[Q]:
    return list(p[:order]) + [Q(0)] * (order - min(len(p), order))


def _ser_mul(a: list[Q], b: list[Q], order: int) -> list[Q]:
    out = [Q(0)] * order
    for i in range(min(order, len(a))):
        x = a[i]
        if x:
            for j in range(min(order - i, len(b))):
                if b[j]:
                    out[i + j] += x * b[j]
    return out


def _ser_inv(u: list[Q], order: int) -> list[Q]:
    inv = [Q(0)] * order
    inv[0] = 1 / u[0]
    for n in range(1, order):
        acc = Q(0)
        for k in range(1, min(n, len(u) - 1) + 1):
            if u[k]:
                acc += u[k] * inv[n - k]
        inv[n] = -acc * inv[0]
    return inv


def _valuation(series: list[Q]) -> int | None:
    for i, x in enumerate(series):
        if x:
            return i
    return None


def smith_exponents(rows: list[list[Poly]], columns: int, order: int) -> list[int]:
    """Exponents of the invariant factors of a polynomial matrix over the local
    ring Q[lam]_(lam), computed exactly in Q[lam]/(lam^order) by elimination with
    minimal-valuation pivots; exponents below `order` are exact."""
    mat = [[_trunc(p, order) for p in row] for row in rows]
    live_rows = list(range(len(mat)))
    live_cols = list(range(columns))
    exponents = []
    while live_rows and live_cols:
        best = None
        for i in live_rows:
            for j in live_cols:
                v = _valuation(mat[i][j])
                if v is not None and (best is None or v < best[0]):
                    best = (v, i, j)
        if best is None:
            break
        v, pi, pj = best
        exponents.append(v)
        inverse = _ser_inv(mat[pi][pj][v:], order - v)
        for i in live_rows:
            if i == pi or _valuation(mat[i][pj]) is None:
                continue
            factor = _ser_mul(mat[i][pj][v:], inverse, order - v)
            for j in live_cols:
                product = _ser_mul(factor, mat[pi][j], order)
                mat[i][j] = [a - b for a, b in zip(mat[i][j], product)]
        live_rows.remove(pi)
        live_cols.remove(pj)
    return sorted(exponents)


def exact_rank(matrix: list[list[Q]]) -> int:
    pivots: list[tuple[int, list[Q]]] = []
    for row in matrix:
        vec = list(row)
        for col, prow in pivots:
            if vec[col]:
                f = vec[col]
                vec = [a - f * b for a, b in zip(vec, prow)]
        lead = next((j for j, x in enumerate(vec) if x), None)
        if lead is not None:
            inv = 1 / vec[lead]
            pivots.append((lead, [x * inv for x in vec]))
    return len(pivots)


def charpoly(c: list[list[Q]]) -> list[Q]:
    """Faddeev-LeVerrier: coefficients of det(zI - C), low degree first."""
    n = len(c)
    coeffs = [Q(0)] * (n + 1)
    coeffs[n] = Q(1)
    mk = [[Q(0)] * n for _ in range(n)]
    for k in range(1, n + 1):
        mk = [
            [sum((c[i][l] * mk[l][j] for l in range(n)), Q(0)) + (coeffs[n - k + 1] if i == j else 0) for j in range(n)]
            for i in range(n)
        ]
        trace = sum((sum((c[i][l] * mk[l][i] for l in range(n)), Q(0)) for i in range(n)), Q(0))
        coeffs[n - k] = -trace / k
    return coeffs


def charpoly_int(c: list[list[int]]) -> list[int]:
    """Faddeev-LeVerrier over Z (every division by k is exact for integer C)."""
    n = len(c)
    coeffs = [0] * (n + 1)
    coeffs[n] = 1
    mk = [[0] * n for _ in range(n)]
    for k in range(1, n + 1):
        mk = [
            [sum(c[i][l] * mk[l][j] for l in range(n)) + (coeffs[n - k + 1] if i == j else 0) for j in range(n)]
            for i in range(n)
        ]
        trace = sum(sum(c[i][l] * mk[l][i] for l in range(n)) for i in range(n))
        assert trace % k == 0
        coeffs[n - k] = -trace // k
    return coeffs


def _int_content_primitive(p: list[int]) -> list[int]:
    g = 0
    for x in p:
        g = gcd(g, x)
    return [x // g for x in p] if g > 1 else p


def _int_positive_remainder(a: list[int], b: list[int]) -> list[int]:
    """A positive multiple of (a mod b) by integer pseudo-division."""
    a = list(a)
    lead_b = b[-1]
    sign_b = 1 if lead_b > 0 else -1
    degree_b = len(b) - 1
    while a and len(a) - 1 >= degree_b:
        lead_a = a[-1]
        a = [x * abs(lead_b) for x in a]
        shift = len(a) - 1 - degree_b
        for j, y in enumerate(b):
            a[shift + j] -= lead_a * sign_b * y
        a.pop()
        while a and a[-1] == 0:
            a.pop()
    return a


def int_sturm_sequence(p: list[int]) -> list[list[int]]:
    """Sturm sequence over Z, each member a positive multiple of the classical one."""
    deriv = [i * p[i] for i in range(1, len(p))]
    seq = [_int_content_primitive(p), _int_content_primitive(deriv)]
    while len(seq[-1]) > 1:
        rem = _int_positive_remainder(seq[-2], seq[-1])
        if not rem:
            break
        seq.append(_int_content_primitive([-x for x in rem]))
    return seq


def _int_values(seq: list[list[int]], num: int, shift: int) -> list[int]:
    """Values of the sequence at x = num * 2^-shift, each times 2^(shift*deg) > 0."""
    out = []
    for poly in seq:
        acc = 0
        for i in range(len(poly) - 1, -1, -1):
            acc = acc * num + (poly[i] << (shift * (len(poly) - 1 - i)))
        out.append(acc)
    return out


def _variations(values: list[int]) -> int:
    signs = [v > 0 for v in values if v]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def smallest_positive_root_int(coeffs: list[int], bisections: int) -> tuple[int, Q, Q]:
    """For an integer polynomial with only real roots >= 0: the multiplicity of
    the root 0 and a dyadic bracket (lo, hi] of the smallest positive root r_1.

    r_1 lies in [e_m/e_{m-1}, m e_m/e_{m-1}], read from the two lowest nonzero
    coefficients.  The predicate "r_1 <= x" is decided exactly: it holds if x is
    a root, and otherwise by Sturm's theorem V(0) - V(x) >= 1, which counts
    distinct roots in (0, x] for any x that is not a root, square-free or not."""
    p = list(coeffs)
    zeros = 0
    while p and p[0] == 0:
        p.pop(0)
        zeros += 1
    degree = len(p) - 1
    p0, p1 = abs(p[0]), abs(p[1])
    seq = int_sturm_sequence(p)
    exponent = p0.bit_length() - p1.bit_length()
    shift = max(0, 64 - exponent) + 40
    h_lo = (p0 << shift) // p1
    h_hi = _ceil_div(p0 << shift, p1)
    lo = h_lo - (h_lo >> 40) - 1
    hi = h_hi * degree
    base = _variations(_int_values(seq, 0, shift))

    def at_or_beyond_first_root(num: int, sh: int) -> bool:
        values = _int_values(seq, num, sh)
        return values[0] == 0 or base - _variations(values) >= 1

    assert not at_or_beyond_first_root(lo, shift)
    assert at_or_beyond_first_root(hi, shift)
    for _ in range(bisections):
        lo, hi, shift = 2 * lo, 2 * hi, shift + 1
        mid = (lo + hi) // 2
        if at_or_beyond_first_root(mid, shift):
            hi = mid
        else:
            lo = mid
    return zeros, Q(lo, 1 << shift), Q(hi, 1 << shift)


def gram_bracket(rows: list[list[Q]], bisections: int = 44) -> tuple[int, Q, Q, Q]:
    """sigma+_min^2 bracket of the coordinate matrix (rows x (r-1)), its nullity,
    and the squared Frobenius norm, computed over Z after clearing denominators."""
    scale = 1
    for row in rows:
        for x in row:
            scale = scale * x.denominator // gcd(scale, x.denominator)
    m = [[int(x * scale) for x in row] for row in rows]
    n = len(m[0])
    g = [[sum(row[i] * row[j] for row in m) for j in range(n)] for i in range(n)]
    zeros, lo, hi = smallest_positive_root_int(charpoly_int(g), bisections)
    square = scale * scale
    return zeros, lo / square, hi / square, Q(sum(g[i][i] for i in range(n)), square)


def metric_bracket(rows: list[list[Q]], children: int) -> tuple[Q, Q]:
    """sigma+_min^2 in the induced Euclidean norm: eigenvalues of H^{-1} G with
    H = B^T B = I + J for the basis e_j - e_0, i.e. of (r I - J) G / r."""
    scale = 1
    for row in rows:
        for x in row:
            scale = scale * x.denominator // gcd(scale, x.denominator)
    m = [[int(x * scale) for x in row] for row in rows]
    n = len(m[0])
    g = [[sum(row[i] * row[j] for row in m) for j in range(n)] for i in range(n)]
    column_sums = [sum(g[k][j] for k in range(n)) for j in range(n)]
    c = [[children * g[i][j] - column_sums[j] for j in range(n)] for i in range(n)]
    _, lo, hi = smallest_positive_root_int(charpoly_int(c), 44)
    square = scale * scale * children
    return lo / square, hi / square


# ----------------------------------------------------------------------------
# Erasure: rigorous 1 - rho_N.


def amplitude_defect_interval(gm: dict[int, Q], k: int, n: int) -> Interval:
    acc: Interval = (0, 0)
    for m, value in gm.items():
        if m == 0 or value == 0:
            continue
        cos_iv, _ = cos_sin_turn(m * k, n)
        acc = iv_add(acc, iv_scale((ONE - cos_iv[1], ONE - cos_iv[0]), value))
    return acc


def defect_by_theta_series(op: Operator, lam: Q, n: int, terms: int = 30) -> tuple[Q, Q]:
    """Second route to D(theta_1) = 1 - |g(theta_1)|^2: the series
    sum_j (-1)^{j+1} M_{2j} theta^{2j}/(2j)! with exact rational coefficients at
    lam, evaluated on the theta_1 = 2 pi/n enclosure, plus the explicit tail bound
    sum_m |g^_m| sum_{j>J} (m theta)^{2j}/(2j)! <= 2 sum_m |g^_m| (m theta)^{2J+2}/(2J+2)!
    (valid for m theta <= 1, where cosh(m theta) < 2)."""
    coeffs = at_lambda(op, lam)
    width = max(coeffs) - min(coeffs)
    gm = {m: 2 * sum((coeffs[s] * coeffs.get(s + m, Q(0)) for s in coeffs), Q(0)) for m in range(1, width + 1)}
    th_lo, th_hi = theta_one(n)
    th_lo = Q((th_lo.numerator << BITS) // th_lo.denominator, ONE)
    th_hi = Q(_ceil_div(th_hi.numerator << BITS, th_hi.denominator), ONE)
    assert width * th_hi <= 1
    lo = hi = Q(0)
    for j in range(1, terms + 1):
        c = Q((-1) ** (j + 1), factorial(2 * j)) * sum((g * Q(m) ** (2 * j) for m, g in gm.items()), Q(0))
        a, b = c * th_lo ** (2 * j), c * th_hi ** (2 * j)
        lo, hi = lo + min(a, b), hi + max(a, b)
    tail = 2 * sum((abs(g) * (m * th_hi) ** (2 * terms + 2) for m, g in gm.items()), Q(0)) / factorial(2 * terms + 2)
    return lo - tail, hi + tail


def contraction_factor(op: Operator, lam: Q, n: int) -> dict[str, Any]:
    """rho_N = max_{1<=k<=N-1} |g(theta_k)| through D_k = 1 - |g(theta_k)|^2.

    Returns the enclosure of min_k D_k and whether that minimum is attained
    at k = 1 and k = N-1 only, strictly separated from every other k."""
    coeffs = at_lambda(op, lam)
    width = max(coeffs) - min(coeffs)
    gm = {0: sum((c * c for c in coeffs.values()), Q(0))}
    for m in range(1, width + 1):
        gm[m] = 2 * sum((coeffs[s] * coeffs.get(s + m, Q(0)) for s in coeffs), Q(0))
    assert sum(gm.values()) == 1
    defects = [amplitude_defect_interval(gm, k, n) for k in range(1, n)]
    first = defects[0]
    at_first = defects[n - 2] == first and all(defects[i][0] > first[1] for i in range(1, n - 2))
    lowest = min(defects, key=lambda iv: iv[0])
    return {"max_at_k1_strict": at_first, "defect": iv_fraction(first if at_first else lowest)}


# ----------------------------------------------------------------------------
# Reference data transcribed from the authors' published outputs.

AUTHORS_EROSION_ORDERS = {
    # erasure_orders.csv: two_s, A_exact, A_at_0.5, A_at_0.9, nondissipative lambdas in (0,1].
    "uw1": (2, "1*lam + -1*lam^2", "0.25", "0.08999999999999997", "1"),
    "lw": (4, "1/4*lam^2 + -1/4*lam^4", "0.046875", "0.03847500000000001", "1"),
    "bw": (4, "1/2*lam + -5/4*lam^2 + 1*lam^3 + -1/4*lam^4", "0.046875", "0.0024749999999999217", ""),
    "fromm": (4, "1/4*lam + -1/2*lam^2 + 1/2*lam^3 + -1/4*lam^4", "0.046875", "0.02047500000000002", "1"),
    "ub3": (4, "1/6*lam + -1/12*lam^2 + -1/6*lam^3 + 1/12*lam^4", "0.046875", "0.01567499999999998", "1"),
    "ub5": (
        6,
        "1/30*lam + -1/90*lam^2 + -1/24*lam^3 + 1/72*lam^4 + 1/120*lam^5 + -1/360*lam^6",
        "0.009765625",
        "0.0031820249999999946",
        "1",
    ),
    "weno5l": (2, "-1*lam^2", "-0.25", "-0.81", ""),
    "weno5l_rk3": (4, "1/12*lam^4", "0.005208333333333333", "0.054675", ""),
}

AUTHORS_STABILITY_CERTIFICATE = {
    # stability_certificate.csv: object -> (x box, lambda box, certified, lower bound, cells).
    "S": ((Q(-1), Q(1)), (Q(0), Q(1, 2)), True, "0.1440179856729157", 1),
    "Q_over_lambda": ((Q(-1), Q(1)), (Q(1, 2), Q(1)), True, "0.041666666666666664", 1),
}

AUTHORS_ERASURE_RATES = {
    # erasure_rates.csv: (scheme, lambda, N) -> (rho_N, one_minus_rho, ratio).
    ("uw1", "0.5", 24): ("0.9914448613738104", "0.008555138626189618", "0.9985729211479477"),
    ("uw1", "0.5", 48): ("0.9978589232386036", "0.00214107676139641", "0.9996430774292928"),
    ("uw1", "0.5", 96): ("0.9994645874763657", "0.0005354125236343155", "0.9999107598009139"),
    ("uw1", "0.5", 192): ("0.9998661379095618", "0.00013386209043819708", "0.9999776893527832"),
    ("uw1", "0.5", 384): ("0.9999665339174012", "3.3466082598798685e-05", "0.9999944222979765"),
    ("uw1", "0.9", 24): ("0.9969286076405032", "0.0030713923594968273", "0.9958307497416042"),
    ("uw1", "0.9", 48): ("0.9992297408740823", "0.0007702591259176605", "0.9989576492709107"),
    ("uw1", "0.9", 96): ("0.9998072845218465", "0.00019271547815347123", "0.9997394100586907"),
    ("uw1", "0.9", 192): ("0.9999518117118175", "4.818828818253795e-05", "0.9999348523730587"),
    ("uw1", "0.9", 384): ("0.9999879523392875", "1.2047660712499741e-05", "0.9999837130848378"),
    ("lw", "0.5", 24): ("0.9998911457021759", "0.0001088542978241458", "0.9886891974602828"),
    ("lw", "0.5", 48): ("0.9999931383767484", "6.861623251586124e-06", "0.9971512998919093"),
    ("lw", "0.5", 96): ("0.9999995702302484", "4.2976975156427955e-07", "0.9992864970019618"),
    ("lw", "0.5", 192): ("0.9999999731250085", "2.6874991454484132e-08", "0.9998215451788901"),
    ("lw", "0.5", 384): ("0.9999999983200882", "1.6799117652510631e-09", "0.9999553553672407"),
    ("lw", "0.9", 24): ("0.9999106532638617", "8.934673613825161e-05", "0.9886795535729233"),
    ("lw", "0.9", 48): ("0.9999943679830976", "5.63201690240156e-06", "0.997150686854938"),
    ("lw", "0.9", 96): ("0.9999996472450012", "3.527549987669687e-07", "0.9992864592775076"),
    ("lw", "0.9", 192): ("0.999999977941007", "2.2058992965590107e-08", "0.9998215442610395"),
    ("lw", "0.9", 384): ("0.9999999986211284", "1.3788715724771805e-09", "0.999955352146713"),
    ("bw", "0.5", 24): ("0.9998911457021757", "0.00010885429782425682", "0.9886891974612911"),
    ("bw", "0.5", 48): ("0.9999931383767484", "6.861623251586124e-06", "0.9971512998919093"),
    ("bw", "0.5", 96): ("0.9999995702302484", "4.2976975156427955e-07", "0.9992864970019618"),
    ("bw", "0.5", 192): ("0.9999999731250087", "2.687499134346183e-08", "0.9998215410485636"),
    ("bw", "0.5", 384): ("0.9999999983200881", "1.6799118762733656e-09", "0.9999554214524651"),
    ("bw", "0.9", 24): ("0.9999942527893801", "5.747210619855281e-06", "0.9886382268719962"),
    ("bw", "0.9", 48): ("0.9999996377074697", "3.6229253030928277e-07", "0.9971480593706376"),
    ("bw", "0.9", 96): ("0.9999999773081618", "2.2691838186261748e-08", "0.999286290188409"),
    ("bw", "0.9", 192): ("0.9999999985810007", "1.4189992514346272e-09", "0.9998213356587218"),
    ("bw", "0.9", 384): ("0.9999999999113008", "8.86991591286801e-11", "0.999953302720152"),
    ("fromm", "0.5", 24): ("0.9998905274816261", "0.00010947251837389871", "0.9943042994030292"),
    ("fromm", "0.5", 48): ("0.9999931285930251", "6.871406974862104e-06", "0.9985730993735658"),
    ("fromm", "0.5", 96): ("0.9999995700768868", "4.2992311322098686e-07", "0.9996430884841395"),
    ("fromm", "0.5", 192): ("0.9999999731226106", "2.687738942519502e-08", "0.9999107561014503"),
    ("fromm", "0.5", 384): ("0.9999999983200508", "1.679949179766993e-09", "0.9999776260878502"),
    ("fromm", "0.9", 24): ("0.9999523737838233", "4.762621617671048e-05", "0.990324806026032"),
    ("fromm", "0.9", 48): ("0.9999970015807895", "2.9984192104759444e-06", "0.9975712240440809"),
    ("fromm", "0.9", 96): ("0.9999998122567199", "1.8774328014448116e-07", "0.9993921763864496"),
    ("fromm", "0.9", 192): ("0.9999999882606933", "1.1739306748914657e-08", "0.9998479891935205"),
    ("fromm", "0.9", 384): ("0.9999999992662096", "7.337903618065411e-10", "0.9999620365108014"),
    ("ub3", "0.5", 24): ("0.999890527481626", "0.00010947251837400973", "0.9943042994040376"),
    ("ub3", "0.5", 48): ("0.9999931285930251", "6.871406974862104e-06", "0.9985730993735658"),
    ("ub3", "0.5", 96): ("0.9999995700768868", "4.2992311322098686e-07", "0.9996430884841395"),
    ("ub3", "0.5", 192): ("0.9999999731226106", "2.687738942519502e-08", "0.9999107561014503"),
    ("ub3", "0.5", 384): ("0.9999999983200508", "1.679949179766993e-09", "0.9999776260878502"),
    ("ub3", "0.9", 24): ("0.9999635260230898", "3.647397691020071e-05", "0.990674668880397"),
    ("ub3", "0.9", 48): ("0.9999977043006261", "2.2956993739331466e-06", "0.9976608682818359"),
    ("ub3", "0.9", 96): ("0.9999998562665532", "1.437334468112894e-07", "0.9994147280788898"),
    ("ub3", "0.9", 192): ("0.9999999910127143", "8.987285693429214e-09", "0.9998536470647409"),
    ("ub3", "0.9", 384): ("0.9999999994382329", "5.617670773006012e-10", "0.9999635132561822"),
    ("ub5", "0.5", 24): ("0.9999984446418991", "1.555358100868709e-06", "0.9893468786478486"),
    ("ub5", "0.5", 48): ("0.9999999755015241", "2.449847591456944e-08", "0.9973262122124643"),
    ("ub5", "0.5", 96): ("0.9999999996164419", "3.835580741906597e-10", "0.9993307929979963"),
    ("ub5", "0.5", 192): ("0.9999999999940038", "5.996203533698008e-12", "0.9998491469040733"),
    ("ub5", "0.5", 384): ("0.9999999999999063", "9.370282327836321e-14", "0.9999787353805927"),
    ("ub5", "0.9", 24): ("0.9999994952842195", "5.047157805027069e-07", "0.9852842306504344"),
    ("ub5", "0.9", 48): ("0.9999999920256363", "7.974363724905231e-09", "0.9963012221786118"),
    ("ub5", "0.9", 96): ("0.999999999875054", "1.249460535035496e-10", "0.999072809669394"),
    ("ub5", "0.9", 192): ("0.9999999999980467", "1.9533263895255004e-12", "0.9996072279647245"),
    ("ub5", "0.9", 384): ("0.9999999999999698", "3.019806626980426e-14", "0.9890395944304833"),
    ("weno5l", "0.5", 24): ("1.1869662543176571", "-0.18696625431765712", "21.82307580131255"),
    ("weno5l", "0.5", 48): ("1.1869662543176571", "-0.18696625431765712", "87.2923032052502"),
    ("weno5l", "0.5", 96): ("1.18751825270195", "-0.1875182527019501", "350.20009853899023"),
    ("weno5l", "0.5", 192): ("1.1875701767643612", "-0.18757017676436116", "1401.1882777141934"),
    ("weno5l", "0.5", 384): ("1.1876285881211999", "-0.18762858812119987", "5606.498490850537"),
    ("weno5l", "0.9", 24): ("1.6089362348606064", "-0.6089362348606064", "21.937116472456974"),
    ("weno5l", "0.9", 48): ("1.6091561002828316", "-0.6091561002828316", "87.78014876957457"),
    ("weno5l", "0.9", 96): ("1.6121856193348887", "-0.6121856193348887", "352.8668248736913"),
    ("weno5l", "0.9", 192): ("1.6121856193348887", "-0.6121856193348887", "1411.4672994947653"),
    ("weno5l", "0.9", 384): ("1.6121856193348887", "-0.6121856193348887", "5645.869197979061"),
    ("weno5l_rk3", "0.5", 24): ("0.9999851990710125", "1.4800928987535045e-05", "1.2098894582365327"),
    ("weno5l_rk3", "0.5", 48): ("0.9999991947672193", "8.052327806762349e-07", "1.0531691934603817"),
    ("weno5l_rk3", "0.5", 96): ("0.9999999515764239", "4.8423576148870495e-08", "1.0133361653626667"),
    ("weno5l_rk3", "0.5", 192): ("0.9999999970033912", "2.9966088499477905e-09", "1.0033367586937312"),
    ("weno5l_rk3", "0.5", 384): ("0.9999999998131796", "1.868204480004465e-10", "1.0008317115274865"),
    ("weno5l_rk3", "0.9", 24): ("0.9998692002595524", "0.0001307997404476069", "1.01852942362734"),
    ("weno5l_rk3", "0.9", 48): ("0.9999919356996859", "8.06430031408123e-06", "1.004739259909956"),
    ("weno5l_rk3", "0.9", 96): ("0.9999994977608916", "5.022391084130717e-07", "1.0011915835101732"),
    ("weno5l_rk3", "0.9", 192): ("0.999999968638062", "3.1361938046536864e-08", "1.0002983165201227"),
    ("weno5l_rk3", "0.9", 384): ("0.9999999980403177", "1.959682305319177e-09", "1.0000743744937066"),
}

AUTHORS_ERASURE_SLOPES = {
    # erasure_rate_fits.csv: slope_fit over the last three N.
    ("uw1", "0.5"): "-1.9999396474038997",
    ("uw1", "0.9"): "-1.9998237482169545",
    ("lw", "0.5"): "-3.999517337694035",
    ("lw", "0.9"): "-3.9995173127853887",
    ("bw", "0.5"): "-3.999517290021494",
    ("bw", "0.9"): "-3.9995186691425726",
    ("fromm", "0.5"): "-3.9997586363539566",
    ("fromm", "0.9"): "-3.9995888000273467",
    ("ub3", "0.5"): "-3.9997586363539566",
    ("ub3", "0.9"): "-3.999604012058697",
    ("ub5", "0.5"): "-5.999532446936501",
    ("ub5", "0.9"): "-6.00728077181369",
    ("weno5l_rk3", "0.5"): "-4.008956723776011",
    ("weno5l_rk3", "0.9"): "-4.000805386244981",
}

AUTHORS_QUEUE_EXPONENTS = {
    # tab:queue-exponents / results/tables/queue_exponents.tex: fitted (predicted) per L.
    "uw1": ("1.00 (1)", "2.00 (2)", "3.01 (3)", "4.01 (4)", "5.02 (5)"),
    "lw": ("1.01 (1)", "1.99 (2)", "3.00 (3)", "3.00 (3)", "3.00 (3)"),
    "bw": ("1.00 (1)", "2.00 (2)", "3.00 (3)", "3.95 (4)", "3.95 (4)"),
    "fromm": ("0.99 (1)", "1.99 (2)", "3.00 (3)", "3.00 (3)", "2.99 (3)"),
    "ub3": ("0.99 (1)", "1.98 (2)", "3.02 (3)", "3.01 (3)", "3.01 (3)"),
    "ub5": ("1.00 (1)", "1.98 (2)", "3.01 (3)", "3.01 (3)", "3.01 (3)"),
    "weno5l": ("1.00 (1)", "2.00 (2)", "3.00 (3)", "3.00 (3)", "3.00 (3)"),
}

AUTHORS_QUEUE_FITS = {
    # queue_conditioning_fits.csv: (q, gamma_fit) per (scheme, L).
    ("uw1", 1): (1, "0.9999999999999929"),
    ("uw1", 2): (2, "2.0034048031290745"),
    ("uw1", 3): (3, "3.0081023794138355"),
    ("uw1", 4): (4, "4.012656183970116"),
    ("uw1", 5): (5, "5.017272732389844"),
    ("lw", 1): (2, "1.0053509509151486"),
    ("lw", 2): (4, "1.9949937627438314"),
    ("lw", 3): (5, "3.0014030108442387"),
    ("lw", 4): (5, "3.0012375814298777"),
    ("lw", 5): (5, "3.0010758962560526"),
    ("bw", 1): (1, "0.9965743365175631"),
    ("bw", 2): (2, "1.998834013989332"),
    ("bw", 3): (3, "3.0029888815906287"),
    ("bw", 4): (5, "3.954677050984206"),
    ("bw", 5): (5, "3.9519310164899153"),
    ("fromm", 1): (2, "0.9922361924175401"),
    ("fromm", 2): (4, "1.9881812726217594"),
    ("fromm", 3): (5, "3.001049207844211"),
    ("fromm", 4): (5, "2.996213779011643"),
    ("fromm", 5): (5, "2.9912409956092767"),
    ("ub3", 1): (2, "0.9938505039332135"),
    ("ub3", 2): (4, "1.9780795958250799"),
    ("ub3", 3): (5, "3.0151779911787537"),
    ("ub3", 4): (5, "3.0117988877052064"),
    ("ub3", 5): (5, "3.0084221163791938"),
    ("ub5", 1): (2, "0.99891966094897"),
    ("ub5", 2): (4, "1.9818238663283154"),
    ("ub5", 3): (5, "3.014380838584226"),
    ("ub5", 4): (5, "3.01288859252921"),
    ("ub5", 5): (5, "3.0114008335703057"),
    ("weno5l", 1): (2, "0.999999999999999"),
    ("weno5l", 2): (4, "2.0041910174233606"),
    ("weno5l", 3): (5, "3.0016595247478146"),
    ("weno5l", 4): (5, "3.0003728217195107"),
    ("weno5l", 5): (5, "2.9990901935018566"),
}

QUEUE_LAMBDAS = (
    # The E3 Courant grid of queue_conditioning.csv (fits use lambda <= 0.05).
    "0.001",
    "0.0015587755020547783",
    "0.0024297810658061286",
    "0.003787483200735143",
    "0.005903836027749967",
    "0.009202754968205043",
    "0.014345028995850929",
    "0.022360679774997897",
    "0.03485527984255849",
    "0.05433155633584393",
    "0.08469069900482264",
    "0.1320137868606126",
    "0.2057798568918039",
    "0.3207645997392824",
    "0.5",
)

AUTHORS_QUEUE_SIGMA = {
    # queue_conditioning.csv sigma_min_plus at lambda = 0.001 and 0.5.
    ("uw1", 1, "0.001"): "0.0002357022603955225",
    ("uw1", 1, "0.5"): "0.11785113019775792",
    ("uw1", 2, "0.001"): "1.0545142536000952e-07",
    ("uw1", 2, "0.5"): "0.031830500937508756",
    ("uw1", 3, "0.001"): "5.4125066840862496e-11",
    ("uw1", 3, "0.5"): "0.01100153298656112",
    ("uw1", 4, "0.001"): "2.8417677365641474e-14",
    ("uw1", 4, "0.5"): "0.003877216159864174",
    ("uw1", 5, "0.001"): "1.490201107436457e-17",
    ("uw1", 5, "0.5"): "0.0013574099802340347",
    ("lw", 1, "0.001"): "7.986285074944274e-05",
    ("lw", 1, "0.5"): "0.039232966673617045",
    ("lw", 2, "0.001"): "1.8633502739900617e-08",
    ("lw", 2, "0.5"): "0.001528161288348921",
    ("lw", 3, "0.001"): "8.278375008545204e-12",
    ("lw", 3, "0.5"): "0.0023469127148638895",
    ("lw", 4, "0.001"): "2.4075761211993468e-11",
    ("lw", 4, "0.5"): "0.004385258430951231",
    ("lw", 5, "0.001"): "5.234557789475536e-11",
    ("lw", 5, "0.5"): "0.006256478219656945",
    ("bw", 1, "0.001"): "0.0003725289325080013",
    ("bw", 1, "0.5"): "0.15023130314433286",
    ("bw", 2, "0.001"): "3.531197717639761e-07",
    ("bw", 2, "0.5"): "0.06791120401599661",
    ("bw", 3, "0.001"): "3.8932954224596313e-10",
    ("bw", 3, "0.5"): "0.04184225812476911",
    ("bw", 4, "0.001"): "1.428074721831081e-14",
    ("bw", 4, "0.5"): "1.7072032409239418e-05",
    ("bw", 5, "0.001"): "7.256818963417702e-14",
    ("bw", 5, "0.5"): "0.02332142866822184",
    ("fromm", 1, "0.001"): "9.378298014073632e-05",
    ("fromm", 1, "0.5"): "0.024603440438150558",
    ("fromm", 2, "0.001"): "4.938935410911794e-09",
    ("fromm", 2, "0.5"): "0.0004582199576738181",
    ("fromm", 3, "0.001"): "8.940352619276333e-11",
    ("fromm", 3, "0.5"): "0.000853306112950595",
    ("fromm", 4, "0.001"): "2.598709816481987e-10",
    ("fromm", 4, "0.5"): "0.005065098076654768",
    ("fromm", 5, "0.001"): "5.647134158524118e-10",
    ("fromm", 5, "0.5"): "0.010085524444154548",
    ("ub3", 1, "0.001"): "0.00010626878114153037",
    ("ub3", 1, "0.5"): "0.024603440438150558",
    ("ub3", 2, "0.001"): "8.764759383059032e-09",
    ("ub3", 2, "0.5"): "0.0004582199576738181",
    ("ub3", 3, "0.001"): "4.580058280844092e-11",
    ("ub3", 3, "0.5"): "0.000853306112950595",
    ("ub3", 4, "0.001"): "1.3315134717066035e-10",
    ("ub3", 4, "0.5"): "0.005065098076654768",
    ("ub3", 5, "0.001"): "2.8939201994649493e-10",
    ("ub3", 5, "0.5"): "0.010085524444154548",
    ("ub5", 1, "0.001"): "0.00012348073119571525",
    ("ub5", 1, "0.5"): "0.03412127764778298",
    ("ub5", 2, "0.001"): "1.985702072978931e-08",
    ("ub5", 2, "0.5"): "0.0013646949808023509",
    ("ub5", 3, "0.001"): "4.933661869345951e-11",
    ("ub5", 3, "0.5"): "0.002671610074303655",
    ("ub5", 4, "0.001"): "1.4346166155140804e-10",
    ("ub5", 4, "0.5"): "0.006561249157067288",
    ("ub5", 5, "0.001"): "3.118657366184629e-10",
    ("ub5", 5, "0.5"): "0.013382022771767704",
    ("weno5l", 1, "0.001"): "0.00012348898035182884",
    ("weno5l", 1, "0.5"): "0.061744490175913894",
    ("weno5l", 2, "0.001"): "1.990885901684776e-08",
    ("weno5l", 2, "0.5"): "0.006483166715073911",
    ("weno5l", 3, "0.001"): "4.926406039707148e-11",
    ("weno5l", 3, "0.5"): "0.006187367342664743",
    ("weno5l", 4, "0.001"): "1.43250787202976e-10",
    ("weno5l", 4, "0.5"): "0.01811144109433808",
    ("weno5l", 5, "0.001"): "3.114075434320395e-10",
    ("weno5l", 5, "0.5"): "0.038872912912450375",
}

AUTHORS_HALF_LIFE = {
    # tab:half-life (N=48, lambda=1/2): n_1/2 (pred.) 4 s.f., retention vs UW1;
    # delayed_collisions.csv: half_life_predicted and rho_N.
    "uw1": ("3.234e2", 1, "323.3909625780552", "0.9978589232386036"),
    "lw": ("1.010e5", 312, "101017.61304653295", "0.9999931383767484"),
    "bw": ("1.010e5", 312, "101017.61304653295", "0.9999931383767484"),
    "fromm": ("1.009e5", 312, "100873.78052919685", "0.9999931285930251"),
    "ub3": ("1.009e5", 312, "100873.78052919685", "0.9999931285930251"),
    "ub5": ("2.829e7", 87500, "28293481.37763951", "0.9999999755015241"),
    "weno5l_rk3": ("8.608e5", 2662, "860803.1343500284", "0.9999991947672193"),
}


def parse_csv_poly(text: str) -> Poly:
    coeffs: dict[int, Q] = {}
    for term in text.split(" + "):
        if "*lam" in term:
            c, _, power = term.partition("*lam")
            degree = int(power[1:]) if power.startswith("^") else 1
        else:
            c, degree = term, 0
        coeffs[degree] = coeffs.get(degree, Q(0)) + Q(c)
    top = max(coeffs)
    return ptrim(coeffs.get(i, Q(0)) for i in range(top + 1))


def significant_round(q: Q, digits: int) -> Q:
    text = decimal_text(q, digits)
    mantissa, _, exponent = text.partition("e")
    return Q(mantissa) * Q(10) ** int(exponent)


# ----------------------------------------------------------------------------


def build_certificate() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    discrepancies: list[dict[str, str]] = []
    operators = build_operators()

    # ------------------------------------------------------------------
    # Dissipative orders and leading coefficients (items 4 and 5).
    closed_forms = {
        "uw1": (2, poly_from_factors(Q(1), [(LAM, 1), (lin(1, -1), 1)])),
        "lw": (4, poly_from_factors(Q(1, 4), [(LAM, 2), (lin(1, -1), 1), (lin(1, 1), 1)])),
        "bw": (4, poly_from_factors(Q(1, 4), [(LAM, 1), (lin(1, -1), 2), (lin(2, -1), 1)])),
        "fromm": (4, poly_from_factors(Q(1, 4), [(LAM, 1), (lin(1, -1), 1), ((Q(1), Q(-1), Q(1)), 1)])),
        "ub3": (4, poly_from_factors(Q(1, 12), [(LAM, 1), (lin(1, -1), 1), (lin(1, 1), 1), (lin(2, -1), 1)])),
        "ub5": (
            6,
            poly_from_factors(
                Q(1, 360),
                [(LAM, 1), (lin(1, -1), 1), (lin(1, 1), 1), (lin(2, -1), 1), (lin(2, 1), 1), (lin(3, -1), 1)],
            ),
        ),
        "weno5l": (2, pscale(ppow(LAM, 2), Q(-1))),
        "weno5l_rk3": (4, pscale(ppow(LAM, 4), Q(1, 12))),
    }
    dissipation: dict[str, Any] = {}
    leading: dict[str, Poly] = {}
    orders: dict[str, int] = {}
    discrete_p = {"uw1": 1, "lw": 2, "bw": 2, "fromm": 2, "ub3": 3, "ub5": 5, "weno5l": 1, "weno5l_rk3": 3}
    for name in SCHEME_ORDER:
        op = operators[name]
        by_moments = defect_series_by_moments(op, 8)
        direct = defect_series_direct(op, 8)
        checks[f"amplitude_series_two_routes_agree_{name}"] = by_moments == direct
        two_s, coeff = dissipative_order(by_moments)
        orders[name] = two_s
        leading[name] = coeff
        expected_order, expected = closed_forms[name]
        checks[f"dissipative_order_and_leading_coefficient_{name}"] = (
            two_s == expected_order and coeff == expected
        )
        ref = AUTHORS_EROSION_ORDERS[name]
        checks[f"erasure_orders_csv_two_s_and_A_exact_{name}"] = (
            ref[0] == two_s and parse_csv_poly(ref[1]) == coeff
        )
        checks[f"erasure_orders_csv_A_values_{name}"] = all(
            abs(peval(coeff, lam) - Q(value)) <= Q(1, 10**14)
            for lam, value in ((Q(1, 2), ref[2]), (Q(9, 10), ref[3]))
        )
        checks[f"consistency_order_table_scheme_zoo_{name}"] = consistency_order(op) == discrete_p[name]
        if name in STABLE:
            zeros_open = roots_in_open_unit_interval(coeff)
            nondiss = ["1"] if peval(coeff, Q(1)) == 0 else []
            positive = zeros_open == 0 and peval(coeff, Q(1, 2)) > 0
            checks[f"leading_coefficient_positive_on_open_interval_{name}"] = positive
            dissipation[name] = {
                "two_s": two_s,
                "A": poly_text(coeff),
                "nondissipative_lambdas_in_0_1": nondiss,
                "authors_csv_nondissipative": ref[4],
            }
            if ",".join(nondiss) != ref[4]:
                discrepancies.append(
                    {
                        "location": "results/erasure_orders.csv, row " + name,
                        "claim": f"nondissipative_lambdas_in_(0,1] = '{ref[4]}'",
                        "finding": (
                            f"A(lambda) = {poly_text(coeff)} vanishes at lambda = 1 "
                            f"(exact zeros in (0,1]: {nondiss}); the manuscript's tab:scheme-zoo "
                            "lists lambda=1 as nondissipative, so the CSV omits it"
                        ),
                    }
                )
        else:
            dissipation[name] = {"two_s": two_s, "A": poly_text(coeff)}
    checks["only_zero_of_A_in_0_1_is_lambda_1_for_interpolation_members"] = all(
        peval(leading[n], Q(1)) == 0 and roots_in_open_unit_interval(leading[n]) == 0
        for n in ("uw1", "lw", "bw", "fromm", "ub3", "ub5")
    )
    checks["bw_leading_coefficient_double_root_at_lambda_1"] = (
        peval(leading["bw"], Q(1)) == 0 and peval(pderiv(leading["bw"]), Q(1)) == 0
    )
    checks["rk3_leading_coefficient_nonzero_on_0_1"] = peval(leading["weno5l_rk3"], Q(1)) != 0
    checks["fe_antidissipative_every_lambda"] = leading["weno5l"] == pscale(ppow(LAM, 2), Q(-1))
    fe_moment = defect_series_by_moments(operators["weno5l"], 1)[0]
    checks["fe_second_moment_minus_two_lambda_squared"] = pscale(fe_moment, Q(2)) == pscale(ppow(LAM, 2), Q(-2))

    # rem:coefficient-structure (i): odd-degree closed form.
    odd_ok = True
    for name, nodes in (("uw1", (-1, 0)), ("ub3", (-2, -1, 0, 1)), ("ub5", (-3, -2, -1, 0, 1, 2))):
        omega: Poly = ONE_P
        for d in nodes:
            omega = pmul(omega, lin(-d, -1))  # (x - d) at x = -lam
        candidate = pscale(omega, Q(2, factorial(len(nodes))))
        if peval(candidate, Q(1, 2)) < 0:
            candidate = pscale(candidate, Q(-1))
        odd_ok &= candidate == leading[name] and roots_in_open_unit_interval(omega) == 0
    checks["odd_degree_closed_form_A_equals_2_abs_omega_over_factorial"] = odd_ok
    # (ii) reflection duality |g_BW(theta; lam)| = |g_LW(theta; lam - 1)|.
    g_lw = autocorrelations(operators["lw"])
    g_bw = autocorrelations(operators["bw"])
    shifted = lin(-1, 1)
    checks["reflection_duality_bw_lw_autocorrelations"] = set(g_lw) == set(g_bw) and all(
        pcompose(g_lw[m], shifted) == g_bw[m] for m in g_lw
    ) and pcompose(leading["lw"], lin(1, -1)) == leading["bw"]
    checks["common_value_A_half_3_over_64"] = all(
        peval(leading[n], Q(1, 2)) == Q(3, 64) for n in ("lw", "bw", "fromm", "ub3")
    )
    nu = {n: peval(leading[n], Q(1, 2)) / (2 * Q(1, 2)) for n in ("lw", "ub5", "weno5l_rk3")}
    checks["modified_equation_constants_at_lambda_half"] = nu == {
        "lw": Q(3, 64),
        "ub5": Q(5, 512),
        "weno5l_rk3": Q(1, 192),
    }

    # prop:weno-fe: RK3 stability polynomial and flux-difference symbol.
    re_r = (Q(1), Q(0), Q(-1, 2))
    im_r = (Q(0), Q(1), Q(0), Q(-1, 6))
    checks["rk3_imaginary_axis_amplitude_identity"] = padd(pmul(re_r, re_r), pmul(im_r, im_r)) == (
        Q(1), Q(0), Q(0), Q(0), Q(-1, 12), Q(0), Q(1, 36)
    )
    diff = weno5_flux_difference()
    flux_moments = [sum((v * Q(s) ** n for s, v in diff.items()), Q(0)) for n in range(8)]
    checks["weno5_flux_symbol_i_theta_plus_real_theta6"] = (
        flux_moments[0] == 0 and flux_moments[1] == -1 and all(flux_moments[n] == 0 for n in range(2, 6))
        and flux_moments[6] != 0
    )

    # ------------------------------------------------------------------
    # prop:interior-stability: (1-x)^s Q factorizations.
    quotients: dict[str, Bivariate] = {}
    factor_ok = True
    for name in ("uw1", "lw", "bw", "fromm", "ub3", "ub5", "weno5l_rk3"):
        f = amplitude_defect_in_x(operators[name])
        s = orders[name] // 2
        for _ in range(s):
            f, remainder = divide_by_one_minus_x(f)
            factor_ok &= remainder == ()
        quotients[name] = f
        factor_ok &= bivariate_eval_x(f, Q(1)) == pscale(leading[name], Q(2**s))
    checks["interior_stability_divisible_by_one_minus_x_power_s"] = factor_ok
    expected_q = {
        "uw1": [poly_from_factors(Q(2), [(LAM, 1), (lin(1, -1), 1)])],
        "lw": [poly_from_factors(Q(1), [(LAM, 2), (lin(1, -1), 1), (lin(1, 1), 1)])],
        "bw": [poly_from_factors(Q(1), [(LAM, 1), (lin(1, -1), 2), (lin(2, -1), 1)])],
    }
    for name, value in expected_q.items():
        checks[f"interior_stability_quotient_{name}"] = quotients[name] == value
    fr, ub3, ub5 = quotients["fromm"], quotients["ub3"], quotients["ub5"]
    fr_one = poly_from_factors(Q(1), [(LAM, 1), (lin(1, -1), 1), ((Q(1), Q(-1), Q(1)), 1)])
    fr_minus = poly_from_factors(Q(1), [(LAM, 1), (lin(1, -1), 1)])
    checks["interior_stability_quotient_fromm_affine"] = (
        len(fr) == 2 and bivariate_eval_x(fr, Q(1)) == fr_one and bivariate_eval_x(fr, Q(-1)) == fr_minus
        and positive_on_open_unit_interval(fr_one) and positive_on_open_unit_interval(fr_minus)
    )
    base3 = [(LAM, 1), (lin(1, -1), 1), (lin(1, 1), 1), (lin(2, -1), 1)]
    ub3_one = poly_from_factors(Q(1, 3), base3)
    ub3_minus = poly_from_factors(Q(1, 9), base3 + [(lin(3, -2), 1), (lin(1, 2), 1)])
    checks["interior_stability_quotient_ub3_affine"] = (
        len(ub3) == 2 and bivariate_eval_x(ub3, Q(1)) == ub3_one and bivariate_eval_x(ub3, Q(-1)) == ub3_minus
        and positive_on_open_unit_interval(ub3_one) and positive_on_open_unit_interval(ub3_minus)
    )
    ub5_one = poly_from_factors(
        Q(1, 45), [(LAM, 1), (lin(1, -1), 1), (lin(1, 1), 1), (lin(2, -1), 1), (lin(2, 1), 1), (lin(3, -1), 1)]
    )
    ub5_q2 = poly_from_factors(
        Q(1, 450), [(LAM, 2), (lin(1, -1), 2), (lin(1, 1), 2), (lin(2, -1), 2), (lin(2, 1), 1), (lin(3, -1), 1)]
    )
    ub5_slope = poly_from_factors(
        Q(-1, 60), [(LAM, 2), (lin(1, -1), 2), (lin(1, 1), 1), (lin(2, -1), 1), (lin(2, 1), 1), (lin(3, -1), 1)]
    )
    checks["interior_stability_quotient_ub5_quadratic"] = (
        len(ub5) == 3 and bivariate_eval_x(ub5, Q(1)) == ub5_one and ub5[2] == ub5_q2
        and bivariate_eval_x(bivariate_deriv_x(ub5), Q(1)) == ub5_slope
        and padd(ub5[1], pscale(ub5[2], Q(2))) == ub5_slope
        and positive_on_open_unit_interval(ub5_q2) and positive_on_open_unit_interval(pscale(ub5_slope, Q(-1)))
        and positive_on_open_unit_interval(ub5_one)
    )

    # prop:rk3-stability: decomposition and tensor-Bernstein certificates.
    q_rk3 = quotients["weno5l_rk3"]
    divisible_by_lambda = all(not p or p[0] == 0 for p in q_rk3)
    q_over_lambda = [ptrim(p[1:]) for p in q_rk3]
    shifted_q = [psub(q_over_lambda[0], pscale(ppow(LAM, 3), Q(1, 3)))] + q_over_lambda[1:]
    s_poly, remainder = divide_by_one_minus_x(shifted_q)
    checks["rk3_decomposition_Q_equals_lambda_times_one_minus_x_S_plus_lambda3_over_3"] = (
        divisible_by_lambda and remainder == () and len(q_rk3) - 1 == 13 and len(s_poly) - 1 == 12
        and max(len(p) for p in s_poly) - 1 == 5
        and [p[0] if p else Q(0) for p in s_poly] == [Q(4, 15)] + [Q(0)] * 12
        and bivariate_eval_x(q_rk3, Q(1)) == pscale(ppow(LAM, 4), Q(1, 3))
    )
    stability: dict[str, Any] = {}
    for label, poly in (("S", s_poly), ("Q_over_lambda", q_over_lambda)):
        x_box, l_box, ref_cert, ref_bound, ref_cells = AUTHORS_STABILITY_CERTIFICATE[label]
        bound, nx, nl = tensor_bernstein_min(poly, x_box, l_box)
        stability[label] = {
            "box": [str(x_box[0]), str(x_box[1]), str(l_box[0]), str(l_box[1])],
            "bernstein_degrees": [nx, nl],
            "min_bernstein_coefficient": str(bound),
            "min_bernstein_decimal": decimal_text(bound, 16),
            "authors_certified_lower_bound": ref_bound,
        }
        checks[f"rk3_bernstein_certificate_positive_{label}"] = bound > 0
        checks[f"rk3_bernstein_bound_matches_authors_csv_{label}"] = (
            ref_cert and ref_cells == 1 and abs(bound - Q(ref_bound)) <= Q(1, 10**15)
        )
    checks["rk3_bernstein_bounds_quoted_0_1440_and_1_24"] = (
        Q(stability["S"]["min_bernstein_coefficient"]) >= Q(1440, 10**4)
        and Q(stability["Q_over_lambda"]["min_bernstein_coefficient"]) == Q(1, 24)
    )

    # Forward Euler: the maximum of |g|^2 over x = cos(theta) in [-1, 1].
    fe_max: dict[str, Any] = {}
    fe_max_exact: dict[Q, tuple[Q, Q]] = {}
    fe_poly = amplitude_defect_in_x(operators["weno5l"])
    for lam in (Q(1, 2), Q(9, 10)):
        amp = ptrim([(1 if i == 0 else 0) - peval(p, lam) for i, p in enumerate(fe_poly)])
        seq = sturm_sequence(pderiv(amp))
        lipschitz = sum((abs(c) * i for i, c in enumerate(amp)), Q(0))
        endpoint = max(peval(amp, Q(-1)), peval(amp, Q(1)))
        brackets = []
        stack = [(Q(-1), Q(1))]
        while stack:
            a, b = stack.pop()
            count = sign_changes(seq, a) - sign_changes(seq, b)
            if count == 0:
                continue
            if count == 1 and b - a < Q(1, 2**90):
                brackets.append((a, b))
                continue
            mid = (a + b) / 2
            stack.extend(((a, mid), (mid, b)))
        interior_values = [
            (peval(amp, a) - lipschitz * (b - a), peval(amp, a) + lipschitz * (b - a)) for a, b in brackets
        ] or [(endpoint, endpoint)]
        best = max(interior_values, key=lambda iv: iv[0])
        upper = max(iv[1] for iv in interior_values)
        interior_max = best[0] > endpoint and best[0] > 1
        r_lo, r_hi = sqrt_bounds(best[0], max(upper, endpoint))
        fe_max_exact[lam] = (r_lo - 1, r_hi - 1)
        fe_max[str(lam)] = {
            "max_abs_g_minus_1": [decimal_text(r_lo - 1, 12), decimal_text(r_hi - 1, 12)],
            "attained_at_interior_frequency": interior_max,
        }
        checks[f"fe_maximum_interior_and_unstable_lambda_{lam}"] = interior_max
    checks["fe_rho_minus_one_about_0_19_at_half"] = (
        Q(18, 100) < fe_max_exact[Q(1, 2)][0] and fe_max_exact[Q(1, 2)][1] < Q(195, 1000)
    )

    # ------------------------------------------------------------------
    # conj:conditioning (item 6).
    conditioning: dict[str, Any] = {}
    exact_exponents: dict[tuple[str, int, int], int] = {}
    rank_determined = True
    conjecture_instances = True
    bracket_slopes_ok = True
    lam_seq = (Q(1, 100), Q(1, 1000), Q(1, 10**4), Q(1, 10**5))
    for children in (6, 8):
        for name in SINGLE_STAGE + ("weno5l_rk3",):
            horizon_max = 1 if (name == "weno5l_rk3" and children == 8) else children - 1
            powers = laurent_powers(operators[name], horizon_max)
            blocks_all = local_polynomial_blocks(powers, 8, children, horizon_max)
            ladder = [0]
            report = []
            last_slope = (Q(0), Q(0))
            for horizon in range(1, horizon_max + 1):
                blocks = blocks_all[: horizon + 1]
                zero_at_t0 = all(not any(row) for row in blocks[0])
                conserved = all(
                    all(
                        sum_rows == ()
                        for sum_rows in (
                            _sum_polys([row[j] for row in block]) for j in range(children - 1)
                        )
                    )
                    for block in blocks
                )
                reduced = []
                full = []
                for block in blocks[1:]:
                    live = [k for k, row in enumerate(block) if any(row)]
                    full.extend(block[k] for k in live)
                    reduced.extend(block[k] for k in live if k != 0)
                exps = smith_exponents(reduced, children - 1, 12)
                rho = exact_rank([[peval(p, Q(1, 2)) for p in row] for row in full])
                determined = len(exps) == min(len(reduced), children - 1) == rho
                rank_determined &= determined and zero_at_t0 and conserved
                ladder.append(rho)
                q = rho
                l_q = min(L for L, value in enumerate(ladder) if value >= q)
                gamma = max(exps)
                exact_exponents[(name, children, horizon)] = gamma
                entry: dict[str, Any] = {
                    "L": horizon,
                    "q": q,
                    "L_q": l_q,
                    "smith_exponents": exps,
                    "exact_exponent": gamma,
                }
                if name != "weno5l_rk3":
                    conjecture_instances &= gamma == l_q
                if children == 6 and (name != "weno5l_rk3" or horizon == 1):
                    brackets = []
                    slopes = []
                    previous = None
                    for lam in lam_seq:
                        matrix = [[peval(p, lam) for p in row] for row in full]
                        zeros, lo, hi, _ = gram_bracket(matrix)
                        brackets.append([decimal_text(lo, 12), decimal_text(hi, 12)])
                        if previous is not None:
                            plo, phi = previous
                            s_lo = (ln_bounds(plo)[0] - ln_bounds(hi)[1]) / (2 * ln_bounds(Q(10))[1])
                            s_hi = (ln_bounds(phi)[1] - ln_bounds(lo)[0]) / (2 * ln_bounds(Q(10))[0])
                            slopes.append(decimal_text((s_lo + s_hi) / 2, 6))
                            last_slope = (s_lo, s_hi)
                        previous = (lo, hi)
                    entry["sigma_squared_brackets_lambda_1e-2_to_1e-5"] = brackets
                    entry["decade_slopes"] = slopes
                    bracket_slopes_ok &= abs((last_slope[0] + last_slope[1]) / 2 - gamma) < Q(1, 100)
                report.append(entry)
            conditioning[f"{name}_P8_r{children}"] = report
    checks["conditioning_generic_rank_determined_exactly"] = rank_determined
    checks["conjecture_conditioning_exact_exponent_equals_L_q_single_stage_r6_r8"] = conjecture_instances
    checks["conditioning_exact_brackets_slopes_approach_exact_exponent"] = bracket_slopes_ok
    table_ok = True
    fit_ok = True
    for name, cells in AUTHORS_QUEUE_EXPONENTS.items():
        for horizon, cell in enumerate(cells, start=1):
            fit_text, predicted_text = cell.split(" ")
            predicted = int(predicted_text.strip("()"))
            entry = conditioning[f"{name}_P8_r6"][horizon - 1]
            table_ok &= entry["L_q"] == predicted == exact_exponents[(name, 6, horizon)]
            q_ref, fit_ref = AUTHORS_QUEUE_FITS[(name, horizon)]
            table_ok &= q_ref == entry["q"]
            fit_ok &= abs(Q(fit_ref) - predicted) <= Q(5, 100)
            fit_ok &= Q(fit_text) == Q(round(Q(fit_ref) * 100), 100)
    checks["queue_exponents_table_predicted_equals_exact_exponent"] = table_ok
    checks["queue_exponents_fits_within_0_05_and_table_rounding"] = fit_ok

    # The E3 data reproduced: exact sigma+_min at the authors' Courant grid,
    # in the coordinate convention xi_1..xi_{r-1} (basis e_j - e_0).
    epsilon = Q(1, 2**52)
    sigma_ok = True
    sigma_half_ok = True
    max_sigma_deviation = Q(0)
    conditioning_ratio = Q(1)
    orthonormal_mismatch = False
    refits: dict[str, Any] = {}
    sigma_comparison: dict[str, Any] = {}
    refit_within_005 = True
    max_refit_deviation = Q(0)
    for name in SINGLE_STAGE:
        powers = laurent_powers(operators[name], 5)
        blocks_all = local_polynomial_blocks(powers, 8, 6, 5)
        for horizon in range(1, 6):
            full = [row for block in blocks_all[1 : horizon + 1] for row in block if any(row)]
            xs, ys = [], []
            for lam_text in QUEUE_LAMBDAS:
                lam = Q(lam_text)
                if lam > Q(5, 100) and lam_text != "0.5":
                    continue
                matrix = [[peval(p, lam) for p in row] for row in full]
                _, lo, hi, frob = gram_bracket(matrix, 32)
                if lam <= Q(5, 100):
                    xs.append(ln_mid(lam))
                    ys.append(ln_mid((lo + hi) / 2) / 2)
                key = (name, horizon, lam_text)
                if key in AUTHORS_QUEUE_SIGMA:
                    ref = Q(AUTHORS_QUEUE_SIGMA[key])
                    s_lo, s_hi = sqrt_bounds(lo, hi)
                    s_mid = (s_lo + s_hi) / 2
                    relative = abs(ref - s_mid) / s_mid
                    max_sigma_deviation = max(max_sigma_deviation, relative)
                    sigma_ok &= relative < Q(1, 10**3)
                    if lam_text == "0.5":
                        sigma_half_ok &= relative < Q(1, 10**9)
                    conditioning_ratio = min(conditioning_ratio, s_mid / sqrt_bounds(frob, frob)[1])
                    sigma_comparison[f"{name}_L{horizon}_lambda_{lam_text}"] = {
                        "exact_sigma": decimal_text(s_mid, 10),
                        "authors_sigma": AUTHORS_QUEUE_SIGMA[key],
                        "relative_deviation": decimal_text(relative, 3) if relative else "0",
                    }
                    tolerance = 1024 * epsilon * sqrt_bounds(frob, frob)[1]
                    if lam_text == "0.5":
                        o_lo, o_hi = sqrt_bounds(*metric_bracket(matrix, 6))
                        orthonormal_mismatch |= not (o_lo - tolerance <= ref <= o_hi + tolerance)
            mean_x = sum(xs, Q(0)) / len(xs)
            mean_y = sum(ys, Q(0)) / len(ys)
            slope = sum(((x - mean_x) * (yy - mean_y) for x, yy in zip(xs, ys)), Q(0)) / sum(
                ((x - mean_x) ** 2 for x in xs), Q(0)
            )
            authors_fit = Q(AUTHORS_QUEUE_FITS[(name, horizon)][1])
            refits[f"{name}_L{horizon}"] = {
                "fit_from_exact_sigma": decimal_text(slope, 8),
                "authors_fit": AUTHORS_QUEUE_FITS[(name, horizon)][1],
                "exact_exponent": exact_exponents[(name, 6, horizon)],
            }
            refit_within_005 &= abs(slope - exact_exponents[(name, 6, horizon)]) <= Q(5, 100)
            max_refit_deviation = max(max_refit_deviation, abs(slope - authors_fit))
    checks["queue_conditioning_csv_sigma_70_cells_relative_1e-3_coordinate_convention"] = sigma_ok
    checks["queue_conditioning_csv_sigma_lambda_half_relative_1e-9"] = sigma_half_ok
    checks["queue_conditioning_csv_not_in_induced_euclidean_norm"] = orthonormal_mismatch
    sigma_comparison["max_relative_deviation"] = decimal_text(max_sigma_deviation, 3)
    sigma_comparison["min_sigma_over_frobenius_norm"] = decimal_text(conditioning_ratio, 3)
    checks["queue_exponent_fits_from_exact_sigma_within_0_05_of_exact_exponent"] = refit_within_005
    refits["max_abs_difference_exact_data_fit_vs_authors_fit"] = decimal_text(max_refit_deviation, 4)

    # Exclusion of WENO5-LIN + SSP-RK3 at L = 1.
    exclusion: dict[str, Any] = {}
    exclusion_ok = True
    t_matrix = [[Q(1), Q(0), Q(0)], [Q(0), Q(1), Q(0)], [Q(0), Q(0), Q(1)], [Q(-1), Q(-1), Q(-1)]]
    tt = [[sum((t_matrix[k][i] * t_matrix[k][j] for k in range(4)), Q(0)) for j in range(3)] for i in range(3)]
    tt_poly = charpoly(tt)
    norm_t_is_two = tt_poly == [Q(-4), Q(9), Q(-6), Q(1)]  # (z-1)^2 (z-4)
    for children in (6, 8):
        powers = laurent_powers(operators["weno5l_rk3"], 1)
        block = local_polynomial_blocks(powers, 8, children, 1)[1]
        live = [k for k, row in enumerate(block) if any(row)]
        row_k = block[0]
        others = [block[7], block[1], block[2]]
        conservation = all(
            _sum_polys([row_k[j]] + [o[j] for o in others]) == () for j in range(children - 1)
        )
        cubic_only = all(not p or (len(p) > 3 and p[:3] == (Q(0), Q(0), Q(0))) for p in block[2]) and any(
            len(p) == 4 for p in block[2]
        )
        bound_ok = True
        for lam in lam_seq:
            matrix = [[peval(p, lam) for p in row] for row in (block[k] for k in live)]
            _, lo, hi, _ = gram_bracket(matrix)
            row_norm = sum((peval(p, lam) ** 2 for p in block[2]), Q(0))
            bound_ok &= hi <= 4 * row_norm
        gamma = exact_exponents[("weno5l_rk3", children, 1)]
        exclusion[f"r{children}"] = {
            "nonzero_parents": live,
            "smith_exponents": conditioning[f"weno5l_rk3_P8_r{children}"][0]["smith_exponents"],
            "rho_1": conditioning[f"weno5l_rk3_P8_r{children}"][0]["q"],
            "exact_exponent": gamma,
        }
        exclusion_ok &= (
            live == [0, 1, 2, 7] and conservation and cubic_only and bound_ok and norm_t_is_two
            and gamma == 3 and conditioning[f"weno5l_rk3_P8_r{children}"][0]["L_q"] == 1
        )
    checks["rk3_exclusion_exponent_three_not_one_at_L1"] = exclusion_ok

    # ------------------------------------------------------------------
    # Erasure rates and the N^{-2s} law (item 7).
    erasure: dict[str, Any] = {}
    within_claimed = True
    csv_one_minus_rho_ok = True
    argmax_ok = True
    exact_u: dict[tuple[str, str, int], tuple[Q, Q]] = {}
    exact_ratio: dict[tuple[str, str, int], tuple[Q, Q]] = {}
    max_ratio_csv_deviation = Q(0)
    series_agrees = True
    for name in STABLE:
        for lam, lam_text in ((Q(1, 2), "0.5"), (Q(9, 10), "0.9")):
            a_val = peval(leading[name], lam)
            two_s = orders[name]
            for n in (24, 48, 96, 192, 384):
                info = contraction_factor(operators[name], lam, n)
                argmax_ok &= info["max_at_k1_strict"]
                d_lo, d_hi = info["defect"]
                rho_lo, rho_hi = sqrt_bounds(1 - d_hi, 1 - d_lo)
                u_lo, u_hi = 1 - rho_hi, 1 - rho_lo
                exact_u[(name, lam_text, n)] = (u_lo, u_hi)
                th_lo, th_hi = theta_one(n)
                ratio_lo = u_lo / (a_val / 2 * th_hi**two_s)
                ratio_hi = u_hi / (a_val / 2 * th_lo**two_s)
                exact_ratio[(name, lam_text, n)] = (ratio_lo, ratio_hi)
                _, ref_u, ref_ratio = AUTHORS_ERASURE_RATES[(name, lam_text, n)]
                u_mid = (u_lo + u_hi) / 2
                ratio_mid = (ratio_lo + ratio_hi) / 2
                ulps = abs(Q(ref_u) - u_mid) * 2**53
                # 1 - rho for a double rho just below 1 is a multiple of 2^-53.
                csv_one_minus_rho_ok &= ulps <= 8
                max_ratio_csv_deviation = max(max_ratio_csv_deviation, abs(Q(ref_ratio) - ratio_mid))
                erasure[f"{name}_lambda_{lam_text}_N{n}"] = {
                    "one_minus_rho": decimal_text(u_mid, 12),
                    "ratio": decimal_text(ratio_mid, 12),
                    "authors_one_minus_rho": ref_u,
                    "authors_ratio": ref_ratio,
                    "one_minus_rho_in_units_of_2^-53": decimal_text(u_mid * 2**53, 8),
                    "authors_error_in_units_of_2^-53": decimal_text(ulps, 3) if ulps else "0",
                }
                if n == 384:
                    within_claimed &= Q(9999, 10**4) <= ratio_lo and ratio_hi <= Q(10009, 10**4)
                    s_lo, s_hi = defect_by_theta_series(operators[name], lam, n)
                    series_agrees &= s_lo <= d_hi and d_lo <= s_hi and s_hi - s_lo < Q(1, 10**40)
    checks["erasure_N384_defect_theta_series_route_overlaps_cosine_route"] = series_agrees
    checks["erasure_maximum_attained_at_k_1_strictly_all_N"] = argmax_ok
    checks["erasure_csv_one_minus_rho_within_8_units_of_2^-53"] = csv_one_minus_rho_ok
    checks["erasure_ratio_N384_all_schemes_both_lambda_in_0_9999_1_0009"] = within_claimed

    ub5 = erasure["ub5_lambda_0.9_N384"]
    ub5_lo, ub5_hi = exact_ratio[("ub5", "0.9", 384)]
    checks["ub5_lambda_0_9_N384_ratio_is_0_99994"] = (
        Q(99994, 10**5) - Q(1, 10**5) < ub5_lo and ub5_hi < Q(99994, 10**5) + Q(1, 10**5)
        and ub5_hi - ub5_lo < Q(1, 10**30)
    )
    # Next-order prediction: ratio = 1 + (B/A) theta^2 + (C/A) theta^4 + O(theta^6), with
    # 1 - |g|^2 = A theta^6 + B theta^8 + C theta^10 + ...
    series = defect_series_by_moments(operators["ub5"], 5)
    a_ub5, b_ub5, c_ub5 = (peval(series[i], Q(9, 10)) for i in (2, 3, 4))
    th_lo, _ = theta_one(384)
    predicted_correction = b_ub5 / a_ub5 * th_lo**2 + c_ub5 / a_ub5 * th_lo**4
    actual_correction = (ub5_lo + ub5_hi) / 2 - 1
    checks["ub5_next_order_correction_matches_theta8_term"] = (
        abs(actual_correction - predicted_correction) < Q(1, 10**9)
    )
    erasure["max_abs_ratio_difference_authors_csv_vs_exact"] = decimal_text(max_ratio_csv_deviation, 4)
    for lam_text in ("0.5", "0.9"):
        lo, hi = exact_ratio[("ub5", lam_text, 384)]
        erasure[f"ub5_lambda_{lam_text}_N384"]["ratio_20_digits"] = decimal_text((lo + hi) / 2, 20)
        erasure[f"ub5_lambda_{lam_text}_N384"]["ratio_enclosure_width"] = decimal_text(hi - lo, 2)
    # rem:coefficient-structure (iii): at lambda = 1/2 the quartet gives two values of
    # 1 - rho_N (LW = BW exactly, FR = UB3 exactly) within 3e-5 relative at N = 384.
    half = Q(1, 2)
    same_lw_bw = all(
        peval(g_lw[m], half) == peval(g_bw[m], half) for m in g_lw
    )
    pair_lw = exact_u[("lw", "0.5", 384)]
    pair_fr = exact_u[("fromm", "0.5", 384)]
    checks["quartet_two_values_at_lambda_half_within_3e-5"] = (
        same_lw_bw
        and exact_u[("lw", "0.5", 384)] == exact_u[("bw", "0.5", 384)]
        and exact_u[("fromm", "0.5", 384)] == exact_u[("ub3", "0.5", 384)]
        and abs(pair_fr[1] - pair_lw[0]) / pair_lw[0] < Q(3, 10**5)
    )
    # Figure fig:erasure-flattening caption: UW1 at N=64, lambda=1/2.
    info64 = contraction_factor(operators["uw1"], half, 64)
    d_lo, d_hi = info64["defect"]
    rho64 = sqrt_bounds(1 - d_hi, 1 - d_lo)
    steps = (ln_bounds(Q(10) ** 17)[0] / -ln_bounds(rho64[0])[0], ln_bounds(Q(10) ** 17)[1] / -ln_bounds(rho64[1])[1])
    checks["uw1_N64_rho_0_99880_and_1e-17_after_3_25e4_steps"] = (
        info64["max_at_k1_strict"]
        and significant_round((rho64[0] + rho64[1]) / 2, 5) == Q(99880, 10**5)
        and significant_round((steps[0] + steps[1]) / 2, 3) == Q(32500)
    )
    erasure["uw1_lambda_0.5_N64"] = {
        "rho_N": decimal_text((rho64[0] + rho64[1]) / 2, 12),
        "steps_to_1e-17": decimal_text((steps[0] + steps[1]) / 2, 8),
    }
    ub5_finding = {
        "one_minus_rho_N": ub5["one_minus_rho"],
        "ratio": ub5["ratio"],
        "B_over_A": decimal_text(b_ub5 / a_ub5, 12),
        "predicted_next_order_correction": decimal_text(predicted_correction, 6),
        "authors_csv_ratio": ub5["authors_ratio"],
        "authors_csv_one_minus_rho": ub5["authors_one_minus_rho"],
    }
    discrepancies.append(
        {
            "location": "article.tex line 1010 (E4c), label sec:exp-erasure",
            "claim": "ratio at N=384 in [0.9999,1.0009] except UB5 at lambda=0.9 (0.989, a visible next-order correction at 2s=6)",
            "finding": (
                f"rigorous value for UB5 at lambda=0.9 is {ub5['ratio']} (1-rho_N = {ub5['one_minus_rho']}), "
                "inside [0.9999, 1.0009]; the next-order theta^8 term predicts a correction of "
                f"{decimal_text(predicted_correction, 4)}, not -0.011; 0.989 is a double-precision artifact "
                f"(1-rho_N is about {ub5['one_minus_rho_in_units_of_2^-53']} units of 2^-53 and the CSV value is 3 units low)"
            ),
        }
    )
    rk3_half = erasure["weno5l_rk3_lambda_0.5_N384"]
    checks["rk3_lambda_half_N384_ratio_outside_1e-4_of_unity"] = (
        exact_ratio[("weno5l_rk3", "0.5", 384)][0] - 1 > Q(1, 10**4)
    )
    checks["interpolation_members_lambda_half_N384_ratio_within_1e-4_of_unity"] = all(
        abs(exact_ratio[(name, "0.5", 384)][0] - 1) < Q(1, 10**4)
        and abs(exact_ratio[(name, "0.5", 384)][1] - 1) < Q(1, 10**4)
        for name in ("uw1", "lw", "bw", "fromm", "ub3", "ub5")
    )
    discrepancies.append(
        {
            "location": "article.tex line 745 (proof of prop:contraction-general)",
            "claim": "the measured ratio is within 1e-4 of unity at N=384 for every scheme at lambda=1/2",
            "finding": (
                f"exact ratio for WENO5-LIN+SSP-RK3 at lambda=1/2, N=384 is {rk3_half['ratio']} "
                "(deviation 8.3e-4); the 1e-4 statement holds for the six interpolation members only"
            ),
        }
    )
    # Slopes over the last three N (least squares over equally spaced log N).
    slopes: dict[str, Any] = {}
    slope_values: dict[tuple[str, str], Q] = {}
    ln4 = ln_bounds(Q(4))
    for name in STABLE:
        for lam_text in ("0.5", "0.9"):
            lo96, hi96 = exact_u[(name, lam_text, 96)]
            lo384, hi384 = exact_u[(name, lam_text, 384)]
            # least squares over three equally spaced log N is (y_384 - y_96)/ln 4
            s_lo = (ln_bounds(lo384)[0] - ln_bounds(hi96)[1]) / ln4[0]
            s_hi = (ln_bounds(hi384)[1] - ln_bounds(lo96)[0]) / ln4[1]
            mid = (min(s_lo, s_hi) + max(s_lo, s_hi)) / 2
            slope_values[(name, lam_text)] = mid
            slopes[f"{name}_lambda_{lam_text}"] = {
                "exact_slope": decimal_text(mid, 8),
                "authors_slope": AUTHORS_ERASURE_SLOPES[(name, lam_text)],
            }
    checks["erasure_slopes_match_minus_2s_to_three_decimals_except_ub5_lambda_0_9"] = all(
        abs(slope_values[(n, lt)] + orders[n]) < Q(5, 10**4)
        for n in STABLE
        if n != "weno5l_rk3"
        for lt in ("0.5", "0.9")
        if (n, lt) != ("ub5", "0.9")
    )
    checks["erasure_slopes_rk3_round_to_minus_4_009_and_minus_4_001"] = (
        abs(slope_values[("weno5l_rk3", "0.5")] - Q(-4009, 1000)) < Q(5, 10**4)
        and abs(slope_values[("weno5l_rk3", "0.9")] - Q(-4001, 1000)) < Q(5, 10**4)
    )
    # UB5 at lambda=0.5 differs by about 1.5e-5: its N=384 double is correctly rounded but
    # 1 - rho_N is only 844 units of 2^-53, a 2e-5 quantization of the input.
    checks["erasure_slopes_reproduce_authors_fits_except_ub5_lambda_0_9"] = all(
        abs(slope_values[key] - Q(text)) < Q(5, 10**5)
        for key, text in AUTHORS_ERASURE_SLOPES.items()
        if key != ("ub5", "0.9")
    )
    ub5_slope = slope_values[("ub5", "0.9")]
    checks["ub5_lambda_0_9_exact_slope_rounds_to_minus_5_999"] = (
        Q(-59995, 10**4) <= ub5_slope < Q(-59985, 10**4)
    )
    discrepancies.append(
        {
            "location": "article.tex line 1010 (E4c) and results/erasure_rate_fits.csv (ub5, 0.9)",
            "claim": "log-log slopes on the last three N match -2s to three decimals for every stable scheme (-6.000)",
            "finding": (
                f"for UB5 at lambda=0.9 the authors' fit is {AUTHORS_ERASURE_SLOPES[('ub5', '0.9')]} "
                f"(contaminated by the N=384 float value) and the exact slope is "
                f"{slopes['ub5_lambda_0.9']['exact_slope']}; neither rounds to -6.000 (the exact one "
                "rounds to -5.999); all other cells reproduce the authors' fits to 1e-5"
            ),
        }
    )

    # Forward Euler: rho_N > 1, flat in N.
    fe_cells = {}
    fe_ok = True
    for lam, lam_text in ((Q(1, 2), "0.5"), (Q(9, 10), "0.9")):
        coeffs = at_lambda(operators["weno5l"], lam)
        gm = {0: sum((c * c for c in coeffs.values()), Q(0))}
        for m in range(1, 6):
            gm[m] = 2 * sum((coeffs[s] * coeffs.get(s + m, Q(0)) for s in coeffs), Q(0))
        for n in (24, 48, 96, 192, 384):
            defects = [amplitude_defect_interval(gm, k, n) for k in range(1, n)]
            worst = min(defects, key=lambda iv: iv[0])
            d_lo, d_hi = iv_fraction(worst)
            r_lo, r_hi = sqrt_bounds(1 - d_hi, 1 - d_lo)
            ref = Q(AUTHORS_ERASURE_RATES[("weno5l", lam_text, n)][0])
            fe_ok &= r_lo > 1 and abs(ref - (r_lo + r_hi) / 2) < Q(1, 10**14)
            fe_cells[f"lambda_{lam_text}_N{n}"] = decimal_text((r_lo + r_hi) / 2 - 1, 12)
    checks["fe_rho_N_unstable_matches_authors_csv"] = fe_ok

    # ------------------------------------------------------------------
    # tab:half-life at N = 48, lambda = 1/2.
    half_life: dict[str, Any] = {}
    half_ok = True
    uw1_half = None
    for name in STABLE:
        u_lo, u_hi = exact_u[(name, "0.5", 48)]
        rho_lo, rho_hi = 1 - u_hi, 1 - u_lo
        neg_ln_lo = -ln_bounds(rho_hi)[1]
        neg_ln_hi = -ln_bounds(rho_lo)[0]
        n_lo = LN2[0] / neg_ln_hi
        n_hi = LN2[1] / neg_ln_lo
        n_mid = (n_lo + n_hi) / 2
        if name == "uw1":
            uw1_half = n_mid
        table_text, retention, csv_half, csv_rho = AUTHORS_HALF_LIFE[name]
        assert uw1_half is not None
        ratio = n_mid / uw1_half
        mantissa, _, exponent = table_text.partition("e")
        table_value = Q(mantissa) * Q(10) ** int(exponent)
        half_ok &= significant_round(n_mid, 4) == table_value
        half_ok &= abs(Q(csv_half) - n_mid) / n_mid < Q(1, 10**7)
        half_ok &= abs(Q(csv_rho) - (rho_lo + rho_hi) / 2) < Q(1, 10**15)
        # retention entries are rounded to three or four significant digits
        half_ok &= abs(ratio - retention) / retention < Q(5, 10**3)
        half_life[name] = {
            "n_half": decimal_text(n_mid, 10),
            "retention_vs_uw1": decimal_text(ratio, 8),
            "authors_table": table_text,
            "authors_retention": retention,
        }
    checks["half_life_table_and_csv_reproduced"] = half_ok

    check_count = len(checks)
    return {
        "schema": "certified-simulation/coarse-observability-high-order/independent-dissipation-conditioning-erasure/v1",
        "arithmetic": (
            "fractions.Fraction and Python integers; exact polynomial identities in Q[x, lambda]; "
            "exact Smith exponents over Q[lambda]/(lambda^12); exact Sturm brackets; rigorous "
            "outward-rounded intervals (2^-320 grid) for cos(2 pi v/N), pi (Machin), square "
            "roots (integer isqrt) and logarithms (atanh series with tail bounds)"
        ),
        "results": (
            "2s and A(lambda) for all eight operators; interior-stability factorizations; "
            "prop:weno-fe; prop:rk3-stability Bernstein bounds; exact conditioning exponents "
            "and the WENO5-LIN+SSP-RK3 exclusion; rigorous erasure ratios N=24..384; half-lives"
        ),
        "scope": {
            "established": (
                "Exact asymptotic exponents of sigma+_min(O_L iota_K) as lambda->0 for the seven "
                "single-stage operators at (P,r)=(8,6) and (8,8), L=1..r-1, from lambda-adic Smith "
                "forms of the polynomial matrix; they equal L_q in every case, so conj:conditioning "
                "holds at these two grids. For WENO5-LIN+SSP-RK3 at L=1 the exponent is exactly 3."
            ),
            "numerical_evidence": (
                "Exact Sturm brackets of sigma+_min at lambda=1e-2..1e-5 and their decade slopes; "
                "the conjecture for general (P, r) remains unproved."
            ),
            "convention": (
                "Singular values are reported in the coordinates xi_1..xi_{r-1} of the zero-mean "
                "profile (basis e_j - e_0), the convention that reproduces queue_conditioning.csv; "
                "the induced Euclidean norm changes each singular value by a factor in [1, sqrt(r)] "
                "and no exponent."
            ),
            "independence": (
                "No file under the authors' src/ was opened; reference values are transcribed "
                "from results/*.csv and results/tables/*.tex only for comparison."
            ),
        },
        "dissipation": dissipation,
        "rk3_stability_certificate": stability,
        "fe_interior_maximum": fe_max,
        "fe_rho_N_minus_one": fe_cells,
        "conditioning": conditioning,
        "conditioning_exclusion": exclusion,
        "queue_exponent_refits_from_exact_sigma": refits,
        "queue_sigma_comparison": sigma_comparison,
        "erasure": erasure,
        "erasure_slopes": slopes,
        "ub5_lambda_0_9_N384": ub5_finding,
        "half_life": half_life,
        "discrepancies": discrepancies,
        "checks": checks,
        "check_count": check_count,
        "passed": sum(checks.values()),
        "verdict": "pass" if all(checks.values()) else "fail",
    }


def _sum_polys(polys: list[Poly]) -> Poly:
    total: Poly = ()
    for p in polys:
        total = padd(total, p)
    return total


def canonical_bytes(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
        "utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "artifacts"
        / "dissipation-conditioning-erasure-certificate.json",
    )
    args = parser.parse_args()
    document = build_certificate()
    payload = canonical_bytes(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(
        f"checks={document['passed']}/{document['check_count']} "
        f"verdict={document['verdict']} "
        f"sha256={hashlib.sha256(payload).hexdigest()} "
        f"output={args.output.as_posix()}"
    )


if __name__ == "__main__":
    main()

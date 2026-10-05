#!/usr/bin/env python3
"""Independent exact certificate: rank ladders, invisible subspace, local
observability, and spectral-collision sets for the paper "Coarse Observability
Across High-Order Finite-Volume Reconstructions" (Polemitis, Kokkinakis,
Christakis, Drikakis).

Every scheme coefficient is derived here from the manuscript text alone:

* the five semi-Lagrangian members from the Lagrange formula and node sets of
  Appendix A.1, c_{-d}(lam) = prod_{d' != d} (-lam - d')/(d - d');
* Fromm as the arithmetic mean of Lax-Wendroff and Beam-Warming (Sec. 2.2);
* WENO5-LIN + FE from the five-point linear-weights interface flux of
  Appendix A.2 and u_i - lam (f_{i+1/2} - f_{i-1/2});
* WENO5-LIN + SSP-RK3 from the three Shu-Osher stages applied to the
  forward-Euler operator, checked against I + Z + Z^2/2 + Z^3/6.

The script replays, in exact rational arithmetic:

* thm:rank-universal: exact ranks over Q (Gaussian elimination over
  Fractions) of O_L = [R; RA; ...; RA^L] for L = 0..r+1, for all eight fully
  discrete operators, against both the general class-count formula
  sum_m min(L+1, d_m) and the universal ladder P + (P-1) min(L, r-1); the class
  node counts d_m are exact, from zero tests in Q(omega) = Q[x]/Phi_N(x);
  collision controls (the central average, lam = 0) exercise the general
  formula where the ladder must drop;
* thm:invisible-general: nullity of the saturated observation matrix and the
  explicit repeated zero-mean profiles as its basis;
* prop:local-rank, lem:local-upper, conj:local-rank: single-parent ranks
  rho(L) against sigma*L, r-1, and the wrap-around bound, and against the
  authors' local_observability.csv;
* prop:collision-certificate: whole-interval emptiness of the collision set at
  (P, r) in {(8,6), (8,8)} by exact cyclotomic valuations and Bernstein
  coefficients of |delta~|^2 enclosed with rigorous fixed-point intervals,
  compared value by value with collision_certificate_exact.csv;
* cor:detection-delay(ii) on the E5 compensated pairs.

The script is self-contained, deterministic, and uses only the Python
standard library.  It never reads the authors' code; their published outputs
are embedded below as transcribed reference data, and the companion test
re-reads the published files to confirm each transcription.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from fractions import Fraction as Q
from math import comb
from pathlib import Path
from typing import Any

type Poly = tuple[Q, ...]
type Operator = dict[int, Poly]
type Interval = tuple[int, int]

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


def peval(p: Poly, x: Q) -> Q:
    acc = Q(0)
    for coeff in reversed(p):
        acc = acc * x + coeff
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
    """Number of distinct real roots of p in the half-open interval (lo, hi]
    (Sturm's theorem on the square-free part, valid at root endpoints)."""
    p = squarefree(ptrim(p))
    if len(p) <= 1:
        return 0
    seq = [p, pderiv(p)]
    while len(seq[-1]) > 1:
        _, rem = pdivmod(seq[-2], seq[-1])
        if not rem:
            break
        seq.append(pscale(rem, Q(-1)))

    def changes(x: Q) -> int:
        signs = [peval(f, x) for f in seq]
        signs = [s for s in signs if s != 0]
        return sum(1 for a, b in zip(signs, signs[1:]) if (a > 0) != (b > 0))

    return changes(lo) - changes(hi)


def roots_in_open_unit_interval(p: Poly) -> int:
    """Distinct real roots of p in (0, 1), after removing the factor lam^v."""
    p = ptrim(p)
    while p and p[0] == 0:
        p = p[1:]
    if not p:
        raise ValueError("zero polynomial")
    return sturm_roots_in(p, Q(0), Q(1)) - (1 if peval(p, Q(1)) == 0 else 0)


def poly_text(p: Poly) -> str:
    if not p:
        return "0"
    terms = []
    for i, c in enumerate(p):
        if c:
            terms.append(f"{c}" if i == 0 else f"{c}*lam^{i}")
    return " + ".join(terms)


# ----------------------------------------------------------------------------
# Operators as Laurent polynomials in the shift S, (S^s u)_i = u_{i-s}.


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
    """Appendix A.1: c_{-d}(lam) = prod_{d' != d} (-lam - d')/(d - d')."""
    ops: Operator = {}
    for d in nodes:
        poly: Poly = (Q(1),)
        for dp in nodes:
            if dp != d:
                poly = pmul(poly, (Q(-dp, d - dp), Q(-1, d - dp)))
        ops[-d] = poly
    return dict(sorted(ops.items()))


WENO5_FLUX = {2: Q(2, 60), 1: Q(-13, 60), 0: Q(47, 60), -1: Q(27, 60), -2: Q(-3, 60)}
"""Appendix A.2: f_{i+1/2} = sum_s h_s u_{i-s} with (2,-13,47,27,-3)/60."""


def weno5_forward_euler() -> Operator:
    diff = {s: WENO5_FLUX.get(s, Q(0)) - WENO5_FLUX.get(s - 1, Q(0)) for s in range(-3, 4)}
    ops: Operator = {}
    for s in range(-3, 4):
        poly = ptrim(((Q(1) if s == 0 else Q(0)), -diff[s]))
        if poly:
            ops[s] = poly
    return ops


def ssp_rk3(fe: Operator) -> Operator:
    """Shu-Osher SSP-RK3 stages for a linear step u -> fe u."""
    stage1 = fe
    stage2 = op_add(op_scale(IDENTITY, Q(3, 4)), op_scale(op_mul(fe, stage1), Q(1, 4)))
    return op_add(op_scale(IDENTITY, Q(1, 3)), op_scale(op_mul(fe, stage2), Q(2, 3)))


SCHEME_ORDER = ("uw1", "lw", "bw", "fromm", "ub3", "ub5", "weno5l", "weno5l_rk3")
DISPLAY = {
    "uw1": "UW1",
    "lw": "LW",
    "bw": "BW",
    "fromm": "FR",
    "ub3": "UB3",
    "ub5": "UB5",
    "weno5l": "WENO5-LIN+FE",
    "weno5l_rk3": "WENO5-LIN+SSP-RK3",
}
INTERPOLATION_MEMBERS = ("uw1", "lw", "bw", "fromm", "ub3", "ub5")


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


def reaches(coeffs: dict[int, Any]) -> tuple[int, int]:
    support = [s for s, v in coeffs.items() if v]
    return max(0, max(support)), max(0, -min(support))


# ----------------------------------------------------------------------------
# The authors' generated coefficient fragment (results/tables/coefficients.tex),
# embedded verbatim line by line; the test confirms each line in the file.

AUTHORS_COEFFICIENT_LINES = (
    r"UW1 & $1$ & $\lambda$ \\",
    r" & $0$ & $1 - \lambda$ \\",
    r"LW & $1$ & $\tfrac{1}{2}\lambda + \tfrac{1}{2}\lambda^{2}$ \\",
    r" & $0$ & $1 - \lambda^{2}$ \\",
    r" & $-1$ & $-\tfrac{1}{2}\lambda + \tfrac{1}{2}\lambda^{2}$ \\",
    r"BW & $2$ & $-\tfrac{1}{2}\lambda + \tfrac{1}{2}\lambda^{2}$ \\",
    r" & $1$ & $2\lambda - \lambda^{2}$ \\",
    r" & $0$ & $1 - \tfrac{3}{2}\lambda + \tfrac{1}{2}\lambda^{2}$ \\",
    r"FR & $2$ & $-\tfrac{1}{4}\lambda + \tfrac{1}{4}\lambda^{2}$ \\",
    r" & $1$ & $\tfrac{5}{4}\lambda - \tfrac{1}{4}\lambda^{2}$ \\",
    r" & $0$ & $1 - \tfrac{3}{4}\lambda - \tfrac{1}{4}\lambda^{2}$ \\",
    r" & $-1$ & $-\tfrac{1}{4}\lambda + \tfrac{1}{4}\lambda^{2}$ \\",
    r"UB3 & $2$ & $-\tfrac{1}{6}\lambda + \tfrac{1}{6}\lambda^{3}$ \\",
    r" & $1$ & $\lambda + \tfrac{1}{2}\lambda^{2} - \tfrac{1}{2}\lambda^{3}$ \\",
    r" & $0$ & $1 - \tfrac{1}{2}\lambda - \lambda^{2} + \tfrac{1}{2}\lambda^{3}$ \\",
    r" & $-1$ & $-\tfrac{1}{3}\lambda + \tfrac{1}{2}\lambda^{2} - \tfrac{1}{6}\lambda^{3}$ \\",
    r"UB5 & $3$ & $\tfrac{1}{30}\lambda - \tfrac{1}{24}\lambda^{3} + \tfrac{1}{120}\lambda^{5}$ \\",
    r" & $2$ & $-\tfrac{1}{4}\lambda - \tfrac{1}{24}\lambda^{2} + \tfrac{7}{24}\lambda^{3}"
    r" + \tfrac{1}{24}\lambda^{4} - \tfrac{1}{24}\lambda^{5}$ \\",
    r" & $1$ & $\lambda + \tfrac{2}{3}\lambda^{2} - \tfrac{7}{12}\lambda^{3}"
    r" - \tfrac{1}{6}\lambda^{4} + \tfrac{1}{12}\lambda^{5}$ \\",
    r" & $0$ & $1 - \tfrac{1}{3}\lambda - \tfrac{5}{4}\lambda^{2} + \tfrac{5}{12}\lambda^{3}"
    r" + \tfrac{1}{4}\lambda^{4} - \tfrac{1}{12}\lambda^{5}$ \\",
    r" & $-1$ & $-\tfrac{1}{2}\lambda + \tfrac{2}{3}\lambda^{2} - \tfrac{1}{24}\lambda^{3}"
    r" - \tfrac{1}{6}\lambda^{4} + \tfrac{1}{24}\lambda^{5}$ \\",
    r" & $-2$ & $\tfrac{1}{20}\lambda - \tfrac{1}{24}\lambda^{2} - \tfrac{1}{24}\lambda^{3}"
    r" + \tfrac{1}{24}\lambda^{4} - \tfrac{1}{120}\lambda^{5}$ \\",
    r"c_{3}=\tfrac{1}{30}\lambda,\quad",
    r"c_{2}=-\tfrac{1}{4}\lambda,\quad",
    r"c_{1}=\lambda,\quad",
    r"c_{0}=1 - \tfrac{1}{3}\lambda,\quad",
    r"c_{-1}=-\tfrac{1}{2}\lambda,\quad",
    r"c_{-2}=\tfrac{1}{20}\lambda",
    r"$9$ & $\tfrac{1}{162000}\lambda^{3}$ & $1$ & $\lambda - \tfrac{31}{150}\lambda^{2}"
    r" - \tfrac{49}{200}\lambda^{3}$ \\",
    r"$8$ & $-\tfrac{1}{7200}\lambda^{3}$ & $0$ & $1 - \tfrac{1}{3}\lambda"
    r" - \tfrac{329}{720}\lambda^{2} + \tfrac{10211}{64800}\lambda^{3}$ \\",
    r"$7$ & $\tfrac{23}{14400}\lambda^{3}$ & $-1$ & $-\tfrac{1}{2}\lambda"
    r" + \tfrac{13}{60}\lambda^{2} + \tfrac{6253}{72000}\lambda^{3}$ \\",
    r"$6$ & $\tfrac{1}{1800}\lambda^{2} - \tfrac{961}{86400}\lambda^{3}$ & $-2$ &"
    r" $\tfrac{1}{20}\lambda + \tfrac{13}{120}\lambda^{2} - \tfrac{1849}{28800}\lambda^{3}$ \\",
    r"$5$ & $-\tfrac{1}{120}\lambda^{2} + \tfrac{121}{2400}\lambda^{3}$ & $-3$ &"
    r" $-\tfrac{1}{40}\lambda^{2} - \tfrac{9}{800}\lambda^{3}$ \\",
    r"$4$ & $\tfrac{31}{480}\lambda^{2} - \tfrac{427}{3000}\lambda^{3}$ & $-4$ &"
    r" $\tfrac{1}{800}\lambda^{2} + \tfrac{7}{1200}\lambda^{3}$ \\",
    r"$3$ & $\tfrac{1}{30}\lambda - \tfrac{47}{180}\lambda^{2} + \tfrac{9467}{43200}\lambda^{3}$"
    r" & $-5$ & $-\tfrac{1}{1600}\lambda^{3}$ \\",
    r"$2$ & $-\tfrac{1}{4}\lambda + \tfrac{17}{30}\lambda^{2} - \tfrac{449}{9600}\lambda^{3}$"
    r" & $-6$ & $\tfrac{1}{48000}\lambda^{3}$ \\",
)

_TERM = re.compile(r"^(?:\\tfrac\{(\d+)\}\{(\d+)\}|(\d+))?(\\lambda(?:\^\{(\d+)\})?)?$")


def parse_latex_poly(text: str) -> Poly:
    compact = text.replace(" ", "")
    coeffs: dict[int, Q] = {}
    for raw in re.split(r"(?=[+-])", compact):
        if not raw:
            continue
        sign = -1 if raw[0] == "-" else 1
        body = raw.lstrip("+-")
        match = _TERM.match(body)
        if match is None or not body:
            raise ValueError(f"cannot parse term {raw!r} in {text!r}")
        num, den, integer, lam, power = match.groups()
        coeff = Q(int(num), int(den)) if num else Q(int(integer)) if integer else Q(1)
        degree = (int(power) if power else 1) if lam else 0
        coeffs[degree] = coeffs.get(degree, Q(0)) + sign * coeff
    top = max(coeffs) if coeffs else -1
    return ptrim(coeffs.get(i, Q(0)) for i in range(top + 1))


def parse_authors_coefficients() -> dict[str, Operator]:
    tables: dict[str, Operator] = {key: {} for key in SCHEME_ORDER}
    name_of = {"UW1": "uw1", "LW": "lw", "BW": "bw", "FR": "fromm", "UB3": "ub3", "UB5": "ub5"}
    current = None
    for line in AUTHORS_COEFFICIENT_LINES:
        if line.startswith("c_{"):
            match = re.match(r"c_\{(-?\d+)\}=(.*?)(?:,\\quad)?$", line)
            assert match is not None
            tables["weno5l"][int(match.group(1))] = parse_latex_poly(match.group(2))
        elif line.startswith("$"):
            cells = [c.strip().strip("$") for c in line.rstrip("\\").split("&")]
            for s_cell, p_cell in ((cells[0], cells[1]), (cells[2], cells[3])):
                tables["weno5l_rk3"][int(s_cell)] = parse_latex_poly(p_cell)
        else:
            cells = [c.strip() for c in line.rstrip("\\").split("&")]
            if cells[0]:
                current = name_of[cells[0]]
            assert current is not None
            tables[current][int(cells[1].strip("$"))] = parse_latex_poly(cells[2].strip("$"))
    return {k: dict(sorted(v.items())) for k, v in tables.items()}


# ----------------------------------------------------------------------------
# Exact arithmetic in Q(omega) = Q[x]/Phi_N(x), omega = exp(2 pi i/N).


def _int_poly_divmod(num: list[int], den: list[int]) -> tuple[list[int], list[int]]:
    num = list(num)
    quot = [0] * (len(num) - len(den) + 1)
    for i in range(len(num) - len(den), -1, -1):
        coeff = num[i + len(den) - 1]
        quot[i] = coeff
        if coeff:
            for j, d in enumerate(den):
                num[i + j] -= coeff * d
    return quot, num[: len(den) - 1]


_CYCLOTOMIC: dict[int, list[int]] = {}


def cyclotomic_polynomial(n: int) -> list[int]:
    if n not in _CYCLOTOMIC:
        poly = [-1] + [0] * (n - 1) + [1]
        for d in range(1, n):
            if n % d == 0:
                poly, rem = _int_poly_divmod(poly, cyclotomic_polynomial(d))
                assert not any(rem)
        _CYCLOTOMIC[n] = poly
    return _CYCLOTOMIC[n]


class CyclotomicField:
    """Power-basis coordinates of elements sum_e a_e omega^e."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.phi = cyclotomic_polynomial(n)
        self.degree = len(self.phi) - 1
        table = []
        vec = [0] * self.degree
        vec[0] = 1
        for _ in range(n):
            table.append(tuple(vec))
            lead = vec[-1]
            vec = [0] + vec[:-1]
            if lead:
                vec = [v - lead * c for v, c in zip(vec, self.phi[:-1])]
        self.monomial = table

    def element(self, terms: dict[int, Q]) -> tuple[Q, ...]:
        acc = [Q(0)] * self.degree
        for exponent, coeff in terms.items():
            if coeff:
                for i, c in enumerate(self.monomial[exponent % self.n]):
                    if c:
                        acc[i] += coeff * c
        return tuple(acc)


def symbol_element(field: CyclotomicField, coeffs: dict[int, Q], k: int) -> tuple[Q, ...]:
    """g_k = sum_s c_s omega^{-k s} in power-basis coordinates."""
    terms: dict[int, Q] = {}
    for s, c in coeffs.items():
        e = (-k * s) % field.n
        terms[e] = terms.get(e, Q(0)) + c
    return field.element(terms)


def class_node_counts(coeffs: dict[int, Q], parents: int, children: int) -> list[int]:
    """d_0 = 1 and, for m != 0, the number of distinct class symbols (exact)."""
    field = CyclotomicField(parents * children)
    counts = [1]
    for m in range(1, parents):
        values = {symbol_element(field, coeffs, m + j * parents) for j in range(children)}
        counts.append(len(values))
    return counts


# ----------------------------------------------------------------------------
# Exact observation ranks over Q.


def restriction_rows(parents: int, children: int) -> list[list[Q]]:
    size = parents * children
    rows = []
    for k in range(parents):
        row = [Q(0)] * size
        for j in range(children):
            row[k * children + j] = Q(1, children)
        rows.append(row)
    return rows


def right_apply(row: list[Q], items: list[tuple[int, Q]]) -> list[Q]:
    """(x A)_n = sum_s c_s x_{n+s}, the transpose action of (A u)_i = sum_s c_s u_{i-s}."""
    size = len(row)
    return [sum((c * row[(n + s) % size] for s, c in items), Q(0)) for n in range(size)]


def apply(items: list[tuple[int, Q]], vec: list[Q]) -> list[Q]:
    size = len(vec)
    return [sum((c * vec[(i - s) % size] for s, c in items), Q(0)) for i in range(size)]


class Echelon:
    """Incremental exact Gaussian elimination over Q with Fraction entries."""

    def __init__(self) -> None:
        self.pivots: list[tuple[int, list[tuple[int, Q]]]] = []

    def add(self, row: list[Q]) -> bool:
        vec = list(row)
        for col, prow in self.pivots:
            factor = vec[col]
            if factor:
                for j, x in prow:
                    vec[j] -= factor * x
        support = [j for j, x in enumerate(vec) if x]
        if not support:
            return False
        lead = support[0]
        inv = 1 / vec[lead]
        self.pivots.append((lead, [(j, vec[j] * inv) for j in support]))
        return True

    @property
    def rank(self) -> int:
        return len(self.pivots)


def observation_rank_ladder(
    coeffs: dict[int, Q], parents: int, children: int, horizon: int
) -> list[int]:
    items = sorted(coeffs.items())
    rows = restriction_rows(parents, children)
    echelon = Echelon()
    ladder = []
    for _ in range(horizon + 1):
        for row in rows:
            echelon.add(row)
        ladder.append(echelon.rank)
        rows = [right_apply(row, items) for row in rows]
    return ladder


def parent_averages(vec: list[Q], parents: int, children: int) -> list[Q]:
    return [
        sum(vec[k * children : (k + 1) * children], Q(0)) / children for k in range(parents)
    ]


def invisible_subspace_report(
    coeffs: dict[int, Q], parents: int, children: int, horizon: int
) -> dict[str, bool]:
    """Repeated zero-mean child profiles: invisible at every tested horizon,
    mapped into themselves by A, and linearly independent."""
    items = sorted(coeffs.items())
    size = parents * children
    basis = []
    for j in range(1, children):
        profile = [Q(0)] * children
        profile[j] = Q(1)
        profile[0] = Q(-1)
        basis.append([profile[i % children] for i in range(size)])
    invisible = True
    invariant = True
    for vec in basis:
        current = vec
        for _ in range(horizon + 1):
            invisible &= all(v == 0 for v in parent_averages(current, parents, children))
            nxt = apply(items, current)
            invariant &= all(
                nxt[i] == nxt[i % children] for i in range(size)
            ) and sum(nxt[:children], Q(0)) == 0
            current = nxt
    echelon = Echelon()
    independent = all(echelon.add(vec) for vec in basis)
    return {"invisible": invisible, "invariant": invariant, "independent": independent}


def local_rank_ladder(
    coeffs: dict[int, Q], parents: int, children: int, horizon: int, host: int = 0
) -> list[int]:
    """rho(L) = rank(O_L iota_K) with iota_K on the basis e_j - e_0 of the
    zero-mean child profiles of parent K (rank is basis independent)."""
    items = sorted(coeffs.items())
    size = parents * children
    columns = []
    for j in range(1, children):
        vec = [Q(0)] * size
        vec[host * children + j] = Q(1)
        vec[host * children] = Q(-1)
        columns.append(vec)
    echelon = Echelon()
    ladder = []
    for _ in range(horizon + 1):
        averages = [parent_averages(col, parents, children) for col in columns]
        for k in range(parents):
            echelon.add([averages[j][k] for j in range(children - 1)])
        ladder.append(echelon.rank)
        columns = [apply(items, col) for col in columns]
    return ladder


def wraparound_bound(m_left: int, m_right: int, children: int, horizon: int) -> int:
    total = sum(-(-t * m_left // children) - (-t * m_right // children) for t in range(1, horizon + 1))
    return min(children - 1, total)


# ----------------------------------------------------------------------------
# Rigorous fixed-point interval arithmetic for cos/sin(2 pi v/N).

BITS = 320
ONE = 1 << BITS


def _floor_div(a: int, b: int) -> int:
    return a // b


def _ceil_div(a: int, b: int) -> int:
    return -((-a) // b)


def _pi_bounds() -> Interval:
    """Machin: pi = 16 atan(1/5) - 4 atan(1/239), alternating-series brackets."""

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


def iv_sub(x: Interval, y: Interval) -> Interval:
    return (x[0] - y[1], x[1] - y[0])


def iv_mul(x: Interval, y: Interval) -> Interval:
    products = (x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1])
    return (min(products) >> BITS, -((-max(products)) >> BITS))


def iv_scale(x: Interval, q: Q) -> Interval:
    n, d = q.numerator, q.denominator
    if n >= 0:
        return (_floor_div(x[0] * n, d), _ceil_div(x[1] * n, d))
    return (_floor_div(x[1] * n, d), _ceil_div(x[0] * n, d))


def iv_fraction(x: Interval) -> tuple[Q, Q]:
    return Q(x[0], ONE), Q(x[1], ONE)


_TRIG: dict[tuple[int, int], tuple[Interval, Interval]] = {}


def cos_sin_turn(v: int, n: int) -> tuple[Interval, Interval]:
    """Enclosures of cos(2 pi v/n) and sin(2 pi v/n): Taylor series at an exact
    fixed-point angle with Lagrange remainder, plus the Lipschitz margin of the
    angle enclosure (|d cos|, |d sin| <= 1)."""
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
        term = (_floor_div(term[0] * x, k * ONE), _ceil_div(term[1] * x, k * ONE))
        if k > 8 and term[1] <= 1:
            break
    margin = term[1] + 2 + (a_hi - a_lo)
    cos_iv = (max(c_lo - margin, -ONE), min(c_hi + margin, ONE))
    sin_iv = (max(s_lo - margin, -ONE), min(s_hi + margin, ONE))
    if mirrored:
        sin_iv = (-sin_iv[1], -sin_iv[0])
    _TRIG[key] = (cos_iv, sin_iv)
    return cos_iv, sin_iv


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
# Whole-interval collision certificate (prop:collision-certificate).


def collision_certificate(op: Operator, parents: int, children: int) -> dict[str, Any]:
    size = parents * children
    field = CyclotomicField(size)
    degree = max(len(p) for p in op.values()) - 1
    by_power = [
        {s: (p[n] if n < len(p) else Q(0)) for s, p in op.items()} for n in range(degree + 1)
    ]
    exact = [[symbol_element(field, by_power[n], k) for n in range(degree + 1)] for k in range(size)]
    trig = [cos_sin_turn(v, size) for v in range(size)]
    pairs = 0
    certified = 0
    max_q_degree = 0
    min_valuation = None
    identically_zero = 0
    worst: tuple[Interval, tuple[int, int]] | None = None
    second_order: list[tuple[int, int, int, int, Q, Q]] = []
    for m in range(1, parents):
        for j in range(children):
            for jj in range(j + 1, children):
                k, kk = m + j * parents, m + jj * parents
                pairs += 1
                nonzero = [n for n in range(degree + 1) if exact[k][n] != exact[kk][n]]
                if not nonzero:
                    identically_zero += 1
                    continue
                a, top = nonzero[0], nonzero[-1]
                min_valuation = a if min_valuation is None else min(min_valuation, a)
                parts = []
                for n in range(a, top + 1):
                    re_iv: Interval = (0, 0)
                    im_iv: Interval = (0, 0)
                    for s, c in by_power[n].items():
                        if not c:
                            continue
                        cos_k, sin_k = trig[(k * s) % size]
                        cos_kk, sin_kk = trig[(kk * s) % size]
                        re_iv = iv_add(re_iv, iv_scale(iv_sub(cos_k, cos_kk), c))
                        im_iv = iv_add(im_iv, iv_scale(iv_sub(sin_kk, sin_k), c))
                    parts.append((re_iv, im_iv))
                q_degree = 2 * (top - a)
                max_q_degree = max(max_q_degree, q_degree)
                q_coeffs: list[Interval] = []
                for d in range(q_degree + 1):
                    acc: Interval = (0, 0)
                    for i in range(max(0, d - (top - a)), min(d, top - a) + 1):
                        x, y = parts[i], parts[d - i]
                        acc = iv_add(acc, iv_add(iv_mul(x[0], y[0]), iv_mul(x[1], y[1])))
                    q_coeffs.append(acc)
                bernstein = []
                for i in range(q_degree + 1):
                    acc = (0, 0)
                    for d in range(i + 1):
                        acc = iv_add(acc, iv_scale(q_coeffs[d], Q(comb(i, d), comb(q_degree, d))))
                    bernstein.append(acc)
                low = min(bernstein, key=lambda iv: iv[0])
                if low[0] > 0:
                    certified += 1
                if worst is None or low[0] < worst[0][0]:
                    worst = (low, (k, kk))
                if a >= 2:
                    q_lo, q_hi = iv_fraction(q_coeffs[0])
                    second_order.append((k, kk, a, q_degree, q_lo, q_hi))
    assert worst is not None
    lo, hi = iv_fraction(worst[0])
    return {
        "pairs": pairs,
        "certified": certified,
        "identically_zero_pairs": identically_zero,
        "min_valuation": min_valuation,
        "max_q_degree": max_q_degree,
        "min_bernstein_lower": lo,
        "min_bernstein_upper": hi,
        "worst_pair": worst[1],
        "second_order_pairs": second_order,
    }


# ----------------------------------------------------------------------------
# Reference data transcribed from the authors' published outputs.

AUTHORS_LOCAL_RANKS = {
    # (scheme, r, lambda): (m_L, m_R, rho(0..r+3)) from local_observability.csv, P = 8.
    ("bw", 6, "0.5"): (2, 0, (0, 1, 2, 3, 5, 5, 5, 5, 5, 5)),
    ("bw", 6, "0.9"): (2, 0, (0, 1, 2, 3, 5, 5, 5, 5, 5, 5)),
    ("bw", 8, "0.5"): (2, 0, (0, 1, 2, 3, 4, 6, 7, 7, 7, 7, 7, 7)),
    ("bw", 8, "0.9"): (2, 0, (0, 1, 2, 3, 4, 6, 7, 7, 7, 7, 7, 7)),
    ("fromm", 6, "0.5"): (2, 1, (0, 2, 4, 5, 5, 5, 5, 5, 5, 5)),
    ("fromm", 6, "0.9"): (2, 1, (0, 2, 4, 5, 5, 5, 5, 5, 5, 5)),
    ("fromm", 8, "0.5"): (2, 1, (0, 2, 4, 6, 7, 7, 7, 7, 7, 7, 7, 7)),
    ("fromm", 8, "0.9"): (2, 1, (0, 2, 4, 6, 7, 7, 7, 7, 7, 7, 7, 7)),
    ("lw", 6, "0.5"): (1, 1, (0, 2, 4, 5, 5, 5, 5, 5, 5, 5)),
    ("lw", 6, "0.9"): (1, 1, (0, 2, 4, 5, 5, 5, 5, 5, 5, 5)),
    ("lw", 8, "0.5"): (1, 1, (0, 2, 4, 6, 7, 7, 7, 7, 7, 7, 7, 7)),
    ("lw", 8, "0.9"): (1, 1, (0, 2, 4, 6, 7, 7, 7, 7, 7, 7, 7, 7)),
    ("ub3", 6, "0.5"): (2, 1, (0, 2, 4, 5, 5, 5, 5, 5, 5, 5)),
    ("ub3", 6, "0.9"): (2, 1, (0, 2, 4, 5, 5, 5, 5, 5, 5, 5)),
    ("ub3", 8, "0.5"): (2, 1, (0, 2, 4, 6, 7, 7, 7, 7, 7, 7, 7, 7)),
    ("ub3", 8, "0.9"): (2, 1, (0, 2, 4, 6, 7, 7, 7, 7, 7, 7, 7, 7)),
    ("ub5", 6, "0.5"): (3, 2, (0, 2, 4, 5, 5, 5, 5, 5, 5, 5)),
    ("ub5", 6, "0.9"): (3, 2, (0, 2, 4, 5, 5, 5, 5, 5, 5, 5)),
    ("ub5", 8, "0.5"): (3, 2, (0, 2, 4, 7, 7, 7, 7, 7, 7, 7, 7, 7)),
    ("ub5", 8, "0.9"): (3, 2, (0, 2, 4, 7, 7, 7, 7, 7, 7, 7, 7, 7)),
    ("uw1", 6, "0.5"): (1, 0, (0, 1, 2, 3, 4, 5, 5, 5, 5, 5)),
    ("uw1", 6, "0.9"): (1, 0, (0, 1, 2, 3, 4, 5, 5, 5, 5, 5)),
    ("uw1", 8, "0.5"): (1, 0, (0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7)),
    ("uw1", 8, "0.9"): (1, 0, (0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7)),
    ("weno5l", 6, "0.5"): (3, 2, (0, 2, 4, 5, 5, 5, 5, 5, 5, 5)),
    ("weno5l", 6, "0.9"): (3, 2, (0, 2, 4, 5, 5, 5, 5, 5, 5, 5)),
    ("weno5l", 8, "0.5"): (3, 2, (0, 2, 4, 7, 7, 7, 7, 7, 7, 7, 7, 7)),
    ("weno5l", 8, "0.9"): (3, 2, (0, 2, 4, 7, 7, 7, 7, 7, 7, 7, 7, 7)),
    ("weno5l_rk3", 6, "0.5"): (9, 6, (0, 3, 5, 5, 5, 5, 5, 5, 5, 5)),
    ("weno5l_rk3", 6, "0.9"): (9, 6, (0, 3, 5, 5, 5, 5, 5, 5, 5, 5)),
    ("weno5l_rk3", 8, "0.5"): (9, 6, (0, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7)),
    ("weno5l_rk3", 8, "0.9"): (9, 6, (0, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7)),
}

AUTHORS_COLLISION_CERTIFICATE = {
    # (scheme, r): (n_pairs, certified, certified_min_q, max_bernstein_cells), P = 8.
    ("uw1", 6): (105, True, "1.0", 1),
    ("lw", 6): (105, True, "0.017037086855465858", 1),
    ("bw", 6): (105, True, "1.0", 1),
    ("fromm", 6): (105, True, "0.9075165176855672", 1),
    ("ub3", 6): (105, True, "0.5487820222673201", 1),
    ("ub5", 6): (105, True, "0.3827763452672109", 1),
    ("weno5l", 6): (105, True, "0.3827763452672109", 1),
    ("weno5l_rk3", 6): (105, True, "0.16419994343317945", 1),
    ("uw1", 8): (196, True, "0.585786437626905", 1),
    ("lw", 8): (196, True, "0.005627861071295757", 1),
    ("bw", 8): (196, True, "0.585786437626905", 1),
    ("fromm", 8): (196, True, "0.5233126323246691", 1),
    ("ub3", 8): (196, True, "0.31590508338973555", 1),
    ("ub5", 8): (196, True, "0.22782365774656357", 1),
    ("weno5l", 8): (196, True, "0.22782365774656357", 1),
    ("weno5l_rk3", 8): (196, True, "0.09269219719868338", 1),
}

AUTHORS_DELAYED_PAIRS = (
    # (scheme, t0, j1, j2, t_detect_predicted, t_detect_measured) from delayed_collisions.csv.
    ("uw1", 1, 0, 5, 1, 1),
    ("uw1", 2, 0, 4, 2, 2),
    ("uw1", 3, 0, 3, 3, 3),
    ("uw1", 4, 0, 2, 4, 4),
    ("uw1", 5, 0, 1, 5, 5),
    ("lw", 1, 1, 5, 1, 1),
    ("lw", 2, 2, 4, 2, 2),
    ("bw", 1, 0, 4, 1, 1),
    ("bw", 2, 0, 2, 2, 2),
    ("fromm", 1, 1, 4, 1, 1),
    ("ub3", 1, 1, 4, 1, 1),
    ("ub5", 1, 2, 3, 1, 1),
    ("weno5l_rk3", 0, 0, 5, 1, 1),
)

AUTHORS_HALF_LIFE_LOCAL_COLUMNS = {
    # tab:half-life / results/tables/half_life.tex: (shielding <=, T_loc) at r = 6.
    "uw1": (5, 5),
    "lw": (2, 3),
    "bw": (2, 4),
    "fromm": (1, 3),
    "ub3": (1, 3),
    "ub5": (1, 3),
    "weno5l_rk3": (0, 2),
}

NOMINAL_TABLE = {
    # tab:scheme-zoo: (m_L, m_R), w.
    "uw1": ((1, 0), 1),
    "lw": ((1, 1), 2),
    "bw": ((2, 0), 2),
    "fromm": ((2, 1), 3),
    "ub3": ((2, 1), 3),
    "ub5": ((3, 2), 5),
    "weno5l": ((3, 2), 5),
    "weno5l_rk3": ((9, 6), 15),
}


def first_detection(
    coeffs: dict[int, Q], parents: int, children: int, j1: int, j2: int, steps: int
) -> tuple[int | None, list[list[Q]]]:
    items = sorted(coeffs.items())
    size = parents * children
    vec = [Q(0)] * size
    vec[j2] += 1
    vec[j1] -= 1
    history = []
    detected = None
    for t in range(steps + 1):
        averages = parent_averages(vec, parents, children)
        history.append(averages)
        if detected is None and any(averages):
            detected = t
        vec = apply(items, vec)
    return detected, history


# ----------------------------------------------------------------------------


def build_certificate() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    discrepancies: list[dict[str, str]] = []
    operators = build_operators()

    # ------------------------------------------------------------------
    # Scheme coefficients derived from the text versus the published tables.
    authors = parse_authors_coefficients()
    for name in SCHEME_ORDER:
        checks[f"coefficients_match_authors_table_{name}"] = operators[name] == authors[name]
    lam_poly: Poly = (Q(0), Q(1))
    one = (Q(1),)
    lw_text = {
        1: pmul(pscale(lam_poly, Q(1, 2)), padd(one, lam_poly)),
        0: psub(one, pmul(lam_poly, lam_poly)),
        -1: pscale(pmul(lam_poly, psub(one, lam_poly)), Q(-1, 2)),
    }
    checks["lax_wendroff_matches_section_2_display"] = operators["lw"] == lw_text
    half_lam = pscale(lam_poly, Q(1, 2))
    half_lam2 = pscale(pmul(lam_poly, lam_poly), Q(1, 2))
    bw_text = op_add(
        op_add(IDENTITY, {0: pscale(half_lam, Q(-3)), 1: pscale(half_lam, Q(4)), 2: pscale(half_lam, Q(-1))}),
        {0: half_lam2, 1: pscale(half_lam2, Q(-2)), 2: half_lam2},
    )
    checks["beam_warming_matches_section_2_display"] = operators["bw"] == bw_text
    checks["fromm_c0_factorization_appendix_a"] = operators["fromm"][0] == pscale(
        pmul(padd((Q(4),), lam_poly), psub(one, lam_poly)), Q(1, 4)
    )
    fe = operators["weno5l"]
    z = op_add(fe, op_scale(IDENTITY, Q(-1)))
    taylor = op_add(
        op_add(IDENTITY, z),
        op_add(op_scale(op_mul(z, z), Q(1, 2)), op_scale(op_mul(z, op_mul(z, z)), Q(1, 6))),
    )
    checks["ssp_rk3_shu_osher_equals_third_order_taylor"] = operators["weno5l_rk3"] == taylor
    rk3 = operators["weno5l_rk3"]
    checks["rk3_outer_coefficients_remark_effective_support"] = (
        rk3[9] == (Q(0), Q(0), Q(0), Q(1, 162000)) and rk3[-6] == (Q(0), Q(0), Q(0), Q(1, 48000))
    )
    for name in SCHEME_ORDER:
        op = operators[name]
        total: Poly = ()
        for p in op.values():
            total = padd(total, p)
        checks[f"consistency_sum_of_coefficients_{name}"] = total == (Q(1),)
        (m_left, m_right), width = NOMINAL_TABLE[name]
        checks[f"nominal_reach_table_scheme_zoo_{name}"] = (
            reaches(op) == (m_left, m_right) and m_left + m_right == width
        )
        checks[f"identity_at_lambda_zero_{name}"] = at_lambda(op, Q(0)) == {0: Q(1)}
    for name in INTERPOLATION_MEMBERS:
        op = operators[name]
        checks[f"effective_support_nominal_on_open_interval_{name}"] = all(
            roots_in_open_unit_interval(p) == 0 for p in op.values()
        )
        checks[f"pure_shift_at_lambda_one_{name}"] = at_lambda(op, Q(1)) == {1: Q(1)}
    checks["weno5l_fe_affine_coefficients_nonzero_on_0_1"] = all(
        len(p) <= 2 and roots_in_open_unit_interval(p) == 0 and peval(p, Q(1)) != 0
        for p in fe.values()
    )
    interior_roots = {s: roots_in_open_unit_interval(p) for s, p in rk3.items()}
    checks["rk3_interior_coefficients_s2_to_s6_vanish_once_in_0_1"] = all(
        interior_roots[s] == 1 for s in range(2, 7)
    ) and all(peval(p, Q(1)) != 0 for p in rk3.values())
    checks["rk3_other_coefficients_nonzero_on_0_1"] = all(
        interior_roots[s] == 0 for s in rk3 if not 2 <= s <= 6
    )
    checks["fromm_equals_ub3_at_lambda_half"] = at_lambda(operators["fromm"], Q(1, 2)) == at_lambda(
        operators["ub3"], Q(1, 2)
    ) == {-1: Q(-1, 16), 0: Q(9, 16), 1: Q(9, 16), 2: Q(-1, 16)}
    s_minus_i = {1: (Q(1),), 0: (Q(-1),)}
    newton = op_mul(op_mul(op_mul(s_minus_i, s_minus_i), s_minus_i), {-1: (Q(1),)})
    newton_factor = pscale(pmul(pmul(lam_poly, psub(one, lam_poly)), psub(one, pscale(lam_poly, Q(2)))), Q(-1, 12))
    checks["fromm_minus_ub3_newton_identity"] = op_add(
        operators["fromm"], op_scale(operators["ub3"], Q(-1))
    ) == {s: pmul(newton_factor, p) for s, p in newton.items()}

    checks["bw_pure_shift_at_lambda_two_table_caption"] = at_lambda(operators["bw"], Q(2)) == {2: Q(1)}
    checks["lemma_collision_finite_coefficient_degree_at_most_w"] = all(
        max(len(p) for p in operators[name].values()) - 1 <= NOMINAL_TABLE[name][1] for name in SCHEME_ORDER
    ) and max(len(p) for p in rk3.values()) - 1 == 3

    def valuation(p: Poly) -> int:
        return next(i for i, c in enumerate(p) if c)

    checks["remark_lambda_orders_single_stage_coefficients_theta_lambda"] = all(
        valuation(p) == 1 for name in SCHEME_ORDER[:-1] for s, p in operators[name].items() if s != 0
    )
    checks["remark_lambda_orders_rk3_graded_by_stage"] = all(
        valuation(p) == (-(-s // 3) if s > 0 else -(-(-s) // 2)) for s, p in rk3.items() if s != 0
    )
    velocity = {name: {s: p[1] for s, p in operators[name].items() if len(p) > 1} for name in ("uw1", "lw", "bw")}
    checks["first_order_node_velocities_section_e2"] = (
        velocity["uw1"] == {0: Q(-1), 1: Q(1)}
        and velocity["lw"] == {-1: Q(-1, 2), 0: Q(0), 1: Q(1, 2)}
        and velocity["bw"] == {0: Q(-3, 2), 1: Q(2), 2: Q(-1, 2)}
    )

    coefficient_listing = {
        DISPLAY[name]: {str(s): poly_text(p) for s, p in sorted(operators[name].items())}
        for name in SCHEME_ORDER
    }

    # ------------------------------------------------------------------
    # thm:rank-universal and thm:invisible-general.
    rank_cases: dict[str, Any] = {}
    courants = (Q(1, 10), Q(1, 3), Q(1, 2), Q(9, 10), Q(1))
    grids = [(p, r, courants) for p in (2, 3, 4) for r in (3, 4, 6)]
    grids.append((8, 6, (Q(1, 3), Q(1, 2), Q(9, 10))))
    grids.append((8, 8, (Q(1, 2),)))
    universal_all = True
    general_all = True
    invisible_all = True
    detection_kernels = True
    collision_cases: list[str] = []
    case_count = 0
    for parents, children, lams in grids:
        horizon = children + 1
        size = parents * children
        for name in SCHEME_ORDER:
            for lam in lams:
                coeffs = at_lambda(operators[name], lam)
                counts = class_node_counts(coeffs, parents, children)
                ladder = observation_rank_ladder(coeffs, parents, children, horizon)
                general = [sum(min(L + 1, d) for d in counts) for L in range(horizon + 1)]
                universal = [parents + (parents - 1) * min(L, children - 1) for L in range(horizon + 1)]
                collision_free = all(d == children for d in counts[1:])
                key = f"{name}_P{parents}_r{children}_lambda_{lam}"
                case_count += 1
                general_ok = ladder == general
                general_all &= general_ok
                if collision_free:
                    universal_all &= ladder == universal
                else:
                    collision_cases.append(key)
                saturated = ladder[children - 1 :]
                nullity_ok = all(size - x == children - 1 for x in saturated) if collision_free else True
                bound_ok = all(x <= size - children + 1 for x in ladder)
                report = invisible_subspace_report(coeffs, parents, children, horizon)
                invisible_ok = nullity_ok and bound_ok and all(report.values())
                invisible_all &= invisible_ok
                if (parents, children) in ((8, 6), (8, 8)) or (parents == 3 and children == 4):
                    rank_cases[key] = {
                        "ladder": ladder,
                        "class_node_counts": counts,
                        "collision_free": collision_free,
                    }
                    checks[f"rank_universal_ladder_{key}"] = collision_free and ladder == universal
                    checks[f"invisible_subspace_{key}"] = invisible_ok
                if parents == 8:
                    # cor:detection-delay(i): dim ker O_{r-2} = P + r - 2 and dim ker O_{r-1} = r - 1.
                    detection_kernels &= (
                        size - ladder[children - 2] == parents + children - 2
                        and size - ladder[children - 1] == children - 1
                    )
    checks["rank_general_formula_all_cases"] = general_all
    checks["rank_universal_ladder_all_collision_free_cases"] = universal_all
    checks["invisible_subspace_all_cases"] = invisible_all
    checks["detection_delay_kernel_dimensions_working_grids"] = detection_kernels
    checks["no_spectral_collision_at_any_tested_courant_number"] = not collision_cases

    # Controls where the ladder must drop below the universal law.
    central = {-1: Q(1, 2), 1: Q(1, 2)}
    control_cases = {}
    control_ok = True
    for parents, children in ((2, 3), (4, 4), (4, 6), (8, 6)):
        counts = class_node_counts(central, parents, children)
        ladder = observation_rank_ladder(central, parents, children, children + 1)
        general = [sum(min(L + 1, d) for d in counts) for L in range(children + 2)]
        universal = [parents + (parents - 1) * min(L, children - 1) for L in range(children + 2)]
        expected_counts = [1] + [
            (children + 1) // 2 if 2 * m == parents else children for m in range(1, parents)
        ]
        ok = ladder == general and counts == expected_counts and ladder[-1] < universal[-1]
        control_ok &= ok
        control_cases[f"P{parents}_r{children}"] = {"ladder": ladder, "class_node_counts": counts}
    checks["collision_control_central_average_general_formula"] = control_ok
    zero_ok = True
    for name in SCHEME_ORDER:
        coeffs = at_lambda(operators[name], Q(0))
        ladder = observation_rank_ladder(coeffs, 3, 4, 5)
        zero_ok &= ladder == [3] * 6 and class_node_counts(coeffs, 3, 4) == [1, 1, 1]
    checks["lambda_zero_control_rank_collapses_to_P"] = zero_ok

    # ------------------------------------------------------------------
    # prop:local-rank, lem:local-upper, conj:local-rank, against local_observability.csv.
    local: dict[str, Any] = {}
    csv_match = True
    closed_early = True
    closed_saturation = True
    wrap_attained = True
    collars_rule = True
    for children in (6, 8):
        for name in SCHEME_ORDER:
            for lam in (Q(1, 10), Q(1, 2), Q(9, 10), Q(1)):
                coeffs = at_lambda(operators[name], lam)
                m_left, m_right = reaches(coeffs)
                width = m_left + m_right
                sigma = 1 + (1 if m_right >= 1 else 0)
                horizon = children + 3
                ladder = local_rank_ladder(coeffs, 8, children, horizon)
                for L, rho in enumerate(ladder):
                    if L * width <= children - 1:
                        closed_early &= rho == sigma * L
                    if L >= children - 1:
                        closed_saturation &= rho == children - 1
                    wrap_attained &= rho == wraparound_bound(m_left, m_right, children, L)
                key = f"{name}_r{children}_lambda_{lam}"
                local[key] = {"m_L": m_left, "m_R": m_right, "rho": ladder}
                lam_text = {Q(1, 2): "0.5", Q(9, 10): "0.9"}.get(lam)
                if lam_text is not None:
                    ref = AUTHORS_LOCAL_RANKS[(name, children, lam_text)]
                    csv_match &= (m_left, m_right) == ref[:2] and tuple(ladder) == ref[2]
                if lam == Q(1) and name in INTERPOLATION_MEMBERS:
                    collars_rule &= (m_left, m_right) == (1, 0) and ladder[: children] == list(range(children))
    checks["local_rank_early_horizon_sigma_L"] = closed_early
    checks["local_rank_saturation_r_minus_1"] = closed_saturation
    checks["local_rank_wraparound_bound_attained_all_horizons"] = wrap_attained
    checks["local_rank_matches_authors_local_observability_csv"] = csv_match
    checks["local_rank_lambda_one_pure_shift_single_collar"] = collars_rule
    rk3_r6 = local["weno5l_rk3_r6_lambda_1/2"]["rho"]
    rk3_r8 = local["weno5l_rk3_r8_lambda_1/2"]["rho"]
    checks["rk3_local_ladder_0_3_5_and_0_3_7"] = rk3_r6[:3] == [0, 3, 5] and rk3_r8[:3] == [0, 3, 7]
    bw6 = local["bw_r6_lambda_1/2"]["rho"]
    checks["bw_local_ladder_jumps_by_two_at_L4"] = bw6[:5] == [0, 1, 2, 3, 5]
    checks["width_law_refuted_collars_law_holds"] = (
        local["ub5_r8_lambda_1/2"]["rho"][1] == 2 != NOMINAL_TABLE["ub5"][1]
    )
    t_loc_ok = True
    for name, (shield, t_loc) in AUTHORS_HALF_LIFE_LOCAL_COLUMNS.items():
        ladder = local[f"{name}_r6_lambda_1/2"]["rho"]
        width = NOMINAL_TABLE[name][1]
        t_loc_ok &= min(L for L, rho in enumerate(ladder) if rho == 5) == t_loc
        t_loc_ok &= (5 // width) == shield
    checks["half_life_table_T_loc_and_shielding_columns"] = t_loc_ok

    # ------------------------------------------------------------------
    # prop:collision-certificate: whole-interval emptiness at the working grids.
    collision: dict[str, Any] = {}
    second_order_pairs: dict[str, list[tuple[int, int, int, int, Q, Q]]] = {}
    total_pairs = 0
    global_worst = None
    for children in (6, 8):
        for name in SCHEME_ORDER:
            report = collision_certificate(operators[name], 8, children)
            total_pairs += report["pairs"]
            ref_pairs, ref_cert, ref_min, ref_cells = AUTHORS_COLLISION_CERTIFICATE[(name, children)]
            ref_value = Q(ref_min)
            lo, hi = report["min_bernstein_lower"], report["min_bernstein_upper"]
            tolerance = Q(1, 10**12) * ref_value
            agrees = lo - tolerance <= ref_value <= hi + tolerance
            key = f"{name}_P8_r{children}"
            checks[f"collision_set_empty_on_0_1_{key}"] = (
                report["certified"] == report["pairs"] and report["identically_zero_pairs"] == 0
                and report["min_valuation"] >= 1 and report["max_q_degree"] <= 8
            )
            checks[f"collision_certificate_matches_authors_csv_{key}"] = (
                report["pairs"] == ref_pairs and ref_cert and ref_cells == 1 and agrees
            )
            collision[key] = {
                "pairs": report["pairs"],
                "certified_pairs": report["certified"],
                "min_valuation": report["min_valuation"],
                "max_q_degree": report["max_q_degree"],
                "min_bernstein_coefficient": decimal_text(lo, 15),
                "authors_certified_min_q": ref_min,
                "worst_pair_k_kprime": list(report["worst_pair"]),
            }
            if global_worst is None or lo < global_worst[0]:
                global_worst = (lo, key)
            if children == 6:
                second_order_pairs[name] = report["second_order_pairs"]
    assert global_worst is not None
    # E2 (sec:exp-spectra): at (8,6) only Lax-Wendroff has second-order class-node gaps,
    # exactly the reflected pairs (4,20) and (28,44), with |g_4 - g_20| = sqrt(3) lam^2.
    lw_pairs = second_order_pairs["lw"]
    checks["lax_wendroff_second_order_node_gaps_sqrt3_lambda_squared"] = (
        [(k, kk, a, d) for k, kk, a, d, _, _ in lw_pairs] == [(4, 20, 2, 0), (28, 44, 2, 0)]
        and all(q_lo <= 3 <= q_hi for *_, q_lo, q_hi in lw_pairs)
        and all(not second_order_pairs[n] for n in SCHEME_ORDER if n != "lw")
    )
    checks["collision_certificate_2408_pairs"] = total_pairs == 2408
    checks["collision_worst_pair_lax_wendroff_r8_bound_5_62e-3"] = (
        global_worst[1] == "lw_P8_r8" and global_worst[0] >= Q(562, 10**5)
    )

    # ------------------------------------------------------------------
    # cor:detection-delay(ii) on the E5 compensated pairs (P = 8, r = 6, lambda = 1/2).
    delayed: dict[str, Any] = {}
    delayed_ok = True
    for name, t0, j1, j2, t_pred, t_meas in AUTHORS_DELAYED_PAIRS:
        coeffs = at_lambda(operators[name], Q(1, 2))
        m_left, m_right = reaches(coeffs)
        expected_j = (0, 5) if t0 == 0 else (t0 * m_right, 6 - t0 * m_left)
        detected, history = first_detection(coeffs, 8, 6, j1, j2, max(t0, 1) + 1)
        exact_value = True
        if t0 >= 1:
            exact_value = history[t0][1] == coeffs[m_left] ** t0 / 6
        ok = (j1, j2) == expected_j and detected == t_pred == t_meas == max(t0, 1) and exact_value
        delayed_ok &= ok
        delayed[f"{name}_t0_{t0}"] = {"j1": j1, "j2": j2, "first_detection": detected}
    checks["detection_delay_compensated_pairs_exact"] = delayed_ok
    discrepancies.append(
        {
            "location": "article.tex line 1041 (E5)",
            "claim": "across all 13 (scheme, depth) pairs the first separation occurs exactly at t*=t0",
            "finding": (
                "for the WENO5-LIN+SSP-RK3 pair (t0=0, j1=0, j2=5) the first separation is at "
                "step 1, not t0=0 (zero-mean data are never visible at step 0); the authors' "
                "delayed_collisions.csv records t_detect_predicted=1 for that row"
            ),
        }
    )

    check_count = len(checks)
    return {
        "schema": "certified-simulation/coarse-observability-high-order/independent-rank-structure/v1",
        "arithmetic": (
            "fractions.Fraction and Python integers; exact ranks over Q; exact zero tests in "
            "Q[x]/Phi_N(x); rigorous outward-rounded fixed-point intervals (2^-320 grid) for "
            "cos/sin(2 pi v/N); tolerance zero for every exact statement"
        ),
        "results": (
            "Scheme coefficients from the manuscript text; thm:rank-universal (general and "
            "universal formulas); thm:invisible-general; prop:local-rank, lem:local-upper and "
            "conj:local-rank at r=6,8; prop:collision-certificate at (8,6),(8,8); "
            "cor:detection-delay(ii) on the E5 pairs"
        ),
        "scope": {
            "executed": (
                f"{case_count} exact rank ladders (P in 2,3,4 with r in 3,4,6 at lambda in "
                "1/10,1/3,1/2,9/10,1; (8,6) at 1/3,1/2,9/10; (8,8) at 1/2; L=0..r+1); local "
                "ranks at P=8, r=6,8, lambda in 1/10,1/2,9/10,1, L=0..r+3; 2408 collision pairs"
            ),
            "exclusion": (
                "Finite sweeps do not prove the quantified statements; the collision certificate "
                "is specific to the two grid pairs, as in the manuscript."
            ),
            "independence": (
                "No file under the authors' src/ was opened; reference values are transcribed "
                "from results/*.csv and results/tables/*.tex only for comparison."
            ),
        },
        "scheme_coefficients": coefficient_listing,
        "rank_cases": rank_cases,
        "collision_controls": control_cases,
        "collision_cases_found": collision_cases,
        "local_ranks": local,
        "collision_certificate": collision,
        "detection_delays": delayed,
        "discrepancies": discrepancies,
        "checks": checks,
        "check_count": check_count,
        "passed": sum(checks.values()),
        "verdict": "pass" if all(checks.values()) else "fail",
    }


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
        / "observation-rank-structure-certificate.json",
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

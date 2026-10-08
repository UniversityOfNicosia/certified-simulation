#!/usr/bin/env python3
"""Independent exact certificate: long-horizon image and fibers, the lag-r
recurrence, and the sharp minimal child-count law, for the note "Exact
Lifting of Coarse Histories in Periodic Upwind Finite Volumes" (Polemitis,
Christakis, Drikakis).

Every object is built here from the manuscript's definitions alone
(sec:hierarchy and sec:long-horizon): the fine-grid shift S, the upwind
operator A = (1-lambda) I + lambda S, the parent average R, the history map
O_{L,r}, the parent shift Pi_P, the all-horizon invisible subspace N_r, and
the recurrence eq:history-recurrence with its displayed coefficients.

The script replays, in exact rational arithmetic:

* the identity sum_j C(r,j) [-(1-lambda)]^{r-j} A^j = lambda^r S^r as a
  matrix identity, R S^r = Pi_P R, the triangular change of basis from
  (R, RS, ..., RS^{r-1}) to (R, RA, ..., RA^{r-1}), and the layer-difference
  identity r (RS^{t+1} d - RS^t d)_K = d_{K-1,r-1-t} - d_{K,r-1-t};
* thm:long-realization for P = 2..4, r = 1..5, L in {r-1, r, r+1, 2r} and
  lambda in {1/10, 1/3, 1/2, 2/3, 9/10, 1}.  The image equals the set cut
  out by the equal-sum condition and the recurrence: every column of O_{L,r}
  satisfies both, and the exact rank equals the dimension of that set.
  Admissible histories are lifted constructively through
  thm:short-realization at horizon r-1.  ker O_{L,r} = N_r and
  dim N_r = r-1, and fibers are x* + N_r;
* the kernel ladder dim ker O_{L,r} = P(r-L-1)+L for L <= r-1 and r-1 for
  L >= r-1, for L = 0..2r+1;
* cor:minimal-r: r_min = L+1, found by exhaustive search over r for
  P = 2..4, L = 0..5 and all six lambdas.  For every r <= L the proof's
  history (0, ..., 0, v, 0, ...) is mass-conserving, and exact rank shows it
  is not realizable.  The codimension of the image in M_{P,L} is
  (P-1)(L+1-r).  The hypotheses P >= 2 and lambda > 0 are both needed;
* ex:nonrealizable value by value, rows 2 and 3 of tab:certificates, and
  the authors' generated CSV and table fragment;
* the bibliography: every cited key has an entry, and the companion-paper
  entries are recorded for the staging review.

The script is self-contained, deterministic, and uses only the Python
standard library.  It never reads the authors' code.  Values quoted from the
manuscript, the bibliography and the authors' generated outputs are embedded
below as transcriptions, and the companion test re-reads those files to
confirm each one.  The finite sweeps detect errors; they do not replace the
note's proofs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections.abc import Sequence
from fractions import Fraction as Q
from math import comb, gcd, lcm
from pathlib import Path
from typing import Any

type Vector = tuple[Q, ...]
type History = tuple[Vector, ...]

LAMBDAS: tuple[Q, ...] = (Q(1, 10), Q(1, 3), Q(1, 2), Q(2, 3), Q(9, 10), Q(1))
SWEEP_PARENTS = (2, 3, 4)
SWEEP_CHILDREN = (1, 2, 3, 4, 5)

# ---------------------------------------------------------------------------
# Transcriptions from manuscript/main.tex.  Each entry is (label, template,
# values): the whitespace-free LaTeX displayed after that label is the template
# with each "#" replaced, in order, by the values.

MANUSCRIPT_FORMULAS: dict[str, tuple[str, str]] = {
    "shift_identity": ("sec:long-horizon", r"\bigl(A_\lambda-(1-\lambda)I\bigr)^r=\lambda^rS^r."),
    "parent_closure": ("sec:long-horizon", r"RS^r=\Pi_PR,"),
    "invisible_subspace": (
        "sec:long-horizon",
        r"x_{K,j}=c_j\\text{forall}K,\quad\sum_{j=0}^{r-1}c_j=0",
    ),
    "invisible_dimension": ("sec:long-horizon", r"Ithasdimension\(r-1\)."),
    "long_hypotheses": ("thm:long-realization", r"\(L\ger-1\)"),
    "equal_sums": (
        "thm:long-realization",
        r"\mathbf1_P^\topy^0=\cdots=\mathbf1_P^\topy^{r-1};",
    ),
    "recurrence": (
        "thm:long-realization",
        r"y^{n+r}=\lambda^r\Pi_Py^n-\sum_{j=0}^{r-1}\binom{r}{j}\bigl[-(1-\lambda)\bigr]^{r-j}"
        r"y^{n+j}.\label{eq:history-recurrence}",
    ),
    "recurrence_range": ("thm:long-realization", r"\(n=0,\ldots,L-r\),"),
    "long_fiber": ("thm:long-realization", r"{\calO}_{L,r}^{-1}(Y)=x_\star+{\calN}_r"),
    "long_fiber_dimension": ("thm:long-realization", r"\dim{\calO}_{L,r}^{-1}(Y)=r-1."),
    "expanded_identity": (
        "thm:long-realization",
        r"\sum_{j=0}^{r}\binom{r}{j}\bigl[-(1-\lambda)\bigr]^{r-j}A_\lambda^j=\lambda^rS^r.",
    ),
    "layer_difference": (
        "thm:long-realization",
        r"r(RS^{t+1}d-RS^td)_K=d_{K-1,r-1-t}-d_{K,r-1-t}.",
    ),
    "kernel_saturated": ("thm:long-realization", r"\ker{\calO}_{L,r}={\calN}_r\qquad(L\ger-1)."),
    "minimal_count": ("cor:minimal-r", r"r_{\min}=L+1."),
    "counterexample_history": ("cor:minimal-r", r"y^0=\cdots=y^{r-1}=0,\qquady^r=v,"),
    "counterexample_tail": ("cor:minimal-r", r"y^{r+1}=\cdots=y^L=0."),
}

MANUSCRIPT_EXAMPLE_NONREALIZABLE: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "parameters": (
        "ex:nonrealizable",
        r"P=#,\qquadr=#,\qquadL=#,\qquad\lambda=\frac##.",
        ("2", "2", "2", "1", "2"),
    ),
    "history": (
        "ex:nonrealizable",
        r"y^0=(#,#),\qquady^1=(#,#),\qquady^2=(#,#).",
        ("1", "1", "1", "1", "2", "0"),
    ),
    "prediction": ("ex:nonrealizable", r"predicts\[y^2=(#,#).\]", ("1", "1")),
    "residual": ("ex:nonrealizable", r"Therecurrenceresidualistherefore\[(#,#),\]", ("1", "-1")),
}

# tab:certificates as typed in main.tex: header and rows, cells stripped.
MANUSCRIPT_TABLE: tuple[tuple[str, str, str], ...] = (
    ("Certificate", "Parameters", "Result"),
    ("Normalized short-horizon lift", r"\(P=2,\ L=1,\ r=2,\ \lambda=1/2\)", "exact lift"),
    ("Long-horizon recurrence", r"\(P=3,\ r=3,\ L=6,\ \lambda=1/2\)", "residual zero"),
    (
        "Nonrealizable nonnegative history",
        r"\(P=2,\ r=2,\ L=2,\ \lambda=1/2\)",
        r"residual \((1,-1)\)",
    ),
)

# The authors' generated outputs, verbatim (CSV lines without terminators;
# the fragment is the whole file).
AUTHORS_CSV_HEADER = "certificate,parameters,result,details"
AUTHORS_CSV_LINE_2 = (
    'long_horizon_recurrence,"P=3, r=3, L=6, lambda=1/2",residual zero,"residuals='
    "[[Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)], "
    "[Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)], "
    "[Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)], "
    '[Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)]]"'
)
AUTHORS_CSV_LINE_3 = (
    'nonrealizable_mass_conserving_history,"P=2, r=2, L=2, lambda=1/2","residual (1,-1)",'
    '"y0=(1,1), y1=(1,1), y2=(2,0)"'
)
AUTHORS_TABLE_FRAGMENT = (
    r"\begin{tabular}{@{}lll@{}}\toprule"
    r"Certificate & Parameters & Result \\\midrule"
    r"Normalized short-horizon lift & P=2, L=1, r=2, lambda=1/2 & exact lift \\"
    r"Long-horizon recurrence & P=3, r=3, L=6, lambda=1/2 & residual zero \\"
    r"Nonrealizable nonnegative history & P=2, r=2, L=2, lambda=1/2 & residual (1,-1) \\"
    r"\bottomrule\end{tabular}"
)

# Bibliography transcriptions: \cite keys of main.tex in order of first
# citation, and the entry keys of references.bib in file order.
CITED_KEYS: tuple[str, ...] = (
    "LeVeque2002",
    "EymardGallouetHerbin2000",
    "Berger1987",
    "BergerColella1989",
    "McCorquodaleColella2011",
    "OsherSanders1983",
    "ConstantinescuSandu2007",
    "BarSinai2019",
    "KochkovEtAl2021",
    "polemitis2026exactfinitehorizonmemoryconditioning",
    "PolemitisChristakisDrikakis2026OneStep",
    "Willems1986ExactModelling",
    "PoldermanWillems1998",
    "MarkovskyDorfler2021",
    "CohnDee1988",
    "HoKalman1966",
    "Moore1981",
    "Forney2011",
    "Antoulas2005",
    "BennerGugercinWillcox2015",
    "PeherstorferWillcox2016",
    "Kevrekidis2003",
)
BIB_ENTRY_KEYS: tuple[str, ...] = (
    "LeVeque2002",
    "EymardGallouetHerbin2000",
    "Berger1987",
    "BergerColella1989",
    "McCorquodaleColella2011",
    "OsherSanders1983",
    "ConstantinescuSandu2007",
    "CohnDee1988",
    "HoKalman1966",
    "Moore1981",
    "Forney2011",
    "Antoulas2005",
    "PoldermanWillems1998",
    "Willems1986ExactModelling",
    "MarkovskyDorfler2021",
    "BennerGugercinWillcox2015",
    "PeherstorferWillcox2016",
    "ChorinHaldKupferman2002",
    "ParishDuraisamy2017",
    "Stinis2015",
    "BarSinai2019",
    "KochkovEtAl2021",
    "AgdesteinSanderse2025",
    "AgdesteinVerstappenSanderse2026",
    "SanderseEtAl2025",
    "polemitis2026exactfinitehorizonmemoryconditioning",
    "PolemitisChristakisDrikakis2026OneStep",
    "Kevrekidis2003",
)
# Whitespace-free field text of the entries that the findings concern.
BIB_FIELDS: dict[str, tuple[str, ...]] = {
    "polemitis2026exactfinitehorizonmemoryconditioning": (
        r"@misc{polemitis2026exactfinitehorizonmemoryconditioning,",
        r"eprint={2608.08633},",
        r"archivePrefix={arXiv},",
        r"url={https://arxiv.org/abs/2608.08633}",
    ),
    "PolemitisChristakisDrikakis2026OneStep": (
        r"@misc{PolemitisChristakisDrikakis2026OneStep,",
        r"note={ManuscriptsubmittedtoAppliedNumericalMathematics},",
    ),
    "EymardGallouetHerbin2000": (r'Gallou{"e}t',),
    "MarkovskyDorfler2021": (r'D{"o}rfler',),
}


def fill(template: str, values: Sequence[str]) -> str:
    for value in values:
        template = template.replace("#", value, 1)
    return template


# ---------------------------------------------------------------------------
# Encoding, pseudo-random rationals, exact linear algebra.


def encode(value: Q) -> str:
    return f"{value.numerator}/{value.denominator}"


def encode_vector(values: Sequence[Q]) -> list[str]:
    return [encode(Q(value)) for value in values]


class Stream:
    """Deterministic small rationals from a 64-bit LCG (no `random` module)."""

    def __init__(self, seed: int) -> None:
        self.state = (seed * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019) % 2**64

    def next_int(self, bound: int) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.state >> 33) % bound

    def rational(self) -> Q:
        return Q(self.next_int(41) - 20, 1 + self.next_int(6))


def rank(rows: Sequence[Sequence[Q]]) -> int:
    """Exact rank over Q: rows scaled to integers, fraction-free elimination."""
    matrix: list[list[int]] = []
    for row in rows:
        values = [Q(v) for v in row]
        scale = lcm(*(v.denominator for v in values)) if values else 1
        ints = [v.numerator * (scale // v.denominator) for v in values]
        if any(ints):
            matrix.append(ints)
    if not matrix:
        return 0
    width = len(matrix[0])
    found = 0
    for col in range(width):
        pivot = next((i for i in range(found, len(matrix)) if matrix[i][col]), None)
        if pivot is None:
            continue
        matrix[found], matrix[pivot] = matrix[pivot], matrix[found]
        top = matrix[found]
        head = top[col]
        for i in range(found + 1, len(matrix)):
            lead = matrix[i][col]
            if lead:
                row = [head * a - lead * b for a, b in zip(matrix[i], top, strict=True)]
                common = gcd(*row)
                matrix[i] = [a // common for a in row] if common > 1 else row
        found += 1
        if found == len(matrix):
            break
    return found


def dot(a: Sequence[Q], b: Sequence[Q]) -> Q:
    return sum((x * y for x, y in zip(a, b, strict=True)), Q(0))


def unit(size: int, index: int) -> Vector:
    return tuple(Q(int(i == index)) for i in range(size))


def flatten(history_: History) -> Vector:
    return tuple(v for state in history_ for v in state)


# ---------------------------------------------------------------------------
# The hierarchy (sec:hierarchy), re-implemented here so the script stands alone.


def shift_s(x: Vector, parents: int, children: int) -> Vector:
    """(Sx)_{K,j} = x_{K,j-1} for j >= 1 and x_{K-1,r-1} for j = 0."""
    out: list[Q] = []
    for k in range(parents):
        for j in range(children):
            if j >= 1:
                out.append(x[k * children + j - 1])
            else:
                out.append(x[((k - 1) % parents) * children + children - 1])
    return tuple(out)


def upwind(x: Vector, lam: Q, parents: int, children: int) -> Vector:
    shifted = shift_s(x, parents, children)
    return tuple((1 - lam) * a + lam * b for a, b in zip(x, shifted, strict=True))


def restrict(x: Vector, parents: int, children: int) -> Vector:
    return tuple(
        sum(x[k * children : (k + 1) * children], Q(0)) / children for k in range(parents)
    )


def history(x: Vector, lam: Q, parents: int, children: int, horizon: int) -> History:
    states = []
    state = tuple(x)
    for n in range(horizon + 1):
        states.append(restrict(state, parents, children))
        if n < horizon:
            state = upwind(state, lam, parents, children)
    return tuple(states)


def parent_shift(y: Vector) -> Vector:
    return tuple(y[k - 1] for k in range(len(y)))


def register_inverse(g: Vector) -> Vector:
    partial = [Q(0)]
    for k in range(1, len(g)):
        partial.append(partial[-1] + g[k])
    mean = sum(partial, Q(0)) / len(g)
    return tuple(value - mean for value in partial)


def flux_to_collar(horizon: int, lam: Q) -> tuple[Vector, ...]:
    return tuple(
        tuple(
            lam ** (-(t + 1)) * comb(t, j) * (-(1 - lam)) ** (t - j) if j <= t else Q(0)
            for j in range(horizon)
        )
        for t in range(horizon)
    )


def normalized_lift(y: History, lam: Q, parents: int, children: int) -> Vector:
    """thm:short-realization's lift (used at horizon r-1, as in the proof)."""
    horizon = len(y) - 1
    if not 0 <= horizon <= children - 1:
        raise ValueError("the short-horizon lift needs 0 <= L <= r-1")
    x = [[Q(0)] * children for _ in range(parents)]
    if horizon >= 1:
        registers = [
            register_inverse(tuple(children * (a - b) for a, b in zip(y[t], y[t + 1], strict=True)))
            for t in range(horizon)
        ]
        inverse = flux_to_collar(horizon, lam)
        for k in range(parents):
            phi = [registers[t][k] for t in range(horizon)]
            for j in range(horizon):
                x[k][children - 1 - j] = dot(inverse[j], phi)
    for k in range(parents):
        x[k][0] = children * y[0][k] - sum(x[k][1:], Q(0))
    return tuple(v for row in x for v in row)


def random_mass_conserving(stream: Stream, parents: int, horizon: int) -> History:
    first = tuple(stream.rational() for _ in range(parents))
    total = sum(first, Q(0))
    states = [first]
    for _ in range(horizon):
        state = [stream.rational() for _ in range(parents)]
        state[0] += total - sum(state, Q(0))
        states.append(tuple(state))
    return tuple(states)


def mass_constraint_rows(parents: int, horizon: int, last: int | None = None) -> list[Vector]:
    """1^T y^n - 1^T y^0 = 0 for n = 1..last (default L)."""
    width = parents * (horizon + 1)
    rows = []
    for n in range(1, (horizon if last is None else min(last, horizon)) + 1):
        row = [Q(0)] * width
        for k in range(parents):
            row[n * parents + k] += 1
            row[k] -= 1
        rows.append(tuple(row))
    return rows


# ---------------------------------------------------------------------------
# sec:long-horizon.


def recurrence_coefficients(children: int, lam: Q) -> tuple[Q, tuple[Q, ...]]:
    """(lambda^r, (C(r,j) [-(1-lambda)]^{r-j} for j = 0..r-1)) of eq:history-recurrence."""
    return lam**children, tuple(
        comb(children, j) * (-(1 - lam)) ** (children - j) for j in range(children)
    )


def recurrence_prediction(y: History, lam: Q, children: int, n: int) -> Vector:
    lead, coefficients = recurrence_coefficients(children, lam)
    predicted = [lead * v for v in parent_shift(y[n])]
    for j, coefficient in enumerate(coefficients):
        predicted = [p - coefficient * v for p, v in zip(predicted, y[n + j], strict=True)]
    return tuple(predicted)


def recurrence_residual(y: History, lam: Q, children: int, n: int) -> Vector:
    """y^{n+r} minus the recurrence's prediction."""
    return tuple(
        a - p for a, p in zip(y[n + children], recurrence_prediction(y, lam, children, n), strict=True)
    )


def extend_by_recurrence(head: History, lam: Q, children: int, horizon: int) -> History:
    states = list(head)
    while len(states) < horizon + 1:
        states.append(recurrence_prediction(tuple(states), lam, children, len(states) - children))
    return tuple(states)


def condition_rows(parents: int, children: int, horizon: int, lam: Q) -> list[Vector]:
    """Equal sums of y^0..y^{r-1}, then the recurrence for n = 0..L-r."""
    rows = mass_constraint_rows(parents, horizon, last=children - 1)
    lead, coefficients = recurrence_coefficients(children, lam)
    width = parents * (horizon + 1)
    for n in range(horizon - children + 1):
        for k in range(parents):
            row = [Q(0)] * width
            row[(n + children) * parents + k] += 1
            row[n * parents + (k - 1) % parents] -= lead
            for j, coefficient in enumerate(coefficients):
                row[(n + j) * parents + k] += coefficient
            rows.append(tuple(row))
    return rows


def invisible_basis(parents: int, children: int) -> list[Vector]:
    """N_r: the zero-sum child profile e_j - e_0 repeated in every parent."""
    basis = []
    for j in range(1, children):
        profile = [Q(0)] * children
        profile[j], profile[0] = Q(1), Q(-1)
        basis.append(tuple(profile) * parents)
    return basis


def unit_columns(parents: int, children: int, lam: Q, horizon: int) -> list[Vector]:
    """Columns of O_{L,r}: flattened histories of the fine unit vectors.

    With lambda = p/q, q^n A^n = ((q-p) I + p S)^n has integer entries, so each
    unit vector is evolved in exact integers and divided by r q^n only when
    averaged.  A check below compares this with the direct Fraction evolution.
    """
    p, q = lam.numerator, lam.denominator
    size = parents * children
    columns = []
    for c in range(size):
        z = [0] * size
        z[c] = 1
        flat: list[Q] = []
        for n in range(horizon + 1):
            scale = children * q**n
            flat.extend(
                Q(sum(z[k * children : (k + 1) * children]), scale) for k in range(parents)
            )
            if n < horizon:
                shifted = shift_s(tuple(z), parents, children)
                z = [(q - p) * a + p * b for a, b in zip(z, shifted, strict=True)]
        columns.append(tuple(flat))
    return columns


class UnitHistories:
    """Cache of unit_columns per (P, r, lambda), truncated to the requested L."""

    def __init__(self) -> None:
        self.cache: dict[tuple[int, int, Q], tuple[int, list[Vector]]] = {}

    def columns(self, parents: int, children: int, lam: Q, horizon: int) -> list[Vector]:
        key = (parents, children, lam)
        stored = self.cache.get(key)
        if stored is None or stored[0] < horizon:
            depth = max(horizon, 2 * children + 1)
            stored = (depth, unit_columns(parents, children, lam, depth))
            self.cache[key] = stored
        width = parents * (horizon + 1)
        return [column[:width] for column in stored[1]]


def combine(x: Sequence[Q], columns: Sequence[Vector]) -> Vector:
    """O_{L,r} x as the combination sum_c x_c O e_c (linearity of the history map)."""
    total = [Q(0)] * len(columns[0])
    for coefficient, column in zip(x, columns, strict=True):
        if coefficient:
            total = [u + coefficient * v for u, v in zip(total, column, strict=True)]
    return tuple(total)


def split_states(flat: Vector, parents: int) -> History:
    return tuple(tuple(flat[i : i + parents]) for i in range(0, len(flat), parents))


def admissible(y: History, lam: Q, children: int) -> bool:
    """Condition 1 (equal sums of y^0..y^{r-1}) and condition 2 (the recurrence)."""
    ok = len({sum(state, Q(0)) for state in y[:children]}) == 1
    for n in range(len(y) - children):
        ok &= not any(recurrence_residual(y, lam, children, n))
    return ok


def long_case(
    parents: int, children: int, horizon: int, lam: Q, seed: int, units: UnitHistories
) -> dict[str, Any]:
    """Every claim of thm:long-realization at one (P, r, L, lambda)."""
    size = parents * children
    stream = Stream(seed)
    columns = units.columns(parents, children, lam, horizon)
    satisfied = all(admissible(split_states(column, parents), lam, children) for column in columns)
    rows = condition_rows(parents, children, horizon, lam)
    probe = tuple(tuple(stream.rational() for _ in range(parents)) for _ in range(horizon + 1))
    expected = [
        sum(probe[n], Q(0)) - sum(probe[0], Q(0)) for n in range(1, min(children - 1, horizon) + 1)
    ] + [
        value
        for n in range(horizon - children + 1)
        for value in recurrence_residual(probe, lam, children, n)
    ]
    rows_encode_conditions = [dot(row, flatten(probe)) for row in rows] == expected
    rank_o = rank(columns)
    dim_c = parents * (horizon + 1) - rank(rows)
    image = satisfied and rows_encode_conditions and rank_o == dim_c == size - children + 1

    zero = (Q(0),) * (parents * (horizon + 1))
    basis = invisible_basis(parents, children)
    kernel = (
        size - rank_o == children - 1
        and len(basis) == children - 1
        and rank(basis) == children - 1
        and all(combine(d, columns) == zero for d in basis)
    )

    # Direct evolution of a pseudo-random state agrees with the columns and is admissible.
    x = tuple(stream.rational() for _ in range(size))
    y = history(x, lam, parents, children, horizon)
    necessity = flatten(y) == combine(x, columns) and admissible(y, lam, children)

    sufficiency = fiber = True
    for _ in range(2):
        head = random_mass_conserving(stream, parents, children - 1)
        y = extend_by_recurrence(head, lam, children, horizon)
        x = normalized_lift(head, lam, parents, children)
        sufficiency &= admissible(y, lam, children) and combine(x, columns) == flatten(y)
        shifted = list(x)
        for d in basis:
            c = stream.rational()
            shifted = [u + c * v for u, v in zip(shifted, d, strict=True)]
        fiber &= combine(shifted, columns) == flatten(y)
    sufficiency &= history(x, lam, parents, children, horizon) == y

    return {
        "rank": rank_o,
        "dim_conditions": dim_c,
        "nullity": size - rank_o,
        "image": image,
        "kernel": kernel,
        "necessity": necessity,
        "sufficiency": sufficiency,
        "fiber": fiber,
    }


def parse_parameters(text: str) -> dict[str, Q]:
    found = re.findall(r"(P|L|r|\\lambda|lambda)=([0-9/]+)", text)
    return {key.removeprefix("\\"): Q(value) for key, value in found}


def parse_vectors(text: str) -> dict[str, Vector]:
    return {
        key: tuple(Q(v) for v in body.split(","))
        for key, body in re.findall(r"(y\d)=\(([^)]*)\)", text)
    }


# ---------------------------------------------------------------------------


def build_certificate() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    discrepancies: list[dict[str, str]] = []
    units = UnitHistories()

    # ------------------------------------------------------------------
    # The identities behind the recurrence.
    for parents in (2, 3):
        identity_ok = closure_ok = triangular_ok = layer_ok = True
        stream = Stream(10 + parents)
        for children in SWEEP_CHILDREN:
            size = parents * children
            for lam in LAMBDAS:
                for c in range(size):
                    e = unit(size, c)
                    powers = [e]
                    for _ in range(children):
                        powers.append(upwind(powers[-1], lam, parents, children))
                    left = [Q(0)] * size
                    for j in range(children + 1):
                        coefficient = comb(children, j) * (-(1 - lam)) ** (children - j)
                        left = [u + coefficient * v for u, v in zip(left, powers[j], strict=True)]
                    shifted = e
                    for _ in range(children):
                        shifted = shift_s(shifted, parents, children)
                    identity_ok &= tuple(left) == tuple(lam**children * v for v in shifted)
                    closure_ok &= restrict(shifted, parents, children) == parent_shift(
                        restrict(e, parents, children)
                    )
                d = tuple(stream.rational() for _ in range(size))
                s_powers = [d]
                a_powers = [d]
                for _ in range(children):
                    s_powers.append(shift_s(s_powers[-1], parents, children))
                    a_powers.append(upwind(a_powers[-1], lam, parents, children))
                for t in range(children):
                    combination = [Q(0)] * parents
                    for j in range(t + 1):
                        weight = comb(t, j) * (1 - lam) ** (t - j) * lam**j
                        combination = [
                            u + weight * v
                            for u, v in zip(
                                combination, restrict(s_powers[j], parents, children), strict=True
                            )
                        ]
                    triangular_ok &= restrict(a_powers[t], parents, children) == tuple(combination)
                    triangular_ok &= lam**t != 0
                for t in range(children - 1):
                    lhs = tuple(
                        children * (u - v)
                        for u, v in zip(
                            restrict(s_powers[t + 1], parents, children),
                            restrict(s_powers[t], parents, children),
                            strict=True,
                        )
                    )
                    rhs = tuple(
                        d[((k - 1) % parents) * children + children - 1 - t]
                        - d[k * children + children - 1 - t]
                        for k in range(parents)
                    )
                    layer_ok &= lhs == rhs
        checks[f"identity_binomial_expansion_equals_lambda^r_S^r_P{parents}"] = identity_ok
        checks[f"identity_RS^r_equals_Pi_P_R_P{parents}"] = closure_ok
        checks[f"identity_RA^t_triangular_in_RS^j_diagonal_lambda^t_P{parents}"] = triangular_ok
        checks[f"identity_layer_difference_P{parents}"] = layer_ok

    evolution_ok = True
    for parents, children in ((2, 2), (2, 3), (3, 2), (3, 3), (4, 3)):
        size = parents * children
        for lam in LAMBDAS + (Q(3, 2), Q(-1, 2), Q(0)):
            direct = [
                flatten(history(unit(size, c), lam, parents, children, 2 * children + 1))
                for c in range(size)
            ]
            evolution_ok &= unit_columns(parents, children, lam, 2 * children + 1) == direct
    checks["columns_by_integer_evolution_match_fraction_evolution"] = evolution_ok

    # ------------------------------------------------------------------
    # thm:long-realization on the sweep, and the kernel ladder.
    cases: dict[str, Any] = {}
    ladders: dict[str, list[int]] = {}
    seed = 2000
    for parents in SWEEP_PARENTS:
        for children in SWEEP_CHILDREN:
            aggregate = dict.fromkeys(
                ("image", "kernel", "necessity", "sufficiency", "fiber", "invisible_all_L"), True
            )
            horizons = sorted({children - 1, children, children + 1, 2 * children})
            for horizon in horizons:
                summaries = set()
                for lam in LAMBDAS:
                    seed += 1
                    result = long_case(parents, children, horizon, lam, seed, units)
                    for key in ("image", "kernel", "necessity", "sufficiency", "fiber"):
                        aggregate[key] &= result[key]
                    summaries.add((result["rank"], result["dim_conditions"], result["nullity"]))
                aggregate["image"] &= len(summaries) == 1
                rank_o, dim_c, nullity = sorted(summaries)[0]
                cases[f"P{parents}_r{children}_L{horizon}"] = {
                    "rank_O": rank_o,
                    "dim_conditions": dim_c,
                    "nullity": nullity,
                    "lambdas": [encode(lam) for lam in LAMBDAS],
                }
            ladder_ok = True
            for lam in LAMBDAS:
                top = 2 * children + 1
                ladder = []
                for horizon in range(top + 1):
                    ladder.append(
                        parents * children
                        - rank(units.columns(parents, children, lam, horizon))
                    )
                expected = [
                    parents * (children - h - 1) + h if h <= children - 1 else children - 1
                    for h in range(top + 1)
                ]
                ladder_ok &= ladder == expected
                top_columns = units.columns(parents, children, lam, top)
                aggregate["invisible_all_L"] &= all(
                    combine(d, top_columns) == (Q(0),) * (parents * (top + 1))
                    for d in invisible_basis(parents, children)
                )
                if lam == Q(1, 2):
                    ladders[f"P{parents}_r{children}_lambda_1/2"] = ladder
            tag = f"P{parents}_r{children}"
            checks[f"thm_long_image_equals_equal_sums_plus_recurrence_{tag}"] = aggregate["image"]
            checks[f"thm_long_generated_histories_satisfy_recurrence_{tag}"] = aggregate["necessity"]
            checks[f"thm_long_admissible_histories_lift_via_short_theorem_{tag}"] = aggregate["sufficiency"]
            checks[f"thm_long_kernel_equals_N_r_dimension_r-1_{tag}"] = aggregate["kernel"]
            checks[f"thm_long_fiber_is_lift_plus_N_r_{tag}"] = aggregate["fiber"]
            checks[f"thm_long_N_r_invisible_at_every_horizon_{tag}"] = aggregate["invisible_all_L"]
            checks[f"kernel_ladder_short_then_saturated_at_r-1_{tag}"] = ladder_ok
            # At L = r-1 both theorems apply: the condition set is M_{P,r-1}.
            overlap = cases[f"P{parents}_r{children}_L{children - 1}"]
            checks[f"thm_long_and_short_agree_at_L=r-1_{tag}"] = (
                overlap["nullity"]
                == parents * (children - (children - 1) - 1) + (children - 1)
                == children - 1
                and overlap["dim_conditions"] == parents * children - (children - 1)
            )

    # ------------------------------------------------------------------
    # cor:minimal-r: exhaustive search for the least r, and the proof's
    # counterexamples for every r <= L.
    minimal: dict[str, dict[str, int]] = {}
    counterexamples: dict[str, Any] = {}
    for parents in SWEEP_PARENTS:
        rmin_ok = sharp_ok = codim_ok = True
        stream = Stream(30 + parents)
        for lam in LAMBDAS:
            found: dict[str, int] = {}
            for horizon in range(6):
                dim_m = parents * (horizon + 1) - rank(mass_constraint_rows(parents, horizon))
                onto: dict[int, bool] = {}
                for children in range(1, horizon + 3):
                    columns = units.columns(parents, children, lam, horizon)
                    in_m = all(
                        dot(row, column) == 0
                        for row in mass_constraint_rows(parents, horizon)
                        for column in columns
                    )
                    rank_o = rank(columns)
                    onto[children] = in_m and rank_o == dim_m
                    if children <= horizon + 1:
                        codim_ok &= dim_m - rank_o == (parents - 1) * (horizon + 1 - children)
                    if children <= horizon:
                        zero = (Q(0),) * parents
                        v_simple = tuple(
                            Q(1) if k == 0 else Q(-1) if k == 1 else Q(0) for k in range(parents)
                        )
                        raw = [stream.rational() for _ in range(parents)]
                        v_random = tuple(v - sum(raw, Q(0)) / parents for v in raw)
                        if not any(v_random):
                            v_random = v_simple
                        for v in (v_simple, v_random):
                            y = tuple(
                                v if n == children else zero for n in range(horizon + 1)
                            )
                            in_mass = len({sum(state, Q(0)) for state in y}) == 1
                            outside = rank(columns + [flatten(y)]) == rank_o + 1
                            residual = recurrence_residual(y, lam, children, 0)
                            sharp_ok &= in_mass and outside and residual == v and any(v)
                        if lam == Q(1, 2):
                            counterexamples[f"P{parents}_L{horizon}_r{children}"] = {
                                "v": encode_vector(v_simple),
                                "rank_O": rank_o,
                                "dim_M": dim_m,
                                "codimension": dim_m - rank_o,
                            }
                least = min((r for r, ok in onto.items() if ok), default=0)
                rmin_ok &= least == horizon + 1 and onto[horizon + 2]
                found[f"L{horizon}"] = least
            minimal[f"P{parents}_lambda_{encode(lam)}"] = found
        checks[f"cor_minimal_child_count_is_L+1_by_search_P{parents}"] = rmin_ok
        checks[f"cor_sharpness_history_mass_conserving_not_realizable_P{parents}"] = sharp_ok
        checks[f"cor_image_codimension_in_M_is_(P-1)(L+1-r)_P{parents}"] = codim_ok

    # Hypotheses: P = 1 makes every mass-conserving history realizable with
    # r = 1, and lambda = 0 destroys surjectivity even at r = L+1.
    p1_ok = True
    for lam in LAMBDAS:
        for horizon in range(1, 5):
            columns = units.columns(1, 1, lam, horizon)
            p1_ok &= rank(columns) == 1 == 1 * (horizon + 1) - horizon
    checks["hypothesis_P>=2_needed_P1_realizes_all_with_r1"] = p1_ok
    lam0_columns = [flatten(history(unit(6, c), Q(0), 2, 3, 2)) for c in range(6)]
    checks["hypothesis_lambda>0_needed_not_onto_at_r=L+1"] = rank(lam0_columns) == 2 < 2 * 3 - 2
    beyond = True
    for lam in (Q(3, 2), Q(-1, 2)):
        for parents, children, horizon in ((2, 2, 4), (3, 3, 6)):
            seed += 1
            result = long_case(parents, children, horizon, lam, seed, units)
            beyond &= result["image"] and result["kernel"] and result["sufficiency"]
    checks["observation_long_theorem_algebra_also_holds_at_lambda_3/2_and_-1/2"] = beyond

    # ------------------------------------------------------------------
    # ex:nonrealizable.
    ex = MANUSCRIPT_EXAMPLE_NONREALIZABLE
    p_, r_, l_, lam_num, lam_den = ex["parameters"][2]
    parents, children, horizon = int(p_), int(r_), int(l_)
    lam = Q(int(lam_num), int(lam_den))
    values = [Q(v) for v in ex["history"][2]]
    y = tuple(tuple(values[2 * n : 2 * n + 2]) for n in range(horizon + 1))
    prediction = recurrence_prediction(y, lam, children, 0)
    residual = recurrence_residual(y, lam, children, 0)
    lead, coefficients = recurrence_coefficients(children, lam)
    expanded_lhs = tuple(
        sum((comb(children, j) * (-(1 - lam)) ** (children - j) * y[j][k] for j in range(children + 1)), Q(0))
        - lead * parent_shift(y[0])[k]
        for k in range(parents)
    )
    columns = units.columns(parents, children, lam, horizon)
    checks["example_nonrealizable_mass_conserving_and_nonnegative"] = (
        len({sum(state, Q(0)) for state in y}) == 1 and all(v >= 0 for state in y for v in state)
    )
    checks["example_nonrealizable_recurrence_coefficients_r2_lambda_1/2"] = (lead, coefficients) == (
        Q(1, 4),
        (Q(1, 4), Q(-1)),
    )
    checks["example_nonrealizable_prediction"] = prediction == tuple(Q(v) for v in ex["prediction"][2])
    checks["example_nonrealizable_residual"] = residual == tuple(Q(v) for v in ex["residual"][2])
    checks["example_nonrealizable_residual_sign_convention"] = expanded_lhs == residual and tuple(
        -v for v in residual
    ) != residual
    checks["example_nonrealizable_not_in_image_by_exact_rank"] = (
        rank(columns + [flatten(y)]) == rank(columns) + 1
    )
    # One more child (r = L+1 = 3) makes the same history realizable, but not by
    # a nonnegative state: x_{0,1} + x_{1,0} + x_{0,2} = -9 on the whole fiber.
    lift3 = normalized_lift(y, lam, parents, 3)
    witness = (Q(0), Q(1), Q(1), Q(1), Q(0), Q(0))  # x_{0,1} + x_{0,2} + x_{1,0}
    checks["example_nonrealizable_lifts_with_r3"] = history(lift3, lam, parents, 3, horizon) == y
    zero3 = tuple((Q(0),) * parents for _ in range(horizon + 1))
    checks["observation_example_r3_fiber_has_no_nonnegative_state"] = (
        dot(witness, lift3) == -9
        and all(dot(witness, d) == 0 for d in invisible_basis(parents, 3))
        and all(history(d, lam, parents, 3, horizon) == zero3 for d in invisible_basis(parents, 3))
        and parents * 3 - rank(units.columns(parents, 3, lam, horizon)) == 2
    )
    example = {
        "parameters": {"P": parents, "r": children, "L": horizon, "lambda": encode(lam)},
        "history": [encode_vector(state) for state in y],
        "recurrence_lead_lambda^r": encode(lead),
        "recurrence_coefficients_C(r,j)[-(1-lambda)]^(r-j)": encode_vector(coefficients),
        "predicted_y2": encode_vector(prediction),
        "residual_y2_minus_prediction": encode_vector(residual),
        "residual_opposite_sign": encode_vector(tuple(-v for v in residual)),
        "rank_O_2_2": rank(columns),
        "rank_with_history_appended": rank(columns + [flatten(y)]),
        "normalized_lift_with_r3": encode_vector(lift3),
    }

    # ------------------------------------------------------------------
    # tab:certificates rows 2 and 3, and the authors' generated outputs.
    row2, row3 = MANUSCRIPT_TABLE[2], MANUSCRIPT_TABLE[3]
    params2 = parse_parameters(row2[1])
    p2, r2, l2, lam2 = int(params2["P"]), int(params2["r"]), int(params2["L"]), params2["lambda"]
    stream = Stream(77)
    generated = [
        history(tuple(stream.rational() for _ in range(p2 * r2)), lam2, p2, r2, l2) for _ in range(3)
    ]
    generated += [split_states(column, p2) for column in units.columns(p2, r2, lam2, l2)]
    residuals = [
        [recurrence_residual(y_, lam2, r2, n) for n in range(l2 - r2 + 1)] for y_ in generated
    ]
    checks["table_row_2_generated_histories_residual_zero"] = row2[2] == "residual zero" and all(
        v == 0 for block in residuals for vector in block for v in vector
    )
    params3 = parse_parameters(row3[1])
    checks["table_row_3_parameters_match_example"] = (
        int(params3["P"]),
        int(params3["r"]),
        int(params3["L"]),
        params3["lambda"],
    ) == (parents, children, horizon, lam)
    residual_text = re.fullmatch(r"residual \\\((\(.*\))\\\)", row3[2])
    checks["table_row_3_residual_matches_exact"] = residual_text is not None and tuple(
        Q(v) for v in residual_text.group(1).strip("()").split(",")
    ) == residual

    header, line2, line3 = (next(csv.reader([text])) for text in (AUTHORS_CSV_HEADER, AUTHORS_CSV_LINE_2, AUTHORS_CSV_LINE_3))
    authors_residuals = [
        tuple(Q(int(a), int(b)) for a, b in re.findall(r"Fraction\((-?\d+), (\d+)\)", block))
        for block in re.findall(r"\[(Fraction[^\[\]]*)\]", line2[3])
    ]
    checks["authors_csv_header"] = header == ["certificate", "parameters", "result", "details"]
    checks["authors_csv_row_2_matches_exact_residuals"] = (
        line2[0] == "long_horizon_recurrence"
        and parse_parameters(line2[1]) == params2
        and line2[2] == row2[2]
        and authors_residuals == residuals[0]
        and len(authors_residuals) == l2 - r2 + 1
    )
    checks["authors_csv_row_3_matches_exact_values"] = (
        line3[0] == "nonrealizable_mass_conserving_history"
        and parse_parameters(line3[1]) == params3
        and line3[2] == "residual (1,-1)"
        and tuple(Q(v) for v in line3[2].removeprefix("residual ").strip("()").split(",")) == residual
        and parse_vectors(line3[3]) == {"y0": y[0], "y1": y[1], "y2": y[2]}
    )
    fragment = AUTHORS_TABLE_FRAGMENT
    segments = fragment.split("\\\\")
    fragment_rows = [
        [cell.strip() for cell in segment.split("&")]
        for segment in segments
        if "&" in segment
    ]
    manuscript_cells = [
        [
            cell.strip(r"\(\)").replace(r"\ ", " ").replace(r"\lambda", "lambda")
            if cell.startswith(r"\(")
            else cell.replace(r"\(", "").replace(r"\)", "")
            for cell in row
        ]
        for row in MANUSCRIPT_TABLE
    ]
    glued = [re.sub(r"^.*\\(toprule|midrule)", "", row[0]) for row in fragment_rows]
    checks["authors_fragment_cells_match_table_1"] = [
        [g] + row[1:] for g, row in zip(glued, fragment_rows, strict=True)
    ] == manuscript_cells
    undefined = sorted(set(re.findall(r"\\(?:toprule|midrule|bottomrule)[A-Za-z]+", fragment)))
    checks["authors_fragment_lacks_line_breaks_and_glues_rule_commands"] = (
        "\n" not in fragment and undefined == [r"\midruleNormalized", r"\topruleCertificate"]
    )
    discrepancies.append(
        {
            "location": (
                "experiments/tables/history_realizability_certificates.tex "
                "(generated body of tab:certificates)"
            ),
            "claim": (
                "the generated LaTeX table fragment reproduces Table 1 (Data and code "
                "availability: 'LaTeX table fragments supporting this study')"
            ),
            "finding": (
                "the fragment has no line terminators (one 327-byte line), so \\toprule and "
                "\\midrule run into the following cell text and TeX reads the undefined control "
                "words \\topruleCertificate and \\midruleNormalized; as provided it cannot be "
                "\\input. Its parameter cells are plain text ('lambda=1/2'), not math. Cell by "
                "cell the values agree with Table 1, which main.tex types by hand rather than "
                "inputting the fragment."
            ),
        }
    )

    # ------------------------------------------------------------------
    # Mutations a wrong implementation would accept.
    y_rand = generated[0]
    flipped = [
        a
        - lam2**r2 * parent_shift(y_rand[0])[k]
        - sum(comb(r2, j) * (-(1 - lam2)) ** (r2 - j) * y_rand[j][k] for j in range(r2))
        for k, a in enumerate(y_rand[r2])
    ]
    checks["mutation_recurrence_with_plus_sign_detected"] = any(flipped)
    lag_short = recurrence_coefficients(r2 - 1, lam2)
    short_prediction = [lag_short[0] * v for v in parent_shift(y_rand[0])]
    for j, coefficient in enumerate(lag_short[1]):
        short_prediction = [p - coefficient * v for p, v in zip(short_prediction, y_rand[j], strict=True)]
    checks["mutation_lag_r-1_recurrence_detected"] = tuple(short_prediction) != y_rand[r2 - 1]
    checks["mutation_unshifted_parent_term_detected"] = any(
        a
        - (
            lam2**r2 * y_rand[0][k]
            - sum(comb(r2, j) * (-(1 - lam2)) ** (r2 - j) * y_rand[j][k] for j in range(r2))
        )
        for k, a in enumerate(y_rand[r2])
    )

    # ------------------------------------------------------------------
    # Bibliography (transcribed; the test confirms main.tex and references.bib).
    cited = list(CITED_KEYS)
    entries = list(BIB_ENTRY_KEYS)
    checks["bibliography_every_cited_key_has_an_entry"] = set(cited) <= set(entries)
    checks["bibliography_no_duplicate_entry_keys"] = len(entries) == len(set(entries))
    bibliography = {
        "cited_keys_in_order_of_first_citation": cited,
        "cited_count": len(cited),
        "entries_not_cited": [key for key in entries if key not in set(cited)],
        "transcribed_fields": {key: list(value) for key, value in BIB_FIELDS.items()},
        "missing_entries": [key for key in cited if key not in set(entries)],
        "findings": [
            {
                "location": "references.bib entry PolemitisChristakisDrikakis2026OneStep (printed as [11])",
                "claim": "note = {Manuscript submitted to Applied Numerical Mathematics}",
                "finding": (
                    "stated status is inaccurate: as of 2026-10-08 the companion note has not been "
                    "submitted; the repository records it as staged for sign-off, to be posted to "
                    "arXiv and submitted after its release is frozen "
                    "(papers/initial-trace-entropy-separation/README.md)"
                ),
            },
            {
                "location": (
                    "references.bib entry polemitis2026exactfinitehorizonmemoryconditioning "
                    "(printed as [10])"
                ),
                "claim": "eprint = {2608.08633}, archivePrefix = {arXiv}, url = {...}",
                "finding": (
                    "the identifier is correct (arXiv:2608.08633, as recorded in "
                    "papers/finite-horizon-memory/README.md) but the unsrt style prints none of "
                    "eprint, archivePrefix or url, so reference [10] in Exact_lifting.pdf reads "
                    "only 'Exact finite-horizon memory, ... prediction, 2026.' with no arXiv "
                    "identifier or status; a note or howpublished field is needed"
                ),
            },
            {
                "location": "references.bib entries EymardGallouetHerbin2000 and MarkovskyDorfler2021 ([2], [14])",
                "claim": 'author fields Gallou{"e}t and D{"o}rfler',
                "finding": (
                    'the umlaut macro lacks its backslash; Exact_lifting.pdf prints Gallou"et and '
                    "D\"orfler (a closing double quote before the vowel); the fields need "
                    '{\\"e} and {\\"o}'
                ),
            },
        ],
    }

    check_count = len(checks)
    return {
        "schema": "certified-simulation/coarse-history-realizability/independent-long-horizon-minimal-r/v1",
        "arithmetic": (
            "fractions.Fraction and Python integers; exact ranks over Q by fraction-free "
            "integer elimination; tolerance zero"
        ),
        "results": (
            "The identity behind eq:history-recurrence, R S^r = Pi_P R, the triangular change of "
            "basis and the layer-difference identity; thm:long-realization (image, constructive "
            "sufficiency, kernel N_r, fibers); the kernel ladder; cor:minimal-r (exhaustive "
            "search for r_min, the proof's counterexamples, codimension); ex:nonrealizable; "
            "tab:certificates rows 2 and 3; the authors' CSV and table fragment; bibliography keys"
        ),
        "scope": {
            "analytic": (
                "The note proves thm:long-realization for P >= 2, r >= 1, 0 < lambda <= 1, "
                "L >= r-1, and cor:minimal-r for P >= 2, L >= 0."
            ),
            "executed": (
                "Exact checks for P = 2,3,4, r = 1..5, L in {r-1, r, r+1, 2r} and lambda in "
                "{1/10, 1/3, 1/2, 2/3, 9/10, 1}. Per instance: every column of O_{L,r} satisfies "
                "the equal-sum condition and the recurrence, and the exact rank equals the "
                "dimension of the condition set, so the image is exactly that set. Also checked "
                "per instance: constructive lifts of pseudo-random admissible histories, "
                "ker O = N_r, and fiber translates. Kernel ladders cover L = 0..2r+1. The "
                "minimal child count is found by search over r = 1..L+2 for P = 2,3,4, "
                "L = 0..5 and all six lambdas. The counterexample of cor:minimal-r is checked "
                "by exact rank at every r <= L."
            ),
            "exclusion": (
                "Finite sweeps do not prove the quantified statements; the proofs in the note "
                "carry the general claims. Bibliographic status is a dated observation, not a "
                "computation."
            ),
            "independence": (
                "No file of the authors' code was opened. Manuscript values, bibliography keys "
                "and the authors' generated outputs are transcribed only for comparison."
            ),
        },
        "interpretations": [
            "The recurrence residual is y^{n+r} minus the right-hand side of "
            "eq:history-recurrence, equal to the left side minus the right side of the "
            "expanded identity; for ex:nonrealizable it is (1,-1), as printed and as in the "
            "authors' outputs. The opposite sign convention would give (-1,1).",
            "tab:certificates row 2 does not specify its generated history; the claim is "
            "checked for every fine state of that instance (each column of O_{6,3} satisfies "
            "the recurrence) and for three pseudo-random states. The authors' CSV lists the "
            "four residual vectors n = 0..3, all zero.",
            "Condition 1 of thm:long-realization (equal sums of the first r states) together "
            "with the recurrence implies equal sums at all times, because the recurrence "
            "coefficients sum to one.",
        ],
        "long_horizon_cases": cases,
        "kernel_ladders_lambda_1/2": ladders,
        "minimal_child_count_by_search": minimal,
        "sharpness_counterexamples_lambda_1/2": counterexamples,
        "example_nonrealizable": example,
        "table_1_row_2_residual_blocks": len(residuals[0]),
        "bibliography": bibliography,
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
        / "long-horizon-minimal-r-certificate.json",
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

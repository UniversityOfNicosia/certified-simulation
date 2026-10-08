#!/usr/bin/env python3
"""Independent exact certificate: the upwind hierarchy, the short-horizon
image and fibers, and the normalized lift, for the note "Exact Lifting of
Coarse Histories in Periodic Upwind Finite Volumes" (Polemitis, Christakis,
Drikakis).

Every object is built here from the manuscript's definitions alone
(sec:hierarchy and sec:short-horizon):

* fine states x_{K,j} (parents K = 0..P-1, children j = 0..r-1), stored
  parent-major at index K*r + j; the downstream shift S from its two-case
  definition; A = (1-lambda) I + lambda S; the parent average R; the history
  map O_{L,r} = [R; RA; ...; RA^L]; the parent shift Pi_P, the periodic
  difference B = I - Pi_P, and the note's concrete formula for B_0^{-1};
* the collar-to-flux map T_{L,lambda} and its binomial inverse;
* the normalized lift of thm:short-realization, assembled step by step as in
  its proof: zero-mean flux registers, collar inversion, zero interior
  children, and the j = 0 child for the initial parent averages.

The script replays, in exact rational arithmetic:

* the setting: S and A against the displayed upwind update, conservation,
  R S^r = Pi_P R, the range and kernel of B, the B_0^{-1} formula, and the
  parent-update identity y^{t+1} = y^t - B Phi^t / r;
* the flux formula, the triangular structure and diagonal of T_{L,lambda},
  and its binomial inverse, also as the integer identity that proves the
  inverse for every lambda != 0;
* thm:short-realization for P = 2..4, r = 1..5, every L = 0..r-1, and
  lambda in {1/10, 1/3, 1/2, 2/3, 9/10, 1}, plus P = 5, r = 6.  The image
  equals M_{P,L}: every column of O_{L,r} lies in M_{P,L} and the exact rank
  equals dim M_{P,L}.  The normalized lift is a linear right inverse,
  checked on a basis of M_{P,L} and on pseudo-random histories.  It obeys
  the convention and is the only fiber point that does.  The kernel has
  dimension P(r-L-1)+L and is spanned by the note's L gauge directions and
  P(r-L-1) interior directions, and fibers are translates of the kernel;
* ex:canonical-lift value by value, and row 1 of tab:certificates against
  the manuscript and the authors' generated outputs;
* the remark on algebraic versus stable lifting: every lift of a fixed O(1)
  history has a collar spread of exactly r lambda^{-L}, and B_0^{-1} grows
  linearly in P;
* the hypothesis lambda > 0: at lambda = 0 the history map is not onto.

The script is self-contained, deterministic, and uses only the Python
standard library.  It never reads the authors' code.  Values quoted from the
manuscript and from the authors' generated outputs are embedded below as
transcriptions, and the companion test re-reads those files to confirm each
one.  The finite sweeps detect errors; they do not replace the note's proofs.
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
# Transcriptions.  Each manuscript entry is (label, template, values): the
# whitespace-free LaTeX displayed after that label is the template with each
# "#" replaced, in order, by the values.  The test confirms every filled
# template against manuscript/main.tex.

MANUSCRIPT_FORMULAS: dict[str, tuple[str, str]] = {
    "upwind_update": ("sec:hierarchy", r"u_i^{n+1}=(1-\lambda)u_i^n+\lambdau_{i-1}^n,"),
    "restriction": ("sec:hierarchy", r"(Rx)_K=\frac1r\sum_{j=0}^{r-1}x_{K,j}."),
    "shift": (
        "sec:hierarchy",
        r"(Sx)_{K,j}=\begin{cases}x_{K,j-1},&j\ge1,\\x_{K-1,r-1},&j=0,\end{cases}",
    ),
    "upwind_operator": ("sec:hierarchy", r"A_\lambda=(1-\lambda)I+\lambdaS."),
    "parent_shift": ("sec:hierarchy", r"(\Pi_Py)_K=y_{K-1},"),
    "difference": ("sec:hierarchy", r"B=I-\Pi_P,\qquad(B\Phi)_K=\Phi_K-\Phi_{K-1}."),
    "register_partial_sums": (
        "sec:hierarchy",
        r"\widehat\Phi_0=0,\qquad\widehat\Phi_K=\sum_{m=1}^{K}g_m,\qquadK=1,\ldots,P-1.",
    ),
    "register_zero_mean": (
        "sec:hierarchy",
        r"B_0^{-1}g=\widehat\Phi-\frac1P\left(\sum_{K=0}^{P-1}\widehat\Phi_K\right)\mathbf1_P.",
    ),
    "mass_conserving": ("sec:hierarchy", r"\mathbf1_P^\topy^n=\mathbf1_P^\topy^0"),
    "collar": ("sec:short-horizon", r"c_{K,j}=x_{K,r-1-j},"),
    "flux_formula": (
        "sec:short-horizon",
        r"\Phi_{K,t}=\lambda(A_\lambda^tx)_{K,r-1}=\lambda\sum_{j=0}^{t}"
        r"\binom{t}{j}(1-\lambda)^{t-j}\lambda^jc_{K,j},",
    ),
    "collar_map_diagonal": ("sec:short-horizon", r"\lambda,\lambda^2,\ldots,\lambda^L,"),
    "binomial_inverse": (
        "sec:short-horizon",
        r"c_{K,t}=\lambda^{-(t+1)}\sum_{j=0}^{t}\binom{t}{j}\bigl[-(1-\lambda)\bigr]^{t-j}\Phi_{K,j},",
    ),
    "short_hypotheses": ("thm:short-realization", r"\(0\leL\ler-1\)"),
    "short_image": ("thm:short-realization", r"\im{\calO}_{L,r}={\calM}_{P,L}."),
    "short_right_inverse": ("thm:short-realization", r"{\calO}_{L,r}{\calL}_{L,r}=I_{{\calM}_{P,L}}."),
    "short_fiber_dimension": ("thm:short-realization", r"P(r-L-1)+L."),
    "short_interior_count": ("thm:short-realization", r"\[P(r-L-1)\]unexposedinteriorchildvalues."),
    "lift_zero_order": (
        "thm:short-realization",
        r"x_{K,0}=ry_K^0,\qquadx_{K,j}=0\quad(j=1,\ldots,r-1).",
    ),
    "lift_registers": ("thm:short-realization", r"g^t=r(y^t-y^{t+1})."),
    "lift_interior": ("thm:short-realization", r"x_{K,j}=0,\qquadj=1,\ldots,r-L-1,"),
    "lift_mass_child": ("thm:short-realization", r"x_{K,0}=ry_K^0-\sum_{j=1}^{r-1}x_{K,j}."),
    "kernel_dimension": ("thm:short-realization", r"L+P(r-L-1)."),
    "sharp_count_fiber": ("ex:canonical-lift", r"\[r=L+1,\]thefiberdimensionin\cref{thm:short-realization}reducesto\[L.\]"),
}

# ex:canonical-lift, value by value.
MANUSCRIPT_EXAMPLE_CANONICAL_LIFT: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "parameters": (
        "ex:canonical-lift",
        r"P=#,\qquadL=#,\qquadr=#,\qquad\lambda=\frac##.",
        ("2", "1", "2", "1", "2"),
    ),
    "history": ("ex:canonical-lift", r"y^0=(#,#),\qquady^1=(#,#).", ("0", "0", "1", "-1")),
    "g0": ("ex:canonical-lift", r"g^0=r(y^0-y^1)=(#,#).", ("-2", "2")),
    "register": ("ex:canonical-lift", r"\Phi^0=(#,#).", ("-1", "1")),
    "flux_factor": ("ex:canonical-lift", r"\Phi_{K,0}=\frac##c_{K,0},", ("1", "2")),
    "collar": ("ex:canonical-lift", r"c_{0,0}=#,\qquadc_{1,0}=#.", ("-2", "2")),
    "lift": ("ex:canonical-lift", r"x=(#,#\mid#,#).", ("2", "-2", "-2", "2")),
    "observed": ("ex:canonical-lift", r"Rx=(#,#),\qquadRA_{1/2}x=(#,#).", ("0", "0", "1", "-1")),
    "reproduces": ("ex:canonical-lift", r"{\calO}_{1,2}x=(y^0,y^1).", ()),
}

# Row 1 of tab:certificates as typed in main.tex (cells stripped).
MANUSCRIPT_TABLE_ROW_1: tuple[str, str, str] = (
    "Normalized short-horizon lift",
    r"\(P=2,\ L=1,\ r=2,\ \lambda=1/2\)",
    "exact lift",
)

# Row 1 of the authors' generated outputs, verbatim.
AUTHORS_CSV_LINE_1 = (
    'canonical_short_horizon_lift,"P=2, L=1, r=2, lambda=1/2",exact lift,'
    '"y0=(0,0), y1=(1,-1), x=(2,-2|-2,2)"'
)
AUTHORS_FRAGMENT_ROW_1 = r"Normalized short-horizon lift & P=2, L=1, r=2, lambda=1/2 & exact lift \\"


def fill(template: str, values: Sequence[str]) -> str:
    for value in values:
        template = template.replace("#", value, 1)
    return template


# ---------------------------------------------------------------------------
# Encoding and deterministic pseudo-random rationals.


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


# ---------------------------------------------------------------------------
# Exact linear algebra over Q.


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


def transpose(rows: Sequence[Sequence[Q]]) -> list[Vector]:
    return [tuple(column) for column in zip(*rows, strict=True)]


def matmul(a: Sequence[Sequence[Q]], b: Sequence[Sequence[Q]]) -> tuple[Vector, ...]:
    columns = transpose(b)
    return tuple(tuple(dot(row, column) for column in columns) for row in a)


def identity(size: int) -> tuple[Vector, ...]:
    return tuple(tuple(Q(int(i == j)) for j in range(size)) for i in range(size))


def unit(size: int, index: int) -> Vector:
    return tuple(Q(int(i == index)) for i in range(size))


def flatten(history_: History) -> Vector:
    return tuple(v for state in history_ for v in state)


# ---------------------------------------------------------------------------
# The hierarchy of sec:hierarchy.


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
    """A_lambda x = (1 - lambda) x + lambda S x."""
    shifted = shift_s(x, parents, children)
    return tuple((1 - lam) * a + lam * b for a, b in zip(x, shifted, strict=True))


def upwind_by_formula(u: Vector, lam: Q) -> Vector:
    """The displayed update u_i^{n+1} = (1 - lambda) u_i + lambda u_{i-1}, periodic."""
    size = len(u)
    return tuple((1 - lam) * u[i] + lam * u[(i - 1) % size] for i in range(size))


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


def difference_b(phi: Vector) -> Vector:
    return tuple(phi[k] - phi[k - 1] for k in range(len(phi)))


def register_inverse(g: Vector) -> Vector:
    """The note's concrete B_0^{-1}: partial sums from K = 1, then remove the mean."""
    partial = [Q(0)]
    for k in range(1, len(g)):
        partial.append(partial[-1] + g[k])
    mean = sum(partial, Q(0)) / len(g)
    return tuple(value - mean for value in partial)


def outgoing_fluxes(x: Vector, lam: Q, parents: int, children: int, t: int) -> Vector:
    """Phi_{K,t} = lambda (A^t x)_{K,r-1}."""
    state = tuple(x)
    for _ in range(t):
        state = upwind(state, lam, parents, children)
    return tuple(lam * state[k * children + children - 1] for k in range(parents))


def collar_to_flux(horizon: int, lam: Q) -> tuple[Vector, ...]:
    """T_{L,lambda}: entry (t, j) = lambda C(t,j) (1-lambda)^{t-j} lambda^j for j <= t."""
    return tuple(
        tuple(
            lam * comb(t, j) * (1 - lam) ** (t - j) * lam**j if j <= t else Q(0)
            for j in range(horizon)
        )
        for t in range(horizon)
    )


def flux_to_collar(horizon: int, lam: Q) -> tuple[Vector, ...]:
    """The note's binomial inverse: lambda^{-(t+1)} C(t,j) [-(1-lambda)]^{t-j}."""
    return tuple(
        tuple(
            lam ** (-(t + 1)) * comb(t, j) * (-(1 - lam)) ** (t - j) if j <= t else Q(0)
            for j in range(horizon)
        )
        for t in range(horizon)
    )


def mass_constraint_rows(parents: int, horizon: int) -> list[Vector]:
    """1^T y^n - 1^T y^0 = 0 for n = 1..L, as functionals on flattened histories."""
    width = parents * (horizon + 1)
    rows = []
    for n in range(1, horizon + 1):
        row = [Q(0)] * width
        for k in range(parents):
            row[n * parents + k] += 1
            row[k] -= 1
        rows.append(tuple(row))
    return rows


def mass_conserving_basis(parents: int, horizon: int) -> list[History]:
    zero = (Q(0),) * parents
    basis = []
    for k in range(parents):
        e = unit(parents, k)
        basis.append(tuple(e for _ in range(horizon + 1)))
    for n in range(1, horizon + 1):
        for k in range(1, parents):
            d = tuple(Q(1) if i == k else Q(-1) if i == 0 else Q(0) for i in range(parents))
            basis.append(tuple(d if m == n else zero for m in range(horizon + 1)))
    return basis


def random_mass_conserving(stream: Stream, parents: int, horizon: int) -> History:
    first = tuple(stream.rational() for _ in range(parents))
    total = sum(first, Q(0))
    states = [first]
    for _ in range(horizon):
        state = [stream.rational() for _ in range(parents)]
        state[0] += total - sum(state, Q(0))
        states.append(tuple(state))
    return tuple(states)


# ---------------------------------------------------------------------------
# The normalized lift and the fiber description of thm:short-realization.


def normalized_lift(
    y: History,
    lam: Q,
    parents: int,
    children: int,
    inverse: Sequence[Sequence[Q]] | None = None,
    register: Any = register_inverse,
) -> Vector:
    """The proof's lift; `inverse` and `register` are swappable only for mutation checks."""
    horizon = len(y) - 1
    if not 0 <= horizon <= children - 1:
        raise ValueError("the short-horizon lift needs 0 <= L <= r-1")
    x = [[Q(0)] * children for _ in range(parents)]
    if horizon >= 1:
        registers = [
            register(tuple(children * (a - b) for a, b in zip(y[t], y[t + 1], strict=True)))
            for t in range(horizon)
        ]
        if inverse is None:
            inverse = flux_to_collar(horizon, lam)
        for k in range(parents):
            phi = [registers[t][k] for t in range(horizon)]
            collar = [dot(inverse[t], phi) for t in range(horizon)]
            for j in range(horizon):
                x[k][children - 1 - j] = collar[j]
    # Unexposed interior children j = 1..r-L-1 stay zero; the j = 0 child
    # carries the initial parent average.
    for k in range(parents):
        x[k][0] = children * y[0][k] - sum(x[k][1:], Q(0))
    return tuple(v for row in x for v in row)


def satisfies_convention(x: Vector, y: History, lam: Q, parents: int, children: int) -> bool:
    horizon = len(y) - 1
    ok = restrict(x, parents, children) == y[0]
    for t in range(horizon):
        g = tuple(children * (a - b) for a, b in zip(y[t], y[t + 1], strict=True))
        fluxes = outgoing_fluxes(x, lam, parents, children, t)
        ok &= fluxes == register_inverse(g) and sum(fluxes, Q(0)) == 0
    for k in range(parents):
        for j in range(1, children - horizon):
            ok &= x[k * children + j] == 0
    return ok


def normalization_rows(parents: int, children: int, horizon: int, lam: Q) -> list[Vector]:
    """Functionals fixed by the convention: sum_K Phi_{K,t} (t < L), interior children."""
    size = parents * children
    rows = []
    for t in range(horizon):
        rows.append(
            tuple(
                sum(outgoing_fluxes(unit(size, c), lam, parents, children, t), Q(0))
                for c in range(size)
            )
        )
    for k in range(parents):
        for j in range(1, children - horizon):
            rows.append(unit(size, k * children + j))
    return rows


def kernel_basis(parents: int, children: int, horizon: int, lam: Q) -> tuple[list[Vector], list[Vector]]:
    """The proof's kernel: L gauge directions, then P(r-L-1) interior directions."""
    size = parents * children
    inverse = flux_to_collar(horizon, lam)
    gauge = []
    for t in range(horizon):
        collar = [inverse[s][t] for s in range(horizon)]  # T^{-1} e_t
        d = [Q(0)] * size
        for k in range(parents):
            for j in range(horizon):
                d[k * children + children - 1 - j] = collar[j]
            d[k * children] = -sum(d[k * children + 1 : (k + 1) * children], Q(0))
        gauge.append(tuple(d))
    interior = []
    for k in range(parents):
        for j in range(1, children - horizon):
            d = [Q(0)] * size
            d[k * children + j] = Q(1)
            d[k * children] = Q(-1)
            interior.append(tuple(d))
    return gauge, interior


def short_case(parents: int, children: int, horizon: int, lam: Q, seed: int) -> dict[str, Any]:
    """Every claim of thm:short-realization at one (P, r, L, lambda)."""
    size = parents * children
    stream = Stream(seed)
    columns = [
        flatten(history(unit(size, c), lam, parents, children, horizon)) for c in range(size)
    ]
    mass_rows = mass_constraint_rows(parents, horizon)
    dim_m = parents * (horizon + 1) - rank(mass_rows)
    rank_o = rank(columns)
    in_m = all(dot(row, column) == 0 for row in mass_rows for column in columns)
    image_equals_m = in_m and rank_o == dim_m == parents * (horizon + 1) - horizon

    # Right inverse on a basis of M_{P,L} and on pseudo-random histories; linearity.
    basis = mass_conserving_basis(parents, horizon)
    right_inverse = len(basis) == dim_m and rank([flatten(b) for b in basis]) == dim_m
    convention = True
    for y in basis:
        x = normalized_lift(y, lam, parents, children)
        right_inverse &= history(x, lam, parents, children, horizon) == y
        convention &= satisfies_convention(x, y, lam, parents, children)
    samples = [random_mass_conserving(stream, parents, horizon) for _ in range(2)]
    lifts = [normalized_lift(y, lam, parents, children) for y in samples]
    for x, y in zip(lifts, samples, strict=True):
        right_inverse &= history(x, lam, parents, children, horizon) == y
        convention &= satisfies_convention(x, y, lam, parents, children)
    a, b = stream.rational(), stream.rational()
    combined = tuple(
        tuple(a * u + b * v for u, v in zip(s0, s1, strict=True))
        for s0, s1 in zip(samples[0], samples[1], strict=True)
    )
    linear = normalized_lift(combined, lam, parents, children) == tuple(
        a * u + b * v for u, v in zip(lifts[0], lifts[1], strict=True)
    )
    if horizon == 0:
        zero_order = all(
            normalized_lift(y, lam, parents, children)
            == tuple(
                children * y[0][k] if j == 0 else Q(0)
                for k in range(parents)
                for j in range(children)
            )
            for y in samples
        )
        right_inverse &= zero_order

    # The convention fixes a unique point of each fiber.
    norm_rows = normalization_rows(parents, children, horizon, lam)
    unique = len(norm_rows) == size - rank_o and rank(transpose(columns) + norm_rows) == size

    # Kernel: dimension, explicit spanning set, gauge structure.
    nullity = size - rank_o
    formula = parents * (children - horizon - 1) + horizon
    gauge, interior = kernel_basis(parents, children, horizon, lam)
    kernel = gauge + interior
    zero_history = tuple((Q(0),) * parents for _ in range(horizon + 1))
    kernel_ok = (
        nullity == formula
        and len(gauge) == horizon
        and len(interior) == parents * (children - horizon - 1)
        and len(kernel) == nullity
        and all(history(d, lam, parents, children, horizon) == zero_history for d in kernel)
        and rank(kernel) == nullity
    )
    for t, d in enumerate(gauge):
        for s in range(horizon):
            kernel_ok &= outgoing_fluxes(d, lam, parents, children, s) == (Q(int(s == t)),) * parents
        for j in range(horizon):
            kernel_ok &= len({d[k * children + children - 1 - j] for k in range(parents)}) == 1

    # Fibers are translates: lift + any kernel combination reproduces the history.
    shifted = list(lifts[0])
    for d in kernel:
        c = stream.rational()
        shifted = [u + c * v for u, v in zip(shifted, d, strict=True)]
    fiber = history(tuple(shifted), lam, parents, children, horizon) == samples[0]

    return {
        "rank": rank_o,
        "dim_M": dim_m,
        "nullity": nullity,
        "formula": formula,
        "image_equals_M": image_equals_m,
        "right_inverse": right_inverse and linear,
        "convention": convention,
        "unique": unique,
        "fiber_dimension": nullity == formula,
        "kernel": kernel_ok,
        "fiber_translate": fiber,
    }


# ---------------------------------------------------------------------------


def build_certificate() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    discrepancies: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # sec:hierarchy.
    for parents in SWEEP_PARENTS:
        shift_ok = conserve_ok = closure_ok = update_ok = True
        stream = Stream(100 + parents)
        for children in SWEEP_CHILDREN:
            size = parents * children
            for lam in LAMBDAS:
                states = [unit(size, c) for c in range(size)]
                states += [tuple(stream.rational() for _ in range(size)) for _ in range(2)]
                for x in states:
                    ax = upwind(x, lam, parents, children)
                    shift_ok &= ax == upwind_by_formula(x, lam)
                    shift_ok &= shift_s(x, parents, children) == tuple(x[i - 1] for i in range(size))
                    conserve_ok &= sum(ax, Q(0)) == sum(x, Q(0))
                    shifted = x
                    for _ in range(children):
                        shifted = shift_s(shifted, parents, children)
                    closure_ok &= restrict(shifted, parents, children) == parent_shift(
                        restrict(x, parents, children)
                    )
                for x in states[size:]:
                    state = x
                    for t in range(2 * children + 1):
                        nxt = upwind(state, lam, parents, children)
                        fluxes = tuple(
                            lam * state[k * children + children - 1] for k in range(parents)
                        )
                        before = restrict(state, parents, children)
                        update_ok &= restrict(nxt, parents, children) == tuple(
                            u - v / children
                            for u, v in zip(before, difference_b(fluxes), strict=True)
                        )
                        update_ok &= fluxes == outgoing_fluxes(x, lam, parents, children, t)
                        state = nxt
        checks[f"setting_shift_matches_displayed_upwind_update_P{parents}"] = shift_ok
        checks[f"setting_update_conserves_fine_sum_P{parents}"] = conserve_ok
        checks[f"setting_RS^r_equals_Pi_P_R_P{parents}"] = closure_ok
        checks[f"setting_parent_update_identity_P{parents}"] = update_ok

    for parents in (2, 3, 4, 5, 8):
        stream = Stream(200 + parents)
        b_rows = transpose([difference_b(unit(parents, k)) for k in range(parents)])
        ones = (Q(1),) * parents
        range_kernel = (
            all(sum(difference_b(unit(parents, k)), Q(0)) == 0 for k in range(parents))
            and difference_b(ones) == (Q(0),) * parents
            and rank(b_rows) == parents - 1
        )
        zero_mean_basis = [
            tuple(Q(1) if i == k else Q(-1) if i == 0 else Q(0) for i in range(parents))
            for k in range(1, parents)
        ]
        injective = rank([difference_b(z) for z in zero_mean_basis]) == parents - 1
        formula_ok = True
        samples = zero_mean_basis + [
            tuple(v - m for v in g)
            for g in (tuple(stream.rational() for _ in range(parents)) for _ in range(4))
            for m in (sum(g, Q(0)) / parents,)
        ]
        for g in samples:
            phi = register_inverse(g)
            formula_ok &= difference_b(phi) == g and sum(phi, Q(0)) == 0
        checks[f"setting_B_range_zero_mean_kernel_constants_P{parents}"] = range_kernel
        checks[f"setting_B0_inverse_formula_unique_zero_mean_solution_P{parents}"] = (
            formula_ok and injective
        )

    # ------------------------------------------------------------------
    # The collar-to-flux map and its binomial inverse.
    collar_map: dict[str, Any] = {}
    for horizon in range(1, 7):
        triangular = inverse_ok = True
        for lam in LAMBDAS:
            t_matrix = collar_to_flux(horizon, lam)
            t_inverse = flux_to_collar(horizon, lam)
            triangular &= all(
                t_matrix[t][j] == 0 for t in range(horizon) for j in range(t + 1, horizon)
            ) and all(t_matrix[t][t] == lam ** (t + 1) for t in range(horizon))
            inverse_ok &= matmul(t_matrix, t_inverse) == identity(horizon)
            inverse_ok &= matmul(t_inverse, t_matrix) == identity(horizon)
        checks[f"collar_map_lower_triangular_diagonal_lambda^(t+1)_L{horizon}"] = triangular
        checks[f"collar_map_binomial_inverse_exact_L{horizon}"] = inverse_ok
    collar_map["T_3_lambda_1/2"] = [encode_vector(row) for row in collar_to_flux(3, Q(1, 2))]
    collar_map["T_3_lambda_1/2_inverse"] = [encode_vector(row) for row in flux_to_collar(3, Q(1, 2))]
    # (T T^{-1})_{ts} = (1-lambda)^{t-s} sum_j C(t,j) C(j,s) (-1)^{j-s}; the integer
    # identity below therefore proves the inverse for every lambda != 0.
    checks["collar_map_binomial_inversion_integer_identity_t_le_12"] = all(
        sum(comb(t, j) * comb(j, s) * (-1) ** (j - s) for j in range(s, t + 1)) == int(t == s)
        for t in range(13)
        for s in range(t + 1)
    )
    flux_ok = True
    stream = Stream(300)
    for parents in (2, 3):
        for horizon in range(1, 5):
            for children in (horizon + 1, horizon + 2):
                for lam in LAMBDAS:
                    t_matrix = collar_to_flux(horizon, lam)
                    x = tuple(stream.rational() for _ in range(parents * children))
                    for k in range(parents):
                        collar = [x[k * children + children - 1 - j] for j in range(horizon)]
                        phi = tuple(
                            outgoing_fluxes(x, lam, parents, children, t)[k] for t in range(horizon)
                        )
                        flux_ok &= phi == tuple(dot(row, collar) for row in t_matrix)
    checks["collar_flux_formula_matches_upwind_evolution"] = flux_ok

    # ------------------------------------------------------------------
    # thm:short-realization on the sweep.
    cases: dict[str, Any] = {}
    seed = 1000
    for parents in SWEEP_PARENTS:
        for children in SWEEP_CHILDREN:
            aggregate = {
                key: True
                for key in (
                    "image_equals_M",
                    "right_inverse",
                    "convention",
                    "unique",
                    "fiber_dimension",
                    "kernel",
                    "fiber_translate",
                )
            }
            for horizon in range(children):
                summaries = set()
                for lam in LAMBDAS:
                    seed += 1
                    result = short_case(parents, children, horizon, lam, seed)
                    for key in aggregate:
                        aggregate[key] &= result[key]
                    summaries.add((result["rank"], result["dim_M"], result["nullity"], result["formula"]))
                (rank_o, dim_m, nullity, formula), *rest = sorted(summaries)
                aggregate["fiber_dimension"] &= not rest
                cases[f"P{parents}_r{children}_L{horizon}"] = {
                    "rank_O": rank_o,
                    "dim_M": dim_m,
                    "nullity": nullity,
                    "P(r-L-1)+L": formula,
                    "lambdas": [encode(lam) for lam in LAMBDAS],
                }
            tag = f"P{parents}_r{children}"
            checks[f"thm_short_image_equals_M_{tag}"] = aggregate["image_equals_M"]
            checks[f"thm_short_normalized_lift_is_linear_right_inverse_{tag}"] = aggregate["right_inverse"]
            checks[f"thm_short_lift_obeys_zero_mean_gauge_zero_interior_convention_{tag}"] = aggregate["convention"]
            checks[f"thm_short_lift_is_unique_fiber_point_obeying_convention_{tag}"] = aggregate["unique"]
            checks[f"thm_short_fiber_dimension_P(r-L-1)+L_{tag}"] = aggregate["fiber_dimension"]
            checks[f"thm_short_kernel_spanned_by_gauge_and_interior_{tag}"] = aggregate["kernel"]
            checks[f"thm_short_fiber_is_lift_plus_kernel_{tag}"] = aggregate["fiber_translate"]
            if children >= 2:
                sharp = cases[f"P{parents}_r{children}_L{children - 1}"]
                checks[f"thm_short_sharp_count_r=L+1_fiber_is_L_gauge_only_{tag}"] = (
                    sharp["nullity"] == children - 1 == sharp["P(r-L-1)+L"]
                )
    large_ok = True
    for horizon in (0, 3, 5):
        for lam in (Q(1, 3), Q(1)):
            seed += 1
            result = short_case(5, 6, horizon, lam, seed)
            large_ok &= all(
                result[key]
                for key in (
                    "image_equals_M",
                    "right_inverse",
                    "convention",
                    "unique",
                    "fiber_dimension",
                    "kernel",
                    "fiber_translate",
                )
            )
            cases[f"P5_r6_L{horizon}_lambda_{encode(lam)}"] = {
                "rank_O": result["rank"],
                "dim_M": result["dim_M"],
                "nullity": result["nullity"],
                "P(r-L-1)+L": result["formula"],
            }
    checks["thm_short_all_claims_P5_r6_L0_3_5"] = large_ok

    # ------------------------------------------------------------------
    # ex:canonical-lift, value by value.
    ex = MANUSCRIPT_EXAMPLE_CANONICAL_LIFT
    p_, l_, r_, lam_num, lam_den = ex["parameters"][2]
    parents, horizon, children = int(p_), int(l_), int(r_)
    lam = Q(int(lam_num), int(lam_den))
    hist_values = [Q(v) for v in ex["history"][2]]
    y = (tuple(hist_values[:2]), tuple(hist_values[2:]))
    g0 = tuple(children * (a - b) for a, b in zip(y[0], y[1], strict=True))
    phi0 = register_inverse(g0)
    t_matrix = collar_to_flux(horizon, lam)
    collar = tuple(dot(flux_to_collar(horizon, lam)[0], (phi0[k],)) for k in range(parents))
    x = normalized_lift(y, lam, parents, children)
    rx = restrict(x, parents, children)
    rax = restrict(upwind(x, lam, parents, children), parents, children)
    checks["example_canonical_lift_history_mass_conserving"] = sum(y[0], Q(0)) == sum(y[1], Q(0))
    checks["example_canonical_lift_g0"] = g0 == tuple(Q(v) for v in ex["g0"][2])
    checks["example_canonical_lift_register_unique_zero_mean"] = (
        phi0 == tuple(Q(v) for v in ex["register"][2])
        and difference_b(phi0) == g0
        and sum(phi0, Q(0)) == 0
    )
    checks["example_canonical_lift_flux_factor_T_1_lambda"] = t_matrix == (
        (Q(int(ex["flux_factor"][2][0]), int(ex["flux_factor"][2][1])),),
    )
    checks["example_canonical_lift_collar"] = collar == tuple(Q(v) for v in ex["collar"][2]) and all(
        x[k * children + children - 1] == collar[k] for k in range(parents)
    )
    checks["example_canonical_lift_x"] = x == tuple(Q(v) for v in ex["lift"][2])
    checks["example_canonical_lift_Rx_and_RAx"] = rx + rax == tuple(Q(v) for v in ex["observed"][2])
    checks["example_canonical_lift_reproduces_history"] = history(x, lam, parents, children, horizon) == y
    # Alternative readings, recorded for the interpretation notes.
    upstream = restrict(
        tuple((1 - lam) * x[i] + lam * x[(i + 1) % len(x)] for i in range(len(x))),
        parents,
        children,
    )
    raw_register = (Q(0), g0[1])  # partial sums without the mean removed
    raw_collar = tuple(v / lam for v in raw_register)
    raw_lift = (-raw_collar[0], raw_collar[0], -raw_collar[1], raw_collar[1])
    gauge, _ = kernel_basis(parents, children, horizon, lam)
    gauge_gap = tuple(u - v for u, v in zip(raw_lift, x, strict=True))
    checks["example_canonical_lift_upstream_shift_reading_fails"] = upstream != y[1]
    checks["example_canonical_lift_unnormalized_gauge_differs_by_gauge_direction"] = (
        history(raw_lift, lam, parents, children, horizon) == y
        and raw_lift != x
        and rank([gauge[0], gauge_gap]) == 1
    )
    example = {
        "parameters": {"P": parents, "L": horizon, "r": children, "lambda": encode(lam)},
        "y0": encode_vector(y[0]),
        "y1": encode_vector(y[1]),
        "g0": encode_vector(g0),
        "Phi0": encode_vector(phi0),
        "T_1_lambda": [encode_vector(row) for row in t_matrix],
        "collar_c_K0": encode_vector(collar),
        "normalized_lift_x": encode_vector(x),
        "Rx": encode_vector(rx),
        "RAx": encode_vector(rax),
        "upstream_shift_RAx": encode_vector(upstream),
        "unnormalized_gauge_lift": encode_vector(raw_lift),
    }

    # ------------------------------------------------------------------
    # tab:certificates row 1 against the manuscript and the authors' outputs.
    name, params_tex, result_tex = MANUSCRIPT_TABLE_ROW_1
    params = dict(re.findall(r"(P|L|r|\\lambda)=([0-9/]+)", params_tex))
    checks["table_row_1_parameters_match_example"] = (
        int(params["P"]) == parents
        and int(params["L"]) == horizon
        and int(params["r"]) == children
        and Q(params[r"\lambda"]) == lam
    )
    checks["table_row_1_result_exact_lift"] = result_tex == "exact lift" and history(
        x, lam, parents, children, horizon
    ) == y
    csv_fields = next(csv.reader([AUTHORS_CSV_LINE_1]))
    csv_params = dict(re.findall(r"(P|L|r|lambda)=([0-9/]+)", csv_fields[1]))
    vectors = re.findall(r"(y0|y1|x)=\(([^)]*)\)", csv_fields[3])
    parsed = {key: tuple(Q(v) for v in body.replace("|", ",").split(",")) for key, body in vectors}
    checks["authors_csv_row_1_matches_exact_values"] = (
        csv_fields[0] == "canonical_short_horizon_lift"
        and csv_fields[2] == result_tex
        and {k: Q(v) for k, v in csv_params.items()}
        == {"P": parents, "L": horizon, "r": children, "lambda": lam}
        and parsed == {"y0": y[0], "y1": y[1], "x": x}
    )
    fragment_cells = [c.strip() for c in AUTHORS_FRAGMENT_ROW_1.removesuffix("\\\\").split("&")]
    checks["authors_fragment_row_1_matches_table_row_1"] = fragment_cells == [
        name,
        params_tex.strip(r"\(\)").replace(r"\ ", " ").replace(r"\lambda", "lambda"),
        result_tex,
    ]

    # ------------------------------------------------------------------
    # The remark on algebraic versus stable lifting.
    spread_ok = True
    spreads: dict[str, str] = {}
    stream = Stream(400)
    for lam in (Q(1, 10), Q(1, 100), Q(1, 1000)):
        for horizon in (1, 2, 3):
            for children in (horizon + 1, horizon + 2):
                zero = (Q(0), Q(0))
                y = tuple(zero for _ in range(horizon)) + ((Q(1), Q(-1)),)
                lift = normalized_lift(y, lam, 2, children)
                gauge, interior = kernel_basis(2, children, horizon, lam)
                members = [lift]
                for _ in range(3):
                    member = list(lift)
                    for d in gauge + interior:
                        c = stream.rational()
                        member = [u + c * v for u, v in zip(member, d, strict=True)]
                    members.append(tuple(member))
                expected = -children * lam ** (-horizon)
                for member in members:
                    spread_ok &= history(member, lam, 2, children, horizon) == y
                    gap = member[children - horizon] - member[children + children - horizon]
                    spread_ok &= gap == expected
                    spread_ok &= max(abs(v) for v in member) >= abs(expected) / 2
                spreads[f"L{horizon}_r{children}_lambda_{encode(lam)}"] = encode(expected)
    checks["remark_every_lift_has_collar_spread_r_lambda^-L"] = spread_ok
    growth_ok = True
    for parents in (4, 8, 12, 16, 20):
        g = tuple(Q(1) if k < parents // 2 else Q(-1) for k in range(parents))
        growth_ok &= max(abs(v) for v in register_inverse(g)) == Q(parents, 4)
    checks["remark_B0_inverse_square_wave_grows_as_P/4"] = growth_ok

    # ------------------------------------------------------------------
    # Hypotheses and scope.
    size = 4
    columns = [flatten(history(unit(size, c), Q(0), 2, 2, 1)) for c in range(size)]
    checks["hypothesis_lambda_zero_history_map_not_onto"] = (
        rank(columns) == 2 < 2 * 2 - 1 and collar_to_flux(1, Q(0)) == ((Q(0),),)
    )
    beyond = True
    for lam in (Q(3, 2), Q(-1, 2)):
        for parents, children, horizon in ((2, 3, 2), (3, 3, 1)):
            seed += 1
            result = short_case(parents, children, horizon, lam, seed)
            beyond &= all(
                result[key] for key in ("image_equals_M", "right_inverse", "unique", "kernel")
            )
    checks["observation_short_theorem_algebra_also_holds_at_lambda_3/2_and_-1/2"] = beyond

    # ------------------------------------------------------------------
    # Mutations a wrong implementation would accept.
    lam, parents, children, horizon = Q(1, 3), 3, 4, 2
    y = random_mass_conserving(Stream(500), parents, horizon)
    no_powers = tuple(
        tuple(comb(t, j) * (-(1 - lam)) ** (t - j) if j <= t else Q(0) for j in range(horizon))
        for t in range(horizon)
    )
    mutant = normalized_lift(y, lam, parents, children, inverse=no_powers)
    checks["mutation_inverse_without_lambda^-(t+1)_breaks_right_inverse"] = (
        history(mutant, lam, parents, children, horizon) != y
    )

    def partial_sums(g: Vector) -> Vector:
        out = [Q(0)]
        for value in g[1:]:
            out.append(out[-1] + value)
        return tuple(out)

    mutant = normalized_lift(y, lam, parents, children, register=partial_sums)
    checks["mutation_register_without_mean_removal_lifts_but_breaks_convention"] = (
        history(mutant, lam, parents, children, horizon) == y
        and not satisfies_convention(mutant, y, lam, parents, children)
    )
    case = cases[f"P{parents}_r{children}_L{horizon}"]
    checks["mutation_fiber_formula_P(r-L)+L_rejected"] = (
        case["nullity"] != parents * (children - horizon) + horizon
    )

    check_count = len(checks)
    return {
        "schema": "certified-simulation/coarse-history-realizability/independent-short-horizon-lifting/v1",
        "arithmetic": (
            "fractions.Fraction and Python integers; exact ranks over Q by fraction-free "
            "integer elimination; tolerance zero"
        ),
        "results": (
            "sec:hierarchy (S, A, R, conservation, R S^r = Pi_P R, B and the B_0^{-1} formula, "
            "the parent-update identity); the collar-to-flux map T_{L,lambda} and its binomial "
            "inverse; thm:short-realization (image, normalized right inverse, convention, fiber "
            "dimension, kernel basis, fibers); ex:canonical-lift; tab:certificates row 1; the "
            "remark on algebraic versus stable lifting"
        ),
        "scope": {
            "analytic": (
                "The note proves thm:short-realization for all P >= 2, r >= 1, 0 < lambda <= 1 "
                "and 0 <= L <= r-1."
            ),
            "executed": (
                "Exact checks for P = 2,3,4, r = 1..5, every L = 0..r-1 and lambda in "
                "{1/10, 1/3, 1/2, 2/3, 9/10, 1} (270 instances), and P = 5, r = 6, L = 0,3,5 at "
                "lambda = 1/3, 1. Per instance: im O_{L,r} = M_{P,L} by an exact rank count with "
                "every column of O_{L,r} in M_{P,L}; O L = I on a basis of M_{P,L}; linearity; "
                "the convention; uniqueness by full column rank of O stacked with the "
                "convention functionals; the kernel's dimension and explicit basis; fiber "
                "translates."
            ),
            "exclusion": (
                "Finite sweeps do not prove the quantified statements; the proofs in the note "
                "carry the general claims. The binomial inverse is proved here for every "
                "lambda != 0 through an integer identity."
            ),
            "independence": (
                "No file of the authors' code was opened. Manuscript values and the authors' "
                "generated outputs are transcribed only for comparison."
            ),
        },
        "interpretations": [
            "Fine states are stored parent-major, x_{K,j} at index K*r + j; the note's "
            "x=(2,-2|-2,2) lists (x_{0,0}, x_{0,1} | x_{1,0}, x_{1,1}).",
            "S is the downstream shift of the displayed two-case definition; it equals the "
            "periodic index shift i -> i-1 of the displayed update. Under the opposite "
            "(upstream) reading the example's lift would give RAx = (-1,1), not (1,-1).",
            "Flux registers carry the factor lambda: Phi_{K,t} = lambda (A^t x)_{K,r-1}, and "
            "y^{t+1} = y^t - B Phi^t / r.",
            "The normalized lift uses the zero-mean register B_0^{-1}g; the partial-sum "
            "register without the mean removed gives the lift (0,0|-4,4), also in the fiber, "
            "which differs from the normalized lift by a gauge direction. The authors' output "
            "matches the zero-mean convention.",
        ],
        "short_horizon_cases": cases,
        "collar_to_flux": collar_map,
        "example_canonical_lift": example,
        "remark_collar_spreads_c0_minus_c1": spreads,
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
        / "short-horizon-lifting-certificate.json",
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

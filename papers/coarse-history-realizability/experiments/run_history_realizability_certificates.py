"""
Exact rational certificates for the paper

    Exact Lifting of Coarse Histories in Periodic Upwind Finite Volumes

This script verifies representative finite instances of the analytical
theorems using Python's fractions.Fraction type.  It does not perform
floating-point simulation.  All reported identities and residuals are checked
in exact rational arithmetic.

The checks are:

1. Normalized short-horizon lift
   Parameters:
       P = 2, L = 1, r = 2, lambda = 1/2.

   Prescribed parent history:
       y^0 = (0, 0), y^1 = (1, -1).

   The zero-mean parent-interface register is:
       Phi^0 = (-1, 1).

   The normalized fine-grid lift is:
       x = (2, -2 | -2, 2).

   The script verifies exactly that:
       R x = y^0,
       R A_lambda x = y^1.

2. Long-horizon recurrence
   Parameters:
       P = 3, r = 3, L = 6, lambda = 1/2.

   A rational fine-grid state is evolved by the periodic upwind update.
   The generated parent history is checked against the lag-r recurrence
   from the paper.  The script verifies that every recurrence residual is
   exactly zero.

3. Nonrealizable nonnegative mass-conserving history
   Parameters:
       P = 2, r = 2, L = 2, lambda = 1/2.

   Prescribed parent history:
       y^0 = (1, 1), y^1 = (1, 1), y^2 = (2, 0).

   This history is nonnegative and conserves total parent mass, but it
   violates the lag-two recurrence.  The script verifies that the recurrence
   residual is:
       (1, -1).

Outputs:
    results/history_realizability_certificates.csv
    tables/history_realizability_certificates.tex

Run:
    python run_history_realizability_certificates.py

Notes:
    The generated LaTeX table is intended to be included in the manuscript.
    The certificates are reproducibility checks for finite algebraic examples;
    they are not substitutes for the analytical proofs.
"""

from __future__ import annotations

import csv
from fractions import Fraction
from pathlib import Path
from typing import List


Vector = List[Fraction]
History = List[Vector]


def ensure_dirs() -> None:
    Path("results").mkdir(exist_ok=True)
    Path("tables").mkdir(exist_ok=True)


def fmt_fraction(x: Fraction) -> str:
    """Return a fraction in LaTeX math notation."""
    if x.denominator == 1:
        return rf"${x.numerator}$"
    return rf"$\frac{{{x.numerator}}}{{{x.denominator}}}$"


def restrict_parent(x: Vector, P: int, r: int) -> Vector:
    """Compute parent averages from a fine-grid state."""
    if len(x) != P * r:
        raise ValueError("Incompatible fine-grid vector length.")

    return [
        sum(x[K * r:(K + 1) * r], Fraction(0)) / r
        for K in range(P)
    ]


def upwind_periodic(x: Vector, lam: Fraction) -> Vector:
    """One exact-rational periodic positive-upwind step."""
    N = len(x)
    return [(1 - lam) * x[i] + lam * x[(i - 1) % N] for i in range(N)]


def history_from_fine(
    x: Vector,
    P: int,
    r: int,
    lam: Fraction,
    L: int,
) -> History:
    """Return parent averages at times 0 through L."""
    u = list(x)
    history: History = []

    for _ in range(L + 1):
        history.append(restrict_parent(u, P, r))
        u = upwind_periodic(u, lam)

    return history


def parent_shift(y: Vector) -> Vector:
    """Parent-level periodic shift Pi_P y, with (Pi_P y)_K = y_{K-1}."""
    P = len(y)
    return [y[(K - 1) % P] for K in range(P)]


def solve_zero_mean_difference(g: Vector) -> Vector:
    """
    Solve B phi = g exactly, where (B phi)_K = phi_K - phi_{K-1},
    subject to sum(phi) = 0.
    """
    P = len(g)

    if sum(g, Fraction(0)) != 0:
        raise ValueError("The difference right-hand side must have zero sum.")

    anchored = [Fraction(0) for _ in range(P)]
    for K in range(1, P):
        anchored[K] = anchored[K - 1] + g[K]

    mean = sum(anchored, Fraction(0)) / P
    return [value - mean for value in anchored]


def verify_difference(phi: Vector, g: Vector) -> bool:
    """Check B phi = g."""
    P = len(phi)
    return all(phi[K] - phi[(K - 1) % P] == g[K] for K in range(P))


def collar_to_flux_matrix(L: int, lam: Fraction) -> List[List[Fraction]]:
    """Construct the short-horizon lower-triangular collar-to-flux matrix."""
    import math

    T = [[Fraction(0) for _ in range(L)] for _ in range(L)]

    for t in range(L):
        for j in range(t + 1):
            T[t][j] = (
                lam
                * Fraction(math.comb(t, j), 1)
                * (1 - lam) ** (t - j)
                * lam**j
            )

    return T


def solve_lower_triangular(T: List[List[Fraction]], rhs: Vector) -> Vector:
    """Solve T x = rhs for a lower-triangular exact-rational matrix."""
    n = len(rhs)
    x = [Fraction(0) for _ in range(n)]

    for i in range(n):
        residual = rhs[i] - sum(T[i][j] * x[j] for j in range(i))
        x[i] = residual / T[i][i]

    return x


def canonical_lift(
    Y: History,
    P: int,
    r: int,
    lam: Fraction,
) -> Vector:
    """
    Construct the canonical short-horizon lift.

    The procedure:
    1. choose zero-mean parent-interface registers;
    2. invert the collar-to-flux map in every parent;
    3. set unexposed interior children to zero;
    4. choose child 0 to match the prescribed initial parent mean.
    """
    L = len(Y) - 1

    if r < L + 1:
        raise ValueError("Canonical arbitrary-history lift requires r >= L + 1.")

    x = [Fraction(0) for _ in range(P * r)]

    if L == 0:
        for K in range(P):
            x[K * r] = r * Y[0][K]
        return x

    T = collar_to_flux_matrix(L, lam)

    registers_by_time: List[Vector] = []
    for t in range(L):
        g = [r * (Y[t][K] - Y[t + 1][K]) for K in range(P)]
        phi = solve_zero_mean_difference(g)

        if not verify_difference(phi, g):
            raise RuntimeError("Difference equation check failed.")

        registers_by_time.append(phi)

    for K in range(P):
        phi_K = [registers_by_time[t][K] for t in range(L)]
        collar = solve_lower_triangular(T, phi_K)

        for j in range(L):
            x[K * r + (r - 1 - j)] = collar[j]

        interior_sum = sum(x[K * r + j] for j in range(1, r))
        x[K * r] = r * Y[0][K] - interior_sum

    return x


def recurrence_next(
    Y: History,
    n: int,
    r: int,
    lam: Fraction,
) -> Vector:
    """Evaluate the lag-r parent recurrence."""
    import math

    P = len(Y[0])
    result = [lam**r * value for value in parent_shift(Y[n])]

    for j in range(r):
        coeff = Fraction(math.comb(r, j), 1) * (-(1 - lam)) ** (r - j)
        for K in range(P):
            result[K] -= coeff * Y[n + j][K]

    return result


def canonical_lift_certificate() -> dict:
    P, L, r = 2, 1, 2
    lam = Fraction(1, 2)

    Y = [
        [Fraction(0), Fraction(0)],
        [Fraction(1), Fraction(-1)],
    ]

    x = canonical_lift(Y, P, r, lam)
    generated = history_from_fine(x, P, r, lam, L)

    return {
        "certificate": "canonical_short_horizon_lift",
        "parameters": "P=2, L=1, r=2, lambda=1/2",
        "result": "exact lift" if generated == Y else "failed",
        "details": "y0=(0,0), y1=(1,-1), x=(2,-2|-2,2)",
    }


def long_recurrence_certificate() -> dict:
    P, r, L = 3, 3, 6
    lam = Fraction(1, 2)

    x = [
        Fraction(1), Fraction(2), Fraction(-1),
        Fraction(0), Fraction(3), Fraction(1),
        Fraction(-2), Fraction(1), Fraction(4),
    ]

    Y = history_from_fine(x, P, r, lam, L)

    residuals = []
    for n in range(L - r + 1):
        prediction = recurrence_next(Y, n, r, lam)
        residual = [Y[n + r][K] - prediction[K] for K in range(P)]
        residuals.append(residual)

    passed = all(all(value == 0 for value in residual) for residual in residuals)

    return {
        "certificate": "long_horizon_recurrence",
        "parameters": "P=3, r=3, L=6, lambda=1/2",
        "result": "residual zero" if passed else "failed",
        "details": f"residuals={residuals}",
    }


def nonrealizable_certificate() -> dict:
    P, r, L = 2, 2, 2
    lam = Fraction(1, 2)

    Y = [
        [Fraction(1), Fraction(1)],
        [Fraction(1), Fraction(1)],
        [Fraction(2), Fraction(0)],
    ]

    predicted = recurrence_next(Y, 0, r, lam)
    residual = [Y[2][K] - predicted[K] for K in range(P)]

    passed = residual == [Fraction(1), Fraction(-1)]

    return {
        "certificate": "nonrealizable_mass_conserving_history",
        "parameters": "P=2, r=2, L=2, lambda=1/2",
        "result": "residual (1,-1)" if passed else "failed",
        "details": "y0=(1,1), y1=(1,1), y2=(2,0)",
    }


def write_csv(rows: List[dict]) -> None:
    path = Path("results/history_realizability_certificates.csv")

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["certificate", "parameters", "result", "details"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_table(rows: List[dict]) -> None:
    path = Path("tables/history_realizability_certificates.tex")

    labels = {
    "canonical_short_horizon_lift": "Normalized short-horizon lift",
    "long_horizon_recurrence": "Long-horizon recurrence",
    "nonrealizable_mass_conserving_history": "Nonrealizable nonnegative history",
}

    with path.open("w") as f:
        f.write(r"\begin{tabular}{@{}lll@{}}" + "")
        f.write(r"\toprule" + "")
        f.write(r"Certificate & Parameters & Result \\" + "")
        f.write(r"\midrule" + "")

        for row in rows:
            f.write(
                f"{labels[row['certificate']]} & "
                f"{row['parameters']} & "
                f"{row['result']} \\\\"
            )

        f.write(r"\bottomrule" + "")
        f.write(r"\end{tabular}" + "")


def main() -> None:
    ensure_dirs()

    rows = [
        canonical_lift_certificate(),
        long_recurrence_certificate(),
        nonrealizable_certificate(),
    ]

    write_csv(rows)
    write_table(rows)

    print("Exact coarse-history realizability certificates:")
    for row in rows:
        print(f"  {row['certificate']}: {row['result']}")
        print(f"    {row['details']}")

    print("Wrote:")
    print("  results/history_realizability_certificates.csv")
    print("  tables/history_realizability_certificates.tex")


if __name__ == "__main__":
    main()

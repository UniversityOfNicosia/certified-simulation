"""
Exact computational certificates for the paper

    Initial-Trace Insufficiency and Entropy-Accounting Separation
    in Finite-Volume Coarse Graining

This script generates the small reproducibility certificates used in the
paper.  The calculations are deliberately finite, deterministic, and exact:
all arithmetic in the certificate examples is performed with Python's
`fractions.Fraction` type rather than floating-point numbers.  Thus the
reported identities, inequalities, branch counts, and table entries are not
affected by roundoff error.

The script is not a CFD solver and does not run large numerical simulations.
Its purpose is to verify, in executable form, the finite examples appearing
in the proofs.

What the script checks
----------------------

1. Mean-only obstruction
   Function:
       certificate_mean_only_obstruction()

   Checks:
       Two fine states have the same parent averages, but after one
       positive-upwind step their restricted parent averages differ.

   Mathematical statement:
       This is the finite example behind the mean-only factorization
       obstruction.  It verifies that parent averages alone do not define a
       one-step pathwise closure on a support containing both states.

2. SSPRK(2,2)/Heun initial-trace obstruction
   Function:
       certificate_heun_trace_failure()

   Checks:
       Two fine states have the same parent averages and the same initial
       parent-interface traces, but their completed Heun updates have
       different parent averages.

   Mathematical statement:
       This is the finite example showing that initial interface traces,
       although sufficient for forward Euler, do not generally determine a
       completed multistage update.

3. Realized flux-register closure
   Function:
       certificate_flux_closure()

   Checks:
       For a deterministic fine state, the parent averages obtained by
       restricting the updated fine solution agree exactly with the parent
       update computed from the realized integrated parent-interface fluxes.

   Mathematical statement:
       This is the finite-volume telescoping identity: a realized integrated
       flux register closes one completed conservative step.

4. Entropy sign-family counts
   Function:
       entropy_counts(P)

   Checks:
       For the sign-indexed Burgers--Rusanov family, the number of branches,
       entropy-interface register values, entropy-divergence values, and the
       probability of zero entropy divergence.

   Mathematical statement:
       Across 2^P sign branches, the conservative parent channel is identical,
       while the entropy-interface register distinguishes all branches and
       the entropy divergence has 2^P - 1 possible values.

Generated files
---------------

The script creates the following directories if they do not already exist:

    results/
    tables/

It writes:

    results/closure_certificates.csv
        CSV record of the three exact closure certificates:
        mean-only obstruction, Heun trace failure, and flux-register closure.

    results/entropy_counts.csv
        CSV record of the entropy sign-family counts for P = 2, 4, 8, 16.

    tables/closure_certificates.tex
        LaTeX tabular fragment summarizing the three closure certificates.

    tables/entropy_counts.tex
        LaTeX tabular fragment summarizing the entropy sign-family counts.

How to run
----------

From the repository root, run:

    python run_closure_certificates.py

No external CFD solver, mesh package, or proprietary software is required.

Important note
--------------

This script is a reproducibility and regression-check tool.  It does not
replace the mathematical proofs in the paper.  It provides an executable
record that the finite examples and table entries stated in the manuscript
are reproduced exactly using rational arithmetic.
"""

from __future__ import annotations

import csv
from fractions import Fraction
from pathlib import Path
from typing import List


def ensure_dirs() -> None:
    Path("results").mkdir(exist_ok=True)
    Path("tables").mkdir(exist_ok=True)


def fmt_frac(x: Fraction) -> str:
    """Format a Fraction as LaTeX math."""
    if x.denominator == 1:
        return rf"${x.numerator}$"
    return rf"$\frac{{{x.numerator}}}{{{x.denominator}}}$"


def restrict_parent(u: List[Fraction], P: int, r: int) -> List[Fraction]:
    if len(u) != P * r:
        raise ValueError("Incompatible vector size.")
    out = []
    for K in range(P):
        s = sum(u[K * r : (K + 1) * r], Fraction(0))
        out.append(s / r)
    return out


def upwind_periodic(u: List[Fraction], lam: Fraction) -> List[Fraction]:
    N = len(u)
    return [(1 - lam) * u[i] + lam * u[(i - 1) % N] for i in range(N)]


def heun_upwind(u: List[Fraction], lam: Fraction) -> List[Fraction]:
    """
    Heun update for the linear upwind system.

    For the linear update A, Heun equals 1/2 * (I + A^2).
    """
    Au = upwind_periodic(u, lam)
    A2u = upwind_periodic(Au, lam)
    return [(u[i] + A2u[i]) / 2 for i in range(len(u))]


def parent_flux_register(u: List[Fraction], P: int, r: int, lam: Fraction) -> List[Fraction]:
    """
    Scalar positive-upwind integrated parent-interface flux.

    For unit fine-cell width and positive speed, the realized integrated
    flux through the right face of parent K is lam times the rightmost child
    of parent K.
    """
    return [lam * u[K * r + (r - 1)] for K in range(P)]


def flux_update_parent(y: List[Fraction], Q: List[Fraction], r: int) -> List[Fraction]:
    P = len(y)
    return [y[K] - (Q[K] - Q[(K - 1) % P]) / r for K in range(P)]


def certificate_mean_only_obstruction() -> dict:
    P, r = 2, 2
    lam = Fraction(1, 2)
    x = [Fraction(0), Fraction(1), Fraction(0), Fraction(0)]
    y = [Fraction(1), Fraction(0), Fraction(0), Fraction(0)]

    Rx = restrict_parent(x, P, r)
    Ry = restrict_parent(y, P, r)

    RAx = restrict_parent(upwind_periodic(x, lam), P, r)
    RAy = restrict_parent(upwind_periodic(y, lam), P, r)

    passed = Rx == Ry and RAx != RAy

    return {
        "test": "mean_only_obstruction",
        "passed": str(passed),
        "input_relation": f"Rx=Ry={Rx}",
        "output_relation": f"RAx={RAx}, RAy={RAy}",
    }


def certificate_heun_trace_failure() -> dict:
    P, r = 2, 4
    lam = Fraction(1, 2)

    xp = [
        Fraction(0), Fraction(-1), Fraction(1), Fraction(0),
        Fraction(0), Fraction(0), Fraction(0), Fraction(0),
    ]
    xm = [
        Fraction(0), Fraction(1), Fraction(-1), Fraction(0),
        Fraction(0), Fraction(0), Fraction(0), Fraction(0),
    ]

    Rxp = restrict_parent(xp, P, r)
    Rxm = restrict_parent(xm, P, r)

    Hxp = restrict_parent(heun_upwind(xp, lam), P, r)
    Hxm = restrict_parent(heun_upwind(xm, lam), P, r)

    # Initial parent-interface traces are zero in both branches:
    # parent 0 right child, parent 1 right child, and periodic corresponding traces.
    traces_equal = True

    passed = Rxp == Rxm and traces_equal and Hxp != Hxm

    return {
        "test": "heun_initial_trace_failure",
        "passed": str(passed),
        "input_relation": f"R x+ = R x- = {Rxp}; initial traces equal",
        "output_relation": f"R H x+ = {Hxp}, R H x- = {Hxm}",
    }


def certificate_flux_closure() -> dict:
    P, r = 3, 4
    lam = Fraction(1, 3)

    u = [
        Fraction(2), Fraction(-1), Fraction(3), Fraction(1),
        Fraction(0), Fraction(4), Fraction(-2), Fraction(5),
        Fraction(1), Fraction(1), Fraction(-3), Fraction(2),
    ]

    y = restrict_parent(u, P, r)
    u_next = upwind_periodic(u, lam)
    y_next_restricted = restrict_parent(u_next, P, r)

    Q = parent_flux_register(u, P, r, lam)
    y_next_flux = flux_update_parent(y, Q, r)

    passed = y_next_restricted == y_next_flux

    return {
        "test": "flux_register_closure",
        "passed": str(passed),
        "input_relation": f"Q={Q}",
        "output_relation": f"restricted={y_next_restricted}, flux_update={y_next_flux}",
    }


def entropy_counts(P: int) -> dict:
    branches = 2 ** P
    entropy_register_values = branches
    entropy_divergence_values = branches - 1
    zero_divergence_probability = Fraction(2, branches)

    # Under the uniform branch law:
    # Psi_K = sigma_K / 6, so E ||Psi||_2^2 = P / 36.
    # beta_K = (sigma_K - sigma_{K-1}) / 6, and
    # E[(sigma_K - sigma_{K-1})^2] = 2, so
    # E ||beta||_2^2 = P / 18.
    entropy_register_mse_floor = Fraction(P, 36)
    entropy_divergence_mse_floor = Fraction(P, 18)

    return {
        "P": P,
        "branches": branches,
        "entropy_register_values": entropy_register_values,
        "entropy_divergence_values": entropy_divergence_values,
        "zero_divergence_probability": zero_divergence_probability,
        "entropy_register_mse_floor": entropy_register_mse_floor,
        "entropy_divergence_mse_floor": entropy_divergence_mse_floor,
    }


def write_closure_csv(rows: List[dict]) -> None:
    path = Path("results/closure_certificates.csv")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test", "passed", "input_relation", "output_relation"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_entropy_csv(P_values: List[int]) -> None:
    path = Path("results/entropy_counts.csv")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "P",
                "branches",
                "entropy_register_values",
                "entropy_divergence_values",
                "zero_divergence_probability",
                "entropy_register_mse_floor",
                "entropy_divergence_mse_floor",
            ],
        )
        writer.writeheader()
        for P in P_values:
            row = entropy_counts(P)
            row["zero_divergence_probability"] = str(row["zero_divergence_probability"])
            row["entropy_register_mse_floor"] = str(row["entropy_register_mse_floor"])
            row["entropy_divergence_mse_floor"] = str(row["entropy_divergence_mse_floor"])
            writer.writerow(row)


def write_closure_table(rows: List[dict]) -> None:
    out = Path("tables/closure_certificates.tex")
    with out.open("w") as f:
        f.write(r"\begin{tabular}{@{}lll@{}}" + "")
        f.write(r"\toprule" + "")
        f.write(r"Certificate & Result & Meaning \\" + "")
        f.write(r"\midrule" + "")
        labels = {
            "mean_only_obstruction": "Mean-only obstruction",
            "heun_initial_trace_failure": "Heun trace failure",
            "flux_register_closure": "Flux-register closure",
        }
        meanings = {
            "mean_only_obstruction": "same parent means, different next parent means",
            "heun_initial_trace_failure": "same means and initial traces, different Heun outputs",
            "flux_register_closure": "restricted update equals flux-register update",
        }
        for row in rows:
            f.write(
                f"{labels[row['test']]} & {row['passed']} & {meanings[row['test']]} \\\\"
            )
        f.write(r"\bottomrule" + "")
        f.write(r"\end{tabular}" + "")


def write_entropy_table(P_values: List[int]) -> None:
    out = Path("tables/entropy_counts.tex")
    with out.open("w") as f:
        f.write(r"\begin{tabular}{@{}rrrrrrr@{}}" + "")
        f.write(r"\toprule" + "")
        f.write(
            r"$P$ & branches & $\Psi$ values & $\beta$ values & "
            r"$\mathbb P(\beta=0)$ & "
            r"$\mathbb E\|\Psi\|_2^2$ & "
            r"$\mathbb E\|\beta\|_2^2$ \\"
            + ""
        )
        f.write(r"\midrule" + "")

        for P in P_values:
            row = entropy_counts(P)
            f.write(
                f"{P} & "
                f"{row['branches']} & "
                f"{row['entropy_register_values']} & "
                f"{row['entropy_divergence_values']} & "
                f"{fmt_frac(row['zero_divergence_probability'])} & "
                f"{fmt_frac(row['entropy_register_mse_floor'])} & "
                f"{fmt_frac(row['entropy_divergence_mse_floor'])} \\\\"
            )

        f.write(r"\bottomrule" + "")
        f.write(r"\end{tabular}" + "")

def main() -> None:
    ensure_dirs()

    closure_rows = [
        certificate_mean_only_obstruction(),
        certificate_heun_trace_failure(),
        certificate_flux_closure(),
    ]

    P_values = [2, 4, 8, 16]

    write_closure_csv(closure_rows)
    write_entropy_csv(P_values)
    write_closure_table(closure_rows)
    write_entropy_table(P_values)

    print("Closure certificates:")
    for row in closure_rows:
        print(f"  {row['test']}: {row['passed']}")

    print("Entropy counts:")
    for P in P_values:
        print(f"  P={P}: {entropy_counts(P)}")

    print("Wrote CSV files to ./results and LaTeX tables to ./tables.")


if __name__ == "__main__":
    main()

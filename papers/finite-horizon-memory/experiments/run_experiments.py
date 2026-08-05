"""
Run all numerical experiments for the paper

    Exact Finite-Horizon Memory, Conditioning, and Dissipative
    Decay in Coarse Upwind Finite-Volume Prediction

This script generates the figures and CSV data files used in the manuscript.
The experiments are deliberately small, deterministic, and designed to
illustrate the exact mathematical results in the paper.

Overview of the experiments
---------------------------

1. Exact rank versus effective rank
   Function:
       experiment_rank_effective()

   Purpose:
       Build the finite-horizon observation matrix

           O_L = [R; R A; R A^2; ...; R A^L]

       for a periodic first-order upwind finite-volume hierarchy.  Compare
       the exact rank formula

           rank(O_L) = P + (P - 1) min(L, r - 1)

       against a tolerance-dependent effective rank computed from the
       singular values of O_L.

   Parameters used in the paper:
       P = 8 parent cells
       r = 6 fine cells per parent
       N = P*r = 48 fine cells
       lambda in {1, 0.5, 0.1, 0.01, 0.001}
       L = 0, ..., r + 3

   Outputs:
       figures/rank_effective.pdf
       figures/singular_values_saturated.pdf
       results/rank_effective.csv


2. Queue conditioning
   Function:
       experiment_queue_conditioning()

   Purpose:
       Build the collar-to-queue matrix T_q, which maps q right-collar
       subcell values inside a parent cell to q future outflow entries.
       The paper proves that this map is triangular and that its smallest
       singular value scales like lambda^q for small lambda.  This experiment
       plots sigma_min(T_q) and the reference scaling lambda^q.

   Parameters used in the paper:
       q = 1, ..., 5
       lambda logarithmically spaced from 1e-4 to 1

   Outputs:
       figures/queue_conditioning.pdf
       results/queue_conditioning.csv

   Note:
       This implementation uses double precision.  The analytical theorem in
       the paper establishes the small-lambda scaling exactly; very small
       singular values in the figure should be interpreted as a visual
       diagnostic rather than as high-relative-accuracy measurements.


3. Periodic step-function flattening
   Function:
       experiment_step_flattening()

   Purpose:
       Evolve a periodic step function with the first-order upwind scheme.
       For 0 < lambda < 1, the scheme is dissipative and the nonconstant
       component decays toward a perturbation floor.  For lambda = 1, the
       scheme is a pure cyclic shift and the step does not flatten.

   Initial condition:
       u_i^0 = 1 for the first half of the periodic grid,
       u_i^0 = 0 for the second half.

   Parameters used in the paper:
       N = 64 fine cells
       nsteps = 60000
       lambda in {0.5, 0.1, 1.0}

   Diagnostics:
       E2(n)    = ||u^n - mean(u^n) 1||_2
       Einf(n)  = ||u^n - mean(u^n) 1||_infty
       TV(n)    = sum_i |u_i^n - u_{i-1}^n|
       mean drift = |mean(u^n) - mean(u^0)|

   Outputs:
       figures/step_flattening.pdf
       results/step_flattening.csv


4. Delayed collision and dissipative decay
   Function:
       experiment_delayed_collision_erasure()

   Purpose:
       Construct two fine-grid states x^0 and y^0 with identical parent
       averages but different subcell arrangements.  The parent averages
       initially agree, then separate after a prescribed delay when the
       hidden subcell perturbation reaches a parent interface.  After the
       difference becomes visible at the parent level, the upwind scheme
       dissipates it.

   Parameters used in the paper:
       P = 8 parent cells
       r = 8 fine cells per parent
       N = P*r = 64 fine cells
       lambda = 0.5
       delay_m = 3
       perturbation amplitude = 0.25
       nsteps = 60000

   Diagnostic:
       D(n) = ||R A^n x^0 - R A^n y^0||_2

   Outputs:
       figures/delayed_collision_erasure.pdf
       results/delayed_collision_erasure.csv


5. One-step flux-closure verification
   Function:
       experiment_flux_closure_check()

   Purpose:
       Verify the telescoping finite-volume identity used in the paper:
       restricting the updated fine solution gives the same parent update as
       applying the realized parent-interface flux difference.

   Parameters used in the paper:
       P = 10 parent cells
       r = 4 fine cells per parent
       N = 40 fine cells
       lambda = 0.37
       100 random trials
       NumPy default random-number generator with seed 12345
       standard normal random fine states

   Output:
       results/flux_closure_check.csv

Generated files
---------------
Figures:
    figures/rank_effective.pdf
    figures/singular_values_saturated.pdf
    figures/queue_conditioning.pdf
    figures/step_flattening.pdf
    figures/delayed_collision_erasure.pdf

CSV files:
    results/rank_effective.csv
    results/queue_conditioning.csv
    results/step_flattening.csv
    results/delayed_collision_erasure.csv
    results/flux_closure_check.csv

How to run
----------
From the repository root, run:

    python run_experiments.py

Then, to generate LaTeX tables from the CSV files, run:

    python make_tables.py

Dependencies
------------
Required:
    Python 3.10 or newer
    NumPy
    Matplotlib

The script uses only double-precision NumPy arrays.  No external CFD solver
is required.

Reproducibility notes
---------------------
The experiments are deterministic.  The only randomized experiment is the
one-step flux-closure check, which uses NumPy's default random-number
generator with seed 12345.

The generated CSV files are intended to be part of the reproducibility record
for the manuscript.  The PDF figures are produced directly from those
experiments.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ensure_dirs() -> None:
    Path("figures").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)


def build_R(P: int, r: int) -> np.ndarray:
    """Parent-average restriction matrix R, shape P by P*r."""
    if P < 2 or r < 2:
        raise ValueError("Require P >= 2 and r >= 2.")
    N = P * r
    R = np.zeros((P, N), dtype=float)
    for K in range(P):
        R[K, K * r : (K + 1) * r] = 1.0 / r
    return R


def build_shift(N: int) -> np.ndarray:
    """Cyclic downstream shift S satisfying (S x)[i] = x[i-1 mod N]."""
    if N < 2:
        raise ValueError("Require N >= 2.")
    I = np.eye(N, dtype=float)
    return np.roll(I, shift=1, axis=0)


def build_A(P: int, r: int, lam: float) -> np.ndarray:
    """First-order periodic upwind matrix."""
    if not (0.0 <= lam <= 1.0):
        raise ValueError("Require 0 <= lambda <= 1.")
    N = P * r
    S = build_shift(N)
    return (1.0 - lam) * np.eye(N) + lam * S


def build_O(P: int, r: int, lam: float, L: int) -> np.ndarray:
    """Observation matrix [R; R A; ...; R A^L]."""
    if L < 0:
        raise ValueError("Require L >= 0.")
    R = build_R(P, r)
    A = build_A(P, r, lam)
    rows = []
    Apow = np.eye(P * r)
    for _ in range(L + 1):
        rows.append(R @ Apow)
        Apow = A @ Apow
    return np.vstack(rows)


def exact_rank_formula(P: int, r: int, L: int) -> int:
    q = min(L, r - 1)
    return P + (P - 1) * q


def effective_rank(M: np.ndarray, factor: float = 100.0) -> tuple[int, float, np.ndarray]:
    """SVD-based effective rank."""
    s = np.linalg.svd(M, compute_uv=False)
    tau = factor * np.finfo(float).eps * max(M.shape) * s[0]
    return int(np.sum(s > tau)), float(tau), s


def experiment_rank_effective() -> None:
    P = 8
    r = 6
    lambdas = [1.0, 0.5, 0.1, 0.01, 0.001]
    L_values = list(range(0, r + 4))

    csv_path = Path("results/rank_effective.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "P",
                "r",
                "lambda",
                "L",
                "exact_rank_formula",
                "effective_rank",
                "tau",
                "largest_singular_value",
                "smallest_singular_value",
            ]
        )

        plt.figure(figsize=(7.5, 4.8))
        exact_values = [exact_rank_formula(P, r, L) for L in L_values]
        plt.plot(L_values, exact_values, "k--", linewidth=2.0, label="exact formula")

        for lam in lambdas:
            eff_values = []
            for L in L_values:
                O = build_O(P, r, lam, L)
                erank, tau, s = effective_rank(O)
                eff_values.append(erank)
                writer.writerow(
                    [
                        P,
                        r,
                        lam,
                        L,
                        exact_rank_formula(P, r, L),
                        erank,
                        tau,
                        float(s[0]),
                        float(s[-1]),
                    ]
                )
            plt.plot(L_values, eff_values, marker="o", label=fr"$\lambda={lam:g}$")

    plt.xlabel("horizon L")
    plt.ylabel("rank")
    #plt.title("Exact rank and effective floating-point rank")
    #plt.grid(True, alpha=0.3)
    plt.legend()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("figures/rank_effective.pdf")
    plt.close()

    # Singular values at the saturated horizon.
    L = r - 1
    plt.figure(figsize=(7.5, 4.8))
    for lam in lambdas:
        O = build_O(P, r, lam, L)
        _, tau, s = effective_rank(O)
        idx = np.arange(1, len(s) + 1)
        plt.semilogy(idx, s, marker="o", label=fr"$\lambda={lam:g}$")
        plt.axhline(tau, color="gray", linestyle=":", linewidth=0.7)
    plt.xlabel("singular-value index")
    plt.ylabel("singular value")
    #plt.title("Singular values of saturated observation matrix")
    #plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("figures/singular_values_saturated.pdf")
    plt.close()


def queue_matrix(lam: float, q: int) -> np.ndarray:
    """Map q right-collar values to q future outflow entries."""
    if q < 1:
        raise ValueError("Require q >= 1.")
    T = np.zeros((q, q), dtype=float)
    for t in range(q):
        for j in range(t + 1):
            T[t, j] = (
                lam
                * math.comb(t, j)
                * (1.0 - lam) ** (t - j)
                * lam**j
            )
    return T


def experiment_queue_conditioning() -> None:
    q_values = [1, 2, 3, 4, 5]
    lambdas = np.logspace(-4, 0, 200)

    csv_path = Path("results/queue_conditioning.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lambda", "q", "sigma_min", "condition_number", "lambda_power_q"])

        plt.figure(figsize=(7.5, 4.8))
        for q in q_values:
            sigma_min_values = []
            lambda_power_values = []
            for lam in lambdas:
                T = queue_matrix(float(lam), q)
                s = np.linalg.svd(T, compute_uv=False)
                sigma_min = float(s[-1])
                cond = float(s[0] / s[-1])
                sigma_min_values.append(sigma_min)
                lambda_power_values.append(float(lam**q))
                writer.writerow([float(lam), q, sigma_min, cond, float(lam**q)])

            plt.loglog(
                lambdas,
                sigma_min_values,
                linewidth=2.0,
                label=fr"$\sigma_{{\min}}(T_{q})$, q={q}",
            )
            plt.loglog(
                lambdas,
                lambda_power_values,
                linestyle=":",
                linewidth=1.0,
                color="gray",
            )

    plt.xlabel(r"$\lambda$")
    plt.ylabel("smallest singular value")
    #plt.title("Conditioning of collar-to-queue map")
    #plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=10)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("figures/queue_conditioning.pdf")
    plt.close()


def upwind_periodic(u: np.ndarray, lam: float) -> np.ndarray:
    """One periodic first-order upwind step."""
    return (1.0 - lam) * u + lam * np.roll(u, 1)


def rho_value(N: int, lam: float) -> float:
    return float(
        math.sqrt(
            1.0
            - 4.0 * lam * (1.0 - lam) * math.sin(math.pi / N) ** 2
        )
    )


def total_variation_periodic(u: np.ndarray) -> float:
    return float(np.sum(np.abs(u - np.roll(u, 1))))


def experiment_step_flattening() -> None:
    N = 64
    nsteps = 60000
    store_every = 20
    lambdas = [0.5, 0.1, 1.0]
    eps = np.finfo(float).eps

    # Fixed visual style for each lambda.  This prevents the dashed bound in
    # the E2 panel from advancing Matplotlib's color cycle and changing the
    # colors relative to the other panels.
    styles = {
        0.5: {"color": "C0", "marker": None, "linestyle": "-"},
        0.1: {"color": "C1", "marker": None, "linestyle": "-"},
        1.0: {"color": "C2", "marker": None, "linestyle": "-"},
    }

    csv_path = Path("results/step_flattening.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "lambda",
                "n",
                "mean",
                "E2",
                "Einf",
                "TV",
                "mean_drift",
                "theory_bound_E2",
            ]
        )

        fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.5), sharex=True)
        ax_e2, ax_einf, ax_tv, ax_mean = axes.ravel()

        for lam in lambdas:
            u = np.zeros(N, dtype=float)
            u[: N // 2] = 1.0
            alpha = float(np.mean(u))
            e20 = float(np.linalg.norm(u - alpha))

            if lam < 1.0:
                rho = rho_value(N, lam)
                eta = 200.0 * eps * math.sqrt(N)
            else:
                rho = 1.0
                eta = 0.0

            ns = []
            e2s = []
            einfs = []
            tvs = []
            mean_drifts = []
            bounds = []

            for n in range(nsteps + 1):
                if n % store_every == 0 or n == nsteps:
                    mean = float(np.mean(u))
                    v = u - mean
                    e2 = float(np.linalg.norm(v))
                    einf = float(np.max(np.abs(u - mean)))
                    tv = total_variation_periodic(u)
                    mean_drift = abs(mean - alpha)

                    if lam < 1.0:
                        bound = (rho**n) * e20 + (1.0 - rho**n) * eta / (1.0 - rho)
                    else:
                        bound = e20

                    ns.append(n)
                    e2s.append(e2)
                    einfs.append(einf)
                    tvs.append(tv)
                    mean_drifts.append(mean_drift)
                    bounds.append(bound)

                    writer.writerow([lam, n, mean, e2, einf, tv, mean_drift, bound])

                if n < nsteps:
                    u = upwind_periodic(u, lam)

            style = styles[lam]
            color = style["color"]
            label = fr"$\lambda={lam:g}$"

            # Actual diagnostics, same color in all four panels.
            ax_e2.semilogy(ns, e2s, color=color, linestyle="-", label=label)
            ax_einf.semilogy(ns, einfs, color=color, linestyle="-", label=label)
            ax_tv.semilogy(ns, tvs, color=color, linestyle="-", label=label)
            ax_mean.semilogy(
                ns,
                np.maximum(mean_drifts, eps),
                color=color,
                linestyle="-",
                label=label,
            )

            # Theoretical E2 bound, same color as the corresponding E2 curve.
            # Use a no-legend label so the legend is not duplicated.
            if lam < 1.0:
                ax_e2.semilogy(
                    ns,
                    bounds,
                    color=color,
                    linestyle="--",
                    alpha=0.8,
                    label="_nolegend_",
                )

    ax_e2.set_title(r"$E_2(n)=\|u^n-\bar{u}^{\,n}\mathbf{1}\|_2$")
    ax_einf.set_title(r"$E_\infty(n)=\|u^n-\bar{u}^{\,n}\mathbf{1}\|_\infty$")
    ax_tv.set_title("periodic total variation")
    ax_mean.set_title("mean drift")

    ax_e2.legend()
    ax_einf.legend()
    ax_tv.legend()
    ax_mean.legend(loc="lower right")

    for ax in axes.ravel():
        ax.set_xlabel("time step n")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig("figures/step_flattening.pdf")
    plt.close(fig)


def restrict_parent(u: np.ndarray, P: int, r: int) -> np.ndarray:
    if u.size != P * r:
        raise ValueError("u has incompatible size.")
    return u.reshape(P, r).mean(axis=1)


def experiment_delayed_collision_erasure() -> None:
    P = 8
    r = 8
    N = P * r
    lam = 0.5
    nsteps = 60000
    delay_m = 3
    ell = r - 1 - delay_m
    delta0 = 0.25

    x = 0.5 * np.ones(N, dtype=float)
    y = 0.5 * np.ones(N, dtype=float)

    # Parent 0 perturbation with zero parent mean.
    x[0] -= delta0
    x[ell] += delta0

    Rx0 = restrict_parent(x, P, r)
    Ry0 = restrict_parent(y, P, r)

    if not np.allclose(Rx0, Ry0, atol=1.0e-15, rtol=0.0):
        raise RuntimeError("Initial parent averages do not match.")

    rho = rho_value(N, lam)
    fine_diff_norm = float(np.linalg.norm(x - y))
    parent_bound_prefactor = fine_diff_norm / math.sqrt(r)

    ns = []
    Ds = []
    bounds = []

    for n in range(nsteps + 1):
        D = float(np.linalg.norm(restrict_parent(x, P, r) - restrict_parent(y, P, r)))
        bound = (rho**n) * parent_bound_prefactor

        ns.append(n)
        Ds.append(D)
        bounds.append(bound)

        if n < nsteps:
            x = upwind_periodic(x, lam)
            y = upwind_periodic(y, lam)

    csv_path = Path("results/delayed_collision_erasure.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "D_parent", "bound_parent"])
        for n, D, bound in zip(ns, Ds, bounds):
            writer.writerow([n, D, bound])

    eps = np.finfo(float).eps

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Short-time panel.
    n_short = 25
    axes[0].plot(ns[: n_short + 1], Ds[: n_short + 1], marker="o")
    axes[0].axvline(delay_m + 1, color="red", linestyle=":", label="first separation")
    axes[0].set_xlabel("time step n")
    axes[0].set_ylabel(r"$D(n)$")
    #axes[0].set_title("short-time delayed separation")
    #axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    ax = plt.gca()
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Long-time panel.  Use NaN for exact zeros so they do not dominate the log scale.
    Ds_log = np.array(Ds, dtype=float)
    Ds_log[Ds_log <= 10.0 * eps] = np.nan

    axes[1].semilogy(ns, Ds_log, label=r"$D(n)$")
    axes[1].semilogy(ns, bounds, "k--", label="exact-arithmetic decay estimate")
    axes[1].axhline(eps, color="gray", linestyle=":", label=r"$\epsilon_{\rm mach}$")
    axes[1].set_xlabel("time step n")
    axes[1].set_ylabel(r"$D(n)$")
    #axes[1].set_title("long-time dissipative erasure")
    #axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    
    fig.tight_layout()
    fig.savefig("figures/delayed_collision_erasure.pdf")
    plt.close(fig)



def experiment_flux_closure_check() -> None:
    rng = np.random.default_rng(12345)
    P = 10
    r = 4
    N = P * r
    lam = 0.37
    ntrials = 100
    errors = []

    for _ in range(ntrials):
        x = rng.standard_normal(N)
        x_next = upwind_periodic(x, lam)

        y = restrict_parent(x, P, r)
        y_next_restricted = restrict_parent(x_next, P, r)

        # Parent right-face flux for scalar positive upwind.
        Phi = np.zeros(P)
        for K in range(P):
            Phi[K] = lam * x[K * r + (r - 1)]

        div = Phi - np.roll(Phi, 1)
        y_next_flux = y - div / r

        errors.append(float(np.max(np.abs(y_next_restricted - y_next_flux))))

    csv_path = Path("results/flux_closure_check.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "max_abs_error"])
        for i, err in enumerate(errors):
            writer.writerow([i, err])

    print("Flux closure check")
    print("  max error:", max(errors))
    print("  mean error:", sum(errors) / len(errors))


def main() -> None:
    ensure_dirs()
    experiment_rank_effective()
    experiment_queue_conditioning()
    experiment_step_flattening()
    experiment_delayed_collision_erasure()
    experiment_flux_closure_check()
    print("Done. Figures are in ./figures and CSV files are in ./results.")


if __name__ == "__main__":
    main()

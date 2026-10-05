"""
Numerical experiments for article_4 (Advances in Computational Mathematics).

Currently implemented
---------------------
E1  Rank ladders across the linear scheme family.
    For every scheme in `schemes.SCHEMES`, every Courant number in the
    sweep, and horizons L = 0, ..., r+3, this experiment compares:

      (a) the effective (SVD, tolerance-based) rank of the observation
          matrix O_L = [R; RA; ...; RA^L], computed exactly as in the
          companion paper (tau = 100 eps max(shape) ||O_L||_2);
      (b) the scheme-independent universal ladder P + (P-1) min(L, r-1);
      (c) the exact Fourier node-counting prediction
          rank O_L = sum_m min(L+1, d_m),
          which differs from (b) only on the spectral-collision set.

    It also writes a collision catalogue: any (scheme, lambda) pair whose
    class symbols contain coincident nodes.

E1b Collision certificate (whole-interval, not sampled).
    For every scheme and every mode pair (k, k') within a residue class,
    the symbol difference delta(lambda) = g_k - g_{k'} is a polynomial in
    lambda with exactly computable coefficients.  All roots of every pair
    polynomial are found; real roots in (0, 1] would be collision Courant
    numbers.  An empty output certifies that the spectral-collision set is
    empty on the entire interval (0, 1] for the given (P, r).

E1c Single-parent (local) observability ranks.
    rho(L) = rank of O_L restricted to zero-mean perturbations of one
    parent cell.  This is the quantity entering the product-local encoder
    lower bound m_K >= rho(L); measured per scheme to identify the closed
    form (one-sided schemes vs. schemes with two active collars).

E2  Saturated singular spectra and effective rank.
    Full singular spectrum of O_{r-1} per scheme and Courant number;
    effective-rank tables (Paper A Table 1 analogue); per-class Vandermonde
    block spectra sigma_min(V_m) and intra-class node gaps
    min |g_k - g_k'|, which localize effective-rank loss to specific
    residue classes; exact first-order node velocities mu_k = dg_k/dlambda
    at lambda=0 (class-node geometry explanation of the E1 findings);
    Gramian-monotonicity check (effective rank vs. horizon L past
    saturation at small lambda).

E3  Queue conditioning.
    sigma_min^+ (smallest nonzero singular value) of the single-parent
    observation map O_L iota_K versus lambda, per scheme and horizon;
    log-log slope fits of the conditioning exponents gamma_sch, compared
    with the prediction gamma = L_q (first horizon exposing q local
    dimensions; = ceil(q/sigma) pre-wraparound; = q for UW1, Paper A).
    By the queue-observation equivalence lemma the exponents equal those
    of the generalized collar-to-queue map T_q.

E4  Dissipative erasure.
    (a) Exact dissipative orders 2s and leading coefficients A(lambda)
        from the autocorrelation moment expansion
        1 - |g|^2 = A(lambda) theta^{2s} + O(theta^{2s+2}) (exact rational
        arithmetic; includes the WENO5L-FE antidissipation A = -lambda^2
        and the SSP-RK3 pairing with A = lambda^4/12 exactly).
    (b) Interior-stability scan max_k |g_k| over a lambda sweep (the only
        violator should be WENO5L with forward Euler).
    (c) Contraction factors rho_N for N = 24..384: verification of
        1 - rho_N = (A/2)(2 pi/N)^{2s} (1 + o(1)) and of the log-log
        slope -2s (the N^{2s} erasure-time law).
    (d) Time-domain step flattening at N = 64: fitted per-step contraction
        vs. rho_N; conservation (mean-drift) check.

E5  Delayed collisions and information half-life.
    Sharp compensated two-cell pairs per scheme (construction of the
    detection-delay corollary) for every admissible depth t0: measured
    first-separation step vs. the proven law t* = t0; peak separation;
    information half-life n_1/2 = ln 2/(-ln rho_N) (predicted; exact for
    geometric decay -- ln 2/(1 - rho_N) is its asymptotic approximation)
    and the measured peak-to-half decay; tail decay rate vs. rho_N.  Headline
    table: hiding time vs. retention factor relative to UW1.

Outputs (written to article_4/results and article_4/figures)
------------------------------------------------------------
    results/rank_ladders_linear.csv
    results/rank_collision_catalogue.csv
    results/collision_certificate.csv
    results/local_observability.csv
    results/singular_values_saturated.csv
    results/effective_rank_saturated.csv
    results/class_blocks_saturated.csv
    results/node_velocities.csv
    results/rank_vs_horizon_smalllambda.csv
    results/queue_conditioning.csv
    results/queue_conditioning_fits.csv
    results/erasure_orders.csv
    results/erasure_stability.csv
    results/erasure_rates.csv
    results/erasure_rate_fits.csv
    results/erasure_flattening.csv
    results/erasure_flattening_fits.csv
    results/delayed_collisions.csv
    results/delayed_collisions_series.csv
    results/nonlinear_sensitivity.csv          (E6s, REVISION_1 m4)
    figures/rank_ladders_linear.pdf
    figures/singular_spectra_saturated.pdf
    figures/queue_conditioning.pdf
    figures/erasure_rates.pdf
    figures/erasure_flattening.pdf
    figures/delayed_collisions.pdf

Interpretation notes
--------------------
The universality theorem concerns exact algebraic rank.  The effective
rank is a floating-point diagnostic: for small lambda it drops below the
exact rank (conditioning loss, cf. companion paper).  Agreement between
(a) and (b) is therefore expected for moderate lambda; discrepancies at
lambda <= 0.01 measure stable observability, not failures of the theorem.

How to run
----------
    python run_experiments.py

Dependencies: Python >= 3.10, NumPy, Matplotlib.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import certificates as cert
import nonlinear as nl
import schemes as sch

BASE = Path(__file__).resolve().parent.parent   # article_4/
RESULTS = BASE / "results"
FIGURES = BASE / "figures"

# Experiment E1 parameters.
P = 8
r = 6
LAMBDAS = [1.0, 0.9, 0.5, 0.1, 0.01, 0.001]

# Compact panel titles for the 4x2 grid figures (E1/E2/E3): the long
# descriptive labels overlap horizontally at the font sizes needed for
# readability after the ~0.42x scaling to text width.
PANEL_TITLES = {
    "uw1": "UW1: first-order upwind",
    "lw": "LW: Lax\u2013Wendroff",
    "bw": "BW: Beam\u2013Warming",
    "fromm": "FR: Fromm",
    "ub3": "UB3: upwind-biased, $p=3$",
    "ub5": "UB5: upwind-biased, $p=5$",
    "weno5l": "WENO5-LIN: linear-weights WENO5 (+FE)",
}

# Panel identifiers (left-to-right, then top-to-bottom).  The multi-panel
# figures label each panel with one of these in a corner and expand the
# scheme names in the caption, so long per-panel titles no longer collide.
PANEL_LETTERS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]


def _panel_label(ax, idx, loc="tl", fontsize=19):
    """Place a corner subpanel label ``(a)``, ``(b)``, ...

    ``loc`` picks the corner that is clear of the data: ``"tl"`` (top-left,
    just inside the top of the y-axis) or ``"tr"`` (top-right, above the
    right end of the x-axis).  A white bounding box keeps it legible where
    a curve passes underneath.
    """
    x, ha = (0.05, "left") if loc == "tl" else (0.95, "right")
    ax.text(x, 0.95, PANEL_LETTERS[idx], transform=ax.transAxes,
            ha=ha, va="top", fontsize=fontsize, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="0.75", alpha=0.9))


def _panel_key_title(ax, idx, name, fontsize=16):
    """Left-aligned ``(a) UW1`` style panel title with the short scheme key.

    Placed above the axes, so it can never collide with the plotted curves
    in any panel, and short enough (the key only) not to overrun into the
    neighbouring panel the way the full descriptive titles did.  The
    caption expands each key to the full scheme name.
    """
    key = PANEL_TITLES[name].split(":", 1)[0]
    ax.set_title(f"{PANEL_LETTERS[idx]} {key}", loc="left", fontsize=fontsize)


L_VALUES = list(range(0, r + 4))


def effective_rank(M: np.ndarray, factor: float = 100.0) -> tuple[int, float, np.ndarray]:
    """SVD-based effective rank with the companion paper's tolerance."""
    s = np.linalg.svd(M, compute_uv=False)
    tau = factor * np.finfo(float).eps * max(M.shape) * s[0]
    return int(np.sum(s > tau)), float(tau), s


def experiment_rank_ladders(r_values: tuple[int, ...] = (6, 8)) -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    csv_path = RESULTS / "rank_ladders_linear.csv"
    collision_path = RESULTS / "rank_collision_catalogue.csv"

    n_schemes = len(sch.SCHEMES)
    ncols = 4
    nrows = -(-n_schemes // ncols)

    mismatches_moderate = []   # effective vs node prediction, lambda >= 0.1
    collisions_found = []

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scheme", "P", "r", "lambda", "L",
            "effective_rank", "tau",
            "universal_formula", "node_prediction",
            "largest_singular_value", "smallest_singular_value",
            "m_L", "m_R",
        ])

        for r_val in r_values:
            L_vals = list(range(0, r_val + 4))
            make_fig = (r_val == r)   # manuscript figure stays on r = 6
            if make_fig:
                fig, axes = plt.subplots(nrows, ncols,
                                         figsize=(4.0 * ncols, 3.2 * nrows),
                                         sharex=True, sharey=True)
                axes = np.atleast_2d(axes)

            for idx, (name, (fn, order, label)) in \
                    enumerate(sch.SCHEMES.items()):
                if make_fig:
                    ax = axes[idx // ncols][idx % ncols]
                    universal = [sch.universal_formula(P, r_val, L)
                                 for L in L_vals]
                    ax.plot(L_vals, universal, "k--", linewidth=2.6,
                            label="universal formula")

                for lam in LAMBDAS:
                    coeffs = fn(lam)
                    mL, mR = sch.reach(coeffs)

                    hits = sch.find_collisions(coeffs, P, r_val)
                    if hits:
                        collisions_found.append((name, r_val, lam, hits))

                    eff_values = []
                    for L in L_vals:
                        O = sch.build_O(coeffs, P, r_val, L)
                        erank, tau, s = effective_rank(O)
                        upred = sch.universal_formula(P, r_val, L)
                        npred = sch.predicted_rank(coeffs, P, r_val, L)
                        eff_values.append(erank)
                        writer.writerow([
                            name, P, r_val, lam, L, erank, tau, upred,
                            npred, float(s[0]), float(s[-1]), mL, mR,
                        ])
                        if lam >= 0.1 and erank != npred:
                            mismatches_moderate.append(
                                (name, r_val, lam, L, erank, npred))

                    if make_fig:
                        ax.plot(L_vals, eff_values, marker="o",
                                markersize=4.5, linewidth=1.6,
                                label=fr"$\lambda={lam:g}$")

                if make_fig:
                    # Panel figures are scaled to text width (~0.42x) in
                    # the manuscript, so sources land at ~7-8 pt.  Short
                    # "(a) UW1" key as a left title (above the curves, so
                    # no collision in any panel); caption expands the key.
                    _panel_key_title(ax, idx, name)
                    ax.tick_params(labelsize=16)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

            if make_fig:
                for ax in axes[-1]:
                    ax.set_xlabel("horizon L", fontsize=18)
                for row in axes:
                    row[0].set_ylabel("rank", fontsize=18)
                # Shared legend in the unused 8th grid slot: in every
                # panel the ladder crosses the lower-left corner and the
                # rank plateaus bound the lower-right strip, so no
                # in-panel placement clears all curves at readable size.
                for idx in range(n_schemes, nrows * ncols):
                    axes[idx // ncols][idx % ncols].axis("off")
                handles_, labels_ = \
                    axes[0][0].get_legend_handles_labels()
                axes[nrows - 1][ncols - 1].legend(
                    handles_, labels_, fontsize=17, loc="center",
                    frameon=False)
                fig.tight_layout()
                fig.savefig(FIGURES / "rank_ladders_linear.pdf")
                plt.close(fig)

    with collision_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scheme", "r", "lambda", "class_m", "j", "j_prime"])
        for name, r_val, lam, hits in collisions_found:
            for (m, j, jp) in hits:
                writer.writerow([name, r_val, lam, m, j, jp])

    # Terminal summary.
    print("E1: rank ladders across the linear scheme family")
    print(f"  schemes: {', '.join(sch.SCHEMES)}")
    print(f"  P={P}, r in {r_values}, lambdas={LAMBDAS}, L=0..r+3")
    if collisions_found:
        print("  spectral collisions detected (exceptional set):")
        for name, r_val, lam, hits in collisions_found:
            print(f"    {name} at r={r_val}, lambda={lam}: {hits}")
    else:
        print("  spectral collisions detected: none "
              "(node prediction == universal formula everywhere)")
    if mismatches_moderate:
        print("  effective-rank mismatches at lambda >= 0.1 "
              "(UNEXPECTED if theorem holds):")
        for row in mismatches_moderate:
            print(f"    scheme={row[0]} r={row[1]} lambda={row[2]} "
                  f"L={row[3]} effective={row[4]} predicted={row[5]}")
    else:
        print("  effective rank == exact node prediction for every scheme "
              "at lambda >= 0.1 (both grids): universality confirmed")
    print(f"  wrote {csv_path.relative_to(BASE)}, "
          f"{collision_path.relative_to(BASE)}, "
          f"figures/rank_ladders_linear.pdf")


def experiment_collision_certificate() -> None:
    """E1b: certify emptiness of the collision set on (0, 1] per operator.

    Covers all eight fully discrete operators of the study: the seven
    spatial schemes advanced with forward Euler plus the composed
    WENO5L + SSP-RK3 one-step operator (REVISION_2 M2 -- the composition
    needs its own certificate, since the RK3 stability polynomial is not
    injective and FE-level emptiness does not transfer analytically).

    Two independent layers (REVISION_1 M1):
      (i)  exact rational-arithmetic certificate (certificates.py):
           Bernstein positivity of |delta_tilde|^2 on [0, 1], computed
           in the cyclotomic group ring Q[Z_N] with certified rational
           cosine enclosures -- no floating-point number enters any
           accepted step;
      (ii) floating-point root isolation on the same exact-coefficient
           pair polynomials (independent cross-check, tolerance 1e-8).
    """
    RESULTS.mkdir(exist_ok=True)
    csv_path = RESULTS / "collision_certificate.csv"
    exact_path = RESULTS / "collision_certificate_exact.csv"

    grid_pairs = [(8, 6), (8, 8)]
    operators = (*sch.SCHEMES, "weno5l_rk3")

    # ---- (i) exact certificate ----------------------------------------
    exact_rows = []
    with exact_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scheme", "P", "r", "n_pairs", "certified",
                         "certified_min_q", "max_bernstein_cells"])
        for (Pc, rc) in grid_pairs:
            tab = cert.cos_table(Pc * rc)
            for name in operators:
                res = cert.collision_certificate_exact(name, Pc, rc, tab)
                exact_rows.append(res)
                writer.writerow([name, Pc, rc, res["n_pairs"],
                                 res["certified"], res["min_bound"],
                                 res["max_cells"]])

    # ---- (ii) floating-point cross-check ------------------------------
    findings_all: list[tuple] = []
    n_pairs_total = 0
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scheme", "P", "r", "class_m", "k", "k_prime",
                         "collision_lambda_or_flag"])
        for (Pc, rc) in grid_pairs:
            n_pairs_total += (Pc - 1) * rc * (rc - 1) // 2 * len(operators)
            for name in operators:
                findings = sch.collision_certificate(name, Pc, rc)
                for (m, k, kp, what) in findings:
                    findings_all.append((name, Pc, rc, m, k, kp, what))
                    writer.writerow([name, Pc, rc, m, k, kp, what])

    print("E1b: collision certificate")
    print(f"  grid pairs (P, r): {grid_pairs}; "
          f"pair polynomials examined: {n_pairs_total}")
    all_cert = all(r["certified"] for r in exact_rows)
    gmin = min(r["min_bound"] for r in exact_rows)
    worst = min(exact_rows, key=lambda r: r["min_bound"])
    print(f"  exact certificate: all pairs certified = {all_cert}; "
          f"uniform bound min |delta_tilde|^2 >= {gmin:.4e} on [0,1] "
          f"(worst pair family: {worst['scheme']}, r={worst['r']}); "
          f"max Bernstein cells = "
          f"{max(r['max_cells'] for r in exact_rows)}")
    if findings_all:
        print("  FLOAT CROSS-CHECK found candidate roots in (0, 1]:")
        for row in findings_all:
            print(f"    {row}")
    else:
        print("  float cross-check: no real roots in (0, 1] for any pair "
              "polynomial")
    print(f"  wrote {csv_path.relative_to(BASE)}, "
          f"{exact_path.relative_to(BASE)}")


def experiment_local_rank() -> None:
    """E1c: single-parent zero-mean observability rank rho(L) per scheme."""
    RESULTS.mkdir(exist_ok=True)
    csv_path = RESULTS / "local_observability.csv"

    lams = [0.9, 0.5]
    r_values = [6, 8]

    # The seven spatial schemes plus the composed WENO5L+SSP-RK3 one-step
    # operator (itself linear circulant).  Its reach (9,6) spans several
    # parents in a single step, exercising the multi-parent regime of the
    # wrap-around bound where rho can grow faster than sigma <= 2 per
    # step (REVISION_1 M5).
    members = [(name, fn) for name, (fn, _order, _label)
               in sch.SCHEMES.items()]
    members.append(("weno5l_rk3", sch.weno5l_ssprk3))
    fns = dict(members)

    tables: dict[tuple, dict[str, list[int]]] = {}
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scheme", "P", "r", "lambda", "L",
                         "local_rank", "m_L", "m_R"])
        for rc in r_values:
            Ls = list(range(0, rc + 4))
            for lam in lams:
                tab: dict[str, list[int]] = {}
                for name, fn in members:
                    coeffs = fn(lam)
                    mL, mR = sch.reach(coeffs)
                    ranks = []
                    for L in Ls:
                        M = sch.local_observability_matrix(coeffs, P, rc, L)
                        erank, _tau, _s = effective_rank(M)
                        ranks.append(erank)
                        writer.writerow([name, P, rc, lam, L, erank, mL, mR])
                    tab[name] = ranks
                tables[(rc, lam)] = tab

    print("E1c: single-parent local observability rank rho(L)")
    for rc in r_values:
        Ls = list(range(0, rc + 4))
        print(f"  P={P}, r={rc}, lambda=0.5   (columns: L = "
              f"{Ls[0]}..{Ls[-1]})")
        for name, ranks in tables[(rc, 0.5)].items():
            mL, mR = sch.reach(fns[name](0.5))
            sigma = (1 if mL > 0 else 0) + (1 if mR > 0 else 0)
            closed = [sch.rho_formula(mL, mR, rc, L) for L in Ls]
            match = "matches closed form" if ranks == closed else \
                    "DOES NOT match closed form"
            print(f"    {name:12s} (m_L={mL}, m_R={mR}, sigma={sigma}): "
                  f"{ranks}  <- {match}")
        agree = all(tables[(rc, 0.5)][n] == tables[(rc, 0.9)][n]
                    for n in fns)
        print(f"  lambda=0.9 table {'identical' if agree else 'DIFFERS'}"
              f" (r={rc})")
    print(f"  wrote {csv_path.relative_to(BASE)}")


def _class_basis(P: int, r: int, m: int) -> np.ndarray:
    """Orthonormal (complex) basis of the class-m fine Fourier modes."""
    N = P * r
    n = np.arange(N)
    cols = [np.exp(2j * np.pi * (m + j * P) * n / N) / np.sqrt(N)
            for j in range(r)]
    return np.stack(cols, axis=1)


def experiment_saturated_spectra() -> None:
    """E2: saturated spectra, per-class conditioning, node geometry."""
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    L_sat = r - 1
    lams_e2 = [1.0, 0.9, 0.5, 0.1, 0.05, 0.01, 0.001]
    exact_sat = sch.universal_formula(P, r, L_sat)

    spectra_path = RESULTS / "singular_values_saturated.csv"
    table_path = RESULTS / "effective_rank_saturated.csv"
    class_path = RESULTS / "class_blocks_saturated.csv"
    mu_path = RESULTS / "node_velocities.csv"
    gram_path = RESULTS / "rank_vs_horizon_smalllambda.csv"

    # --- full spectra + effective-rank table + class blocks ---
    eff_table: dict[str, list[int]] = {}
    worst_class: dict[tuple, tuple] = {}
    spectra_store: dict[tuple, np.ndarray] = {}
    with spectra_path.open("w", newline="") as fs, \
            table_path.open("w", newline="") as ft, \
            class_path.open("w", newline="") as fc:
        ws = csv.writer(fs)
        wt = csv.writer(ft)
        wc = csv.writer(fc)
        ws.writerow(["scheme", "lambda", "index", "singular_value", "tau"])
        wt.writerow(["scheme", "lambda", "L", "exact_rank",
                     "effective_rank", "tau", "sigma_max", "sigma_min"])
        wc.writerow(["scheme", "lambda", "class_m", "sigma_min_block",
                     "sigma_max_block", "min_node_gap"])
        for name, (fn, _order, _label) in sch.SCHEMES.items():
            eff_row = []
            for lam in lams_e2:
                coeffs = fn(lam)
                O = sch.build_O(coeffs, P, r, L_sat)
                erank, tau, s = effective_rank(O)
                eff_row.append(erank)
                spectra_store[(name, lam)] = s
                for i, sv in enumerate(s):
                    ws.writerow([name, lam, i, float(sv), tau])
                wt.writerow([name, lam, L_sat, exact_sat, erank, tau,
                             float(s[0]), float(s[-1])])
                # Per-class Vandermonde blocks (images of distinct classes
                # are orthogonal, so the union of block spectra is the full
                # spectrum of O).
                worst = None
                for m in range(1, P):
                    U = _class_basis(P, r, m)
                    sb = np.linalg.svd(O @ U, compute_uv=False)
                    gs = [sch.symbol(coeffs, m + j * P, P * r)
                          for j in range(r)]
                    gap = min(abs(gs[a] - gs[b])
                              for a in range(r) for b in range(a + 1, r))
                    wc.writerow([name, lam, m, float(sb[-1]),
                                 float(sb[0]), gap])
                    if worst is None or sb[-1] < worst[1]:
                        worst = (m, float(sb[-1]), gap)
                worst_class[(name, lam)] = worst
            eff_table[name] = eff_row

    # --- exact first-order node velocities ---
    mu_report: dict[str, tuple] = {}
    with mu_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "k", "Re_mu", "Im_mu", "class_m"])
        for name in sch.SCHEMES:
            mu = sch.node_velocities(name, P * r)
            for k in range(P * r):
                w.writerow([name, k, float(mu[k].real), float(mu[k].imag),
                            k % P])
            # min intra-class first-order separation (classes m != 0)
            min_gap, min_info = None, None
            for m in range(1, P):
                ks = [m + j * P for j in range(r)]
                for a in range(r):
                    for b in range(a + 1, r):
                        d = abs(mu[ks[a]] - mu[ks[b]])
                        if min_gap is None or d < min_gap:
                            min_gap, min_info = d, (m, ks[a], ks[b])
            mu_report[name] = (float(np.max(np.abs(mu.real))),
                               min_gap, min_info)

    # --- Gramian monotonicity: effective rank vs horizon past saturation ---
    gram_L = [L_sat, L_sat + 2, L_sat + 4, L_sat + 7]
    gram_rows: dict[tuple, list[int]] = {}
    with gram_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "lambda", "L", "effective_rank", "exact_rank"])
        for name, (fn, _order, _label) in sch.SCHEMES.items():
            for lam in (0.01, 0.001):
                coeffs = fn(lam)
                row = []
                for L in gram_L:
                    O = sch.build_O(coeffs, P, r, L)
                    erank, _tau, _s = effective_rank(O)
                    row.append(erank)
                    w.writerow([name, lam, L, erank,
                                sch.universal_formula(P, r, L)])
                gram_rows[(name, lam)] = row

    # --- figure: saturated spectra ---
    n_schemes = len(sch.SCHEMES)
    ncols = 4
    nrows = -(-n_schemes // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.0 * ncols, 3.2 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for idx, (name, (_fn, _order, label)) in enumerate(sch.SCHEMES.items()):
        ax = axes[idx // ncols][idx % ncols]
        for lam in lams_e2:
            s = spectra_store[(name, lam)]
            ax.semilogy(np.arange(1, len(s) + 1), s, linewidth=1.6,
                        label=fr"$\lambda={lam:g}$")
        ax.axvline(exact_sat, color="k", linestyle=":", linewidth=1.2)
        # Short "(a) UW1" key as a left title (above the curves, collision
        # free in every panel); the caption expands the key.
        _panel_key_title(ax, idx, name)
        ax.tick_params(labelsize=16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("singular value index", fontsize=18)
    for row_ in axes:
        row_[0].set_ylabel(r"$\sigma_i$", fontsize=18)
    # Shared legend in the unused 8th grid slot: the lambda=0.001
    # staircase descends through the lower-left corner of every panel,
    # so no in-panel placement clears all curves at readable size.
    for idx in range(n_schemes, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    handles_, labels_ = axes[0][0].get_legend_handles_labels()
    axes[nrows - 1][ncols - 1].legend(handles_, labels_, fontsize=17,
                                      loc="center", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "singular_spectra_saturated.pdf")
    plt.close(fig)

    # --- terminal summary ---
    print("E2: saturated spectra, effective rank, class-node geometry")
    print(f"  P={P}, r={r}, L_sat={L_sat}, exact saturated rank {exact_sat}")
    header = "  " + "scheme".ljust(9) + "".join(
        f"{lam:>8g}" for lam in lams_e2)
    print("  effective rank at saturation vs lambda:")
    print(header)
    for name, row in eff_table.items():
        print("  " + name.ljust(9) + "".join(f"{v:>8d}" for v in row))
    print("  worst residue class (sigma_min of class block) at lambda=0.01:")
    for name in sch.SCHEMES:
        m, smin, gap = worst_class[(name, 0.01)]
        print(f"    {name:8s}: class m={m}, sigma_min_block={smin:.3e}, "
              f"min node gap={gap:.3e}")
    print("  first-order node velocities mu_k (exact):")
    for name, (max_re, min_gap, info) in mu_report.items():
        flag = " <- purely imaginary (degenerate geometry)" \
            if max_re < 1e-14 else ""
        print(f"    {name:8s}: max|Re mu|={max_re:.3e}, "
              f"min intra-class |mu_j-mu_j'|={min_gap:.3e} "
              f"(class m={info[0]}, k={info[1]},{info[2]}){flag}")
    print("  Gramian monotonicity (effective rank at L = "
          f"{gram_L}, lambda=0.001):")
    for name in sch.SCHEMES:
        print(f"    {name:8s}: {gram_rows[(name, 0.001)]}")
    print(f"  wrote {spectra_path.name}, {table_path.name}, "
          f"{class_path.name}, {mu_path.name}, {gram_path.name}, "
          f"figures/singular_spectra_saturated.pdf")


def experiment_queue_conditioning() -> None:
    """E3: sigma_min^+ of the local observation map vs lambda; exponents."""
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    lam_grid = np.logspace(-3, np.log10(0.5), 15)
    fit_mask_max = 0.05          # fit exponents on lambda <= 0.05
    L_values = list(range(1, r))  # 1..r-1

    csv_path = RESULTS / "queue_conditioning.csv"
    fits_path = RESULTS / "queue_conditioning_fits.csv"

    data: dict[tuple, list[tuple[float, float]]] = {}
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "L", "q", "lambda", "sigma_min_plus"])
        for name, (fn, _order, _label) in sch.SCHEMES.items():
            mL, mR = sch.reach(fn(0.5))
            for L in L_values:
                q = sch.rho_formula(mL, mR, r, L)
                for lam in lam_grid:
                    M = sch.local_observability_matrix(fn(lam), P, r, L)
                    s = np.linalg.svd(M, compute_uv=False)
                    smin_plus = float(s[q - 1])
                    w.writerow([name, L, q, float(lam), smin_plus])
                    data.setdefault((name, L), []).append(
                        (float(lam), smin_plus))

    # --- slope fits and comparison with the prediction gamma = L_q ---
    fit_rows = []
    with fits_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "L", "q", "gamma_fit", "gamma_predicted",
                    "abs_error"])
        for name, (fn, _order, _label) in sch.SCHEMES.items():
            mL, mR = sch.reach(fn(0.5))
            for L in L_values:
                q = sch.rho_formula(mL, mR, r, L)
                pts = [(lam, sv) for lam, sv in data[(name, L)]
                       if lam <= fit_mask_max and sv > 0.0]
                x = np.log([p[0] for p in pts])
                y = np.log([p[1] for p in pts])
                slope = float(np.polyfit(x, y, 1)[0])
                # Predicted exponent: first horizon exposing q dimensions.
                Lq = next(Lp for Lp in range(1, r + 4)
                          if sch.rho_formula(mL, mR, r, Lp) >= q)
                w.writerow([name, L, q, slope, Lq, abs(slope - Lq)])
                fit_rows.append((name, L, q, slope, Lq))

    # --- figure ---
    n_schemes = len(sch.SCHEMES)
    ncols = 4
    nrows = -(-n_schemes // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.0 * ncols, 3.2 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for idx, (name, (_fn, _order, label)) in enumerate(sch.SCHEMES.items()):
        ax = axes[idx // ncols][idx % ncols]
        for L in L_values:
            pts = data[(name, L)]
            ax.loglog([p[0] for p in pts], [p[1] for p in pts],
                      marker="o", markersize=4, linewidth=1.6,
                      label=fr"$L={L}$")
        _panel_key_title(ax, idx, name)
        ax.tick_params(labelsize=16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\lambda$", fontsize=18)
    for row_ in axes:
        row_[0].set_ylabel(r"$\sigma^{+}_{\min}$", fontsize=18)
    # Shared legend in the unused 8th grid slot: the conditioning fans
    # fill the panels diagonally, leaving no clean in-panel corner.
    for idx in range(n_schemes, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    handles_, labels_ = axes[0][0].get_legend_handles_labels()
    axes[nrows - 1][ncols - 1].legend(handles_, labels_, fontsize=17,
                                      loc="center", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "queue_conditioning.pdf")
    plt.close(fig)

    # --- terminal summary ---
    print("E3: queue conditioning (local map), exponent fits")
    print(f"  lambda grid: {lam_grid[0]:.3g}..{lam_grid[-1]:.3g} "
          f"(fit on lambda <= {fit_mask_max})")
    print("  scheme    L  q   gamma_fit   gamma_pred")
    max_err = 0.0
    for name, L, q, slope, Lq in fit_rows:
        max_err = max(max_err, abs(slope - Lq))
        print(f"  {name:8s} {L:2d} {q:2d}   {slope:9.3f}   {Lq:5d}")
    print(f"  max |gamma_fit - gamma_pred| = {max_err:.3f}")
    print(f"  wrote {csv_path.name}, {fits_path.name}, "
          f"figures/queue_conditioning.pdf")


def _eval_poly(p, lam: float) -> float:
    """Evaluate an exact (Fraction-coefficient) polynomial at a float."""
    return float(sum(float(a) * lam ** i for i, a in enumerate(p)))


def _erasure_family() -> dict[str, tuple]:
    """Schemes entering the erasure/collision experiments (E4d, E5).

    The registry with weno5l replaced by its SSP-RK3 pairing: the
    forward-Euler pairing is antidissipative (1 - |g|^2 = -lam^2 th^2 + ...),
    hence linearly unstable, and cannot be evolved.
    """
    fam = {}
    for name, (fn, order, label) in sch.SCHEMES.items():
        if name == "weno5l":
            continue
        fam[name] = (fn, label)
    fam["weno5l_rk3"] = (sch.weno5l_ssprk3, "WENO5-LIN+SSP-RK3")
    return fam


def experiment_erasure() -> None:
    """E4: exact dissipative orders, contraction factors, erasure rates."""
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    all_polys = {name: sch.coeff_polys(name) for name in sch.SCHEMES}
    all_polys["weno5l_rk3"] = sch.weno5l_rk3_polys()

    # ---- (a) exact orders and leading coefficients --------------------
    orders_path = RESULTS / "erasure_orders.csv"
    orders: dict[str, tuple[int, list]] = {}
    with orders_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "two_s", "A_exact", "A_at_0.5", "A_at_0.9",
                    "nondissipative_lambdas_in_(0,1]"])
        for name, polys in all_polys.items():
            two_s, A, _ = sch.dissipation_expansion(polys)
            orders[name] = (two_s, A)
            coeffs = [float(a) for a in A]
            roots = np.roots(coeffs[::-1]) if any(coeffs) else np.array([])
            nondiss = sorted(float(z.real) for z in np.atleast_1d(roots)
                             if abs(z.imag) < 1e-9
                             and 1e-9 < z.real <= 1 + 1e-9)
            w.writerow([name, two_s, sch.poly_to_string(A),
                        _eval_poly(A, 0.5), _eval_poly(A, 0.9),
                        ";".join(f"{x:.12g}" for x in nondiss)])

    print("E4a: exact dissipative orders 2s and leading coefficients A(lam)")
    print("     (1 - |g|^2 = A(lam) theta^{2s} + O(theta^{2s+2}), exact)")
    for name, (two_s, A) in orders.items():
        print(f"  {name:12s} 2s={two_s}  A = {sch.poly_to_string(A)}")

    # ---- (b) interior-stability scan ----------------------------------
    stab_path = RESULTS / "erasure_stability.csv"
    lam_scan = [round(0.05 * i, 2) for i in range(1, 21)]
    violators = []
    with stab_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "lambda", "max_abs_g"])
        for name in all_polys:
            fn = (sch.weno5l_ssprk3 if name == "weno5l_rk3"
                  else sch.SCHEMES[name][0])
            for lam in lam_scan:
                mx = sch.contraction_factor(fn(lam), 2048)
                w.writerow([name, lam, mx])
                if mx > 1 + 1e-12:
                    violators.append((name, lam, mx))
    unstable_schemes = sorted({v[0] for v in violators})
    print(f"E4b: stability scan (max_k |g_k|, N=2048, lambda=0.05..1.0): "
          f"violations only for {unstable_schemes}")

    # ---- (c) contraction-factor sweep and N^{-2s} law -----------------
    rates_path = RESULTS / "erasure_rates.csv"
    fits_path = RESULTS / "erasure_rate_fits.csv"
    Ns = [24, 48, 96, 192, 384]
    lam_rates = [0.5, 0.9]
    rate_data: dict[tuple, list[tuple[int, float, float]]] = {}
    with rates_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "lambda", "N", "rho_N", "one_minus_rho",
                    "predicted_A_half_theta1_2s", "ratio"])
        for name in all_polys:
            fn = (sch.weno5l_ssprk3 if name == "weno5l_rk3"
                  else sch.SCHEMES[name][0])
            two_s, A = orders[name]
            for lam in lam_rates:
                Aval = _eval_poly(A, lam)
                for N in Ns:
                    rho = sch.contraction_factor(fn(lam), N)
                    omr = 1.0 - rho
                    pred = 0.5 * Aval * (2 * np.pi / N) ** two_s
                    ratio = omr / pred if pred != 0 else np.nan
                    w.writerow([name, lam, N, rho, omr, pred, ratio])
                    rate_data.setdefault((name, lam), []).append(
                        (N, omr, pred))

    fit_summary = []
    with fits_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "lambda", "slope_fit", "expected_minus_2s",
                    "ratio_at_largest_N"])
        for (name, lam), rows in rate_data.items():
            two_s, _A = orders[name]
            x = np.log([row[0] for row in rows[-3:]])
            y = np.log([abs(row[1]) for row in rows[-3:]])
            slope = float(np.polyfit(x, y, 1)[0])
            ratio = rows[-1][1] / rows[-1][2]
            w.writerow([name, lam, slope, -two_s, ratio])
            fit_summary.append((name, lam, slope, -two_s, ratio))

    print("E4c: 1 - rho_N vs N (fit on last 3 N), prediction (A/2) th1^{2s}")
    print("  scheme        lam   slope_fit  -2s   ratio(N=384)")
    for name, lam, slope, m2s, ratio in fit_summary:
        print(f"  {name:12s} {lam:4.2f}  {slope:9.3f}  {m2s:4d}   {ratio:8.5f}")

    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

    # Manuscript display names for the legend (CSV keys stay unchanged):
    # short scheme keys as in Figs. 1-3, with the two WENO5-LIN pairings
    # spelled out (REVISION_2 m4 display convention).
    disp = {name: PANEL_TITLES[name].split(":", 1)[0] for name in sch.SCHEMES}
    disp["weno5l"] = "WENO5-LIN+FE"
    disp["weno5l_rk3"] = "WENO5-LIN+SSP-RK3"

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, lam in zip(axes, lam_rates):
        for name in all_polys:
            rows = rate_data[(name, lam)]
            ax.loglog([row[0] for row in rows],
                      [abs(row[1]) for row in rows],
                      marker="o", markersize=3, linewidth=1.0,
                      label=disp[name])
        ax.set_title(fr"$\lambda={lam}$", fontsize=10)
        ax.set_xlabel(r"$N$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Label the x-axis only at the actual grid sizes.  The default
        # log-scale minor labels (2-9 x 10^k) collide at this figure
        # width; show plain integers at the sampled N instead.
        ax.xaxis.set_major_locator(FixedLocator(Ns))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _pos: f"{int(round(v))}"))
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel(r"$|1-\rho_N|$")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES / "erasure_rates.pdf")
    plt.close(fig)

    # ---- (d) time-domain step flattening ------------------------------
    N_flat, lam_flat, n_steps, rec_every = 64, 0.5, 200_000, 20
    n_mode_steps = 5_000
    flat_path = RESULTS / "erasure_flattening.csv"
    flat_fits_path = RESULTS / "erasure_flattening_fits.csv"
    family = _erasure_family()
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fit_rows = []
    for name, (fn, _label) in family.items():
        A = sch.circulant_matrix(fn(lam_flat), N_flat)

        # Step-function flattening (figure + CSV).  The tail slope of this
        # curve is a mode mixture and is pre-asymptotic for 2s >= 6 at this
        # horizon; the rho_N verification below therefore uses the slowest
        # mode directly.
        u0 = np.zeros(N_flat)
        u0[: N_flat // 2] = 1.0
        v = u0 - u0.mean()
        e0 = np.linalg.norm(v)
        steps, e2 = [0], [1.0]
        drift = 0.0
        for t in range(1, n_steps + 1):
            v = A @ v
            drift = max(drift, abs(v.sum()) / N_flat)
            if t % rec_every == 0:
                steps.append(t)
                e2.append(np.linalg.norm(v) / e0)
        steps_a, e2_a = np.asarray(steps), np.asarray(e2)
        series[name] = (steps_a, e2_a)

        # Slowest-mode decay: v0 = cos(2 pi i / N) evolves in the k = +-1
        # eigenplane, so its per-step contraction equals |g_1| = rho_N
        # whenever the maximum of |g_k| is attained at k = +-1.
        v = np.cos(2 * np.pi * np.arange(N_flat) / N_flat)
        m0 = np.linalg.norm(v)
        tgrid = np.arange(1, n_mode_steps + 1)
        vals = np.empty(n_mode_steps)
        for t in range(n_mode_steps):
            v = A @ v
            vals[t] = np.linalg.norm(v) / m0
        slope = float(np.polyfit(tgrid, np.log(vals), 1)[0])
        rho_fit = float(np.exp(slope))
        rho_N = sch.contraction_factor(fn(lam_flat), N_flat)
        fit_rows.append((name, rho_N, rho_fit, abs(rho_fit - rho_N), drift))

    with flat_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "step", "E2_over_E2_0"])
        for name, (steps_a, e2_a) in series.items():
            for t, e in zip(steps_a[::10], e2_a[::10]):
                w.writerow([name, int(t), float(e)])
    with flat_fits_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "rho_N", "rho_fit_slowest_mode", "abs_error",
                    "max_mean_drift"])
        for row in fit_rows:
            w.writerow(row)

    print(f"E4d: step flattening (N={N_flat}, lambda={lam_flat}, "
          f"{n_steps} steps); slowest-mode contraction fit vs rho_N")
    print("  scheme        rho_N            rho_fit          |diff|      drift")
    for name, rho_N, rho_fit, err, drift in fit_rows:
        print(f"  {name:12s} {rho_N:.12f}  {rho_fit:.12f}  {err:.2e}  "
              f"{drift:.2e}")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, (steps_a, e2_a) in series.items():
        ax.semilogy(steps_a, np.maximum(e2_a, 1e-30), linewidth=1.0,
                    label=family[name][1])
    ax.set_xlabel("step $n$")
    ax.set_ylabel(r"$E_2(n)/E_2(0)$")
    # Clip the axis at 1e-1: UW1 alone plunges to ~1e-17, and letting it set
    # the scale flattens the high-order curves (0.58--0.91) into one band.
    # UW1's full descent (geometric at rate rho_N, below 1e-17 by n ~ 3.2e4)
    # is reported in the figure caption instead.
    ax.set_ylim(1e-1, 1.2)
    ax.legend(fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "erasure_flattening.pdf")
    plt.close(fig)

    print(f"  wrote {orders_path.name}, {stab_path.name}, {rates_path.name}, "
          f"{fits_path.name}, {flat_path.name}, {flat_fits_path.name}, "
          f"figures/erasure_rates.pdf, figures/erasure_flattening.pdf")


def experiment_stability_certificate() -> None:
    """E4e: exact interior-stability certificate for WENO5L+SSP-RK3.

    Certifies, in exact rational arithmetic (REVISION_1 M3), the
    decomposition Q(x, lam) = lam [ (1-x) S(x, lam) + lam^3/3 ] of the
    pairing's Chebyshev quotient together with the two tensor-Bernstein
    positivity certificates

        S > 0      on [-1, 1] x [0, 1/2],
        Q/lam > 0  on [-1, 1] x [1/2, 1],

    which give Q > 0 on [-1, 1] x (0, 1], i.e. |g(theta)| < 1 for all
    theta in (0, 2*pi) at every fixed lam in (0, 1].  The dense
    frequency-grid scan of E4b remains as an independent cross-check.
    """
    RESULTS.mkdir(exist_ok=True)
    res = cert.stability_certificate()
    csv_path = RESULTS / "stability_certificate.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["object", "x_lo", "x_hi", "lam_lo", "lam_hi",
                    "certified", "certified_lower_bound",
                    "bernstein_cells"])
        w.writerow(["S", -1, 1, 0, 0.5, res["S_positive_on_low_lambda"],
                    res["S_lower_bound"], res["S_cells"]])
        w.writerow(["Q_over_lambda", -1, 1, 0.5, 1,
                    res["Qt_positive_on_high_lambda"],
                    res["Qt_lower_bound"], res["Qt_cells"]])

    print("E4e: exact interior-stability certificate (WENO5L+SSP-RK3)")
    print("  decomposition Q = lam[(1-x) S + lam^3/3] verified exactly; "
          "S(x,0) = 4/15")
    print(f"  S >= {res['S_lower_bound']:.6f} on [-1,1]x[0,1/2] "
          f"({res['S_cells']} Bernstein cell(s)); "
          f"Q/lam >= {res['Qt_lower_bound']:.6f} on [-1,1]x[1/2,1] "
          f"({res['Qt_cells']} cell(s))")
    print(f"  certified: {res['certified']}  ->  |g| < 1 on (0, 2pi) "
          f"for every lam in (0, 1]")
    print(f"  wrote {csv_path.relative_to(BASE)}")


def experiment_delayed_collisions() -> None:
    """E5: detection delay of compensated pairs; information half-life."""
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    lam, n_steps, zeta = 0.5, 300_000, 1e-12
    N = P * r
    R = sch.build_R(P, r)
    family = _erasure_family()

    rows = []
    deepest_series: dict[str, tuple[np.ndarray, str]] = {}
    series_path = RESULTS / "delayed_collisions_series.csv"
    with series_path.open("w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["scheme", "t0", "t", "coarse_separation"])
        for name, (fn, label) in family.items():
            coeffs = fn(lam)
            mL, mR = sch.reach(coeffs)
            wreach = mL + mR
            A = sch.circulant_matrix(coeffs, N)
            rho_N = sch.contraction_factor(coeffs, N)
            # Exact half-life of geometric decay rho^n; ln2/(1 - rho) is
            # the asymptotic approximation as rho -> 1 (REVISION_1 M9).
            n_half_pred = np.log(2.0) / (-np.log(rho_N))
            t0_max = (r - 1) // wreach

            pairs = []
            for t0 in range(1, t0_max + 1):
                j2, j1 = r - t0 * mL, t0 * mR
                pairs.append((t0, j1, j2, t0))
            if not pairs:
                # Reach exceeds r-1: no hiding is possible; deepest pair is
                # the extreme two-cell perturbation, detected at step 1.
                pairs.append((0, 0, r - 1, 1))

            for t0, j1, j2, t_pred in pairs:
                v = np.zeros(N)
                v[j2] += 1.0
                v[j1] -= 1.0
                sep = np.empty(n_steps + 1)
                sep[0] = np.linalg.norm(R @ v)
                for t in range(1, n_steps + 1):
                    v = A @ v
                    sep[t] = np.linalg.norm(R @ v)
                t_detect = int(np.argmax(sep > zeta))
                t_peak = int(np.argmax(sep))
                peak = float(sep[t_peak])
                # Early-time relaxation (geometric spreading, not erasure).
                after = np.nonzero(sep[t_peak:] <= 0.5 * peak)[0]
                peak_relax = int(after[0]) if len(after) else None
                # Asymptotic tail fit on the last half of the available
                # data.  The cutoff 1e-140 keeps the window well above the
                # level where np.linalg.norm underflows (squares of entries
                # ~1e-162 flush to zero).  For 2s >= 6 the window is
                # pre-asymptotic (mode mixture) and overestimates the decay.
                pos = np.nonzero(sep > 1e-140)[0]
                t_last = int(pos[-1])
                lo = max(t_peak + 5, t_last // 2)
                window = np.arange(lo, t_last + 1)
                rho_fit = (float(np.exp(np.polyfit(
                    window, np.log(sep[window]), 1)[0]))
                    if len(window) > 10 else None)
                n_half_fit = (np.log(2.0) / (-np.log(rho_fit))
                              if rho_fit is not None and rho_fit < 1 else None)
                rows.append((name, label, t0, j1, j2, t_pred, t_detect,
                             peak, t_peak, n_half_pred, peak_relax,
                             rho_N, rho_fit, t0_max, n_half_fit))
                keep = np.concatenate([np.arange(0, min(101, n_steps + 1)),
                                       np.arange(200, n_steps + 1, 100)])
                for t in keep:
                    wcsv.writerow([name, t0, int(t), float(sep[t])])
                if t0 == max(p[0] for p in pairs):
                    deepest_series[name] = (sep, label)

    table_path = RESULTS / "delayed_collisions.csv"
    with table_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "t0", "j1", "j2", "t_detect_predicted",
                    "t_detect_measured", "peak_separation", "t_peak",
                    "peak_relaxation_steps", "half_life_predicted",
                    "half_life_fit_tail", "rho_N", "rho_fit_tail"])
        for (name, _label, t0, j1, j2, t_pred, t_det, peak, t_peak,
             nh_pred, peak_relax, rho_N, rho_fit, _t0max, nh_fit) in rows:
            w.writerow([name, t0, j1, j2, t_pred, t_det, peak, t_peak,
                        peak_relax if peak_relax is not None else "",
                        nh_pred,
                        nh_fit if nh_fit is not None else "",
                        rho_N, rho_fit if rho_fit is not None else ""])

    print(f"E5: delayed collisions (P={P}, r={r}, N={N}, lambda={lam}, "
          f"threshold {zeta:g})")
    print("  scheme        t0  pair(j1,j2)  t_pred  t_det   peak      "
          "n_half_pred   n_half_fit")
    uw1_half = next(row[9] for row in rows if row[0] == "uw1")
    for (name, _label, t0, j1, j2, t_pred, t_det, peak, _t_peak,
         nh_pred, _peak_relax, rho_N, rho_fit, _t0max, nh_fit) in rows:
        nh = f"{nh_fit:12.4g}" if nh_fit is not None else "         n/a"
        print(f"  {name:12s} {t0:3d}  ({j1},{j2})      {t_pred:4d}  "
              f"{t_det:4d}   {peak:.3e}  {nh_pred:12.4g}  {nh}")
    print("  information half-life relative to UW1 "
          f"(n_half_pred / {uw1_half:.4g}):")
    seen = set()
    for row in rows:
        if row[0] in seen:
            continue
        seen.add(row[0])
        print(f"    {row[0]:12s} hiding <= {row[13]:2d} steps,  "
              f"retention x {row[9] / uw1_half:10.4g}")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for name, (sep, label) in deepest_series.items():
        tgrid = np.arange(len(sep))
        axes[0].semilogy(tgrid[:31], np.maximum(sep[:31], 1e-18),
                         marker="o", markersize=2.5, linewidth=1.0,
                         label=label)
        axes[1].semilogy(tgrid[::100], np.maximum(sep[::100], 1e-30),
                         linewidth=1.0, label=label)
    axes[0].axhline(zeta, color="0.6", linestyle=":", linewidth=0.8)
    axes[0].set_xlabel("step $n$")
    axes[0].set_ylabel(r"$\Vert R A^n v \Vert_2$")
    axes[0].set_title("detection window", fontsize=10)
    axes[1].set_xlabel("step $n$")
    axes[1].set_title("long-time decay", fontsize=10)
    axes[1].set_ylim(1e-18, 1)
    axes[0].legend(fontsize=6)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "delayed_collisions.pdf")
    plt.close(fig)

    print(f"  wrote {table_path.name}, {series_path.name}, "
          f"figures/delayed_collisions.pdf")


def experiment_nonlinear_ensembles() -> None:
    """E6: collision ensembles for MUSCL/WENO; response-span vs linearized rank."""
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    lam, M, L_max, n_steps, zeta = 0.5, 400, 12, 60, 1e-12
    eps_list = [1e-2, 1e-5]
    fd_delta, tau_lin_factor = 1e-5, 1e-8
    N = P * r
    R = sch.build_R(P, r)
    rng = np.random.default_rng(0)

    xgrid = (np.arange(N) + 0.5) / N
    sine = 0.5 + 0.25 * np.sin(2 * np.pi * (xgrid - 0.0625))
    square = np.where(xgrid < 0.5, 1.0, 0.2)
    # regime -> (base state, perturbed parent K)
    regimes = {
        "smooth": (sine, 0),      # max-slope monotone parent
        "extremum": (sine, 2),    # peak centered in the parent (cell 15)
        "jump": (square, 3),      # flat plateau upstream of the x=0.5 jump
    }

    B0, _ = np.linalg.qr(sch.local_basis(P, r))   # orthonormal, parent 0

    theta_lead = 1e-3    # declared leading-order distinguishability floor

    ladders_path = RESULTS / "nonlinear_rank_ladders.csv"
    table_path = RESULTS / "nonlinear_ensembles.csv"
    lad_f = ladders_path.open("w", newline="")
    lad_w = csv.writer(lad_f)
    lad_w.writerow(["scheme", "regime", "eps", "L",
                    "rank_emp_lead", "rank_emp_mach", "rank_lin"])
    tab_f = table_path.open("w", newline="")
    tab_w = csv.writer(tab_f)
    tab_w.writerow(["scheme", "regime", "eps",
                    "nstar_deep", "nstar_min", "nstar_median", "nstar_max",
                    "peak_median",
                    "rank_lead_L3", "rank_lead_L5", "rank_lead_L12",
                    "rank_lin_L3", "rank_lin_L5", "rank_lin_L12",
                    "rank_mach_L5", "sigma6_over_sigma1_L5"])

    ladder_store: dict[tuple, tuple[list[int], list[int]]] = {}
    summary_rows = []

    for regime, (base, K) in regimes.items():
        BK = np.roll(B0, K * r, axis=0)
        xi = rng.standard_normal((r - 1, M))
        xi /= np.linalg.norm(xi, axis=0)
        v_deep = np.zeros(N)
        v_deep[K * r + 2] = 1.0 / np.sqrt(2.0)
        v_deep[K * r + 1] = -1.0 / np.sqrt(2.0)
        V_all = np.column_stack([v_deep, BK @ xi])   # N x (M+1)

        for name, (step, _label) in nl.STEPPERS.items():
            # ---- linearized ladder (eps-independent) -------------------
            y = base.copy()
            W = BK.copy()
            Gs = [R @ W]
            for _n in range(1, L_max + 1):
                norms = np.linalg.norm(W, axis=0)
                norms[norms == 0.0] = 1.0
                Wn = W / norms
                plus = step(y[:, None] + fd_delta * Wn, lam)
                minus = step(y[:, None] - fd_delta * Wn, lam)
                W = (plus - minus) / (2.0 * fd_delta) * norms
                y = step(y, lam)
                Gs.append(R @ W)
            ranks_lin = []
            for L in range(L_max + 1):
                OL = np.vstack(Gs[: L + 1])
                s = np.linalg.svd(OL, compute_uv=False)
                # absolute floor: a numerically zero matrix has rank 0
                ranks_lin.append(int(np.sum(s > tau_lin_factor * s[0]))
                                 if s[0] > 1e-10 else 0)

            # ---- ensembles per amplitude ------------------------------
            for eps in eps_list:
                U = np.column_stack([base[:, None] + eps * V_all,
                                     base])          # last column = base
                sep = np.zeros((n_steps + 1, M + 1))
                hist = np.zeros((L_max + 1, sch.build_R(P, r).shape[0],
                                 M + 1))
                d0 = R @ (U[:, :-1] - U[:, -1:])
                hist[0] = d0
                sep[0] = np.linalg.norm(d0, axis=0)
                for n in range(1, n_steps + 1):
                    U = step(U, lam)
                    d = R @ (U[:, :-1] - U[:, -1:])
                    sep[n] = np.linalg.norm(d, axis=0)
                    if n <= L_max:
                        hist[n] = d
                detected = sep > zeta
                nstar = np.where(detected.any(axis=0),
                                 detected.argmax(axis=0), -1)
                peaks = sep.max(axis=0)

                ranks_lead, ranks_mach, sig6 = [], [], []
                for L in range(L_max + 1):
                    D = hist[: L + 1].reshape((L + 1) * P, M + 1) / eps
                    rk_mach, _tau, s = effective_rank(D)
                    if s[0] <= 1e-10:      # numerically zero data matrix
                        rk_mach, rk_lead = 0, 0
                    else:
                        rk_lead = int(np.sum(s > theta_lead * s[0]))
                    ranks_lead.append(rk_lead)
                    ranks_mach.append(rk_mach)
                    sig6.append(float(s[5] / s[0]) if len(s) > 5 and s[0] > 0
                                else 0.0)
                ladder_store[(name, regime, eps)] = (ranks_lead, ranks_lin)

                for L in range(L_max + 1):
                    lad_w.writerow([name, regime, eps, L, ranks_lead[L],
                                    ranks_mach[L], ranks_lin[L]])
                ens = nstar[1:]
                row = [name, regime, eps,
                       int(nstar[0]), int(ens.min()),
                       float(np.median(ens)), int(ens.max()),
                       float(np.median(peaks[1:])),
                       ranks_lead[3], ranks_lead[5], ranks_lead[12],
                       ranks_lin[3], ranks_lin[5], ranks_lin[12],
                       ranks_mach[5], sig6[5]]
                tab_w.writerow(row)
                summary_rows.append(row)

    lad_f.close()
    tab_f.close()

    # ---- figure: rank ladders (small amplitude) ------------------------
    eps_fig = 1e-5
    muscl_names = [n for n in nl.STEPPERS if n.startswith("muscl")]
    weno_names = [n for n in nl.STEPPERS if n.startswith("weno")]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.4),
                             sharex=True, sharey=True)
    Lgrid = np.arange(L_max + 1)
    for col, regime in enumerate(regimes):
        for rowi, group in enumerate((muscl_names, weno_names)):
            ax = axes[rowi][col]
            # Nested line widths (thick drawn first, thin last) so that
            # ladders that coincide -- e.g. the three WENO5 variants in
            # the smooth and extremum regimes at this amplitude, where all
            # reduce to the linear WENO5 sigma=2 ladder -- stay visible as
            # a stack of colours instead of hiding under whichever line is
            # drawn last.
            n_in_group = len(group)
            for j, name in enumerate(group):
                emp, lin = ladder_store[(name, regime, eps_fig)]
                lw = 2.8 - 1.7 * j / max(n_in_group - 1, 1)
                line, = ax.plot(Lgrid, emp, marker="o", markersize=3,
                                linewidth=lw, label=nl.STEPPERS[name][1])
                ax.plot(Lgrid, lin, linestyle="--", linewidth=0.9,
                        color=line.get_color(), alpha=0.6)
            # Gray reference laws are kept out of the legend (they are
            # described in the caption) so the scheme legend stays small
            # and fits the empty lower-right corner without touching a line.
            ax.plot(Lgrid, np.minimum(2 * Lgrid, r - 1), color="0.55",
                    linestyle=":", linewidth=1.0, label="_nolegend_")
            ax.plot(Lgrid, np.minimum(Lgrid, r - 1), color="0.55",
                    linestyle="-.", linewidth=1.0, label="_nolegend_")
            if rowi == 0:
                ax.set_title(regime, fontsize=11)
            if col == 0:
                ax.set_ylabel("rank")
            _panel_label(ax, rowi * 3 + col, "tl", fontsize=13)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(r"horizon $L$")
    axes[0][0].legend(fontsize=9, loc="lower right", framealpha=0.9)
    axes[1][0].legend(fontsize=9, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURES / "nonlinear_rank_ladders.pdf")
    plt.close(fig)

    # ---- terminal summary ----------------------------------------------
    print(f"E6: nonlinear collision ensembles (P={P}, r={r}, lambda={lam}, "
          f"M={M}+deep pair, eps={eps_list}, threshold {zeta:g}, "
          f"leading floor theta={theta_lead:g})")
    print("  scheme          regime    eps    n*deep  "
          "rk_lead(3/5/12)  rk_lin(3/5/12)  rk_mach(5)  s6/s1(5)")
    for row in summary_rows:
        (name, regime, eps, nd, nmin, nmed, nmax, pk,
         e3, e5, e12, l3, l5, l12, m5, s65) = row
        print(f"  {name:15s} {regime:9s} {eps:7.0e} {nd:5d}   "
              f"{e3}/{e5}/{e12}            {l3}/{l5}/{l12}        "
              f"{m5:3d}      {s65:.2e}")
    print(f"  wrote {table_path.name}, {ladders_path.name}, "
          f"figures/nonlinear_rank_ladders.pdf")


def experiment_sensitivity_sweeps() -> None:
    """E6s (REVISION_1 m4): one-at-a-time sensitivity of the E6 ranks.

    Baseline configuration as in E6: lambda = 1/2, (P, r) = (8, 6),
    M = 400 random members plus the deterministic deep pair, seed 0,
    amplitude eps = 1e-5, FD-Jacobian step delta = 1e-5 (linearized
    tolerance 1e-8), leading floor theta = 1e-3, horizons L <= 12.
    Axes, varied one at a time around that baseline:

        floor     theta in {1e-2, 1e-3, 1e-4}  (post-hoc on the same SVDs)
        M         in {100, 200, 400, 800}      (nested prefixes, seed 0)
        seed      in {0, 1, 2}                 (fresh directions, M = 400)
        fd_delta  in {1e-4, 1e-5, 1e-6}        (linearized ladder only)
        eps       in {1e-2, 1e-3, 1e-5, 1e-7}  (physical amplitude)

    One CSV row per (axis, setting, scheme, regime): the leading-floor
    response-span ladder rho(1..5), rho(12), the machine-floor rank at
    L = 5, and the gap sigma6/sigma1 at L = 5; for the fd_delta axis
    the rank columns hold the linearized ladder at tolerance 1e-8
    instead (rank_mach5/gap5 empty).  Each axis includes its baseline
    setting, so deviations are read off within the file.  Writes
    results/nonlinear_sensitivity.csv.
    """
    RESULTS.mkdir(exist_ok=True)
    lam, L_max = 0.5, 12
    theta_lead = 1e-3
    theta_list = [1e-2, 1e-3, 1e-4]
    M_list = [100, 200, 400, 800]
    seed_list = [0, 1, 2]
    delta_list = [1e-4, 1e-5, 1e-6]
    eps_list = [1e-2, 1e-3, 1e-5, 1e-7]
    M_base, seed_base, eps_base = 400, 0, 1e-5
    N = P * r
    R = sch.build_R(P, r)

    xgrid = (np.arange(N) + 0.5) / N
    sine = 0.5 + 0.25 * np.sin(2 * np.pi * (xgrid - 0.0625))
    square = np.where(xgrid < 0.5, 1.0, 0.2)
    regimes = {"smooth": (sine, 0), "extremum": (sine, 2),
               "jump": (square, 3)}
    B0, _ = np.linalg.qr(sch.local_basis(P, r))

    # Direction banks: per (regime, seed) an (r-1, max M) block of unit
    # columns; the M axis takes nested prefixes of the seed-0 bank.
    banks: dict[tuple, np.ndarray] = {}
    for ri, regime in enumerate(regimes):
        for seed in seed_list:
            rng = np.random.default_rng(1000 * seed + ri)
            xi = rng.standard_normal((r - 1, max(M_list)))
            xi /= np.linalg.norm(xi, axis=0)
            banks[(regime, seed)] = xi

    def ensemble_svals(step, base, K, BK, xi, eps):
        """Per-horizon singular values of D_L for one ensemble."""
        M = xi.shape[1]
        v_deep = np.zeros(N)
        v_deep[K * r + 2] = 1.0 / np.sqrt(2.0)
        v_deep[K * r + 1] = -1.0 / np.sqrt(2.0)
        V = np.column_stack([v_deep, BK @ xi])
        U = np.column_stack([base[:, None] + eps * V, base])
        hist = np.zeros((L_max + 1, P, M + 1))
        hist[0] = R @ (U[:, :-1] - U[:, -1:])
        for n in range(1, L_max + 1):
            U = step(U, lam)
            hist[n] = R @ (U[:, :-1] - U[:, -1:])
        return [np.linalg.svd(hist[: L + 1].reshape((L + 1) * P, M + 1)
                              / eps, compute_uv=False)
                for L in range(L_max + 1)]

    def lead_ladder(svals, theta):
        return [int(np.sum(s > theta * s[0])) if s[0] > 1e-10 else 0
                for s in svals]

    def mach_rank5(svals, M):
        s = svals[5]
        tau = 100.0 * np.finfo(float).eps * max(6 * P, M + 1) * s[0]
        return int(np.sum(s > tau))

    def gap5(svals):
        s = svals[5]
        return float(s[5] / s[0]) if len(s) > 5 and s[0] > 0 else 0.0

    def lin_ladder(step, base, BK, delta):
        y = base.copy()
        W = BK.copy()
        Gs = [R @ W]
        for _n in range(1, L_max + 1):
            norms = np.linalg.norm(W, axis=0)
            norms[norms == 0.0] = 1.0
            Wn = W / norms
            plus = step(y[:, None] + delta * Wn, lam)
            minus = step(y[:, None] - delta * Wn, lam)
            W = (plus - minus) / (2.0 * delta) * norms
            y = step(y, lam)
            Gs.append(R @ W)
        out = []
        for L in range(L_max + 1):
            s = np.linalg.svd(np.vstack(Gs[: L + 1]), compute_uv=False)
            out.append(int(np.sum(s > 1e-8 * s[0])) if s[0] > 1e-10 else 0)
        return out

    csv_path = RESULTS / "nonlinear_sensitivity.csv"
    rows = []
    for regime, (base, K) in regimes.items():
        BK = np.roll(B0, K * r, axis=0)
        for name, (step, _label) in nl.STEPPERS.items():
            cache: dict[tuple, list] = {}

            def svals_for(seed, M, eps, _step=step, _base=base, _K=K,
                          _BK=BK, _regime=regime, _cache=cache):
                key = (seed, M, eps)
                if key not in _cache:
                    xi = banks[(_regime, seed)][:, :M]
                    _cache[key] = ensemble_svals(_step, _base, _K, _BK,
                                                 xi, eps)
                return _cache[key]

            def emit(axis, setting, svals, theta, M):
                lad = lead_ladder(svals, theta)
                rows.append([axis, setting, name, regime,
                             *lad[1:6], lad[12],
                             mach_rank5(svals, M), f"{gap5(svals):.3e}"])

            for theta in theta_list:
                emit("floor", f"{theta:g}",
                     svals_for(seed_base, M_base, eps_base), theta, M_base)
            for M in M_list:
                emit("M", f"{M:g}",
                     svals_for(seed_base, M, eps_base), theta_lead, M)
            for seed in seed_list:
                emit("seed", f"{seed:g}",
                     svals_for(seed, M_base, eps_base), theta_lead, M_base)
            for eps in eps_list:
                emit("eps", f"{eps:g}",
                     svals_for(seed_base, M_base, eps), theta_lead, M_base)
            for delta in delta_list:
                lad = lin_ladder(step, base, BK, delta)
                rows.append(["fd_delta", f"{delta:g}", name, regime,
                             *lad[1:6], lad[12], "", ""])

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["axis", "setting", "scheme", "regime",
                    "rank1", "rank2", "rank3", "rank4", "rank5",
                    "rank12", "rank_mach5", "gap5"])
        w.writerows(rows)

    # terminal summary: per axis setting, deviations from the axis baseline
    base_setting = {"floor": f"{theta_lead:g}", "M": f"{M_base:g}",
                    "seed": f"{seed_base:g}", "eps": f"{eps_base:g}",
                    "fd_delta": "1e-05"}
    # key -> (ladder rho(1..5) tuple, rho(12))
    table = {(row[0], row[1], row[2], row[3]):
             (tuple(row[4:9]), row[9]) for row in rows}
    print("E6s: sensitivity sweeps (one at a time around the E6 baseline)")
    print("  axis      setting   cells with changed rho(1..5) / rho(12) "
          "(of 21)   max|d rho(12)|")
    cells = [(sc, rg) for rg in regimes for sc in nl.STEPPERS]
    for axis in ("floor", "M", "seed", "eps", "fd_delta"):
        settings = sorted({st for (a, st, _, _) in table if a == axis},
                          key=float)
        for st in settings:
            if st == base_setting[axis]:
                continue
            ch_lad = ch_12 = max_d12 = 0
            for sc, rg in cells:
                lad, r12 = table[(axis, st, sc, rg)]
                lad0, r12_0 = table[(axis, base_setting[axis], sc, rg)]
                ch_lad += lad != lad0
                ch_12 += r12 != r12_0
                max_d12 = max(max_d12, abs(int(r12) - int(r12_0)))
            print(f"  {axis:9s} {st:9s} {ch_lad:2d} / {ch_12:2d}"
                  f"{'':31s} {max_d12}")
    print(f"  wrote {csv_path.relative_to(BASE)}")


def experiment_flux_closure() -> None:
    """E7: one-step flux-closure (telescoping) sanity check per scheme."""
    RESULTS.mkdir(exist_ok=True)
    csv_path = RESULTS / "flux_closure.csv"
    rng = np.random.default_rng(3)
    rows = []

    def residuals(u, F, unew, lam, r_val, R):
        fine = float(np.max(np.abs(
            unew - (u - lam * (F - np.roll(F, 1))))))
        right = F[(np.arange(P) + 1) * r_val - 1]
        left = np.roll(right, 1)
        coarse = float(np.max(np.abs(
            R @ unew - (R @ u - (lam / r_val) * (right - left)))))
        return fine, coarse

    # Linear family (incl. the SSP-RK3 pairing) on both grids.
    linear = dict(sch.SCHEMES)
    for r_val in (6, 8):
        N = P * r_val
        R = sch.build_R(P, r_val)
        u = rng.standard_normal(N)
        for name, (fn, _order, _label) in linear.items():
            for lam in LAMBDAS:
                coeffs = fn(lam)
                F = sch.apply_flux(sch.flux_weights(coeffs, lam), u)
                unew = sch.circulant_matrix(coeffs, N) @ u
                fine, coarse = residuals(u, F, unew, lam, r_val, R)
                rows.append(["linear", name, r_val, lam, "random",
                             fine, coarse])
        for lam in LAMBDAS:
            coeffs = sch.weno5l_ssprk3(lam)
            F = sch.apply_flux(sch.flux_weights(coeffs, lam), u)
            unew = sch.circulant_matrix(coeffs, N) @ u
            fine, coarse = residuals(u, F, unew, lam, r_val, R)
            rows.append(["linear", "weno5l_rk3", r_val, lam, "random",
                         fine, coarse])

    # Nonlinear family on the three E6 base regimes plus noise
    # (activates many limiter branches at once).
    r_val = r
    N = P * r_val
    R = sch.build_R(P, r_val)
    xgrid = (np.arange(N) + 0.5) / N
    sine = 0.5 + 0.25 * np.sin(2 * np.pi * (xgrid - 0.0625))
    square = np.where(xgrid < 0.5, 1.0, 0.2)
    datasets = {
        "sine+noise": sine + 0.05 * rng.standard_normal(N),
        "square+noise": square + 0.05 * rng.standard_normal(N),
        "random": rng.standard_normal(N),
    }
    for name, (step, _label) in nl.STEPPERS.items():
        for data_name, u in datasets.items():
            for lam in (0.9, 0.5, 0.1):
                F = nl.FLUXES[name](u, lam)
                unew = step(u, lam)
                fine, coarse = residuals(u, F, unew, lam, r_val, R)
                rows.append(["nonlinear", name, r_val, lam, data_name,
                             fine, coarse])

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["family", "scheme", "r", "lambda", "data",
                         "residual_fine", "residual_coarse"])
        writer.writerows(rows)

    worst_fine = max(row[5] for row in rows)
    worst_coarse = max(row[6] for row in rows)
    n_lin = sum(1 for row in rows if row[0] == "linear")
    n_nl = len(rows) - n_lin
    print("E7: flux-closure sanity check "
          f"({n_lin} linear + {n_nl} nonlinear cases)")
    print(f"  worst fine-level residual:   {worst_fine:.3e}")
    print(f"  worst coarse-level residual: {worst_coarse:.3e}")
    ok = worst_fine < 1e-12 and worst_coarse < 1e-12
    print(f"  telescoping identity: {'PASS' if ok else 'FAIL'} "
          "(threshold 1e-12)")
    print(f"  wrote {csv_path.relative_to(BASE)}")


def experiment_coarse_recurrence() -> None:
    """E7b: universal depth-r coarse recurrence (saturated autonomous state).

    Per residue class m the coarse Fourier mode obeys the scalar linear
    recurrence whose characteristic polynomial is prod_j (z - g_{m+jP});
    equivalently the last r coarse vectors form an autonomous state.  This
    verifies the recurrence residual to machine precision per scheme.
    """
    RESULTS.mkdir(exist_ok=True)
    csv_path = RESULTS / "coarse_recurrence.csv"
    rng = np.random.default_rng(11)
    rows = []

    family = dict(sch.SCHEMES)
    family["weno5l_rk3"] = (sch.weno5l_ssprk3, 4, "WENO5-LIN+SSP-RK3")
    for r_val in (6, 8):
        N = P * r_val
        R = sch.build_R(P, r_val)
        n_extra = 24
        for name, (fn, _order, _label) in family.items():
            for lam in (0.9, 0.5, 0.1):
                coeffs = fn(lam)
                A = sch.circulant_matrix(coeffs, N)
                x = rng.standard_normal(N)
                coarse = np.empty((r_val + n_extra + 1, P))
                for n in range(coarse.shape[0]):
                    coarse[n] = R @ x
                    x = A @ x
                yhat = np.fft.fft(coarse, axis=1)  # (time, class m)
                worst = 0.0
                for m in range(P):
                    G = np.array([sch.symbol(coeffs, m + j * P, N)
                                  for j in range(r_val)])
                    a = np.poly(G)  # leading coeff 1, length r_val+1
                    scale = np.sum(np.abs(a)) * np.max(np.abs(yhat[:, m]))
                    if scale == 0.0:
                        continue
                    for n in range(n_extra):
                        res = np.abs(np.dot(
                            a, yhat[n:n + r_val + 1, m][::-1]))
                        worst = max(worst, float(res) / scale)
                rows.append([name, r_val, lam, worst])

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scheme", "r", "lambda", "max_residual"])
        writer.writerows(rows)

    worst_all = max(row[3] for row in rows)
    # Threshold 1e-10: the antidissipative LW5+FE pairing (|g|>1) grows
    # over the window and amplifies roundoff to ~5e-12; all stable
    # pairings sit below 5e-15.
    ok = worst_all < 1e-10
    print(f"E7b: coarse recurrence check ({len(rows)} cases, depth r)")
    print(f"  worst normalized residual: {worst_all:.3e}")
    print(f"  depth-r recurrence: {'PASS' if ok else 'FAIL'} "
          "(threshold 1e-10)")
    print(f"  wrote {csv_path.relative_to(BASE)}")


def main() -> None:
    sch.self_test()
    nl.self_test()
    cert.self_test()
    experiment_rank_ladders()
    experiment_collision_certificate()
    experiment_local_rank()
    experiment_saturated_spectra()
    experiment_queue_conditioning()
    experiment_erasure()
    experiment_stability_certificate()
    experiment_delayed_collisions()
    experiment_nonlinear_ensembles()
    experiment_sensitivity_sweeps()
    experiment_flux_closure()
    experiment_coarse_recurrence()
    print("Done.")


if __name__ == "__main__":
    main()

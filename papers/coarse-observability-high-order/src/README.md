# Numerical experiments

Software used to obtain the results of the manuscript

> *Coarse Observability Across High-Order Finite-Volume Reconstructions:
> Universal Rank, Conditioning, and Dissipative Memory*
> (target: Advances in Computational Mathematics).

This package supersedes the companion paper's `run_experiments.py`
(`FV_erasure-main`), which hard-codes the first-order upwind scheme; that
original package is kept untouched as Paper A's reproducibility record.

## Contents

| File | Purpose |
|---|---|
| `schemes.py` | Linear circulant scheme registry (UW1, Lax–Wendroff, Beam–Warming, Fromm, UB3, UB5, linear-weights WENO5 + Euler), matrix builders, Fourier symbols, exact rank prediction via node counting, spectral-collision detection, exact coefficient polynomials in λ (`Fraction` arithmetic), whole-interval collision certificate via polynomial root-finding, single-parent (local) observability matrices, contraction factors, exact dissipation expansions (autocorrelation moments → fully discrete order 2s and leading coefficient A(λ), exact), the exact Chebyshev amplitude quotient `amplitude_quotient` (1−&#124;g&#124;² = (1−x)^s·Q(x,λ) with x = cos θ, used by the strict-interior-stability proposition), and the WENO5L + SSP-RK3 pairing built by convolution algebra. Run directly for the self-test (includes the closed forms A_UW1 = λ(1−λ), A_LW = λ²(1−λ²)/4, A_RK3 = λ⁴/12, the complete interior-stability factorizations of Q(±1), q₂, and ∂ₓQ(1) for FR/UB3/UB5 as exact polynomial identities, and the λ = 1/2 coincidences behind the overlapping curves in the E4 figures: FR − UB3 = [λ(1−λ)(1−2λ)/12]·(S⁻¹ − 3I + 3S − S²), so FR ≡ UB3 as operators at λ = 1/2, and LW/BW share their full amplitude spectrum there). |
| `certificates.py` | Exact rational-arithmetic certificates (no floating-point step in any accepted computation). **M1/E1b:** collision-set emptiness — the squared symbol-difference modulus &#124;δ̃&#124;² of every residue-class mode pair is certified strictly positive on λ ∈ [0,1] via Bernstein enclosure, with coefficients computed exactly in the cyclotomic group ring ℚ[ℤ_N] (zero tests modulo Φ_N for the exact λ-adic valuation) and signs certified through rational interval enclosures of cos(2πv/N) (Taylor remainder bounds; π enclosed by Machin's formula with alternating-series tails). **M3/E4e:** strict interior stability of WENO5L+SSP-RK3 — verifies the exact decomposition Q(x,λ) = λ[(1−x)·S(x,λ) + λ³/3] of the degree-13 Chebyshev quotient (S(x,0) ≡ 4/15) and certifies S > 0 on [−1,1]×[0,½] and Q/λ > 0 on [−1,1]×[½,1] by exact tensor-Bernstein positivity with de Casteljau subdivision, giving &#124;g&#124; < 1 on (0,2π) for every λ ∈ (0,1]. Run directly for the self-test (π/cosine enclosures, cyclotomic reduction, Bernstein accept/reject cases, the decomposition identity). |
| `nonlinear.py` | Nonlinear schemes for E6/E7: MUSCL flux-limiter updates (minmod, MC, van Leer, superbee; forward Euler, TVD for λ ∈ (0,1]) and WENO5 with Jiang–Shu, mapped, and Z weights advanced with SSP-RK3 (forward Euler is antidissipative for the 5th-order pairing, see the manuscript's `prop:weno-fe`), plus the realized interface fluxes of each full step (`FLUXES`, including the stage-summed SSP-RK3 flux (F⁰+F¹+4F²)/6). All steppers act columnwise on (N, C) arrays, so whole collision ensembles evolve in one vectorized call. Run directly for the self-test (φ≡1 ≡ Lax–Wendroff, φ≡r ≡ Beam–Warming, linear weights ≡ the `weno5l` circulant, conservation, flux-form reconstruction, smooth-weight consistency, TVD sanity). |
| `make_tables.py` | Regenerates the numeric bodies of the manuscript's data tables (`tab:eff-rank-sat`, `tab:queue-exponents`, `tab:half-life`, `tab:nonlinear`) from the results CSVs, plus the exact coefficient tables of Appendix A (`tab:coeff-sl`, the WENO5-LIN flux-difference display) and of the Supplementary Material (`tab:coeff-rk3`, Table S1) directly from `schemes.coeff_polys`/`schemes.weno5l_rk3_polys` in rational arithmetic, and the E6s sensitivity summary (`tab:sensitivity`, Supplementary Table S2), as LaTeX fragments in `results/tables/`, so every number in the paper is machine-regenerable and no coefficient is transcribed by hand. |
| `run_experiments.py` | Experiment drivers. Currently: **E1** (rank ladders across the scheme family, testing the universality theorem + collision catalogue), **E1b** (collision certificate over all eight fully discrete operators — the seven spatial schemes plus the composed WENO5L+SSP-RK3 pairing, certified directly on its exact composed cubic coefficients since the RK3 stability polynomial is not injective (REVISION_2 M2) — two independent layers: the exact rational-arithmetic Bernstein certificate of `certificates.py` — all 2408 pairs certified with a single Bernstein cell each and a uniform bound min &#124;δ̃&#124;² ≥ 5.6×10⁻³ on [0,1] — plus floating-point root isolation on the same exact-coefficient pair polynomials as a cross-check; certifies the collision set empty on the whole interval (0,1], not just a sampled λ grid), **E1c** (single-parent observability ranks ρ(L) for the seven spatial schemes plus the composed WENO5L+SSP-RK3 one-step operator, testing the local rank law σL, its wrap-around closed form, and the multi-parent regime where reach exceeds r−1; also yields the local injectivity horizon T_loc reported in the E5 table), **E2** (saturated singular spectra, effective-rank tables, per-class Vandermonde block conditioning, exact first-order node velocities μ_k, Gramian-monotonicity check), **E3** (queue conditioning: σ_min⁺ of the local observation map vs. λ, exponent fits vs. the prediction γ = L_q), **E4** (dissipative erasure: exact orders/leading coefficients, interior-stability scan, ρ_N sweep vs. the (A/2)(2π/N)^{2s} law, step flattening + slowest-mode contraction fits), **E4e** (exact interior-stability certificate for the SSP-RK3 pairing via `certificates.py`; the E4b grid scan becomes an independent cross-check), **E5** (delayed collisions: sharp compensated pairs vs. the detection-delay law, information half-life table), **E6** (nonlinear collision ensembles: three base-state regimes × seven MUSCL/WENO schemes, M = 400 members + deterministic deep pair, empirical response-span ranks at two declared floors vs. FD-linearized ladders, nonlinearity gap σ₆/σ₁ at two amplitudes), **E6s** (sensitivity sweeps, REVISION_1 m4: one-at-a-time variation of the rank floor θ ∈ {10⁻², 10⁻³, 10⁻⁴}, ensemble size M ∈ {100, 200, 400, 800} as nested prefixes, seed ∈ {0, 1, 2}, FD-Jacobian step δ ∈ {10⁻⁴, 10⁻⁵, 10⁻⁶}, and amplitude ε ∈ {10⁻², 10⁻³, 10⁻⁵, 10⁻⁷} around the E6 baseline, over all 21 scheme–regime cells; the M/seed/FD-step knobs leave every reported ladder unchanged, while the floor and amplitude axes move ranks exactly as the tolerance-indexed definition and the crossover analysis predict (REVISION_2 M5), Supplementary Material Sec. S2, `app:sensitivity`), **E7** (flux-closure sanity check: one-step telescoping identity at fine and coarse level for all linear schemes — physical-flux weights F = φ/λ by partial summation, manuscript `rem:flux-normalization` — and all nonlinear schemes — realized stage-summed fluxes; machine-precision residuals), **E7b** (universal coarse recurrence: each coarse Fourier mode obeys the depth-r recurrence with characteristic polynomial ∏ⱼ(z − g_{m+jP}); machine-precision residuals for all linear schemes + the SSP-RK3 pairing at r ∈ {6,8}). E1 sweeps both r = 6 and r = 8. |

## Scheme keys vs manuscript display names

Code and CSV artifacts keep the original scheme keys; the manuscript
(REVISION_2 m4) displays the linear-weights WENO5 operators under
unambiguous names, so that `LW` (Lax–Wendroff) cannot be confused with
the former `LW5`:

| Code/CSV key | Manuscript display name |
|---|---|
| `weno5l` | WENO5-LIN (+FE where the pairing matters) |
| `weno5l_rk3` | WENO5-LIN+SSP-RK3 |
| `weno5-js` / `weno5-m` / `weno5-z` | WENO5-JS / WENO5-M / WENO5-Z (all advanced with SSP-RK3, stated once in the manuscript's family definition) |

Figure legends and the generated table fragments use the display
names; the keys appear only in code, CSV headers/rows, and this README.

## How to run

```bash
python run_experiments.py   # E1-E7: all CSVs and figures
python make_tables.py       # LaTeX table bodies from the CSVs
```

Outputs are written to `../results/` (CSV, plus LaTeX fragments in
`results/tables/`) and `../figures/` (PDF, referenced by
`article.tex`). Dependencies: Python ≥ 3.10, NumPy, Matplotlib (same as
the companion package). All computations are deterministic (fixed random
seeds); the SVD tolerance convention follows the companion paper.
The full run (E1–E7) takes roughly four minutes on a laptop; the
long-time evolutions in E4d/E5 dominate.  E6 takes about 20 seconds and
E7 about one second.

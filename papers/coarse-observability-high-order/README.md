# Coarse Observability Across High-Order Finite-Volume Reconstructions: Universal Rank, Conditioning, and Dissipative Memory

The program's fourth paper, by Antonis Polemitis, Ioannis Kokkinakis,
Nicholas Christakis, and Dimitris Drikakis. The
[first paper](../finite-horizon-memory/) determined, for first-order
upwind advection, how much extra memory exact coarse prediction needs,
how stably that memory can be recovered, and how long it lives. This
paper answers all three questions for the high-order finite-volume
reconstructions used in practice. For every linear scheme in the
family, the finite-horizon observation rank follows the same universal
ladder as first-order upwind. The scheme sets everything else: when
hidden information is detected, how well it can be recovered, and how
fast dissipation erases it. The manuscript is staged for author
sign-off. It will be posted to arXiv and submitted to a journal after
the release is frozen, and identifiers will be recorded here as they
become permanent.

## Contents

| Path | Contents | Status |
| --- | --- | --- |
| `article.tex`, `article.bib`, `article.pdf` | Manuscript source and the authors' compiled PDF | staged |
| `supplementary.tex`, `supplementary.pdf` | Supplementary Material | staged |
| `sn-jnl.cls`, `sn-mathphys-num.bst` | Springer Nature class and bibliography style | staged |
| [`src/`](src/) | The authors' code: scheme registry, exact certificates, nonlinear schemes, experiment drivers E1 to E7, and the table generator | staged |
| [`results/`](results/) | The authors' reference outputs: CSV files, and the manuscript's table bodies in `results/tables/` | staged |
| [`figures/`](figures/) | The seven figures the manuscript includes | staged |
| [`certificates/`](certificates/), [`artifacts/`](artifacts/), [`tests/`](tests/) | Independent exact-arithmetic certificate suite, its canonical outputs, and replay tests | staged |

Keep this layout. `src/run_experiments.py` writes `results/` and
`figures/` into its parent directory, so `article.tex`, `src/`,
`results/`, and `figures/` must stay siblings.

## Replaying the paper

Two verification layers.

The authors' code regenerates every CSV file, figure, and table body.
From `src/`, with Python 3.12 or later, NumPy 2, and Matplotlib 3.9 or
later:

```bash
cd papers/coarse-observability-high-order/src
python schemes.py          # self-test
python certificates.py     # self-test of the exact certificates
python nonlinear.py        # self-test
python run_experiments.py  # E1 to E7: every CSV and figure
python make_tables.py      # the manuscript's table bodies
```

The run is deterministic and takes one to two minutes. These outputs
reproduce byte for byte, and continuous integration requires it:
every table body in `results/tables/`, and `collision_certificate.csv`,
`collision_certificate_exact.csv`, `erasure_orders.csv`,
`stability_certificate.csv`, `local_observability.csv`,
`rank_collision_catalogue.csv`, and `rank_vs_horizon_smalllambda.csv`.
On Windows the table bodies are written with CRLF line endings, so
compare them with line endings normalized. The other CSV files are
floating-point diagnostics that agree to about 13 significant digits.
Quantities at the limit of double precision, such as singular values
near 1e-17 and machine-precision empirical ranks, can differ between
machines. Figure PDFs embed the Matplotlib version and a timestamp, so
a regenerated figure never hash-matches the committed one.

The independent certificate suite re-derives the paper's exact results
with a separate implementation written from the manuscript's
definitions, and pins its outputs as canonical artifacts:

```bash
python papers/coarse-observability-high-order/certificates/certify_observation_rank_structure.py
python papers/coarse-observability-high-order/certificates/certify_dissipation_conditioning_erasure.py
```

The two scripts run 355 exact checks between them and use only the
Python standard library. They cover the universal rank ladder, the
invisible subspace, the local ranks, the spectral-collision
certificate, the dissipation coefficients, interior stability, the
conditioning exponents at the working grids, and the erasure rates. A
successful replay reproduces the committed artifacts byte for byte.
Each artifact also records where the manuscript or the authors'
outputs disagree with the exact values. The replay tests in `tests/`
also compare every value the certificates transcribe against the
committed `results/`. Rerunning the experiments rewrites the
floating-point CSV files, so run the tests on a clean checkout.
Continuous integration reruns both layers on every change.

## Release

The release will be tagged `coarse-observability-high-order-v1` once
all four authors have signed off. Identifiers will be recorded here as
they become permanent.

## Licensing

The code in `src/` is MIT licensed by its author; see
[`src/LICENSE`](src/LICENSE). The certificate suite in `certificates/`
and `tests/` is under Apache-2.0. `artifacts/`, `results/`, and
`figures/` are data under CC BY 4.0. The manuscript and supplement,
in source and PDF, are © the authors.
[LICENSES/](../../LICENSES/README.md) is the authority.

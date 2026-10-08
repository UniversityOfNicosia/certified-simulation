# Exact Lifting of Coarse Histories in Periodic Upwind Finite Volumes

The program's third paper, a short note by Antonis Polemitis, Nicholas
Christakis, and Dimitris Drikakis. The
[first paper](../finite-horizon-memory/) asked which coarse history a
given fine-grid state produces. This note asks the inverse question:
which parent-average histories can be lifted to a fine-grid initial
state at all. For periodic scalar advection under first-order positive
upwinding with at least two parent cells, every mass-conserving L-step
parent history is realizable exactly when each parent has at least
L + 1 children. The note constructs a normalized lift, describes every
lift of a realizable short-horizon history, and shows that with fewer
children the admissible histories must satisfy an explicit lag-r
parent-level recurrence. The manuscript is staged for author sign-off.
It will be posted to arXiv and submitted to a journal after the
release is frozen, and identifiers will be recorded here as they
become permanent.

## Contents

| Directory | Contents | Status |
| --- | --- | --- |
| [`manuscript/`](manuscript/) | Note source and the authors' compiled PDF | staged |
| [`experiments/`](experiments/) | The authors' exact-arithmetic certificate script, which regenerates the checks of Table 1 as a CSV file and a LaTeX table fragment | staged |
| [`certificates/`](certificates/), [`artifacts/`](artifacts/), [`tests/`](tests/) | Independent certificate suite, its canonical outputs, and replay tests | staged |

## Replaying the note

Two verification layers, both exact and both deterministic.

The authors' script reproduces the three checks of Table 1 in exact
rational arithmetic and writes `results/history_realizability_certificates.csv`
and `tables/history_realizability_certificates.tex` in the working
directory:

```bash
cd papers/coarse-history-realizability/experiments
python run_history_realizability_certificates.py
```

The independent certificate suite re-derives the note's results with a
separate implementation written from the manuscript's definitions, and
pins its outputs as canonical artifacts:

```bash
python papers/coarse-history-realizability/certificates/certify_short_horizon_lifting.py
python papers/coarse-history-realizability/certificates/certify_long_horizon_minimal_r.py
```

The two scripts run 337 exact checks between them and use only the
Python standard library. They cover the hierarchy and the
collar-to-flux map, the short-horizon image, fibers, and normalized
lift, the long-horizon image and its lag-r recurrence, the minimal
child-count law including its sharpness, both worked examples, and the
three rows of Table 1. A successful replay reproduces the committed
artifacts byte for byte. Each artifact also records where the
manuscript or the authors' outputs disagree with the exact values. The
replay tests in `tests/` also compare the certified values with
`manuscript/main.tex` and, when the authors' script has been run, with
its outputs in `experiments/`. Continuous integration reruns both
layers on every change.

## Licensing

The code in `experiments/` is MIT licensed by its author; see
[`experiments/LICENSE`](experiments/LICENSE). The certificate suite in
`certificates/` and `tests/` is under Apache-2.0, and `artifacts/` is
data under CC BY 4.0. The manuscript, in source and PDF, is © the
authors. [LICENSES/](../../LICENSES/README.md) is the authority.

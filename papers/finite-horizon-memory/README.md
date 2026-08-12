# Exact Finite-Horizon Memory, Conditioning, and Dissipative Decay in Coarse Upwind Finite-Volume Prediction

The program's first paper, by Antonis Polemitis, Nicholas Christakis,
and Dimitris Drikakis. It solves one family completely: for periodic
scalar advection under a first-order upwind finite-volume method, it
determines exactly what additional information coarse parent averages
need for exact prediction, when the algebraically visible information
is recoverable in practice, and how unresolved information decays
under the scheme's own dissipation. The paper is posted at
[arXiv:2608.08633](https://arxiv.org/abs/2608.08633); submission to
the Journal of Scientific Computing is in progress, and the journal
identifier will be recorded here when assigned.

## Contents

| Directory | Contents | Status |
| --- | --- | --- |
| [`manuscript/`](manuscript/) | Paper source and the compiled PDF | staged |
| [`certificates/`](certificates/) | One exact-arithmetic certificate program per theorem | staged |
| [`artifacts/`](artifacts/) | The certificates' canonical outputs, replayed byte-for-byte by CI | staged |
| [`tests/`](tests/) | Replay tests: regenerate each certificate and require byte identity | staged |
| [`experiments/`](experiments/) | Numerical experiments: figures, CSV data, and LaTeX tables used in the paper | staged |

## Replaying the paper

The three theorems each have an executable certificate; the certificates
and their exact scope are described in
[`certificates/README.md`](certificates/README.md). From the repository
root, with Python 3.12 or later:

```bash
python papers/finite-horizon-memory/certificates/certify_finite_horizon_memory.py
python papers/finite-horizon-memory/certificates/certify_queue_conditioning.py
python papers/finite-horizon-memory/certificates/certify_dissipative_decay.py
```

A successful replay reproduces the committed artifacts byte for byte.
The floating-point illustrations regenerate with:

```bash
cd papers/finite-horizon-memory/experiments
uv run --group experiments python run_experiments.py
uv run --group experiments python make_tables.py
```

## Release

This folder is released as
[`finite-horizon-memory-v1`](https://github.com/UniversityOfNicosia/certified-simulation/releases/tag/finite-horizon-memory-v1)
(9 August 2026), approved for submission by all three authors. The
release carries the per-file hash manifest and the arXiv source
package, and is archived at Zenodo under version DOI
[10.5281/zenodo.21859861](https://doi.org/10.5281/zenodo.21859861).
The posted preprint is
[arXiv:2608.08633](https://arxiv.org/abs/2608.08633)
(DOI [10.48550/arXiv.2608.08633](https://doi.org/10.48550/arXiv.2608.08633)),
produced from this release's source package.
The verification ladder for checking the release is in
[REPRODUCING.md](../../REPRODUCING.md).

## Licensing

The `experiments/` directory is MIT licensed by its author (see its
`LICENSE` file). Everything else follows the repository's terms in
[LICENSES/](../../LICENSES/README.md).

# Exact Finite-Horizon Memory, Conditioning, and Dissipative Decay in Coarse Upwind Finite-Volume Prediction

The program's first paper, by Antonis Polemitis, Nicholas Christakis,
and Dimitris Drikakis. It solves one family completely: for periodic
scalar advection under a first-order upwind finite-volume method, it
determines exactly what additional information coarse parent averages
need for exact prediction, when the algebraically visible information
is recoverable in practice, and how unresolved information decays
under the scheme's own dissipation. The manuscript is in final
preparation for submission to the Journal of Scientific Computing.

## Contents

| Directory | Contents | Status |
| --- | --- | --- |
| [`experiments/`](experiments/) | Numerical experiments: figures, CSV data, and LaTeX tables used in the paper | staged |
| `certificates/` | Exact-arithmetic certificate programs for the paper's results | staging |
| `artifacts/` | The certificates' canonical outputs | staging |
| `tests/` | Regeneration and pinning tests | staging |
| [`manuscript/`](manuscript/) | Paper and supplement sources, compiled PDFs | staged |

This folder becomes the content of the paper's release tag when the
manuscript is finalized; the release adds the hash manifest and the
literal replay commands, per [REPRODUCING.md](../../REPRODUCING.md).

## Licensing

The `experiments/` directory is MIT licensed by its author (see its
`LICENSE` file). Everything else follows the repository's terms in
[LICENSES/](../../LICENSES/README.md).

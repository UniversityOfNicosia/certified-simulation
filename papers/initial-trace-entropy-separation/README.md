# Initial-Trace Insufficiency and Entropy-Accounting Separation in Finite-Volume Coarse Graining

The program's second paper, a short communication by Antonis Polemitis,
Nicholas Christakis, and Dimitris Drikakis. It establishes two one-step
information boundaries for finite-volume coarse graining: parent
averages with complete initial parent-interface traces close a
forward-Euler transition but provably fail to close the SSPRK(2,2)/Heun
transition, and exact conservative one-step closure does not determine
the entropy account, shown by an explicit Burgers–Rusanov family whose
branches are conservatively identical yet entropy-distinguishable. It
is the one-step companion to the program's
[first paper](../finite-horizon-memory/) on multi-step memory. The
manuscript is staged for author sign-off; it will be posted to arXiv
and submitted to a journal after the release is frozen, and identifiers
will be recorded here as they become permanent.

## Contents

| Directory | Contents | Status |
| --- | --- | --- |
| [`manuscript/`](manuscript/) | Paper source and the compiled PDF | staged |
| [`certificates/`](certificates/) | Independent exact-arithmetic certificate programs, one per result | staged |
| [`artifacts/`](artifacts/) | The certificates' canonical outputs, replayed byte-for-byte by CI | staged |
| [`tests/`](tests/) | Replay tests: regenerate each certificate and require byte identity | staged |
| [`experiments/`](experiments/) | The authors' certificate script: regenerates the paper's tables and CSV data in exact rational arithmetic | staged |

## Replaying the paper

Two verification layers, both exact and both deterministic:

The authors' script regenerates every table in the paper:

```bash
cd papers/initial-trace-entropy-separation/experiments
python run_closure_certificates.py
```

The independent certificate suite replays the paper's results with a
separate implementation and canonical, byte-pinned artifacts:

```bash
python papers/initial-trace-entropy-separation/certificates/certify_initial_trace_insufficiency.py
python papers/initial-trace-entropy-separation/certificates/certify_entropy_accounting_separation.py
```

A successful replay reproduces the committed artifacts byte for byte
(135 exact checks in total). Continuous integration reruns both layers
on every change.

## Licensing

Everything in this folder follows the repository's terms in
[LICENSES/](../../LICENSES/README.md): code under Apache-2.0, the
certificate artifacts under CC BY 4.0, and the manuscript copyright
the authors.

# Certified Simulation

**Mathematical certification for simulation, learned models, and
digital twins.**

![Code: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue)
![Artifacts: CC BY 4.0](https://img.shields.io/badge/artifacts-CC%20BY%204.0-green)
![Manuscripts: reserved](https://img.shields.io/badge/manuscripts-%C2%A9%20the%20authors-lightgrey)

This repository is the public release record of the Certified
Simulation research program at the University of Nicosia. The program
establishes exact information-theoretic limits for coarse-grained
prediction: when a reduced account of a finer computation determines
its own future, what such an account must retain in order to do so,
and the exact best performance that any consumer of the same retained
data can achieve. Each result ships with machine-checked certificates
that replay every identity and every counterexample in exact
arithmetic.

Two ideas organize the program. First, fidelity claims about
simulations, learned surrogates, and digital twins should be measured
against proved reference points rather than against other models.
These reference points are theorems, not benchmarks of practice:
exact ceilings on success, equivalently floors on mismatch. Second,
such claims should be *certifiable*. Every quantitative statement
published here is paired with an executable certificate that anyone
can re-run.

## Releases

Papers are released under frozen tags containing the submitted
sources, compiled PDFs, executable certificates with their canonical
artifacts, and a hash manifest. Released tags are never rewritten;
corrections arrive as new releases that name what they correct. The
first releases are in preparation. The index lives in
[RELEASES.md](RELEASES.md).

## Start here

| If you are… | Read |
| --- | --- |
| Checking whether the claims hold | [REPRODUCING.md](REPRODUCING.md): the five-level verification ladder, from hashing the files to reimplementing the certificates independently |
| Asking why you should not have to trust us | [PROVENANCE.md](PROVENANCE.md): the six-link chain that makes every release checkable from its own contents |
| Replicating, correcting, or building on the work | [CONTRIBUTING.md](CONTRIBUTING.md): replication reports, independent reimplementation, and the corrections process |
| Reusing code, data, or text | [LICENSES/](LICENSES/README.md): three components, three terms |
| Wondering who conducts this program | [ABOUT.md](ABOUT.md): the university, the institute, and UNIC Evolve |

Policies: [SECURITY.md](SECURITY.md) ·
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) · [NOTICE](NOTICE)

## Licensing

Three components, three terms, with [LICENSES/](LICENSES/README.md)
as the authority: code under Apache-2.0, certificate artifacts and
data under CC BY 4.0, manuscript sources and PDFs © the authors.

## Contributing

No pull requests, by design: releases are byte-anchored to reviewed,
frozen artifacts. Replication reports, independent reimplementations,
corrections, and questions are welcome as issues.
[CONTRIBUTING.md](CONTRIBUTING.md) explains what makes each one
useful.

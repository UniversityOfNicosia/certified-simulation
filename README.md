# Certified Simulation

Mathematical certification for simulation, learned models, and digital
twins.

This repository is the public release record of the Certified
Simulation research program at the University of Nicosia. The program
establishes exact information-theoretic limits for coarse-grained
prediction — when a reduced account of a finer computation determines
its own future, what such an account must retain in order to do so,
and the exact best performance that any consumer of the same retained
data can achieve — and ships each result with machine-checked
certificates that replay every identity and every counterexample in
exact arithmetic.

Two ideas organize the program. First, fidelity claims about
simulations, learned surrogates, and digital twins should be measured
against proved reference points rather than against other models;
these reference points — exact ceilings on success, equivalently
floors on mismatch — are theorems, not benchmarks of practice.
Second, such claims should be *certifiable*: every quantitative
statement published here is paired with an executable certificate
that anyone can re-run.

## Releases

Papers are released here as they ship, each under a frozen tag
containing the submitted sources, compiled PDFs, executable
certificates with their canonical artifacts, and a hash manifest
tying the release to the program's internal research record. The
first releases are in preparation; see [RELEASES.md](RELEASES.md)
for the release index and [REPRODUCING.md](REPRODUCING.md) for
replay instructions.

## About

This research program is conducted at the **Institute for Advanced
Modelling and Simulation** and **UNIC Evolve** at the [University of
Nicosia](https://www.unic.ac.cy/). IAMS is the university's institute
for computational science and simulation. UNIC Evolve is the
university's interdisciplinary research and innovation initiative on
the opportunities and risks of emerging artificial superintelligence;
the work in this repository is produced under its AI-assisted,
human-responsible research workflow, with the verification protocol
documented in [PROVENANCE.md](PROVENANCE.md).

## Licensing

Three components, three terms — [LICENSES/](LICENSES/README.md) is
the authority:

- **Code** (certifiers, tests, tooling): Apache License 2.0.
- **Certificate artifacts and data**: Creative Commons Attribution
  4.0 International.
- **Manuscript sources and PDFs**: © the authors, all rights
  reserved; preprints are distributed through arXiv under its
  license, and journal versions under the terms of the publishing
  agreement.

## Contributing

This repository is a curated release record rather than a development
tree; see [CONTRIBUTING.md](CONTRIBUTING.md). Replication reports and
questions are welcome as issues.

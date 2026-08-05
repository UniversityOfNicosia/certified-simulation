# Certified Simulation

**Mathematical certification for simulation, learned models, and
digital twins.**

![Code: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue)
![Artifacts: CC BY 4.0](https://img.shields.io/badge/artifacts-CC%20BY%204.0-green)
![Manuscripts: reserved](https://img.shields.io/badge/manuscripts-%C2%A9%20the%20authors-lightgrey)

Certified Simulation is a joint initiative of [UNIC
Evolve](ABOUT.md#unic-evolve) and the [Institute for Advanced
Modelling and Simulation](ABOUT.md#institute-for-advanced-modelling-and-simulation)
at the University of Nicosia; this repository is its public release
record.

Simulation is becoming infrastructure. Learned surrogates stand in
for solvers, foundation models act as simulators, and digital twins
of machines, buildings, and increasingly people inform decisions that
once required physical tests or expert judgment. Assurance practice
has not kept pace: models are validated against other models, against
held-out data, and through expert review. All of this answers how
well a particular model did. None of it answers the prior question:
what could any model know from the data a system actually retains,
and where is the line past which no training, architecture, or scale
can help?

Questions of this kind have exact answers, and this program is
proving them, class by class. For declared classes of coarse-grained
computation it establishes when a reduced account of a finer process
determines its own future, what the account must retain to do so, and
the best performance any consumer of the same retained data can
achieve: exact ceilings on success, equivalently floors on mismatch,
proved as theorems. Each result is also
*certifiable*: every quantitative statement published here comes
paired with an executable certificate that replays every identity and
every counterexample in exact arithmetic, and that anyone can re-run.

The first releases are foundations: complete, exact answers on
deliberately small, fully declared classes. The work that follows
extends the theory to more realistic settings, among them noisy
observation, tolerance, and stochastic prediction, and builds
practical tools on top: benchmarks that score learned models against
proved limits, and in time a certification that a deployed simulation
or digital twin can pass. The goal is for a certified simulation
to carry mathematical meaning: specific theorems, and a certificate
anyone can re-run.

## How this repository works

Every paper the program publishes lands here as a release. A release
is a frozen tag that carries everything needed to verify the work:
the paper and supplement sources exactly as submitted, the compiled
PDFs, the certificate programs together with the canonical artifacts
they produce, and a manifest listing the SHA-256 hash of every file.
Once tagged, a release never changes. If we find an error, or someone
shows us one, we publish a new release that states what it corrects,
and the flawed tag stays in place as part of the record.

The documents at the top level, indexed below, set the standing
rules: how to verify a release, why no trust is required, how to
contribute, and who runs the program. [RELEASES.md](RELEASES.md)
indexes every release with its hashes; the first releases are in
preparation.

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

We take no pull requests: releases are byte-anchored to reviewed,
frozen artifacts, and an outside commit would break the anchoring.
Everything else comes in through issues, from replication reports to
corrections and questions. [CONTRIBUTING.md](CONTRIBUTING.md) covers
the details, and Working with us, below, covers collaboration.

## Working with us

The open problems are bigger than one group. If you work on the
theory side and want to extend these results to new classes or
settings, if you build learned simulators or digital twins and want
to score them against proved limits, or if you want to help turn the
theorems into benchmarks and certification instruments, we want to
hear from you. Each paper's discussion section states the open
directions we see; you will see others. Start with an issue
describing your interest, or reach the program through
[UNIC Evolve](https://evolve.unic.ac.cy/).

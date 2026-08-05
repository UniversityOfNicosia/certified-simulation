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
theorems rather than benchmarks of practice. Each result is also
*certifiable*: every quantitative statement published here comes
paired with an executable certificate that replays every identity and
every counterexample in exact arithmetic, and that anyone can re-run.

The first releases are foundations: complete, exact answers on
deliberately small, fully declared classes. The work that follows
extends the theory to more realistic settings, among them noisy
observation, tolerance, and stochastic prediction, and builds
practical tools on top: benchmarks that score learned models against
proved limits, and in time a certification that a deployed simulation
or digital twin can pass. The goal is for a certified simulation to
mean an exact, checkable claim, not a label.

## How this repository works

Work arrives here as releases: frozen tags, each containing the
submitted sources, compiled PDFs, executable certificates with their
canonical artifacts, and a hash manifest tying everything together.
Released tags are never rewritten; corrections arrive as new releases
that name what they correct. Between releases, the repository carries
the general documents indexed below. The release index lives in
[RELEASES.md](RELEASES.md); the first releases are in preparation.

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

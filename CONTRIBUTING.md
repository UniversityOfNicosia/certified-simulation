# Contributing

This repository is the curated public release record of a research
program whose development happens in an internal record under its own
review protocol. Every release is byte-anchored: hashes tie the
published sources, certificates, and PDFs to reviewed, frozen
artifacts. That shapes what contribution means here. **The repository
accepts no pull requests, and its releases are never rewritten**, but
there are several ways to contribute that matter more than patches.

## Ways to contribute

### 1. Replicate a release

The single most valuable contribution. Check out a release tag,
follow [REPRODUCING.md](REPRODUCING.md), and file an issue with your
result, **whether or not it matched**. A useful replication report
includes:

- the release tag and the commands you ran;
- your platform (OS, architecture) and Python version;
- the executed check counts each certificate printed;
- the SHA-256 hashes of your regenerated artifacts next to the
  committed ones;
- for a mismatch: the first differing file, and anything unusual
  about your environment.

Confirmations build the public evidence base. Mismatches are gold:
they either uncover an environment sensitivity worth documenting or
a genuine defect worth a correction.

### 2. Reimplement independently

The certificates here deliberately re-derive every operator from the
definitions in the papers rather than sharing library code, so that
agreement is evidence about the mathematics rather than about a
common implementation. The strongest possible check is one we cannot
produce ourselves: **an independent reimplementation in your own
language and toolchain**. If you build one, in Julia, Rust, a proof
assistant, or anything else, file an issue with a link, whether it
agrees or disagrees. Disagreement reports receive the same triage as
corrections, below.

### 3. Report a suspected error

For a suspected error in a released paper, proof, supplement, or
certificate, open an issue with:

- the release tag and the precise location (page, theorem, equation,
  file, or check);
- a self-contained statement of the problem, ideally the smallest
  case that exhibits it.

Reports are triaged against the internal research record. The
outcome (confirmed, not confirmed, or clarified) is posted on the
issue either way. Confirmed corrections are addressed in a
subsequent tagged release with the correction noted and the reporter
credited, unless you prefer otherwise. Released tags themselves are
never modified.

### 4. Ask a question

Questions about the results, their declared scopes, or the replay
procedure are welcome as issues. Please search existing issues
first, and keep one topic per issue.

## Why pull requests are closed

A release is checkable precisely because every file in it is pinned
by hash to artifacts that passed the program's review. An external
commit, however good, would either break that chain or bypass the
review that gives the chain meaning. Improvements suggested in
issues can enter the internal record, pass review, and appear in a
future release; they cannot enter through the front door of this
mirror.

## Using the work

Reuse is encouraged under the licenses in
[LICENSES/](LICENSES/README.md): code under Apache-2.0, certificate
artifacts and data under CC BY 4.0 (cite the corresponding paper;
each release's `CITATION.cff` gives the reference), manuscripts ©
the authors. No permission is needed for anything the licenses
already grant.

## Collaboration and contact

For public matters, the issue tracker is the front door. For
academic collaboration inquiries, contact the authors at the
addresses given in the released papers. Security-sensitive reports:
see [SECURITY.md](SECURITY.md).

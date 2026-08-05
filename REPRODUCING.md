# Reproducing

Every quantitative claim released here is designed to be replayed on
an ordinary computer, in minutes, from the release alone. This page
is the contract for what "reproducing" means. Each release records
its own literal commands and expected outputs in a `REPLAY` section
of its release notes, and those per-release instructions always
follow the structure below.

Verification is a ladder. Climb as far as your purpose requires.

## Level 0: Integrity (a minute; no toolchain)

Confirm you hold exactly the released bytes: recompute SHA-256 hashes
of the files in the release and compare them against the release
manifest, which lists every file. The manifest also records the exact
source commit and tree identifiers.

*Establishes:* nothing was corrupted or altered between us and you.

## Level 1: Certificates (minutes; Python only)

Create the pinned environment (each release ships a lockfile), then
run each certificate program. Every certificate:

- re-derives its operators from the definitions printed in the paper,
  sharing no code with other certificates or with any internal
  library;
- replays every certified identity, probability law, and
  counterexample in **exact rational arithmetic**, with no floating
  point, no tolerances, and no random seeds in any certified
  quantity;
- prints its executed check count and writes a canonical, byte-stable
  JSON artifact.

Then run the shipped comparison step, which regenerates the artifacts
and compares them **byte for byte** against the committed ones. Exact
arithmetic makes this platform-independent: the same bytes on Linux,
macOS, and Windows, on any architecture. There is no "close enough".
The comparison passes identically or fails.

Where a release also contains auxiliary empirical material (for
example, a numerical illustration involving trained models), that
material is seed-pinned, labeled as empirical in the paper itself,
and gated by shipped tests. The release notes state which checks are
exact and which are empirical.

*Establishes:* every computation the paper relies on happens as
printed.

## Level 2: Documents (optional; a TeX toolchain)

Rebuild the PDFs from the released sources. Releases are
self-contained: class files, bibliography, and all inputs ship in the
bundle, with no external dependencies. Under the pinned toolchain
recorded in the release notes, the rebuilt PDFs match the released
ones by hash. Under a different TeX distribution byte-identity is not
guaranteed, since TeX embeds toolchain details, and the check
becomes: the build completes with no unresolved references or
citations, and the rendered content matches.

*Establishes:* the released PDFs are what the released sources say.

## Level 3: Mathematics (human time)

Read the proofs. Every load-bearing theorem is proved in full in the
released paper and its supplement; nothing defers to an external
record, and the certificates of Level 1 cross-check each proof's
computational content on explicit instances. This level is the only
one a machine cannot do for you, and the releases are built so that
it is the only thing left for you to do.

*Establishes:* the theorems themselves.

## Level 4: Independence (the gold standard)

Reimplement the certificates from the papers' definitions in your own
language and toolchain, sharing nothing with our code, and compare
results. This is the strongest possible check and the one we cannot
perform on ourselves; see
[CONTRIBUTING.md](CONTRIBUTING.md#2-reimplement-independently).
Agreement and disagreement reports are equally welcome.

## Environment

- Python 3.12 or later with [`uv`](https://docs.astral.sh/uv/); each
  release ships the lockfile that pins everything else.
- No network access is needed after setup, no GPU, and no unusual
  hardware: certificate replays are laptop-scale, typically seconds
  to a few minutes each.
- Level 2 additionally needs a TeX toolchain; the release notes name
  the pinned one.

## If something does not match

1. Confirm you are on the release tag itself, not a branch, and that
   Level 0 passes. Most mismatches are a modified or partial
   checkout.
2. Re-create the environment from the lockfile (`--frozen`), not from
   loose versions.
3. Still differing? File a replication report; the checklist in
   [CONTRIBUTING.md](CONTRIBUTING.md#1-replicate-a-release) says what
   to include. Mismatch reports are the most valuable input this
   repository can receive: each one is either an environment
   sensitivity worth documenting or a defect worth a correction
   release.

## What reproduction does and does not establish

Levels 0 through 2 establish that the released record is internally
exact: the bytes, the computations, and the documents agree with each
other. They do not by themselves prove the theorems; that is Level 3,
by design in your hands. Every certificate checks exactly the
instances it declares, at the scopes the paper states. The releases
are engineered so that trust is never part of the chain; see
[PROVENANCE.md](PROVENANCE.md).

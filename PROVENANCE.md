# Provenance

Provenance in this repository means exactly one thing: **every claim
in a release can be checked from the release itself.** The chain
below is designed so that validity is established by checking, not by
trust — in the authors, the institution, or the tools that produced
the work. Who or what wrote an artifact is irrelevant to whether it
is correct; the release carries everything needed to decide that.

## The chain

**1. Frozen releases.** Each release is an annotated tag and is never
rewritten. Corrections appear as new releases that name what they
correct.

**2. Complete identity.** Each release ships a manifest recording the
SHA-256 hash of every released file — sources, certificates,
artifacts, and compiled PDFs — together with the exact source commit
and tree identifiers. Anything you rebuild or regenerate can be
compared byte for byte.

**3. Complete proofs.** Every load-bearing theorem is proved in full
in the released paper and its supplement. The submitted pair is
self-contained: verifying the mathematics requires nothing outside
the release.

**4. Executable certificates.** Each quantitative statement is paired
with a standalone program that re-derives the relevant operators from
the definitions printed in the paper — deliberately sharing no
library code with anything else — and replays every certified
identity, probability law, and counterexample in exact rational
arithmetic. Certificates emit canonical, byte-stable artifacts, and
the tests that force regeneration to match ship in the release.
Agreement is therefore evidence about the mathematics, not about a
common implementation. Certificates are supplementary evidence; no
proof relies on them.

**5. Import fidelity.** Where a paper composes previously established
results, an executable audit pins every imported statement to its
source by content hash and byte-compares the quoted text, so no
statement can drift from what it cites.

**6. Independent archival.** Each release is deposited with a
persistent DOI, so the record outlives this hosting platform.

## What this asks of you

Nothing, except to check: rebuild the PDFs from the released sources,
replay the certificates, recompute the hashes, and read the proofs.
[REPRODUCING.md](REPRODUCING.md) gives the commands.

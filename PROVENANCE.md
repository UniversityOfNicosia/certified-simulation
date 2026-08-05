# Provenance

Every result released here follows one pipeline, and this file states
it once so the releases can be checked rather than trusted.

## The research record

Results are developed in an internal research record in which every
load-bearing claim has a written derivation, an explicit declared
scope, and a status ledger. Claims are frozen under content-addressed
tags before review; frozen records are never edited afterwards.

## Adversarial two-model review

The program operates an AI-assisted, human-responsible workflow.
Construction and review are performed by two different AI systems
from different model families in separated roles: a constructor
produces derivations, certificates, and manuscript text; an
independent reviewer examines only frozen, tagged artifacts — never
conversations — and issues written verdicts with findings. Repairs
are bounded and re-reviewed; a claim is promoted only on a passing
verdict, and the review evidence is preserved. The human authors
check the cited sources, the mathematical arguments, the executable
evidence, and the final text, and assume responsibility for all
content, as stated in each paper's acknowledgments.

## Executable certificates

Each quantitative statement is paired with a certificate program
that re-derives the relevant operators from the definitions in the
paper and replays every identity, every probability law, and every
counterexample in exact rational arithmetic — no floating point in
any certified quantity. Certificates emit canonical byte-stable JSON
artifacts whose regeneration is enforced by a permanent test suite.
Certificates are supplementary computational evidence: complete
analytic proofs of every load-bearing theorem are contained in the
released papers and their supplements, and no proof relies on a
certificate.

## Import fidelity

Where a paper composes previously established results, an executable
audit pins every imported statement to its frozen source record by
content hash and byte-compares the quoted statements, so that no
paraphrase can drift from what was reviewed.

## Release anchoring

Each public release tag records, in its manifest and in
[RELEASES.md](RELEASES.md): the internal frozen tag it mirrors, the
exact source commit and tree hashes, and SHA-256 hashes of every
released file, including the compiled PDFs. Releases are archived
with a persistent DOI. Verifying a release therefore requires no
access to the internal record: the sources, certificates, and hashes
in the release are self-contained, and the replay commands in
[REPRODUCING.md](REPRODUCING.md) regenerate every certified artifact.

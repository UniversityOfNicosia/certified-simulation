# Executable certificates

Each script in this directory replays one theorem of the paper in exact
rational arithmetic and writes a canonical JSON certificate to
[`../artifacts/`](../artifacts). The scripts are self-contained, use only
the Python standard library (Python 3.12 or later), take no external input,
and are deterministic: the same script always produces the same bytes, on
any platform.

| Script | Theorem | Certificate |
| --- | --- | --- |
| `certify_finite_horizon_memory.py` | Finite-horizon upwind memory (the rank law for the observability map, the encoder lower bounds, and the flux queue) | `finite-horizon-memory-certificate.json` |
| `certify_queue_conditioning.py` | Conditioning of delayed queue recovery (the collar-to-queue matrix: determinant, explicit inverse, norms, condition number, singular-value bounds) | `queue-conditioning-certificate.json` |
| `certify_dissipative_decay.py` | Dissipative decay under bounded arithmetic perturbations (the dissipation identity, the Chebyshev spectrum identity, the sharp contraction rate, the restriction inequality, mean bookkeeping) | `dissipative-decay-certificate.json` |

## Replaying

From the repository root:

```bash
python papers/finite-horizon-memory/certificates/certify_finite_horizon_memory.py
python papers/finite-horizon-memory/certificates/certify_queue_conditioning.py
python papers/finite-horizon-memory/certificates/certify_dissipative_decay.py
```

Each run prints the number of executed checks, the verdict, and the SHA-256
of the certificate it wrote. A successful replay reproduces the committed
certificate byte for byte; `git status` will show no change. The test suite
in [`../tests/`](../tests) automates exactly this comparison:

```bash
pytest papers/finite-horizon-memory/tests
```

## How to read a certificate

Every certificate carries the same frame:

- `arithmetic` states the number system. All checks run in exact rational
  arithmetic with zero tolerance. Where a quantity is irrational (a
  trigonometric eigenvalue), the certificate encloses it in a rational
  bracket produced by exact Sturm-sequence bisection and certifies both
  sides of the bracket.
- `scope` separates three things honestly: what the paper proves
  analytically, what this certificate executes, and what the executed
  checks do not establish. Finite sweeps detect implementation errors and
  replay the paper's exact values; they do not replace the quantified
  proofs.
- `checks` is the flat list of named boolean checks, `check_count` and
  `passed` count them, and `verdict` is `pass` only if every check passed.
- The remaining fields are fixtures: small exact instances (matrices,
  polynomials, brackets, witnesses) recorded so a reader can inspect
  concrete values without rerunning anything.

Each certificate also contains mutation checks: deliberately wrong variants
of the claims (a dropped sign, a wrong exponent, a wrong subspace) that the
certificate must detect as wrong. A certificate that cannot fail is not
evidence; these rows demonstrate that this one can.

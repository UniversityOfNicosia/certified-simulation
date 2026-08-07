"""Replay tests for the executable certificates.

Each test rebuilds a certificate from scratch with its script and requires
byte identity with the committed artifact, then re-checks a few statements
directly so that a stale artifact and a stale script cannot pass together.
"""

from __future__ import annotations

import importlib.util
import sys
from fractions import Fraction as Q
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parent.parent
CERTIFICATES = PAPER_ROOT / "certificates"
ARTIFACTS = PAPER_ROOT / "artifacts"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, CERTIFICATES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def replay(name: str, artifact: str) -> tuple[object, dict]:
    module = load(name)
    document = module.build_certificate()
    committed = (ARTIFACTS / artifact).read_bytes()
    assert module.canonical_bytes(document) == committed, (
        f"{artifact} does not replay byte-for-byte; regenerate it with "
        f"certificates/{name}.py"
    )
    assert document["verdict"] == "pass"
    assert document["passed"] == document["check_count"]
    return module, document


def test_finite_horizon_memory_certificate_replays() -> None:
    module, document = replay(
        "certify_finite_horizon_memory", "finite-horizon-memory-certificate.json"
    )
    # Rank law spot checks, independent of the certificate dictionary.
    assert module.rank(module.observability(3, 4, 2, Q(1, 2))) == 7
    assert module.expected_rank(3, 4, 2, Q(1, 2)) == 7
    assert module.rank(module.observability(4, 3, 9, Q(1, 3))) == 4 + 3 * 2
    assert module.rank(module.observability(3, 4, 3, Q(0))) == 3
    assert document["check_count"] == 381


def test_queue_conditioning_certificate_replays() -> None:
    module, document = replay(
        "certify_queue_conditioning", "queue-conditioning-certificate.json"
    )
    matrix = module.queue_matrix(3, Q(1, 2))
    assert matrix[0] == (Q(1, 2), Q(0), Q(0))
    assert matrix[1] == (Q(1, 4), Q(1, 4), Q(0))
    assert matrix[2] == (Q(1, 8), Q(1, 4), Q(1, 8))
    assert module.determinant(matrix) == Q(1, 2) ** 6
    inverse = module.stated_inverse(3, Q(1, 2))
    assert module.multiply(inverse, matrix) == module.identity(3)
    assert document["check_count"] == 837


def test_dissipative_decay_certificate_replays() -> None:
    module, document = replay(
        "certify_dissipative_decay", "dissipative-decay-certificate.json"
    )
    size, courant = 5, Q(1, 3)
    update = module.upwind_update(size, courant)
    gram = module.multiply(module.transpose(update), update)
    stated = module.add_scaled(
        module.identity(size), module.cyclic_laplacian(size), -courant * (1 - courant)
    )
    assert gram == stated
    # charpoly(L) for N=4 is x^4 - 8x^3 + 20x^2 - 16x, i.e. x(x-2)^2(x-4).
    charpoly = module.characteristic_polynomial(module.cyclic_laplacian(4))
    assert charpoly == (Q(0), Q(-16), Q(20), Q(-8), Q(1))
    bracket = document["bracket_fixtures"]["N6"]
    low = Q(bracket["smallest_nonzero_eigenvalue_low"])
    high = Q(bracket["smallest_nonzero_eigenvalue_high"])
    assert low < 1 <= high
    assert document["check_count"] == 547

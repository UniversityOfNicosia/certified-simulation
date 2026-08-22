"""Replay tests for the short note's executable certificates.

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


def test_initial_trace_certificate_replays() -> None:
    module, document = replay(
        "certify_initial_trace_insufficiency", "initial-trace-certificate.json"
    )
    # The note's collision, checked directly at lambda = 1/2.
    x_plus = (Q(0), Q(-1), Q(1), Q(0), Q(0), Q(0), Q(0), Q(0))
    x_minus = (Q(0), Q(1), Q(-1), Q(0), Q(0), Q(0), Q(0), Q(0))
    plus = module.parent_means(module.heun_step(x_plus, Q(1, 2)), 2, 4)
    minus = module.parent_means(module.heun_step(x_minus, Q(1, 2)), 2, 4)
    assert plus == (Q(-1, 32), Q(1, 32))
    assert minus == (Q(1, 32), Q(-1, 32))
    # Forward Euler does not separate the pair.
    assert module.parent_means(module.upwind_step(x_plus, Q(1, 2)), 2, 4) == (
        module.parent_means(module.upwind_step(x_minus, Q(1, 2)), 2, 4)
    )
    assert document["check_count"] == 65


def test_entropy_accounting_certificate_replays() -> None:
    module, document = replay(
        "certify_entropy_accounting_separation", "entropy-accounting-certificate.json"
    )
    # A residual-table row recomputed by hand: (a,b) = (1,1).
    table = document["residual_table"]["a=1,b=1"]
    assert table == ["13/24", "13/24", "5/24", "5/24"]
    # The declared entropy register on equal sign traces is sigma/6.
    assert module.DT * module.rusanov_entropy(Q(1), Q(1)) == Q(1, 6)
    assert module.DT * module.rusanov_entropy(Q(-1), Q(-1)) == Q(-1, 6)
    # The P=4 fixture matches the note's table.
    fixture = document["family_fixtures"]["P4"]
    assert fixture["branches"] == 16
    assert fixture["distinct_registers"] == 16
    assert fixture["distinct_divergences"] == 15
    assert fixture["zero_divergence_probability"] == "1/8"
    assert fixture["register_mse_floor"] == "1/9"
    assert fixture["divergence_mse_floor"] == "2/9"
    assert document["check_count"] == 70

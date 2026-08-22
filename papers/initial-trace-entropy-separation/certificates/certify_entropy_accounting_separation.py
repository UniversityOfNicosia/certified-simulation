#!/usr/bin/env python3
"""Exact certificate for the entropy-accounting results of the short note.

This script replays, in exact rational arithmetic, the note's separation
between conservative one-step closure and entropy accounting for the
Burgers--Rusanov setting:

* the four-cell local update and the full residual table of the note's
  lemma, all sixteen entries, with nonnegativity;
* the augmented parent entropy balance as an exact algebraic identity on
  pseudo-random rational states, with the Jensen gap nonnegative;
* the entropy-blind sign family for every P from 2 to 10: all 2^P branches
  share parent means, conservative interface registers, and next parent
  means, while the raw entropy-interface registers distinguish every
  branch; the distinct-value counts, the zero-divergence probability, and
  the mean-squared-error floors of the note's table are recomputed
  independently, in closed form and by full enumeration;
* semantic mutations that a wrong implementation would accept.

The script is self-contained, deterministic, and uses only the Python
standard library.  The finite sweeps detect implementation errors; they do
not replace the note's proofs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
from typing import Any

type Vector = tuple[Q, ...]


def encode(value: Q) -> str:
    return f"{value.numerator}/{value.denominator}"


def flux(u: Q) -> Q:
    return u * u / 2


def entropy(u: Q) -> Q:
    return u * u / 2


def entropy_flux(u: Q) -> Q:
    return u * u * u / 3


def rusanov(left: Q, right: Q) -> Q:
    return (flux(left) + flux(right)) / 2 - (right - left) / 2


def rusanov_entropy(left: Q, right: Q) -> Q:
    return (entropy_flux(left) + entropy_flux(right)) / 2 - (entropy(right) - entropy(left)) / 2


DT = Q(1, 2)  # unit cells, Delta t = 1/2


def forward_euler(cells: Vector) -> Vector:
    size = len(cells)
    fluxes = tuple(rusanov(cells[i], cells[(i + 1) % size]) for i in range(size))
    return tuple(
        cells[i] - DT * (fluxes[i] - fluxes[(i - 1) % size]) for i in range(size)
    )


def build_certificate() -> dict[str, Any]:
    checks: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # The four-cell lemma: update values and the full residual table.
    expected_residuals = {
        (1, 1): (Q(13, 24), Q(13, 24), Q(5, 24), Q(5, 24)),
        (1, -1): (Q(13, 24), Q(1, 2), Q(1, 2), Q(13, 24)),
        (-1, 1): (Q(5, 24), Q(1, 2), Q(1, 2), Q(5, 24)),
        (-1, -1): (Q(5, 24), Q(5, 24), Q(13, 24), Q(13, 24)),
    }
    residual_table: dict[str, list[str]] = {}
    table_matches = True
    for a_int, b_int in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        a, b = Q(a_int), Q(b_int)
        cells = (a, -a, -b, b)
        # Compatible exterior states: a on the left, b on the right.
        extended = (a, *cells, b)
        fluxes = tuple(rusanov(extended[i], extended[i + 1]) for i in range(5))
        updated = tuple(
            extended[i + 1] - DT * (fluxes[i + 1] - fluxes[i]) for i in range(4)
        )
        expected_update = (a / 2, -(a + b) / 4, -(a + b) / 4, b / 2)
        table_matches &= updated == expected_update
        entropy_fluxes = tuple(
            DT * rusanov_entropy(extended[i], extended[i + 1]) for i in range(5)
        )
        residuals = tuple(
            entropy(extended[i + 1])
            - entropy(updated[i])
            - (entropy_fluxes[i + 1] - entropy_fluxes[i])
            for i in range(4)
        )
        table_matches &= residuals == expected_residuals[(a_int, b_int)]
        table_matches &= all(r >= 0 for r in residuals)
        residual_table[f"a={a_int},b={b_int}"] = [encode(r) for r in residuals]
    checks["lemma_update_and_residual_table_exact"] = table_matches

    # For sign states the declared entropy flux reduces to (u_i + u_{i+1})/12.
    reduction = all(
        rusanov_entropy(Q(u), Q(v)) == (Q(u) + Q(v)) / 6
        for u in (-1, 1)
        for v in (-1, 1)
    )
    # With Delta t = 1/2 the integrated register is (u_i + u_{i+1})/12.
    checks["sign_state_entropy_flux_reduction"] = reduction

    # ------------------------------------------------------------------
    # Augmented parent entropy balance: exact identity on rational states.
    identity_holds = True
    jensen_nonneg = True
    for parent_count, child_count in ((2, 4), (3, 4)):
        size = parent_count * child_count
        for seed in range(3):
            cells = tuple(Q((seed * 5 + 2 * i * i + i) % 13 - 6, 7) for i in range(size))
            updated = forward_euler(cells)
            fluxes = tuple(
                DT * rusanov_entropy(cells[i], cells[(i + 1) % size]) for i in range(size)
            )
            residuals = tuple(
                entropy(cells[i])
                - entropy(updated[i])
                - (fluxes[i] - fluxes[(i - 1) % size])
                for i in range(size)
            )
            for k in range(parent_count):
                lo, hi = k * child_count, (k + 1) * child_count
                h_total = Q(child_count)
                mean_before = sum(cells[lo:hi], Q(0)) / child_count
                mean_after = sum(updated[lo:hi], Q(0)) / child_count
                jensen_before = sum(
                    (entropy(cells[i]) for i in range(lo, hi)), Q(0)
                ) / child_count - entropy(mean_before)
                jensen_after = sum(
                    (entropy(updated[i]) for i in range(lo, hi)), Q(0)
                ) / child_count - entropy(mean_after)
                jensen_nonneg &= jensen_before >= 0 and jensen_after >= 0
                dissipation = sum((residuals[i] for i in range(lo, hi)), Q(0))
                right_register = fluxes[hi - 1]
                left_register = fluxes[(lo - 1) % size]
                lhs = entropy(mean_after) + jensen_after + dissipation / h_total
                rhs = entropy(mean_before) + jensen_before - (
                    right_register - left_register
                ) / h_total
                identity_holds &= lhs == rhs
    checks["augmented_parent_balance_identity"] = identity_holds
    checks["jensen_gaps_nonnegative"] = jensen_nonneg

    # ------------------------------------------------------------------
    # The entropy-blind sign family, P = 2..10.
    family_fixtures: dict[str, Any] = {}
    for parent_count in range(2, 11):
        branches = list(product((-1, 1), repeat=parent_count))
        first_means = None
        first_registers = None
        first_next = None
        conservative_identical = True
        entropy_registers: list[tuple[Q, ...]] = []
        divergences: list[tuple[Q, ...]] = []
        register_formula = True
        for sigma in branches:
            cells: list[Q] = []
            for k in range(parent_count):
                prev = Q(sigma[(k - 1) % parent_count])
                cur = Q(sigma[k])
                cells.extend((prev, -prev, -cur, cur))
            state = tuple(cells)
            size = len(state)
            means = tuple(
                sum(state[4 * k : 4 * k + 4], Q(0)) / 4 for k in range(parent_count)
            )
            conservative = tuple(
                DT * rusanov(state[4 * k + 3], state[(4 * k + 4) % size])
                for k in range(parent_count)
            )
            updated = forward_euler(state)
            next_means = tuple(
                sum(updated[4 * k : 4 * k + 4], Q(0)) / 4 for k in range(parent_count)
            )
            if first_means is None:
                first_means, first_registers, first_next = means, conservative, next_means
            conservative_identical &= (
                means == first_means
                and conservative == first_registers
                and next_means == first_next
            )
            register = tuple(
                DT * rusanov_entropy(state[4 * k + 3], state[(4 * k + 4) % size])
                for k in range(parent_count)
            )
            register_formula &= register == tuple(Q(s, 6) for s in sigma)
            entropy_registers.append(register)
            divergences.append(
                tuple(
                    register[k] - register[(k - 1) % parent_count]
                    for k in range(parent_count)
                )
            )
        distinct_registers = len(set(entropy_registers))
        distinct_divergences = len(set(divergences))
        zero_divergence = sum(
            1 for d in divergences if all(v == 0 for v in d)
        )
        # Independent closed forms, then cross-checked against enumeration:
        # registers sigma_K/12 are injective in sigma; the divergence map
        # collapses exactly the two constant sign vectors.
        checks[f"family_conservative_identical_P{parent_count}"] = conservative_identical
        checks[f"family_register_formula_P{parent_count}"] = register_formula
        checks[f"family_registers_distinguish_P{parent_count}"] = (
            distinct_registers == 2**parent_count
        )
        checks[f"family_divergence_count_P{parent_count}"] = (
            distinct_divergences == 2**parent_count - 1
        )
        checks[f"family_zero_divergence_probability_P{parent_count}"] = Q(
            zero_divergence, 2**parent_count
        ) == Q(1, 2 ** (parent_count - 1))
        # Mean-squared-error floors under the uniform branch law: the
        # conservative channel is constant across branches, so the best
        # predictor of either entropy vector is its mean, and the floor is
        # the total variance.
        register_floor = Q(0)
        for k in range(parent_count):
            values = [reg[k] for reg in entropy_registers]
            mean = sum(values, Q(0)) / len(values)
            register_floor += sum(((v - mean) ** 2 for v in values), Q(0)) / len(values)
        divergence_floor = Q(0)
        for k in range(parent_count):
            values = [d[k] for d in divergences]
            mean = sum(values, Q(0)) / len(values)
            divergence_floor += sum(((v - mean) ** 2 for v in values), Q(0)) / len(values)
        checks[f"family_register_mse_floor_P{parent_count}"] = register_floor == Q(
            parent_count, 36
        )
        checks[f"family_divergence_mse_floor_P{parent_count}"] = divergence_floor == Q(
            parent_count, 18
        )
        if parent_count <= 4:
            family_fixtures[f"P{parent_count}"] = {
                "branches": 2**parent_count,
                "distinct_registers": distinct_registers,
                "distinct_divergences": distinct_divergences,
                "zero_divergence_probability": encode(
                    Q(zero_divergence, 2**parent_count)
                ),
                "register_mse_floor": encode(register_floor),
                "divergence_mse_floor": encode(divergence_floor),
            }

    # ------------------------------------------------------------------
    # Semantic mutations.
    a, b = Q(1), Q(1)
    extended = (a, a, -a, -b, b, b)
    fluxes = tuple(rusanov(extended[i], extended[i + 1]) for i in range(5))
    updated = tuple(extended[i + 1] - DT * (fluxes[i + 1] - fluxes[i]) for i in range(4))
    wrong_update = (a / 2, (-a + b) / 4, -(a + b) / 4, b / 2)
    checks["mutation_middle_cell_sign_detected"] = updated != wrong_update
    wrong_registers = all(
        rusanov_entropy(Q(u), Q(v)) == (Q(u) + Q(v)) / 12 for u in (-1, 1) for v in (-1, 1)
    )
    checks["mutation_register_normalization_detected"] = not wrong_registers
    # Claiming the conservative register distinguishes branches must fail:
    # for P=2 all four branches share it.
    sigma_states = []
    for sigma in product((-1, 1), repeat=2):
        cells = []
        for k in range(2):
            prev, cur = Q(sigma[(k - 1) % 2]), Q(sigma[k])
            cells.extend((prev, -prev, -cur, cur))
        state = tuple(cells)
        sigma_states.append(
            tuple(DT * rusanov(state[4 * k + 3], state[(4 * k + 4) % 8]) for k in range(2))
        )
    checks["mutation_conservative_register_blind_confirmed"] = len(set(sigma_states)) == 1

    check_count = len(checks)
    return {
        "schema": "certified-simulation/initial-trace-entropy-separation/entropy-accounting/v1",
        "arithmetic": "fractions.Fraction; exact; tolerance zero",
        "results": (
            "The four-cell residual lemma with its full table; the augmented "
            "parent entropy balance as an exact identity; the entropy-blind "
            "Burgers--Rusanov sign family for P=2..10 with register and "
            "divergence counts and mean-squared-error floors"
        ),
        "scope": {
            "analytic": (
                "The note proves the residual lemma, the conditional parent "
                "balance, and the all-P entropy-blindness statement for the "
                "declared flux and entropy-flux pair."
            ),
            "executed": (
                "This certificate recomputes the sixteen residual-table entries, "
                "verifies the parent balance identity on pseudo-random rational "
                "states, enumerates all branches for P=2..10 (up to 1024), and "
                "cross-checks the distinct-value counts, the zero-divergence "
                "probability, and both error floors against independent closed "
                "forms."
            ),
            "exclusion": (
                "Finite sweeps do not prove quantified statements; the proofs in "
                "the note carry the general claims.  The floors are stated for "
                "the declared uniform branch law and the declared registers."
            ),
        },
        "residual_table": residual_table,
        "family_fixtures": family_fixtures,
        "checks": checks,
        "check_count": check_count,
        "passed": sum(checks.values()),
        "verdict": "pass" if all(checks.values()) else "fail",
    }


def canonical_bytes(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
        "utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "artifacts"
        / "entropy-accounting-certificate.json",
    )
    args = parser.parse_args()
    document = build_certificate()
    payload = canonical_bytes(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(
        f"checks={document['passed']}/{document['check_count']} "
        f"verdict={document['verdict']} "
        f"sha256={hashlib.sha256(payload).hexdigest()} "
        f"output={args.output.as_posix()}"
    )


if __name__ == "__main__":
    main()

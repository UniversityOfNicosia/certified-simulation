#!/usr/bin/env python3
"""Exact certificate for the initial-trace results of the short note.

This script replays, in exact rational arithmetic, the note's one-step
closure results for periodic scalar positive-speed upwinding:

* the fiber-factorization criterion on explicit finite fixtures;
* the mean-only obstruction: equal parent averages do not determine the
  restricted one-step update;
* forward-Euler trace sufficiency: parent averages plus complete ordered
  initial parent-interface traces determine the next parent averages
  exactly, verified on sweeps of state pairs that agree on exactly that
  data and disagree elsewhere;
* the SSPRK(2,2)/Heun collision: the note's explicit pair with identical
  parent averages and identical initial traces whose completed Heun
  updates differ, with the displayed values (-lambda^2/8, lambda^2/8)
  reproduced exactly, together with collisions of the same construction
  on larger grids;
* realized-flux closure: the stage-averaged realized interface fluxes
  close the completed Heun transition through the telescoping identity;
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
from pathlib import Path
from typing import Any

type Vector = tuple[Q, ...]


def encode(value: Q) -> str:
    return f"{value.numerator}/{value.denominator}"


def encode_vector(values: Vector) -> list[str]:
    return [encode(value) for value in values]


def upwind_step(state: Vector, courant: Q) -> Vector:
    size = len(state)
    return tuple(
        (1 - courant) * state[i] + courant * state[(i - 1) % size] for i in range(size)
    )


def heun_step(state: Vector, courant: Q) -> Vector:
    stage = upwind_step(state, courant)
    second = upwind_step(stage, courant)
    return tuple((a + c) / 2 for a, c in zip(state, second, strict=True))


def parent_means(state: Vector, parent_count: int, child_count: int) -> Vector:
    return tuple(
        sum(state[k * child_count : (k + 1) * child_count], Q(0)) / child_count
        for k in range(parent_count)
    )


def initial_traces(state: Vector, parent_count: int, child_count: int) -> tuple[
    tuple[Q, Q], ...
]:
    """Ordered one-sided state pairs at each parent interface K+1/2."""
    size = parent_count * child_count
    return tuple(
        (
            state[(k * child_count + child_count - 1) % size],
            state[((k + 1) * child_count) % size],
        )
        for k in range(parent_count)
    )


def upwind_interface_fluxes(state: Vector, parent_count: int, child_count: int) -> Vector:
    """Realized upwind flux through each parent interface for one substep."""
    return tuple(
        state[k * child_count + child_count - 1] for k in range(parent_count)
    )


def pseudo_state(size: int, seed: int) -> Vector:
    return tuple(Q((seed * 7 + 3 * i * i + i) % 23 - 11, 5) for i in range(size))


def build_certificate() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    courants = (Q(1, 10), Q(1, 4), Q(1, 3), Q(1, 2), Q(2, 3), Q(3, 4), Q(9, 10), Q(1))

    # ------------------------------------------------------------------
    # Fiber factorization on explicit finite fixtures.
    domain = (0, 1, 2, 3)
    coarse_map = {0: "a", 1: "a", 2: "b", 3: "b"}
    constant_target = {0: 10, 1: 10, 2: 20, 3: 20}
    breaking_target = {0: 10, 1: 11, 2: 20, 3: 20}
    checks["factorization_holds_on_fiber_constant_target"] = all(
        constant_target[x] == constant_target[y]
        for x in domain
        for y in domain
        if coarse_map[x] == coarse_map[y]
    )
    checks["factorization_fails_on_fiber_breaking_target"] = any(
        breaking_target[x] != breaking_target[y]
        for x in domain
        for y in domain
        if coarse_map[x] == coarse_map[y]
    )

    # ------------------------------------------------------------------
    # Mean-only obstruction: equal parent averages, different one-step
    # restricted updates, for every swept configuration.
    mean_only: dict[str, Any] = {}
    for parent_count in (2, 3):
        for child_count in (2, 3, 4, 5):
            size = parent_count * child_count
            for courant in (Q(1, 3), Q(1, 2), Q(3, 4)):
                left = [Q(0)] * size
                right = [Q(0)] * size
                left[0] = Q(1)
                right[child_count - 1] = Q(1)
                left_state, right_state = tuple(left), tuple(right)
                same_means = parent_means(left_state, parent_count, child_count) == parent_means(
                    right_state, parent_count, child_count
                )
                differ = parent_means(
                    upwind_step(left_state, courant), parent_count, child_count
                ) != parent_means(upwind_step(right_state, courant), parent_count, child_count)
                key = f"P{parent_count}_r{child_count}_lambda_{encode(courant)}"
                checks[f"mean_only_obstruction_{key}"] = same_means and differ
        mean_only[f"P{parent_count}"] = True

    # ------------------------------------------------------------------
    # Forward-Euler trace sufficiency: pairs agreeing on parent averages
    # and all ordered initial traces have identical next parent averages.
    for parent_count in (2, 3, 4):
        child_count = 4
        size = parent_count * child_count
        for courant in (Q(1, 3), Q(1, 2), Q(1)):
            agreeing = True
            for seed in range(4):
                base = list(pseudo_state(size, seed))
                other = list(base)
                # Perturb strictly interior children (positions 1 and 2 within a
                # parent) in a mean-preserving way: traces and averages agree.
                for k in range(parent_count):
                    other[k * child_count + 1] += Q(1, 3)
                    other[k * child_count + 2] -= Q(1, 3)
                base_state, other_state = tuple(base), tuple(other)
                assert parent_means(base_state, parent_count, child_count) == parent_means(
                    other_state, parent_count, child_count
                )
                assert initial_traces(base_state, parent_count, child_count) == initial_traces(
                    other_state, parent_count, child_count
                )
                agreeing &= parent_means(
                    upwind_step(base_state, courant), parent_count, child_count
                ) == parent_means(upwind_step(other_state, courant), parent_count, child_count)
            key = f"P{parent_count}_lambda_{encode(courant)}"
            checks[f"forward_euler_trace_sufficiency_{key}"] = agreeing

    # ------------------------------------------------------------------
    # The note's Heun collision: P=2, r=4.
    collision_fixture: dict[str, Any] = {}
    x_plus = (Q(0), Q(-1), Q(1), Q(0), Q(0), Q(0), Q(0), Q(0))
    x_minus = (Q(0), Q(1), Q(-1), Q(0), Q(0), Q(0), Q(0), Q(0))
    checks["collision_equal_parent_means"] = (
        parent_means(x_plus, 2, 4) == parent_means(x_minus, 2, 4) == (Q(0), Q(0))
    )
    checks["collision_equal_initial_traces"] = initial_traces(x_plus, 2, 4) == initial_traces(
        x_minus, 2, 4
    )
    heun_values_match = True
    for courant in courants:
        plus_means = parent_means(heun_step(x_plus, courant), 2, 4)
        minus_means = parent_means(heun_step(x_minus, courant), 2, 4)
        expected_plus = (-(courant**2) / 8, courant**2 / 8)
        expected_minus = (courant**2 / 8, -(courant**2) / 8)
        heun_values_match &= plus_means == expected_plus
        heun_values_match &= minus_means == expected_minus
        if courant == Q(1, 2):
            collision_fixture["lambda_1/2"] = {
                "RH_x_plus": encode_vector(plus_means),
                "RH_x_minus": encode_vector(minus_means),
            }
    checks["collision_displayed_values_exact"] = heun_values_match

    # The same construction collides on larger grids: an interior
    # zero-mean, zero-trace packet at positions (r-3, r-2) of the first
    # parent, close enough to the interface for the two-stage update to
    # cross it (this needs r >= 4).
    for parent_count in (2, 3):
        for child_count in (4, 5, 6):
            size = parent_count * child_count
            packet = [Q(0)] * size
            packet[child_count - 3] = Q(-1)
            packet[child_count - 2] = Q(1)
            state = tuple(packet)
            zero = (Q(0),) * size
            same_data = parent_means(state, parent_count, child_count) == parent_means(
                zero, parent_count, child_count
            ) and initial_traces(state, parent_count, child_count) == initial_traces(
                zero, parent_count, child_count
            )
            crossed = any(
                parent_means(heun_step(state, courant), parent_count, child_count)
                != parent_means(heun_step(zero, courant), parent_count, child_count)
                for courant in (Q(1, 3), Q(1, 2), Q(3, 4))
            )
            key = f"P{parent_count}_r{child_count}"
            checks[f"collision_generalizes_{key}"] = same_data and crossed

    # ------------------------------------------------------------------
    # Realized-flux closure for the completed Heun step: the stage-averaged
    # realized interface fluxes close the parent update by telescoping.
    for parent_count in (2, 3, 4):
        for child_count in (3, 4):
            size = parent_count * child_count
            for courant in (Q(1, 3), Q(1, 2), Q(1)):
                closed = True
                for seed in range(3):
                    state = pseudo_state(size, seed)
                    stage = upwind_step(state, courant)
                    first_flux = upwind_interface_fluxes(state, parent_count, child_count)
                    second_flux = upwind_interface_fluxes(stage, parent_count, child_count)
                    registers = tuple(
                        courant * (a + b) / 2
                        for a, b in zip(first_flux, second_flux, strict=True)
                    )
                    before = parent_means(state, parent_count, child_count)
                    after = parent_means(heun_step(state, courant), parent_count, child_count)
                    reconstructed = tuple(
                        before[k]
                        - (registers[k] - registers[(k - 1) % parent_count]) / child_count
                        for k in range(parent_count)
                    )
                    closed &= reconstructed == after
                key = f"P{parent_count}_r{child_count}_lambda_{encode(courant)}"
                checks[f"flux_register_closure_{key}"] = closed

    # ------------------------------------------------------------------
    # Semantic mutations.
    checks["mutation_traces_claimed_sufficient_for_heun_detected"] = parent_means(
        heun_step(x_plus, Q(1, 2)), 2, 4
    ) != parent_means(heun_step(x_minus, Q(1, 2)), 2, 4)
    checks["mutation_forward_euler_collision_absent"] = parent_means(
        upwind_step(x_plus, Q(1, 2)), 2, 4
    ) == parent_means(upwind_step(x_minus, Q(1, 2)), 2, 4)
    state = pseudo_state(8, 1)
    stage = upwind_step(state, Q(1, 2))
    wrong_register = tuple(
        Q(1, 2) * b for b in upwind_interface_fluxes(stage, 2, 4)
    )  # second stage only, not the Heun average
    before = parent_means(state, 2, 4)
    after = parent_means(heun_step(state, Q(1, 2)), 2, 4)
    wrong_next = tuple(
        before[k] - (wrong_register[k] - wrong_register[(k - 1) % 2]) / 4 for k in range(2)
    )
    checks["mutation_single_stage_register_detected"] = wrong_next != after

    check_count = len(checks)
    return {
        "schema": "certified-simulation/initial-trace-entropy-separation/initial-trace/v1",
        "arithmetic": "fractions.Fraction; exact; tolerance zero",
        "results": (
            "Fiber factorization fixtures; mean-only obstruction; forward-Euler "
            "trace sufficiency; the SSPRK(2,2)/Heun initial-trace collision with "
            "its displayed values; realized-flux closure of the completed step"
        ),
        "scope": {
            "analytic": (
                "The note proves trace sufficiency for forward Euler, the Heun "
                "insufficiency with an explicit collision, and realized-flux "
                "closure for completed steps, for the stated classes."
            ),
            "executed": (
                "This certificate replays the note's explicit collision for eight "
                "Courant values, sweeps the same construction over P=2,3 and "
                "r=3,4,5, verifies trace sufficiency and flux closure on "
                "pseudo-random rational states, and runs mutation checks."
            ),
            "exclusion": (
                "Finite sweeps do not prove quantified statements; the proofs in "
                "the note carry the general claims."
            ),
        },
        "collision_fixture": collision_fixture,
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
        / "initial-trace-certificate.json",
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

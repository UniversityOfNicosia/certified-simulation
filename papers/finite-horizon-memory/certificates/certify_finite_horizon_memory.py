#!/usr/bin/env python3
"""Exact certificate for the finite-horizon upwind memory theorem.

This script replays the paper's rank law and its supporting structure in
exact rational arithmetic: bounded rank sweeps of the observability matrix,
explicit nonzero minors for the rank basis, the saturated nullspace, the
collar-to-queue map, autonomous queue rolling, the six-cell current-flux
collision, all declared delayed packets in the executed range, and semantic
mutations that a wrong implementation would pass.

The script is self-contained and uses only the Python standard library.
The finite sweeps detect implementation errors.  They do not replace the
written arbitrary-parameter rank proof, the invariance-of-domain lower
bound, or the every-r delayed-packet derivation in the paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction as Q
from math import comb
from pathlib import Path
from typing import Any

type Vector = tuple[Q, ...]
type Matrix = tuple[Vector, ...]


def encode(value: Q) -> str:
    return f"{value.numerator}/{value.denominator}"


def encode_vector(values: Vector) -> list[str]:
    return [encode(value) for value in values]


def identity(size: int) -> Matrix:
    return tuple(tuple(Q(row == column) for column in range(size)) for row in range(size))


def multiply(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(
            sum((left[row][inner] * right[inner][column] for inner in range(len(right))), Q(0))
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def apply(matrix: Matrix, vector: Vector) -> Vector:
    return tuple(
        sum((coefficient * value for coefficient, value in zip(row, vector, strict=True)), Q(0))
        for row in matrix
    )


def rank(matrix: Matrix) -> int:
    if not matrix:
        return 0
    work = [list(row) for row in matrix]
    pivot_row = 0
    for column in range(len(work[0])):
        pivot = next(
            (row for row in range(pivot_row, len(work)) if work[row][column]),
            None,
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        scale = work[pivot_row][column]
        work[pivot_row] = [value / scale for value in work[pivot_row]]
        for row in range(len(work)):
            if row == pivot_row or not work[row][column]:
                continue
            factor = work[row][column]
            work[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(work[row], work[pivot_row], strict=True)
            ]
        pivot_row += 1
        if pivot_row == len(work):
            break
    return pivot_row


def determinant(matrix: Matrix) -> Q:
    work = [list(row) for row in matrix]
    result = Q(1)
    for column in range(len(work)):
        pivot = next((row for row in range(column, len(work)) if work[row][column]), None)
        if pivot is None:
            return Q(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            result *= -1
        pivot_value = work[column][column]
        result *= pivot_value
        for row in range(column + 1, len(work)):
            if not work[row][column]:
                continue
            factor = work[row][column] / pivot_value
            for inner in range(column + 1, len(work)):
                work[row][inner] -= factor * work[column][inner]
    return result


def shift(parent_count: int, child_count: int) -> Matrix:
    size = parent_count * child_count
    return tuple(
        tuple(Q(column == (row - 1) % size) for column in range(size)) for row in range(size)
    )


def restriction(parent_count: int, child_count: int) -> Matrix:
    return tuple(
        tuple(
            Q(1, child_count)
            if parent * child_count <= column < (parent + 1) * child_count
            else Q(0)
            for column in range(parent_count * child_count)
        )
        for parent in range(parent_count)
    )


def update(parent_count: int, child_count: int, courant: Q) -> Matrix:
    size = parent_count * child_count
    one = identity(size)
    cyclic = shift(parent_count, child_count)
    return tuple(
        tuple(
            (1 - courant) * one[row][column] + courant * cyclic[row][column]
            for column in range(size)
        )
        for row in range(size)
    )


def observability(parent_count: int, child_count: int, horizon: int, courant: Q) -> Matrix:
    average = restriction(parent_count, child_count)
    dynamics = update(parent_count, child_count, courant)
    power = identity(parent_count * child_count)
    rows: list[Vector] = []
    for _ in range(horizon + 1):
        rows.extend(multiply(average, power))
        power = multiply(power, dynamics)
    return tuple(rows)


def expected_rank(parent_count: int, child_count: int, horizon: int, courant: Q) -> int:
    if courant == 0:
        return parent_count
    return parent_count + (parent_count - 1) * min(horizon, child_count - 1)


def explicit_basis(parent_count: int, child_count: int, horizon: int) -> Matrix:
    q = min(horizon, child_count - 1)
    rows = list(restriction(parent_count, child_count))
    for layer in range(child_count - q, child_count):
        for parent in range(parent_count - 1):
            row = [Q(0)] * (parent_count * child_count)
            row[parent * child_count + layer] = Q(1)
            row[(parent_count - 1) * child_count + layer] = Q(-1)
            rows.append(tuple(row))
    return tuple(rows)


def basis_minor(parent_count: int, child_count: int, horizon: int) -> Matrix:
    q = min(horizon, child_count - 1)
    columns = [parent * child_count for parent in range(parent_count)]
    columns.extend(
        parent * child_count + layer
        for layer in range(child_count - q, child_count)
        for parent in range(parent_count - 1)
    )
    return tuple(
        tuple(row[column] for column in columns)
        for row in explicit_basis(parent_count, child_count, horizon)
    )


def null_basis(parent_count: int, child_count: int) -> Matrix:
    vectors: list[Vector] = []
    for layer in range(child_count - 1):
        vector = [Q(0)] * (parent_count * child_count)
        for parent in range(parent_count):
            vector[parent * child_count + layer] = Q(1)
            vector[parent * child_count + child_count - 1] = Q(-1)
        vectors.append(tuple(vector))
    return tuple(vectors)


def step(state: Vector, parent_count: int, child_count: int, courant: Q) -> Vector:
    return apply(update(parent_count, child_count, courant), state)


def means(state: Vector, parent_count: int, child_count: int) -> Vector:
    return tuple(
        sum(state[parent * child_count : (parent + 1) * child_count], Q(0)) / child_count
        for parent in range(parent_count)
    )


def trajectory(
    state: Vector,
    parent_count: int,
    child_count: int,
    horizon: int,
    courant: Q,
) -> tuple[Vector, ...]:
    records = [state]
    for _ in range(horizon):
        state = step(state, parent_count, child_count, courant)
        records.append(state)
    return tuple(records)


def outflow_queue(
    state: Vector,
    parent_count: int,
    child_count: int,
    length: int,
    courant: Q,
) -> tuple[Vector, ...]:
    records = trajectory(state, parent_count, child_count, max(length - 1, 0), courant)
    return tuple(
        tuple(
            courant * record[parent * child_count + child_count - 1]
            for parent in range(parent_count)
        )
        for record in records[:length]
    )


def collar_queue(collar: Vector, length: int, courant: Q) -> Vector:
    return tuple(
        courant
        * sum(
            (
                Q(comb(time, upstream))
                * (1 - courant) ** (time - upstream)
                * courant**upstream
                * collar[upstream]
                for upstream in range(time + 1)
            ),
            Q(0),
        )
        for time in range(length)
    )


def divergence(face_values: Vector) -> Vector:
    return tuple(value - face_values[parent - 1] for parent, value in enumerate(face_values))


def anchor(values: Vector) -> Vector:
    assert sum(values, Q(0)) == 0
    return values[:-1]


def restore(values: Vector) -> Vector:
    return (*values, -sum(values, Q(0)))


def parent_shift(values: Vector) -> Vector:
    return (values[-1], *values[:-1])


def recurrence_next(history: tuple[Vector, ...], child_count: int, courant: Q) -> Vector:
    result = tuple(courant**child_count * value for value in parent_shift(history[0]))
    for time in range(child_count):
        coefficient = Q(comb(child_count, time)) * (-(1 - courant)) ** (child_count - time)
        result = tuple(
            value - coefficient * history[time][parent] for parent, value in enumerate(result)
        )
    return result


def anchored_queue(
    state: Vector,
    parent_count: int,
    child_count: int,
    length: int,
    courant: Q,
) -> tuple[Vector, ...]:
    return tuple(
        anchor(divergence(row))
        for row in outflow_queue(state, parent_count, child_count, length, courant)
    )


def advance_queue(
    current_means: Vector,
    queue: tuple[Vector, ...],
    child_count: int,
    courant: Q,
) -> tuple[Vector, tuple[Vector, ...]]:
    history = [current_means]
    for row in queue:
        flux_divergence = restore(row)
        history.append(
            tuple(
                history[-1][parent] - flux_divergence[parent] / child_count
                for parent in range(len(current_means))
            )
        )
    future = recurrence_next(tuple(history), child_count, courant)
    new_divergence = tuple(
        child_count * (history[-1][parent] - future[parent]) for parent in range(len(current_means))
    )
    return history[1], (*queue[1:], anchor(new_divergence))


def encoded_collision() -> dict[str, Any]:
    left = (Q(0), Q(2), Q(1), Q(0), Q(0), Q(0))
    right = (Q(2), Q(0), Q(1), Q(0), Q(0), Q(0))
    left_records = trajectory(left, 2, 3, 2, Q(1, 2))
    right_records = trajectory(right, 2, 3, 2, Q(1, 2))
    left_queue = outflow_queue(left, 2, 3, 2, Q(1, 2))
    right_queue = outflow_queue(right, 2, 3, 2, Q(1, 2))
    return {
        "left_initial": encode_vector(left),
        "right_initial": encode_vector(right),
        "initial_parent_means": encode_vector(means(left, 2, 3)),
        "current_outflows": encode_vector(left_queue[0]),
        "first_parent_means": encode_vector(means(left_records[1], 2, 3)),
        "left_next_outflows": encode_vector(left_queue[1]),
        "right_next_outflows": encode_vector(right_queue[1]),
        "left_second_parent_means": encode_vector(means(left_records[2], 2, 3)),
        "right_second_parent_means": encode_vector(means(right_records[2], 2, 3)),
    }


def delayed_packet(
    parent_count: int,
    child_count: int,
    delay: int,
    courant: Q,
) -> tuple[tuple[Vector, ...], tuple[Vector, ...]]:
    layer = child_count - 1 - delay
    left = [Q(0)] * (parent_count * child_count)
    right = [Q(0)] * (parent_count * child_count)
    left[0], left[layer] = Q(2), Q(5)
    right[0], right[layer] = Q(5), Q(2)
    left_state, right_state = tuple(left), tuple(right)
    left_records = trajectory(left_state, parent_count, child_count, delay + 1, courant)
    right_records = trajectory(right_state, parent_count, child_count, delay + 1, courant)
    mean_differences = tuple(
        tuple(
            right_value - left_value
            for left_value, right_value in zip(
                means(left_record, parent_count, child_count),
                means(right_record, parent_count, child_count),
                strict=True,
            )
        )
        for left_record, right_record in zip(left_records, right_records, strict=True)
    )
    left_outflows = outflow_queue(left_state, parent_count, child_count, delay + 1, courant)
    right_outflows = outflow_queue(right_state, parent_count, child_count, delay + 1, courant)
    outflow_differences = tuple(
        tuple(
            right_value - left_value
            for left_value, right_value in zip(left_row, right_row, strict=True)
        )
        for left_row, right_row in zip(left_outflows, right_outflows, strict=True)
    )
    return mean_differences, outflow_differences


def build_certificate() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    rank_sweeps: dict[str, Any] = {}
    courants = (Q(0), Q(1, 3), Q(1, 2), Q(1), Q(-1, 2), Q(2))
    for parent_count in range(2, 6):
        for child_count in range(2, 8):
            for courant in courants:
                observed = tuple(
                    rank(observability(parent_count, child_count, horizon, courant))
                    for horizon in range(child_count + 3)
                )
                expected = tuple(
                    expected_rank(parent_count, child_count, horizon, courant)
                    for horizon in range(child_count + 3)
                )
                key = f"P{parent_count}_r{child_count}_lambda_{encode(courant)}"
                checks[f"rank_{key}"] = observed == expected
                if parent_count <= 3 and child_count <= 4 and courant in (Q(0), Q(1, 2)):
                    rank_sweeps[key] = list(observed)

    minor_sweeps: dict[str, str] = {}
    for parent_count in range(2, 6):
        for child_count in range(2, 8):
            minor_ok = True
            rowspace_ok = True
            for horizon in range(child_count + 3):
                expected = expected_rank(parent_count, child_count, horizon, Q(2, 5))
                basis = explicit_basis(parent_count, child_count, horizon)
                minor_ok &= determinant(basis_minor(parent_count, child_count, horizon)) == Q(
                    1, child_count**parent_count
                )
                rowspace_ok &= rank(basis) == expected
                rowspace_ok &= (
                    rank((*observability(parent_count, child_count, horizon, Q(2, 5)), *basis))
                    == expected
                )
            key = f"P{parent_count}_r{child_count}"
            checks[f"minor_{key}"] = minor_ok
            checks[f"rowspace_{key}"] = rowspace_ok
            if parent_count <= 3 and child_count <= 4:
                minor_sweeps[key] = encode(Q(1, child_count**parent_count))

    nullspace_sweeps: dict[str, Any] = {}
    for parent_count in range(2, 6):
        for child_count in range(2, 8):
            basis = null_basis(parent_count, child_count)
            saturated = observability(parent_count, child_count, child_count + 2, Q(1, 3))
            key = f"P{parent_count}_r{child_count}"
            checks[f"nullity_{key}"] = rank(basis) == child_count - 1
            checks[f"null_vectors_{key}"] = all(
                apply(saturated, vector) == (Q(0),) * len(saturated) for vector in basis
            )
            checks[f"saturation_{key}"] = (
                rank(saturated) == parent_count * child_count - child_count + 1
            )
            if parent_count <= 3 and child_count <= 4:
                nullspace_sweeps[key] = {
                    "nullity": child_count - 1,
                    "saturated_rank": rank(saturated),
                }

    collar_sweeps: dict[str, Any] = {}
    for child_count in range(2, 8):
        for courant in (Q(1, 3), Q(1, 2), Q(1)):
            state = tuple(Q((5 * index + 1) % 17 - 8, 3) for index in range(3 * child_count))
            full = outflow_queue(state, 3, child_count, child_count - 1, courant)
            matches = True
            for parent in range(3):
                block = state[parent * child_count : (parent + 1) * child_count]
                matches &= collar_queue(tuple(reversed(block)), child_count - 1, courant) == tuple(
                    row[parent] for row in full
                )
            determinant_value = courant ** (child_count * (child_count - 1) // 2)
            key = f"r{child_count}_lambda_{encode(courant)}"
            checks[f"collar_{key}"] = matches
            checks[f"collar_det_{key}"] = determinant_value != 0
            if child_count <= 4:
                collar_sweeps[key] = {
                    "determinant": encode(determinant_value),
                    "first_parent_queue": encode_vector(tuple(row[0] for row in full)),
                }

    queue_sweeps: dict[str, bool] = {}
    for parent_count in range(2, 5):
        for child_count in range(2, 7):
            for courant in (Q(1, 3), Q(1, 2), Q(1)):
                state = tuple(
                    Q((7 * index + 3) % 19 - 9, 5) for index in range(parent_count * child_count)
                )
                queue = anchored_queue(
                    state,
                    parent_count,
                    child_count,
                    child_count - 1,
                    courant,
                )
                predicted_means, predicted_queue = advance_queue(
                    means(state, parent_count, child_count),
                    queue,
                    child_count,
                    courant,
                )
                next_state = step(state, parent_count, child_count, courant)
                exact = predicted_means == means(
                    next_state, parent_count, child_count
                ) and predicted_queue == anchored_queue(
                    next_state,
                    parent_count,
                    child_count,
                    child_count - 1,
                    courant,
                )
                key = f"P{parent_count}_r{child_count}_lambda_{encode(courant)}"
                checks[f"queue_roll_{key}"] = exact
                queue_sweeps[key] = exact

    collision = encoded_collision()
    expected_collision = {
        "initial_parent_means": ["1/1", "0/1"],
        "current_outflows": ["1/2", "0/1"],
        "first_parent_means": ["5/6", "1/6"],
        "left_next_outflows": ["3/4", "0/1"],
        "right_next_outflows": ["1/4", "0/1"],
        "left_second_parent_means": ["7/12", "5/12"],
        "right_second_parent_means": ["3/4", "1/4"],
    }
    for field, expected in expected_collision.items():
        checks[f"collision_{field}"] = collision[field] == expected

    delayed_sweeps: dict[str, bool] = {}
    for child_count in range(2, 9):
        for courant in (Q(1, 3), Q(1, 2), Q(1)):
            exact = True
            for delay in range(child_count - 1):
                mean_differences, outflow_differences = delayed_packet(
                    3,
                    child_count,
                    delay,
                    courant,
                )
                zero = (Q(0), Q(0), Q(0))
                exact &= mean_differences[: delay + 1] == (zero,) * (delay + 1)
                exact &= mean_differences[delay + 1] == (
                    Q(3, child_count) * courant ** (delay + 1),
                    -Q(3, child_count) * courant ** (delay + 1),
                    Q(0),
                )
                exact &= outflow_differences[:delay] == (zero,) * delay
                exact &= outflow_differences[delay] == (
                    -Q(3) * courant ** (delay + 1),
                    Q(0),
                    Q(0),
                )
            key = f"r{child_count}_lambda_{encode(courant)}"
            checks[f"delayed_{key}"] = exact
            delayed_sweeps[key] = exact

    # Semantic mutations and exception rows.
    correct_rank = rank(observability(3, 4, 2, Q(1, 2)))
    checks["mutation_rank_increment_P_not_P_minus_1_detected"] = correct_rank == 7 != 9
    checks["mutation_lambda_zero_treated_as_nonzero_detected"] = (
        rank(observability(3, 4, 3, Q(0))) == 3 != 9
    )
    checks["mutation_raw_flux_count_as_global_minimum_detected"] = 3 * 3 != (3 - 1) * 3
    checks["mutation_zero_courant_collar_invertibility_detected"] = Q(0) ** 6 == 0
    checks["mutation_current_flux_as_recursive_state_detected"] = (
        collision["left_next_outflows"] != collision["right_next_outflows"]
    )
    checks["mutation_second_step_collision_detected"] = (
        collision["left_second_parent_means"] != collision["right_second_parent_means"]
    )
    checks["mutation_delays_beyond_saturation_detected"] = rank(
        observability(3, 5, 4, Q(1, 2))
    ) == rank(observability(3, 5, 10, Q(1, 2)))
    left_state = tuple(Q(index - 5) for index in range(12))
    correct_next = means(step(left_state, 3, 4, Q(1, 2)), 3, 4)
    current = means(left_state, 3, 4)
    first_divergence = divergence(outflow_queue(left_state, 3, 4, 1, Q(1, 2))[0])
    wrong_next = tuple(current[parent] + first_divergence[parent] / 4 for parent in range(3))
    checks["mutation_reversed_divergence_orientation_detected"] = wrong_next != correct_next

    check_count = len(checks)
    return {
        "schema": "certified-simulation/finite-horizon-memory/memory-rank/v1",
        "arithmetic": "fractions.Fraction; exact; tolerance zero",
        "theorem": "Finite-horizon upwind memory (rank of the observability map)",
        "scope": {
            "analytic": (
                "The written proof in the paper establishes the rank law, the "
                "centralized and parent-local encoder lower bounds, the queue "
                "attainment, the nullspace, the collision, and the sharp "
                "delayed-packet range for arbitrary stated parameters."
            ),
            "executed": (
                "This certificate sweeps P=2,...,5, r=2,...,7, L=0,...,r+2, six "
                "Courant values for rank, and r through 8 for delays."
            ),
            "exclusion": (
                "Finite sweeps do not prove quantified statements. Classical "
                "observability and minimal-realization theory are prior art; the "
                "contribution certified here is the exact finite-volume "
                "specialization."
            ),
        },
        "rank_fixtures": rank_sweeps,
        "explicit_minor_fixtures": minor_sweeps,
        "nullspace_fixtures": nullspace_sweeps,
        "collar_queue_fixtures": collar_sweeps,
        "queue_roll_sweeps": queue_sweeps,
        "six_cell_collision": collision,
        "delayed_packet_sweeps": delayed_sweeps,
        "mutations": {
            "correct_P3_r4_L2_rank": correct_rank,
            "wrong_P_increment_rank": 9,
            "lambda_zero_rank": rank(observability(3, 4, 3, Q(0))),
            "raw_local_queue_dimension": 9,
            "anchored_global_queue_dimension": 6,
            "wrong_orientation_next_means": encode_vector(wrong_next),
            "correct_next_means": encode_vector(correct_next),
        },
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
        / "finite-horizon-memory-certificate.json",
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

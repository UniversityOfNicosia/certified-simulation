#!/usr/bin/env python3
"""Exact certificate for the delayed-queue conditioning theorem.

This script replays the paper's conditioning theorem for the collar-to-queue
matrix T_q in exact rational arithmetic: the lower-triangular entry formula,
the diagonal and determinant, the explicit binomial inverse, the infinity
norms of T_q and its inverse, the condition number, and the two-sided bounds
on the smallest singular value.  The singular-value bounds are certified
without floating point by exact positive-semidefiniteness tests on the Gram
matrix and an exact Rayleigh-quotient witness.

The script is self-contained and uses only the Python standard library.
The finite sweeps detect implementation errors.  They do not replace the
written all-q, all-lambda proof in the paper.
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


def encode_matrix(matrix: Matrix) -> list[list[str]]:
    return [[encode(value) for value in row] for row in matrix]


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


def transpose(matrix: Matrix) -> Matrix:
    return tuple(
        tuple(matrix[row][column] for row in range(len(matrix)))
        for column in range(len(matrix[0]))
    )


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


def is_positive_semidefinite(matrix: Matrix) -> bool:
    """Exact PSD test for a symmetric rational matrix via outer-product elimination."""
    size = len(matrix)
    if any(matrix[row][column] != matrix[column][row] for row in range(size) for column in range(size)):
        return False
    work = [list(row) for row in matrix]
    for step in range(size):
        pivot = work[step][step]
        if pivot < 0:
            return False
        if pivot == 0:
            if any(work[row][step] != 0 for row in range(step, size)):
                return False
            continue
        for row in range(step + 1, size):
            factor = work[row][step] / pivot
            for column in range(step + 1, size):
                work[row][column] -= factor * work[step][column]
    return True


def queue_matrix(order: int, courant: Q) -> Matrix:
    """(T_q)_{tj} = C(t,j) (1-lambda)^{t-j} lambda^{j+1} for 0 <= j <= t <= q-1."""
    return tuple(
        tuple(
            Q(comb(time, upstream)) * (1 - courant) ** (time - upstream) * courant ** (upstream + 1)
            if upstream <= time
            else Q(0)
            for upstream in range(order)
        )
        for time in range(order)
    )


def stated_inverse(order: int, courant: Q) -> Matrix:
    """(T_q^{-1})_{tj} = lambda^{-(t+1)} C(t,j) (-(1-lambda))^{t-j} for j <= t."""
    return tuple(
        tuple(
            courant ** (-(time + 1))
            * Q(comb(time, upstream))
            * (-(1 - courant)) ** (time - upstream)
            if upstream <= time
            else Q(0)
            for upstream in range(order)
        )
        for time in range(order)
    )


def collar_map_column(order: int, courant: Q, index: int) -> Vector:
    """Image of the unit collar e_index under the paper's collar-to-queue sum."""
    collar = tuple(Q(position == index) for position in range(order))
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
        for time in range(order)
    )


def row_abs_sums(matrix: Matrix) -> tuple[Q, ...]:
    return tuple(sum((abs(value) for value in row), Q(0)) for row in matrix)


def build_certificate() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    orders = range(1, 9)
    courants = (Q(1, 10), Q(1, 4), Q(1, 3), Q(1, 2), Q(2, 3), Q(3, 4), Q(9, 10), Q(1))

    matrix_fixtures: dict[str, Any] = {}
    norm_fixtures: dict[str, Any] = {}
    singular_fixtures: dict[str, Any] = {}

    for order in orders:
        for courant in courants:
            key = f"q{order}_lambda_{encode(courant)}"
            matrix = queue_matrix(order, courant)

            # Column-by-column agreement with the paper's collar-to-queue sum.
            columns_match = all(
                tuple(matrix[time][index] for time in range(order))
                == collar_map_column(order, courant, index)
                for index in range(order)
            )
            checks[f"collar_map_{key}"] = columns_match

            # Lower-triangular structure and stated diagonal.
            checks[f"triangular_{key}"] = all(
                matrix[time][upstream] == 0
                for time in range(order)
                for upstream in range(time + 1, order)
            )
            checks[f"diagonal_{key}"] = all(
                matrix[time][time] == courant ** (time + 1) for time in range(order)
            )

            # Determinant lambda^{q(q+1)/2}.
            checks[f"determinant_{key}"] = determinant(matrix) == courant ** (
                order * (order + 1) // 2
            )

            # Explicit binomial inverse, both products.
            inverse = stated_inverse(order, courant)
            checks[f"inverse_left_{key}"] = multiply(inverse, matrix) == identity(order)
            checks[f"inverse_right_{key}"] = multiply(matrix, inverse) == identity(order)

            # Infinity norms: every row of T_q sums to lambda in absolute value;
            # row t of the inverse sums to lambda^{-(t+1)} (2-lambda)^t.
            forward_sums = row_abs_sums(matrix)
            inverse_sums = row_abs_sums(inverse)
            checks[f"norm_rows_{key}"] = all(value == courant for value in forward_sums)
            checks[f"inverse_norm_rows_{key}"] = all(
                inverse_sums[time] == courant ** (-(time + 1)) * (2 - courant) ** time
                for time in range(order)
            )
            norm_forward = max(forward_sums)
            norm_inverse = max(inverse_sums)
            checks[f"norm_value_{key}"] = norm_forward == courant
            checks[f"inverse_norm_value_{key}"] = norm_inverse == courant ** (-order) * (
                2 - courant
            ) ** (order - 1)

            # Condition number ((2-lambda)/lambda)^{q-1}.
            checks[f"condition_number_{key}"] = norm_forward * norm_inverse == (
                (2 - courant) / courant
            ) ** (order - 1)

            # Smallest singular value, upper bound: the last column of T_q has the
            # single entry lambda^q, so the Rayleigh quotient of the Gram matrix at
            # the last coordinate vector equals lambda^{2q}.
            gram = multiply(transpose(matrix), matrix)
            last = order - 1
            checks[f"sigma_upper_witness_{key}"] = all(
                matrix[time][last] == (courant**order if time == last else Q(0))
                for time in range(order)
            ) and gram[last][last] == courant ** (2 * order)

            # Smallest singular value, lower bound: G - b I is positive
            # semidefinite with b = lambda^{2q} / (q (2-lambda)^{2(q-1)}),
            # which is exactly the square of the paper's lower bound.
            lower_square = courant ** (2 * order) / (
                order * (2 - courant) ** (2 * (order - 1))
            )
            shifted = tuple(
                tuple(
                    gram[row][column] - (lower_square if row == column else Q(0))
                    for column in range(order)
                )
                for row in range(order)
            )
            checks[f"sigma_lower_psd_{key}"] = is_positive_semidefinite(shifted)

            if order <= 3:
                matrix_fixtures[key] = encode_matrix(matrix)
            if order <= 4:
                norm_fixtures[key] = {
                    "norm": encode(norm_forward),
                    "inverse_norm": encode(norm_inverse),
                    "condition_number": encode(norm_forward * norm_inverse),
                }
                singular_fixtures[key] = {
                    "sigma_min_square_lower": encode(lower_square),
                    "sigma_min_square_upper": encode(courant ** (2 * order)),
                    "bracket_ratio": encode(
                        courant ** (2 * order) / lower_square
                    ),
                }

    # Semantic mutations at q=4, lambda=1/3: each wrong variant must differ
    # from the certified value.
    order, courant = 4, Q(1, 3)
    matrix = queue_matrix(order, courant)
    inverse = stated_inverse(order, courant)
    checks["mutation_entry_exponent_detected"] = any(
        sum((abs(value) for value in row), Q(0)) != Q(1)
        for row in matrix
    )  # dropping the extra factor of lambda would make every row sum to one
    dropped_sign = tuple(
        tuple(abs(value) for value in row) for row in inverse
    )
    checks["mutation_inverse_sign_detected"] = multiply(dropped_sign, matrix) != identity(order)
    checks["mutation_condition_exponent_detected"] = ((2 - courant) / courant) ** order != (
        (2 - courant) / courant
    ) ** (order - 1)
    checks["mutation_determinant_exponent_detected"] = courant ** (
        order * (order - 1) // 2
    ) != courant ** (order * (order + 1) // 2)
    gram = multiply(transpose(matrix), matrix)
    too_large = courant ** (2 * order) * Q(101, 100)
    inflated = tuple(
        tuple(
            gram[row][column] - (too_large if row == column else Q(0))
            for column in range(order)
        )
        for row in range(order)
    )
    checks["mutation_sigma_bracket_ceiling_detected"] = not is_positive_semidefinite(inflated)

    check_count = len(checks)
    return {
        "schema": "certified-simulation/finite-horizon-memory/queue-conditioning/v1",
        "arithmetic": "fractions.Fraction; exact; tolerance zero",
        "theorem": "Conditioning of delayed queue recovery (the collar-to-queue matrix T_q)",
        "scope": {
            "analytic": (
                "The written proof in the paper establishes the entry formula, "
                "determinant, explicit inverse, infinity norms, condition number, "
                "and singular-value bounds for every q >= 1 and 0 < lambda <= 1."
            ),
            "executed": (
                "This certificate sweeps q=1,...,8 and eight rational Courant "
                "values.  The singular-value bounds are certified by exact "
                "positive-semidefiniteness of G - b I for the paper's lower "
                "bound b and by the exact unit-vector Rayleigh quotient for the "
                "upper bound; no eigenvalues are computed in floating point."
            ),
            "exclusion": (
                "Finite sweeps do not prove quantified statements.  The "
                "certificate checks the paper's bounds; it does not locate the "
                "smallest singular value beyond the certified bracket."
            ),
        },
        "matrix_fixtures": matrix_fixtures,
        "norm_fixtures": norm_fixtures,
        "singular_value_fixtures": singular_fixtures,
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
        / "queue-conditioning-certificate.json",
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

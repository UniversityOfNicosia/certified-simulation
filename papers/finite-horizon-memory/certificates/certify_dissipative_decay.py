#!/usr/bin/env python3
"""Exact certificate for the dissipative-decay theorem.

This script replays the paper's decay theorem for the periodic upwind update
A = (1-lambda) I + lambda S in exact rational arithmetic:

* the dissipation identity  A^T A = I - lambda (1-lambda) L  with the cyclic
  Laplacian  L = 2I - S - S^T = (I-S)^T (I-S);
* mean preservation in both senses (row and column sums of A equal one);
* the exact spectrum of L through the integer polynomial identity
  charpoly(L) = (-1)^N 2 (T_N(1 - x/2) - 1)  with T_N the Chebyshev
  polynomial, which places the eigenvalues at 2 - 2 cos(2 pi k / N),
  equivalently 4 sin^2(pi k / N);
* a certified rational bracket for the smallest nonzero eigenvalue of L,
  produced by exact Sturm-sequence bisection with no floating point, which
  encloses the paper's sharp rate rho_N^2 = 1 - 4 lambda (1-lambda)
  sin^2(pi/N) between two rationals;
* exact positive-semidefiniteness of  rho_upper^2 I - A^T A  on the
  zero-mean subspace, certifying the contraction at the bracketed rate;
* an exact rational power-iteration witness whose Rayleigh quotient meets
  the bracket from below, certifying sharpness to the stated width;
* the restriction inequality  ||R z - mean(z) 1||_2^2 <= (1/r) ||(I-P_0)
  z||_2^2  as an exact matrix semidefiniteness statement, with an exact
  equality witness showing the constant 1/r is sharp;
* exact mean bookkeeping under per-step perturbations, and the geometric-sum
  identity as a polynomial identity.

The script is self-contained, deterministic, and uses only the Python
standard library.  The finite sweeps detect implementation errors.  They do
not replace the written all-N, all-lambda proof in the paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction as Q
from pathlib import Path
from typing import Any

type Vector = tuple[Q, ...]
type Matrix = tuple[Vector, ...]
type Poly = tuple[Q, ...]  # coefficients, low degree first


def encode(value: Q) -> str:
    return f"{value.numerator}/{value.denominator}"


def encode_poly(poly: Poly) -> list[str]:
    return [encode(value) for value in poly]


# ---------------------------------------------------------------------------
# Exact matrix helpers.


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


def add_scaled(left: Matrix, right: Matrix, scale: Q) -> Matrix:
    return tuple(
        tuple(a + scale * b for a, b in zip(lrow, rrow, strict=True))
        for lrow, rrow in zip(left, right, strict=True)
    )


def apply(matrix: Matrix, vector: Vector) -> Vector:
    return tuple(
        sum((coefficient * value for coefficient, value in zip(row, vector, strict=True)), Q(0))
        for row in matrix
    )


def dot(left: Vector, right: Vector) -> Q:
    return sum((a * b for a, b in zip(left, right, strict=True)), Q(0))


def is_positive_semidefinite(matrix: Matrix) -> bool:
    """Exact PSD test for a symmetric rational matrix via outer-product elimination."""
    size = len(matrix)
    if any(
        matrix[row][column] != matrix[column][row]
        for row in range(size)
        for column in range(size)
    ):
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


def cyclic_shift(size: int) -> Matrix:
    return tuple(
        tuple(Q(column == (row - 1) % size) for column in range(size)) for row in range(size)
    )


def upwind_update(size: int, courant: Q) -> Matrix:
    return add_scaled(
        tuple(tuple((1 - courant) * value for value in row) for row in identity(size)),
        cyclic_shift(size),
        courant,
    )


def cyclic_laplacian(size: int) -> Matrix:
    shift_matrix = cyclic_shift(size)
    two_identity = tuple(tuple(2 * value for value in row) for row in identity(size))
    minus_both = add_scaled(
        add_scaled(two_identity, shift_matrix, Q(-1)), transpose(shift_matrix), Q(-1)
    )
    return minus_both


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


# ---------------------------------------------------------------------------
# Exact polynomial helpers (coefficients low degree first).


def poly_trim(poly: list[Q]) -> Poly:
    while poly and poly[-1] == 0:
        poly.pop()
    return tuple(poly) if poly else (Q(0),)


def poly_add(left: Poly, right: Poly) -> Poly:
    size = max(len(left), len(right))
    return poly_trim(
        [
            (left[index] if index < len(left) else Q(0))
            + (right[index] if index < len(right) else Q(0))
            for index in range(size)
        ]
    )


def poly_scale(poly: Poly, scale: Q) -> Poly:
    return poly_trim([value * scale for value in poly])


def poly_mul(left: Poly, right: Poly) -> Poly:
    result = [Q(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        if a == 0:
            continue
        for j, b in enumerate(right):
            result[i + j] += a * b
    return poly_trim(result)


def poly_divmod(numerator: Poly, denominator: Poly) -> tuple[Poly, Poly]:
    remainder = list(numerator)
    quotient = [Q(0)] * max(len(numerator) - len(denominator) + 1, 1)
    lead = denominator[-1]
    while len(remainder) >= len(denominator) and any(value != 0 for value in remainder):
        shift_amount = len(remainder) - len(denominator)
        factor = remainder[-1] / lead
        quotient[shift_amount] += factor
        for index, value in enumerate(denominator):
            remainder[shift_amount + index] -= factor * value
        while remainder and remainder[-1] == 0:
            remainder.pop()
    return poly_trim(quotient), poly_trim(remainder if remainder else [Q(0)])


def poly_eval(poly: Poly, point: Q) -> Q:
    result = Q(0)
    for value in reversed(poly):
        result = result * point + value
    return result


def poly_derivative(poly: Poly) -> Poly:
    if len(poly) <= 1:
        return (Q(0),)
    return poly_trim([poly[index] * index for index in range(1, len(poly))])


def poly_normalize_positive(poly: Poly) -> Poly:
    lead = poly[-1]
    if lead == 0:
        return poly
    return poly_scale(poly, 1 / abs(lead))


def poly_gcd(left: Poly, right: Poly) -> Poly:
    a, b = left, right
    while b != (Q(0),):
        _, remainder = poly_divmod(a, b)
        a, b = b, poly_normalize_positive(remainder) if remainder != (Q(0),) else (Q(0),)
    return poly_normalize_positive(a)


def square_free_part(poly: Poly) -> Poly:
    gcd = poly_gcd(poly, poly_derivative(poly))
    quotient, remainder = poly_divmod(poly, gcd)
    assert remainder == (Q(0),)
    return poly_normalize_positive(quotient)


def sturm_chain(poly: Poly) -> tuple[Poly, ...]:
    chain: list[Poly] = [poly, poly_derivative(poly)]
    while chain[-1] != (Q(0),) and len(chain[-1]) > 1:
        _, remainder = poly_divmod(chain[-2], chain[-1])
        if remainder == (Q(0),):
            break
        chain.append(poly_normalize_positive(poly_scale(remainder, Q(-1))))
    if chain[-1] == (Q(0),):
        chain.pop()
    return tuple(chain)


def sign_variations(chain: tuple[Poly, ...], point: Q) -> int:
    signs = []
    for member in chain:
        value = poly_eval(member, point)
        if value != 0:
            signs.append(1 if value > 0 else -1)
    return sum(1 for left, right in zip(signs, signs[1:], strict=False) if left != right)


def count_roots_between(chain: tuple[Poly, ...], low: Q, high: Q) -> int:
    """Distinct real roots in (low, high); both endpoints must be non-roots."""
    return sign_variations(chain, low) - sign_variations(chain, high)


def characteristic_polynomial(matrix: Matrix) -> Poly:
    """Monic characteristic polynomial det(xI - M) via Faddeev-LeVerrier."""
    size = len(matrix)
    coefficients = [Q(0)] * (size + 1)
    coefficients[size] = Q(1)
    auxiliary = identity(size)
    for step in range(1, size + 1):
        product = multiply(matrix, auxiliary)
        trace = sum((product[index][index] for index in range(size)), Q(0))
        coefficient = -trace / step
        coefficients[size - step] = coefficient
        auxiliary = add_scaled(product, identity(size), coefficient)
    return poly_trim(coefficients)


def chebyshev_t(order: int) -> Poly:
    previous: Poly = (Q(1),)
    if order == 0:
        return previous
    current: Poly = (Q(0), Q(1))
    for _ in range(order - 1):
        previous, current = current, poly_add(
            poly_scale(poly_mul((Q(0), Q(2)), current), Q(1)), poly_scale(previous, Q(-1))
        )
    return current


def compose(outer: Poly, inner: Poly) -> Poly:
    result: Poly = (Q(0),)
    for coefficient in reversed(outer):
        result = poly_add(poly_mul(result, inner), (coefficient,))
    return result


# ---------------------------------------------------------------------------
# Certified bracket for the smallest nonzero Laplacian eigenvalue.


def smallest_positive_root_bracket(
    chain: tuple[Poly, ...], square_free: Poly
) -> tuple[Q, Q]:
    """Exact Sturm bisection: returns rationals (low, high) with the smallest
    positive root of the square-free polynomial inside (low, high)."""
    low, high = Q(0), Q(9, 2)
    assert poly_eval(square_free, low) != 0
    assert poly_eval(square_free, high) != 0
    assert count_roots_between(chain, low, high) >= 1
    target_width = Q(1, 10**10)
    while high - low > target_width:
        middle = (low + high) / 2
        while poly_eval(square_free, middle) == 0:
            middle += Q(1, 10**15)
        if count_roots_between(chain, low, middle) == 0:
            low = middle
        else:
            high = middle
    return low, high


def zero_mean_basis(size: int) -> tuple[Vector, ...]:
    return tuple(
        tuple(Q(index == position) - Q(position == size - 1) for position in range(size))
        for index in range(size - 1)
    )


def projected_form(matrix: Matrix, basis: tuple[Vector, ...]) -> Matrix:
    images = [apply(matrix, vector) for vector in basis]
    return tuple(
        tuple(dot(basis[row], images[column]) for column in range(len(basis)))
        for row in range(len(basis))
    )


def power_witness(gram: Matrix, size: int, squarings: int) -> Vector:
    """Exact power iteration: apply gram^(2^squarings) to a fixed zero-mean seed."""
    power = gram
    for _ in range(squarings):
        power = multiply(power, power)
    seed = tuple(Q(index == 0) - Q(index == 1) for index in range(size))
    return apply(power, seed)


# ---------------------------------------------------------------------------
# Certificate body.


def build_certificate() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    sizes = range(3, 13)
    courants = (Q(1, 10), Q(1, 4), Q(1, 3), Q(1, 2), Q(2, 3), Q(3, 4), Q(9, 10))
    sharpness_courants = (Q(1, 2), Q(9, 10))

    spectrum_fixtures: dict[str, Any] = {}
    bracket_fixtures: dict[str, Any] = {}
    rate_fixtures: dict[str, Any] = {}

    brackets: dict[int, tuple[Q, Q]] = {}

    for size in sizes:
        shift_matrix = cyclic_shift(size)
        laplacian = cyclic_laplacian(size)
        ones = tuple(Q(1) for _ in range(size))

        # Structural identities that do not depend on lambda.
        checks[f"shift_orthogonal_N{size}"] = multiply(
            transpose(shift_matrix), shift_matrix
        ) == identity(size)
        difference = add_scaled(identity(size), shift_matrix, Q(-1))
        checks[f"laplacian_factorization_N{size}"] = (
            multiply(transpose(difference), difference) == laplacian
        )
        checks[f"laplacian_kernel_N{size}"] = apply(laplacian, ones) == tuple(
            Q(0) for _ in range(size)
        )

        # Spectrum identity: charpoly(L) = (-1)^N 2 (T_N(1 - x/2) - 1).
        charpoly = characteristic_polynomial(laplacian)
        chebyshev = compose(chebyshev_t(size), (Q(1), Q(-1, 2)))
        stated = poly_scale(poly_add(chebyshev, (Q(-1),)), Q(2 * (-1) ** size))
        checks[f"spectrum_identity_N{size}"] = charpoly == stated

        # Certified bracket for the smallest nonzero eigenvalue.
        reduced, remainder = poly_divmod(charpoly, (Q(0), Q(1)))
        checks[f"kernel_simple_N{size}"] = remainder == (Q(0),) and poly_eval(reduced, Q(0)) != 0
        square_free = square_free_part(reduced)
        chain = sturm_chain(square_free)
        low, high = smallest_positive_root_bracket(chain, square_free)
        brackets[size] = (low, high)
        checks[f"bracket_low_N{size}"] = count_roots_between(chain, Q(0), low) == 0
        checks[f"bracket_high_N{size}"] = count_roots_between(chain, Q(0), high) == 1
        checks[f"bracket_below_four_N{size}"] = high < 4

        if size <= 6:
            spectrum_fixtures[f"N{size}"] = encode_poly(charpoly)
        bracket_fixtures[f"N{size}"] = {
            "smallest_nonzero_eigenvalue_low": encode(low),
            "smallest_nonzero_eigenvalue_high": encode(high),
        }

    basis_cache = {size: zero_mean_basis(size) for size in sizes}

    for size in sizes:
        low, high = brackets[size]
        for courant in courants:
            key = f"N{size}_lambda_{encode(courant)}"
            update = upwind_update(size, courant)
            gram = multiply(transpose(update), update)
            laplacian = cyclic_laplacian(size)

            # Dissipation identity A^T A = I - lambda (1-lambda) L.
            stated_gram = add_scaled(identity(size), laplacian, -courant * (1 - courant))
            checks[f"dissipation_identity_{key}"] = gram == stated_gram

            # Mean preservation: rows and columns of A sum to one.
            checks[f"row_sums_{key}"] = all(
                sum(row, Q(0)) == 1 for row in update
            )
            checks[f"column_sums_{key}"] = all(
                sum((update[row][column] for row in range(size)), Q(0)) == 1
                for column in range(size)
            )

            # Contraction at the bracketed rate on the zero-mean subspace:
            # rho_upper^2 = 1 - lambda (1-lambda) low  >=  rho_N^2.
            rho_upper_square = 1 - courant * (1 - courant) * low
            rho_lower_square = 1 - courant * (1 - courant) * high
            checks[f"rate_positive_{key}"] = rho_lower_square > 0
            checks[f"rate_below_one_{key}"] = rho_upper_square < 1
            shifted = add_scaled(
                tuple(
                    tuple(rho_upper_square * value for value in row) for row in identity(size)
                ),
                gram,
                Q(-1),
            )
            projected = projected_form(shifted, basis_cache[size])
            checks[f"contraction_psd_{key}"] = is_positive_semidefinite(projected)

            # Sharpness witness: exact power iteration reaches the bracket.
            if courant in sharpness_courants:
                witness = power_witness(gram, size, 9)
                rayleigh = dot(witness, apply(gram, witness)) / dot(witness, witness)
                gap = rho_upper_square - rayleigh
                checks[f"sharpness_witness_{key}"] = Q(0) <= gap <= Q(1, 10**8)
                if size <= 6 and courant == Q(1, 2):
                    rate_fixtures[key] = {
                        "rho_square_low": encode(rho_lower_square),
                        "rho_square_high": encode(rho_upper_square),
                        "witness_rayleigh_gap": encode(gap),
                    }

    # Restriction inequality: || R z - mean(z) 1 ||^2 <= (1/r) || (I-P0) z ||^2,
    # with exact sharpness witness.
    restriction_fixtures: dict[str, Any] = {}
    for parent_count, child_count in ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2)):
        size = parent_count * child_count
        key = f"P{parent_count}_r{child_count}"
        average = restriction(parent_count, child_count)
        centered = tuple(
            tuple(average[row][column] - Q(1, size) for column in range(size))
            for row in range(parent_count)
        )
        zero_mean_projector = tuple(
            tuple(Q(row == column) - Q(1, size) for column in range(size))
            for row in range(size)
        )
        gap_matrix = add_scaled(
            tuple(
                tuple(Q(1, child_count) * value for value in row)
                for row in zero_mean_projector
            ),
            multiply(transpose(centered), centered),
            Q(-1),
        )
        checks[f"restriction_psd_{key}"] = is_positive_semidefinite(gap_matrix)
        checks[f"restriction_ones_{key}"] = apply(average, tuple(Q(1) for _ in range(size))) == tuple(
            Q(1) for _ in range(parent_count)
        )
        witness_state = tuple(
            Q(1) if index < child_count else (Q(-1) if index < 2 * child_count else Q(0))
            for index in range(size)
        )
        left_side = apply(centered, witness_state)
        right_side = apply(zero_mean_projector, witness_state)
        checks[f"restriction_sharp_{key}"] = dot(left_side, left_side) == Q(
            1, child_count
        ) * dot(right_side, right_side)
        if parent_count <= 3 and child_count <= 3:
            restriction_fixtures[key] = {
                "witness_ratio": encode(Q(1, child_count)),
            }

    # Exact mean bookkeeping under perturbations.
    size = 6
    courant = Q(1, 2)
    update = upwind_update(size, courant)
    state = tuple(Q((3 * index + 1) % 11 - 5, 4) for index in range(size))
    perturbations = tuple(
        tuple(Q((5 * step + 2 * index) % 7 - 3, 9) for index in range(size))
        for step in range(4)
    )
    running = state
    drift = Q(0)
    bookkeeping = True
    for perturbation in perturbations:
        following = tuple(
            value + delta
            for value, delta in zip(apply(update, running), perturbation, strict=True)
        )
        mean_before = sum(running, Q(0)) / size
        mean_after = sum(following, Q(0)) / size
        mean_delta = sum(perturbation, Q(0)) / size
        bookkeeping &= mean_after == mean_before + mean_delta
        drift += mean_delta
        running = following
    checks["mean_bookkeeping_exact"] = bookkeeping
    final_mean = sum(running, Q(0)) / size
    initial_mean = sum(state, Q(0)) / size
    checks["mean_drift_sum"] = final_mean - initial_mean == drift
    step_bound = max(abs(sum(perturbation, Q(0)) / size) for perturbation in perturbations)
    checks["mean_drift_bound"] = abs(drift) <= len(perturbations) * step_bound

    # Geometric-sum identity as a polynomial identity in the contraction rate.
    geometric = True
    for terms in range(1, 31):
        partial: Poly = tuple(Q(1) for _ in range(terms))
        left = poly_mul((Q(1), Q(-1)), partial)
        target = poly_trim([Q(1)] + [Q(0)] * (terms - 1) + [Q(-1)])
        geometric &= left == target
    checks["geometric_sum_identity"] = geometric

    # Semantic mutations.
    size = 5
    courant = Q(1, 2)
    update = upwind_update(size, courant)
    gram = multiply(transpose(update), update)
    laplacian = cyclic_laplacian(size)
    wrong_sign = add_scaled(identity(size), laplacian, courant * (1 - courant))
    checks["mutation_laplacian_sign_detected"] = gram != wrong_sign
    checks["mutation_lambda_one_no_decay_detected"] = multiply(
        transpose(upwind_update(size, Q(1))), upwind_update(size, Q(1))
    ) == identity(size)
    checks["mutation_two_cell_half_courant_rate_zero_detected"] = (
        1 - Q(1, 2) * (1 - Q(1, 2)) * 4 == 0
    )
    # Claiming the rate from the second Fourier mode would be violated by the
    # fundamental mode: at N=8, the second-mode eigenvalue is exactly 2, and
    # the certified witness exceeds the second-mode rate.
    size = 8
    update = upwind_update(size, Q(1, 2))
    gram = multiply(transpose(update), update)
    witness = power_witness(gram, size, 9)
    rayleigh = dot(witness, apply(gram, witness)) / dot(witness, witness)
    second_mode_rate = 1 - Q(1, 2) * (1 - Q(1, 2)) * 2
    checks["mutation_second_mode_rate_detected"] = rayleigh > second_mode_rate
    # Dropping the 1/r factor cannot be strengthened to 1/r^2.
    average = restriction(3, 3)
    centered = tuple(
        tuple(average[row][column] - Q(1, 9) for column in range(9)) for row in range(3)
    )
    zero_mean_projector = tuple(
        tuple(Q(row == column) - Q(1, 9) for column in range(9)) for row in range(9)
    )
    too_strong = add_scaled(
        tuple(tuple(Q(1, 9) * value for value in row) for row in zero_mean_projector),
        multiply(transpose(centered), centered),
        Q(-1),
    )
    checks["mutation_restriction_exponent_detected"] = not is_positive_semidefinite(too_strong)

    check_count = len(checks)
    return {
        "schema": "certified-simulation/finite-horizon-memory/dissipative-decay/v1",
        "arithmetic": (
            "fractions.Fraction; exact; tolerance zero; the only inexact objects "
            "are certified rational brackets of width at most 2e-10 around the "
            "trigonometric eigenvalue, produced by exact Sturm bisection"
        ),
        "theorem": "Dissipative decay under bounded arithmetic perturbations",
        "scope": {
            "analytic": (
                "The written proof in the paper establishes the sharp contraction "
                "rate rho_N on the zero-mean subspace, the perturbed decay bound, "
                "the mean-drift bound, and the coarse-residual bound for every "
                "N >= 3 and 0 < lambda < 1 by a Fourier argument."
            ),
            "executed": (
                "This certificate sweeps N=3,...,12 and seven rational Courant "
                "values.  It verifies the dissipation identity, the Chebyshev "
                "spectrum identity, mean preservation, the restriction "
                "inequality with an exact sharpness witness, mean bookkeeping, "
                "and the geometric-sum identity exactly.  The sharp rate is "
                "enclosed in a certified rational bracket; the contraction is "
                "verified as an exact semidefiniteness statement at the bracket "
                "ceiling, and an exact power-iteration witness meets the "
                "bracket from below within 1e-8."
            ),
            "exclusion": (
                "Finite sweeps do not prove quantified statements.  The "
                "triangle-inequality assembly of the per-step bounds into the "
                "n-step bound is elementary real analysis and is not "
                "re-executed here beyond the exact bookkeeping fixtures."
            ),
        },
        "spectrum_fixtures": spectrum_fixtures,
        "bracket_fixtures": bracket_fixtures,
        "rate_fixtures": rate_fixtures,
        "restriction_fixtures": restriction_fixtures,
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
        / "dissipative-decay-certificate.json",
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

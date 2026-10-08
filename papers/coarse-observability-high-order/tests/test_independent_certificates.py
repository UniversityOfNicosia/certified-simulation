"""Replay tests for the independent certificates of the coarse-observability paper.

Each test rebuilds a certificate from scratch with its script and requires
byte identity with the committed artifact, then re-checks a few statements
directly so that a stale artifact and a stale script cannot pass together.
A third test confirms that every value the certificates transcribe from the
authors' published outputs matches those files byte for byte, and that
article.tex quotes the rebuilt values the certificates verify. It reads the
paper's committed results/ directory, or COARSE_OBSERVABILITY_RESULTS when set,
and skips when neither exists. Rerunning the experiments rewrites the
floating-point CSVs, so run this test on a clean checkout.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import re
import sys
from fractions import Fraction as Q
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CERTIFICATES = ROOT / "certificates"
ARTIFACTS = ROOT / "artifacts"

RANK = "certify_observation_rank_structure"
RANK_ARTIFACT = "observation-rank-structure-certificate.json"
DISSIPATION = "certify_dissipation_conditioning_erasure"
DISSIPATION_ARTIFACT = "dissipation-conditioning-erasure-certificate.json"

PUBLISHED_SHA256 = {
    "local_observability.csv": "659cfc5ad6bb0036a44c296e8ba2e20a3041ac95b41dce7afb5733b71a5413cb",
    "collision_certificate_exact.csv": "b791b63ad3f155f1328bcc4cbff019e8354a9311353961acfe53d10267797a2f",
    "stability_certificate.csv": "592fcd8efb41292c2e8bd75d804f5bbc5e710ce8833c826e1cf375d59aa9c259",
    "erasure_orders.csv": "0915e917d9d76609405d153bb5dc824a6bab605523e4d2d919f0eaaa41d9083b",
    "erasure_rates.csv": "ff162133138324797f4c1da0e14019f549f684d18d6e4aa21e7f3b482cb345b8",
    "erasure_rate_fits.csv": "4a03ecfd3c2b15cef6f1592f47e8de4df5e8701fb1048e4b332543cd6795ac7a",
    "delayed_collisions.csv": "2e51a59ab3718e3458bb81f572401085379dc5fae63408bebce5868f3d0d7c99",
    "queue_conditioning_fits.csv": "a6275640cedf27e0c04e8680b0c763cd06725cacaeda0c89c854d056ca505ecb",
    "queue_conditioning.csv": "8bc0e02d9ec320b12e51a3e45b3a12ee13e4205bacf0ebbb9be0678476c435d0",
    "tables/coefficients.tex": "3772da1610d9001373f4e4766d518ccda0e7a0713a92a0fc40619f6e7df4dd32",
    "tables/queue_exponents.tex": "e72ed5e971b0b4f22fdf8c3ed61da83f1e24db9e22fc92e67aed3a83dc272633",
    "tables/half_life.tex": "7e2dcdda2dacb584c57379a0faa27a6bbb227b4ee2dd7a5c59659f657da0b214",
}

# Values quoted in article.tex (5 October 2026 rebuild) that the certificates verify, with
# the number of times each must occur: E4(c) and the proof of prop:contraction-general,
# E5 in sec:exp-collisions, the E3 convention sentence, and the UB5 retention ratio in
# rem:double-penalty, tab:half-life and sec:discussion.
MANUSCRIPT_QUOTES = {
    r"$87{,}490$": 3,
    r"$87{,}500$": 0,
    r"within $10^{-4}$ of unity at $N=384$ for the six interpolation-based members": 1,
    r"equals $1.0008$ for WENO5-LIN + SSP-RK3": 1,
    r"lies in $[0.9999,1.0009]$ for all schemes and both $\lambda$.": 1,
    r"$-5.999$ for UB5 at $\lambda=0.9$": 1,
    r"($0.99994$ and $-5.9994$)": 1,
    r"($0.989$ and $-6.007$)": 1,
    r"the $12$ (scheme, depth) pairs with $t_0\ge1$": 1,
    r"its Gram matrix has eigenvalues $1$ and $r$": 1,
}

_MODULES: dict[str, object] = {}
_DOCUMENTS: dict[str, dict] = {}


def load(name: str):
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location(name, CERTIFICATES / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULES[name] = module
    return _MODULES[name]


def replay(name: str, artifact: str) -> tuple[object, dict]:
    module = load(name)
    if name not in _DOCUMENTS:
        document = module.build_certificate()
        committed = (ARTIFACTS / artifact).read_bytes()
        assert module.canonical_bytes(document) == committed, (
            f"{artifact} does not replay byte-for-byte; regenerate it with "
            f"certificates/{name}.py"
        )
        assert document["verdict"] == "pass"
        assert document["passed"] == document["check_count"]
        _DOCUMENTS[name] = document
    return module, _DOCUMENTS[name]


def results_directory() -> Path | None:
    candidates = []
    if os.environ.get("COARSE_OBSERVABILITY_RESULTS"):
        candidates.append(Path(os.environ["COARSE_OBSERVABILITY_RESULTS"]))
    candidates.append(ROOT / "results")
    for path in candidates:
        if (path / "local_observability.csv").is_file():
            return path
    return None


def test_observation_rank_structure_replays() -> None:
    module, document = replay(RANK, RANK_ARTIFACT)
    ops = module.build_operators()
    # thm:rank-universal at a small grid, recomputed directly.
    coeffs = module.at_lambda(ops["ub5"], Q(1, 2))
    assert module.observation_rank_ladder(coeffs, 3, 4, 5) == [3, 5, 7, 9, 9, 9]
    assert module.class_node_counts(coeffs, 3, 4) == [1, 4, 4]
    # The general formula where the ladder must drop (central average, P=4, r=4).
    central = {-1: Q(1, 2), 1: Q(1, 2)}
    assert module.class_node_counts(central, 4, 4) == [1, 4, 2, 4]
    assert module.observation_rank_ladder(central, 4, 4, 5) == [4, 7, 9, 11, 11, 11]
    # Local ladders of the composed SSP-RK3 operator.
    rk3 = module.at_lambda(ops["weno5l_rk3"], Q(9, 10))
    assert module.local_rank_ladder(rk3, 8, 6, 3) == [0, 3, 5, 5]
    assert module.local_rank_ladder(rk3, 8, 8, 3) == [0, 3, 7, 7]
    # Coefficients from the Shu-Osher stages reproduce Table S1 corner entries.
    assert ops["weno5l_rk3"][9] == (Q(0), Q(0), Q(0), Q(1, 162000))
    assert ops["weno5l_rk3"][0] == (Q(1), Q(-1, 3), Q(-329, 720), Q(10211, 64800))
    # The worst collision pair is Lax-Wendroff at r = 8.
    worst = document["collision_certificate"]["lw_P8_r8"]["min_bernstein_coefficient"]
    assert worst.startswith("5.6278610712957")
    assert document["collision_cases_found"] == []
    # E5: twelve pairs with t0 >= 1; the UB5 pair jumps to sqrt(2)/512 at its first step.
    assert sum(1 for row in module.AUTHORS_DELAYED_PAIRS if row[1] >= 1) == 12
    _, history = module.first_detection(module.at_lambda(ops["ub5"], Q(1, 2)), 8, 6, 2, 3, 2)
    assert history[0] == [Q(0)] * 8 and sum(x * x for x in history[1]) == Q(1, 131072)
    assert document["detection_delays"]["weno5l_rk3_t0_0"]["first_detection"] == 1
    assert document["check_count"] == len(document["checks"])


def test_dissipation_conditioning_erasure_replays() -> None:
    module, document = replay(DISSIPATION, DISSIPATION_ARTIFACT)
    ops = module.build_operators()
    # prop:dissipation for UB5 recomputed from the moments.
    two_s, coeff = module.dissipative_order(module.defect_series_by_moments(ops["ub5"], 4))
    assert two_s == 6
    assert module.peval(coeff, Q(1, 2)) == Q(5, 512)
    # prop:weno-fe: A = lam^4/12 for the SSP-RK3 pairing.
    two_s, coeff = module.dissipative_order(module.defect_series_by_moments(ops["weno5l_rk3"], 3))
    assert (two_s, coeff) == (4, (Q(0), Q(0), Q(0), Q(0), Q(1, 12)))
    # Item 7: UB5 at lambda = 0.9, N = 384, recomputed directly.
    info = module.contraction_factor(ops["ub5"], Q(9, 10), 384)
    assert info["max_at_k1_strict"]
    d_lo, d_hi = info["defect"]
    rho_lo, rho_hi = module.sqrt_bounds(1 - d_hi, 1 - d_lo)
    th_lo, th_hi = module.theta_one(384)
    a = module.peval(coeff_ub5(module, ops), Q(9, 10))
    ratio_lo = (1 - rho_hi) / (a / 2 * th_hi**6)
    ratio_hi = (1 - rho_lo) / (a / 2 * th_lo**6)
    assert Q(99994210, 10**8) < ratio_lo <= ratio_hi < Q(99994211, 10**8)
    assert document["ub5_lambda_0_9_N384"]["ratio"].startswith("9.9994210497")
    # conj:conditioning exclusion: Smith exponents [1, 1, 3] at L = 1.
    powers = module.laurent_powers(ops["weno5l_rk3"], 1)
    block = module.local_polynomial_blocks(powers, 8, 6, 1)[1]
    reduced = [row for k, row in enumerate(block) if any(row) and k != 0]
    assert module.smith_exponents(reduced, 5, 12) == [1, 1, 3]
    assert document["conditioning_exclusion"]["r8"]["exact_exponent"] == 3
    # Exact exponents equal L_q for every single-stage operator at r = 6.
    for name in module.SINGLE_STAGE:
        for entry in document["conditioning"][f"{name}_P8_r6"]:
            assert entry["exact_exponent"] == entry["L_q"]
    # E3: the Gram matrix I + J of the basis e_j - e_0 (r = 6) is (z - 1)^4 (z - 6).
    gram = [[2 if i == j else 1 for j in range(5)] for i in range(5)]
    assert module.charpoly_int(gram) == [-6, 25, -40, 30, -10, 1]
    # Proof of prop:contraction-general: WENO5-LIN + SSP-RK3 at lambda = 1/2, N = 384 is 1.0008.
    info = module.contraction_factor(ops["weno5l_rk3"], Q(1, 2), 384)
    d_lo, d_hi = info["defect"]
    rho_lo, rho_hi = module.sqrt_bounds(1 - d_hi, 1 - d_lo)
    th_lo, th_hi = module.theta_one(384)
    low = (1 - rho_hi) / (Q(1, 384) * th_hi**4)
    high = (1 - rho_lo) / (Q(1, 384) * th_lo**4)
    assert module.decimal_places(low, 4) == module.decimal_places(high, 4) == Q(10008, 10**4)
    # tab:half-life, rem:double-penalty, sec:discussion: UB5 retains 87,490 times longer.
    assert document["half_life"]["ub5"]["authors_retention"] == 87490
    assert document["half_life"]["ub5"]["retention_vs_uw1"].startswith("8.7490019")
    assert document["check_count"] == len(document["checks"])


def coeff_ub5(module, ops):
    return module.dissipative_order(module.defect_series_by_moments(ops["ub5"], 4))[1]


def test_transcriptions_match_published_outputs() -> None:
    results = results_directory()
    if results is None:
        pytest.skip("the authors' results/ directory is not available")
    for rel, digest in PUBLISHED_SHA256.items():
        assert hashlib.sha256((results / rel).read_bytes()).hexdigest() == digest, rel
    rank = load(RANK)
    diss = load(DISSIPATION)

    coefficients = (results / "tables" / "coefficients.tex").read_text(encoding="utf-8")
    joined = coefficients.splitlines()
    for line in rank.AUTHORS_COEFFICIENT_LINES:
        assert line in joined, line
    # The same polynomials appear verbatim in the manuscript's tables: tab:coeff-sl and
    # Appendix A.2 of article.tex, and Table S1 of supplementary.tex.
    article = (results.parent / "article.tex").read_text(encoding="utf-8")
    supplement = (results.parent / "supplementary.tex").read_text(encoding="utf-8")
    for line in rank.AUTHORS_COEFFICIENT_LINES:
        if line.startswith("c_{"):
            # the article breaks the display differently (\\ and a closing " .")
            entry = line.removesuffix(r",\quad")
            assert entry in article, entry
            continue
        target = supplement if line.startswith("$") else article
        for cell in re.findall(r"\$[^$]*\$", line):
            assert cell in target, cell
    for quote, count in MANUSCRIPT_QUOTES.items():
        assert article.count(quote) == count, quote

    def rows(name: str) -> list[dict[str, str]]:
        with open(results / name, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    ladders: dict[tuple[str, int, str], list] = {}
    for row in rows("local_observability.csv"):
        key = (row["scheme"], int(row["r"]), row["lambda"])
        entry = ladders.setdefault(key, [int(row["m_L"]), int(row["m_R"]), []])
        entry[2].append(int(row["local_rank"]))
    assert {k: (v[0], v[1], tuple(v[2])) for k, v in ladders.items()} == rank.AUTHORS_LOCAL_RANKS

    collision = {
        (row["scheme"], int(row["r"])): (
            int(row["n_pairs"]),
            row["certified"] == "True",
            row["certified_min_q"],
            int(row["max_bernstein_cells"]),
        )
        for row in rows("collision_certificate_exact.csv")
    }
    assert collision == rank.AUTHORS_COLLISION_CERTIFICATE

    delayed = rows("delayed_collisions.csv")
    assert tuple(
        (r["scheme"], int(r["t0"]), int(r["j1"]), int(r["j2"]), int(r["t_detect_predicted"]), int(r["t_detect_measured"]))
        for r in delayed
    ) == rank.AUTHORS_DELAYED_PAIRS

    orders = {
        r["scheme"]: (int(r["two_s"]), r["A_exact"], r["A_at_0.5"], r["A_at_0.9"], r["nondissipative_lambdas_in_(0,1]"])
        for r in rows("erasure_orders.csv")
    }
    assert orders == diss.AUTHORS_EROSION_ORDERS

    stability = {r["object"]: r for r in rows("stability_certificate.csv")}
    for label, (x_box, l_box, cert, bound, cells) in diss.AUTHORS_STABILITY_CERTIFICATE.items():
        r = stability[label]
        assert (Q(r["x_lo"]), Q(r["x_hi"])) == x_box and (Q(r["lam_lo"]), Q(r["lam_hi"])) == l_box
        assert (r["certified"] == "True", r["certified_lower_bound"], int(r["bernstein_cells"])) == (cert, bound, cells)

    rates = {
        (r["scheme"], r["lambda"], int(r["N"])): (r["rho_N"], r["one_minus_rho"], r["ratio"])
        for r in rows("erasure_rates.csv")
    }
    assert rates == diss.AUTHORS_ERASURE_RATES
    fit_rows = rows("erasure_rate_fits.csv")
    slopes = {(r["scheme"], r["lambda"]): r["slope_fit"] for r in fit_rows if r["scheme"] != "weno5l"}
    assert slopes == diss.AUTHORS_ERASURE_SLOPES
    # The fits file repeats the N = 384 ratio of erasure_rates.csv (the 0.989 entry for UB5).
    for r in fit_rows:
        assert r["ratio_at_largest_N"] == rates[(r["scheme"], r["lambda"], 384)][2]

    fits = {(r["scheme"], int(r["L"])): (int(r["q"]), r["gamma_fit"]) for r in rows("queue_conditioning_fits.csv")}
    assert fits == diss.AUTHORS_QUEUE_FITS
    sigma = rows("queue_conditioning.csv")
    lambdas = tuple(r["lambda"] for r in sigma if r["scheme"] == "uw1" and r["L"] == "1")
    assert lambdas == diss.QUEUE_LAMBDAS
    published_sigma = {(r["scheme"], int(r["L"]), r["lambda"]): r["sigma_min_plus"] for r in sigma}
    for key, value in diss.AUTHORS_QUEUE_SIGMA.items():
        assert published_sigma[key] == value, key

    table = (results / "tables" / "queue_exponents.tex").read_text(encoding="utf-8").splitlines()
    parsed = {}
    for line in table:
        if "&" in line:
            cells = [c.strip() for c in line.rstrip("\\").split("&")]
            parsed[cells[0]] = tuple(cells[1:])
    display = {"uw1": "UW1", "lw": "LW", "bw": "BW", "fromm": "FR", "ub3": "UB3", "ub5": "UB5", "weno5l": "WENO5-LIN"}
    assert {k: parsed[v] for k, v in display.items()} == diss.AUTHORS_QUEUE_EXPONENTS

    half = (results / "tables" / "half_life.tex").read_text(encoding="utf-8").splitlines()
    half_rows = {}
    for line in half:
        if "&" in line:
            cells = [c.strip() for c in line.rstrip("\\").split("&")]
            half_rows[cells[0]] = cells
    names = {"uw1": "UW1", "lw": "LW", "bw": "BW", "fromm": "FR", "ub3": "UB3", "ub5": "UB5", "weno5l_rk3": "WENO5-LIN+SSP-RK3"}
    by_scheme = {r["scheme"]: r for r in delayed}
    for key, (table_text, retention, csv_half, csv_rho) in diss.AUTHORS_HALF_LIFE.items():
        cells = half_rows[names[key]]
        mantissa, exponent = table_text.split("e")
        assert cells[3] == f"${mantissa}\\times10^{{{exponent}}}$"
        assert cells[5].strip("$").replace("{,}", "") == str(retention)
        assert (cells[1], cells[2]) == tuple(str(x) for x in rank.AUTHORS_HALF_LIFE_LOCAL_COLUMNS[key])
        assert by_scheme[key]["half_life_predicted"] == csv_half
        assert by_scheme[key]["rho_N"] == csv_rho

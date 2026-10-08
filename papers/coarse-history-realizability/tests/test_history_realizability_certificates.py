"""Replay tests for the independent certificates of the coarse-history lifting note.

Each replay test rebuilds a certificate from scratch with its script and
requires byte identity with the committed artifact, then re-checks a few
statements directly so that a stale artifact and a stale script cannot pass
together.  The transcription tests confirm that every value the certificates
quote matches its source.  manuscript/main.tex is committed and always
checked.  The authors' regenerated outputs in experiments/results/ and
experiments/tables/ are not committed, so that comparison skips when they are
absent; the bibliography comparison skips when manuscript/references.bib is
absent.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import re
import sys
from fractions import Fraction as Q
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CERTIFICATES = ROOT / "certificates"
ARTIFACTS = ROOT / "artifacts"
MANUSCRIPT = ROOT / "manuscript" / "main.tex"
BIBLIOGRAPHY = ROOT / "manuscript" / "references.bib"
RESULTS_CSV = ROOT / "experiments" / "results" / "history_realizability_certificates.csv"
TABLE_FRAGMENT = ROOT / "experiments" / "tables" / "history_realizability_certificates.tex"

SHORT = "certify_short_horizon_lifting"
SHORT_ARTIFACT = "short-horizon-lifting-certificate.json"
LONG = "certify_long_horizon_minimal_r"
LONG_ARTIFACT = "long-horizon-minimal-r-certificate.json"

# The authors' generated outputs, hashed with line endings normalized to LF.
PUBLISHED_SHA256 = {
    RESULTS_CSV: "279b31e3b2f97e5ebc92dd88fc9fbee820487d1d93c5c566f65d0d38a90bdaee",
    TABLE_FRAGMENT: "a89cb83c86f6be64d09ba5513bbc54b364f7a3fa9004c804d3f58744087457e3",
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
        assert document["passed"] == document["check_count"] == len(document["checks"])
        _DOCUMENTS[name] = document
    return module, _DOCUMENTS[name]


def squeeze(text: str) -> str:
    return re.sub(r"\s+", "", text)


def region(text: str, label: str) -> str:
    """Whitespace-free source from \\label{label} to the end of its section."""
    start = text.index(f"\\label{{{label}}}")
    end = text.find("\\section", start)
    return squeeze(text[start : end if end != -1 else len(text)])


def table_cells(text: str) -> list[tuple[str, ...]]:
    """Header and rows of the manuscript's only tabular, cells stripped."""
    body = text[text.index("\\begin{tabular}") : text.index("\\end{tabular}")]
    header = body.split("\\toprule", 1)[1].split("\\midrule", 1)[0]
    rows = body.split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
    return [
        tuple(cell.strip() for cell in row.split("&"))
        for chunk in (header, rows)
        for row in chunk.split("\\\\")
        if row.strip()
    ]


def upwind_step(u: list[Q], lam: Q) -> list[Q]:
    """The displayed update u_i <- (1-lambda) u_i + lambda u_{i-1}, written out again."""
    return [(1 - lam) * u[i] + lam * u[i - 1] for i in range(len(u))]


def averages(u: list[Q], parents: int, children: int) -> tuple[Q, ...]:
    return tuple(sum(u[k * children : (k + 1) * children], Q(0)) / children for k in range(parents))


# ---------------------------------------------------------------------------


def test_short_horizon_certificate_replays() -> None:
    module, document = replay(SHORT, SHORT_ARTIFACT)
    # ex:canonical-lift by hand: x = (2,-2|-2,2) observes (0,0) then (1,-1).
    x = [Q(2), Q(-2), Q(-2), Q(2)]
    assert averages(x, 2, 2) == (0, 0)
    assert averages(upwind_step(x, Q(1, 2)), 2, 2) == (1, -1)
    assert document["example_canonical_lift"]["normalized_lift_x"] == ["2/1", "-2/1", "-2/1", "2/1"]
    # A fresh instance outside the sweep: P=6, r=3, L=2, lambda=2/5.
    result = module.short_case(6, 3, 2, Q(2, 5), 99)
    assert (result["rank"], result["dim_M"], result["nullity"]) == (16, 16, 2)
    assert all(
        result[key]
        for key in ("image_equals_M", "right_inverse", "convention", "unique", "kernel", "fiber_translate")
    )
    # T_{3,1/2} and its binomial inverse, recomputed directly.
    assert document["collar_to_flux"]["T_3_lambda_1/2"] == [
        ["1/2", "0/1", "0/1"],
        ["1/4", "1/4", "0/1"],
        ["1/8", "1/4", "1/8"],
    ]
    assert document["collar_to_flux"]["T_3_lambda_1/2_inverse"][2] == ["2/1", "-8/1", "8/1"]
    # Every swept case obeys the fiber-dimension formula of thm:short-realization.
    for key, case in document["short_horizon_cases"].items():
        parents, children, horizon = (int(v) for v in re.findall(r"\d+", key)[:3])
        assert case["nullity"] == parents * (children - horizon - 1) + horizon, key
        assert case["rank_O"] == case["dim_M"] == parents * (horizon + 1) - horizon, key
    assert document["discrepancies"] == []
    assert document["check_count"] == 175


def test_long_horizon_certificate_replays() -> None:
    module, document = replay(LONG, LONG_ARTIFACT)
    # ex:nonrealizable by hand: y^2 = (1/4) Pi y^0 - (1/4) y^0 + y^1 at r=2, lambda=1/2.
    y0, y1, y2 = (Q(1), Q(1)), (Q(1), Q(1)), (Q(2), Q(0))
    predicted = tuple(Q(1, 4) * y0[k - 1] - Q(1, 4) * y0[k] + y1[k] for k in range(2))
    assert predicted == (1, 1)
    assert tuple(a - b for a, b in zip(y2, predicted, strict=True)) == (1, -1)
    assert document["example_nonrealizable"]["residual_y2_minus_prediction"] == ["1/1", "-1/1"]
    assert document["example_nonrealizable"]["normalized_lift_with_r3"] == [
        "9/1", "-6/1", "0/1", "-3/1", "6/1", "0/1",
    ]
    # A fresh instance: the least r realizing all of M_{5,3} at lambda = 3/7 is 4.
    units = module.UnitHistories()
    dim_m = 5 * 4 - 3
    onto = [module.rank(units.columns(5, r, Q(3, 7), 3)) == dim_m for r in range(1, 6)]
    assert onto == [False, False, False, True, True]
    # ker O_{4,3} at P=5 is N_3 (dimension 2), and generated histories obey the recurrence.
    assert 15 - module.rank(units.columns(5, 3, Q(3, 7), 4)) == 2
    result = module.long_case(5, 3, 4, Q(3, 7), 99, units)
    assert all(result[key] for key in ("image", "kernel", "necessity", "sufficiency", "fiber"))
    assert document["minimal_child_count_by_search"]["P3_lambda_1/2"] == {
        f"L{h}": h + 1 for h in range(6)
    }
    assert document["kernel_ladders_lambda_1/2"]["P3_r4_lambda_1/2"] == [9, 7, 5, 3, 3, 3, 3, 3, 3, 3]
    assert len(document["discrepancies"]) == 1
    assert "history_realizability_certificates.tex" in document["discrepancies"][0]["location"]
    assert document["bibliography"]["missing_entries"] == []
    assert len(document["bibliography"]["findings"]) == 3
    assert document["check_count"] == 162


def test_transcriptions_match_manuscript() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    short, long_ = load(SHORT), load(LONG)
    for module in (short, long_):
        for name, (label, snippet) in module.MANUSCRIPT_FORMULAS.items():
            assert snippet in region(text, label), (module.__name__, name)
    for examples in (short.MANUSCRIPT_EXAMPLE_CANONICAL_LIFT, long_.MANUSCRIPT_EXAMPLE_NONREALIZABLE):
        for name, (label, template, values) in examples.items():
            assert short.fill(template, values) in region(text, label), name
    cells = table_cells(text)
    assert cells == list(long_.MANUSCRIPT_TABLE)
    assert cells[1] == short.MANUSCRIPT_TABLE_ROW_1
    cited: list[str] = []
    for group in re.findall(r"\\cite[pt]?\*?\{([^}]*)\}", text):
        for key in (k.strip() for k in group.split(",")):
            if key not in cited:
                cited.append(key)
    assert tuple(cited) == long_.CITED_KEYS


def test_transcriptions_match_authors_outputs() -> None:
    if not (RESULTS_CSV.is_file() and TABLE_FRAGMENT.is_file()):
        pytest.skip("the authors' regenerated outputs are not present under experiments/")
    for path, digest in PUBLISHED_SHA256.items():
        data = path.read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == digest, path.name
    short, long_ = load(SHORT), load(LONG)
    lines = RESULTS_CSV.read_text(encoding="utf-8").splitlines()
    assert lines == [
        long_.AUTHORS_CSV_HEADER,
        short.AUTHORS_CSV_LINE_1,
        long_.AUTHORS_CSV_LINE_2,
        long_.AUTHORS_CSV_LINE_3,
    ]
    with open(RESULTS_CSV, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["result"] for row in rows] == ["exact lift", "residual zero", "residual (1,-1)"]
    assert rows[0]["details"].endswith("x=(2,-2|-2,2)")
    assert rows[1]["details"].count("Fraction(0, 1)") == 12
    fragment = TABLE_FRAGMENT.read_text(encoding="utf-8")
    assert fragment == long_.AUTHORS_TABLE_FRAGMENT
    assert short.AUTHORS_FRAGMENT_ROW_1 in fragment
    # The recorded discrepancy: no line breaks, rule commands glued to cell text.
    assert "\n" not in fragment
    assert "\\topruleCertificate" in fragment and "\\midruleNormalized" in fragment


def test_transcriptions_match_bibliography() -> None:
    if not BIBLIOGRAPHY.is_file():
        pytest.skip("manuscript/references.bib is not present")
    bib = BIBLIOGRAPHY.read_text(encoding="utf-8")
    long_ = load(LONG)
    assert tuple(re.findall(r"@\w+\{([^,\s]+),", bib)) == long_.BIB_ENTRY_KEYS
    entries = {
        match.group(1): squeeze(match.group(0))
        for match in re.finditer(r"@\w+\{([^,\s]+),.*?(?=\n@|\Z)", bib, flags=re.S)
    }
    for key, fields in long_.BIB_FIELDS.items():
        for field in fields:
            assert field in entries[key], (key, field)
    # The companion note's entry carries no arXiv identifier and the stated status.
    one_step = entries["PolemitisChristakisDrikakis2026OneStep"]
    assert "eprint" not in one_step and "submittedtoAppliedNumericalMathematics" in one_step

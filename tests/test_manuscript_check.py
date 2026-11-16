"""Guard derived manuscript comparisons against missing or misattributed values."""
import ast
import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "manuscript_check", ROOT / "scripts/check_against_manuscript.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

SIXMER = "L29_50M - L1_50M on closed"
EXACT = "exact_1L_mean - exact_16L on closed"


@pytest.fixture
def paper(tmp_path):
    sections = tmp_path / "sections"
    sections.mkdir()
    (sections / "04_part2_6mer.tex").write_text(r"""
On the gut test set, the configurations trained on the 50M-read pool are:
\begin{itemize}
\item \ourmodel{} 29 layers (6.45M parameters): \textbf{69.52\%}
\item NT-v2 500M, 29 layers, LoRA: 67.08\%
\item \ourmodel{} 1 layer (0.9M parameters): 53.92\%
\end{itemize}
""")
    (sections / "05_part3_13mer.tex").write_text(r"""
Accuracies on the gut test set range from 91.15\% for the single-layer
exact-vocabulary \ourmodel{}, through MetaTransformer at 87.46\%.
At sixteen layers, exact $d=64$ reaches 85.63\%, hashed $d=64$ 79.26\%.
""")
    return tmp_path


@pytest.mark.parametrize("comparison,expected", [(SIXMER, "15.60"), (EXACT, "5.52")])
def test_difference_can_be_derived_without_printing_it(paper, comparison, expected):
    assert expected not in checker.flatten(paper)
    calculation = checker.derived_difference(paper, comparison, Decimal(expected))
    assert calculation.endswith("= " + expected)


@pytest.mark.parametrize("mutation", ["shift_both", "swap_models", "duplicate", "comment_only"])
def test_bad_source_is_rejected_even_with_expected_numbers_elsewhere(paper, mutation):
    path = paper / "sections/04_part2_6mer.tex"
    source = path.read_text()
    if mutation == "shift_both":
        # Subtraction alone still gives 15.60; each operand must match its model.
        source = source.replace("69.52", "70.52").replace("53.92", "54.92")
    elif mutation == "swap_models":
        source = source.replace("29 layers", "TEMP").replace("1 layer", "29 layers")
        source = source.replace("TEMP", "1 layer")
    elif mutation == "duplicate":
        source += source
    else:
        source = "\n".join("% " + line for line in source.splitlines())
    path.write_text(source + "\nUnrelated results: 69.52, 53.92, 15.60.\n")
    with pytest.raises(ValueError):
        checker.derived_difference(paper, SIXMER, Decimal("15.60"))


def test_different_pooling_result_is_not_accepted(paper):
    path = paper / "sections/05_part3_13mer.tex"
    path.write_text(path.read_text().replace("91.15", "90.74") + "\nOther result: 91.15.\n")
    with pytest.raises(ValueError, match="disagrees"):
        checker.derived_difference(paper, EXACT, Decimal("5.52"))


def test_missing_source_fails(paper):
    (paper / "sections/04_part2_6mer.tex").unlink()
    with pytest.raises(ValueError, match="missing source file"):
        checker.derived_difference(paper, SIXMER, Decimal("15.60"))


def test_wrong_asserted_difference_fails(paper):
    with pytest.raises(ValueError, match="expected 14.60"):
        checker.derived_difference(paper, SIXMER, Decimal("14.60"))


def test_failed_derivation_cannot_fall_back_to_a_printed_difference(paper, monkeypatch, capsys):
    path = paper / "sections/04_part2_6mer.tex"
    path.write_text(path.read_text().replace("69.52", "68.52") + "\nOld gap: 15.60.\n")
    monkeypatch.setattr(checker, "banner_values", lambda: [(["53.92"], "example accuracy")])
    monkeypatch.setattr(checker, "asserted_deltas", lambda: [
        {"forms": ["15.60"], "where": SIXMER, "expected": Decimal("15.60")}])
    monkeypatch.setattr(sys, "argv", ["check_against_manuscript.py", str(paper)])
    assert checker.main() == 1
    assert "INVALID DERIVATION" in capsys.readouterr().out


def test_all_declared_differences_are_collected_including_negatives():
    source = ast.parse((ROOT / "tests/test_reported_numbers.py").read_text())
    declared = next(ast.literal_eval(node.value) for node in source.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "DELTAS" for t in node.targets))
    actual = checker.asserted_deltas()
    assert [(item["where"], item["expected"]) for item in actual] == [
        (f"{a} - {b} on {metric}", Decimal(str(value)))
        for metric, a, b, value, _ in declared]
    assert {item["expected"] for item in actual if item["expected"] < 0} == {
        Decimal("-0.32"), Decimal("-0.16")}


@pytest.mark.parametrize("value,text,expected", [
    ("0.32", "-0.32", False),
    ("-0.32", "$-0.32$", True),
    ("15.6", "115.60", False),
    ("15.60", "15.600", False),
    ("15.60", "The gap is 15.60.", True),
    ("0.08", "$+0.08$", True),
])
def test_number_presence_preserves_sign_and_precision(value, text, expected):
    assert checker.present(value, text) is expected

#!/usr/bin/env python3
"""Check reported accuracies and differences against a manuscript checkout.

WHY THIS IS A SCRIPT AND NOT A TEST. `tests/test_reported_numbers.py` pins the
config banners against constants transcribed from the manuscript. Those constants
are a frozen copy: if the manuscript changes, the constants do not, and the tests
go on passing. They protect banner-against-constant consistency -- which catches a
number attached to the wrong config -- but they cannot see repo-against-manuscript
drift, because the manuscript is not in this repository and cannot be.

So this is the other half, and it has to be run deliberately, pointed at a
manuscript checkout:

    python scripts/check_against_manuscript.py /path/to/paper

It flattens the LaTeX first. Do not skip that step and do not grep for a phrase
directly: LaTeX sources are hard-wrapped, so a phrase that spans a line break will
report as missing when it is present. Three of eight statements did exactly that
on the first attempt at this check.

What it verifies:
  * every accuracy in a `# RESULT:` banner still appears in the manuscript;
  * every difference `tests/test_reported_numbers.py` asserts is printed or
    supported by an explicitly mapped pair of manuscript results.

Two differences omitted from the condensed manuscript are checked by extracting
the named configurations' accuracies, matching each to its config banner and
subtracting them. These checks run even if the difference appears elsewhere.
Missing, ambiguous or changed source values fail the check.

A difference is accepted at two decimals or rounded to one, because the
manuscript quotes some as ranges -- "+8.9 to +9.4" never contains the literal
"8.94". Requiring two decimals reported three of fifteen as drift when nothing
had drifted.

Other entries use numeric presence, which cannot establish that a number still
means the same thing. The DELTAS table retains the original quoted sources for
manual review. This script checks selected reported results, not every number
in the manuscript.
"""

import argparse
import re
import sys
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

BANNER = re.compile(
    r"^# RESULT: closed-set 100K genus Top-1 (?P<closed>[\d.]+)%"
    r"(?:\s*\|\s*Track A out-of-genome (?P<track_a>[\d.]+)%)?"
    r"(?P<exempt>\s*\[not in the manuscript\])?", re.MULTILINE)


# TikZ coordinate pairs -- "at (14.60,2.90)" in the architecture figure -- are
# numbers that are not figures. Left in, they make any short value look present:
# a mutation of 15.60 to 14.60 was reported as found purely because a node sits at
# x=14.60. Stripped before searching.
TIKZ_COORD = re.compile(r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)")
LATEX_COMMENT = re.compile(r"(?<!\\)%[^\n]*")


def clean_source(text: str) -> str:
    text = LATEX_COMMENT.sub(" ", text)
    text = TIKZ_COORD.sub(" ", text)
    return re.sub(r"\s+", " ", text).replace("−", "-")


def flatten(paper_root: Path) -> str:
    """Concatenate the sources, drop comments and coordinates, collapse whitespace."""
    parts = []
    for pattern in ("main.tex", "supplementary.tex", "sections/*.tex"):
        for f in sorted(paper_root.glob(pattern)):
            parts.append(f.read_text(errors="replace"))
    if not parts:
        raise SystemExit(f"no .tex found under {paper_root}")
    return clean_source("\n".join(parts))


def present(value: str, text: str) -> bool:
    """Is `value` in `text` as a number in its own right?

    Substring matching is not enough: "14.6" occurs inside "114.6" and inside a
    coordinate. Require digit/dot boundaries on both sides.
    """
    return re.search(r"(?<![\d.\-])" + re.escape(value) + r"(?!\d|\.\d)", text) is not None


def banner_values():
    """(accepted forms, config) for every accuracy stamped on a config.

    A banner may end with `[not in the manuscript]`, for a measurement this
    repository reports that the paper does not print -- a configuration dropped when a
    section was rewritten, say. Those are skipped and counted, so the exemption
    cannot grow quietly into a way of silencing real drift.
    """
    out, exempt = [], []
    for cfg in sorted((REPO / "configs").rglob("*.yaml")):
        m = BANNER.search(cfg.read_text())
        if not m:
            continue
        rel = cfg.relative_to(REPO / "configs")
        if m.group("exempt"):
            exempt.append(str(rel))
            continue
        for field in ("closed", "track_a"):
            if m.group(field):
                # a banner value is exact; there is one accepted form
                out.append(([m.group(field)], f"{rel} [{field}]"))
    if exempt:
        print(f"exempt, marked 'not in the manuscript': {len(exempt)}")
        for e in exempt:
            print(f"  {e}")
    return out


# The DELTAS table is read out of the test source textually rather than imported,
# so this script needs no pytest and cannot degrade to "0 checked, all passed" if
# an import fails. An empty result is treated as a failure below, not a pass.
DELTA_ENTRY = re.compile(
    r'\(\s*"(?P<metric>closed|track_a)",\s*"(?P<a>[^"]+)",\s*"(?P<b>[^"]+)",\s*'
    r'(?P<expected>-?\d+(?:\.\d+)?),')


# Explicit source mappings for differences omitted from the condensed text.
# Each pattern identifies the configuration and its closed-set accuracy, rather
# than accepting the same numbers elsewhere in the manuscript.
DERIVED_DIFFERENCES = {
    "L29_50M - L1_50M on closed": (
        ("sections/04_part2_6mer.tex",
         r"\\item\s+\\ourmodel\{\}\s+29 layers\s+\([^)]*\):\s*"
         r"(?:\\textbf\{)?(?P<value>\d+(?:\.\d+)?)\\%",
         "6mer/L29_50M.yaml"),
        ("sections/04_part2_6mer.tex",
         r"\\item\s+\\ourmodel\{\}\s+1 layer\s+\([^)]*\):\s*"
         r"(?:\\textbf\{)?(?P<value>\d+(?:\.\d+)?)\\%",
         "6mer/L1_50M.yaml"),
    ),
    "exact_1L_mean - exact_16L on closed": (
        ("sections/05_part3_13mer.tex",
         r"Accuracies on the gut test set range from (?P<value>\d+(?:\.\d+)?)\\% "
         r"for the single-layer exact-vocabulary \\ourmodel\{\}",
         "13mer/exact_1L_50M_meanpool.yaml"),
        ("sections/05_part3_13mer.tex",
         r"At sixteen layers, exact \$d=64\$ reaches (?P<value>\d+(?:\.\d+)?)\\%",
         "13mer/exact_16L_50M.yaml"),
    ),
}


def derived_difference(paper_root: Path, where: str, expected: Decimal) -> str:
    """Validate both mapped source values and their difference; return the calculation."""
    values = []
    for filename, pattern, config in DERIVED_DIFFERENCES[where]:
        path = paper_root / filename
        if not path.is_file():
            raise ValueError(f"missing source file: {filename}")
        matches = list(re.finditer(pattern, clean_source(path.read_text())))
        if len(matches) != 1:
            raise ValueError(f"{filename}: expected one result for {config}, found {len(matches)}")
        value = Decimal(matches[0].group("value"))
        banner = BANNER.search((REPO / "configs" / config).read_text())
        if banner is None or value != Decimal(banner.group("closed")):
            raise ValueError(f"{filename}: {value} disagrees with the closed-set banner for {config}")
        values.append(value)
    actual = values[0] - values[1]
    if actual != expected:
        raise ValueError(f"mapped results give {actual}, expected {expected}")
    return f"{values[0]} - {values[1]} = {actual}"


def asserted_deltas():
    """Accepted forms, description and exact value for each asserted difference.

    A difference may appear in the manuscript at two decimals ("5.52 points") or
    rounded to one, because some are quoted as a range -- the three 5M
    pre-training deltas are stated only as "+8.9 to +9.4", so 8.94 as a literal
    string is never in the text. Accepting either form is the difference between
    this check reporting drift and reporting the truth; requiring two decimals
    flagged all three on the first run.
    """
    src = (REPO / "tests" / "test_reported_numbers.py").read_text()
    out = []
    for m in DELTA_ENTRY.finditer(src):
        v = float(m.group("expected"))
        # Three forms, because the manuscript is not consistent and need not be:
        # "4.30 points" keeps its trailing zero, "15.60" likewise, and the 5M
        # deltas appear only rounded, in "+8.9 to +9.4". Dropping the unstripped
        # two-decimal form loses the 4.30 match once boundaries are enforced.
        forms = {f"{v:.2f}", f"{v:.2f}".rstrip("0").rstrip("."), f"{v:.1f}"}
        out.append({"forms": sorted(forms),
                    "where": f"{m.group('a')} - {m.group('b')} on {m.group('metric')}",
                    "expected": Decimal(m.group("expected"))})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paper_root", type=Path,
                    help="a manuscript checkout containing main.tex and sections/")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    text = flatten(args.paper_root)
    print(f"manuscript: {args.paper_root}  ({len(text):,} chars, flattened)")

    missing, empty, invalid = [], [], []
    banners = [{"forms": forms, "where": where} for forms, where in banner_values()]
    for label, items in (("banner accuracies", banners),
                         ("asserted differences", asserted_deltas())):
        if not items:
            # A check that examined nothing must not report success.
            empty.append(label)
            print(f"{label}: FOUND NOTHING TO CHECK -- this is a failure of the "
                  f"check, not a pass")
            continue
        found, derived = 0, 0
        for item in items:
            forms, where = item["forms"], item["where"]
            if where in DERIVED_DIFFERENCES:
                try:
                    calculation = derived_difference(args.paper_root, where, item["expected"])
                except ValueError as exc:
                    invalid.append(f"{where}: {exc}")
                else:
                    derived += 1
                    print(f"  derived {calculation}  {where}")
                continue
            hit = next((f for f in forms if present(f, text)), None)
            if hit:
                found += 1
                if args.verbose:
                    print(f"  ok      {hit:>7}  {where}")
            else:
                missing.append(("/".join(forms), where))
        if label == "asserted differences":
            print(f"{label}: {found + derived}/{len(items)} verified "
                  f"({found} printed, {derived} derived from mapped results)")
        else:
            print(f"{label}: {found}/{len(items)} present in the manuscript")

    if empty:
        print(f"\nCollected no items for: {', '.join(empty)}. The check did not run "
              "properly -- fix that before reading anything into the result.")
        return 2

    if invalid:
        print("\nINVALID DERIVATION:")
        for error in invalid:
            print(f"  {error}")

    if missing:
        print("\nNOT FOUND -- the repo may have drifted from the manuscript, or the "
              "manuscript may have been revised:")
        for value, where in missing:
            print(f"  {value:>7}  {where}")
        print("\nBefore changing anything, check what the discrepancy is made of. A "
              "number can be absent because it was revised, because it moved to the "
              "supplementary, or because this check is looking in the wrong place.")
    if missing or invalid:
        return 1

    print("\nAll checked config accuracies and differences are supported by the manuscript.")
    print("Scope: numeric presence for unmapped entries; configuration-specific source "
          "and arithmetic checks for mapped differences. Other meanings require manual review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

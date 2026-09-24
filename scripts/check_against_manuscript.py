#!/usr/bin/env python3
"""Check that every number this repo reports still appears in the manuscript.

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
  * every difference `tests/test_reported_numbers.py` asserts still appears.

A difference is accepted at two decimals or rounded to one, because the
manuscript quotes some as ranges -- "+8.9 to +9.4" never contains the literal
"8.94". Requiring two decimals reported three of fifteen as drift when nothing
had drifted.

What it cannot verify: that a number still means the same thing. A figure moved
from one arm to another survives this check. Only reading the sentence catches
that, which is why the DELTAS table carries its quoted source.
"""

import argparse
import re
import sys
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


def flatten(paper_root: Path) -> str:
    """Concatenate the sources, drop comments and coordinates, collapse whitespace."""
    parts = []
    for pattern in ("main.tex", "supplementary.tex", "sections/*.tex"):
        for f in sorted(paper_root.glob(pattern)):
            parts.append(f.read_text(errors="replace"))
    if not parts:
        raise SystemExit(f"no .tex found under {paper_root}")
    text = "\n".join(parts)
    text = LATEX_COMMENT.sub(" ", text)
    text = TIKZ_COORD.sub(" ", text)
    return re.sub(r"\s+", " ", text)


def present(value: str, text: str) -> bool:
    """Is `value` in `text` as a number in its own right?

    Substring matching is not enough: "14.6" occurs inside "114.6" and inside a
    coordinate. Require digit/dot boundaries on both sides.
    """
    return re.search(r"(?<![\d.])" + re.escape(value) + r"(?![\d])", text) is not None


def banner_values():
    """(accepted forms, config) for every accuracy stamped on a config.

    A banner may end with `[not in the manuscript]`, for a measurement this
    repository reports that the paper does not print -- an arm dropped when a
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
    r'(?P<expected>[\d.]+),')


def asserted_deltas():
    """(accepted forms, description) for every difference the test suite asserts.

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
        out.append((sorted(forms),
                    f"{m.group('a')} - {m.group('b')} on {m.group('metric')}"))
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

    missing, empty = [], []
    for label, items in (("banner accuracies", banner_values()),
                         ("asserted differences", asserted_deltas())):
        if not items:
            # A check that examined nothing must not report success.
            empty.append(label)
            print(f"{label}: FOUND NOTHING TO CHECK -- this is a failure of the "
                  f"check, not a pass")
            continue
        found = 0
        for forms, where in items:
            hit = next((f for f in forms if present(f, text)), None)
            if hit:
                found += 1
                if args.verbose:
                    print(f"  ok      {hit:>7}  {where}")
            else:
                missing.append(("/".join(forms), where))
        print(f"{label}: {found}/{len(items)} present in the manuscript")

    if empty:
        print(f"\nCollected no items for: {', '.join(empty)}. The check did not run "
              "properly -- fix that before reading anything into the result.")
        return 2

    if missing:
        print("\nNOT FOUND -- the repo may have drifted from the manuscript, or the "
              "manuscript may have been revised:")
        for value, where in missing:
            print(f"  {value:>7}  {where}")
        print("\nBefore changing anything, check what the discrepancy is made of. A "
              "number can be absent because it was revised, because it moved to the "
              "supplementary, or because this check is looking in the wrong place.")
        return 1

    print("\nEvery number this repo reports is still in the manuscript.")
    print("Note what that does NOT establish: a value moved from one arm to another "
          "would still be found. Same-number is not same-meaning.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Report model directories that exist on disk but were not scored.

WHY. Every scoring step in this pipeline takes `--models` as a comma-separated
list with a hardcoded default naming the arms that existed when it was written.
Add an arm later, run the step, and it is excluded from every level with no
message: the only evidence is a row missing from the output, which is the one
thing nobody notices.

That is not hypothetical. Section 7.3 of the manuscript cites the 29-layer
KmerFormer at r = 0.522 on the L2_lognormal pool, and no such row existed in the
stored summary, because `L29` was not in tb_09's default list. The predictions had
been sitting in `out/L2_lognormal/L29/` the whole time. Re-scoring with it added
reproduces 0.5223 exactly. Two people spent a session establishing that a correct
number was not fabricated.

The omission is wider than that one case. Against what is on disk today, the
defaults skip 10 of 17 model directories per ladder level in tb_07, 12 of 17 in
tb_09 and 13 of 17 in tb_14. Much of that is deliberate -- the ladder figures come
from `scripts/eval/track_b_retention.py`, which discovers arms by listing the
directory instead of naming them -- but "deliberate" and "forgotten" look
identical in the output, and that is the problem this fixes.

Warn, do not fail. Scoring a subset on purpose is legitimate and common. The point
is that the choice appears in the run's own output rather than only in the shape
of what is missing.
"""

from pathlib import Path


def available_models(level_dir) -> set:
    """Model directories under a level that actually hold predictions."""
    level_dir = Path(level_dir)
    if not level_dir.is_dir():
        return set()
    return {d.name for d in level_dir.iterdir()
            if d.is_dir() and (d / "preds.npz").exists()}


def report_unscored(level_dir, scored, *, level=None, allow_unscored=False,
                    stream=None) -> list:
    """Print what was scored against what exists; return the unscored names.

    `scored` is whatever the caller actually processed. Names in `scored` with no
    directory are reported too — asking for an arm that is not there is the
    mirror-image mistake and equally quiet, since the loop simply skips it.
    """
    import sys
    out = stream or sys.stderr
    have = available_models(level_dir)
    scored = {s for s in scored if s}
    label = level or Path(level_dir).name

    unscored = sorted(have - scored)
    missing = sorted(scored - have)

    print(f"  scored {len(scored & have)} of {len(have)} model directories under "
          f"{Path(level_dir)}", file=out)

    if missing:
        print(f"  requested but absent: {', '.join(missing)}", file=out)
    if unscored:
        print(f"  NOT SCORED: {', '.join(unscored)}", file=out)
        if not allow_unscored:
            print(f"    Those directories hold predictions for {label} and are not "
                  f"in --models, so they will simply be absent from the output "
                  f"rather than reported as skipped.", file=out)
            print(f"    Add them to --models, or pass --allow-unscored to say the "
                  f"omission is deliberate.", file=out)
    elif have:
        print("  every model directory present was scored", file=out)

    return unscored


def add_coverage_flag(parser):
    """The opt-out, worded the same way everywhere it appears."""
    parser.add_argument(
        "--allow-unscored", action="store_true",
        help="Silence the note about model directories present on disk but not "
             "in --models. Use when the subset is deliberate.")
    return parser

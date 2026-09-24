#!/usr/bin/env python3
"""Cross-check the `# RESULT:` banners against the differences the paper states.

WHY THIS EXISTS, AND WHY IT IS NOT REDUNDANT WITH THE OTHER TESTS.

The banners were transcribed by hand from the manuscript, one per config. The
failure mode that transcription invites is not a missing number -- that is caught
by `test_result_banner_present` -- but a number attached to the *wrong* config.
Two banners swapped between sibling arms leaves every file well-formed, every link
resolving and every name matching its contents, and is simply wrong. A checker
that only verifies references resolve cannot see it.

What catches it is that the paper does not only state levels, it states
*differences*: "one layer beats sixteen by 5.52 points with an exact vocabulary",
"the cost of hashing is 6.37 points at matched width and depth", "the gain from
depth alone is 15.60 points". Each is an independent arithmetic constraint tying
two specific arms together. Transpose a pair and the arithmetic stops closing.

Every DELTAS entry below is quoted from the manuscript with its section, so this
file doubles as the provenance record for the banners.

An arm leaves this table when the manuscript stops printing its number, not when
the measurement stops being true. `L8_5M` was dropped here after Part 1 was
rewritten around the depth-matched 29-layer arm; the 53.62% it measured is still
in `configs/`, marked so both checkers know it is ours rather than the paper's.

WHAT THIS DOES NOT DO. An earlier version of this docstring claimed that if the
manuscript's numbers changed, these tests would fail loudly. They would not: the
expected differences are constants transcribed from the manuscript, so a revision
there leaves them untouched and every test still passes. These check the banners
against a frozen copy, which is enough to catch a number attached to the wrong
config and nothing more. The manuscript is not in this repository, so no test here
can see repo-against-manuscript drift.

`scripts/check_against_manuscript.py` is the half that can, pointed at a
manuscript checkout. It has to be run deliberately.
"""

import re
from pathlib import Path

import pytest

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

BANNER = re.compile(
    r"^# RESULT: closed-set 100K genus Top-1 (?P<closed>[\d.]+)%"
    r"(?:\s*\|\s*Track A out-of-genome (?P<track_a>[\d.]+)%)?")

# (config path relative to configs/, ...) -- every arm a delta below refers to
ARMS = {
    "L1_50M":                "6mer/L1_50M.yaml",
    "L16_50M":               "6mer/L16_50M.yaml",
    "L29_50M":               "6mer/L29_50M.yaml",
    "L16_50M_stride1":       "6mer/L16_50M_stride1.yaml",
    "L16_500K":              "6mer/data_scale/L16_500K.yaml",
    "L16_1M":                "6mer/data_scale/L16_1M.yaml",
    "L29_500K":              "6mer/data_scale/L29_500K.yaml",
    "L29_1M":                "6mer/data_scale/L29_1M.yaml",
    "ntv2_500K":             "baselines/ntv2_lora_500K.yaml",
    "ntv2_1M":               "baselines/ntv2_lora_1M.yaml",
    "L16_5M":                "6mer/data_scale/L16_5M.yaml",
    "L29_5M":                "6mer/data_scale/L29_5M.yaml",
    "ntv2_50M":              "baselines/ntv2_lora_50M.yaml",
    "ntv2_5M":               "baselines/ntv2_lora_5M.yaml",
    "exact_1L_mean":         "13mer/exact_1L_50M_meanpool.yaml",
    "exact_1L_attn":         "13mer/exact_1L_50M_attnpool.yaml",
    "exact_16L":             "13mer/exact_16L_50M.yaml",
    "hashed_d128_1L":        "13mer/hashed_d128_1L_50M.yaml",
    "hashed_d128_16L":       "13mer/hashed_d128_16L_50M.yaml",
    "hashed_d64_16L":        "13mer/hashed_d64_16L_50M.yaml",
}

# (metric, arm_a, arm_b, expected difference, quoted source)
DELTAS = [
    # ---- Part 2: the 6-mer comparison ----
    ("closed", "L29_50M", "L1_50M", 15.60,
     'Part 2: "the gain from depth alone is 15.60 points"'),
    ("closed", "L29_50M", "ntv2_50M", 2.44,
     'Part 2: "the margin over NT-v2 is 2.44 points at 77x fewer parameters"'),
    ("closed", "L16_50M", "L16_50M_stride1", 5.19,
     'Part 2: stride-1 "lowers accuracy by 5.19 points (62.75% vs 67.94%)"'),
    # ---- Part 1: depth is spent at 5M but not at 50M ----
    ("closed", "L29_50M", "L16_50M", 1.58,
     'Part 1: "the same step is worth 1.58 points at 50M"'),
    ("closed", "L29_5M", "L16_5M", 0.08,
     'Part 1: "16 layers to 29 buys 0.08 points" at the 5M budget'),
    # Part 1's data-scale passage was rewritten to lead with the DEPTH-MATCHED
    # arm -- the 29-layer model, NT-v2's own depth -- rather than the 16-layer
    # one, and the 8-layer 5M figure was dropped. These four pin the primary
    # series; the four after them pin the 16-layer series it still reports
    # alongside.
    ("closed", "ntv2_500K", "L29_500K", 13.69,
     'Part 1: "leads by 13.7 and 13.2 points at 0.5M and 1M (54.72% and 57.34% '
     'against 41.03% and 44.13%)"'),
    ("closed", "ntv2_1M", "L29_1M", 13.21,
     'Part 1: the "13.2" of that same sentence'),
    ("closed", "ntv2_5M", "L29_5M", 8.86,
     'Part 1: "by 8.9 at 5M (63.05% against 54.19%)"'),
    ("closed", "L29_50M", "ntv2_50M", 2.44,
     'Part 1: "overtaken by 2.4 at 50M (67.08% against 69.52%)"'),
    ("closed", "ntv2_500K", "L16_500K", 13.37,
     'Part 1: the 16-layer series, "the lead over that arm is 13.4, 13.1, 8.9 '
     'and -0.9 points"'),
    ("closed", "ntv2_1M", "L16_1M", 13.05,
     'Part 1: the "13.1" of that series'),
    ("closed", "ntv2_5M", "L16_5M", 8.94,
     'Part 1: the "8.9" of that series'),
    # Depth at fixed budget, which is what makes the depth-matched claim work:
    # NT-v2's own depth is slightly WORSE from scratch at the two small budgets.
    ("closed", "L29_500K", "L16_500K", -0.32,
     'Part 1: "sixteen layers to twenty-nine is worth -0.32, -0.16, +0.08 and '
     '+1.58 points as the budget grows"'),
    ("closed", "L29_1M", "L16_1M", -0.16,
     'Part 1: the "-0.16" of that series'),
    # ---- Part 3: the 13-mer arms ----
    ("closed", "exact_1L_mean", "exact_16L", 5.52,
     'Part 3: "one layer beats sixteen by 5.52 points with an exact vocabulary"'),
    ("closed", "hashed_d128_1L", "hashed_d128_16L", 2.97,
     'Part 3: "...and 2.97 with a hashed one"'),
    ("closed", "exact_16L", "hashed_d64_16L", 6.37,
     'Part 3: "the cost of hashing is 6.37 points at matched width and depth '
     '(exact d64 85.63% vs hashed d64 79.26%)"'),
    ("closed", "exact_1L_mean", "exact_1L_attn", 0.41,
     'Part 3: swapping mean for attention pooling "costs 0.41 points '
     '(90.74% against 91.15%)"'),
    # ---- Off the catalogue: the same contrasts, and they behave differently ----
    ("track_a", "exact_1L_mean", "exact_16L", 4.30,
     'Off the catalogue: "one layer beats sixteen by 4.30 points exact"'),
    ("track_a", "hashed_d128_1L", "hashed_d128_16L", 2.14,
     'Off the catalogue: "...and 2.14 hashed"'),
    ("track_a", "exact_1L_mean", "exact_1L_attn", 1.32,
     'Off the catalogue: "mean pooling is worth 1.32 points out of genome '
     'against 0.41 on the closed set"'),
]

# The banners carry two decimals, so a stated one-decimal difference can be off by
# up to 0.005 either way. Anything looser would let a real transposition through.
TOLERANCE = 0.011


def _banner(rel_path):
    text = (CONFIG_DIR / rel_path).read_text()
    m = BANNER.match(text.splitlines()[0])
    assert m, f"{rel_path}: first line is not a parseable RESULT banner"
    return m


@pytest.mark.parametrize("key,rel", sorted(ARMS.items()))
def test_arm_banner_parses(key, rel):
    assert (CONFIG_DIR / rel).exists(), f"{rel} is missing"
    _banner(rel)


@pytest.mark.parametrize(
    "metric,a,b,expected,source", DELTAS,
    ids=[f"{m}:{a}-{b}={e}" for m, a, b, e, _ in DELTAS])
def test_stated_difference_holds(metric, a, b, expected, source):
    """A difference the manuscript states must close over the two banners it names."""
    va, vb = _banner(ARMS[a]).group(metric), _banner(ARMS[b]).group(metric)
    assert va is not None, f"{ARMS[a]} has no {metric} figure, needed for {source}"
    assert vb is not None, f"{ARMS[b]} has no {metric} figure, needed for {source}"
    got = float(va) - float(vb)
    assert abs(got - expected) < TOLERANCE, (
        f"{source}\n"
        f"  expected {a} - {b} = {expected:+.2f} on {metric}\n"
        f"  got      {float(va):.2f} - {float(vb):.2f} = {got:+.2f}\n"
        f"  one of these two banners is on the wrong config, or the manuscript "
        f"changed and the banners did not follow")


# ---------------------------------------------------------------------------
# A different shape of constraint: a weighted average, read out of the prose.
#
# The deltas above pin numbers against numbers. This pins a *count in prose*,
# which is the one thing they cannot reach -- and it is the failure that actually
# happened. An earlier draft of the manuscript said "Thirteen" genomes where the
# arithmetic requires six; this repo faithfully copied the error, and nothing
# caught it, because a count in a sentence is not a number in a table.
#
# What settles it is that the manuscript states enough to over-determine the
# count: the accuracy on the removed subset (48.06%), the accuracy on the rest of
# the pool (26.31%), and the accuracy on the whole pool (27.44%, which is also the
# Track A banner of the 50M arm). Those three close for exactly one value of n:
#
#     n=6  (30,000 of 580,000 reads) -> 27.4350%   agrees with 27.44%
#     n=13 (65,000 of 580,000 reads) -> 28.7475%   does not
#
# so the test reads n out of docs/RESULTS.md rather than hard-coding it. Rewrite
# the passage and the pattern stops matching, which fails loudly; change the count
# alone and the arithmetic fails.
# ---------------------------------------------------------------------------

RESULTS_MD = Path(__file__).resolve().parent.parent / "docs" / "RESULTS.md"

WORD_TO_INT = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
               "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
               "twelve": 12, "thirteen": 13}

READS_PER_GENOME = 5_000
TRACK_A_POOL = 580_000          # after the three protocol exclusions
ACC_ON_REMOVED = 48.06          # manuscript, off-catalogue section
ACC_ON_REST = 26.31             # manuscript, same sentence


def test_near_identical_genome_count_is_arithmetically_consistent():
    text = RESULTS_MD.read_text()

    m = re.search(r"leaving \*\*(?P<count>\w+)\*\* that contribute\s+"
                  r"(?P<reads>[\d,]+) reads", text)
    assert m, ("could not find the near-identical-genome passage in docs/RESULTS.md. "
               "If it was rewritten, update this test rather than deleting it -- it "
               "is the only check that pins that count.")

    word = m.group("count").lower()
    assert word in WORD_TO_INT, f"unrecognised count word {word!r}"
    n = WORD_TO_INT[word]
    stated_reads = int(m.group("reads").replace(",", ""))

    # the count and the read total must agree with each other
    assert n * READS_PER_GENOME == stated_reads, (
        f"the passage says {word} genomes ({n}) and {stated_reads:,} reads, but "
        f"{n} x {READS_PER_GENOME:,} = {n * READS_PER_GENOME:,}")

    # and both must reproduce the whole-pool accuracy the banner carries
    banner_pool_acc = float(_banner(ARMS["exact_1L_mean"]).group("track_a"))
    removed = n * READS_PER_GENOME
    weighted = (ACC_ON_REMOVED * removed
                + ACC_ON_REST * (TRACK_A_POOL - removed)) / TRACK_A_POOL
    assert abs(weighted - banner_pool_acc) < 0.011, (
        f"n={n} does not reconcile the manuscript's own figures:\n"
        f"  {ACC_ON_REMOVED}% on {removed:,} removed reads and {ACC_ON_REST}% on the "
        f"remaining {TRACK_A_POOL - removed:,}\n"
        f"  weighted over {TRACK_A_POOL:,} gives {weighted:.4f}%, but the Track A "
        f"banner says {banner_pool_acc:.2f}%\n"
        f"  n=6 gives 27.4350%; n=13 gives 28.7475%")


def test_every_delta_names_a_known_arm():
    """A typo in ARMS would make a delta silently untested."""
    for metric, a, b, _, source in DELTAS:
        assert a in ARMS, f"{source}: unknown arm {a!r}"
        assert b in ARMS, f"{source}: unknown arm {b!r}"
        assert metric in ("closed", "track_a"), f"{source}: unknown metric {metric!r}"


def test_deltas_cover_every_arm_they_can():
    """Every arm listed in ARMS must take part in at least one cross-check --
    otherwise its banner is asserted to parse but never checked against anything."""
    used = {a for _, a, _, _, _ in DELTAS} | {b for _, _, b, _, _ in DELTAS}
    unchecked = sorted(set(ARMS) - used)
    assert not unchecked, (
        f"these arms are in ARMS but no delta constrains them: {unchecked}. "
        "Either add a constraint from the manuscript or drop them from ARMS.")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

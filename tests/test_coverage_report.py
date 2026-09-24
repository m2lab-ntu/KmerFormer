#!/usr/bin/env python3
"""Tests for scripts/track_b/_coverage.py.

The helper exists because a hardcoded `--models` default silently excluded an arm
whose predictions were on disk: section 7.3 of the manuscript cites the 29-layer
KmerFormer at r = 0.522 on L2_lognormal, no such row was in the stored summary,
and establishing that the number was real rather than fabricated cost a working
session. Re-scoring with `L29` added reproduces 0.522.

So the thing under test is a warning, and a warning that fails to appear is the
whole bug. These check that it appears, names the right directories, and does not
cry wolf when everything present was scored.
"""

import io
from pathlib import Path

import pytest

import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "track_b"))

from _coverage import available_models, report_unscored  # noqa: E402


def make_level(tmp_path, name, models_with_preds, models_without=()):
    level = tmp_path / name
    level.mkdir(parents=True)
    for m in models_with_preds:
        d = level / m
        d.mkdir()
        (d / "preds.npz").write_bytes(b"")
    for m in models_without:
        (level / m).mkdir()
    return level


def capture(level, scored, **kw):
    buf = io.StringIO()
    unscored = report_unscored(level, scored, stream=buf, **kw)
    return unscored, buf.getvalue()


def test_available_models_requires_predictions(tmp_path):
    """A directory with no preds.npz is not a scoreable arm and must not be
    reported as skipped — otherwise every run nags about scratch directories."""
    level = make_level(tmp_path, "L2_far", ["mt_250M", "L29"], models_without=["tmp", "logs"])
    assert available_models(level) == {"mt_250M", "L29"}


def test_the_reported_case(tmp_path):
    """The exact shape that cost a session: five scored, one on disk, unnamed."""
    level = make_level(tmp_path, "L2_lognormal",
                       ["mt_250M", "mt_50M", "mt_6mer", "nt_v9", "kraken2", "L29"])
    unscored, out = capture(level, ["mt_250M", "mt_50M", "mt_6mer", "nt_v9", "kraken2"])
    assert unscored == ["L29"]
    assert "scored 5 of 6" in out
    assert "NOT SCORED: L29" in out
    assert "--allow-unscored" in out


def test_silent_when_everything_present_was_scored(tmp_path):
    level = make_level(tmp_path, "L3_genus", ["mt_250M", "kraken2"])
    unscored, out = capture(level, ["mt_250M", "kraken2"])
    assert unscored == []
    assert "NOT SCORED" not in out
    assert "every model directory present was scored" in out


def test_requesting_an_absent_model_is_also_reported(tmp_path):
    """The mirror-image mistake: naming an arm that is not there. The scoring
    loop just skips it, so without this it is as quiet as the other direction."""
    level = make_level(tmp_path, "L2_far", ["mt_250M"])
    unscored, out = capture(level, ["mt_250M", "kf_exact13_1L_50M"])
    assert "requested but absent: kf_exact13_1L_50M" in out


def test_allow_unscored_keeps_the_fact_and_drops_the_nagging(tmp_path):
    """Opting out must not hide *that* something was skipped, only stop advising
    about it — otherwise the flag reintroduces the silence it exists to fix."""
    level = make_level(tmp_path, "L2_lognormal", ["mt_250M", "L29"])
    unscored, out = capture(level, ["mt_250M"], allow_unscored=True)
    assert unscored == ["L29"]
    assert "NOT SCORED: L29" in out
    assert "scored 1 of 2" in out
    assert "--allow-unscored" not in out


def test_missing_level_directory_does_not_crash(tmp_path):
    unscored, out = capture(tmp_path / "nope", ["mt_250M"])
    assert unscored == []
    assert "scored 0 of 0" in out


@pytest.mark.parametrize("script", ["tb_07_metrics.py", "tb_09_abundance_metrics.py",
                                    "tb_14_ood_detection.py"])
def test_every_scoring_step_wires_it_in(script):
    """All three take a hardcoded --models default, so all three need the report.
    Adding a fourth scoring step without it would reintroduce the bug."""
    src = (REPO / "scripts" / "track_b" / script).read_text()
    assert "from _coverage import" in src, f"{script} does not import the helper"
    assert "report_unscored(" in src, f"{script} imports it but never calls it"
    assert "add_coverage_flag(" in src, f"{script} has no --allow-unscored"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

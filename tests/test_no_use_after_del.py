#!/usr/bin/env python3
"""Static check: nothing reads a local after `del` removed it.

WHY THIS EXISTS. `train.py` loads the best checkpoint at the end of training,
pulls what it needs out, and deletes the dict immediately — the exact-13-mer arms
carry an 8 GiB embedding table and keeping a second copy alive OOM'd the box after
all 15 epochs had already finished. That is a good optimisation and the code does
it correctly today: `best_epoch` is extracted *before* the `del`.

An earlier version did not. It deleted the dict and then, sixty lines later, built
the metrics summary from `best_ckpt["epoch"]`. The `del` sits in straight-line
code with no branch, so this was not an edge case — every run that reached the
final-evaluation stage raised UnboundLocalError there. Nine runs on the training
box did, and the damage is quiet: `best.pt`, `last.pt` and `training_history.csv`
are all written during training, so the run looks complete, but the crash lands
*before* `save_json(metrics, ...)`, so `metrics.json` and `resource_report.json`
never appear and the log never records which epoch was selected. Anyone auditing
such a run by reading its log concludes the epoch was not recorded, rather than
that the summary crashed.

Checkpoint selection is load-bearing here — a repeat of one arm differs from the
original by 0.113 points and part of that is which epoch each run selected — so
the fact this bug erases is one that matters.

The fix is the fragile shape: extract, delete, use the extract much later. A
future memory pass could reintroduce it by moving one line, and no test over
behaviour would notice, because the crash only happens after a full training run.
Hence a static check.
"""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MODULES = sorted((REPO / "kmerformer").glob("*.py")) + \
          sorted((REPO / "scripts").rglob("*.py"))


def _use_after_del(tree):
    """[(function, name, del_line, use_line)] for every load of a deleted local.

    A name that is reassigned after the `del` is fine, so a load only counts when
    it happens after the delete and before any rebinding.
    """
    problems = []
    for func in [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        deletes = {}          # name -> earliest del line
        for node in ast.walk(func):
            if isinstance(node, ast.Delete):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        deletes.setdefault(t.id, node.lineno)
        if not deletes:
            continue
        for name, del_line in deletes.items():
            rebinds = [n.lineno for n in ast.walk(func)
                       if isinstance(n, ast.Name) and n.id == name
                       and isinstance(n.ctx, ast.Store) and n.lineno > del_line]
            next_bind = min(rebinds) if rebinds else float("inf")
            for n in ast.walk(func):
                if (isinstance(n, ast.Name) and n.id == name
                        and isinstance(n.ctx, ast.Load)
                        and del_line < n.lineno < next_bind):
                    problems.append((func.name, name, del_line, n.lineno))
    return problems


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(REPO)))
def test_no_read_of_a_deleted_local(path):
    tree = ast.parse(path.read_text(errors="replace"), filename=str(path))
    problems = _use_after_del(tree)
    assert not problems, "\n".join(
        f"{path.relative_to(REPO)}: {fn}() reads {name!r} at line {use} "
        f"after `del {name}` at line {dl}"
        for fn, name, dl, use in problems)


def test_the_check_would_catch_the_original_bug():
    """The guard is worthless unless it fires on the shape that actually shipped."""
    buggy = '''
def main():
    best_ckpt = load()
    model.load_state_dict(best_ckpt["model_state_dict"])
    del best_ckpt
    metrics = {"best_epoch": int(best_ckpt["epoch"])}
'''
    found = _use_after_del(ast.parse(buggy))
    assert found, "the check does not catch use-after-del"
    fn, name, dl, use = found[0]
    assert (fn, name) == ("main", "best_ckpt")
    assert dl < use


def test_reassignment_after_del_is_not_flagged():
    """Deleting and then rebuilding a name is a normal memory pattern."""
    fine = '''
def main():
    big = load()
    use(big)
    del big
    big = load_again()
    use(big)
'''
    assert not _use_after_del(ast.parse(fine))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

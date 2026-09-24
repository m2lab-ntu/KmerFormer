#!/usr/bin/env python3
"""Invariants over the config tree. No GPU, no data, no network.

These lock in the properties that make `configs/` usable by someone who is not on
the machine the runs were done on, and that make the arms comparable. Each was
verified by hand once; a test is what keeps it true.
"""

import re
import sys
from pathlib import Path

import pytest
import yaml

from kmerformer.utils import load_config

REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO / "configs"
CONFIGS = sorted(CONFIG_DIR.rglob("*.yaml"))

# The three variables every path is written against. See docs/DATA.md.
KNOWN_VARS = {"KF_DATA", "KF_OUT", "KF_VOCAB"}

PATH_KEYS = ("fasta_path", "labels_path", "vocab_path", "cache_path", "dir")


def _walk(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, path + (str(i),))
    else:
        yield path, node


def test_configs_exist():
    assert len(CONFIGS) > 30, f"only found {len(CONFIGS)} configs"


@pytest.mark.parametrize("cfg_path", CONFIGS, ids=lambda p: str(p.relative_to(CONFIG_DIR)))
def test_config_is_valid_yaml_and_loads(cfg_path):
    raw = yaml.safe_load(cfg_path.read_text())
    assert isinstance(raw, dict), "a config must be a mapping"
    assert "model" in raw and "data" in raw and "training" in raw
    assert (raw.get("output") or {}).get("dir"), "every run needs an output dir"
    load_config(cfg_path)  # must survive variable expansion


@pytest.mark.parametrize("cfg_path", CONFIGS, ids=lambda p: str(p.relative_to(CONFIG_DIR)))
def test_paths_are_portable(cfg_path):
    """No absolute path may be baked in: a config must run on any machine.

    Every filesystem path goes through ${KF_DATA} / ${KF_OUT} / ${KF_VOCAB}. A
    leading "/" that is not a variable reference means someone's home directory
    leaked into the repo.
    """
    raw = yaml.safe_load(cfg_path.read_text())
    offenders = []
    for keypath, value in _walk(raw):
        if not isinstance(value, str) or not keypath:
            continue
        if keypath[-1] not in PATH_KEYS:
            continue
        if value.startswith("/"):
            offenders.append(f"{'.'.join(keypath)} = {value}")
        for var in re.findall(r"\$\{?(\w+)\}?", value):
            if var not in KNOWN_VARS:
                offenders.append(f"{'.'.join(keypath)} references unknown ${var}")
    assert not offenders, "non-portable paths: " + "; ".join(offenders)


def test_output_dirs_are_unique():
    """Two configs writing to one directory would silently overwrite each other's
    checkpoints, and the second run would look like a reproduction of the first."""
    seen = {}
    for cfg_path in CONFIGS:
        d = (yaml.safe_load(cfg_path.read_text()).get("output") or {}).get("dir")
        seen.setdefault(d, []).append(str(cfg_path.relative_to(CONFIG_DIR)))
    clashes = {d: v for d, v in seen.items() if len(v) > 1}
    assert not clashes, f"configs sharing an output dir: {clashes}"


@pytest.mark.parametrize("cfg_path", CONFIGS, ids=lambda p: str(p.relative_to(CONFIG_DIR)))
def test_result_banner_present(cfg_path):
    """Every arm the paper reports carries the number it produced, at the top of
    the file, so a config and its result cannot drift apart. The exceptions are
    declared here rather than left implicit."""
    text = cfg_path.read_text()
    rel = cfg_path.relative_to(CONFIG_DIR)
    if rel.parts[0] == "bench":
        return  # benchmark arms report throughput, not accuracy
    if rel.parts[0] == "extra":
        # Nothing in extra/ backs a claim, so it carries no result -- but it must
        # say whether a result exists at all, so an arm that was never run is not
        # mistaken for one whose number is being withheld.
        assert text.startswith("# STATUS:"), (
            f"{rel} is in extra/ and has no '# STATUS:' line saying whether it ran")
        return
    assert text.startswith("# RESULT:"), (
        f"{rel} has no '# RESULT:' banner (add one, or move it to extra/)")
    # A banner may declare its number is ours rather than the manuscript's. The
    # wording is fixed because scripts/check_against_manuscript.py matches on it,
    # and a near-miss would silently stop exempting.
    first = text.splitlines()[0]
    if "not in the manuscript" in first:
        assert first.endswith("[not in the manuscript]"), (
            f"{rel}: the exemption must be exactly the trailing marker "
            f"'[not in the manuscript]' or the checker will not honour it")


def _sparse(raw):
    return bool((raw.get("model", {}).get("shallow_config") or {}).get("sparse_embedding"))


def _tokenizer_type(raw):
    return ((raw.get("data") or {}).get("tokenizer") or {}).get("type")


@pytest.mark.parametrize("cfg_path", CONFIGS, ids=lambda p: str(p.relative_to(CONFIG_DIR)))
def test_sparse_embedding_only_on_exact_vocabularies(cfg_path):
    """sparse_embedding is what makes the 33.5M-row exact table trainable, and it
    is also what rules out DDP. It must not appear on an arm that does not need
    it: a hashed table is 2^22 rows and updates densely, and turning sparse on
    there would forfeit multi-GPU training for nothing.
    """
    raw = yaml.safe_load(cfg_path.read_text())
    if _sparse(raw):
        assert _tokenizer_type(raw) == "exact_kmer", (
            f"{cfg_path.name} sets sparse_embedding but its tokenizer is "
            f"{_tokenizer_type(raw)!r}; only the exact vocabulary needs it")


@pytest.mark.parametrize("cfg_path", CONFIGS, ids=lambda p: str(p.relative_to(CONFIG_DIR)))
def test_ddp_guard_matches_the_sparse_flag(cfg_path):
    """`train_ddp.rejects_data_parallel` must key on sparsity and nothing else.

    Both directions matter, and the second is the one at risk. Rejecting a sparse
    arm prevents an opaque reducer failure tens of minutes into a run. But
    *admitting* the non-sparse arms is what keeps multi-GPU training available to
    the hashed 13-mer arms -- which are long-k too, so a guard rewritten to key on
    `k` or on the tokenizer type would start rejecting them. That failure is silent:
    nobody reports it, the hashed arms just look inherently slow.

    So do not delete this as redundant with the invariant above. That one constrains
    the configs; this one constrains the guard.
    """
    from kmerformer.train_ddp import rejects_data_parallel

    cfg = load_config(cfg_path)
    expected = _sparse(cfg)
    assert rejects_data_parallel(cfg) is expected, (
        f"{cfg_path.name}: guard says "
        f"{'reject' if not expected else 'admit'} but sparse_embedding is {expected}")


def test_ddp_guard_admits_every_hashed_arm():
    """The negative case, stated once over the whole family rather than per file."""
    from kmerformer.train_ddp import rejects_data_parallel

    hashed = [p for p in CONFIGS
              if _tokenizer_type(yaml.safe_load(p.read_text())) == "hashed_kmer"]
    assert hashed, "no hashed arms found -- has the config tree been renamed?"
    rejected = [p.name for p in hashed if rejects_data_parallel(load_config(p))]
    assert not rejected, (
        "the DDP guard rejects hashed arms, which train densely and should keep "
        f"multi-GPU support: {rejected}")


@pytest.mark.parametrize("cfg_path", CONFIGS, ids=lambda p: str(p.relative_to(CONFIG_DIR)))
def test_custom_tokenizer_declares_a_vocab_size(cfg_path):
    """train.py asserts config vocab_size == tokenizer vocab_size. Leaving it out
    means the embedding is sized from NT-v2's 6-mer vocabulary, which for a
    13-mer arm is a silently wrong model rather than an error."""
    raw = yaml.safe_load(cfg_path.read_text())
    if _tokenizer_type(raw) in {"exact_kmer", "hashed_kmer"}:
        assert raw["model"].get("vocab_size"), (
            f"{cfg_path.name} uses a custom k-mer tokenizer but declares no "
            "model.vocab_size")


def test_hashed_arms_agree_on_bucket_count():
    """Every hashed arm in the paper maps into 2^22 buckets. A config that
    quietly used a different count would not be comparable with the others."""
    counts = {}
    for cfg_path in CONFIGS:
        raw = yaml.safe_load(cfg_path.read_text())
        tok = (raw.get("data") or {}).get("tokenizer") or {}
        if tok.get("type") == "hashed_kmer":
            counts[str(cfg_path.relative_to(CONFIG_DIR))] = tok.get("n_buckets")
    assert counts, "no hashed arms found"
    assert set(counts.values()) == {1 << 22}, f"mixed bucket counts: {counts}"


def test_every_config_linked_from_docs_exists():
    """Docs point at configs by path; a rename must not leave a dead reference."""
    missing = []
    for md in list((REPO / "docs").glob("*.md")) + [REPO / "README.md",
                                                    CONFIG_DIR / "README.md",
                                                    REPO / "weights" / "README.md"]:
        for link in re.findall(r"\]\(([^)\s]+\.yaml)\)", md.read_text()):
            target = (md.parent / link).resolve()
            if not target.exists():
                missing.append(f"{md.relative_to(REPO)} -> {link}")
    assert not missing, "dead config links in docs: " + "; ".join(missing)



def test_no_broken_doc_links_or_anchors():
    """Runs scripts/check_doc_links.py, which reproduces GitHub's slug algorithm.

    Kept as a test because the failure mode is invisible locally: a wrong anchor
    resolves fine in most markdown viewers and 404s only on the published page.
    The checker had that bug itself -- it collapsed whitespace where
    github-slugger does not, so an em-dash heading's double hyphen was reported
    as valid. One link in the repo was wrong because of it.
    """
    import subprocess
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "check_doc_links.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"\n{r.stdout}\n{r.stderr}"

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

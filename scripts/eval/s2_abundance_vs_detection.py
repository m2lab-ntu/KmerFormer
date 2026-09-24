#!/usr/bin/env python3
"""Regenerate the abundance-vs-detection comparison (manuscript Fig. S2 / Table 5).

WHY THIS EXISTS. The figure was drawn from five hardcoded triples in the
manuscript's `figures_src/make_figures.py` with no provenance comment, and with
no KmerFormer arm. The constants cannot be reproduced from anything on disk: a
sweep over pools and protocol parameters got no closer to the published ROC AUC
0.905 / sens@95 66.5% than 0.882 / 61.2%, and commit 7455d67 -- which introduced
them -- says the numbers were checked against `THESIS_NUMBERS.md`, where none of
the five appear. So this recomputes every arm from predictions instead, on one
pool, with one scorer, one protocol and one seed, and prints the pool's md5.

THREE THINGS THAT MAKE THAT HARDER THAN IT SOUNDS, each of which produced a
wrong number before it was caught:

1. TWO 100K POOLS EXIST and "100K" in a filename does not say which. Both hold
   100,000 reads over the same 120 genera; their genus_class vectors agree on
   exactly 8.00% of positions, which is above chance and far below identity, so
   no length, class-count or spot check separates them. `reads_100K_val.fa`
   (cf9220f0) is the closed-set pool every printed number comes from.
   `reads_100K.fa` (3436e31e) is a second, disjoint draw. The KmerFormer arms
   already had sample-level output on the wrong one -- r 0.9951, sens 20.41% --
   which looked usable and was not. Hence the label check in check_arm().

2. THE COVERAGE MASK CANNOT CARRY A DETECTION COLUMN. Restricting to the reads
   whose source species is in the Kraken 2 index (85,773 of 100,000) is right
   for read accuracy, and wrong for the sparse-community protocol: it empties
   17 of the 120 genera, so of the 60 genera each sample declares present, 8.3
   on average hold zero reads. 14% of positives become false negatives by
   construction. `evaluate_sample.py` refuses such a run outright (its global
   feasibility cap collapses reads_per_sample to 0);
   `evaluate_sample_kraken2.py` has no such cap and quietly returns a depressed
   number -- AUC 0.899 against 0.954 for the same arm. So the masked pool
   contributes read accuracy here and nothing else.

3. THE KRAKEN 2 ROW WAS ON THE INDEX THE PAPER ARGUES AGAINST. Two indexes
   exist over the same 1,535-genome library: one with 120 genus nodes between
   species and root, one without. Without them a read ambiguous between two
   species of one genus climbs to root and is scored unclassified -- 20.01% of
   this pool, 19.09 points. An earlier version of this table scored that build
   under the name "1535DB", which reports how many genomes went in rather than
   which taxonomy came out. Both rows are printed here so the name can never
   hide it again.

WHAT IT DOES NOT DO. It scores the *raw* Kraken 2 report. Bracken re-estimates
abundance from the whole report and cannot be restricted per read, so its point
comes from `eval_kraken1535_vs_neural.py`, not from here.
"""

import argparse
import csv
import hashlib
import os
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent

# Known pools, by md5 of the FASTA. An unrecognised pool is reported, not
# refused -- but it will never be silently mistaken for one of these.
POOLS = {
    "cf9220f05f8bd0d8f3c6b2fd07cd8bc9":
        "clean_common / val_100K -- the closed-set pool the manuscript prints",
    "3436e31e86aa9ad9992115db285ff561":
        "twcc_test100k -- a second, disjoint draw; no printed number uses it",
}
# Paths follow the repo convention in docs/REPRODUCE.md: set KF_DATA and the
# defaults resolve under it. Nothing here is absolute, so this runs on a fresh
# checkout with the data bundle downloaded anywhere.
KF_DATA = os.environ.get("KF_DATA", "")
DEFAULT_FA = os.path.join(KF_DATA, "val_100K", "reads_100K_val.fa")
DEFAULT_TSV = os.path.join(KF_DATA, "val_100K", "labels_100K_val.tsv")

# The 1,316 of 1,535 species present in the Kraken 2 index, shipped with the
# repository. Keyed by species_name, which is pool-independent -- the same list
# masks either 100K pool, and reconstructing the coverage mask from it
# reproduces the stored in_db_mask.npy bit for bit.
PRESENT_SPECIES = REPO / "docs" / "assets" / "kraken2_index_species.tsv"

PROTOCOL = dict(n_partition_samples=100, reads_per_sample=1000,
                n_sparse_samples=200, genera_present=60, seed=42)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pool(tsv):
    genus, species = [], []
    with open(tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            genus.append(int(r["genus_class"])); species.append(r["species_name"])
    return np.array(genus), species


def coverage_mask(species_names):
    present = set()
    with open(PRESENT_SPECIES) as f:
        lines = [l for l in f if not l.startswith("#")]
    for r in csv.DictReader(lines, delimiter="\t"):
        present.add(r["species_name"])
    if not present:
        raise SystemExit(f"{PRESENT_SPECIES} yielded no species names")
    return np.array([s in present for s in species_names])


def check_arm(name, path, pool_labels):
    """Refuse an arm whose labels are not the pool's, in the pool's order.

    This is the check the missing figure point needed. Candidate files for
    these arms are all 100,000 reads long over 120 genus classes, so only
    elementwise label equality separates the two pools -- they agree on 8% of
    positions, which is neither identity nor chance.
    """
    d = np.load(path)
    preds, labels = d["preds"].astype(np.int64), d["labels"].astype(np.int64)
    if labels.shape != pool_labels.shape:
        raise SystemExit(f"{name}: {labels.shape[0]:,} reads, pool has "
                         f"{pool_labels.shape[0]:,}")
    if not (labels == pool_labels).all():
        agree = float((labels == pool_labels).mean())
        raise SystemExit(
            f"{name}: labels do not match the pool ({agree:.2%} elementwise). "
            f"A different pool or read order, not a different label encoding "
            f"-- its numbers are not comparable. Re-run inference on the pool "
            f"named above.")
    return preds, labels


def score(npz, out_dir):
    """Every arm goes through evaluate_sample_kraken2.py, including the neural
    ones -- not because they need its unclassified handling, but because the
    two scorers do not agree. On the same predictions and seed they give
    identical abundance and different detection (AUC 0.6863 against 0.6858),
    because they consume the RNG in a different order. Scoring Kraken 2 with
    one and the neural arms with the other put them on different sparse draws.
    """
    cmd = [sys.executable,
           str(REPO / "scripts" / "eval" / "evaluate_sample_kraken2.py"),
           "--predictions", str(npz), "--out_dir", str(out_dir)]
    for k, v in PROTOCOL.items():
        cmd += [f"--{k}", str(v)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"scoring failed on {npz}:\n{r.stdout}\n{r.stderr}")
    return json.load(open(Path(out_dir) / "sample_metrics.json"))


def empty_present_genera(labels, mask=None):
    """How many of the 60 genera a sample declares present hold no reads.

    Zero on the full pool. On the coverage-matched pool it is 8.3 of 60, which
    is why only read accuracy is taken from there.
    """
    lab = labels if mask is None else labels[mask]
    hist = np.bincount(lab, minlength=120)
    rng = np.random.default_rng(PROTOCOL["seed"])
    rng.permutation(len(lab))
    counts = []
    for _ in range(PROTOCOL["n_sparse_samples"]):
        pres = rng.choice(120, size=PROTOCOL["genera_present"], replace=False)
        counts.append(int((hist[pres] == 0).sum()))
        avail = np.concatenate([np.where(lab == g)[0] for g in pres])
        if len(avail) >= PROTOCOL["reads_per_sample"]:
            rng.choice(avail, size=PROTOCOL["reads_per_sample"], replace=False)
    return int((hist == 0).sum()), float(np.mean(counts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_fasta", default=DEFAULT_FA)
    ap.add_argument("--pool_labels", default=DEFAULT_TSV)
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=NPZ",
                    help="repeatable; NPZ must be predictions on this pool")
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    work = Path(args.work_dir); (work / "cmp").mkdir(parents=True, exist_ok=True)

    for path, what in ((args.pool_fasta, "pool FASTA"),
                       (args.pool_labels, "pool label TSV")):
        if not Path(path).exists():
            raise SystemExit(
                f"{what} not found: {path or '(empty -- KF_DATA is unset)'}\n"
                f"Set KF_DATA to the downloaded data bundle (see docs/DATA.md), "
                f"or pass --pool_fasta/--pool_labels explicitly.")
    got = md5(args.pool_fasta)
    print(f"pool: {args.pool_fasta}")
    print(f"      md5 {got}")
    print(f"      {POOLS.get(got, 'UNRECOGNISED POOL -- no printed number is known to use it')}")

    pool_labels, species = read_pool(args.pool_labels)
    mask = coverage_mask(species)
    n_empty_full, avg_full = empty_present_genera(pool_labels)
    n_empty_mask, avg_mask = empty_present_genera(pool_labels, mask)
    print(f"      {len(pool_labels):,} reads, "
          f"{len(set(pool_labels.tolist()))} genera, "
          f"{int(mask.sum()):,} in the Kraken 2 index\n")
    print(f"detection protocol feasibility")
    print(f"  full pool            : {n_empty_full} empty genera, "
          f"{avg_full:.1f} of {PROTOCOL['genera_present']} declared-present "
          f"genera hold no reads")
    print(f"  coverage-matched     : {n_empty_mask} empty genera, "
          f"{avg_mask:.1f} of {PROTOCOL['genera_present']} hold no reads"
          f"{'  <- read accuracy only' if n_empty_mask else ''}\n")

    if not args.arm:
        raise SystemExit("no --arm given")
    rows = []
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm wants NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        preds, labels = check_arm(name, path, pool_labels)
        npz = work / "cmp" / f"{name}.npz"
        np.savez(npz, preds=preds, labels=labels)
        m = score(npz, work / "cmp" / f"out_{name}")
        rows.append(dict(
            arm=name,
            preds_sha1=hashlib.sha1(preds.tobytes()).hexdigest()[:16],
            labels_sha1=hashlib.sha1(labels.tobytes()).hexdigest()[:16],
            read_acc_full=float((preds == labels).mean()),
            read_acc_in_db=float((preds[mask] == labels[mask]).mean()),
            unclassified_frac=float((preds < 0).mean()),
            pearson_r=m["abundance_estimation"]["pearson_r_mean"],
            bray_curtis=m["abundance_estimation"]["bray_curtis_mean"],
            roc_auc=m["roc_detection"]["auc"],
            sens_at_95spec=(m["roc_detection"]["operating_points"]
                            ["spec_95pct"]["sensitivity"])))

    rows.sort(key=lambda r: -r["read_acc_full"])
    w = max(len(r["arm"]) for r in rows) + 2
    hdr = (f"{'arm':<{w}}{'read acc':>10}{'in-DB':>9}{'Pearson r':>11}"
           f"{'Bray-C':>9}{'ROC AUC':>9}{'Sens@95':>9}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:<{w}}{r['read_acc_full']*100:>9.2f}%"
              f"{r['read_acc_in_db']*100:>8.2f}%{r['pearson_r']:>11.4f}"
              f"{r['bray_curtis']:>9.3f}{r['roc_auc']:>9.3f}"
              f"{r['sens_at_95spec']*100:>8.2f}%")
    print(f"\nprotocol: {PROTOCOL}")
    print("in-DB column restricts the same predictions to the coverage-matched "
          "reads; read accuracy only, for the reason above.")

    # Record the pool relative to KF_DATA where possible: an absolute path names
    # one machine's disk, and the md5 below is what identifies the pool anyway.
    pool_rel = str(args.pool_fasta)
    if KF_DATA and pool_rel.startswith(KF_DATA.rstrip("/") + "/"):
        pool_rel = "${KF_DATA}/" + pool_rel[len(KF_DATA.rstrip("/")) + 1:]
    out = dict(pool=pool_rel, pool_md5=got,
               pool_identity=POOLS.get(got, "unrecognised"),
               pool_labels_sha1=hashlib.sha1(
                   pool_labels.astype(np.int64).tobytes()).hexdigest()[:16],
               n_reads=int(len(pool_labels)), n_reads_in_db=int(mask.sum()),
               detection_feasibility=dict(
                   full_empty_genera=n_empty_full,
                   full_avg_empty_of_present=avg_full,
                   masked_empty_genera=n_empty_mask,
                   masked_avg_empty_of_present=avg_mask),
               protocol=PROTOCOL, rows=rows)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()

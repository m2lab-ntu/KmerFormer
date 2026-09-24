#!/usr/bin/env python3
"""Turn a Kraken 2 read-level output file into genus predictions for this pool.

WHY IT IS NOT A ONE-LINER. Three things about these databases make the naive
conversion silently wrong, and each cost a real mistake:

1. TWO INDEXES EXIST over the same 1,535-genome library, and they behave very
   differently. `kraken2_db_1535` has 1,535 species nodes hanging off root and
   no genus nodes; `kraken2_db_1535_genus` inserts 120 genus nodes between
   them. Without the genus nodes a read ambiguous between two species of one
   genus has nowhere to stop, climbs to root, and is scored unclassified --
   20.03% of this pool, costing 19.10 points (79.95% against 99.05%). The
   manuscript argues against the species-only build and uses the genus-aware
   one, so a row labelled by genome count ("1535DB") rather than by taxonomy
   tells you nothing about which you have. This script reports the rank
   histogram so the answer is in the output.

2. `species_class` DOES NOT MEAN THE SAME THING in the pool and in the index.
   The index's library filenames carry classes from the 2,505-species
   catalogue (up to 2503); the pool re-indexes densely to 0..1534. They agree
   for the first few entries, so a spot check passes and a join on
   `species_class + offset` is 91% wrong. The join has to go through
   `species_name`, which both sides carry.

3. `seq_id` IS NOT UNIQUE in this pool -- 979 ids repeat, 999 extra rows, in
   both the FASTA and the label TSV. So the join to labels must be positional,
   and this asserts the read names agree in order rather than assuming it.
"""

import argparse
import collections
import csv
import glob
import os
import re

import numpy as np

GENUS_TAXID_BASE = 10000


def load_taxonomy(db_dir):
    parent, rank = {}, {}
    with open(os.path.join(db_dir, "taxonomy", "nodes.dmp")) as f:
        for line in f:
            q = [x.strip() for x in line.split("|")]
            parent[int(q[0])], rank[int(q[0])] = int(q[1]), q[2]
    return parent, rank


def genus_taxid_to_class(db_dir, labels_tsv, parent, rank):
    """Derive genus_taxid -> pool genus_class, joining on species_name.

    Returns the map and refuses anything that is not a clean bijection: a genus
    taxid covering two pool genera means the join is wrong, not that the data
    is messy.
    """
    name2taxid = {}
    for path in sorted(glob.glob(os.path.join(db_dir, "library", "added",
                                              "*.fna"))):
        m = re.match(r"(.+)_taxid(\d+)\.fna$", os.path.basename(path))
        with open(path) as f:
            head = f.readline().strip()
        t = re.match(r">kraken:taxid\|(\d+)\|", head)
        if m and t:
            name2taxid[m.group(1)] = int(t.group(1))
    if not name2taxid:
        raise SystemExit(f"{db_dir}: no library/added/*.fna to join on")

    pool = {}
    with open(labels_tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            pool[r["species_name"]] = int(r["genus_class"])

    absent = set(pool) - set(name2taxid)
    if absent:
        raise SystemExit(f"{len(absent)} pool species are not in the index "
                         f"library (e.g. {sorted(absent)[:3]})")

    has_genus_nodes = any(r == "genus" for r in rank.values())
    if not has_genus_nodes:
        # Species-only index: every species hangs off root, so the taxonomy
        # cannot express a genus call. Genus is assigned post hoc from the
        # species the read was assigned to -- a step Kraken 2 did not take,
        # and one that cannot recover the reads that climbed to root.
        return {name2taxid[n]: gc for n, gc in pool.items()}, False

    cover = collections.defaultdict(set)
    for name, gc in pool.items():
        cover[parent[name2taxid[name]]].add(gc)
    bad = {k: v for k, v in cover.items() if len(v) != 1}
    if bad:
        raise SystemExit(f"{len(bad)} genus taxids span more than one pool "
                         f"genus_class -- the species_name join is wrong")
    gmap = {k: next(iter(v)) for k, v in cover.items()}
    if len(set(gmap.values())) != len(gmap):
        raise SystemExit("genus taxid -> genus_class is not injective")
    return gmap, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kraken_out", required=True, help="Kraken 2 --output file")
    ap.add_argument("--db", required=True, help="the index that produced it")
    ap.add_argument("--labels_tsv", required=True, help="pool label TSV")
    ap.add_argument("--out_npz", required=True)
    args = ap.parse_args()

    parent, rank = load_taxonomy(args.db)
    n_genus_nodes = sum(1 for r in rank.values() if r == "genus")
    print(f"index: {args.db}")
    print(f"  {sum(1 for r in rank.values() if r == 'species')} species nodes, "
          f"{n_genus_nodes} genus nodes "
          f"-> {'genus-aware' if n_genus_nodes else 'SPECIES-ONLY'} build")
    gmap, from_tree = genus_taxid_to_class(args.db, args.labels_tsv,
                                           parent, rank)
    if from_tree:
        print(f"  genus taxid -> genus_class: {len(gmap)} genera, bijective")
    else:
        print(f"  no genus nodes: genus assigned POST HOC from the species "
              f"call, over {len(gmap)} species. Kraken 2 made no genus call "
              f"here, and reads that climbed to root stay unclassified.")

    sids, labels = [], []
    with open(args.labels_tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            sids.append(r["seq_id"]); labels.append(int(r["genus_class"]))

    rows = [l.rstrip("\n").split("\t")
            for l in open(args.kraken_out) if l.strip()]
    if len(rows) != len(sids):
        raise SystemExit(f"{len(rows):,} Kraken 2 rows against "
                         f"{len(sids):,} pool reads")
    for i, (r, s) in enumerate(zip(rows, sids)):
        if r[1] != s:
            raise SystemExit(
                f"Kraken 2 output is not in pool order (row {i}: {r[1]!r} "
                f"against {s!r}). seq_id repeats in this pool, so the join "
                f"must be positional and the order must hold.")
    print(f"  positional join verified on {len(rows):,} read names")

    preds = np.empty(len(rows), dtype=np.int64)
    stats = collections.Counter()
    for i, r in enumerate(rows):
        tid = int(r[2])
        if tid in (0, 1):
            preds[i] = -1; stats["unclassified or root"] += 1
        elif tid in gmap:
            preds[i] = gmap[tid]
            stats["species-level call" if not from_tree
                  else "genus-level call"] += 1
        elif from_tree and rank.get(tid) == "species":
            g = gmap.get(parent[tid])
            if g is None:
                raise SystemExit(f"species taxid {tid} has parent "
                                 f"{parent[tid]}, which is not a pool genus")
            preds[i] = g; stats["species-level call"] += 1
        else:
            preds[i] = -1; stats[f"other rank ({rank.get(tid)})"] += 1
    for k, v in stats.most_common():
        print(f"  {k}: {v:,}")

    labels = np.array(labels, dtype=np.int64)
    print(f"\nread accuracy   : {float((preds == labels).mean()) * 100:.2f}%")
    print(f"unclassified    : {float((preds < 0).mean()) * 100:.2f}%")
    np.savez(args.out_npz, preds=preds, labels=labels)
    print(f"wrote {args.out_npz}")


if __name__ == "__main__":
    main()

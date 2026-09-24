#!/usr/bin/env python3
"""
Track B – Step 13 (E44): novelty-screen the novel-genus candidates.

A genome is only a useful "novel genus" probe if it is not, in fact, the same
organism as something in training under a different name.  Each candidate is
compared by ANI against all 1,535 training genomes and dropped when its closest
training genome is at >= 95% ANI (same species) -- which happens because NCBI
genus names and the UMGS/HGR genus assignments disagree for some clades.

Emits rows in the same schema tb_04 consumes, with level = L3_genus, so read
simulation and the pool build need no special case.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tb_03_ani_stratify import (  # noqa: E402
    read_tsv, read_tsv_pairs, run_skani, collapse_best, TRAIN_GENOME_DIR,
    SPECIES_ANI,
)

OOD_GENUS_CLASS = -1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--survey", required=True, help="candidate_survey_L3.tsv")
    p.add_argument("--training_species", required=True)
    p.add_argument("--genomes_dir", required=True)
    p.add_argument("--train_genome_dir", default=TRAIN_GENOME_DIR)
    p.add_argument("--out", required=True, help="output l3_ani_summary.tsv")
    p.add_argument("--raw_tsv", required=True, help="where to put skani output")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--screen_ani", type=float, default=70.0)
    p.add_argument("--reuse_raw", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    raw = Path(args.raw_tsv)
    raw.parent.mkdir(parents=True, exist_ok=True)

    cands = read_tsv(args.survey)
    root = Path(args.genomes_dir)
    query_paths, cand_by_path = [], {}
    missing = 0
    for r in cands:
        p = root / r["genus"] / f"{r['accession']}.fna"
        if not p.exists() or p.stat().st_size == 0:
            missing += 1
            continue
        query_paths.append(p)
        cand_by_path[str(p)] = r
    print(f"{len(query_paths)} novel-genus genomes present ({missing} missing)")

    train_genus, ref_paths = {}, []
    tdir = Path(args.train_genome_dir)
    for genus, ref in read_tsv_pairs(args.training_species):
        path = tdir / f"{ref}.fa.gz"
        if not path.exists():
            sys.exit(f"missing training genome: {path}")
        ref_paths.append(path)
        train_genus[str(path)] = genus
    print(f"{len(ref_paths)} training reference genomes")

    if not (args.reuse_raw and raw.exists()):
        run_skani(query_paths, ref_paths, raw, args.threads, args.screen_ani)
    _, best_any, n_pairs = collapse_best(raw, cand_by_path, train_genus)
    print(f"Parsed {n_pairs} skani pairs")

    cols = ["genus", "genus_class", "n_training_species_in_genus", "accession",
            "organism_name", "species_key", "named_species", "assembly_level",
            "contig_n50", "total_length", "family",
            "max_ani_same_genus", "closest_training_ref_same_genus",
            "align_frac_query_same_genus",
            "max_ani_any_genus", "closest_training_ref_any_genus",
            "closest_training_genus_any", "level", "cross_genus_leak", "keep"]

    rows, n_drop = [], 0
    for p in query_paths:
        key = str(p)
        c = cand_by_path[key]
        anyg = best_any.get(key)
        max_any = anyg[0] if anyg else None
        same_species_as_training = max_any is not None and max_any >= SPECIES_ANI
        if same_species_as_training:
            n_drop += 1
        rows.append({
            "genus": c["genus"],
            # No valid class exists for these reads; the sentinel keeps the
            # label column typed while making it obvious it is not a class id.
            "genus_class": OOD_GENUS_CLASS,
            "n_training_species_in_genus": 0,
            "accession": c["accession"],
            "organism_name": c["organism_name"],
            "species_key": "",
            "named_species": c.get("named_species", ""),
            "assembly_level": c["assembly_level"],
            "contig_n50": c["contig_n50"],
            "total_length": c["total_length"],
            "family": c.get("family", ""),
            # Same-genus columns are structurally empty: the genus is, by
            # construction, absent from the training set.
            "max_ani_same_genus": "",
            "closest_training_ref_same_genus": "",
            "align_frac_query_same_genus": "",
            "max_ani_any_genus": f"{max_any:.2f}" if max_any is not None else "",
            "closest_training_ref_any_genus":
                Path(anyg[1]).name.replace(".fa.gz", "") if anyg else "",
            "closest_training_genus_any":
                train_genus.get(anyg[1], "") if anyg else "",
            "level": "L3_genus",
            "cross_genus_leak": int(same_species_as_training),
            "keep": int(not same_species_as_training),
        })

    out = Path(args.out)
    with open(out, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    kept = [r for r in rows if r["keep"]]
    with_hit = [r for r in kept if r["max_ani_any_genus"]]
    print(f"\nkept {len(kept)}/{len(rows)} "
          f"({n_drop} dropped as >= {SPECIES_ANI}% ANI to a training genome)")
    print(f"  genera: {len({r['genus'] for r in kept})}   "
          f"families: {len({r['family'] for r in kept})}")
    if with_hit:
        anis = sorted(float(r["max_ani_any_genus"]) for r in with_hit)
        print(f"  closest-training ANI among kept: {anis[0]:.1f}-{anis[-1]:.1f}% "
              f"(median {anis[len(anis)//2]:.1f}%), "
              f"{len(kept)-len(with_hit)} with no hit at all")
    print(f"  {out}")


if __name__ == "__main__":
    main()

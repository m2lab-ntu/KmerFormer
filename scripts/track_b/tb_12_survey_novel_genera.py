#!/usr/bin/env python3
"""
Track B – Step 12 (E44): shortlist genomes from genera the model has never seen.

The 120 training genera are mapped to their families via the UMGS/HGR taxonomy
tables, then each family is queried on NCBI for *sister* genera that are absent
from the training label space.  Sister genera (rather than arbitrary bacteria)
make this a hard rejection test: the reads come from the same neighbourhood of
the tree, so a detector cannot win on gross compositional differences alone.

There is no correct genus label for these reads -- the answer is "none of the
120" -- so E44 scores rejection, never accuracy.

Output
------
  candidate_survey_L3.tsv   tb_02-compatible download list
  l3_family_survey.tsv      per-family: what was found, what was kept
"""

import os
import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

TAXONOMY_DIR = os.environ.get(
    "KF_TAXONOMY", os.path.join(os.environ.get("KF_DATA", "."), "taxonomy_files"))
OOD_GENUS_CLASS = -1

LEVEL_RANK = {"Complete Genome": 0, "Chromosome": 1, "Scaffold": 2, "Contig": 3}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--training_species", required=True)
    p.add_argument("--taxonomy_dir", default=TAXONOMY_DIR)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_genera_per_family", type=int, default=3)
    p.add_argument("--min_n50", type=int, default=50_000)
    return p.parse_args()


def load_family_map(taxonomy_dir: str, train_genera: set[str]) -> dict[str, str]:
    g2f = {}
    for name in ("taxonomy_umgs.tab", "taxonomy_hgr.tab"):
        with open(Path(taxonomy_dir) / name) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                g, f = row.get("Genus", "").strip(), row.get("Family", "").strip()
                if g in train_genera and f:
                    g2f.setdefault(g, f)
    missing = train_genera - set(g2f)
    if missing:
        sys.exit(f"no family for {len(missing)} training genera: {sorted(missing)[:5]}")
    return g2f


def datasets_summary(taxon: str, levels: str) -> list[dict]:
    cmd = ["datasets", "summary", "genome", "taxon", taxon,
           "--assembly-source", "RefSeq", "--assembly-level", levels,
           "--as-json-lines"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print(f"  ! timeout on {taxon}", file=sys.stderr)
        return []
    if proc.returncode != 0:
        print(f"  ! {taxon}: {proc.stderr.strip()[:120]}", file=sys.stderr)
        return []
    out = []
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def genus_of(organism_name: str) -> str | None:
    name = re.sub(r"^\[|\]", "", organism_name.strip())
    name = re.sub(r"^Candidatus\s+", "", name)
    tok = name.split()
    if not tok:
        return None
    g = tok[0]
    return g if re.fullmatch(r"[A-Z][a-z]+", g) else None


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_genera = {l.split("\t")[0] for l in open(args.training_species) if l.strip()}
    g2f = load_family_map(args.taxonomy_dir, train_genera)
    families = sorted(set(g2f.values()))
    print(f"{len(train_genera)} training genera in {len(families)} families")

    cand_rows, fam_rows = [], []
    seen_genera: set[str] = set()

    for i, family in enumerate(families, 1):
        levels = "complete"
        recs = datasets_summary(family, levels)
        if not recs:
            levels = "complete,chromosome"
            recs = datasets_summary(family, levels)

        by_genus: dict[str, list[dict]] = defaultdict(list)
        for rec in recs:
            org = rec.get("organism", {}).get("organism_name", "")
            g = genus_of(org)
            if g is None or g in train_genera or g in seen_genera:
                continue
            info = rec.get("assembly_info", {})
            stats = rec.get("assembly_stats", {})
            n50 = stats.get("contig_n50", 0) or 0
            if n50 < args.min_n50:
                continue
            by_genus[g].append({
                "genus": g,
                "accession": rec.get("accession", ""),
                "organism_name": org,
                "species_key": "",
                "named_species": False,
                "tax_id": rec.get("organism", {}).get("tax_id", ""),
                "assembly_level": info.get("assembly_level", ""),
                "refseq_category": info.get("refseq_category", ""),
                "contig_n50": n50,
                "total_length": stats.get("total_sequence_length", 0) or 0,
                "family": family,
            })

        def best(r):
            return (0 if r["refseq_category"] == "reference genome" else 1,
                    LEVEL_RANK.get(r["assembly_level"], 9), -r["contig_n50"])

        picked = []
        for g in sorted(by_genus, key=lambda g: best(sorted(by_genus[g], key=best)[0])):
            if len(picked) >= args.max_genera_per_family:
                break
            row = dict(sorted(by_genus[g], key=best)[0])
            row["genus_class"] = OOD_GENUS_CLASS
            row["n_training_species_in_genus"] = 0
            row["accession_base"] = re.match(r"(GC[AF]_\d+)", row["accession"]).group(1)
            row["species_key_resolved"] = f"__ood__{row['accession_base']}"
            row["query_levels"] = levels
            picked.append(row)
            seen_genera.add(g)
        cand_rows.extend(picked)

        fam_rows.append({"family": family, "query_levels": levels,
                         "n_assemblies": len(recs),
                         "n_novel_genera_found": len(by_genus),
                         "n_genera_picked": len(picked),
                         "picked": ",".join(r["genus"] for r in picked)})
        print(f"[{i:2d}/{len(families)}] {family:<26} assemblies={len(recs):<5} "
              f"novel_genera={len(by_genus):<4} picked={len(picked)}", flush=True)

    cols = ["genus", "genus_class", "n_training_species_in_genus", "accession",
            "accession_base", "organism_name", "species_key",
            "species_key_resolved", "named_species", "tax_id", "assembly_level",
            "refseq_category", "contig_n50", "total_length", "query_levels",
            "family"]
    cand_path = out_dir / "candidate_survey_L3.tsv"
    with open(cand_path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in cand_rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    fam_cols = ["family", "query_levels", "n_assemblies", "n_novel_genera_found",
                "n_genera_picked", "picked"]
    fam_path = out_dir / "l3_family_survey.tsv"
    with open(fam_path, "w") as fh:
        fh.write("\t".join(fam_cols) + "\n")
        for r in fam_rows:
            fh.write("\t".join(str(r[c]) for c in fam_cols) + "\n")

    print(f"\n{len(cand_rows)} novel-genus genomes over "
          f"{len({r['family'] for r in cand_rows})} families")
    print(f"  {cand_path}\n  {fam_path}")


if __name__ == "__main__":
    main()

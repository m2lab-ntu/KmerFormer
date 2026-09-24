#!/usr/bin/env python3
"""
Track B – Step 1: survey NCBI RefSeq for candidate genomes per training genus.

For each of the 120 training genera, list RefSeq assemblies and pick a
species-diverse candidate set.  Nothing is downloaded here; this only produces
the shortlist so we can see how many genera can actually support a
novel-species (L2) test before spending bandwidth.

Species assignment is by NCBI organism name.  This is ONLY used to diversify
the candidate shortlist -- the authoritative novel-species call is made later
by ANI against the 1,535 training genomes (tb_03), because 1,179/1,535
training references are UMGS MAGs with no species name at all.

Output
------
  candidate_survey.tsv   one row per shortlisted assembly
  survey_by_genus.tsv    one row per genus (counts, whether it is usable)
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

LEVEL_RANK = {
    "Complete Genome": 0,
    "Chromosome": 1,
    "Scaffold": 2,
    "Contig": 3,
}

# Some training genus names cannot be handed to `datasets ... taxon <name>` as-is.
#   Bacillus            -- exact-match collision with the walking-stick genus
#                          (taxid 55087); the firmicute we want is taxid 1386.
#   Massiliomicrobiota  -- training-set spelling; NCBI's current genus name is
#                          Massilimicrobiota (note: Massiliimicrobium is a
#                          *different* genus and must not be accepted).
GENUS_QUERY_OVERRIDE = {
    "Bacillus": "1386",
    "Massiliomicrobiota": "Massilimicrobiota",
}

# Organism-name prefixes accepted in addition to the training genus name.
GENUS_NAME_ALIASES = {
    "Bacillus": ["Bacillus"],
    "Massiliomicrobiota": ["Massilimicrobiota"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--genus_map", required=True,
                   help="TSV with genus_name, genus_class (120 rows)")
    p.add_argument("--training_species", required=True,
                   help="TSV (genus<TAB>species_ref) for the 1,535 training refs")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_species_per_genus", type=int, default=8)
    p.add_argument("--min_n50", type=int, default=50_000,
                   help="Drop fragmented assemblies below this contig N50")
    p.add_argument("--genus", action="append",
                   help="Restrict to these genera (repeatable); default all")
    return p.parse_args()


def accession_base(value: str) -> str:
    m = re.match(r"(GC[AF]_\d+)", value)
    return m.group(1) if m else value.split(".", 1)[0]


def species_key(organism_name: str) -> str | None:
    """Reduce an NCBI organism name to a species-level key.

    Returns None for names that do not pin down a species (``sp.``,
    ``uncultured``, ...), so they never crowd out named species in the
    shortlist -- they are still fine as ANI-defined novel species, they just
    should not be *preferred*.
    """
    name = organism_name.strip()
    name = re.sub(r"^\[|\]", "", name)
    name = re.sub(r"^Candidatus\s+", "", name)
    parts = name.split()
    if len(parts) < 2:
        return None
    genus, epithet = parts[0], parts[1]
    if epithet in {"sp.", "sp"} or epithet.startswith("sp."):
        return None
    if not re.fullmatch(r"[a-z][a-z0-9-]+", epithet):
        return None
    return f"{genus} {epithet}"


def datasets_summary(genus: str, levels: str) -> list[dict]:
    """Run `datasets summary genome taxon` and return parsed JSON lines."""
    query = GENUS_QUERY_OVERRIDE.get(genus, genus)
    cmd = [
        "datasets", "summary", "genome", "taxon", query,
        "--assembly-source", "RefSeq",
        "--assembly-level", levels,
        "--as-json-lines",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        print(f"  ! datasets failed for {genus}: {proc.stderr.strip()[:200]}",
              file=sys.stderr)
        return []
    records = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def flatten(rec: dict, genus: str) -> dict | None:
    acc = rec.get("accession")
    if not acc:
        return None
    org = rec.get("organism", {})
    info = rec.get("assembly_info", {})
    stats = rec.get("assembly_stats", {})
    organism_name = org.get("organism_name", "")
    # Guard against taxon queries pulling in look-alike genera (e.g. querying
    # Massilimicrobiota also returns the unrelated genus Massiliimicrobium).
    accepted = [genus, *GENUS_NAME_ALIASES.get(genus, [])]
    clean_name = organism_name.replace("[", "")
    if not any(clean_name.startswith(a) for a in accepted):
        return None
    return {
        "genus": genus,
        "accession": acc,
        "accession_base": accession_base(acc),
        "organism_name": organism_name,
        "species_key": species_key(organism_name) or "",
        "tax_id": org.get("tax_id", ""),
        "assembly_level": info.get("assembly_level", ""),
        "refseq_category": info.get("refseq_category", ""),
        "contig_n50": stats.get("contig_n50", 0) or 0,
        "total_length": stats.get("total_sequence_length", 0) or 0,
    }


def pick_candidates(rows: list[dict], max_species: int, min_n50: int) -> list[dict]:
    """One best assembly per species key, prioritising named species."""
    usable = [r for r in rows if r["contig_n50"] >= min_n50]
    if not usable and rows:
        # Some genera only have fragmented assemblies in RefSeq; keeping them
        # beats dropping the genus entirely.  assembly_level / contig_n50 stay
        # in the output so the relaxation is auditable downstream.
        print(f"  ~ relaxing N50 floor for {rows[0]['genus']} "
              f"(best N50 = {max(r['contig_n50'] for r in rows):,})")
        usable = rows
    by_species: dict[str, list[dict]] = defaultdict(list)
    for r in usable:
        # Unnamed organisms are kept but bucketed by accession so that several
        # distinct "sp." genomes can each survive as their own candidate.
        key = r["species_key"] or f"__unnamed__{r['accession_base']}"
        by_species[key].append(r)

    def assembly_score(r: dict):
        return (
            0 if r["refseq_category"] == "reference genome" else 1,
            LEVEL_RANK.get(r["assembly_level"], 9),
            -r["contig_n50"],
        )

    best = []
    for key, group in by_species.items():
        group.sort(key=assembly_score)
        chosen = dict(group[0])
        chosen["species_key_resolved"] = key
        chosen["named_species"] = bool(chosen["species_key"])
        best.append(chosen)

    # Named species first, then better assemblies -- keeps the shortlist
    # interpretable while still allowing MAG-like "sp." genomes to fill gaps.
    best.sort(key=lambda r: (0 if r["named_species"] else 1, assembly_score(r)))
    return best[:max_species]


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genera = []
    genus_class = {}
    with open(args.genus_map) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        gi, ci = header.index("genus_name"), header.index("genus_class")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(gi, ci):
                continue
            genera.append(parts[gi])
            genus_class[parts[gi]] = parts[ci]

    train_by_genus: dict[str, set[str]] = defaultdict(set)
    train_gcf: set[str] = set()
    with open(args.training_species) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            g, ref = parts[0], parts[1]
            train_by_genus[g].add(ref)
            if ref.startswith("GC"):
                train_gcf.add(accession_base(ref))

    if args.genus:
        genera = [g for g in genera if g in set(args.genus)]
    print(f"Surveying {len(genera)} genera; {len(train_gcf)} training GCF "
          f"accessions will be excluded.")

    cand_rows, genus_rows = [], []
    for i, genus in enumerate(genera, 1):
        n_train = len(train_by_genus.get(genus, ()))
        levels = "complete"
        rows = [r for r in (flatten(rec, genus)
                            for rec in datasets_summary(genus, levels))
                if r]
        rows = [r for r in rows if r["accession_base"] not in train_gcf]
        n_named = len({r["species_key"] for r in rows if r["species_key"]})

        # Widen only when complete genomes cannot give us species diversity.
        if n_named < 3:
            levels = "complete,chromosome"
            rows = [r for r in (flatten(rec, genus)
                                for rec in datasets_summary(genus, levels))
                    if r]
            rows = [r for r in rows if r["accession_base"] not in train_gcf]
            n_named = len({r["species_key"] for r in rows if r["species_key"]})
        if n_named < 2 and len(rows) < 2:
            levels = "complete,chromosome,scaffold"
            rows = [r for r in (flatten(rec, genus)
                                for rec in datasets_summary(genus, levels))
                    if r]
            rows = [r for r in rows if r["accession_base"] not in train_gcf]
        if not rows:
            # Last resort: several under-studied gut genera only have
            # contig-level assemblies in RefSeq.
            levels = "complete,chromosome,scaffold,contig"
            rows = [r for r in (flatten(rec, genus)
                                for rec in datasets_summary(genus, levels))
                    if r]
            rows = [r for r in rows if r["accession_base"] not in train_gcf]

        picked = pick_candidates(rows, args.max_species_per_genus, args.min_n50)
        for r in picked:
            r["genus_class"] = genus_class[genus]
            r["n_training_species_in_genus"] = n_train
            r["query_levels"] = levels
            cand_rows.append(r)

        genus_rows.append({
            "genus": genus,
            "genus_class": genus_class[genus],
            "n_training_species_in_genus": n_train,
            "query_levels": levels,
            "n_refseq_assemblies": len(rows),
            "n_named_species": len({r["species_key"] for r in rows if r["species_key"]}),
            "n_candidates_picked": len(picked),
        })
        print(f"[{i:3d}/{len(genera)}] {genus:<28} train_sp={n_train:<4} "
              f"refseq={len(rows):<5} named_sp={genus_rows[-1]['n_named_species']:<4} "
              f"picked={len(picked)}", flush=True)

    cand_cols = ["genus", "genus_class", "n_training_species_in_genus",
                 "accession", "accession_base", "organism_name", "species_key",
                 "species_key_resolved", "named_species", "tax_id",
                 "assembly_level", "refseq_category", "contig_n50",
                 "total_length", "query_levels"]
    cand_path = out_dir / "candidate_survey.tsv"
    with open(cand_path, "w") as fh:
        fh.write("\t".join(cand_cols) + "\n")
        for r in cand_rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cand_cols) + "\n")

    genus_cols = ["genus", "genus_class", "n_training_species_in_genus",
                  "query_levels", "n_refseq_assemblies", "n_named_species",
                  "n_candidates_picked"]
    genus_path = out_dir / "survey_by_genus.tsv"
    with open(genus_path, "w") as fh:
        fh.write("\t".join(genus_cols) + "\n")
        for r in genus_rows:
            fh.write("\t".join(str(r.get(c, "")) for c in genus_cols) + "\n")

    usable = sum(1 for r in genus_rows if r["n_candidates_picked"] >= 1)
    print(f"\nWrote {len(cand_rows)} candidates over {usable}/{len(genera)} genera")
    print(f"  {cand_path}\n  {genus_path}")


if __name__ == "__main__":
    main()

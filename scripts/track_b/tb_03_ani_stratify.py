#!/usr/bin/env python3
"""
Track B – Step 3: ANI-stratify the candidate genomes into the OOD ladder.

Every downloaded candidate is compared with **all 1,535 training reference
genomes** (not just those of its own genus) using skani.  Two numbers come out
of that:

  max_ani_same_genus  -- distance to the closest training genome of the same
                         genus.  This is what defines the ladder level.
  max_ani_any_genus   -- distance to the closest training genome anywhere.
                         Used as a leakage guard: if a candidate is >=99% ANI
                         to *some* training genome of a different genus, the
                         label itself is suspect and the candidate is dropped.

Ladder assignment (ANI species boundary = 95%, the standard threshold):

  >= 99            L0_dup      near-duplicate of training -> DROPPED
  95 <= ani < 99   L1_strain   same species, unseen strain
  80 <= ani < 95   L2_species  novel species, known genus      <- main target
  no hit / < 80    L2_far      novel species, distant from any training genome

Note: skani reports nothing below its screening threshold, so "no hit" and
"< threshold" are the same observation; both land in L2_far and are reported
separately from L2_species so the two are never silently merged.

Output
------
  ani/skani_raw.tsv       raw skani dist output
  ani_summary.tsv         one row per candidate with ANI + assigned level
  ani_ladder_counts.tsv   per-level and per-genus counts
"""

import os
import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

TRAIN_GENOME_DIR = os.environ.get(
    "KF_GENOMES", os.path.join(os.environ.get("KF_DATA", "."), "reference_genomes"))

DUP_ANI = 99.0
SPECIES_ANI = 95.0
FAR_ANI = 80.0

# Leakage guard threshold.  Set at the species boundary, not at DUP_ANI: two
# genomes above 95% ANI are the same species and must share a genus, so if the
# nearest training genome carries a *different* genus name the ground-truth
# label for this candidate is contested and it cannot be scored fairly.
LEAK_ANI = SPECIES_ANI

# Audit pass: skani's default --min-af 15 is what keeps L2_far honest.  Dropping
# it surfaces "hits" at 80-96% ANI computed over 0.03-7% of the genome (a few
# conserved genes), which would wrongly promote distant novel species into
# L1_strain.  We record those numbers rather than use them, so the question
# "is L2_far real or just a screening artefact?" is answerable from the table.
LOWAF_SCREEN_ANI = 40.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--survey", required=True, help="candidate_survey.tsv")
    p.add_argument("--training_species", required=True,
                   help="TSV genus<TAB>species_ref for the 1,535 training refs")
    p.add_argument("--genomes_dir", required=True, help="downloaded genomes/ root")
    p.add_argument("--train_genome_dir", default=TRAIN_GENOME_DIR)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--screen_ani", type=float, default=70.0,
                   help="skani -s screening threshold")
    p.add_argument("--reuse_raw", action="store_true",
                   help="Reuse ani/skani_raw.tsv instead of re-running skani")
    return p.parse_args()


def read_tsv(path: str) -> list[dict]:
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        rows = []
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            rows.append(dict(zip(header, parts)))
    return rows


def run_skani(query_paths: list[Path], ref_paths: list[Path], out_tsv: Path,
              threads: int, screen_ani: float, min_af: float | None = None) -> None:
    tmp_dir = out_tsv.parent
    suffix = out_tsv.stem
    ql = tmp_dir / f"query_list_{suffix}.txt"
    rl = tmp_dir / f"ref_list_{suffix}.txt"
    ql.write_text("\n".join(str(p) for p in query_paths) + "\n")
    rl.write_text("\n".join(str(p) for p in ref_paths) + "\n")
    cmd = [
        "skani", "dist",
        "--ql", str(ql), "--rl", str(rl),
        "-t", str(threads),
        "-s", str(screen_ani),
        "-o", str(out_tsv),
    ]
    if min_af is not None:
        cmd += ["--min-af", str(min_af)]
    print(f"  skani dist: {len(query_paths)} queries x {len(ref_paths)} refs "
          f"(-s {screen_ani}"
          + (f", --min-af {min_af}" if min_af is not None else "")
          + f", -t {threads})", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"skani failed:\n{proc.stderr[-2000:]}")
    print(f"  -> {out_tsv}")


def assign_level(max_same_genus: float | None) -> str:
    if max_same_genus is None:
        return "L2_far"
    if max_same_genus >= DUP_ANI:
        return "L0_dup"
    if max_same_genus >= SPECIES_ANI:
        return "L1_strain"
    if max_same_genus >= FAR_ANI:
        return "L2_species"
    return "L2_far"


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ani_dir = out_dir / "ani"
    ani_dir.mkdir(parents=True, exist_ok=True)

    candidates = read_tsv(args.survey)
    genomes_root = Path(args.genomes_dir)

    # Resolve candidate FASTA paths; skip anything that failed to download.
    query_paths, cand_by_path = [], {}
    missing = 0
    for row in candidates:
        path = genomes_root / row["genus"] / f"{row['accession']}.fna"
        if not path.exists() or path.stat().st_size == 0:
            missing += 1
            continue
        query_paths.append(path)
        cand_by_path[str(path)] = row
    print(f"{len(query_paths)} candidate genomes present ({missing} missing).")

    # All 1,535 training references, with their genus.
    train_genus: dict[str, str] = {}
    ref_paths = []
    train_dir = Path(args.train_genome_dir)
    for row in read_tsv_pairs(args.training_species):
        genus, ref = row
        path = train_dir / f"{ref}.fa.gz"
        if not path.exists():
            sys.exit(f"missing training genome: {path}")
        ref_paths.append(path)
        train_genus[str(path)] = genus
    print(f"{len(ref_paths)} training reference genomes.")

    raw_tsv = ani_dir / "skani_raw.tsv"
    if not (args.reuse_raw and raw_tsv.exists()):
        run_skani(query_paths, ref_paths, raw_tsv, args.threads, args.screen_ani)
    else:
        print(f"  reusing {raw_tsv}")

    best_same, best_any, n_pairs = collapse_best(raw_tsv, cand_by_path, train_genus)
    print(f"Parsed {n_pairs} skani pairs above the screening threshold.")

    lowaf_tsv = ani_dir / "skani_raw_lowaf.tsv"
    if not (args.reuse_raw and lowaf_tsv.exists()):
        run_skani(query_paths, ref_paths, lowaf_tsv, args.threads,
                  LOWAF_SCREEN_ANI, min_af=0.0)
    lowaf_same, _, n_lowaf = collapse_best(lowaf_tsv, cand_by_path, train_genus)
    print(f"Audit pass (--min-af 0): {n_lowaf} pairs.")

    build_summary(args, out_dir, query_paths, cand_by_path, train_genus,
                  best_same, best_any, lowaf_same)


def collapse_best(raw_tsv: Path, cand_by_path: dict, train_genus: dict):
    """Reduce skani pairs to the best same-genus and best any-genus hit."""
    best_same: dict[str, tuple[float, str, float, float]] = {}
    best_any: dict[str, tuple[float, str, float, float]] = {}
    n_pairs = 0
    with open(raw_tsv) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            n_pairs += 1
            q = parts[idx["Query_file"]]
            r = parts[idx["Ref_file"]]
            ani = float(parts[idx["ANI"]])
            af_q = float(parts[idx["Align_fraction_query"]])
            af_r = float(parts[idx["Align_fraction_ref"]])
            cand = cand_by_path.get(q)
            if cand is None:
                continue
            rec = (ani, r, af_q, af_r)
            if q not in best_any or ani > best_any[q][0]:
                best_any[q] = rec
            if train_genus.get(r) == cand["genus"]:
                if q not in best_same or ani > best_same[q][0]:
                    best_same[q] = rec
    return best_same, best_any, n_pairs


def build_summary(args, out_dir: Path, query_paths, cand_by_path, train_genus,
                  best_same, best_any, lowaf_same):
    out_cols = [
        "genus", "genus_class", "n_training_species_in_genus",
        "accession", "organism_name", "species_key", "named_species",
        "assembly_level", "contig_n50", "total_length",
        "max_ani_same_genus", "closest_training_ref_same_genus",
        "align_frac_query_same_genus",
        "max_ani_any_genus", "closest_training_ref_any_genus",
        "closest_training_genus_any",
        "audit_ani_same_genus_minaf0", "audit_align_frac_minaf0",
        "level", "cross_genus_leak", "keep",
    ]
    rows_out = []
    for path in query_paths:
        key = str(path)
        cand = cand_by_path[key]
        same = best_same.get(key)
        anyg = best_any.get(key)
        max_same = same[0] if same else None
        level = assign_level(max_same)

        closest_any_genus = train_genus.get(anyg[1], "") if anyg else ""
        # Leakage guard: same species as a training genome that carries a
        # different genus name -> contested ground truth, drop it.
        leak = bool(anyg and anyg[0] >= LEAK_ANI
                    and closest_any_genus != cand["genus"])
        keep = level in {"L1_strain", "L2_species", "L2_far"} and not leak
        audit = lowaf_same.get(key)

        rows_out.append({
            "genus": cand["genus"],
            "genus_class": cand["genus_class"],
            "n_training_species_in_genus": cand["n_training_species_in_genus"],
            "accession": cand["accession"],
            "organism_name": cand["organism_name"],
            "species_key": cand["species_key"],
            "named_species": cand["named_species"],
            "assembly_level": cand["assembly_level"],
            "contig_n50": cand["contig_n50"],
            "total_length": cand["total_length"],
            "max_ani_same_genus": f"{max_same:.2f}" if max_same is not None else "",
            "closest_training_ref_same_genus":
                Path(same[1]).name.replace(".fa.gz", "") if same else "",
            "align_frac_query_same_genus": f"{same[2]:.2f}" if same else "",
            "max_ani_any_genus": f"{anyg[0]:.2f}" if anyg else "",
            "closest_training_ref_any_genus":
                Path(anyg[1]).name.replace(".fa.gz", "") if anyg else "",
            "closest_training_genus_any": closest_any_genus,
            "audit_ani_same_genus_minaf0": f"{audit[0]:.2f}" if audit else "",
            "audit_align_frac_minaf0": f"{audit[2]:.2f}" if audit else "",
            "level": level,
            "cross_genus_leak": int(leak),
            "keep": int(keep),
        })

    summary_path = out_dir / "ani_summary.tsv"
    with open(summary_path, "w") as fh:
        fh.write("\t".join(out_cols) + "\n")
        for r in rows_out:
            fh.write("\t".join(str(r[c]) for c in out_cols) + "\n")

    # Ladder counts, overall and per genus.
    level_counts = defaultdict(int)
    genus_level = defaultdict(lambda: defaultdict(int))
    for r in rows_out:
        level_counts[r["level"]] += 1
        genus_level[r["genus"]][r["level"]] += 1
    levels = ["L0_dup", "L1_strain", "L2_species", "L2_far"]

    counts_path = out_dir / "ani_ladder_counts.tsv"
    with open(counts_path, "w") as fh:
        fh.write("genus\t" + "\t".join(levels) + "\tleak\n")
        for genus in sorted(genus_level):
            leak = sum(1 for r in rows_out
                       if r["genus"] == genus and r["cross_genus_leak"])
            fh.write(genus + "\t"
                     + "\t".join(str(genus_level[genus][lv]) for lv in levels)
                     + f"\t{leak}\n")

    print("\nLadder totals")
    for lv in levels:
        print(f"  {lv:<12} {level_counts[lv]:>4}")
    n_leak = sum(r["cross_genus_leak"] for r in rows_out)
    n_keep = sum(r["keep"] for r in rows_out)
    print(f"  cross-genus leak dropped: {n_leak}")
    print(f"  kept for read simulation: {n_keep}")
    genera_l2 = {r["genus"] for r in rows_out
                 if r["level"] in {"L2_species", "L2_far"} and r["keep"]}
    genera_l1 = {r["genus"] for r in rows_out
                 if r["level"] == "L1_strain" and r["keep"]}
    print(f"  genera with >=1 L2 genome: {len(genera_l2)}")
    print(f"  genera with >=1 L1 genome: {len(genera_l1)}")

    # Evidence that L2_far is genuine distance rather than a screening artefact:
    # the audit pass does find hits, but over a negligible slice of the genome.
    far = [r for r in rows_out if r["level"] == "L2_far"]
    far_audit = [r for r in far if r["audit_align_frac_minaf0"]]
    if far_audit:
        afs = sorted(float(r["audit_align_frac_minaf0"]) for r in far_audit)
        anis = [float(r["audit_ani_same_genus_minaf0"]) for r in far_audit]
        print(f"\nL2_far audit ({len(far_audit)}/{len(far)} have a --min-af 0 hit)")
        print(f"  ANI would read {min(anis):.1f}-{max(anis):.1f}% but over only "
              f"{afs[0]:.2f}-{afs[-1]:.2f}% of the query genome "
              f"(median {afs[len(afs)//2]:.2f}%)")
        print("  -> below skani's --min-af 15 floor; treated as no reliable hit")
    print(f"\n  {summary_path}\n  {counts_path}")


def read_tsv_pairs(path: str):
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                yield parts[0], parts[1]


if __name__ == "__main__":
    main()

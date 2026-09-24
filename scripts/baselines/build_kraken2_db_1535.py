#!/usr/bin/env python3
"""
Build a matched-reference Kraken2 DB from the full 1,535-species catalogue
(UMGS + HGR + GCF) from the reference genome directory (see docs/DATA.md).

Taxid mapping matches the existing evaluation pipeline:
    taxid = species_class + TAXID_OFFSET   (default OFFSET=2)

Also writes library/added/prelim_map.txt required by kraken2-build.

Usage:
    python scripts/baselines/build_kraken2_db_1535.py \
        --labels  $KF_DATA/val_100K/labels_100K_val.tsv \
        --genomes $KF_DATA/reference_genomes \
        --db_dir  $KF_OUT/kraken2_db_1535 \
        --threads 16
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

DEFAULT_TAXID_OFFSET = 2
HEADER_TAXID_RE = re.compile(r"^>kraken:taxid\|(\d+)\|(\S+)")


def build_taxonomy(db_dir: Path, species_classes: set, offset: int,
                   species_to_genus=None, genus_names=None) -> None:
    """Build genus-aware taxonomy, or an explicit species-only control.

    Species IDs remain class + offset; genus IDs are 10000 + genus_class,
    matching kraken2_genus_calls.py. Never replace a different existing taxonomy.
    """
    tax_dir = db_dir / "taxonomy"
    taxids = sorted(int(c) + offset for c in species_classes)
    if not taxids or min(taxids) <= 1:
        raise ValueError("species taxids must be nonempty and greater than root taxid 1")
    genus_names = genus_names or {}
    genera = sorted(set(species_to_genus.values())) if species_to_genus is not None else []
    if species_to_genus is not None and set(species_to_genus) != set(species_classes):
        raise ValueError("every species must map to exactly one genus")
    if any(g < 0 for g in genera) or set(taxids) & {10000 + g for g in genera}:
        raise ValueError("invalid or colliding species/genus taxids")
    nodes = ["1\t|\t1\t|\tno rank\t|\t\t|\t8\t|\t0\t|\t1\t|\t0\t|\t0\t|\t0\t|\t0\t|\t0\t|\t\t|\n"]
    names = ["1\t|\troot\t|\t\t|\tscientific name\t|\n"]
    for genus in genera:
        name = str(genus_names.get(genus, f"genus_{genus}"))
        if any(c in name for c in "\t\r\n|"):
            raise ValueError("genus name contains a taxonomy delimiter")
        nodes.append(f"{10000 + genus}\t|\t1\t|\tgenus\t|\t\t|\t0\t|\t1\t|\t11\t|\t1\t|\t0\t|\t1\t|\t0\t|\t0\t|\t\t|\n")
        names.append(f"{10000 + genus}\t|\t{name}\t|\t\t|\tscientific name\t|\n")
    for taxid in taxids:
        parent = 1 if species_to_genus is None else 10000 + species_to_genus[taxid - offset]
        nodes.append(f"{taxid}\t|\t{parent}\t|\tspecies\t|\t\t|\t0\t|\t1\t|\t11\t|\t1\t|\t0\t|\t1\t|\t0\t|\t0\t|\t\t|\n")
        names.append(f"{taxid}\t|\tspecies_{taxid}\t|\t\t|\tscientific name\t|\n")
    content = {"nodes.dmp": "".join(nodes), "names.dmp": "".join(names)}
    for name, text in content.items():
        path = tax_dir / name
        if path.exists() and path.read_text() != text:
            raise ValueError(f"refusing to replace different taxonomy: {path}; use a new --db_dir")
    tax_dir.mkdir(parents=True, exist_ok=True)
    for name, text in content.items():
        (tax_dir / name).write_text(text)
    print(f"  Wrote taxonomy: {len(taxids)} species, {len(genera)} genera → {tax_dir}")


def _prepare_genome(args):
    genome_path_str, taxid, out_path_str, force = args
    genome_path = Path(genome_path_str)
    out_path = Path(out_path_str)
    if (not force) and out_path.exists() and out_path.stat().st_size > 0:
        # Verify header taxid matches (guards against prior bad builds)
        with open(out_path) as fh:
            first = fh.readline()
        m = HEADER_TAXID_RE.match(first)
        if m and int(m.group(1)) == int(taxid):
            return True, genome_path.name, "skip"
    try:
        opener = gzip.open if genome_path.suffix == ".gz" else open
        mode = "rt" if genome_path.suffix == ".gz" else "r"
        with opener(genome_path, mode) as fh, open(out_path, "w") as out:
            for line in fh:
                if line.startswith(">"):
                    seq_id = line[1:].split()[0].rstrip()
                    out.write(f">kraken:taxid|{taxid}|{seq_id}\n")
                else:
                    out.write(line)
        return True, genome_path.name, "ok"
    except Exception as e:
        return False, f"{genome_path.name}: {e}", "fail"


def write_prelim_map(lib_dir: Path) -> int:
    """Write library/added/prelim_map.txt from kraken:taxid headers."""
    out = lib_dir / "prelim_map_all.txt"
    n = 0
    with open(out, "w") as dest:
        for fa in sorted(lib_dir.glob("*.fna")):
            with open(fa) as fh:
                for line in fh:
                    if not line.startswith(">"):
                        continue
                    m = HEADER_TAXID_RE.match(line)
                    if not m:
                        raise ValueError(f"Bad header in {fa}: {line[:80]!r}")
                    taxid, seqid = m.group(1), m.group(2)
                    # seqid column must be the full token after '>'
                    full_seqid = line[1:].split()[0].rstrip()
                    dest.write(f"TAXID\t{full_seqid}\t{taxid}\n")
                    n += 1
    print(f"  Wrote prelim_map_all.txt with {n} sequence entries → {out}")
    return n


def index_genomes(genomes_dir: Path) -> dict:
    index = {}
    for p in sorted(genomes_dir.rglob("*.fa.gz")) + sorted(
        genomes_dir.rglob("*.fna.gz")
    ):
        stem = p.name.replace(".fa.gz", "").replace(".fna.gz", "")
        if stem not in index or len(p.parts) < len(index[stem].parts):
            index[stem] = p
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--genomes", required=True)
    parser.add_argument("--db_dir", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--taxonomy", choices=("genus-aware", "species-only"),
                        default="genus-aware", help="species-only is the control, not the manuscript baseline")
    parser.add_argument(
        "--taxid_offset",
        type=int,
        default=DEFAULT_TAXID_OFFSET,
        help="taxid = species_class + offset (default 2)",
    )
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument(
        "--force_prepare",
        action="store_true",
        help="Rewrite library FASTAs even if present",
    )
    args = parser.parse_args()

    db_dir = Path(args.db_dir)
    genomes_dir = Path(args.genomes)
    lib_dir = db_dir / "library" / "added"
    db_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)
    offset = args.taxid_offset

    print(f"Loading labels: {args.labels}")
    df = pd.read_csv(args.labels, sep="\t")
    species_map = (
        df[["species_name", "species_class"]]
        .drop_duplicates()
        .set_index("species_name")["species_class"]
        .to_dict()
    )
    print(f"  {len(species_map)} unique species  (taxid_offset={offset})")

    print(f"Indexing genomes under {genomes_dir} ...")
    index = index_genomes(genomes_dir)
    print(f"  {len(index)} unique genome stems")

    missing = [sp for sp in species_map if sp not in index]
    if missing:
        print(f"ERROR: {len(missing)} species missing FASTA", file=sys.stderr)
        for sp in missing[:30]:
            print(f"  {sp}", file=sys.stderr)
        sys.exit(1)

    umgs = sum(1 for sp in species_map if sp.startswith("UMGS"))
    gcf = sum(
        1 for sp in species_map if sp.startswith("GCF") or sp.startswith("GCA")
    )
    hgr = len(species_map) - umgs - gcf
    print(f"  Coverage: UMGS={umgs}  GCF={gcf}  HGR={hgr}  ALL FOUND")

    print("\n[1/4] Building taxonomy...")
    species_to_genus = genus_names = None
    if args.taxonomy == "genus-aware":
        required = {"species_class", "genus_class", "genus_name"}
        if not required <= set(df.columns):
            parser.error("genus-aware taxonomy requires species_class, genus_class and genus_name")
        mapping = df[list(required)].drop_duplicates()
        if mapping.groupby("species_class")["genus_class"].nunique().max() != 1:
            parser.error("a species maps to conflicting genera")
        if mapping.groupby("genus_class")["genus_name"].nunique().max() != 1:
            parser.error("a genus class maps to conflicting names")
        species_to_genus = dict(zip(mapping.species_class, mapping.genus_class))
        genus_names = dict(zip(mapping.genus_class, mapping.genus_name))
    build_taxonomy(db_dir, set(species_map.values()), offset, species_to_genus, genus_names)

    print(f"\n[2/4] Preparing {len(species_map)} genomes ({args.workers} workers)...")
    tasks = []
    for sp, cls in species_map.items():
        taxid = int(cls) + offset
        out_path = lib_dir / f"{sp}_taxid{cls}.fna"
        tasks.append((str(index[sp]), taxid, str(out_path), args.force_prepare))

    ok = fail = skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_prepare_genome, t): t for t in tasks}
        for fut in as_completed(futs):
            success, msg, kind = fut.result()
            if success:
                ok += 1
                if kind == "skip":
                    skipped += 1
            else:
                fail += 1
                print(f"  FAIL: {msg}", file=sys.stderr)
            if (ok + fail) % 200 == 0:
                print(f"    prepared {ok + fail}/{len(tasks)} ...")
    print(f"  Library: {ok} ok ({skipped} reused)  |  {fail} failed")
    if fail:
        sys.exit(1)

    print("\n[3/4] Writing prelim_map.txt ...")
    n_seq = write_prelim_map(lib_dir)

    n_fa = sum(1 for _ in lib_dir.glob("*.fna"))
    meta = {
        "n_species": len(species_map),
        "n_library_fa": n_fa,
        "n_sequences": n_seq,
        "umgs": umgs,
        "gcf": gcf,
        "hgr": hgr,
        "taxid_offset": offset,
        "taxonomy": args.taxonomy,
        "n_genera": len(set(species_to_genus.values())) if species_to_genus else 0,
        "genomes_dir": str(genomes_dir),
        "labels": str(args.labels),
        "species_classes": sorted(int(x) for x in species_map.values()),
    }
    (db_dir / "species_classes_in_db.json").write_text(json.dumps(meta, indent=2))
    print(f"  Wrote metadata → {db_dir / 'species_classes_in_db.json'}")

    if args.skip_build:
        print("\n--skip_build set; stopping before kraken2-build")
        return

    print(f"\n[4/4] Building Kraken2 DB (threads={args.threads})...")
    subprocess.run(
        [
            "kraken2-build",
            "--build",
            "--db",
            str(db_dir),
            "--threads",
            str(args.threads),
        ],
        check=True,
    )
    print(f"\nDone! DB at: {db_dir}")
    print(f"      {n_fa} species / {n_seq} sequences (matched to NT/MT catalogue)")


if __name__ == "__main__":
    main()

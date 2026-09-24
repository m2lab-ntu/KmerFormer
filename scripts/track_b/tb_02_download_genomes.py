#!/usr/bin/env python3
"""
Track B – Step 2: download the shortlisted candidate genomes.

Reads candidate_survey.tsv (tb_01) and fetches each accession with
ncbi-datasets-cli, writing one FASTA per accession:

    genomes/<genus>/<accession>.fna

Downloads happen in accession batches (one zip per batch) to keep the number
of NCBI round-trips low.  Already-present accessions are skipped, so the
script is safe to re-run after an interruption.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--survey", required=True, help="candidate_survey.tsv")
    p.add_argument("--out_dir", required=True, help="genomes/ root")
    p.add_argument("--batch_size", type=int, default=25)
    p.add_argument("--retries", type=int, default=3)
    return p.parse_args()


def read_survey(path: str) -> list[dict]:
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        rows = []
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            rows.append(dict(zip(header, parts)))
    return rows


def download_batch(accessions: list[str], retries: int) -> Path | None:
    """Download a batch into a temp dir; return the extracted data dir."""
    tmp = Path(tempfile.mkdtemp(prefix="tb02_"))
    zip_path = tmp / "dl.zip"
    cmd = [
        "datasets", "download", "genome", "accession", *accessions,
        "--include", "genome",
        "--filename", str(zip_path),
        "--no-progressbar",
    ]
    for attempt in range(1, retries + 1):
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode == 0 and zip_path.exists():
            subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(tmp)],
                           check=True)
            return tmp
        print(f"    retry {attempt}/{retries}: {proc.stderr.strip()[:160]}",
              file=sys.stderr)
    shutil.rmtree(tmp, ignore_errors=True)
    return None


def main():
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = read_survey(args.survey)
    genus_of = {r["accession"]: r["genus"] for r in rows}

    todo = []
    for acc, genus in genus_of.items():
        dest = out_root / genus / f"{acc}.fna"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        todo.append(acc)

    print(f"{len(genus_of)} candidates total, {len(todo)} still to download.")
    if not todo:
        return

    ok, failed = 0, []
    batches = [todo[i:i + args.batch_size]
               for i in range(0, len(todo), args.batch_size)]
    for bi, batch in enumerate(batches, 1):
        print(f"[batch {bi}/{len(batches)}] {len(batch)} accessions", flush=True)
        tmp = download_batch(batch, args.retries)
        if tmp is None:
            failed.extend(batch)
            continue

        # datasets lays files out as ncbi_dataset/data/<accession>/*.fna
        found = defaultdict(list)
        for fna in tmp.rglob("*.fna"):
            found[fna.parent.name].append(fna)

        for acc in batch:
            files = found.get(acc)
            if not files:
                failed.append(acc)
                continue
            dest_dir = out_root / genus_of[acc]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{acc}.fna"
            if len(files) == 1:
                shutil.copyfile(files[0], dest)
            else:
                with open(dest, "wb") as out_fh:
                    for f in sorted(files):
                        with open(f, "rb") as in_fh:
                            shutil.copyfileobj(in_fh, out_fh)
            ok += 1
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\nDownloaded {ok}; failed {len(failed)}")
    if failed:
        fail_path = out_root.parent / "download_failed.txt"
        fail_path.write_text("\n".join(sorted(set(failed))) + "\n")
        print(f"  failed accessions -> {fail_path}")


if __name__ == "__main__":
    main()

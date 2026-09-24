#!/usr/bin/env python3
"""Reconstruct the species -> reference-genome mapping the label ids encode.

There is no "build the catalogue" script in this repo, and there does not need to
be one: the catalogue is *defined by the label file*. The `species_name` column is
the assembly's filename minus the suffix, so the mapping is read off directly. As
an independent check, `species_class` is also that assembly's index in the sorted
file list -- class ids were assigned in sorted-filename order. This script builds
the manifest by name and verifies the index agrees, so a mismatch is caught rather
than propagated.

What that means concretely for the gut catalogue:

  * the genome directory holds 2,505 candidate assemblies (UMGS and HGR
    metagenome-assembled bins plus RefSeq GCF assemblies, from the HGR/UMGS
    release of Almeida et al. 2019);
  * the labels select 1,535 of them, all resolving by name, and the selected
    `species_class` values are a sparse subset of 0..2504 rather than a
    renumbering -- which is why they run to 2503 while numbering only 1,535
    species. A species-level model reusing these ids needs 2,504 output classes;
    renumbering densely to 0..1534 is fine but breaks compatibility with existing
    checkpoints, so nothing here does it;
  * the 120 `genus_class` values are dense, 0..119, and are what every result in
    docs/RESULTS.md is measured on.

The one upstream step this cannot recover is *which* 1,535 of the 2,505 were
chosen and in what order the class ids were assigned. That selection came from
the original MetaTransformer data preparation and is not in this project. It does
not block reproduction, because the ids it produced are recorded in the label
files that ship with the data.

Usage::

    python scripts/data/catalogue_manifest.py \\
        --labels  $KF_DATA/val_100K/labels_100K_val.tsv \\
        --genomes /path/to/reference_genomes \\
        --out     catalogue_manifest.tsv

Any label file over the same catalogue gives the same manifest, so use the
smallest one you have -- the 100K closed-set labels cover all 1,535 species and
are 7 MB, against 3.6 GB for the 50M pool.
"""

import argparse
import csv
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True,
                    help="a labels TSV over the catalogue (any pool will do)")
    ap.add_argument("--genomes", type=Path, required=True,
                    help="directory of .fa.gz reference assemblies")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the manifest here (default: stdout)")
    ap.add_argument("--suffix", default=".fa.gz")
    args = ap.parse_args()

    files = sorted(f.name for f in args.genomes.iterdir()
                   if f.name.endswith(args.suffix))
    if not files:
        raise SystemExit(f"no *{args.suffix} in {args.genomes}")
    index = {f[: -len(args.suffix)]: i for i, f in enumerate(files)}
    print(f"candidate assemblies in {args.genomes}: {len(files)}", file=sys.stderr)

    # species_class -> (species_name, genus_class, genus_name)
    species = {}
    with args.labels.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            species.setdefault(int(row["species_class"]),
                               (row["species_name"], int(row["genus_class"]),
                                row["genus_name"]))

    genera = {g for _, (_, g, _) in species.items()}
    print(f"species in labels: {len(species)}   genera: {len(genera)}", file=sys.stderr)
    print(f"species_class range: {min(species)}..{max(species)}   "
          f"genus_class range: {min(genera)}..{max(genera)}", file=sys.stderr)

    rows, agree, missing = [], 0, []
    for sc in sorted(species):
        name, gc, gname = species[sc]
        i = index.get(name)
        if i is None:
            missing.append(name)
        elif i == sc:
            agree += 1
        rows.append((sc, name, gc, gname,
                     f"{name}{args.suffix}" if i is not None else "MISSING"))

    print(f"species_class == index into the sorted genome list: "
          f"{agree}/{len(species)}", file=sys.stderr)
    if missing:
        print(f"WARNING: {len(missing)} species have no assembly in {args.genomes}: "
              f"{missing[:5]}{' ...' if len(missing) > 5 else ''}", file=sys.stderr)
    if agree != len(species) - len(missing):
        print("WARNING: some ids do not match the sorted-index convention. The "
              "genome directory may hold a different candidate set than the one "
              "the labels were numbered against; the manifest below is still "
              "correct by NAME, only the index check failed.", file=sys.stderr)

    out = args.out.open("w", newline="") if args.out else sys.stdout
    w = csv.writer(out, delimiter="\t")
    w.writerow(["species_class", "species_name", "genus_class", "genus_name",
                "genome_file"])
    w.writerows(rows)
    if args.out:
        out.close()
        print(f"wrote {args.out} ({len(rows)} species)", file=sys.stderr)


if __name__ == "__main__":
    main()

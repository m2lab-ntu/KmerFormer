#!/usr/bin/env python3
"""Generate the README result table from protocol-specific result records."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    records = json.loads((ROOT / "docs/results_manifest.json").read_text())
    lines = ["| Model | Closed-set genus Top-1 | Evaluation |", "|---|---:|---|"]
    for record in records["readme_rows"]:
        lines.append(f"| {record['model']} | {record['reported_percent']:.2f}% | {record['evaluation']} |")
    start, end = "<!-- RESULTS_TABLE_START -->", "<!-- RESULTS_TABLE_END -->"
    path = ROOT / "README.md"
    before = path.read_text()
    prefix, rest = before.split(start, 1)
    _, suffix = rest.split(end, 1)
    after = prefix + start + "\n" + "\n".join(lines) + "\n" + end + suffix
    if args.check:
        if before != after:
            print("README result table differs from docs/results_manifest.json", file=sys.stderr)
            return 1
        print("Result table matches protocol-specific records")
    else:
        path.write_text(after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

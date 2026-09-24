#!/usr/bin/env python3
"""Run CPU-only reviewer checks without training, downloads or release writes."""
import argparse
import ast
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-metrics", action="store_true")
    parser.add_argument("--manuscript", type=Path, help="optional separate manuscript checkout")
    parser.add_argument("--json", type=Path, help="write environment and check outcomes")
    args = parser.parse_args()
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", PYTHONHASHSEED="0")
    report = {"python": platform.python_version(), "dependencies": {}, "checks": []}
    for package in ("torch", "transformers", "peft", "numpy", "scipy", "pytest"):
        try:
            report["dependencies"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["dependencies"][package] = "not installed"
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2), flush=True)

    def run(name, command):
        print(f"\n=== {name} ===", flush=True)
        try:
            result = subprocess.run(command, cwd=REPO, env=env, timeout=900)
            ok = result.returncode == 0
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"FAIL: {exc}")
            ok = False
        report["checks"].append({"name": name, "passed": ok})

    sources = [p for folder in ("kmerformer", "scripts", "tests")
               for p in (REPO / folder).rglob("*.py")]
    try:
        for path in sources:
            ast.parse(path.read_text(), filename=str(path))
        report["checks"].append({"name": "Python syntax", "passed": True, "files": len(sources)})
    except (SyntaxError, UnicodeError) as exc:
        print(f"FAIL: {exc}")
        report["checks"].append({"name": "Python syntax", "passed": False})
    shells = sorted((REPO / "scripts").rglob("*.sh"))
    if shutil.which("bash"):
        for path in shells:
            run(f"shell syntax: {path.relative_to(REPO)}", ["bash", "-n", str(path)])
    else:
        report["checks"].append({"name": "shell syntax (bash missing)", "passed": False})
    run("CPU regression tests", [sys.executable, "-m", "pytest", "tests", "-q"])
    run("Documentation links", [sys.executable, "scripts/check_doc_links.py"])
    run("Training CLI help", [sys.executable, "-m", "kmerformer.train", "--help"])
    run("Evaluation CLI help", [sys.executable, "-m", "kmerformer.evaluate", "--help"])
    run("Synthetic checkpoint round trip", [sys.executable, "scripts/smoke_cpu.py"])
    command = [sys.executable, "scripts/review_predictions.py"]
    if args.full_metrics:
        command.append("--full-metrics")
    run("Bundled prediction evidence", command)
    if args.manuscript:
        run("Manuscript numeric-presence check (not semantic validation)",
            [sys.executable, "scripts/check_against_manuscript.py", str(args.manuscript.resolve())])
    report["passed"] = all(c["passed"] for c in report["checks"])
    report["scope"] = "software and bundled evidence only; not full experimental reproduction"
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print("\n" + ("PASS" if report["passed"] else "FAIL") + ": " + report["scope"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

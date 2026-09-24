#!/usr/bin/env python3
"""Package current reviewable sources and small evidence, without Git history.

Draft mode uses tracked and non-ignored source files, including uncommitted edits.
Release mode requires a clean checkout and includes every tracked file directly
from HEAD, failing on unexpected paths instead of silently omitting them. Excludes
weights, local environments, operational handoff notes and generated run outputs.
Does not commit, publish, upload, or choose any additional licence.
"""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT_FILES = {"README.md", "LICENSE", "CITATION.cff", "pyproject.toml",
              "requirements.txt", "environment.yml", "environment-pools.yml", ".gitignore", ".gitattributes", "CONTRIBUTING.md", "RELEASE.md"}
SOURCE_ROOTS = {"kmerformer", "configs", "scripts", "tests", "docs", "weights", "examples", "reproduction", ".github"}
SUFFIXES = {".py", ".sh", ".md", ".yaml", ".yml", ".json", ".npz", ".tsv", ".csv", ".gz", ".txt", ".fa"}


def include(relative):
    parts = relative.parts
    if len(parts) == 1:
        return relative.name in ROOT_FILES
    if parts[:2] == (".github", "workflows"):
        return len(parts) == 3 and relative.suffix in {".yml", ".yaml"}
    if parts[0] not in SOURCE_ROOTS or any(p.startswith(".") for p in parts):
        return False
    if any(p in {"__pycache__", "agent_prompts"} for p in parts):
        return False
    if parts[0] == "weights":
        return relative.name in {"README.md", "ZENODO.md"} and len(parts) == 2
    if relative.suffix in {".npz", ".tsv", ".json"}:
        return (parts[:2] in {("docs", "assets"), ("kmerformer", "assets"), ("tests", "fixtures"), ("docs", "validation")}
                or parts[0] in {"examples", "reproduction"}
                or relative.as_posix() == "docs/results_manifest.json")
    return relative.suffix in SUFFIXES


def committed_members(root):
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root)
    if status.strip():
        raise ValueError("release requires a clean checkout; commit or resolve pending changes first")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{commit}"], cwd=root, text=True).strip()
    tree = subprocess.check_output(["git", "ls-tree", "-rz", "--full-tree", commit], cwd=root)
    members, modes = {}, {}
    for record in tree.split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        name = path.decode()
        if not include(Path(name)) and name != "weights/.gitkeep":
            raise ValueError(f"tracked file outside release policy: {name}; resolve before publishing")
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(f"unsupported tracked entry: {name} ({mode} {kind})")
        size = int(subprocess.check_output(["git", "cat-file", "-s", oid], cwd=root))
        limit = 64 * 1024 * 1024 if name.startswith("reproduction/") else 10 * 1024 * 1024
        if size > limit:
            raise ValueError(f"unexpectedly large tracked release file: {name}")
        members[name] = subprocess.check_output(["git", "cat-file", "blob", oid], cwd=root)
        modes[name] = mode
    missing = ROOT_FILES - members.keys()
    if missing:
        raise ValueError(f"missing required release files: {sorted(missing)}")
    return commit, members, modes


def release_manifest(commit, members, modes):
    return {
        "format": 2, "mode": "committed-release", "base_commit": commit,
        "snapshot": "all tracked files from the named commit; no uncommitted edits",
        "scope": "code and bundled predictions; no data, checkpoints or Git history",
        "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                         "git_mode": modes[name]} for name, data in sorted(members.items())},
    }


def verify_release(root, archive_path):
    commit, members, modes = committed_members(root)
    expected = release_manifest(commit, members, modes)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        required = {"KmerFormer/" + name for name in members} | {"KmerFormer/RELEASE_MANIFEST.json"}
        if len(names) != len(required) or set(names) != required:
            raise ValueError("archive file list differs from the committed repository")
        if json.loads(archive.read("KmerFormer/RELEASE_MANIFEST.json")) != expected:
            raise ValueError("archive manifest differs from the committed repository")
        for name, data in members.items():
            path = "KmerFormer/" + name
            if archive.read(path) != data:
                raise ValueError(f"archive contents differ from commit: {name}")
            if archive.getinfo(path).external_attr >> 16 != int(modes[name], 8):
                raise ValueError(f"archive mode differs from commit: {name}")
    return expected


def package(root, destination, release=False):
    if release:
        commit, members, modes = committed_members(root)
        manifest = release_manifest(commit, members, modes)
        write_archive(destination, members, manifest, modes)
        return manifest
    found = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)
    relative = sorted({Path(p.decode()) for p in found.split(b"\0") if p and include(Path(p.decode()))})
    missing = ROOT_FILES - {p.as_posix() for p in relative}
    if missing:
        raise ValueError(f"missing required release files: {sorted(missing)}")
    members = {}
    for rel in relative:
        path = root / rel
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"refusing symlink or path outside repository: {rel}")
        limit = 64 * 1024 * 1024 if rel.parts[0] == "reproduction" else 10 * 1024 * 1024
        if not path.is_file() or path.stat().st_size > limit:
            raise ValueError(f"missing or unexpectedly large source/evidence file: {rel}")
        members[rel.as_posix()] = path.read_bytes()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    manifest = {
        "format": 1, "base_commit": commit,
        "snapshot": "current working-tree files; includes uncommitted edits",
        "scope": "code and bundled predictions; no data, checkpoints or Git history",
        "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                  for name, data in members.items()},
    }
    write_archive(destination, members, manifest)
    return manifest


def write_archive(destination, members, manifest, modes=None):
    members = dict(members)
    members["RELEASE_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    # Exclusive creation prevents replacing an earlier reviewer delivery.
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(f"KmerFormer/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            mode = int(modes[name], 8) if modes and name in modes else (0o100755 if name.endswith(".sh") else 0o100644)
            info.external_attr = mode << 16
            archive.writestr(info, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path,
                        help="new zip path; existing files are never overwritten")
    target.add_argument("--verify-archive", type=Path, help="verify exact file/byte/mode parity with clean HEAD")
    parser.add_argument("--release", action="store_true", help="package clean committed HEAD for Zenodo/public delivery")
    args = parser.parse_args()
    try:
        manifest = (verify_release(REPO, args.verify_archive) if args.verify_archive
                    else package(REPO, args.output, release=args.release))
    except (ValueError, OSError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Cannot package reviewer snapshot: {exc}\n")
    archive_path = args.verify_archive or args.output
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if manifest.get("mode") == "committed-release":
        print(f"Committed release: {manifest['base_commit']}")
    print(f"{len(manifest['files'])} files; no Git history or model weights")
    print(f"SHA256 {digest}  {archive_path}")


if __name__ == "__main__":
    main()

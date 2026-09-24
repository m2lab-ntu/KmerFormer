#!/usr/bin/env python3
"""Prepare a version-consistent local Zenodo delivery; never upload or publish."""
import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from urllib.parse import quote, unquote, urlsplit
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from package_reviewer import verify_release


def file_info(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def portable_markdown(source, root, commit):
    """Keep cross-document links valid when copying a report out of the tree."""
    def replace(match):
        link = match.group(1)
        parts = urlsplit(link)
        if parts.scheme or parts.netloc or not parts.path:
            return match.group(0)
        target = (source.parent / unquote(parts.path)).resolve()
        relative = target.relative_to(root.resolve())
        if not target.exists():
            raise ValueError(f"Broken report link: {link}")
        url = f"https://github.com/m2lab-ntu/KmerFormer/blob/{commit}/{quote(relative.as_posix())}"
        if parts.fragment:
            url += "#" + parts.fragment
        return "](" + url + ")"
    return re.sub(r"\]\(([^)\s]+)\)", replace, source.read_text())


def verify_wheel(root, wheel):
    """Reject stale built Python modules or registry/tokenizer assets."""
    tracked = subprocess.check_output(["git", "ls-files", "-z", "kmerformer"], cwd=root)
    files = [p.decode() for p in tracked.split(b"\0") if p]
    with zipfile.ZipFile(wheel) as archive:
        packaged = {n for n in archive.namelist() if n.startswith("kmerformer/") and not n.endswith("/")}
        if packaged != set(files):
            raise ValueError("Wheel package file list differs from the source commit")
        for name in files:
            if archive.read(name) != (root / name).read_bytes():
                raise ValueError(f"Wheel contains stale source or assets: {name}")
        metadata = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        parsed = BytesParser().parsebytes(archive.read(metadata))
        version = parsed["Version"]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    if version != project["version"]:
        raise ValueError("Wheel version differs from pyproject.toml")
    if isinstance(project.get("readme"), str):
        if parsed.get_payload().strip() != (root / project["readme"]).read_text().strip():
            raise ValueError("Wheel contains a stale README description")
    return version


def combine_parts(parts_root, entry, target):
    total = hashlib.sha256()
    with target.open("xb") as dest:
        for part in entry["download"]["parts"]:
            name = part["filename"]
            if Path(name).name != name:
                raise ValueError("Download part name must be a basename")
            source = parts_root / name
            if source.stat().st_size != part["size_bytes"]:
                raise ValueError(f"Download part size mismatch: {name}")
            digest = hashlib.sha256()
            with source.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
                    digest.update(chunk)
                    total.update(chunk)
                    dest.write(chunk)
            if digest.hexdigest() != part["sha256"]:
                raise ValueError(f"Download part checksum mismatch: {name}")
    if total.hexdigest() != entry["download"]["sha256"]:
        raise ValueError("Reassembled model archive checksum mismatch")
    with tarfile.open(target) as archive:
        first = archive.next()
        if not first or first.name != "manifest.json" or not first.isfile():
            raise ValueError("Model archive must begin with its manifest")
        manifest = json.load(archive.extractfile(first))
    for field in ("model_id", "weights_licence", "source_checkpoint_sha256"):
        if manifest[field] != entry[field]:
            raise ValueError(f"Archive and registry disagree on {field}")
    return {"size_bytes": target.stat().st_size, "sha256": total.hexdigest()}, manifest


def separate_records(directory, record):
    """Partition verified files into independently citable code and weight records."""
    groups = {"code": [name for name in record["files"] if name.endswith((".zip", ".whl"))],
              "weights": [model["archive"] for model in record["models"].values()]}
    citation = yaml.safe_load((directory / "CITATION.cff").read_text())
    summaries = {}
    for kind, names in groups.items():
        target = directory / kind
        target.mkdir()
        files = {}
        for name in names:
            (directory / name).rename(target / name)
            files[name] = record["files"][name]
        shutil.copyfile(directory / "VALIDATION.md", target / "VALIDATION.md")
        files["VALIDATION.md"] = file_info(target / "VALIDATION.md")
        if kind == "code":
            shutil.copyfile(directory / "CITATION.cff", target / "CITATION.cff")
            title = f"KmerFormer {record['version']}: software and recorded evidence"
            description = (
                "The source ZIP includes the implementation, configurations, tests,\n"
                "109 recovered evidence files and the 16-read example. The wheel\n"
                "installs the library and CLIs. Trained models are in the companion\n"
                "weights record. Code is MIT; bundled tokenizer assets retain\n"
                "their upstream terms, including the NT-v2 NonCommercial condition.\n")
            models = {}
        else:
            models = record["models"]
            title = f"KmerFormer {record['version']}: trained model bundles"
            terms = sorted({model["weights_licence"] for model in models.values()})
            model_citation = dict(citation, title=title, type="dataset", license=terms,
                                  message="If you use these trained models, please cite the manuscript below.",
                                  abstract="Trained KmerFormer model bundles with tokenizer assets, genus labels and inference settings.")
            model_citation.pop("license-url", None)
            (target / "CITATION.cff").write_text(yaml.safe_dump(model_citation, sort_keys=False))
            description = (
                "Each TAR includes weights, configuration, tokenizer assets, genus\n"
                "labels and an inference protocol. Install the wheel from the\n"
                "companion code record and extract each TAR into its own directory:\n\n"
                "  kmerformer-predict --model model-directory --reads sample.fastq.gz --output predictions.tsv\n\n"
                "Weight licences: " + ", ".join(terms) + ". WEIGHTS_LICENSE.txt\n"
                "applies to model.safetensors. TOKENIZER_LICENSE.txt retains each\n"
                "tokenizer's terms. The 6-mer tokenizer remains CC BY-NC-SA 4.0;\n"
                "the exact vocabulary is CC BY 4.0; the hashed implementation is MIT.\n")
        files["CITATION.cff"] = file_info(target / "CITATION.cff")
        (target / "README.txt").write_text(
            title + "\n\nSource commit: " + record["source_commit"] + "\n"
            "Repository: https://github.com/m2lab-ntu/KmerFormer\n\n" + description + "\n"
            "Verify files with: sha256sum -c SHA256SUMS\n"
            "Link the companion record through Zenodo Related identifiers.\n"
            "This is a local delivery candidate; no Zenodo DOI has been assigned.\n")
        files["README.txt"] = file_info(target / "README.txt")
        metadata = {key: record[key] for key in
                    ("schema_version", "version", "source_commit", "creators", "state", "publication_access")}
        metadata.update(title=title, record_kind=kind, zenodo_record_id=None, doi=None,
                        companion_record={"kind": "weights" if kind == "code" else "code", "doi": None},
                        pending_weight_licences=record["pending_weight_licences"] if models else [],
                        models=models, files=files,
                        payload_bytes=sum(info["size_bytes"] for info in files.values()))
        (target / "DEPOSIT_MANIFEST.json").write_text(json.dumps(metadata, indent=2) + "\n")
        sums = dict(files, **{"DEPOSIT_MANIFEST.json": file_info(target / "DEPOSIT_MANIFEST.json")})
        (target / "SHA256SUMS").write_text("".join(f"{info['sha256']}  {name}\n" for name, info in sorted(sums.items())))
        summaries[kind] = {"title": title, "directory": kind, "file_count": len(sums) + 1,
                           "doi": None, "total_bytes": sum(p.stat().st_size for p in target.iterdir())}
    for name in ("CITATION.cff", "VALIDATION.md", "README.txt", "DEPOSIT_MANIFEST.json", "SHA256SUMS"):
        (directory / name).unlink()
    index = {key: record[key] for key in ("version", "source_commit", "state", "pending_weight_licences")}
    index["records"] = summaries
    (directory / "DELIVERY_MANIFEST.json").write_text(json.dumps(index, indent=2) + "\n")
    (directory / "README.txt").write_text(
        "Upload code/ and weights/ as two separate Zenodo records.\n"
        "Each directory has its own citation metadata, inventory and checksums.\n"
        "Link the two records using Related identifiers and retain the source commit.\n"
        "No upload, publication or DOI reservation has been performed.\n")
    return summaries


def prepare(root, code, wheel, parts, output, *, split_records=False):
    if output.exists():
        raise FileExistsError(output)
    source = verify_release(root, code)
    version = verify_wheel(root, wheel)
    registry = json.loads((root / "kmerformer/assets/models.json").read_text())
    citation = yaml.safe_load((root / "CITATION.cff").read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
    files, models = {}, {}
    try:
        for path in (code, wheel):
            shutil.copyfile(path, temporary / path.name)
            files[path.name] = file_info(temporary / path.name)
        for model_id, entry in registry["models"].items():
            if Path(model_id).name != model_id:
                raise ValueError("Model ID must be a basename")
            if not entry.get("download"):
                raise ValueError(f"Model has no verified download record: {model_id}")
            name = model_id + ".tar"
            files[name], manifest = combine_parts(parts, entry, temporary / name)
            models[model_id] = {"archive": name, "weights_licence": manifest["weights_licence"],
                                "tokenizer_licence": manifest.get("tokenizer_licence"),
                                "source_checkpoint_sha256": manifest["source_checkpoint_sha256"]}
            print(f"Prepared {name}: {files[name]['size_bytes']:,} bytes", flush=True)
        for source_path, name in ((root / "docs/validation" / f"{version}.md", "VALIDATION.md"),
                                  (root / "CITATION.cff", "CITATION.cff")):
            if source_path.suffix == ".md":
                (temporary / name).write_text(portable_markdown(source_path, root, source["base_commit"]))
            else:
                shutil.copyfile(source_path, temporary / name)
            files[name] = file_info(temporary / name)
        pending = [key for key, value in models.items() if value["weights_licence"] == "LicenseRef-Pending"]
        (temporary / "README.txt").write_text(
            f"KmerFormer {version}: software, model bundles and recorded evidence\n\n"
            f"Source commit: {source['base_commit']}\n"
            "Repository: https://github.com/m2lab-ntu/KmerFormer\n\n"
            "The code ZIP contains the full source snapshot, 109 recovered evidence files,\n"
            "primary prediction arrays, configurations, tests and the 16-read example.\n"
            "The wheel contains the installable library and fixed tokenizer assets.\n"
            "Each model TAR contains safetensors, configuration, class map, tokenizer\n"
            "and a manifest. Extract each TAR into its own directory, then use:\n\n"
            "  kmerformer-predict --model model-directory --reads sample.fastq.gz --output predictions.tsv\n\n"
            "Verify download bytes with: sha256sum -c SHA256SUMS\n"
            "Full training pools and reference genomes are external; the evidence\n"
            "collection supports rescoring, not every experiment from raw inputs.\n\n"
            "Code licence: MIT. Fixed NT-v2 tokenizer: CC BY-NC-SA 4.0.\n"
            "Exact 13-mer vocabulary: CC BY 4.0. Notices accompany the assets.\n"
            + ("Weight licences are PENDING author selection; this staging directory\n"
               "does not grant a public weight licence.\n" if pending else
               "Weight licences: " + ", ".join(sorted({m["weights_licence"] for m in models.values()})) + ".\n"
               "WEIGHTS_LICENSE.txt applies to model.safetensors in each bundle.\n"
               "TOKENIZER_LICENSE.txt retains the separate tokenizer terms.\n"
               "The 6-mer tokenizer retains its NonCommercial restriction.\n")
            + "This is a local delivery candidate, not evidence of a Zenodo upload or DOI.\n")
        files["README.txt"] = file_info(temporary / "README.txt")
        record = {"schema_version": 1, "version": version, "source_commit": source["base_commit"],
                  "title": f"KmerFormer {version}: software, trained model bundles and recorded evidence",
                  "creators": [{"name": f"{a['family-names']}, {a['given-names']}",
                                "orcid": a.get("orcid"), "affiliation": a.get("affiliation")}
                               for a in citation["authors"]],
                  "state": "prepared_locally", "zenodo_record_id": None, "doi": None,
                  "publication_access": "not_selected", "pending_weight_licences": pending,
                  "models": models, "files": files, "payload_bytes": sum(x["size_bytes"] for x in files.values())}
        (temporary / "DEPOSIT_MANIFEST.json").write_text(json.dumps(record, indent=2) + "\n")
        sums = dict(files)
        sums["DEPOSIT_MANIFEST.json"] = file_info(temporary / "DEPOSIT_MANIFEST.json")
        (temporary / "SHA256SUMS").write_text("".join(f"{info['sha256']}  {name}\n" for name, info in sorted(sums.items())))
        if len(sums) + 1 > 100 or sum(p.stat().st_size for p in temporary.iterdir()) > 50_000_000_000:
            raise ValueError("Delivery exceeds the verified default Zenodo file quota")
        if split_records:
            record["records"] = separate_records(temporary, record)
        temporary.rename(output)
        return record
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code", "wheel", "parts", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--separate-records", action="store_true",
                        help="Prepare code/ and weights/ for two linked Zenodo records")
    args = parser.parse_args()
    record = prepare(ROOT, args.code, args.wheel, args.parts, args.output, split_records=args.separate_records)
    print(json.dumps({"output": str(args.output), "source_commit": record["source_commit"],
                      "payload_bytes": record["payload_bytes"],
                      "pending_weight_licences": record["pending_weight_licences"],
                      "records": record.get("records")}, indent=2))


if __name__ == "__main__":
    main()

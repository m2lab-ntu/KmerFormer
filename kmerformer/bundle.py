"""Versioned inference bundles: tensors, tokenizer, class names and protocol."""


class ModelUnavailable(RuntimeError):
    """A registered model could not be fetched with the caller's credentials."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.error
import urllib.request
import struct
import subprocess
from urllib.parse import urlparse

import torch
from safetensors.torch import load_file

from .kmer_tokenizers import build_tokenizer
from .model import create_model
from .splits import sha256_file


BUNDLE_LICENCE_FILES = {"WEIGHTS_LICENSE.txt", "TOKENIZER_LICENSE.txt"}
LICENCE_ASSETS = Path(__file__).parent / "assets" / "licences"
SUPPORTED_LICENCES = {"Apache-2.0", "MIT", "CC-BY-4.0", "CC-BY-NC-SA-4.0"}


def copy_licence_texts(directory, weights_licence, tokenizer_licence):
    """Include the selected terms without assigning rights to unknown assets."""
    copied = []
    for licence, name in ((weights_licence, "WEIGHTS_LICENSE.txt"),
                          (tokenizer_licence, "TOKENIZER_LICENSE.txt")):
        if licence in SUPPORTED_LICENCES:
            shutil.copyfile(LICENCE_ASSETS / f"{licence}.txt", Path(directory) / name)
            copied.append(name)
    return copied


def save_tensor_stream(tensors, path):
    """Write safetensors without copying a multi-GiB embedding into Python bytes."""
    names = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
             torch.float64: "F64", torch.int64: "I64", torch.int32: "I32",
             torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL"}
    header, offset = {}, 0
    for key, tensor in tensors.items():
        if tensor.device.type != "cpu" or not tensor.is_contiguous() or tensor.dtype not in names:
            raise ValueError(f"Export requires a contiguous CPU tensor with a supported dtype: {key}")
        size = tensor.numel() * tensor.element_size()
        header[key] = {"dtype": names[tensor.dtype], "shape": list(tensor.shape), "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as dest:
        dest.write(struct.pack("<Q", len(encoded)))
        dest.write(encoded)
        for tensor in tensors.values():
            flat = tensor.detach().reshape(-1).view(torch.uint8).numpy()
            for start in range(0, len(flat), 8 * 1024 * 1024):
                dest.write(memoryview(flat[start:start + 8 * 1024 * 1024]))


def _json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _inside(root, name):
    path = (root / name).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Bundle path escapes its directory: {name}")
    return path


def verify_bundle(path):
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported bundle schema version")
    required = {"model.safetensors", "config.json", "labels.json"}
    if not required.issubset(manifest["files"]):
        raise ValueError("Bundle manifest is missing required files")
    for name, entry in manifest["files"].items():
        source = _inside(root, name)
        if not source.is_file() or source.stat().st_size != entry["size_bytes"] or sha256_file(source) != entry["sha256"]:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    return manifest


class ModelBundle:
    """Load a verified local bundle. Loading performs no network access."""
    def __init__(self, path, *, device="cpu"):
        self.path = Path(path).resolve()
        self.manifest = verify_bundle(self.path)
        self.config = json.loads((self.path / "config.json").read_text())
        self.labels = json.loads((self.path / "labels.json").read_text())
        self.num_classes = int(self.manifest["num_classes"])
        if [entry["class_id"] for entry in self.labels] != list(range(self.num_classes)):
            raise ValueError("Bundle labels must define every output class in ID order")
        cfg = copy.deepcopy(self.config)
        tcfg = cfg["data"]["tokenizer"]
        for field in ("vocab_path",):
            if field in tcfg:
                if tcfg[field] not in self.manifest["files"]:
                    raise ValueError("Tokenizer asset is not covered by the bundle manifest")
                tcfg[field] = str(_inside(self.path, tcfg[field]))
        self.tokenizer = build_tokenizer(cfg)
        if (cfg["model"]["vocab_size"] != self.tokenizer.vocab_size or
                cfg["model"]["pad_id"] != self.tokenizer.pad_token_id):
            raise ValueError("Model and tokenizer vocabulary/PAD settings disagree")
        self.device = torch.device(device)
        with torch.device("meta"):
            self.model = create_model(cfg["model"], self.num_classes)
        self.model.load_state_dict(load_file(str(self.path / "model.safetensors")), strict=True, assign=True)
        self.model.to(self.device).eval()
        self.max_length = int(cfg["data"]["max_token_length"])
        self.protocol = self.manifest["inference"]

    @classmethod
    def from_pretrained(cls, path, *, device="cpu"):
        return cls(path, device=device)

    def predict(self, reads, *, batch_size=128, rc_tta=None, precision="fp32"):
        from .predict import predict_records
        return predict_records(self, reads, batch_size=batch_size, rc_tta=rc_tta, precision=precision)


def export_bundle(checkpoint, config, labels_path, output, *, model_id, licence="LicenseRef-Pending"):
    """Convert a trusted KmerFormer state dict into an inference-only directory."""
    import pandas as pd
    from .utils import load_config
    cfg = load_config(config) if not isinstance(config, dict) else copy.deepcopy(config)
    if cfg["model"].get("type", "kmerformer") not in ("shallow_transformer", "kmerformer"):
        raise ValueError("Bundle export supports from-scratch KmerFormer models")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    task = cfg["data"].get("task", "genus")
    id_col, name_col = f"{task}_class", f"{task}_name"
    mapping = {}
    for frame in pd.read_csv(labels_path, sep="\t", usecols=[id_col, name_col], chunksize=100_000,
                             dtype={id_col: "int64", name_col: str}, keep_default_na=False):
        for key, name in frame[[id_col, name_col]].drop_duplicates().itertuples(index=False, name=None):
            key, name = int(key), str(name)
            if key < 0 or not name.strip() or (key in mapping and mapping[key] != name):
                raise ValueError("Label IDs must be nonnegative and each ID must have one name")
            mapping[key] = name
    if not mapping:
        raise ValueError("Label catalogue is empty")
    num_classes = max(mapping) + 1
    labels = [{"class_id": i, "name": mapping.get(i), "rank": task,
               "observed_in_training_catalogue": i in mapping} for i in range(num_classes)]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    tensors = state.get("model_state_dict", state)
    key = "head.classifier.weight" if "head.classifier.weight" in tensors else "head.classifier.4.weight"
    if tensors[key].shape[0] != num_classes:
        raise ValueError("Checkpoint output width does not match the supplied training label catalogue")
    tokenizer = build_tokenizer(cfg)
    portable = {"model": copy.deepcopy(cfg["model"]),
                "data": {"task": task, "max_token_length": cfg["data"].get("max_token_length", 32)}}
    portable["model"].update(type="kmerformer", vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_token_id)
    for key in ("revision", "trust_remote_code", "gradient_checkpointing"):
        portable["model"].pop(key, None)
    tcfg = copy.deepcopy(cfg["data"].get("tokenizer") or {"type": "ntv2_6mer"})
    tcfg.pop("cache_path", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
    try:
        tokenizer_licence = "LicenseRef-Pending"
        if tcfg["type"] == "ntv2_6mer":
            shutil.copyfile(tokenizer.vocab_path, temporary / "tokenizer.json")
            original = Path(__file__).parent / "assets" / "ntv2_6mer.json"
            if sha256_file(temporary / "tokenizer.json") == sha256_file(original):
                shutil.copyfile(Path(__file__).parent / "assets" / "NOTICE.md", temporary / "TOKENIZER_NOTICE.md")
                tokenizer_licence = "CC-BY-NC-SA-4.0"
            tcfg["vocab_path"] = "tokenizer.json"
        elif tcfg["type"] == "exact_kmer":
            shutil.copyfile(tcfg["vocab_path"], temporary / "vocabulary.txt")
            provenance = Path(__file__).parent / "assets" / "exact13_provenance.json"
            if sha256_file(temporary / "vocabulary.txt") == json.loads(provenance.read_text())["local_vocabulary_sha256"]:
                shutil.copyfile(Path(__file__).parent / "assets" / "EXACT13_NOTICE.md", temporary / "TOKENIZER_NOTICE.md")
                shutil.copyfile(provenance, temporary / "exact13_provenance.json")
                tokenizer_licence = "CC-BY-4.0"
            tcfg["vocab_path"] = "vocabulary.txt"
            tcfg["canonical"] = bool(tokenizer.canonical)
        elif tcfg["type"] == "hashed_kmer":
            tokenizer_licence = "MIT"
        copy_licence_texts(temporary, licence, tokenizer_licence)
        portable["data"]["tokenizer"] = tcfg
        if cfg["data"].get("kmer_preprocess"):
            portable["data"]["kmer_preprocess"] = cfg["data"]["kmer_preprocess"]
        save_tensor_stream(tensors, temporary / "model.safetensors")
        _json(temporary / "config.json", portable)
        _json(temporary / "labels.json", labels)
        manifest = {"schema_version": 1, "model_id": model_id, "num_classes": num_classes,
                    "task": task, "weights_licence": licence,
                    "weights_licence_scope": "model.safetensors", "tokenizer_licence": tokenizer_licence,
                    "source_checkpoint_sha256": sha256_file(checkpoint),
                    "inference": {"rc_tta": True, "aggregation": "mean_logits", "default_precision": "fp32",
                                  "fp32_encoder_fastpath": False},
                    "files": {p.name: {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
                              for p in sorted(temporary.iterdir())}}
        _json(temporary / "manifest.json", manifest)
        verify_bundle(temporary)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def _github_token():
    """The credential download_model will use, or None.

    Checked by `list` as well as `download`, so the availability report and the
    download agree about what the caller has rather than differing by accident.
    """
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token and shutil.which("gh"):
        result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
        token = result.stdout.strip() if result.returncode == 0 else None
    return token or None


def describe_availability(entry, *, token=None):
    """What the caller can actually fetch, without touching the network.

    weights/README.md promises that `list` reports each model's access state.
    Printing the registry does not do that: every entry carries download parts
    whether or not the caller can read them, so the same registry could mean
    "ready" for a maintainer and 404 for everyone else. This returns the
    distinction; `--check` probes.

    Only GitHub release assets are classified from the URL. Any other host is
    reported "unverified" rather than "open": a Hugging Face dataset URL looks
    the same whether the dataset is public or private, and calling it open
    would repeat the misreport this function exists to prevent.
    """
    download = entry.get("download")
    if not download or not download.get("parts"):
        return {"state": "unregistered",
                "detail": "the trained weights are not publicly released; "
                          "see weights/README.md"}
    hosts = {urlparse(part["url"]).hostname for part in download["parts"]}
    if hosts == {"api.github.com"}:
        if token:
            return {"state": "credentialed",
                    "detail": "GitHub release asset; a credential was found"}
        return {"state": "needs-credentials",
                "detail": "GitHub release asset that may require access. Set "
                          "GH_TOKEN, or authenticate the gh CLI, with an account "
                          "that can read it; `list --check` confirms"}
    return {"state": "unverified",
            "detail": "; ".join(sorted(hosts)) + " -- run `list --check` to confirm"}


def check_reachable(entry, *, token=None, timeout=20):
    """Ask the host whether the first part is actually fetchable."""
    part = entry["download"]["parts"][0]
    request = urllib.request.Request(part["url"],
                                     headers={"Accept": "application/octet-stream",
                                              "Range": "bytes=0-0"})
    if token and urlparse(part["url"]).hostname == "api.github.com":
        request.add_unredirected_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return {"reachable": True}
    except urllib.error.HTTPError as error:
        return {"reachable": False, "http_status": error.code,
                "detail": _access_hint(error.code)}
    except Exception as error:                      # network down, DNS, proxy
        return {"reachable": False, "detail": f"{type(error).__name__}: {error}"}


def _access_hint(status):
    if status in (401, 403):
        return ("the credential does not grant access to "
                "m2lab-ntu/KmerFormer's releases")
    if status == 404:
        return ("not found. An asset that is private, or attached to an "
                "unpublished (draft) release, returns 404 to an account "
                "without access rather than a permission error. Set GH_TOKEN "
                "or authenticate the gh CLI with an account that can read it; "
                "see weights/README.md")
    return f"HTTP {status}"


def download_model(model_id, output, *, registry_path=None):
    """Download an immutable, checksummed model archive from the release registry."""
    import tarfile
    registry = json.loads(Path(registry_path or Path(__file__).parent / "assets" / "models.json").read_text())
    entry = registry["models"].get(model_id)
    if not entry:
        raise ValueError(f"Unknown model id {model_id!r}; `kmerformer-model list` shows the known ones")
    if not entry.get("download"):
        raise ModelUnavailable(
            f"{model_id!r} has no public download: the trained weights are not "
            f"publicly released. They are available to reviewers on request; "
            f"see weights/README.md")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    token = _github_token()
    with tempfile.TemporaryDirectory(prefix="kmerformer-download-", dir=output.parent) as temporary:
        archive = Path(temporary) / "bundle.tar"
        with archive.open("wb") as dest:
            for part in entry["download"]["parts"]:
                request = urllib.request.Request(part["url"], headers={"Accept": "application/octet-stream"})
                if urlparse(part["url"]).hostname == "api.github.com" and token:
                    # An unredirected header is not forwarded to the release
                    # storage host when GitHub redirects the download.
                    request.add_unredirected_header("Authorization", "Bearer " + token)
                try:
                    source = urllib.request.urlopen(request, timeout=60)
                except urllib.error.HTTPError as error:
                    # Without this the caller sees a bare urllib traceback and
                    # no way to tell "you lack access" from "the URL is wrong".
                    raise ModelUnavailable(
                        f"Cannot download {model_id!r}: {_access_hint(error.code)}"
                    ) from error
                with source:
                    part_digest = __import__("hashlib").sha256()
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        part_digest.update(chunk)
                        dest.write(chunk)
                if part_digest.hexdigest() != part["sha256"]:
                    raise ValueError("Downloaded model part failed checksum verification")
        if sha256_file(archive) != entry["download"]["sha256"]:
            raise ValueError("Downloaded archive failed checksum verification")
        unpacked = Path(temporary) / "unpacked"
        unpacked.mkdir()
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                if not member.isfile() and not member.isdir():
                    raise ValueError("Model archives may contain only regular files and directories")
                _inside(unpacked.resolve(), member.name)
            source.extractall(unpacked, filter="data")
        manifest = verify_bundle(unpacked)
        if manifest["model_id"] != model_id:
            raise ValueError("Downloaded bundle identity does not match the requested model")
        unpacked.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List model identities and availability")
    listing.add_argument("--check", action="store_true",
                         help="Probe each registered download instead of "
                              "reporting only what the registry and your "
                              "credentials imply")
    verify = commands.add_parser("verify", help="Verify a local model bundle")
    verify.add_argument("bundle")
    download = commands.add_parser("download")
    download.add_argument("model_id")
    download.add_argument("--output", required=True)
    export = commands.add_parser("export", help="Package a trusted local KmerFormer checkpoint")
    for name in ("checkpoint", "config", "labels", "output", "model-id"):
        export.add_argument("--" + name, required=True)
    export.add_argument("--licence", default="LicenseRef-Pending")
    args = parser.parse_args()
    if args.command == "list":
        registry = json.loads(
            (Path(__file__).parent / "assets" / "models.json").read_text())
        token = _github_token()
        for model_id, entry in registry["models"].items():
            entry["availability"] = describe_availability(entry, token=token)
            if args.check and entry.get("download", {}).get("parts"):
                entry["availability"].update(check_reachable(entry, token=token))
        print(json.dumps(registry, indent=2))
    elif args.command == "verify":
        print(json.dumps(verify_bundle(args.bundle), indent=2))
    elif args.command == "download":
        try:
            print(download_model(args.model_id, args.output))
        except ModelUnavailable as error:
            # A traceback here reads as a bug in the package rather than as a
            # statement about the caller's access, which is what it is.
            raise SystemExit(str(error))
    else:
        print(json.dumps(export_bundle(args.checkpoint, args.config, args.labels, args.output,
                                       model_id=args.model_id, licence=args.licence), indent=2))


if __name__ == "__main__":
    main()

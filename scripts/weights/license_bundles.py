#!/usr/bin/env python3
"""Apply an explicitly selected weight licence to verified local model bundles."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from kmerformer.bundle import SUPPORTED_LICENCES, copy_licence_texts, verify_bundle
from kmerformer.splits import sha256_file


def apply_licence(bundle, licence):
    if licence not in SUPPORTED_LICENCES:
        raise ValueError(f"No bundled licence text for {licence!r}")
    bundle = Path(bundle)
    manifest = verify_bundle(bundle)
    tokenizer_licence = manifest.get("tokenizer_licence", "LicenseRef-Pending")
    for name in copy_licence_texts(bundle, licence, tokenizer_licence):
        path = bundle / name
        manifest["files"][name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    manifest["weights_licence"] = licence
    manifest["weights_licence_scope"] = "model.safetensors"
    # Compact metadata avoids shifting existing TAR payloads when it still
    # occupies the same number of 512-byte blocks as the prior manifest.
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", suffix=".json", dir=bundle)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, bundle / "manifest.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundles", nargs="+", type=Path)
    parser.add_argument("--licence", choices=sorted(SUPPORTED_LICENCES), required=True,
                        help="Weight terms selected by the rights holder; tokenizer terms are retained")
    args = parser.parse_args()
    for bundle in args.bundles:
        manifest = apply_licence(bundle, args.licence)
        print(json.dumps({key: manifest[key] for key in
                          ("model_id", "weights_licence", "weights_licence_scope", "tokenizer_licence")}),
              flush=True)


if __name__ == "__main__":
    main()

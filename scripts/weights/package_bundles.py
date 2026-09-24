#!/usr/bin/env python3
"""Create checksummed model download parts without an intermediate large tar."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from kmerformer.bundle import BUNDLE_LICENCE_FILES, verify_bundle


class Parts:
    def __init__(self, directory, stem, max_size):
        self.directory, self.stem, self.max_size = directory, stem, max_size
        self.total = hashlib.sha256()
        self.part_hash = hashlib.sha256()
        self.parts, self.file, self.size = [], None, 0

    def _close_part(self):
        if self.file:
            self.file.close()
            self.parts.append({"filename": Path(self.file.name).name,
                               "sha256": self.part_hash.hexdigest(), "size_bytes": self.size})
            self.file = None

    def write(self, data):
        self.total.update(data)
        remaining = memoryview(data)
        while remaining:
            if self.file is None:
                path = self.directory / f"{self.stem}.tar.part{len(self.parts):03d}"
                self.file = path.open("xb")
                self.part_hash, self.size = hashlib.sha256(), 0
            amount = min(self.max_size - self.size, len(remaining))
            piece, remaining = remaining[:amount], remaining[amount:]
            self.file.write(piece)
            self.part_hash.update(piece)
            self.size += amount
            if self.size == self.max_size:
                self._close_part()
        return len(data)

    def close(self):
        self._close_part()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundles", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part-mib", type=int, default=1024)
    args = parser.parse_args()
    if args.part_mib < 1:
        parser.error("part-mib must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    for bundle in args.bundles:
        manifest = verify_bundle(bundle)
        model_id = manifest["model_id"]
        writer = Parts(args.output, model_id, args.part_mib * 1024**2)
        try:
            with tarfile.open(fileobj=writer, mode="w|") as archive:
                # Append licence texts so updating them can retain existing
                # payload offsets and unchanged remote download parts.
                payload = sorted(set(manifest["files"]) - BUNDLE_LICENCE_FILES)
                licences = sorted(set(manifest["files"]) & BUNDLE_LICENCE_FILES)
                for name in ["manifest.json", *payload, *licences]:
                    info = archive.gettarinfo(str(bundle / name), arcname=name)
                    info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
                    info.mode = 0o644
                    with (bundle / name).open("rb") as source:
                        archive.addfile(info, source)
        finally:
            writer.close()
        record = {"model_id": model_id, "sha256": writer.total.hexdigest(), "parts": writer.parts}
        (args.output / f"{model_id}.download.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

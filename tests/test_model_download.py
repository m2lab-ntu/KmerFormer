"""Real archive round trips and failure paths without a network dependency."""
import hashlib
import io
import json
import tarfile
import urllib.request

import pytest
import torch
from safetensors.torch import load_file

from kmerformer.bundle import download_model, save_tensor_stream


def registry_for(tmp_path, files, *, model_id="fixture", symlink=False):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, value in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(value)
            if symlink:
                member.type, member.linkname, member.size = tarfile.SYMTYPE, "../outside", 0
            archive.addfile(member, io.BytesIO(value))
    data = buffer.getvalue()
    parts = []
    for number, start in enumerate(range(0, len(data), 4096)):
        path = tmp_path / f"part{number}"
        chunk = data[start:start + 4096]
        path.write_bytes(chunk)
        parts.append({"url": path.as_uri(), "sha256": hashlib.sha256(chunk).hexdigest()})
    record = {"schema_version": 1, "models": {model_id: {"download": {
        "parts": parts, "sha256": hashlib.sha256(data).hexdigest()}}}}
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(record))
    return registry


def tiny_bundle_files():
    files = {"model.safetensors": b"tensors", "config.json": b"{}", "labels.json": b"[]"}
    manifest = {"schema_version": 1, "model_id": "fixture", "files": {
        name: {"sha256": hashlib.sha256(value).hexdigest(), "size_bytes": len(value)}
        for name, value in files.items()}}
    files["manifest.json"] = json.dumps(manifest).encode()
    return files


def test_verified_parts_roundtrip_and_no_overwrite(tmp_path):
    files = tiny_bundle_files()
    registry = registry_for(tmp_path, files)
    output = tmp_path / "downloaded"
    assert download_model("fixture", output, registry_path=registry) == output
    assert {p.name: p.read_bytes() for p in output.iterdir()} == files
    with pytest.raises(FileExistsError):
        download_model("fixture", output, registry_path=registry)


@pytest.mark.parametrize("problem", ["part", "archive", "identity", "path", "symlink"])
def test_invalid_download_is_rejected_without_partial_output(tmp_path, problem):
    files = tiny_bundle_files()
    if problem == "path":
        files["../outside"] = b"bad"
    model_id = "different" if problem == "identity" else "fixture"
    registry = registry_for(tmp_path, files, model_id=model_id, symlink=problem == "symlink")
    record = json.loads(registry.read_text())
    if problem in ("part", "archive"):
        target = record["models"][model_id]["download"]
        if problem == "part":
            target = target["parts"][0]
        target["sha256"] = "0" * 64
        registry.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        download_model(model_id, tmp_path / "output", registry_path=registry)
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "outside").exists()
    assert not list(tmp_path.glob("kmerformer-download-*"))


def test_private_github_auth_stays_off_redirected_requests(tmp_path, monkeypatch):
    registry = registry_for(tmp_path, tiny_bundle_files())
    record = json.loads(registry.read_text())
    part = record["models"]["fixture"]["download"]["parts"][0]
    payload = (tmp_path / "part0").read_bytes()
    part["url"] = "https://api.github.com/repos/owner/project/releases/assets/123"
    registry.write_text(json.dumps(record))
    original = urllib.request.urlopen
    seen = []
    def urlopen(request, **kwargs):
        if request.full_url.startswith("https://api.github.com/"):
            assert request.get_header("Authorization") == "Bearer fixture-token"
            redirected = urllib.request.HTTPRedirectHandler().redirect_request(
                request, None, 302, "Found", {}, "https://release-assets.githubusercontent.com/example")
            assert redirected.get_header("Authorization") is None
            seen.append(True)
            return io.BytesIO(payload)
        assert request.get_header("Authorization") is None
        return original(request, **kwargs)
    monkeypatch.setenv("GH_TOKEN", "fixture-token")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    download_model("fixture", tmp_path / "output", registry_path=registry)
    assert seen == [True]


def test_streamed_tensor_format_handles_dtypes_shapes_and_empty_tensors(tmp_path):
    tensors = {str(dtype): torch.arange(6).reshape(2, 3).to(dtype) for dtype in (
        torch.float32, torch.float16, torch.bfloat16, torch.float64,
        torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool)}
    tensors.update(scalar=torch.tensor(7.), empty=torch.empty(0, 3))
    path = tmp_path / "weights.safetensors"
    save_tensor_stream(tensors, path)
    restored = load_file(path)
    assert restored.keys() == tensors.keys()
    assert all(torch.equal(restored[key], value) for key, value in tensors.items())
    with pytest.raises(ValueError, match="contiguous"):
        save_tensor_stream({"bad": torch.ones(2, 3).T}, tmp_path / "bad")

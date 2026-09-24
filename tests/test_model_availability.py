#!/usr/bin/env python3
"""The registry must say what the caller can actually fetch.

WHY. The trained weights are withheld until the paper is accepted, so the
registry lists the models with their checksums and licences but no download
parts. Two earlier states each misled: printing the registry verbatim implied
every model was downloadable, and a refused download surfaced as a bare urllib
traceback. And a non-GitHub URL used to be reported "open" -- a private Hugging
Face dataset looks identical to a public one from its URL alone.
"""

import json
import urllib.error
from pathlib import Path

import pytest

from kmerformer.bundle import (ModelUnavailable, _access_hint,
                               describe_availability, download_model)

REPO = Path(__file__).resolve().parent.parent
REGISTRY = json.loads(
    (REPO / "kmerformer" / "assets" / "models.json").read_text())
GH = {"download": {"parts": [{"url": "https://api.github.com/repos/o/r/releases/assets/1"}]}}


@pytest.mark.parametrize("model_id", sorted(REGISTRY["models"]))
def test_no_model_advertises_a_public_download_while_weights_are_withheld(model_id):
    entry = REGISTRY["models"][model_id]
    assert not entry.get("download"), f"{model_id} still registers download parts"
    assert entry["status"] == "not_publicly_released"
    report = describe_availability(entry, token=None)
    assert report["state"] == "unregistered"
    assert "not publicly released" in report["detail"]


def test_download_of_a_withheld_model_explains_itself(tmp_path):
    model_id = next(iter(REGISTRY["models"]))
    with pytest.raises(ModelUnavailable) as caught:
        download_model(model_id, tmp_path / "out")
    message = str(caught.value)
    assert "not publicly released" in message
    assert "reviewers on request" in message


def test_an_unknown_model_id_is_a_different_error(tmp_path):
    with pytest.raises(ValueError, match="Unknown model id"):
        download_model("no-such-model", tmp_path / "out")


def test_github_asset_state_depends_on_credentials():
    assert describe_availability(GH, token=None)["state"] == "needs-credentials"
    assert describe_availability(GH, token="ghp_example")["state"] == "credentialed"


@pytest.mark.parametrize("url", [
    "https://huggingface.co/datasets/someone/private-weights/resolve/main/a.tar",
    "https://zenodo.org/records/1/files/a.tar"])
def test_other_hosts_are_unverified_not_open(url):
    """A URL cannot tell a private dataset from a public one; only --check can."""
    entry = {"download": {"parts": [{"url": url}]}}
    report = describe_availability(entry, token=None)
    assert report["state"] == "unverified"
    assert "--check" in report["detail"]


@pytest.mark.parametrize("status,must_mention", [
    (404, "without access"), (403, "access"), (401, "access")])
def test_the_refusal_explains_the_status(status, must_mention):
    assert must_mention in _access_hint(status).lower()


def test_a_refused_part_raises_the_typed_error(tmp_path, monkeypatch):
    import kmerformer.bundle as bundle

    def refuse(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    registry = tmp_path / "models.json"
    registry.write_text(json.dumps({"models": {"m": {"download": {
        "sha256": "0" * 64, "parts": [{"url": GH["download"]["parts"][0]["url"],
                                       "sha256": "0" * 64}]}}}}))
    monkeypatch.setattr(bundle.urllib.request, "urlopen", refuse)
    monkeypatch.setattr(bundle, "_github_token", lambda: None)
    with pytest.raises(ModelUnavailable) as caught:
        download_model("m", tmp_path / "out", registry_path=registry)
    assert "GH_TOKEN" in str(caught.value)


def test_readmes_say_the_weights_are_withheld():
    for doc in ("README.md", "weights/README.md"):
        text = (REPO / doc).read_text()
        assert "not currently available for public download" in text, doc
        assert "private repository" not in text, f"{doc} still calls the repository private"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

#!/usr/bin/env python3
"""The registry must say what the caller can actually fetch.

WHY. The registry lists each model with one checksummed archive on Hugging
Face. Two earlier states each misled: printing the registry verbatim implied
every model was downloadable, and a refused download surfaced as a bare urllib
traceback. And a non-GitHub URL used to be reported "open" -- a private Hugging
Face dataset looks identical to a public one from its URL alone, so `list`
says "unverified" and `list --check` asks the host.
"""

import copy
import json
import sys
import urllib.error
from pathlib import Path

import pytest

from kmerformer.bundle import (ModelUnavailable, _access_hint,
                               describe_availability, download_model,
                               probe_availability)

REPO = Path(__file__).resolve().parent.parent
REGISTRY = json.loads(
    (REPO / "kmerformer" / "assets" / "models.json").read_text())
GH = {"download": {"parts": [{"url": "https://api.github.com/repos/o/r/releases/assets/1"}]}}
HOSTED = "https://huggingface.co/ymj1123/kmerformer-0.2.0rc1-weights"


@pytest.mark.parametrize("model_id", sorted(REGISTRY["models"]))
def test_each_model_registers_one_checksummed_archive(model_id):
    entry = REGISTRY["models"][model_id]
    assert entry["status"] == "publicly_released"
    download = entry["download"]
    (part,) = download["parts"]
    assert part["url"] == f"{HOSTED}/resolve/main/{model_id}.tar"
    assert part["filename"] == f"{model_id}.tar"
    assert part["sha256"] == download["sha256"]
    assert len(part["sha256"]) == 64 and set(part["sha256"]) <= set("0123456789abcdef")
    assert part["size_bytes"] > 0
    report = describe_availability(entry, token=None)
    assert report["state"] == "unverified"
    assert "huggingface.co" in report["detail"]


def test_archive_checksums_are_distinct():
    digests = [entry["download"]["sha256"] for entry in REGISTRY["models"].values()]
    assert len(set(digests)) == len(digests)


def test_an_entry_without_a_download_explains_itself(tmp_path):
    entry = {"model_id": "m"}
    report = describe_availability(entry, token=None)
    assert report["state"] == "unregistered"
    assert "weights/README.md" in report["detail"]
    registry = tmp_path / "models.json"
    registry.write_text(json.dumps({"models": {"m": entry}}))
    with pytest.raises(ModelUnavailable) as caught:
        download_model("m", tmp_path / "out", registry_path=registry)
    assert "no registered download" in str(caught.value)


def test_an_unknown_model_id_is_a_different_error(tmp_path):
    with pytest.raises(ValueError, match="Unknown model id"):
        download_model("no-such-model", tmp_path / "out")


def test_github_asset_state_depends_on_credentials():
    assert describe_availability(GH, token=None)["state"] == "needs-credentials"
    assert describe_availability(GH, token="ghp_example")["state"] == "credentialed"


@pytest.mark.parametrize("url", [
    "https://huggingface.co/someone/private-weights/resolve/main/a.tar",
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


@pytest.mark.parametrize("status", [401, 403, 404])
def test_a_refusal_from_another_host_names_that_host(status):
    hint = _access_hint(status, "huggingface.co")
    assert "huggingface.co" in hint
    assert "GH_TOKEN" not in hint and "m2lab-ntu" not in hint


def test_a_refused_hosted_archive_raises_the_typed_error(tmp_path, monkeypatch):
    import kmerformer.bundle as bundle

    def refuse(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(bundle.urllib.request, "urlopen", refuse)
    monkeypatch.setattr(bundle, "_github_token", lambda: None)
    model_id = next(iter(REGISTRY["models"]))
    with pytest.raises(ModelUnavailable) as caught:
        download_model(model_id, tmp_path / "out")
    assert "huggingface.co" in str(caught.value)
    assert not (tmp_path / "out").exists()


def test_readmes_say_where_the_weights_are():
    for doc in ("README.md", "weights/README.md"):
        text = (REPO / doc).read_text()
        assert HOSTED in text, doc
        assert "not currently available for public download" not in text, doc
        assert "private repository" not in text, f"{doc} still calls the repository private"


class _Answered:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _hosted_entry():
    return copy.deepcopy(next(iter(REGISTRY["models"].values())))


def test_a_successful_probe_replaces_unverified(monkeypatch):
    """`list --check` must not tell the caller to run `list --check`."""
    import kmerformer.bundle as bundle
    monkeypatch.setattr(bundle.urllib.request, "urlopen",
                        lambda request, timeout=None: _Answered())
    report = probe_availability(_hosted_entry(), token=None)
    assert report["state"] == "reachable"
    assert report["reachable"] is True
    assert "huggingface.co" in report["detail"]
    assert "--check" not in report["detail"]


@pytest.mark.parametrize("status", [401, 403, 404])
def test_a_refused_probe_reports_unreachable(monkeypatch, status):
    import kmerformer.bundle as bundle

    def refuse(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, status, "refused", {}, None)

    monkeypatch.setattr(bundle.urllib.request, "urlopen", refuse)
    report = probe_availability(_hosted_entry(), token=None)
    assert report["state"] == "unreachable"
    assert report["reachable"] is False
    assert report["http_status"] == status
    assert "huggingface.co" in report["detail"]


def test_a_network_failure_reports_unreachable(monkeypatch):
    import kmerformer.bundle as bundle

    def offline(request, timeout=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(bundle.urllib.request, "urlopen", offline)
    report = probe_availability(_hosted_entry(), token=None)
    assert report["state"] == "unreachable"
    assert "URLError" in report["detail"]
    assert "http_status" not in report


def test_a_probe_without_a_download_stays_unregistered(monkeypatch):
    import kmerformer.bundle as bundle

    def must_not_call(request, timeout=None):
        raise AssertionError("no download is registered, so nothing to probe")

    monkeypatch.setattr(bundle.urllib.request, "urlopen", must_not_call)
    assert probe_availability({"model_id": "m"}, token=None)["state"] == "unregistered"


def test_list_check_prints_the_probed_state(monkeypatch, capsys):
    import kmerformer.bundle as bundle
    monkeypatch.setattr(bundle.urllib.request, "urlopen",
                        lambda request, timeout=None: _Answered())
    monkeypatch.setattr(bundle, "_github_token", lambda: None)
    monkeypatch.setattr(sys, "argv", ["kmerformer-model", "list", "--check"])
    bundle.main()
    listed = json.loads(capsys.readouterr().out)["models"]
    states = {entry["availability"]["state"] for entry in listed.values()}
    assert states == {"reachable"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

"""Regression tests for offline review, safe inspection and package boundaries."""
import importlib.util
import io
import hashlib
import json
import pickle
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent


def module(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_offline_smoke_round_trip():
    report = module("scripts/smoke_cpu.py").run()
    assert report["checkpoint_round_trip"] == "bit-identical"
    assert report["logit_shape"] == [2, 2]


def test_metadata_reads_config_without_tensors(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"config": {"model": {"type": "shallow_transformer"}},
                "epoch": 12, "val_acc": 0.91112,
                "model_state_dict": {"weight": torch.ones(2, 3)}}, path)
    result = module("scripts/weights/inspect_checkpoint.py").read_metadata(path)
    assert result["epoch"] == 12
    assert result["val_acc"] == 0.91112
    assert result["config"]["model"]["type"] == "shallow_transformer"


@pytest.mark.parametrize("global_name", [b"cbuiltins\neval\n.", b"cnumpy\nload\n.",
                                          b"cos\nsystem\n."])
def test_metadata_rejects_executable_globals(global_name):
    reader = module("scripts/weights/inspect_checkpoint.py")
    with pytest.raises(pickle.UnpicklingError, match="unsupported pickle global"):
        reader._MetadataUnpickler(io.BytesIO(global_name)).load()


@pytest.mark.parametrize("path", [".git/config", ".env", "weights/best.pt", "runs/config.yaml",
                                  "docs/agent_prompts/task.md", "scripts/__pycache__/x.py",
                                  "docs/assets/checkpoint.pt", "local_secret.json"])
def test_archive_excludes_local_material(path):
    assert not module("scripts/package_reviewer.py").include(Path(path))


@pytest.mark.parametrize("path", ["docs/REPRODUCE.md", "CITATION.cff", "scripts/reviewer_check.py",
                                  "docs/assets/example_preds_clean_common.npz", "weights/README.md"])
def test_archive_includes_reviewer_material(path):
    assert module("scripts/package_reviewer.py").include(Path(path))


def test_document_check_excludes_ignored_snapshots_but_keeps_new_source(release_repo):
    root, _ = release_repo
    (root / ".gitignore").write_text("staging/\n")
    (root / "staging").mkdir()
    (root / "staging/old.md").write_text("[old report](missing.md)\n")
    (root / "new.md").write_text("new source document\n")
    found = module("scripts/check_doc_links.py").markdown_files(root)
    assert root / "README.md" in found
    assert root / "new.md" in found
    assert root / "staging/old.md" not in found


def test_copied_validation_links_point_to_the_exact_source_commit(tmp_path):
    delivery = module("scripts/weights/prepare_zenodo.py")
    report = tmp_path / "docs/validation/report.md"
    report.parent.mkdir(parents=True)
    (tmp_path / "docs/MISSING_ASSETS.md").write_text("# Inputs\n")
    report.write_text("[inputs](../MISSING_ASSETS.md#inputs) [local](#scope) [web](https://example.org)\n")
    result = delivery.portable_markdown(report, tmp_path, "a" * 40)
    assert "https://github.com/m2lab-ntu/KmerFormer/blob/" + "a" * 40 + "/docs/MISSING_ASSETS.md#inputs" in result
    assert "[local](#scope) [web](https://example.org)" in result
    (tmp_path / "docs/MISSING_ASSETS.md").unlink()
    with pytest.raises(ValueError, match="Broken report link"):
        delivery.portable_markdown(report, tmp_path, "a" * 40)


def test_zenodo_rejects_wheel_with_stale_registry(release_repo, tmp_path):
    root, _ = release_repo
    delivery = module("scripts/weights/prepare_zenodo.py")
    (root / "kmerformer/assets").mkdir(parents=True)
    (root / "kmerformer/__init__.py").write_text('__version__ = "0.2.0rc1"\n')
    (root / "kmerformer/assets/models.json").write_text('{"models": {"current": {}}}\n')
    (root / "pyproject.toml").write_text('[project]\nversion = "0.2.0rc1"\nreadme = "README.md"\n')
    fixture_commit(root)
    for stale in (None, "registry", "readme"):
        wheel = tmp_path / f"model-{stale}.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.write(root / "kmerformer/__init__.py", "kmerformer/__init__.py")
            if stale == "registry":
                archive.writestr("kmerformer/assets/models.json", '{"models": {}}\n')
            else:
                archive.write(root / "kmerformer/assets/models.json", "kmerformer/assets/models.json")
            readme = "old author list" if stale == "readme" else (root / "README.md").read_text()
            archive.writestr("kmerformer-0.2.0rc1.dist-info/METADATA", "Name: kmerformer\nVersion: 0.2.0rc1\n\n" + readme)
        if stale:
            with pytest.raises(ValueError, match="stale"):
                delivery.verify_wheel(root, wheel)
        else:
            assert delivery.verify_wheel(root, wheel) == "0.2.0rc1"


@pytest.mark.parametrize("tokenizer_licence", ["MIT", "CC-BY-4.0", "CC-BY-NC-SA-4.0"])
def test_licensing_existing_bundle_preserves_payloads_and_upstream_terms(tmp_path, tokenizer_licence):
    from kmerformer.bundle import LICENCE_ASSETS, verify_bundle
    files = {}
    for name in ("model.safetensors", "config.json", "labels.json"):
        data = (name + " original bytes\n").encode()
        (tmp_path / name).write_bytes(data)
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    before = {"schema_version": 1, "model_id": "fixture", "files": files,
              "weights_licence": "LicenseRef-Pending", "tokenizer_licence": tokenizer_licence,
              "source_checkpoint_sha256": "a" * 64}
    (tmp_path / "manifest.json").write_text(json.dumps(before, indent=2))
    after = module("scripts/weights/license_bundles.py").apply_licence(tmp_path, "Apache-2.0")
    assert verify_bundle(tmp_path) == after
    assert after["weights_licence_scope"] == "model.safetensors"
    assert after["tokenizer_licence"] == before["tokenizer_licence"]
    assert after["source_checkpoint_sha256"] == before["source_checkpoint_sha256"]
    for name, info in files.items():
        assert after["files"][name] == info
    assert (tmp_path / "TOKENIZER_LICENSE.txt").read_bytes() == (LICENCE_ASSETS / f"{tokenizer_licence}.txt").read_bytes()
    assert (tmp_path / "WEIGHTS_LICENSE.txt").read_bytes() == (LICENCE_ASSETS / "Apache-2.0.txt").read_bytes()


@pytest.mark.parametrize("corrupt", [None, "part", "archive", "identity"])
def test_zenodo_model_concatenation_preserves_archive_and_checks_hashes(tmp_path, corrupt):
    import tarfile
    delivery = module("scripts/weights/prepare_zenodo.py")
    manifest = {"model_id": "fixture", "weights_licence": "LicenseRef-Pending",
                "source_checkpoint_sha256": "a" * 64}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        data = json.dumps(manifest).encode()
        member = tarfile.TarInfo("manifest.json")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    data = buffer.getvalue()
    entry = dict(manifest, download={"sha256": hashlib.sha256(data).hexdigest(), "parts": []})
    for i, start in enumerate(range(0, len(data), 2048)):
        part = data[start:start + 2048]
        name = f"part{i}"
        (tmp_path / name).write_bytes(part)
        entry["download"]["parts"].append({"filename": name, "size_bytes": len(part),
                                             "sha256": hashlib.sha256(part).hexdigest()})
    if corrupt == "part":
        entry["download"]["parts"][0]["sha256"] = "0" * 64
    elif corrupt == "archive":
        entry["download"]["sha256"] = "0" * 64
    elif corrupt == "identity":
        entry["model_id"] = "wrong"
    target = tmp_path / "combined.tar"
    if corrupt:
        with pytest.raises(ValueError):
            delivery.combine_parts(tmp_path, entry, target)
    else:
        info, found = delivery.combine_parts(tmp_path, entry, target)
        assert found == manifest and target.read_bytes() == data
        assert info == {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def test_separate_zenodo_records_have_scoped_licences_and_complete_checksums(tmp_path):
    import yaml
    delivery = module("scripts/weights/prepare_zenodo.py")
    files = {}
    for name in ("source.zip", "library.whl", "model.tar", "README.txt", "VALIDATION.md"):
        (tmp_path / name).write_bytes((name + " payload").encode())
        files[name] = delivery.file_info(tmp_path / name)
    shutil.copyfile(ROOT / "CITATION.cff", tmp_path / "CITATION.cff")
    files["CITATION.cff"] = delivery.file_info(tmp_path / "CITATION.cff")
    citation = yaml.safe_load((tmp_path / "CITATION.cff").read_text())
    models = {"model": {"archive": "model.tar", "weights_licence": "Apache-2.0", "tokenizer_licence": "CC-BY-NC-SA-4.0"}}
    record = {"schema_version": 1, "version": "fixture", "source_commit": "a" * 40,
              "creators": citation["authors"], "state": "prepared_locally", "publication_access": "not_selected",
              "pending_weight_licences": [], "models": models, "files": files}
    (tmp_path / "DEPOSIT_MANIFEST.json").write_text(json.dumps(record))
    (tmp_path / "SHA256SUMS").write_text("initial list")
    summaries = delivery.separate_records(tmp_path, record)
    assert set(summaries) == {"code", "weights"}
    assert not (tmp_path / "model.tar").exists()
    for kind in summaries:
        root = tmp_path / kind
        deposit = json.loads((root / "DEPOSIT_MANIFEST.json").read_text())
        assert deposit["doi"] is None and deposit["companion_record"]["doi"] is None
        assert deposit["creators"] == record["creators"]
        lines = (root / "SHA256SUMS").read_text().splitlines()
        assert {line.split("  ")[1] for line in lines} == {p.name for p in root.iterdir()} - {"SHA256SUMS"}
        for line in lines:
            digest, name = line.split("  ")
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        cff = yaml.safe_load((root / "CITATION.cff").read_text())
        assert cff["authors"] == citation["authors"]
        assert cff["license"] == ("MIT" if kind == "code" else ["Apache-2.0"])
    assert (tmp_path / "weights/model.tar").read_bytes() == b"model.tar payload"
    assert not (tmp_path / "weights/source.zip").exists()
    assert "CC BY-NC-SA 4.0" in (tmp_path / "weights/README.txt").read_text()


def test_archived_prediction_evidence():
    report = module("scripts/review_predictions.py").verify(ROOT / "docs/assets")
    assert len(report["arms"]) == 14
    assert not report["full_metrics"]


def test_genus_aware_taxonomy_and_species_only_control(tmp_path):
    builder = module("scripts/baselines/build_kraken2_db_1535.py")
    builder.build_taxonomy(tmp_path, {3, 8, 20}, 2, {3: 0, 8: 0, 20: 2}, {0: "A", 2: "B"})
    text = (tmp_path / "taxonomy/nodes.dmp").read_text()
    rows = {int(fields[0]): (int(fields[1]), fields[2])
            for line in text.splitlines()
            if (fields := [s.strip() for s in line.split("|")])}
    assert rows[5] == (10000, "species")
    assert rows[10] == (10000, "species")
    assert rows[22] == (10002, "species")
    assert rows[10000] == (1, "genus")
    assert len(rows) == 6
    with pytest.raises(ValueError, match="refusing to replace"):
        builder.build_taxonomy(tmp_path, {3, 8, 20}, 2)
    builder.build_taxonomy(tmp_path / "control", {3, 8, 20}, 2)
    assert "\tgenus\t" not in (tmp_path / "control/taxonomy/nodes.dmp").read_text()


def test_taxonomy_rejects_incomplete_or_colliding_mapping(tmp_path):
    builder = module("scripts/baselines/build_kraken2_db_1535.py")
    with pytest.raises(ValueError):
        builder.build_taxonomy(tmp_path, {3, 8}, 2, {3: 0})
    with pytest.raises(ValueError):
        builder.build_taxonomy(tmp_path, {9998}, 2, {9998: 0})
    assert not (tmp_path / "taxonomy").exists()


@pytest.mark.skipif(shutil.which("git") is None, reason="package construction needs Git")
def test_package_has_current_files_hashes_and_no_history(tmp_path):
    packer = module("scripts/package_reviewer.py")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name in packer.ROOT_FILES:
        (tmp_path / name).write_text("initial\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    # Use a tree object as a synthetic HEAD: this test creates no commit
    # or author identity and never changes the real repository's Git settings.
    tree = subprocess.check_output(["git", "write-tree"], cwd=tmp_path, text=True).strip()
    (tmp_path / ".git/HEAD").write_text(tree + "\n")
    (tmp_path / "README.md").write_text("current uncommitted text\n")
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights/private.pt").write_bytes(b"not for reviewers")
    out = tmp_path / "review.zip"
    packer.package(tmp_path, out)
    with zipfile.ZipFile(out) as archive:
        assert archive.read("KmerFormer/README.md") == b"current uncommitted text\n"
        manifest = json.loads(archive.read("KmerFormer/RELEASE_MANIFEST.json"))
        for name, info in manifest["files"].items():
            contents = archive.read("KmerFormer/" + name)
            assert hashlib.sha256(contents).hexdigest() == info["sha256"]
        assert not any(".git/" in p or p.endswith(".pt") for p in archive.namelist())
    with pytest.raises(FileExistsError):
        packer.package(tmp_path, out)


def test_shell_drivers_resolve_migrated_tools():
    expected = {
        "scripts/track_a/track_a_04_run_inference.sh": ["scripts/eval", "configs/baselines/ntv2_lora_50M.yaml"],
        "scripts/track_a/track_a_05_metrics.sh": ["scripts/eval"],
        "scripts/track_b/tb_05_run_inference.sh": ["scripts/eval/extract_mt_predictions.py", "scripts/eval/run_genus_rctta.py"],
    }
    for script, targets in expected.items():
        text = (ROOT / script).read_text()
        for target in targets:
            assert target in text
            assert (ROOT / target).exists()
    text = (ROOT / "scripts/track_b/tb_05_run_inference.sh").read_text()
    assert "--save_probs" not in text and "--allow_unmapped_labels" not in text


@pytest.mark.skipif(shutil.which("bash") is None, reason="shell checks need Bash")
def test_all_shell_scripts_parse():
    for script in sorted((ROOT / "scripts").rglob("*.sh")):
        subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)


def test_release_metadata_retains_manuscript_authors():
    import yaml
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    authors = [(a["given-names"], a["family-names"]) for a in citation["authors"]]
    assert authors == [("Ming-Ju", "Yang"), ("TING-YU", "YEN"),
                       ("Chien-Yu", "Chen"), ("Joyce Tzu-Yu", "Liu")]
    assert [(a["given-names"], a["family-names"]) for a in citation["preferred-citation"]["authors"]] == authors


def test_track_b_decodes_genus_nodes_from_repaired_builder(tmp_path):
    labels = tmp_path / "labels.tsv"
    labels.write_text("species_class\tgenus_class\n3\t0\n8\t0\n20\t2\n")
    scorer = module("scripts/track_b/tb_06_run_kraken2.py")
    lookup, width = scorer.build_taxid_lut(str(labels))
    assert width == 3
    assert lookup[1] == -1
    assert lookup[5] == lookup[10] == lookup[10000] == 0
    assert lookup[22] == lookup[10002] == 2


@pytest.fixture
def release_repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("release construction needs Git")
    root = tmp_path / "repo"
    root.mkdir()
    packer = module("scripts/package_reviewer.py")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for name in packer.ROOT_FILES:
        (root / name).write_text("initial\n")
    (root / ".gitignore").write_text("*.pt\n")
    (root / "weights").mkdir()
    (root / "weights/.gitkeep").touch()
    (root / "weights/local.pt").write_bytes(b"ignored local weight")
    (root / "README.md").chmod(0o755)
    fixture_commit(root)
    return root, packer


def fixture_commit(root):
    # Identity exists only in the temporary test repository, never the real repo.
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Test Fixture", "-c",
                    "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false",
                    "commit", "-qm", "Fixture"], cwd=root, check=True)


def test_release_matches_entire_commit_and_is_deterministic(release_repo, tmp_path):
    root, packer = release_repo
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    manifest = packer.package(root, first, release=True)
    packer.package(root, second, release=True)
    assert first.read_bytes() == second.read_bytes()
    assert packer.verify_release(root, first) == manifest
    assert "weights/.gitkeep" in manifest["files"]
    assert "weights/local.pt" not in manifest["files"]
    assert manifest["files"]["README.md"]["git_mode"] == "100755"


@pytest.mark.parametrize("state", ["modified", "staged", "untracked", "deleted"])
def test_release_refuses_uncommitted_changes(release_repo, tmp_path, state):
    root, packer = release_repo
    if state == "untracked":
        (root / "new.md").write_text("pending\n")
    elif state == "deleted":
        (root / "README.md").unlink()
    else:
        (root / "README.md").write_text("pending\n")
        if state == "staged":
            subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    with pytest.raises(ValueError, match="clean checkout"):
        packer.package(root, tmp_path / "release.zip", release=True)
    assert not (tmp_path / "release.zip").exists()


def test_release_refuses_silent_tracked_file_omission(release_repo, tmp_path):
    root, packer = release_repo
    (root / "private.txt").write_text("not approved for release\n")
    fixture_commit(root)
    with pytest.raises(ValueError, match="outside release policy"):
        packer.package(root, tmp_path / "release.zip", release=True)


@pytest.mark.parametrize("tamper", ["content", "extra", "manifest", "mode"])
def test_release_verifier_rejects_archive_drift(release_repo, tmp_path, tamper):
    root, packer = release_repo
    original, changed = tmp_path / "original.zip", tmp_path / "changed.zip"
    packer.package(root, original, release=True)
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(changed, "x") as dest:
        for info in source.infolist():
            data = source.read(info)
            if info.filename == "KmerFormer/README.md":
                if tamper == "content":
                    data += b"changed"
                if tamper == "mode":
                    info.external_attr = 0o100644 << 16
            if tamper == "manifest" and info.filename.endswith("RELEASE_MANIFEST.json"):
                data = b"{}"
            dest.writestr(info, data)
        if tamper == "extra":
            dest.writestr("KmerFormer/extra.txt", "unexpected")
    with pytest.raises(ValueError, match="differ"):
        packer.verify_release(root, changed)

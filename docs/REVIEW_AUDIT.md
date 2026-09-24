# Reviewer preparation audit — 2026-09-07

This records software/evidence checks and their limits. It is not a certificate of
full experimental reproducibility. See [REPRODUCE.md](REPRODUCE.md) for current
commands and [the Zenodo guide](../weights/ZENODO.md) for delivery requirements.

## Verified

- CPU regression suite: 519 tests passed, including the synthetic optimizer step
  and strict, bit-identical checkpoint reload.
- All 14 bundled prediction arrays have the expected 100,000 labels, class ranges,
  ordered label fingerprint and recorded prediction fingerprint.
- Read accuracy, declined fraction, Pearson correlation, Bray–Curtis
  dissimilarity, detection AUC and sensitivity at 95% specificity were recomputed
  for all 14 arrays and matched the archived summary within the checker's tolerance.
- Python sources parse, all nine shell scripts pass Bash syntax checks, training
  and evaluation CLI help run, and relative documentation links resolve.
- A source snapshot was extracted outside the checkout, installed non-editably,
  and passed CPU regression tests and the 14-arm full-metrics check without `.git`.
  Importing from outside either source tree resolved to the installed package.
- Ten release-synchronization regression cases cover deterministic committed
  archives, ignored local weights, file-mode preservation, refusal of uncommitted
  or unexpected tracked files, and detection of archive content/manifest drift.
- Three local checkpoints were inspected without loading tensor storages. Their
  stored epochs are 15 / 12 / 15 for 6-mer 50M / exact-13-mer 50M / exact-13-mer
  250M. Metadata inspection does not reproduce their reported test accuracies.
- The separate manuscript numeric-presence guard found 39/39 config banner
  accuracies and 19/19 asserted differences. This does not prove semantic agreement.
- Existing Git authorship and citation author lists were inspected. Release
  attribution retains the existing human identities; no tool identities or
  co-author trailers were added. Git identity settings were not changed. The
  preparation checks themselves do not publish or upload artefacts; any subsequent
  maintainer-authorized commit or push retains the existing human Git identity.

## Material repairs

- Added one CPU reviewer entry point, evidence re-scoring and a source snapshot
  builder with per-file SHA-256 checksums and no Git history or model weights.
- Made genus-only off-catalogue manifests loadable without inventing species
  labels; existing trained genus class IDs are preserved.
- Repaired migrated Track A/B script/config paths, unsupported evaluation flags,
  missing NT output aliases and a shell quoting error. Drivers no longer silently
  switch Conda environments or hide scoring failures.
- Implemented the documented genus-aware Kraken taxonomy, retained species-only
  construction as an explicit control, rejected conflicting existing taxonomies,
  and taught Track B scoring to recognize internal genus calls. This is a tested
  reconstruction, not recovery or validation of the historical full database.
- Restricted checkpoint metadata unpickling to explicitly allowed objects and
  inert placeholders. Arbitrary built-in/NumPy globals are rejected. This is not
  a sandbox against resource exhaustion.
- Corrected evaluation-protocol, parameter-count, model-ranking and release-access
  descriptions. Original prediction arrays and archived metric values were not
  edited to make them agree with the manuscript.

## Environment boundary

Checks were run in a temporary Python 3.11 virtual environment that shares the
host's site-packages, with selected dependencies installed locally into that
environment. No host environment was upgraded. The runner records actual versions
in its JSON output. This is not a clean recreation of either recorded Conda
environment, a proof of all transitive dependencies, or a GPU compatibility test.

The extracted-package check used Python 3.11.7, PyTorch 2.6.0, Transformers 4.46.3,
PEFT 0.20.0, NumPy 2.0.1, pandas 2.2.3, SciPy 1.14.1, scikit-learn 1.5.2,
Matplotlib 3.9.4, seaborn 0.13.2 and pytest 7.4.0. An earlier successful run used
the host's NumPy 1.26.4 and SciPy 1.11.4; that earlier run alone did not satisfy
the package's declared minimum versions.

## Not verified or still missing

- Full training, fresh real-data inference, external downloads and construction
  of the complete Kraken database were not run.
- Genus-only bundled predictions cannot establish species-coverage-masked scores,
  pool provenance or absence of train/test sequence leakage.
- The original 99,742-read filter/membership artefact, full ANI predictions, soil
  configs, training/evaluation FASTAs, frozen accession manifests and exact
  vocabulary are not in the code snapshot.
  Update, 2026-09-08: the NT membership and original prediction records were
  recovered from Nano4 and bundled with the manuscript's
  `supplementary_data/nt_99742_provenance/`. [RESULTS](RESULTS.md) records the
  remaining primary/subset prediction distinction.
- Archived-array versus manuscript score differences remain disclosed in
  [RESULTS.md](RESULTS.md#abundance-against-detection-on-one-pool).
- Reviewer access URLs/DOIs and the weights licence require author decisions.
  No publication or sharing action is implied by preparation of a local archive.

Keep the archive's `RELEASE_MANIFEST.json` with the delivered version. For a
formal `--release` ZIP, its commit identifies the exact source revision and every
tracked file is included with matching bytes and mode. The separate draft mode
records only a starting revision and may include uncommitted edits. Formal Zenodo
delivery must use the committed release mode and pass `--verify-archive` against
the revision pushed to the repository.

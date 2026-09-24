# Asset inventory and release boundaries

Updated for the maintained release on 2026-09-18. This page describes the source
snapshot, separately delivered model bundles and external experimental inputs.

## 1. Included in the code snapshot

- Python implementation, training/configuration files and CPU tests.
- Fourteen small closed-set prediction arrays in [assets/](assets/).
- Their scoring summary and the species-index membership table.
- Offline evidence checks, a trained-model example, synthetic runtime checks and packaging tools.
- Recovered catalogue, NT-v2 membership, soil and ladder evidence in [reproduction/](../reproduction/README.md).

Run [the software and evidence checks](REPRODUCE.md#software-and-evidence-checks). They verify the arrays without reading
the authors' data directories. The coverage-masked accuracy column needs species
labels and is excluded from the offline check.

## 2. Model bundles and external data

Four portable model bundles are uploaded to the private GitHub release draft;
[the registry](../kmerformer/assets/models.json) records their verified URLs and
checksums. The [Zenodo inventory](../weights/ZENODO.md) includes all four bundles.

Three original `.pt` checkpoints are retained in the local `weights/` directory.
Their metadata identifies the expected arms: 6-mer L29 at 50M and exact 13-mer
single-layer mean-pooling models at 50M/250M. Their selected epochs are 15, 12 and
15 respectively. Supplementary A6000 evaluation on 2026-09-08 verified all three
checkpoint hashes and best-validation epochs. The exact one-layer 50M/250M outputs
match every bundled prediction on the verified 100K FASTA. L29 reproduces the
historical and archived predictions under their respective precision/aggregation
settings; see RESULTS. New ordered read/sequence manifests, eight exact-checkpoint
pool evaluations, training records and timing metadata accompany the manuscript's
Supplementary S22; these large inputs/outputs are not bundled in this code snapshot.

Large read pools, full evaluation FASTAs/labels and reference genomes remain
external. Exact model bundles include the vocabulary, and recovered ladder
predictions are now in the evidence catalogue. Earlier operational
inventories recorded local/mounted copies, but this audit does not certify that
every such file is accessible to a reviewer or identical to the original run input.

A download/access record, fixed filenames and checksums are required for a complete
delivery. See [DATA](DATA.md) and [weights](../weights/README.md). Fetch/assembly
scripts under `scripts/fetch/` are maintainer tools for existing machines, not
download instructions for an external reviewer.

### Two traps when comparing fetched metrics against RESULTS.md

**Raw versus filtered Track A.** The raw pool has 595,000 reads from 119 genera.
Removing the three specified accessions gives the reported 580,000-read,
116-genus pool. For the exact 13-mer arms, raw and reported scores differ:

| Arm | Raw 595K | Reported 580K |
|---|---:|---:|
| exact 13-mer 1L, 50M, mean pooling | 28.95% | 27.44% |
| exact 13-mer 1L, 250M, mean pooling | 33.62% | 31.92% |
| exact 13-mer 1L, 50M, attention pooling | 27.63% | 26.12% |

This is not the additional analysis removing six near-identical genomes, whose
effects are 1.13 and 1.75 points at 50M/250M. Keep the filters separate.

**Evaluation checkpoint selection.** The attention-pooling arm's reported 90.74%
comes from `eval_final_rctta`; an earlier `eval_clean_common_rctta` evaluation
records 90.12%. Compare `checkpoint_epoch`, the checkpoint identity and pool
before comparing scores. `scripts/check_epoch_selection.py` checks the selected
epoch against the training history; it does not prove two different codebases
use equivalent selection rules.

Some historical training runs also lack final summary files because of a
post-training reporting bug. The repository contains the fix and a regression
test. Training history and evaluation metadata remain independent checks; finding
an evaluation-side resource report is not evidence of a training-side report.

## 3. Provenance requiring recovery or confirmation

- The original 99,742-read NT-v2 membership/filter was recovered from Nano4 on
  2026-09-08 and exactly reconstructed. It accompanies the manuscript in
  `supplementary_data/nt_99742_provenance/`; the maintained code release also
  includes membership, filters and predictions in `reproduction/nt_99742/`. The v9 primary and subset prediction series still differ on 1,287
  retained reads; membership alone does not explain this difference.
- The upstream training-read simulation driver and original species-selection
  procedure are not included. The label files identify the selected references,
  but do not recreate the upstream selection procedure.
- The main MetaTransformer scores now use the primary 100K arrays (87.458% and
  48.920%). The prior 87.47% matches the recovered 99,742-read MT13 output;
  48.87% is retained only in the historical 5,001,216-read validation comparison.
  Two MT13 predictions differ between the subset archive and restricted primary
  array, without an established implementation cause. The KmerFormer L29
  discrepancy was resolved by the precision/aggregation audit.
  [RESULTS](RESULTS.md#abundance-against-detection-on-one-pool) records these distinctions.
- A validation-versus-test comparison alone does not exclude checkpoint
  selection as an explanation for the remaining MetaTransformer reconstruction
  gap. This remains a manuscript interpretation issue, not a packaging fix.

## 4. Repaired reconstruction path, not a new experiment

The Kraken builder now creates genus nodes at `10000 + genus_class` and species
parents from the supplied labels. The explicit `--taxonomy species-only` mode
reconstructs the control. Regression tests check the tree, collisions and refusal
to replace a different existing taxonomy.

This is a new reconstruction of the documented hierarchy; it is not recovery of
the original historical builder. A complete index build and comparison against the
stored Kraken predictions still need the reference genomes, tools and compute.
No reported Kraken score has been changed by this code repair.

## 5. Deliberately not in this repo

- The full soil training-data bundle. Recovered soil configurations and four
  existing prediction arrays are now in [the evidence catalogue](../reproduction/README.md).
- Large checkpoints are delivered as separate model bundles. Recovered ANI-ladder
  prediction arrays and existing real-mock rescoring summaries are included in
  [the evidence catalogue](../reproduction/README.md); raw mock reads remain external.
- Third-party model implementations and their pretrained weights.
- The manuscript's LaTeX and publication figure generators.
- Private machine handoff notes, credentials and Git history in reviewer archives.

The two NT-v2 250M configurations are present under `configs/extra/` and their
scores are now reported in the manuscript. That directory also contains
exploratory configurations; existence of a config is not evidence that it ran.

## Current recovered evidence

The maintained release includes frozen catalogue records, the NT-v2 99,742-read
membership/filter records, ladder prediction arrays, soil configs/predictions and
mock summaries. Source and delivered checksums are in
[reproduction/manifest.json](../reproduction/manifest.json). These additions
recover existing evidence; they do not reconstruct an absent simulation driver.

## Before reviewer delivery

Provide a working access route for data and the Apache-2.0 weight bundles,
attach frozen input manifests/checksums, and resolve or explicitly disclose the
provenance gaps above. Run the checks on the **extracted archive**, not only on the
author's working checkout. The [Zenodo guide](../weights/ZENODO.md) records the delivery inventory.

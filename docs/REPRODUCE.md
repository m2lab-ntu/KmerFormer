# Reproducing the paper

This page covers software checks, rescoring archived predictions and commands
for the paper's experiments. MetaTransformer uses its own implementation.
Full experimental reproduction requires the external inputs listed in
[DATA.md](DATA.md) and [the asset inventory](MISSING_ASSETS.md).

## Software and evidence checks

From the repository root or an extracted source archive, in Python 3.11:

```bash
python -m pip install -e '.[review,plots]'
python scripts/reviewer_check.py --json /tmp/kmerformer-checks.json
```

The runner checks syntax, CPU tests, documentation links, a synthetic training
step and checkpoint reload, and fourteen archived prediction arrays. Its
subprocesses disable CUDA and Hugging Face downloads. Installation needs network
access unless dependencies are cached. It works without a Git checkout; Git is
needed only for snapshot-construction tests, and Bash for shell syntax checks.
The JSON records installed versions and individual outcomes.

Recompute abundance and detection metrics or verify the recovered evidence:

```bash
python scripts/review_predictions.py --full-metrics
python scripts/check_evidence.py
```

The first command checks accuracy, declined fraction, Pearson correlation,
Bray–Curtis dissimilarity, detection AUC and sensitivity at 95% specificity
against [the archived summary](assets/s2_abundance_vs_detection.json). It preserves
the scoring seed, ordered label fingerprint and prediction fingerprints.
The species-coverage-masked column requires the original FASTA/TSV and
[its scoring script](../scripts/eval/s2_abundance_vs_detection.py); genus-only
arrays cannot reconstruct that mask. These checks evaluate software and existing
predictions, without rerunning training or rebuilding external databases.

See [validation results](validation/0.2.0rc1.md),
[historical environment](history/ENVIRONMENT.md) and
[Zenodo delivery](../weights/ZENODO.md) for execution scope and release packaging.

## Experiment environment

```bash
conda env create -f environment.yml && conda activate kmerformer
pip install -e ".[baselines,plots,review]"

export KF_DATA=/path/to/kmerformer-data
export KF_OUT=/path/to/runs
export KF_VOCAB=$KF_DATA/vocab
```

## The two commands

```bash
# train (single GPU)
kmerformer-train --config configs/6mer/L29_50M.yaml

# evaluate on the primary gut closed-set test pool
kmerformer-eval \
    --config      configs/6mer/L29_50M.yaml \
    --checkpoint  "$KF_OUT/L29_50M/best.pt" \
    --test_fasta  "$KF_DATA/val_100K/reads_100K_val.fa" \
    --test_labels "$KF_DATA/val_100K/labels_100K_val.tsv" \
    --output_dir  "$KF_OUT/L29_50M/eval_closed_set" \
    --rc_tta
```

Use `--rc_tta` for the reported KmerFormer and NT-v2 scores. MetaTransformer follows
its authors' forward-only protocol. The released KmerFormer evaluator averages logits before
softmax and uses bfloat16 autocast when supported, otherwise float16. RC-TTA adds
about 0.7 points to NT-v2, at most 0.21 to the tested non-overlapping KmerFormer
6-mer depths, and 5.44–6.95 to its non-canonical hashed 13-mer arms. Exact one-layer
mean-pooling gains are +0.008 and −0.005 points at 50M and 250M. Record both
aggregation and inference precision; the training `amp_dtype` is not a guarantee
of the evaluation dtype. The recovered Nano4 NT evaluator instead averages
probabilities under float16 autocast, despite its mean-logit docstring. Its
99,742-read series remains separate from the primary 100K predictions; see
[RESULTS](RESULTS.md). Do not silently substitute a different protocol.

Top-1 lands in `<output_dir>/eval_metrics_rc_tta.json` as `micro_accuracy`.

Multi-GPU for dense-gradient arms (6-mer and hashed 13-mer):

```bash
torchrun --nproc_per_node=4 -m kmerformer.train_ddp --config configs/6mer/L29_50M.yaml
```

The exact-13-mer arms take **sparse gradients** on their 33.5M-row embedding table, and
torch's data-parallel reducer rejects sparse gradients — measured, not assumed. They
train single-GPU through `kmerformer-train`, and every rank would hold its own copy of
an 8 GiB table anyway. `train_ddp` will refuse them.

The maintained trainer uses the same explicit split strategy and manifest in
single-process and DDP launches. Historical split distinctions and effective
batch sizes remain documented in [TRAINING.md](TRAINING.md).

## Off-catalogue evaluation

Same command, different pool. Track A and Track B share the closed-set label space —
every test read comes from a genome whose genus is one of the 120 trained on — so
the trained output width remains 120. The streaming evaluator infers output
width from the checkpoint; `--num_classes 120` explicitly checks that width.

```bash
# Track A: out-of-genome, 580,000 reads, 116 genera
kmerformer-eval --config configs/6mer/L29_50M.yaml \
    --checkpoint "$KF_OUT/L29_50M/best.pt" \
    --test_fasta  "$KF_DATA/track_a/newgenome_test.fa" \
    --test_labels "$KF_DATA/track_a/newgenome_test_labels.tsv" \
    --output_dir  "$KF_OUT/L29_50M/eval_track_a" \
    --num_classes 120 --rc_tta --skip_save_logits

# Track B: one rung at a time
for rung in L1_strain L2_species L2_far; do
  kmerformer-eval --config configs/6mer/L29_50M.yaml \
      --checkpoint "$KF_OUT/L29_50M/best.pt" \
      --test_fasta  "$KF_DATA/track_b/$rung.fa" \
      --test_labels "$KF_DATA/track_b/${rung}_labels.tsv" \
      --output_dir  "$KF_OUT/L29_50M/eval_$rung" \
      --num_classes 120 --rc_tta --skip_save_logits
done
```

`--skip_save_logits` streams predictions and labels to disk and omits full-dataset
logit/probability arrays. The RC-TTA output is `predictions_rc_tta.npz`; forward-only
output is `predictions.npz`. Without the flag,
full arrays are written through disk-backed storage. Temporary disk use scales
with read count and class width; RAM use is bounded by the batch and confusion matrix.

## Claim → config

Accuracies are in [`RESULTS.md`](RESULTS.md); each config is also stamped with the
number it produced.

### Part 1 — foundation models

| Claim | Configs |
|---|---|
| NT-v2 is decent but not dominant (67.08% vs 69.52%) | [`baselines/ntv2_lora_50M`](../configs/baselines/ntv2_lora_50M.yaml) vs [`6mer/L29_50M`](../configs/6mer/L29_50M.yaml) |
| NT-v2 is the right foundation-model representative | [`baselines/ntv2_lora_5M`](../configs/baselines/ntv2_lora_5M.yaml), [`dnabert1_lora_5M`](../configs/baselines/dnabert1_lora_5M.yaml), [`dnabert2_lora_5M`](../configs/baselines/dnabert2_lora_5M.yaml) |
| Pre-training pays where labels are scarce, and is crossed at 50M | [`baselines/ntv2_lora_{500K,1M,5M,50M}`](../configs/baselines/) vs [`6mer/data_scale/`](../configs/6mer/data_scale/) |

### Part 2 — the 6-mer comparison

| Claim | Configs |
|---|---|
| Depth is worth 15.60 points | [`6mer/L{1,8,16,29}_50M`](../configs/6mer/) |
| It is *k*-mer length, not token count | [`6mer/L16_50M_stride1`](../configs/6mer/L16_50M_stride1.yaml) vs [`6mer/L16_50M`](../configs/6mer/L16_50M.yaml) |
| Pooling is not what makes it work | [`ablations/L16_5M_meanpool`](../configs/ablations/L16_5M_meanpool.yaml) vs [`6mer/data_scale/L16_5M`](../configs/6mer/data_scale/L16_5M.yaml) |
| Nor is reverse complement | [`ablations/L16_5M_no_rc_augment`](../configs/ablations/L16_5M_no_rc_augment.yaml); and drop `--rc_tta` for the test-time half |
| The 0.12-point reference scale for one configuration | [`ablations/L16_5M_rep{1,2,3}`](../configs/ablations/) plus `L16_5M` |

### Part 3 — the 13-mer comparison

| Claim | Configs |
|---|---|
| KmerFormer leads MetaTransformer by 3.69 points at matched encoder | [`13mer/exact_1L_50M_meanpool`](../configs/13mer/exact_1L_50M_meanpool.yaml) |
| …and the margin does not decompose into anything we can enumerate | [`ablations/mt_direction_{recipe,arch,both,both_pad139}`](../configs/ablations/) |
| The hashed configuration trades 6.37 points for an eightfold smaller table at matched width; canonicalization and embedding updates also differ | [`13mer/hashed_d64_16L_50M`](../configs/13mer/hashed_d64_16L_50M.yaml) vs the exact d64 16-layer arm |
| …and doubling the width recovers most of it, on 2 GiB | [`13mer/hashed_d128_16L_50M`](../configs/13mer/hashed_d128_16L_50M.yaml) |
| Depth *hurts* at *k* = 13 | [`13mer/exact_16L_50M`](../configs/13mer/exact_16L_50M.yaml) and [`hashed_d128_16L_50M`](../configs/13mer/hashed_d128_16L_50M.yaml) vs their 1-layer counterparts |
| Attention pooling is a cost at *k* = 13 | [`13mer/exact_1L_50M_attnpool`](../configs/13mer/exact_1L_50M_attnpool.yaml) |
| Both implementations converge at 250M reads | [`13mer/exact_1L_250M_meanpool`](../configs/13mer/exact_1L_250M_meanpool.yaml) |

### Off the catalogue

The two pools are *generated*, from RefSeq assemblies with ART, by the pipelines in
[`scripts/track_a/`](../scripts/track_a/) (6 steps) and
[`scripts/track_b/`](../scripts/track_b/) (14 steps). Start at
[`scripts/track_a/README.md`](../scripts/track_a/README.md): it lists the external tools
(`art_illumina`, NCBI `datasets`, `skani`, `kraken2`) and the extra environment variables,
and explains which track answers which question. Both use identical ART settings — HiSeq
2500, paired-end, 150 bp, insert 400±50, seed 42 — so they stay comparable.

Once the pools exist, nothing needs training: every claim here re-evaluates arms trained
above.

```bash
# the 6-mer depth series on Track A
python scripts/eval/score_track_a_depth.py

# the ANI-ladder retention table, restricted to the 14 genera present at every rung
python scripts/eval/track_b_retention.py --track_b_out "$KF_OUT/track_b" --markdown
```

`track_b_retention.py` regenerates the table in
[`RESULTS.md §4`](RESULTS.md#the-ani-ladder-track-b) from the stored predictions and
prints the genus ids it restricted to, so the restriction is auditable. It also reports
whether the two tokenizer-family bands still fail to overlap, which is the claim the
table exists to support.

### Non-neural baselines

The naive Bayes arms are what make the ANI-ladder claim structural rather than
architectural: each parameter-free baseline tracks its own *k* family, so *k* sets the
slope and the network only sets the level.

```bash
python scripts/baselines/multinomial_nb.py --help    # 13-mer NB, 74.9% closed set
python scripts/baselines/6mer_baselines.py --help     # 6-mer NB, 25.9%
python scripts/baselines/unique_key_lookup.py --help  # is a long k just a lookup table?
python scripts/baselines/composition_vote.py --help
```

Kraken 2 needs a matched 1,535-genome index, and building it correctly is the whole
difficulty:

```bash
python scripts/baselines/build_kraken2_db_1535.py --help
python scripts/eval/eval_kraken1535_vs_neural.py --help
```

The index **must** carry genus nodes between root and species. Without them a read
ambiguous between two species of one genus has nowhere to stop, climbs to root and is
scored unclassified — costing Kraken 2 20.01% of its reads and 19.09 points (79.97%
against 99.06%). A custom database built the obvious way produces the species-only tree
and understates Kraken 2 badly. See [`RESULTS.md §5`](RESULTS.md#5-kraken-2-and-where-a-learned-classifier-is-the-wrong-tool).

### Sample-level abundance

```bash
python scripts/eval/evaluate_sample.py \
    --predictions "$KF_OUT/L29_50M/eval_closed_set/predictions_rc_tta.npz" \
    --out_dir     "$KF_OUT/L29_50M/eval_sample" \
    --reads_per_sample 1000 --n_partition_samples 100 \
    --n_sparse_samples 200 --genera_present 50
```

Across fourteen arms whose read-level Top-1 spans
48.92% to 99.16%, genus abundance correlation moves only from *r* = 0.9778 to 0.9999.
That compression is the reason everything in this repo is ranked by read-level
accuracy. The full table, and the script that regenerates it from predictions, are in
[`RESULTS.md`](RESULTS.md#abundance-against-detection-on-one-pool).

### The abundance-versus-detection table

Regenerates every arm of
[`RESULTS.md`](RESULTS.md#abundance-against-detection-on-one-pool) from stored
predictions, so it needs no checkpoints and no GPU — the arrays are in
`docs/assets/`:

```bash
python scripts/review_predictions.py --full-metrics
```

This verifies the archived pool's label fingerprint and refuses any arm whose
stored labels are not that pool's in that pool's order. That
check is the only one that works here: the two pools are both 100,000 reads over the
same 120 genera and their label vectors agree on 8.00% of positions, so shape, length,
class count and even accuracy all pass on the wrong one. Coverage-masked accuracy
is excluded from this offline check. The original `s2_abundance_vs_detection.py`
also checks FASTA identity and coverage, but needs `--pool_fasta`, `--pool_labels`
and one `--arm NAME=NPZ` per model.

For a Kraken 2 arm, convert its read-level output first — the same script handles
either index and prints the rank histogram so the build is visible:

```bash
python scripts/eval/kraken2_genus_calls.py \
    --kraken_out  "$KF_OUT/kraken2/reads_100K.kraken2.out" \
    --db          "$KRAKEN2_DB" \
    --labels_tsv  "$KF_DATA/val_100K/labels_100K_val.tsv" \
    --out_npz     "$KF_OUT/kraken2/genus_calls.npz"
```

### Throughput and memory (main-text Table 2)

```bash
python scripts/bench/bench_arms.py --configs configs/bench/*.yaml
```

One GPU, batch 128, 150 bp synthetic reads, warm-up excluded. The two exact-13-mer arms
need a 48 GiB card: they peak at 24.7 GiB to train at one layer and 25.8 GiB at sixteen,
and fail outright on a 24 GiB card. `scripts/bench/embed_mem_probe.py` measures the
embedding table's contribution alone, which is where the separation lives.

## Cost

The reported runs used one RTX A6000 (48 GiB) and one RTX 4090.

| Arm | Wall time | Notes |
|---|---|---|
| 6-mer, 1–16 layers, 5M reads | hours | fine on a 4090 |
| 6-mer, 29 layers, 50M reads | ~1d 8h | 15 epochs, single GPU |
| exact 13-mer, 1 layer, 50M reads | ~1d | needs ≥32 GiB of VRAM |
| exact 13-mer, 1 layer, 250M reads | days | |
| NT-v2 500M + LoRA, 50M reads | ~33h/epoch | the slow one; 500M parameters is heavy |

Two practical notes from the runs. Set `num_workers: 0` for the large-vocabulary arms:
resident memory per worker is ~18 GiB, and four workers exhausted 62 GiB of host RAM
mid-epoch. And `periodic_save_interval` in the 13-mer configs exists so a run that dies
at hour 20 resumes from `last.pt` rather than from nothing —
`kmerformer-train --config <config.yaml> --resume "$KF_OUT/<run>/last.pt"` restores the SparseAdam moments and
the schedule along with the weights.

## Checking the repo against the manuscript

The banners and `tests/test_reported_numbers.py` pin numbers against each other. They
cannot see the manuscript, which is not in this repository, so they cannot detect the
repo drifting away from it. That check is separate and has to be run deliberately,
pointed at a manuscript checkout:

```bash
python scripts/check_against_manuscript.py /path/to/paper
```

It confirms every accuracy stamped on a config, and every difference the test suite
asserts, still appears in the manuscript. Two things it deliberately does not claim:
a value that moved from one arm to another would still be found, and a number absent
from the output may have been revised, moved to the supplementary, or simply be
somewhere the check does not look. Read the sentence before changing anything.

## Checking a run selected the epoch it says it did

A run records its selected epoch twice, in `training_history.csv` and in each eval
JSON's `checkpoint_epoch`, written by different scripts with nothing tying them
together. For nine runs those two are the only record (see
[`MISSING_ASSETS.md`](MISSING_ASSETS.md)), and the manuscript's repeat-spread figure
depends on the difference between two runs' selections, so a drift between them would
be invisible downstream:

```bash
python scripts/check_epoch_selection.py "$KF_OUT"/*/
```

## Verifying a released checkpoint

```bash
python scripts/weights/inspect_checkpoint.py weights/*.pt
```

This reads the config, epoch and validation accuracy out of the checkpoint's pickle
without materialising a single tensor, so it costs nothing on an 8 GiB file and it
tells you which arm you actually have. Confirm the numbers against
[`weights/README.md`](../weights/README.md) before trusting a download.

## What is not reproducible from this repo

**MetaTransformer.** Its rows are its authors' code, run by us on the same training
pool, the same test set and the same GPU. `scripts/eval/extract_mt_predictions.py` and
`scripts/bench/bench_mt.py` are the harnesses that scored and timed it from inside a
MetaTransformer checkout; the model itself is not vendored here.

**The upstream species selection.** Everything from the reference genomes onward is
here: `scripts/data/catalogue_manifest.py` recovers the species → genome mapping from the
labels, `scripts/data/subsample_balanced.py` builds the read pools, and
`scripts/baselines/build_kraken2_db_1535.py` builds the matched index. What is not here
is the step that picked 1,535 of the 2,505 candidate assemblies and assigned their class
ids — that came from the original MetaTransformer data preparation. It does not block
reproduction, since those ids are recorded in the label files. See
[`DATA.md`](DATA.md#there-is-no-build-the-catalogue-script-because-the-labels-are-the-catalogue).

**Soil.** The RefSoil arms are reported in the manuscript's cross-environment check and
run through the same code, but their configs point at a catalogue not bundled here.

**The manuscript's own figures.** What is here draws the Track B ladder
([`scripts/track_b/tb_11_plots.py`](../scripts/track_b/tb_11_plots.py)), training curves
([`scripts/eval/plot_training_curves.py`](../scripts/eval/plot_training_curves.py)) and
the confusion matrices `kmerformer-eval` writes. The scripts that render the figures the
manuscript actually includes — three panels from one generator, two more, and the
generated ladder table — live in the manuscript's own repository alongside the LaTeX,
not here. They read absolute `/nas2` paths, so mirroring them would publish something a
reader could not run, and keeping a second copy of a generator in a second tree is how
this project has already lost time twice. If you need them, ask; they are not secret,
only elsewhere.

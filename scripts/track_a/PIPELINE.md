# Local Real-World / Generalization Evaluation — Agent Handoff Guide

**Purpose.** Evaluate the trained genus classifiers on data they were **NOT trained on**, to
test real-world generalization beyond the closed-set, same-genome memorization that all
current numbers reflect. Runs on a **single RTX 4090 (24 GB)** — this is **inference + CPU
data-prep only; NO training, NO HPC needed**.

> An agent picking this up: read this whole file, then `THESIS_NUMBERS.md` (in
> `benchmark_results/`, or ask the user) for all established closed-set numbers and context.

---

## 0. Context (why this matters)
- Task: classify 150 bp metagenomic reads into **120 human-gut genera** (closed-set; 1,535
  reference genomes, 1 genome/species).
- **All current results are on SAME-genome simulated reads** → closed-set *near-memorization*.
  Genus Top-1 on the clean (train-disjoint) pool: **MT 13-mer 98.7%**, NT-v2 67%, MT 6-mer 45–51%.
- **Open question for publication:** do these transfer to (A) *new genomes* of the same genera,
  and (B) *real / mock communities*? **Expectation: MT 13-mer's 98.7% will drop sharply** on new
  genomes (it memorized the 1,535 genomes' 13-mer content) — that drop *is* the result.

## 1. Hardware / VRAM (measured)
| model | inference VRAM peak | fits 4090 24 GB |
|---|---|---|
| MT 13-mer (2.1 B embedding) | ~9.1 GB | ✅ |
| NT-v2 + LoRA (498 M) | ~6.6 GB | ✅ |
| MT 6-mer | <1.3 GB | ✅ |

## 2. Environment
Conda env (see `environment.yml`). Key pins: **python 3.11, torch 2.5.1 (cu121),
transformers 4.46.3, omegaconf 2.3.0, peft 0.19.1, tokenizers 0.20.3, pandas, numpy, scipy,
einops, accelerate**. Plus **ART** read simulator: `conda install -c bioconda art` (provides
`art_illumina`). For genome download: `conda install -c conda-forge -c bioconda ncbi-datasets-cli`
(or `ncbi-genome-download`).
- Inference does NOT need the MetaTransformer trainer deps (tensorboard etc.) — only the model.

## 3. Files to transfer from Nano4 (NOT in this repo — too large)
Put under `local_realworld_eval/assets/` (or anywhere; pass explicit paths to the scripts).

| what | Nano4 source path | size | note |
|---|---|---|---|
| MT 13-mer **250M** weights | `mt_250M/experiments/mt_13mer_s1_genus_250M_147918/checkpoints/classification_transformer_ckpt_best_inference.pt` | 8.6 GB | **stripped, inference-only** |
| MT 13-mer **250M** config | `mt_250M/experiments/mt_13mer_s1_genus_250M_147918/config.yaml` | tiny | needed by extract |
| MT 13-mer **50M** weights+config | `mt_models/mt_13mer_stride1_genus_895686/{checkpoints/...best_inference.pt, config.yaml}` | 8.6 GB | |
| NT-v2 genus checkpoints | `checkpoints/nt_token_genus_v9_50M_best.pt` (+ v14/v15/gbal/sbal/ov6mer `/best.pt`) | ~2 GB each | pick which to eval |
| 13-mer vocab | `MetaTransformer/vocab_file/vocab_13mer.txt` | 470 MB | |
| 6-mer vocab | `gfm-classifier/small_predictions/vocab_6mer.txt` | tiny | |
| MetaTransformer source | `MetaTransformer/src/` | — | needed for MT inference (`PYTHONPATH`) |
| genus label map | any training `labels.tsv` (e.g. `data/balanced_250M/labels_250M.tsv`) | — | for the 120 genus_class↔name mapping |

**MT checkpoint rename:** `extract_mt_predictions.py` hard-codes the filename
`classification_transformer_ckpt_best.pt`. So in each MT exp dir, rename/symlink the
`*_best_inference.pt` → `classification_transformer_ckpt_best.pt`.

NT-v2 configs are in this repo (`configs/nt_token_genus_*.yaml`). Inference scripts are in
`scripts/` (`run_genus_rctta.py`, `extract_mt_predictions.py`, `evaluate_sample.py`).

---

## 4. Track A — New-genome generalization (DO THIS FIRST; most feasible & informative)
Goal: reads from **different genomes of the same 120 genera** → tests generalization, not memory.

The completed results and their limitations are documented in
[`docs/RESULTS.md`](../../docs/RESULTS.md#track-a-caveat).

1. **Pick genera & download new genomes.** From the genus label map, for each of the 120 genera
   download 1–3 genomes from NCBI/GTDB that are **NOT among the 1,535 training genomes**
   (different accessions/strains). The pipeline passes
   `track_a_training_gcf_accessions.txt` to the downloader and fails if an excluded
   accession is already present. Do not bypass this audit.
2. **Build labeled FASTA.** Assign each genome its training **genus_class** (map by genus name →
   the index in the label map). Header format the pipeline expects:
   `>lbl|<species_class>|<species_name>|<genus_class>|<genus_name>-<readid>` (species_class can be
   a placeholder; only `genus_class` at field 3 is used for genus eval, `class_indices=3`).
3. **Simulate reads** to match training distribution:
   `art_illumina -ss HS25 -i <genome.fa> -p -l 150 -m 400 -s 50 -f <coverage> -o <out>` (paired-end,
   150 bp, HiSeq 2500). Concatenate per-genome FASTAs → single-line `newgenome_test.fa` with the
   headers above. **Single-line sequences** (avoid the 60-col-wrap reader-truncation bug).
4. **Run inference** (Section 6) → **5. metrics** (Section 7). Compare Top-1 on new genomes vs the
   closed-set 98.7%/67%/45%.

## 4b. Track B — Real mock community / CAMI (harder; the true "real world")
- Use a known-composition mock (Zymo/ATCC) or **CAMI2** gut challenge data (cited in thesis).
- No per-read ground truth → evaluate **sample-level abundance** (predicted genus fractions vs
  known composition) + concordance with Kraken2/Bracken. Handle **out-of-set taxa** explicitly
  (taxa not in the 120 genera → forced misclassification; report the in-set fraction).
- Requires a genus-name crosswalk between the mock/CAMI taxonomy and the 120 genera.

---

## 6. Inference commands
**NT-v2 (non-overlap 6-mer)** — saves `rctta.npz` (`preds`,`labels`):
```
python scripts/run_genus_rctta.py --config configs/nt_token_genus_v9_50M.yaml \
  --checkpoint assets/nt_token_genus_v9_50M_best.pt \
  --test_fasta <newgenome_test.fa> --test_labels <labels.tsv> --train_labels <labels.tsv> \
  --out_dir out/v9 --batch_size 256
```
**NT-v2 overlap-6mer** — identical but use `configs/nt_token_genus_ov6mer_17M.yaml` (has
`kmer_preprocess: {k:6, stride:1}`).

**MT (13-mer or 6-mer)** — saves npz (`preds`,`probs`,`labels`); run from MetaTransformer/src:
```
PYTHONPATH=<MetaTransformer>/src python scripts/extract_mt_predictions.py \
  --exp_dir <mt_exp_dir_with_config.yaml_and_checkpoints/> \
  --val_dir <dir_containing_only newgenome_test.fa> --vocab assets/vocab_13mer.txt \
  --out out/mt13_250M/preds.npz --class_indices 3 --batch_size 1024
```
(`--val_dir` globs `*.fa`; keep only the test fasta in it.)

## 7. Metrics
- **Per-read Top-1:** `python -c "import numpy as np;d=np.load('out/<m>/preds.npz');print((d['preds']==d['labels']).mean())"` (NT npz uses keys `preds`/`labels` too).
- **Sample-level abundance:** `python scripts/evaluate_sample.py --predictions out/<m>/preds.npz
  --out_dir out/<m>/sample --exp_name <m> --n_partition_samples 50 --reads_per_sample 10000`
  → Pearson r, Bray-Curtis, `abundance_scatter.png`.

## 8. Gotchas
- **Genus only** is clean (`class_indices=3`, global 0–119 mapping). **Species (class_indices=1)
  label indices do NOT align across models/datasets** → species Top-1 will read ~0% (a mapping
  artefact, not real). Stick to genus, or build a per-model species crosswalk first.
- Match training read params (150 bp, HS25) so the comparison is fair.
- Use the **stripped** MT checkpoints (8.6 GB) renamed to `classification_transformer_ckpt_best.pt`.

## 9. What to report (publication framing)
Top-1 (and sample-level r) on NEW genomes for each setting, side-by-side with the closed-set
numbers. The **drop from closed-set → new-genome** is the headline: it quantifies how much each
model *memorized* vs *generalized*, and whether the tokenization advantage (13-mer) survives on
unseen genomes. Treat this as a robustness analysis rather than a clean tokenization ablation:
the evaluated models also differ in model size and training volume. Do not report the
sample-level abundance correlation from the uniformly sampled Track A pool.

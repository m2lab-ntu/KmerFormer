# Building the evaluation pools

The two off-catalogue pools are *generated*, not downloaded — from RefSeq assemblies,
with ART. This directory and [`../track_b/`](../track_b/) hold the pipelines that do it,
end to end: pick candidate genomes, fetch them, simulate reads, run inference, score.

Which one you want depends on the question:

| | Track A (here) | [Track B](../track_b/) |
|---|---|---|
| Guarantees | unseen *assemblies* only | ANI-stratified rungs |
| Answers | "how far" on average | how accuracy decays *with* distance |
| Steps | 6 | 14 |
| In the manuscript | supplementary | main text |

Track A came first and Track B supersedes it for the distance question; the manuscript
reports both because they are different genome sets. Their ART settings are deliberately
identical — HiSeq 2500, paired-end, 150 bp, insert 400±50, seed 42 — so the two are
directly comparable.

## Prerequisites

Neither pipeline is pure Python — four command-line tools do the real work, and
they are the part that is easy to miss. There is an environment for exactly this:

```bash
conda env create -f environment-pools.yml && conda activate kmerformer-pools
```

It is separate from the root `environment.yml`, which is for training and knows
nothing about genomics tooling. Read the notes at the bottom of the file before
trusting it: the `peft` pin differs from the training environment on purpose, and
the four bioconda tools are transcribed rather than verified to solve.

What the pipelines invoke, if you would rather install them your own way — they
only need to be on `PATH`:

| Tool | Used by | For |
|---|---|---|
| `art_illumina` | both, at the simulate step | read simulation (ART, Huang et al. 2012) |
| `datasets` | both, at the download step | NCBI datasets CLI, fetching assemblies |
| `skani` | Track B | ANI against every reference genome |
| `kraken2`, `bracken` | Track B | the non-neural baseline |

Environment, on top of the three from [`../../docs/DATA.md`](../../docs/DATA.md):

```bash
export KF_TRACK_A="$KF_OUT/track_a"     # working dir: genomes/, reads/, test_data/, out/
export KF_TRACK_B="$KF_OUT/track_b"
export KF_GENOMES="$KF_DATA/reference_genomes"   # the 2,505 training assemblies
export KF_PYTHON=python                 # interpreter for this repo's env
export MT_PYTHON=python                 # MetaTransformer's env, if scoring that baseline
export KF_TRAINING_ACCESSIONS="$KF_DATA/training_accessions.txt"
export KF_TRACK_A_EXCLUSIONS="$KF_DATA/track_a_exclusions.txt"
export MT6_EXP=/path/to/metatransformer-6mer-experiment
export MT6_VOCAB=/path/to/metatransformer/vocab_6mer.txt
```

`KF_GENOMES` matters for Track B only, and matters a lot: `tb_03` measures every
candidate's ANI *against the training catalogue*, so the rungs are defined relative to
what the models actually saw.

## Track A, in order

```bash
cd <this repository>
export KF_TRACK_A="$KF_OUT/track_a"

python scripts/track_a/track_a_01_download_genomes.py --help   # one assembly per genus
python scripts/track_a/track_a_02_simulate_reads.py   --help   # ART, 150 bp, seed 42
python scripts/track_a/track_a_03_build_test_fasta.py --help   # balance and label
bash   scripts/track_a/track_a_04_run_inference.sh             # every arm
bash   scripts/track_a/track_a_05_metrics.sh
python scripts/track_a/track_a_06_clean_metrics.py    --help   # the strict-pool variant
```

`run_track_a.sh` chains 01–06 and requires the two accession lists above, which are
external data assets rather than guessed from filenames. Step 06 scores after
excluding the three specified training-overlapping accessions (595K → 580K reads).
The additional six near-identical genomes are a different filter; their removal
costs the 13-mer arms 1.13–1.75 points. See
[`../../docs/RESULTS.md`](../../docs/RESULTS.md#track-a-caveat).

The shell inference drivers cover historical baseline arms, not every KmerFormer
configuration. Use the direct evaluator commands in
[`../../docs/REPRODUCE.md`](../../docs/REPRODUCE.md#off-catalogue-evaluation) for
KmerFormer and keep the 120-class output space. Fresh NCBI candidate selection may
not reproduce the original assemblies or pool membership; frozen manifests are
required for exact reproduction. The external-data/GPU pipelines are not exercised
by the offline reviewer checks.

## What this does not rebuild

**The training pool.** These pipelines build the *evaluation* pools. The 258.67M-read
gut training pool was simulated upstream, in the data preparation this project inherited,
and that step is not here — the same boundary as the species-selection step described in
[`../../docs/DATA.md`](../../docs/DATA.md#there-is-no-build-the-catalogue-script-because-the-labels-are-the-catalogue).
What *is* here is everything downstream: `scripts/data/subsample_balanced.py` builds the
balanced pools from it, and these two pipelines build everything the models are tested on.

The ART parameters are recorded in `track_a_02_simulate_reads.py` and are the ones the
paper's Methods describe, so a from-scratch regeneration of the training pool has the
settings it needs even though the original driver script is absent.

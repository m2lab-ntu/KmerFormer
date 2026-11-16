# Evaluation protocols and results

This guide describes the manuscript's evaluation protocols and results, checked on
2026-11-09. The original submission tag preserves the earlier protocols.
[The added evidence bundle](../reproduction/evaluation/README.md) distinguishes
recomputable primary results from summary-only uncertainty and strand controls.

## Output policies and fair comparison

Ordinary read Top-1 comparisons retain the neural model's argmax for every read.
The uncertainty analysis adds temperature scaling and validation-fitted cutoffs
to both neural models as diagnostics. The main common-read comparison instead
uses the original MetaTransformer raw maximum softmax probability (MSP) >= 0.5.
This original rule is used for its thresholded precision/recall and community
calls; unthresholded accuracy is a separate metric.

KmerFormer 250M uses T = 0.7833955, fitted by validation NLL, then a cutoff of
0.798195943 to retain the most confident 99.065% of its own validation reads.
The coverage target comes from Kraken2's primary-pool coverage. The fitted
cutoff gives 99.081% coverage on the primary pool; validation and test coverage
need not match. Temperature is not fitted to a coverage target.

Reads declined by any method are removed. The main policy keeps MetaTransformer's
original output rule, a choice made after both versions had been run; it compares
complete pipelines on a selected population.
All numbers below use the identical 98,272 retained reads, with detection
communities redrawn from that pool. Declines count reads from the original 100K.

| Method | Declined | Wrong among retained | Retained accuracy | Detection Sens@95% | Sens@99% |
|---|---:|---:|---:|---:|---:|
| KmerFormer 250M, TS-MSP | 919 | 148 | 99.849% | 90.50% | 87.43% |
| MetaTransformer 250M, native MSP >= 0.5 | 293 | 604 | 99.385% | 90.02% | 77.51% |
| Kraken2, genus-aware, native decline | 935 | 2 | 99.998% | 90.64% | 90.25% |

KF minus MT accuracy is +0.464 percentage points (paired read-bootstrap 95%
interval +0.413 to +0.511). KF minus Kraken2 is -0.149 points (interval -0.173
to -0.125). KF leads MT by 9.92 points in sensitivity at exactly 99% specificity,
while Kraken2 leads KF by 2.82 points. These are conditional primary-pool results.
The smaller gaps are measured on a different read pool with redrawn
communities. Giving MT the additional
TS-MSP policy retains 97,950 reads and gives KF a +0.34-point accuracy margin.

## Detection operating points

The earlier repository table reports the highest sensitivity at a realizable
threshold whose specificity is at least 95%. The revised paper also interpolates
at exactly 95% and 99% specificity. ROC points still group tied scores; the
interpolation can represent randomization between adjacent thresholds. The
interpolated operating points were added in the revised analysis to compare
methods at equal specificity; the earlier values are retained.

On all primary reads, the interpolated Sens@95% values are 91.44% for Kraken2,
88.60% for KF 250M and 83.07% for MT 250M. At 99% specificity, Kraken2 and KF
reach 91.08% and 71.01%. Thus the earlier statement that the 250M neural methods
are tied refers to their realizable-threshold values at different specificities,
77.49% at 98.42% and 77.47% at 97.56%. Read accuracy and abundance are unchanged.

## Kraken2 database correction

The primary results already used a genus-aware index. The earlier ANI-ladder,
L3 and log-normal records used a species-only tree. The October 30 rerun uses
genus-aware Kraken2 throughout, with confidence 0 and minimum-hit-groups 2.
A primary rerun matched every stored primary prediction. The old ladder arrays
are retained as historical evidence and are not the current manuscript's Kraken2
baseline. L3 contains 123 evaluated genomes from 130 candidates.

On the fourteen shared far-rung genera, Kraken2's correct/wrong/declined shares
are 7.57%/3.92%/88.51%. On all far-rung reads its coverage is 9.56%, compared with
8.00% on L3. These scopes must not be mixed. For the log-normal community, raw
Kraken2 has Pearson r = 0.814 and Bray–Curtis = 0.351; with Bracken these are
0.818 and 0.204. Bracken now leads both abundance criteria. Real-mock runs use a
separate database and are outside this rerun.

## Uncertainty and strand interpretation

Temperature scaling improves calibration; MC-dropout mutual information lowers
AURC for the tested exact 13-mer models, with the opposite direction for the
tested 6-mer model. Last-layer Laplace changes little. Neither strong
primary-versus-shift separation nor low error among selected answers establishes
novel-genus recognition: far known genera and novel genera remain weakly separated.

Small gains from RC-TTA do not imply strand invariance. Canonical k-mers share
token identities across strands, but token order and positional encoding still
change. The revised controls test prediction disagreement, architectural
invariance and a consistency-loss-free control. The training gain attributed
initially to consistency training is also obtained by the control trained on
both strands. These control summaries are in the added evidence bundle; they
are not new defaults for the public predictor.

## Historical hashed-model data loading

Recovered Taiwania-2 launch scripts for the one-layer d128, sixteen-layer d64/d128
and extended d128 hashed runs invoke `train_ddp.py`. That archived trainer always
constructs `LazyFASTADataset`, whose split uses `default_rng(42).permutation` and
a 10% validation fraction. A missing `data.lazy` key in those historical configs
did not select the eager path. The current four hashed configs now explicitly
set `lazy: true` and `split_strategy: random` to retain that recorded behavior.
These hashed runs used the lazy/random path; split type alone does not
distinguish exact from hashed 13-mers.
This audit establishes the recovered script path, not an independently hashed
train/validation membership manifest for every historical checkpoint.

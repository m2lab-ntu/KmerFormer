# Results

This page distinguishes manuscript/config scores from the later bundled-array
audit (see [citation](../README.md#citation)). Config links identify the reported
configurations; the abundance/detection section is recomputed from archived predictions.
See [the reproduction guide](REPRODUCE.md) for executable checks and input requirements.

**Revision scope:** this page retains the original submission comparisons and
realizable-threshold detection table. [Evaluation protocols and results](EVALUATION.md)
report the interpolated operating points, native-MT common-read analysis and
genus-aware Kraken2 reruns used by the revised manuscript. Do not combine the
old species-only ladder results with the revised primary comparison.

**Read the two accuracy columns together.** They rank the models differently. The closed-set column measures
reads from genomes the reference catalogue contains; Track A measures reads from
genomes it does not. The ordering inverts between them.

Conventions:

- **Closed set** — 100,000 held-out reads from the same 1,535 gut reference genomes,
  genus Top-1, 120 classes. KmerFormer and NT-v2 use RC-TTA; MetaTransformer is
  evaluated forward-only under its authors' protocol.
- **Track A** — 580,000 reads from 116 genera whose source assemblies are absent from
  the reference; 5,000 reads per genus, so micro and balanced accuracy coincide.
- All configurations train on the same 50M-read balanced gut pool unless the row says otherwise.
  Each takes its own 90/10 train/validation split of it, so "matched on training data"
  means one source pool and one test set, not identical training indices.
- Run-to-run standard deviation is **0.12 points** across four runs of `L16_5M`
  (original plus three repeats). This is a reference scale for that configuration only.

---

## 1. Part 1 — foundation models

| Model | Params | Closed set | Track A | Config |
|---|---:|---:|---:|---|
| NT-v2 500M + LoRA, 6-mer | 500M | 67.08% | 25.81% | [`baselines/ntv2_lora_50M.yaml`](../configs/baselines/ntv2_lora_50M.yaml) |
| KmerFormer 6-mer, 29 layers | 6.45M | **69.52%** | **28.90%** | [`6mer/L29_50M.yaml`](../configs/6mer/L29_50M.yaml) |
| KmerFormer 6-mer, 16 layers | 3.88M | 67.94% | 27.46% | [`6mer/L16_50M.yaml`](../configs/6mer/L16_50M.yaml) |

A 6.45M-parameter model trained from scratch leads the 500M pre-trained model by
2.44 points with 77× fewer stored parameters, and by more off the catalogue. LoRA adapts
5.54M NT-v2 parameters, against all 6.45M in KmerFormer. The representation is
behind by the same margin as the classifier: NT-v2's pooled embeddings put a read's
genus on 20.6% of its twenty nearest neighbours against the 29-layer model's 34.6%, so
the gap lies in the representation as well as the classifier head.

### Backbone screen (5M reads)

| Backbone | Tokenizer | Closed set | Config |
|---|---|---:|---|
| NT-v2 | non-overlapping 6-mer | **63.05%** | [`baselines/ntv2_lora_5M.yaml`](../configs/baselines/ntv2_lora_5M.yaml) |
| DNABERT-1 | overlapping 6-mer | 61.78% | [`baselines/dnabert1_lora_5M.yaml`](../configs/baselines/dnabert1_lora_5M.yaml) |
| DNABERT-2 | byte-pair encoding | 58.88% | [`baselines/dnabert2_lora_5M.yaml`](../configs/baselines/dnabert2_lora_5M.yaml) |

At the matched 5M budget, NT-v2 leads the tested DNABERT adaptation pipelines.
This screen supports selecting NT-v2 for the subsequent comparisons.

### The low-data regime

The comparison is **depth-matched**: the 29-layer configuration has NT-v2's depth.

| Labelled reads | NT-v2 + LoRA | KmerFormer 6-mer **L29** | Advantage | KmerFormer L16 | vs L16 |
|---|---:|---:|---:|---:|---:|
| 0.5M | 54.72% | 41.03% | **+13.7** | 41.35% | +13.4 |
| 1M | 57.34% | 44.13% | **+13.2** | 44.29% | +13.1 |
| 5M | 63.05% | 54.19% | **+8.9** | 54.11% | +8.9 |
| 50M | 67.08% | 69.52% | **−2.4** | 67.94% | −0.9 |

Configs: [`baselines/ntv2_lora_{500K,1M,5M,50M}`](../configs/baselines/) ·
[`6mer/data_scale/`](../configs/6mer/data_scale/) ·
[`6mer/L29_50M`](../configs/6mer/L29_50M.yaml).

The advantage shrinks monotonically and changes sign between 5M and 50M. The
crossing recurs on a second soil catalogue (RefSoil, 309 genera), also at matched depth: NT-v2 leads
by **+4.3** at 5M (31.98% vs 27.68%) and is crossed by **+6.5** at 50M (46.35% vs
39.84%) — a wider margin than the 2.4 it takes on gut. The pre-trained line is nearly
flat over these two budgets: a 10× increase in reads adds 4.0 points.
The advantage belongs to the tested pipelines: initialization, width and LoRA
adaptation are not isolated by matching depth. The recovered soil NT configs also
differ from gut in RC augmentation, validation split, batch size and schedule
(Supplementary S10). The soil comparison tests whether the ordering recurs across
the two study settings.

**Soil's depth differences differ from gut's, so the soil cell was measured directly.** From 16 layers to 29, soil gains **+0.26** and **+2.67**
points at the two budgets where gut gains +0.08 and +1.58. Carrying gut's delta across
would have put the 50M soil cell at 45.26% against the measured 46.35%. Depth pays
more on the harder catalogue at both budgets.

> [!NOTE]
> The soil configurations are not in this repository
> ([`MISSING_ASSETS.md`](MISSING_ASSETS.md#5-deliberately-not-in-this-repo)), so the
> soil numbers have no config banner or test coverage here. RC-TTA moves the 50M soil
> configuration by 0.04 points (46.31% forward, 46.35% RC-TTA), consistent with the
> small gains of the tested non-overlapping 6-mer configurations.

**Matching depth widens the pre-trained lead.** Sixteen
layers to twenty-nine is worth −0.32, −0.16, +0.08 and +1.58 points as the budget
grows: at the two smallest budgets NT-v2's own depth makes the from-scratch model
slightly *worse*, and only at 50M does depth add a material gain. Matching depth
therefore enlarges the 13-point lead at 0.5M and 1M.

> [!NOTE]
> An 8-layer configuration trained at 5M reaches 53.62%
> ([`data_scale/L8_5M.yaml`](../configs/6mer/data_scale/L8_5M.yaml)). The manuscript
> does not report it, so its config banner is marked `[not in the manuscript]`, a form
> recognised by `scripts/check_against_manuscript.py` and the test suite.

---

## 2. Part 2 — depth at a shared 6-mer tokenizer

KmerFormer and NT-v2 share the non-overlapping tokenizer, source training pool,
label space and test set, but differ in architecture and LoRA versus full training.
MetaTransformer is a reference configuration with its own stride-1 tokenizer and protocol.

| Configuration | Params | Closed set | Track A | Config |
|---|---:|---:|---:|---|
| KmerFormer, 29 layers | 6.45M | **69.52%** | **28.90%** | [`6mer/L29_50M.yaml`](../configs/6mer/L29_50M.yaml) |
| KmerFormer, 16 layers | 3.88M | 67.94% | 27.46% | [`6mer/L16_50M.yaml`](../configs/6mer/L16_50M.yaml) |
| KmerFormer, 8 layers | 2.29M | 65.44% | 25.17% | [`6mer/L8_50M.yaml`](../configs/6mer/L8_50M.yaml) |
| KmerFormer, 1 layer | 0.90M | 53.92% | 15.22% | [`6mer/L1_50M.yaml`](../configs/6mer/L1_50M.yaml) |
| NT-v2 500M + LoRA | 500M | 67.08% | 25.81% | [`baselines/ntv2_lora_50M.yaml`](../configs/baselines/ntv2_lora_50M.yaml) |
| MetaTransformer 6-mer stride-1, 1 layer | 0.5M | 48.92% | — | (its authors' code) |

Depth adds 15.60 points on the closed set, with diminishing positive returns, and
continues to add accuracy off the catalogue after the closed-set curve flattens (15.22 → 25.17 → 27.46 → 28.90). The 20.6-point margin over
MetaTransformer's single-layer design spans a tokenizer difference too; at one layer
each the gap is 5.00 points.

### Four controls

| Control | Result | Config |
|---|---|---|
| Stride-1 instead of non-overlapping 6-mer | 62.75% vs 67.94% — **5.19 points worse** at 4× the training memory (1,799 vs 453 MiB) for 146 tokens instead of 25 | [`6mer/L16_50M_stride1.yaml`](../configs/6mer/L16_50M_stride1.yaml) |
| Mean instead of attention pooling (5M) | 54.29% vs 54.11% — comparable to the 0.12 reference scale | [`ablations/L16_5M_meanpool.yaml`](../configs/ablations/L16_5M_meanpool.yaml) |
| No RC augmentation during training (5M) | moves accuracy by 0.13 — on the reference scale | [`ablations/L16_5M_no_rc_augment.yaml`](../configs/ablations/L16_5M_no_rc_augment.yaml) |
| Run-to-run spread (5M, four repeats) | s.d. **0.12** | [`ablations/L16_5M_rep{1,2,3}.yaml`](../configs/ablations/) |

RC test-time augmentation is worth about +0.7 points to NT-v2 and at most +0.21
to the tested non-overlapping KmerFormer 6-mer depths. It requires inference on
both strands; the non-canonical hashed 13-mer configurations instead gain 5.44–6.95 points.

---

## 3. Part 3 — exact and hashed 13-mer vocabularies

A full overlapping 13-mer table is 4^13 = 67.1M keys; the observed canonical
vocabulary is 33,545,099, which at `d_model=64` is 8 GiB of parameters and 24 GiB once
SparseAdam materialises its two dense moment buffers. The configurations differ in
how they handle this table.

| Configuration | Table | Closed set | Track A | Config |
|---|---|---:|---:|---|
| KmerFormer exact, 1 layer, mean pool | 8 GiB | **91.15%** | 27.44% | [`13mer/exact_1L_50M_meanpool.yaml`](../configs/13mer/exact_1L_50M_meanpool.yaml) |
| KmerFormer exact, 1 layer, attention pool | 8 GiB | 90.74% | 26.12% | [`13mer/exact_1L_50M_attnpool.yaml`](../configs/13mer/exact_1L_50M_attnpool.yaml) |
| MetaTransformer exact, 1 layer | 8 GiB | 87.46% | 27.05% | (its authors' code) |
| KmerFormer hashed d128, 1 layer | 2 GiB | 86.80% | 23.37% | [`13mer/hashed_d128_1L_50M.yaml`](../configs/13mer/hashed_d128_1L_50M.yaml) |
| KmerFormer exact, 16 layers | 8 GiB | 85.63% | 23.14% | [`13mer/exact_16L_50M.yaml`](../configs/13mer/exact_16L_50M.yaml) |
| KmerFormer hashed d128, 16 layers | 2 GiB | 83.83% | 21.23% | [`13mer/hashed_d128_16L_50M.yaml`](../configs/13mer/hashed_d128_16L_50M.yaml) |
| KmerFormer hashed d64, 16 layers | 1 GiB | 79.26% | 18.81% | [`13mer/hashed_d64_16L_50M.yaml`](../configs/13mer/hashed_d64_16L_50M.yaml) |
| 13-mer naive Bayes (no network) | — | 74.9% | — | [`scripts/baselines/multinomial_nb.py`](../scripts/baselines/multinomial_nb.py) |

Observations:

**The head-to-head matches the encoder dimensions.** Row 1 and row 3 match on tokenizer, on
vocabulary (both enumerate the same 33,545,099 canonical 13-mers), on training data,
and encoder width/depth settings — `d_model=64`, 2 heads,
`d_ff=512`, one layer, dropout 0.1. KmerFormer leads by 3.69 points at equal
throughput and equal memory. Moving KmerFormer's configuration toward
MetaTransformer's accounts for 0.98 of the 3.69 points: its optimisation settings
cost 0.21 points, its architecture 1.53, and both together reach 90.17% from a
configuration that is parameter-identical above the embedding table. The remaining
2.71 points are unattributed. Checkpoint selection and shuffling remain procedural differences between
the codebases. See [`configs/ablations/mt_direction_*.yaml`](../configs/ablations/).
Forward-only KmerFormer reaches 91.144% against MetaTransformer's 87.458%, a
3.686-point margin; RC-TTA adds only 0.008 points to the KmerFormer score. The
advantage therefore also holds under forward-only inference (Supplementary S7).

**The hashed configuration trades 6.37 points for an eightfold smaller table at matched width and depth** (exact d64 85.63% vs hashed
d64 79.26%), and doubling the width to d128 recovers most of it (83.83%) on 2 GiB
instead of 8 GiB. Canonicalization and embedding optimization also differ, so this
measures a complete-configuration trade-off rather than an isolated hashing cost.

**Fewer layers score higher in these tested 13-mer configurations**: with matched attention pooling,
one exact layer leads sixteen by 5.11 points (90.74% vs 85.63%). The 5.52-point
headline gap additionally includes mean pooling; the hashed d128 gap is 2.97 points.
Nearly all parameters sit in the embedding table:
99.99% of that configuration's 2.15B parameters; encoder, pooling and classifier
together are 187,704 against MetaTransformer's 90,936. Both are a very large lookup
table with a very small network on top, so embedding-table size alone does not
explain the difference between them.

### At MetaTransformer's own data budget

| Configuration | Closed set | Track A | Config |
|---|---:|---:|---|
| KmerFormer exact, 1 layer, 250M reads | **99.16%** | 31.92% | [`13mer/exact_1L_250M_meanpool.yaml`](../configs/13mer/exact_1L_250M_meanpool.yaml) |
| MetaTransformer, 250M reads | 98.68% | **32.67%** | (its authors' code) |

Five times the data lifts the closed set from 91.15% to 99.16%, and both
implementations approach saturation. Their unrounded primary scores are 99.155%
and 98.683%, a 0.472-point margin. Lower-budget repeats do not estimate training
variance for this pair. On Track A the ordering reverses.

### NT-v2 at the 250M budget

Both rows in the table above are from-scratch 13-mer models. NT-v2 was also trained on
the 250M pool, twice. Retrieved training logs confirm both runs finished
(25/25 and 15/15 epochs); the selected checkpoints are epochs 24 and 14:

| Configuration | Labelled reads | fwd | rc | **RC-TTA** |
|---|---|---:|---:|---:|
| NT-v2 + LoRA (v9) | 50M | 66.36% | 66.31% | **67.12%** |
| NT-v2 + LoRA, warm-started from v9 | 250M | 66.63% | 66.55% | **67.29%** |
| NT-v2 + LoRA, adapters from scratch | 250M | 64.13% | 64.16% | 64.84% |

Configs: [`extra/ntv2_lora_250M_warmstart.yaml`](../configs/extra/ntv2_lora_250M_warmstart.yaml)
and [`extra/ntv2_lora_250M.yaml`](../configs/extra/ntv2_lora_250M.yaml).

**The 250M continuation improves on the 50M checkpoint by 0.17 points.** The
other completed 250M recipe scores 2.27 points lower than the 50M checkpoint.
Both use the pretrained backbone. Initialization, learning rates and training
schedule change across recipes, so this is not a controlled data-only effect.

> [!IMPORTANT]
> These scores use the archived 99,742-read pool. The recovered exclusion lists
> remove 258 read IDs and reconstruct its FASTA byte for byte from the primary
> 100K file, preserving sequence order and labels. The lists filter identifiers;
> they do not de-duplicate sequences. Original predictions reproduce all three scores.
> The manuscript's `supplementary_data/nt_99742_provenance/` contains the manifest,
> exclusion lists, predictions, checkpoint hashes and training records.
>
> The v9 primary array restricted to these same reads scores 67.083%, whereas
> the subset archive scores 67.117%, with 1,287 prediction differences. Membership
> alone does not explain the two series. Their precise implementation difference
> remains unverified; retain the series separately and record RC-TTA aggregation
> and precision when reproducing them.

The current manuscript reports these scores in Part 1, separately from the
unfiltered 100,000-read curve. The two 250M runs differ in initialization, head
and adapter learning rates, epoch budget, warmup ratio and stopping patience.
Both use batch 128 on 64 GPUs (8,192 effective) and a 2% validation split; v9
uses batch 256 and a 10% split. The runs are not seed replicates. The retrieved
NT evaluator averages probabilities under float16 autocast despite its mean-logit
docstring; the primary KmerFormer evaluator averages logits.

---

## 4. Off-catalogue evaluation

### The ANI ladder (Track B)

Three rungs at defined distances from the reference (skani, 95% species boundary).

> [!IMPORTANT]
> **These accuracies are computed only over the fourteen genera present at every
> rung**, not over each rung's full pool. That restriction is the point of the ladder:
> it makes the three points on a line differ in *distance from the reference* rather
> than in class count. Accuracies over each rung's full pool differ from these and
> are not comparable across rungs.

| Configuration | L1 strain<br>same species | L2 species<br>novel species | L2_far<br>no reference in reach | Retention<br>far / L1 |
|---|---:|---:|---:|---:|
| **KmerFormer 6-mer, 29 layers** | 45.87% | 45.77% | **22.37%** | **48.8%** |
| NT-v2 500M + LoRA, 6-mer | 41.87% | 42.26% | 20.79% | 49.6% |
| MetaTransformer 6-mer | 26.07% | 27.79% | 13.31% | 51.1% |
| 6-mer naive Bayes | 22.31% | 24.23% | 12.00% | 53.8% |
| MetaTransformer 13-mer, 250M | **78.12%** | 46.96% | 11.07% | 14.2% |
| KmerFormer exact 13-mer, 1L, 250M | 77.89% | 45.97% | 10.80% | 13.9% |
| KmerFormer exact 13-mer, 1L, 50M | 64.94% | 41.62% | 10.08% | 15.5% |
| MetaTransformer 13-mer, 50M | 62.66% | 41.54% | 11.10% | 17.7% |
| KmerFormer exact 13-mer, 16L | 55.89% | 35.59% | 8.47% | 15.2% |
| 13-mer naive Bayes | 55.62% | 41.96% | 12.36% | 22.2% |
| KmerFormer hashed d128, 1L | 50.03% | 31.51% | 6.57% | 13.1% |
| KmerFormer hashed d128, 16L | 44.41% | 29.50% | 6.55% | 14.8% |
| KmerFormer hashed d64, 16L | 39.14% | 28.11% | 6.39% | 16.3% |

The 6-mer depth series is listed separately because it is one architecture at four
depths:

| KmerFormer 6-mer, depth series | L1 strain | L2 species | L2_far | Retention |
|---|---:|---:|---:|---:|
| 1 layer | 26.69% | 27.79% | 12.01% | 45.0% |
| 8 layers | 40.68% | 41.16% | 20.10% | 49.4% |
| 16 layers | 43.60% | 43.82% | 21.27% | 48.8% |
| 29 layers *(also the row above)* | 45.87% | 45.77% | 22.37% | 48.8% |

Accuracy rises monotonically with depth at every rung, including the far one; 16
layers over 1 adds 16.91, 16.03 and 9.26 points at the three rungs, a gain that shrinks
with distance. Both 13-mer vocabularies show the reverse depth effect at all three.

Every cell is measured; regenerate the whole table from the stored predictions with

```bash
python scripts/eval/track_b_retention.py --track_b_out $KF_OUT/track_b --markdown
```

which also prints the fourteen genus ids it restricted to, so the restriction can be
audited.

**Two 6-mer groupings; the manuscript uses the narrower one.** Its band covers the
four *architecturally distinct* 6-mer configurations — the 29-layer model, NT-v2,
MetaTransformer's 6-mer and a 6-mer naive Bayes — quoted rounded as 49–54%, measured
at **48.8–53.8%**. The three rows marked *(depth)* are the same architecture at 1, 8
and 16 layers; counting them too gives **45.0–53.8% over seven configurations**. The script
prints both, so a number here can always be matched to the grouping it came from.
Long-*k* is unaffected either way: **13.1–22.2%** over nine configurations, quoted as 13–22%.

Retention therefore separates the two tokenizer families **with no overlap on either
grouping** — a 26.5-point gap over the four, 22.8 over all seven — across one to
twenty-nine encoder layers, exact and hashed vocabularies, 0.9M to 500M parameters,
and in two cases no neural network at all. In these comparisons the tokenizer family
determines the retention band; the depth series widens the 6-mer band downward without
reaching the long-*k* band.

> [!NOTE]
> Kraken 2 is not in this table because read-level accuracy is a different quantity
> for a method that declines reads: on the far rung its low accuracy mostly reflects
> declined reads. Its correct, wrong and declined shares on these rungs are in
> [the evaluation guide](EVALUATION.md#kraken2-database-correction), and
> `track_b_retention.py --include_abstaining` scores it from the stored predictions.
> The closed-set, abundance and throughput comparisons are in [§5](#5-kraken-2-comparison).

Retention describes the shape of a curve: the 6-mer naive
Bayes retains the most of anything here (53.8%) by starting at 22.31% and ending at
12.00%. What recommends the 29-layer 6-mer model is that it does both — the highest
first rung of any 6-mer configuration and a retention that leaves it highest at the last rung,
under the far rung's no-reliable-hit conditions.

### Embedding neighbourhoods

Fraction of a read's twenty nearest neighbours carrying its genus, measured in each
model's own pooled space (1.0% expected by chance):

| Model | Closed set | Track A | Retained |
|---|---:|---:|---:|
| KmerFormer exact 13-mer | 75.7% | 21.1% | 28% |
| MetaTransformer 13-mer | 73.7% | 22.8% | 31% |
| KmerFormer 6-mer, 29 layers | 34.6% | **27.0%** | 78% |
| NT-v2 | 20.6% | 20.1% | 98% |

The more of a representation is built from keys specific to the reference, the less of
it survives when the reference no longer contains the genome. Part of a long *k*-mer's
accuracy on held-out reads from known genomes therefore reflects **recognition of those
genomes**.

### Track A caveat

> [!NOTE]
> The manuscript reports Track A in the supplementary and leads its off-catalogue
> analysis with the ANI ladder (Track B), because Track A mixes distances. This page
> keeps a Track A column as one off-catalogue number per configuration; the values
> match the manuscript.

Track A guarantees only unseen *assemblies*, and it is a mixture: of its 119 genomes,
48 are the same species as a reference genome and 71 have no reference within 95% ANI.
Nine sit at or above 99.9% ANI — the same genome under another accession — and the
protocol already excludes three of those by name, leaving **six** that contribute
30,000 reads. Removing those six costs the exact 13-mer configuration 1.13 points at 50M
(27.44% → 26.31%) and 1.75 at 250M (31.92% → 30.17%), and leaves the 29-layer 6-mer
model unchanged (28.90% → 28.90%).

The configurations respond differently: on the six removed genomes the exact
13-mer configuration scores 48.06% against 26.31% on the rest of the pool, while the 29-layer
6-mer model scores 28.98% against 28.90%. This is consistent with greater reference
specificity in the exact 13-mer configuration; it does not establish that a 6-mer
model cannot memorize genomes.

Separately and additionally, the three accessions the protocol removes by name take the
pool from 595,000 reads to 580,000. Those three are scored 87–99% by the 13-mer configurations,
so evaluating on the raw pool inflates them by about 1.5 points — the same mechanism on
a disjoint subset, which is why 1.5 and 1.13 are not two estimates of one quantity.
See [`MISSING_ASSETS.md`](MISSING_ASSETS.md#two-traps-when-comparing-fetched-metrics-against-resultsmd),
which matters if you fetch the stored eval metrics and compare them with this table.

---

## 5. Kraken 2 comparison

| Setting | Kraken 2 | Best neural |
|---|---:|---:|
| Closed set, complete reference (1,535-genome index) | **99.06%** | 99.16% (KmerFormer 13-mer, 250M reads, 2.1B params) |
| Novel-species log-normal community, abundance Pearson *r* (revised genus-aware index with Bracken) | **0.818** | 0.522 (KmerFormer 6-mer, 29 layers) |
| Throughput, 100K reads, 1 / 20 CPU threads | 45,496 / 138,889 reads/s | see [compute](#6-compute) |

With reads drawn from the indexed genomes, both approaches approach saturation:
99.155% for the 250M-read exact configuration and 99.062% for raw Kraken 2, a 0.093-point
fixed-prediction gap. The lower-budget repeats do not estimate training variance
for this comparison. Kraken 2 reaches this accuracy without neural training, so
the tested complete-reference setting offers little accuracy incentive for the
larger learned classifier.

On the revised genus-aware log-normal community run, Kraken 2 with Bracken
leads both abundance correlation and composition error (r = 0.818, Bray–Curtis
0.204), compared with raw Kraken 2 (0.814, 0.351). The earlier 0.850 raw
correlation belongs to the species-only index. The neural rows in this
community analysis use unthresholded predictions; the new common-read
abstention comparison is restricted to the primary pool. See
[the evaluation guide](EVALUATION.md#kraken2-database-correction).

Building the matched index correctly matters. It needs genus nodes between root and
species, not the species-only tree a custom database most easily produces: without
them a read ambiguous between two species of one genus has nowhere to stop, climbs to
root, and is scored unclassified — which costs Kraken 2 20.01% of its reads and 19.09
points (79.97% against 99.06%) on the closed-set pool. Both figures are reproduced by
[`scripts/eval/kraken2_genus_calls.py`](../scripts/eval/kraken2_genus_calls.py) from
each index's own Kraken 2 output.

**The builder now defaults to the genus-aware tree.**
[`scripts/baselines/build_kraken2_db_1535.py`](../scripts/baselines/build_kraken2_db_1535.py)
writes genus nodes at `10000 + genus_class` and attaches each species to its genus.
Use `--taxonomy species-only` only for the explicit control. This repairs the
reconstruction path; a full index rebuild and score comparison have not been run
as part of repository preparation. See [`MISSING_ASSETS.md`](MISSING_ASSETS.md).
Check which index you have by counting
ranks in `taxonomy/nodes.dmp`;
[`scripts/eval/kraken2_genus_calls.py`](../scripts/eval/kraken2_genus_calls.py)
prints that histogram and refuses to treat the two alike.

### Abundance against detection, on one pool

Two questions come apart on the closed-set pool: *how much* of a genus is there
(abundance) and *whether it is there at all* (detection). Every configuration below is scored
from predictions on the same 100,000 reads
(`reads_100K_val.fa`, md5 `cf9220f0…`), through the same scorer, with one protocol
and one seed — 100 partition samples of 1,000 reads for abundance, 200 sparse
communities of 60 genera for detection, seed 42. Regenerate with
[`scripts/eval/s2_abundance_vs_detection.py`](../scripts/eval/s2_abundance_vs_detection.py);
values in [`docs/assets/s2_abundance_vs_detection.json`](assets/s2_abundance_vs_detection.json),
predictions alongside it.

> **Detection columns use tie-grouped ROC points (since 2026-09-26).** ROC points are formed only at
> the end of each group of tied scores ([`scripts/eval/_roc.py`](../scripts/eval/_roc.py)).
> Between 10,289 and 13,074 of the 24,000 pooled genus-community pairs score exactly
> 0, and the earlier one-pair-at-a-time accumulation placed the 95%-specificity
> operating point inside that tie, where no threshold can reach. Sensitivity at
> 95% specificity fell by 0.04-10.52 points and AUC rose by 0.0003-0.0011;
> read accuracy, *r* and Bray-Curtis are unaffected. The earlier 250M sensitivity lead
> over MetaTransformer becomes a tie under this realizable-threshold rule.
> The revised paper separately reports interpolated sensitivity at equal
> specificity; see [the operating-point comparison](EVALUATION.md#detection-operating-points).

| Configuration | Read acc. | Pearson *r* | Bray–Curtis | ROC AUC | Sens@95% spec. |
|---|---:|---:|---:|---:|---:|
| Kraken 2 raw, **genus-aware** index | 99.06% | **0.9999** | **0.005** | **0.955** | **90.99%** |
| KmerFormer exact 13-mer, 1L, 250M | **99.16%** | **0.9999** | 0.006 | 0.945 | 77.49% |
| MetaTransformer 13-mer, 250M | 98.68% | 0.9998 | 0.009 | 0.940 | 77.47% |
| KmerFormer exact 13-mer, 1L, 50M | 91.15% | 0.9981 | 0.043 | 0.864 | 46.19% |
| MetaTransformer 13-mer, 50M | 87.46% | 0.9976 | 0.054 | 0.837 | 41.49% |
| KmerFormer hashed 13-mer, 1L, d128, 50M | 86.80% | 0.9949 | 0.064 | 0.845 | 39.68% |
| KmerFormer exact 13-mer, 16L, 50M | 85.63% | 0.9978 | 0.056 | 0.815 | 37.30% |
| KmerFormer hashed 13-mer, 16L, d128, 50M | 83.83% | 0.9949 | 0.072 | 0.805 | 35.05% |
| Kraken 2 raw, species-only index | 79.97% | 0.8311 | 0.111 | 0.954 | 90.92% |
| KmerFormer hashed 13-mer, 16L, d64, 50M | 79.26% | 0.9881 | 0.099 | 0.766 | 27.98% |
| KmerFormer 6-mer, 29 layers, 50M | 69.53% | 0.9920 | 0.107 | 0.692 | 19.18% |
| NT-v2 6-mer + LoRA, 50M | 67.08% | 0.9894 | 0.127 | 0.671 | 16.31% |
| KmerFormer overlapping 6-mer, 50M | 62.75% | 0.9873 | 0.127 | 0.640 | 15.45% |
| MetaTransformer 6-mer, 50M | 48.92% | 0.9778 | 0.195 | 0.569 | 9.23% |

> The main primary-pool comparison now uses the MetaTransformer arrays:
> 87.458% for 13-mer and 48.920% for 6-mer. The recovered 99,742-read 13-mer
> archive scores 87.4667%, rounding to the previously quoted 87.47%; restricting
> the primary predictions to that subset gives 87.4657%, with two predictions
> differing. The historical 48.87% 6-mer value remains only in the separate
> 5,001,216-read validation comparison. The 2026-09-08 audit resolved L29:
> bfloat16 with logit averaging
> reproduces all historical predictions at 69.519%; float16 with probability
> averaging reproduces every bundled prediction at 69.527% (rounded here to 69.53%).
> The exact cause of the two retained-read MetaTransformer 13-mer prediction differences has not
> been established. No numerical protocol is selected by its resulting accuracy.

Each configuration is named by tokenizer, depth and data budget, and each Kraken 2
row by its taxonomy, because a genome count or a "100K" pool name alone does not
identify the index or the pool.

Observations:

**Raw Kraken 2 on the genus-aware index leads both abundance and detection**, with no
abundance re-estimation. The species-only build's *r* of 0.831 is not a property
of *k*-mer lookup; it is the 20.01% of reads that climb to root when no genus node
exists to stop at, and those missing counts deflate every genus at once. Insert the
genus nodes and unclassified falls to 0.94%, *r* goes to 0.9999 and Bray–Curtis to
0.005. Detection barely moves (AUC 0.954 against 0.955), because the reads that were
climbing to root were ambiguous *within* a genus and never decided whether a genus
is present. Both indexes come from the same 1,535-genome library, so a row named by
genome count cannot distinguish them — name the taxonomy.

**Against MetaTransformer, KmerFormer leads on every column at 50M and on all but
one at 250M.** At 50M: 91.15% against 87.46% read accuracy, *r* 0.9981 against
0.9976, AUC 0.864 against 0.837, sensitivity 46.19% against 41.49%. At 250M: 99.16%
against 98.68%, *r* 0.9999 against 0.9998 and AUC 0.945 against 0.940, while
sensitivity under the realizable-threshold rule is level, 77.49% against 77.47%
(88.60% against 83.07% at exactly 95% specificity; see
[the evaluation guide](EVALUATION.md#detection-operating-points)). The 250M
configuration also has the highest read accuracy here, Kraken 2 included. Kraken 2
still leads detection.

**Exact vocabularies lead hashed ones on abundance by more than read accuracy predicts.** Every
exact 13-mer configuration sits at *r* ≥ 0.9976 and every hashed one at ≤ 0.9949, and the
ordering crosses: the 16-layer exact configuration is 1.2 points *worse* on read accuracy
than the 1-layer hashed configuration (85.63% against 86.80%) and 0.0029 *better* on *r*
(0.9978 against 0.9949). These measured configuration differences also include
canonicalization, width, depth and embedding updates; they do not isolate hash
collisions as the source of the abundance gap.

**The 6-mer models form one group** — high *r*, weak detection, all four of them
across two architectures and two tokenizer variants. The 29-layer KmerFormer is in
that group, at 0.9920 and 19.18%. The pattern recurs across the tested 6-mer configurations; these
comparisons do not isolate tokenization from every other modeling choice.

#### The exact 13-mer predictions reproduce across machines

The two exact 13-mer configurations were re-run on an RTX 4090 from the checkpoints
described in [`weights/`](../weights/) and compared with predictions written in August
by separate jobs on an RTX A6000 from a different checkout. The arrays are
**elementwise identical**:

| Configuration | Re-run here (4090) | A6000, separate job | `preds` SHA-1 |
|---|---:|---:|---|
| exact 13-mer, 1L, 50M | 91.1520% | 91.1520% | `ddab9e2c74462755` both |
| exact 13-mer, 1L, 250M | 99.1550% | 99.1550% | `a31f6f2bc3acabfe` both |

Both label vectors hash to `9b9ec0c313f7f5fc`, which is this pool's `genus_class`
column. These two numbers therefore rest on two independent paths to the same array.

It also pins down what the repeat spread measures. The supplement quotes 0.113
points for this configuration, from the A6000's two 50M runs (91.152% and 91.265%), and those
two differ on **7,176 reads**. Since a re-run of the same checkpoint reproduces its
predictions exactly, that spread is **training** variance, not evaluation noise —
inference here is deterministic, and re-scoring an existing checkpoint will not
reproduce it.

Both runs' predictions are in the repository, so the pair can be checked directly:
the first is `docs/assets/KmerFormer_exact13mer_1L_50M_preds_clean_common.npz`, the
second `docs/assets/repeats/KmerFormer_exact13mer_1L_50M_run2_preds_clean_common.npz`.
Their label vectors are identical (SHA-1 `9b9ec0c313f7f5fc`); 2,155 reads are right
in the first and wrong in the second and 2,268 the other way round, which gives the
nominal standard error of 0.067 points the supplement quotes when the read pairs
are treated as independent. The second run sits in its own directory so the
regeneration command below, which loops over the table's arrays, does not treat it
as a fifteenth configuration.

#### Coverage-masked scoring

Restricting to the 85,773 reads whose source species is in the Kraken 2 index is
right for a read-accuracy comparison — it is the `read_acc_in_db` field, and it
moves every neural configuration by 2–3 points. It cannot produce a detection column. The
mask empties 17 of the 120 genera, so of the 60 genera each sparse sample declares
present, **8.3 on average hold no reads at all**: 14% of the positives are false
negatives by construction, for every method alike, and the same configuration reads AUC 0.899
masked against 0.955 unmasked. The two scorers in this repo disagree about what to
do here.
[`evaluate_sample.py`](../scripts/eval/evaluate_sample.py) caps
`reads_per_sample` by the smallest genus over all 120, so one empty genus collapses
it to zero and the run refuses itself;
[`evaluate_sample_kraken2.py`](../scripts/eval/evaluate_sample_kraken2.py) has no
such cap and returns the depressed number without comment. The regeneration script
prints the empty-genus count for both the full and the masked pool so the choice is
never implicit.

#### Identifying the 100K pool

Two 100,000-read pools over these same 120 genera exist, and their `genus_class`
vectors agree on exactly **8.00%** of positions — above chance for a 120-way label,
far below identity, so no length check, class count or spot check tells them apart.
`reads_100K_val.fa` (md5 `cf9220f0…`) is this one, and every number this repo and
the manuscript print comes from it. `reads_100K.fa` under `twcc_test100k`
(md5 `3436e31e…`) is a second, disjoint draw; no printed number uses it. Scoring
the same checkpoint on both gives 69.53% and 69.88%, so accuracy will not warn you
either. Filenames do not distinguish them:
`mt_genus_preds_100K.npz` and `mt_genus_preds_remapped_100K.npz` differ by *pool*,
not by a remapping. The regeneration script prints the pool md5, names which pool
it is, and refuses any configuration whose stored labels are not that pool's in that pool's
order — which is the only check that separates them.

### Real mock communities provide limited ranking evidence

Across D6331 and the correctly identified Mori cell-mix (DRR466867), historical
correlations span about 0.07–0.68. Every genus-bootstrap interval includes zero;
this does not by itself test paired model differences or establish model equivalence.
Kraken 2 + Bracken gives the highest D6331 point estimate (0.580); MetaTransformer
6-mer gives the highest Mori estimate (0.681). Revised Mori expectations use its
18 equal-cell strains weighted by published genome mass, not the old `Kim` composition.
The new exact one-layer 50M/250M evaluations give D6331 correlations 0.487–0.509,
and 0.601/0.576 on a shared 1.5M-read Mori pool. Their intervals are also broad.
The manuscript reports these as limited real-community evidence, not a general ranking.

---

## 6. Compute

Measured on one RTX A6000 (48 GiB) at batch 128 with 150 bp synthetic reads, warm-up
excluded. The KmerFormer and NT-v2 rows share one code path; the MetaTransformer rows
run that project's own model and tokenizer transform, timed identically. Parameters
are split into the vocabulary table and everything else, because for long-*k* configurations the
table is almost the whole model.

| Configuration | tok/read | reads/s infer | reads/s train | peak MiB infer | peak MiB train | params (M) embed / rest | sparse |
|---|---:|---:|---:|---:|---:|---|:-:|
| KmerFormer 6-mer, 1 layer | 32 | 108,414 | 12,434 | 27 | 68 | 0.5 / 0.4 | |
| KmerFormer 6-mer, 29 layers | 32 | 9,749 | 2,370 | 62 | 788 | 0.5 / 5.9 | |
| MetaTransformer 6-mer stride-1 | 145 | 148,014 | 17,583 | 66 | 167 | 0.3 / 0.2 | |
| NT-v2 500M + LoRA | 32 | 1,285 | 526 | 2,092 | 8,779 | 4.2 / 494.4 | |
| KmerFormer 13-mer hashed d128, 1L | 140 | 130,969 | 1,643 | 2,119 | 10,263 | 536.9 / 0.4 | |
| KmerFormer 13-mer exact d64, 1L | 140 | 187,862 | 16,088 | 8,248 | 24,712 | 2,146.9 / 0.2 | ✓ |
| MetaTransformer 13-mer stride-1 | 138 | 193,942 | 18,248 | 8,244 | 24,698 | 2,146.9 / 0.1 | ✓ |

Reproduce with [`scripts/bench/bench_arms.py`](../scripts/bench/bench_arms.py) and
[`configs/bench/`](../configs/bench/). NT-v2 is LoRA-adapted, so 5.54M of its 498.6M
parameters are trained; every KmerFormer configuration trains all of its own.

Observations:

**KmerFormer is cheaper by more than it is more accurate.** Against NT-v2 the 29-layer
configuration infers 7.6× faster in 34× less memory and trains 4.5× faster in 11× less; at one
layer, 84× faster on 27 MiB.

**The long-*k* cost is vocabulary, not arithmetic.** The 13-mer configurations infer at 6-mer
speed but need 20–175× the memory, and their embedding table is 160–10,000× the
encoder. This is what the frequently quoted "~5M parameters" for a long-*k* read
classifier omits.

**Training the enumerated long-*k* table exceeds the tested 24 GiB GPU's memory.**
The exact configuration peaks at 24,712 MiB (24.1 GiB) at one layer and 25,812 MiB (25.2 GiB)
at sixteen. Both fail on the tested 24 GiB card. The hashed vocabulary reduces
memory: the single-layer d128 configuration peaks at 10,263 MiB (10.0 GiB).

**Sparse gradients, not hashing, set the training speed.** The exact configurations take sparse
gradients under SparseAdam, so only the rows a batch touches are updated; the hashed
configurations use dense updates over all 2^22 rows every step. That is why the exact
single-layer configuration trains an order of magnitude faster than the hashed one (16,088 vs
1,643 reads/s) despite a table four times larger. A sparse hashed table would combine
both advantages, but sparse embeddings failed under the tested data-parallel
reducer and backend. This does not rule out other distributed implementations.

### Read accuracy and abundance correlation

Across fourteen configurations whose read-level Top-1 spans 48.92% to 99.16%, sample-level
genus abundance correlation moves only from *r* = 0.9778 to 0.9999 — a 50-point
spread compressed into 2.2 points of Pearson *r*. Abundance correlation, the quantity
most microbiome studies report, therefore hides most of these rankings, so this page
ranks configurations by read-level accuracy.

The exception is raw Kraken 2 on the species-only index (*r* = 0.8311), and its cause
is the index build:
because 20.01% of its reads climb to root and the missing counts deflate every
genus at once; on the genus-aware index the same tool reads 0.9999 with **no
abundance re-estimation at all**. So it is the index build, not Bracken, that
closes that gap — see the table above.

---

## 7. Additional ablations and controls

The effects of the convolutional front-end, reverse-complement processing and
pooling vary across the tested configurations, as summarized below.

| Idea | Result | Config |
|---|---|---|
| Multiscale convolutional front-end (kernels 2, 3, 5 over the token axis, summed residually) | 69.52% → 69.72% closed set, 28.90% → 29.23% Track A. This small change does not establish which local patterns the model fails to learn; ordered non-overlapping tokens preserve the input sequence. | [`ablations/L29_50M_conv_frontend.yaml`](../configs/ablations/L29_50M_conv_frontend.yaml) |
| Reverse complement | About +0.7 points to NT-v2, at most +0.21 to the tested non-overlapping KmerFormer 6-mer depths, and +5.44–6.95 to hashed 13-mer; ablating training-time RC augmentation moves the tested 5M 6-mer configuration by 0.13. | [`ablations/L16_5M_no_rc_augment.yaml`](../configs/ablations/L16_5M_no_rc_augment.yaml) |
| Attention pooling at *k* = 13 | Lower accuracy: −0.41 closed set, −1.32 Track A. | [`13mer/exact_1L_50M_attnpool.yaml`](../configs/13mer/exact_1L_50M_attnpool.yaml) |

The conv front-end is off by default and every configuration above except that one ran without
it.

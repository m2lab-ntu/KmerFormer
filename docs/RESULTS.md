# Results

This page distinguishes manuscript/config scores from the later bundled-array
audit (see [citation](../README.md#citation)). Config links identify the reported
arms; the abundance/detection section is recomputed from archived predictions.
See [the reproduction guide](REPRODUCE.md) for executable checks and input requirements.

**Read the two accuracy columns together.** They disagree about which model is best,
and the disagreement is the finding rather than noise. The closed-set column measures
reads from genomes the reference catalogue contains; Track A measures reads from
genomes it does not. The ordering inverts between them.

Conventions:

- **Closed set** — 100,000 held-out reads from the same 1,535 gut reference genomes,
  genus Top-1, 120 classes. KmerFormer and NT-v2 use RC-TTA; MetaTransformer is
  evaluated forward-only under its authors' protocol.
- **Track A** — 580,000 reads from 116 genera whose source assemblies are absent from
  the reference; 5,000 reads per genus, so micro and balanced accuracy coincide.
- All arms train on the same 50M-read balanced gut pool unless the row says otherwise.
  Each takes its own 90/10 train/validation split of it, so "matched on training data"
  means one source pool and one test set, not identical training indices.
- Run-to-run standard deviation is **0.12 points** across four runs of `L16_5M`
  (original plus three repeats). This is a reference scale for that configuration,
  not a universal uncertainty interval or a variance estimate for NT-v2.

---

## 1. Part 1 — foundation models: decent, not dominant

| Model | Params | Closed set | Track A | Config |
|---|---:|---:|---:|---|
| NT-v2 500M + LoRA, 6-mer | 500M | 67.08% | 25.81% | [`baselines/ntv2_lora_50M.yaml`](../configs/baselines/ntv2_lora_50M.yaml) |
| KmerFormer 6-mer, 29 layers | 6.45M | **69.52%** | **28.90%** | [`6mer/L29_50M.yaml`](../configs/6mer/L29_50M.yaml) |
| KmerFormer 6-mer, 16 layers | 3.88M | 67.94% | 27.46% | [`6mer/L16_50M.yaml`](../configs/6mer/L16_50M.yaml) |

A 6.45M-parameter model with no pre-training beats a 500M pre-trained one by 2.44
points at 77× fewer stored parameters, and by more off the catalogue. LoRA adapts
5.54M NT-v2 parameters, against all 6.45M in KmerFormer. The representation is
behind by the same margin as the classifier: NT-v2's pooled embeddings put a read's
genus on 20.6% of its twenty nearest neighbours against the 29-layer model's 34.6%, so
this is not a strong representation read out by a weak head.

### Backbone screen (5M reads, which selects NT-v2)

| Backbone | Tokenizer | Closed set | Config |
|---|---|---:|---|
| NT-v2 | non-overlapping 6-mer | **63.05%** | [`baselines/ntv2_lora_5M.yaml`](../configs/baselines/ntv2_lora_5M.yaml) |
| DNABERT-1 | overlapping 6-mer | 61.78% | [`baselines/dnabert1_lora_5M.yaml`](../configs/baselines/dnabert1_lora_5M.yaml) |
| DNABERT-2 | byte-pair encoding | 58.88% | [`baselines/dnabert2_lora_5M.yaml`](../configs/baselines/dnabert2_lora_5M.yaml) |

At the matched 5M budget, NT-v2 leads the tested DNABERT adaptation pipelines.
This screen supports selecting NT-v2 for the subsequent comparisons; it does not
establish a tokenizer-independent accuracy ceiling at 50M.

### Where pre-training does pay: the low-data regime

The comparison that carries this is **depth-matched** — against the 29-layer arm,
NT-v2's own depth — so the lead cannot be an artefact of pitting a deep backbone
against something shallower:

| Labelled reads | NT-v2 + LoRA | KmerFormer 6-mer **L29** | Advantage | KmerFormer L16 | vs L16 |
|---|---:|---:|---:|---:|---:|
| 0.5M | 54.72% | 41.03% | **+13.7** | 41.35% | +13.4 |
| 1M | 57.34% | 44.13% | **+13.2** | 44.29% | +13.1 |
| 5M | 63.05% | 54.19% | **+8.9** | 54.11% | +8.9 |
| 50M | 67.08% | 69.52% | **−2.4** | 67.94% | −0.9 |

Configs: [`baselines/ntv2_lora_{500K,1M,5M,50M}`](../configs/baselines/) ·
[`6mer/data_scale/`](../configs/6mer/data_scale/) ·
[`6mer/L29_50M`](../configs/6mer/L29_50M.yaml).

The advantage shrinks monotonically and changes sign, so this is a curve with a
crossing rather than two endpoints that happen to differ. It reproduces on an
second soil catalogue (RefSoil, 309 genera), also at matched depth: NT-v2 leads
by **+4.3** at 5M (31.98% vs 27.68%) and is crossed by **+6.5** at 50M (46.35% vs
39.84%) — a wider margin than the 2.4 it takes on gut. The pre-trained line is nearly
flat over these two budgets — a 10× increase in reads buys it +4.0 points.
The advantage belongs to the tested pipelines: initialization, width and LoRA
adaptation are not isolated by matching depth. The recovered soil NT configs also
differ from gut in RC augmentation, validation split, batch size and schedule
(Supplementary S10). The soil comparison tests recurrence of the ordering across
the two study settings, not an isolated catalogue effect.

**Soil's depth deltas do not transfer from gut, which is why the cell was measured
rather than extrapolated.** From 16 layers to 29, soil gains **+0.26** and **+2.67**
points at the two budgets where gut gains +0.08 and +1.58. Carrying gut's delta across
would have put the 50M soil cell at 45.26% against the measured 46.35%. Depth pays
more on the harder catalogue at both budgets.

> [!NOTE]
> The soil arms have no configs in this repository — the manuscript reports soil as a
> cross-environment check and shipping it means shipping a second catalogue, which is
> still an open decision ([`MISSING_ASSETS.md`](MISSING_ASSETS.md#5-deliberately-not-in-this-repo)).
> So unlike every other figure on this page, the soil numbers are quoted prose with no
> config banner behind them and nothing in the test suite constraining them. The
> 50M arm ran on Taiwania2; RC-TTA moves it by 0.04 points, 46.31% forward to 46.35%,
> consistent with the small gains measured for the tested non-overlapping 6-mer arms.

**Matching depth widens the pre-trained lead rather than explaining it.** Sixteen
layers to twenty-nine is worth −0.32, −0.16, +0.08 and +1.58 points as the budget
grows: at the two smallest budgets NT-v2's own depth makes the from-scratch model
slightly *worse*, and only at 50M does depth pay enough to matter. So the 13-point
lead at 0.5M and 1M is not a depth artefact — matching depth makes it larger.

> [!NOTE]
> An 8-layer arm was also trained at 5M and reaches 53.62%
> ([`data_scale/L8_5M.yaml`](../configs/6mer/data_scale/L8_5M.yaml)). Part 1 quoted
> it while the passage was organised around a range over three depths; the rewrite
> around the depth-matched arm dropped it. The measurement stands, so the config is
> kept with its banner marked `[not in the manuscript]` — a form both
> `scripts/check_against_manuscript.py` and the test suite recognise, so a number
> that is ours rather than the paper's cannot be mistaken for drift in either
> direction.

---

## 2. Part 2 — at a shared 6-mer tokenizer, depth is what pays

KmerFormer and NT-v2 share the non-overlapping tokenizer, source training pool,
label space and test set, but differ in architecture and LoRA versus full training.
MetaTransformer is a reference arm with its own stride-1 tokenizer and protocol.

| Arm | Params | Closed set | Track A | Config |
|---|---:|---:|---:|---|
| KmerFormer, 29 layers | 6.45M | **69.52%** | **28.90%** | [`6mer/L29_50M.yaml`](../configs/6mer/L29_50M.yaml) |
| KmerFormer, 16 layers | 3.88M | 67.94% | 27.46% | [`6mer/L16_50M.yaml`](../configs/6mer/L16_50M.yaml) |
| KmerFormer, 8 layers | 2.29M | 65.44% | 25.17% | [`6mer/L8_50M.yaml`](../configs/6mer/L8_50M.yaml) |
| KmerFormer, 1 layer | 0.90M | 53.92% | 15.22% | [`6mer/L1_50M.yaml`](../configs/6mer/L1_50M.yaml) |
| NT-v2 500M + LoRA | 500M | 67.08% | 25.81% | [`baselines/ntv2_lora_50M.yaml`](../configs/baselines/ntv2_lora_50M.yaml) |
| MetaTransformer 6-mer stride-1, 1 layer | 0.5M | 48.92% | — | (its authors' code) |

Depth alone is worth 15.60 points on the closed set, with diminishing but never
negative returns, and it keeps paying out-of-genome after the closed-set curve has
flattened (15.22 → 25.17 → 27.46 → 28.90). The 20.6-point margin over
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
both strands; the non-canonical hashed 13-mer arms instead gain 5.44–6.95 points.

---

## 3. Part 3 — at 13-mer the vocabulary stops being free

A full overlapping 13-mer table is 4^13 = 67.1M keys; the observed canonical
vocabulary is 33,545,099, which at `d_model=64` is 8 GiB of parameters and 24 GiB once
SparseAdam materialises its two dense moment buffers. The arms differ in how they
resolve that.

| Arm | Table | Closed set | Track A | Config |
|---|---|---:|---:|---|
| KmerFormer exact, 1 layer, mean pool | 8 GiB | **91.15%** | 27.44% | [`13mer/exact_1L_50M_meanpool.yaml`](../configs/13mer/exact_1L_50M_meanpool.yaml) |
| KmerFormer exact, 1 layer, attention pool | 8 GiB | 90.74% | 26.12% | [`13mer/exact_1L_50M_attnpool.yaml`](../configs/13mer/exact_1L_50M_attnpool.yaml) |
| MetaTransformer exact, 1 layer | 8 GiB | 87.46% | 27.05% | (its authors' code) |
| KmerFormer hashed d128, 1 layer | 2 GiB | 86.80% | 23.37% | [`13mer/hashed_d128_1L_50M.yaml`](../configs/13mer/hashed_d128_1L_50M.yaml) |
| KmerFormer exact, 16 layers | 8 GiB | 85.63% | 23.14% | [`13mer/exact_16L_50M.yaml`](../configs/13mer/exact_16L_50M.yaml) |
| KmerFormer hashed d128, 16 layers | 2 GiB | 83.83% | 21.23% | [`13mer/hashed_d128_16L_50M.yaml`](../configs/13mer/hashed_d128_16L_50M.yaml) |
| KmerFormer hashed d64, 16 layers | 1 GiB | 79.26% | 18.81% | [`13mer/hashed_d64_16L_50M.yaml`](../configs/13mer/hashed_d64_16L_50M.yaml) |
| 13-mer naive Bayes (no network) | — | 74.9% | — | [`scripts/baselines/multinomial_nb.py`](../scripts/baselines/multinomial_nb.py) |

Three things follow.

**The head-to-head matches the encoder dimensions.** Row 1 and row 3 match on tokenizer, on
vocabulary (both enumerate the same 33,545,099 canonical 13-mers), on training data,
and encoder width/depth settings — `d_model=64`, 2 heads,
`d_ff=512`, one layer, dropout 0.1. KmerFormer leads by 3.69 points at equal
throughput and equal memory. Walking our configuration toward MetaTransformer's
accounts for only part of it: its optimisation settings cost 0.21 points, its
architecture 1.53, both together reach 90.17% from a configuration that is
parameter-identical above the embedding table. Two thirds of the margin sits outside
everything we can enumerate, and we report that rather than a decomposition we cannot
support. Checkpoint selection and shuffling remain procedural differences between
the codebases. See [`configs/ablations/mt_direction_*.yaml`](../configs/ablations/).
Forward-only KmerFormer reaches 91.144% against MetaTransformer's 87.458%, a
3.686-point margin; RC-TTA adds only 0.008 points to the KmerFormer score. The
advantage therefore also holds under forward-only inference (Supplementary S22).

**The hashed configuration trades 6.37 points for an eightfold smaller table at matched width and depth** (exact d64 85.63% vs hashed
d64 79.26%), and doubling the width to d128 recovers most of it (83.83%) on 2 GiB
instead of 8 GiB. Canonicalization and embedding optimization also differ, so this
measures a complete-configuration trade-off rather than an isolated hashing cost.

**Depth hurts in these tested configurations**: with matched attention pooling,
one exact layer leads sixteen by 5.11 points (90.74% vs 85.63%). The 5.52-point
headline gap additionally includes mean pooling; the hashed d128 gap is 2.97 points.
And "one layer" is generous — 99.99% of
that arm's 2.15B parameters are the embedding table; encoder, pooling and classifier
together are 187,704 against MetaTransformer's 90,936. Both are a very large lookup
table with a very small network on top, which is why the difference between them
cannot be explained by embedding-table size alone.

### At MetaTransformer's own data budget

| Arm | Closed set | Track A | Config |
|---|---:|---:|---|
| KmerFormer exact, 1 layer, 250M reads | **99.16%** | 31.92% | [`13mer/exact_1L_250M_meanpool.yaml`](../configs/13mer/exact_1L_250M_meanpool.yaml) |
| MetaTransformer, 250M reads | 98.68% | **32.67%** | (its authors' code) |

Five times the data lifts the closed set from 91.15% to 99.16%, and both
implementations approach saturation. Their unrounded primary scores are 99.155%
and 98.683%, a 0.472-point margin. Lower-budget repeats do not estimate training
variance for this pair. On Track A the ordering reverses.

### NT-v2 at the 250M budget

> [!NOTE]
> An earlier version of this page said the pre-trained backbone was *unmeasured* at
> this budget, because no output existed on any machine reachable from here. It was
> measured. The runs completed on Nano4, and the absence of a local copy was
> mistaken for the absence of a run. After authenticated access on 2026-09-08,
> the original records and predictions were retrieved and checked.

Both rows in the table above are from-scratch 13-mer models. NT-v2 was also trained on
the 250M pool, twice. Retrieved Nano4 training logs confirm both runs finished
(25/25 and 15/15 epochs); the selected checkpoints are epochs 24 and 14:

| Arm | Labelled reads | fwd | rc | **RC-TTA** |
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
> 100K file, preserving sequence order and labels. This is identifier filtering,
> not DNA de-duplication. Original predictions reproduce all three scores.
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

## 4. Off the catalogue — the ordering crosses over

### The ANI ladder (Track B)

Three rungs at defined distances from the reference (skani, 95% species boundary).

> [!IMPORTANT]
> **These accuracies are computed only over the fourteen genera present at every
> rung**, not over each rung's full pool. That restriction is the point of the ladder:
> it makes the three points on a line differ in *distance from the reference* rather
> than in class count. Recomputing over the full pools gives different numbers that
> are not comparable across rungs — so a reader who does that and finds a mismatch
> has changed the measurement, not found an error.

| Arm | L1 strain<br>same species | L2 species<br>novel species | L2_far<br>no reference in reach | Retention<br>far / L1 |
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

The same series at four depths, kept apart from the table above rather than sorted
into it — one architecture at 1, 8, 16 and 29 layers is not four architectures, and
interleaved its 45.0% reads as a member of a band it is not in:

| KmerFormer 6-mer, depth series | L1 strain | L2 species | L2_far | Retention |
|---|---:|---:|---:|---:|
| 1 layer | 26.69% | 27.79% | 12.01% | 45.0% |
| 8 layers | 40.68% | 41.16% | 20.10% | 49.4% |
| 16 layers | 43.60% | 43.82% | 21.27% | 48.8% |
| 29 layers *(also the row above)* | 45.87% | 45.77% | 22.37% | 48.8% |

Depth rises monotonically at every rung, including the far one, and 16 layers over 1
is worth 16.91, 16.03 and 9.26 points at the three rungs — shrinking with distance,
like everything else here. Both 13-mer vocabularies do the reverse at all three.

Every cell is measured, not derived — regenerate the whole table from the stored
predictions with

```bash
python scripts/eval/track_b_retention.py --track_b_out $KF_OUT/track_b --markdown
```

which also prints the fourteen genus ids it restricted to, so the restriction is
auditable rather than asserted.

**Two 6-mer groupings, and the manuscript uses the narrower one.** Its band is over
the four *architecturally distinct* 6-mer arms — the 29-layer model, NT-v2,
MetaTransformer's 6-mer and a 6-mer naive Bayes — quoted rounded as 49–54%, measured
at **48.8–53.8%**. The three rows marked *(depth)* are the same architecture at 1, 8
and 16 layers; counting them too gives **45.0–53.8% over seven arms**. The script
prints both, so a number here can always be matched to the grouping it came from.
Long-*k* is unaffected either way: **13.1–22.2%** over nine arms, quoted as 13–22%.

Retention therefore separates the two tokenizer families **with no overlap on either
grouping** — a 26.5-point gap over the four, 22.8 over all seven — across one to
twenty-nine encoder layers, exact and hashed vocabularies, 0.9M to 500M parameters,
and in two cases no neural network at all. Nothing about the architecture moves an arm
out of the band its tokenizer puts it in. That the depth series widens the 6-mer band
downward without approaching the long-*k* one is the same finding read at four depths
instead of one.

> [!NOTE]
> **No Kraken 2 row here, following the manuscript.** Kraken 2 appears in the paper's
> framing, setup, cross-cutting compute section, discussion and supplementary, but in
> **none of its four results sections** — so its absence from this ladder is a
> consistent editorial choice rather than an omission in one table.
>
> `track_b_retention.py` *can* score it, and the predictions are stored alongside the
> others (`--include_abstaining`), but read-level accuracy on these rungs is not a
> like-for-like quantity for a method that abstains: a low far-rung number means
> "declined to call" where a neural arm's means "called it wrong". And adding it would
> not be adding a row — it would be Kraken 2's first appearance in a results section,
> which is a question about the paper's structure. That question is open, and this repo
> is not the place to settle it.
>
> Where the paper does compare Kraken 2 — the closed set, abundance under an incomplete
> reference, throughput — is [§5](#5-kraken-2-and-where-a-learned-classifier-is-the-wrong-tool).

Retention describes the shape of a curve, not the worth of a model: the 6-mer naive
Bayes retains the most of anything here (53.8%) by starting at 22.31% and ending at
12.00%. What recommends the 29-layer 6-mer model is that it does both — the highest
first rung of any 6-mer arm and a retention that leaves it highest at the last rung,
under the far rung's no-reliable-hit conditions.

### Embedding structure says the same thing without a classifier

Fraction of a read's twenty nearest neighbours carrying its genus, measured in each
model's own pooled space (1.0% expected by chance):

| Model | Closed set | Track A | Retained |
|---|---:|---:|---:|
| KmerFormer exact 13-mer | 75.7% | 21.1% | 28% |
| MetaTransformer 13-mer | 73.7% | 22.8% | 31% |
| KmerFormer 6-mer, 29 layers | 34.6% | **27.0%** | 78% |
| NT-v2 | 20.6% | 20.1% | 98% |

The more of a representation is built from keys specific to the reference, the less of
it survives when the reference no longer contains the genome. So part of what a long
*k*-mer buys on held-out reads from known genomes is **recognition of those genomes**
rather than transferable genus signal.

### Track A caveat

> [!NOTE]
> **This page and the manuscript place Track A differently.** The manuscript reports
> it in the supplementary and leads its off-catalogue argument with the ANI ladder
> (Track B), which is the stratified comparison; Track A is a mixture of distances
> and answers "how far" only on average. This page keeps a Track A column on every
> table because it is a single per-arm number for "off the catalogue" and is useful
> when picking a checkpoint. The figures are the same in both. If you are checking
> the repo against the paper, expect the emphasis to differ and the numbers not to.

Track A guarantees only unseen *assemblies*, and it is a mixture: of its 119 genomes,
48 are the same species as a reference genome and 71 have no reference within 95% ANI.
Nine sit at or above 99.9% ANI — the same genome under another accession — and the
protocol already excludes three of those by name, leaving **six** that contribute
30,000 reads. Removing those six costs the exact 13-mer arm 1.13 points at 50M
(27.44% → 26.31%) and 1.75 at 250M (31.92% → 30.17%), and costs the 29-layer 6-mer
model **nothing at all** (28.90% → 28.90%).

The arms respond differently: on the six removed genomes the exact
13-mer arm scores 48.06% against 26.31% on the rest of the pool, while the 29-layer
6-mer model scores 28.98% against 28.90%. This is consistent with greater reference
specificity in the exact 13-mer configuration; it does not establish that a 6-mer
model cannot memorize genomes.

Separately and additionally, the three accessions the protocol removes by name take the
pool from 595,000 reads to 580,000. Those three are scored 87–99% by the 13-mer arms,
so evaluating on the raw pool inflates them by about 1.5 points — the same mechanism on
a disjoint subset, which is why 1.5 and 1.13 are not two estimates of one quantity.
See [`MISSING_ASSETS.md`](MISSING_ASSETS.md#two-traps-when-comparing-fetched-metrics-against-resultsmd),
which matters if you fetch the stored eval metrics and compare them with this table.

---

## 5. Kraken 2, and where a learned classifier is the wrong tool

This section is here because leaving it out would misrepresent the work.

| Setting | Kraken 2 | Best neural |
|---|---:|---:|
| Closed set, complete reference (1,535-genome index) | **99.06%** | 99.16% (KmerFormer 13-mer, 250M reads, 2.1B params) |
| Novel-species log-normal community, abundance Pearson *r* | **0.850** | 0.522 (KmerFormer 6-mer, 29 layers) |
| Throughput, 100K reads, 1 / 20 CPU threads | 45,496 / 138,889 reads/s | see [compute](#6-compute) |

With reads drawn from the indexed genomes, both approaches approach saturation:
99.155% for the 250M-read exact arm and 99.062% for raw Kraken 2, a 0.093-point
fixed-prediction gap. The lower-budget repeats do not estimate training variance
for this comparison. Kraken 2 reaches this accuracy without neural training, so
the tested complete-reference setting offers little accuracy incentive for the
larger learned classifier.

On the novel-species log-normal community, raw Kraken 2 leads Pearson correlation,
while Bracken improves composition error and detection (Supplementary S22).
Kraken 2 declines to call 53.9% of reads. Three genera
exceed 0.1% of its calls while accounting for less than 0.1% of the true composition;
two are absent from the community. Every
neural method commits on every read and reports 8–12 false-positive genera.
These are method-level comparisons; a matched-coverage neural abstention control
would be needed to isolate the contribution of declining reads.

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
(abundance) and *whether it is there at all* (detection). Every arm below is scored
from predictions on the same 100,000 reads
(`reads_100K_val.fa`, md5 `cf9220f0…`), through the same scorer, with one protocol
and one seed — 100 partition samples of 1,000 reads for abundance, 200 sparse
communities of 60 genera for detection, seed 42. Regenerate with
[`scripts/eval/s2_abundance_vs_detection.py`](../scripts/eval/s2_abundance_vs_detection.py);
values in [`docs/assets/s2_abundance_vs_detection.json`](assets/s2_abundance_vs_detection.json),
predictions alongside it.

| Arm | Read acc. | Pearson *r* | Bray–Curtis | ROC AUC | Sens@95% spec. |
|---|---:|---:|---:|---:|---:|
| Kraken 2 raw, **genus-aware** index | 99.06% | **0.9999** | **0.005** | **0.954** | **91.44%** |
| KmerFormer exact 13-mer, 1L, 250M | **99.16%** | **0.9999** | 0.006 | 0.944 | 88.02% |
| MetaTransformer 13-mer, 250M | 98.68% | 0.9998 | 0.009 | 0.939 | 83.30% |
| KmerFormer exact 13-mer, 1L, 50M | 91.15% | 0.9981 | 0.043 | 0.863 | 50.63% |
| MetaTransformer 13-mer, 50M | 87.46% | 0.9976 | 0.054 | 0.836 | 43.20% |
| KmerFormer hashed 13-mer, 1L, d128, 50M | 86.80% | 0.9949 | 0.064 | 0.844 | 40.78% |
| KmerFormer exact 13-mer, 16L, 50M | 85.63% | 0.9978 | 0.056 | 0.814 | 40.87% |
| KmerFormer hashed 13-mer, 16L, d128, 50M | 83.83% | 0.9949 | 0.072 | 0.804 | 35.62% |
| Kraken 2 raw, species-only index | 79.97% | 0.8311 | 0.111 | 0.954 | 91.37% |
| KmerFormer hashed 13-mer, 16L, d64, 50M | 79.26% | 0.9881 | 0.099 | 0.765 | 28.55% |
| KmerFormer 6-mer, 29 layers, 50M | 69.53% | 0.9920 | 0.107 | 0.692 | 19.44% |
| NT-v2 6-mer + LoRA, 50M | 67.08% | 0.9894 | 0.127 | 0.670 | 17.22% |
| KmerFormer overlapping 6-mer, 50M | 62.75% | 0.9873 | 0.127 | 0.639 | 15.76% |
| MetaTransformer 6-mer, 50M | 48.92% | 0.9778 | 0.195 | 0.568 | 9.28% |

> The main primary-pool comparison now uses the MetaTransformer arrays:
> 87.458% for 13-mer and 48.920% for 6-mer. The recovered 99,742-read 13-mer
> archive scores 87.4667%, rounding to the previously quoted 87.47%; restricting
> the primary predictions to that subset gives 87.4657%, with two predictions
> differing. The historical 48.87% 6-mer value remains only in the separate
> 5,001,216-read validation comparison. The 2026-09-08 audit resolved L29:
> bfloat16 with logit averaging
> reproduces all historical predictions at 69.519%; float16 with probability
> averaging reproduces every bundled prediction at 69.527% (rounded here to 69.53%).
> The exact cause of the two retained-read MT13 prediction differences has not
> been established. No numerical protocol is selected by its resulting accuracy.

Every arm is named by tokenizer, depth and data budget, because the two mistakes this
table has already produced were both a name hiding a variable — a Kraken 2 row named
by genome count when the taxonomy was what differed, and a "100K" pool name shared by
two disjoint pools.

Three things to read off it.

**Raw Kraken 2 on the correct index is best at both**, and no abundance
re-estimation is involved. The species-only build's *r* of 0.831 is not a property
of *k*-mer lookup; it is the 20.01% of reads that climb to root when no genus node
exists to stop at, and those missing counts deflate every genus at once. Insert the
genus nodes and unclassified falls to 0.94%, *r* goes to 0.9999 and Bray–Curtis to
0.005. Detection is unchanged (0.954 either way), because the reads that were
climbing to root were ambiguous *within* a genus and never decided whether a genus
is present. Both indexes come from the same 1,535-genome library, so a row named by
genome count cannot distinguish them — name the taxonomy.

**The head-to-head against MetaTransformer holds on every column, at both
budgets.** At 250M, 99.16% against 98.68% read accuracy, *r* 0.9999 against 0.9998,
AUC 0.944 against 0.939, sensitivity 88.02% against 83.30%. At 50M, 91.15% against
87.46%, 0.9981 against 0.9976, 0.863 against 0.836, 50.63% against 43.20%. The
250M arm also has the highest read accuracy of anything here, Kraken 2 included —
though Kraken 2 still leads detection, so "best at both" stays Kraken 2's.

**Exact beats hashed on abundance by more than read accuracy predicts.** Every
exact 13-mer arm sits at *r* ≥ 0.9976 and every hashed one at ≤ 0.9949, and the
ordering crosses: the 16-layer exact arm is 1.2 points *worse* on read accuracy
than the 1-layer hashed arm (85.63% against 86.80%) and 0.0029 *better* on *r*
(0.9978 against 0.9949). These measured configuration differences also include
canonicalization, width, depth and embedding updates; they do not isolate hash
collisions as the source of the abundance gap.

**The 6-mer models sit together** — high *r*, weak detection, all four of them
across two architectures and two tokenizer variants. Ours is in that group, 0.9920
and 19.44%. The pattern recurs across the tested 6-mer configurations; these
comparisons do not isolate tokenization from every other modeling choice.

#### Two of these rows reproduce bit-for-bit across machines

The two exact 13-mer arms had no predictions on `/nas2`, so they were re-run here
from the local checkpoints described in [`weights/`](../weights/); reviewer download
access is not yet recorded. They also exist on the
A6000, written in August by separate jobs from a different checkout. The arrays are
**elementwise identical** — not merely equal on accuracy:

| Arm | Re-run here (4090) | A6000, separate job | `preds` SHA-1 |
|---|---:|---:|---|
| exact 13-mer, 1L, 50M | 91.1520% | 91.1520% | `ddab9e2c74462755` both |
| exact 13-mer, 1L, 250M | 99.1550% | 99.1550% | `a31f6f2bc3acabfe` both |

Both label vectors hash to `9b9ec0c313f7f5fc`, which is this pool's `genus_class`
column. So these two numbers rest on two independent paths to the same array rather
than on one path agreeing with a written-down constant — which matters for a table
whose previous version was built on constants nobody could regenerate.

It also pins down what the repeat spread measures. The supplement quotes 0.113
points for this arm, from the A6000's two 50M runs (91.152% and 91.265%), and those
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
as a fifteenth arm.

#### The coverage mask gives read accuracy and nothing else

Restricting to the 85,773 reads whose source species is in the Kraken 2 index is
right for a read-accuracy comparison — it is the `read_acc_in_db` field, and it
moves every neural arm by 2–3 points. It cannot produce a detection column. The
mask empties 17 of the 120 genera, so of the 60 genera each sparse sample declares
present, **8.3 on average hold no reads at all**: 14% of the positives are false
negatives by construction, for every method alike, and the same arm reads AUC 0.899
masked against 0.954 unmasked. The two scorers in this repo disagree about what to
do here, which is worth knowing before trusting either.
[`evaluate_sample.py`](../scripts/eval/evaluate_sample.py) caps
`reads_per_sample` by the smallest genus over all 120, so one empty genus collapses
it to zero and the run refuses itself;
[`evaluate_sample_kraken2.py`](../scripts/eval/evaluate_sample_kraken2.py) has no
such cap and returns the depressed number without comment. The regeneration script
prints the empty-genus count for both the full and the masked pool so the choice is
never implicit.

#### "100K" does not identify the pool

Two 100,000-read pools over these same 120 genera exist, and their `genus_class`
vectors agree on exactly **8.00%** of positions — above chance for a 120-way label,
far below identity, so no length check, class count or spot check tells them apart.
`reads_100K_val.fa` (md5 `cf9220f0…`) is this one, and every number this repo and
the manuscript print comes from it. `reads_100K.fa` under `twcc_test100k`
(md5 `3436e31e…`) is a second, disjoint draw; no printed number uses it. Scoring
the same checkpoint on both gives 69.53% and 69.88%, so accuracy will not warn you
either. Filenames are worse than useless here:
`mt_genus_preds_100K.npz` and `mt_genus_preds_remapped_100K.npz` differ by *pool*,
not by a remapping. The regeneration script prints the pool md5, names which pool
it is, and refuses any arm whose stored labels are not that pool's in that pool's
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
are split into the vocabulary table and everything else, because for long-*k* arms the
table is almost the whole model.

| Arm | tok/read | reads/s infer | reads/s train | peak MiB infer | peak MiB train | params (M) embed / rest | sparse |
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
parameters are trained; every KmerFormer arm trains all of its own.

Four things this table settles.

**KmerFormer is cheaper by more than it is more accurate.** Against NT-v2 the 29-layer
arm infers 7.6× faster in 34× less memory and trains 4.5× faster in 11× less; at one
layer, 84× faster on 27 MiB.

**The long-*k* cost is vocabulary, not arithmetic.** The 13-mer arms infer at 6-mer
speed but need 20–175× the memory, and their embedding table is 160–10,000× the
encoder. This is what the frequently quoted "~5M parameters" for a long-*k* read
classifier omits.

**Training the enumerated long-*k* table exceeds the tested 24 GiB GPU's memory.**
The exact arm peaks at 24,712 MiB (24.1 GiB) at one layer and 25,812 MiB (25.2 GiB)
at sixteen. Both fail on the tested 24 GiB card. The hashed vocabulary reduces
memory: the single-layer d128 arm peaks at 10,263 MiB (10.0 GiB).

**Sparse gradients, not hashing, set the training speed.** The exact arms take sparse
gradients under SparseAdam, so only the rows a batch touches are updated; the hashed
arms use dense updates over all 2^22 rows every step. That is why the exact
single-layer arm trains an order of magnitude faster than the hashed one (16,088 vs
1,643 reads/s) despite a table four times larger. A sparse hashed table would combine
both advantages, but sparse embeddings failed under the tested data-parallel
reducer and backend. This does not rule out other distributed implementations.

### What the reported metric cannot see

Across fourteen arms whose read-level Top-1 spans 48.92% to 99.16%, sample-level
genus abundance correlation moves only from *r* = 0.9778 to 0.9999 — a 50-point
spread compressed into 2.2 points of Pearson *r*. Every ranking on this page is
invisible to the quantity most microbiome studies report, which is why everything
is ranked by read-level accuracy.

The one arm that breaks the pattern breaks it for a fixable reason and not a
methodological one. Raw Kraken 2 on the species-only index reads *r* = 0.8311,
because 20.01% of its reads climb to root and the missing counts deflate every
genus at once; on the genus-aware index the same tool reads 0.9999 with **no
abundance re-estimation at all**. So it is the index build, not Bracken, that
closes that gap — see the table above.

---

## 7. What did not work

Reported so it is not retried on the strength of the same intuition.

| Idea | Result | Config |
|---|---|---|
| Multiscale convolutional front-end (kernels 2, 3, 5 over the token axis, summed residually) | 69.52% → 69.72% closed set, 28.90% → 29.23% Track A. Composing short tokens downstream does not recover what the tokenizer discarded upstream. | [`ablations/L29_50M_conv_frontend.yaml`](../configs/ablations/L29_50M_conv_frontend.yaml) |
| Reverse complement | About +0.7 points to NT-v2, at most +0.21 to the tested non-overlapping KmerFormer 6-mer depths, and +5.44–6.95 to hashed 13-mer; ablating training-time RC augmentation moves the tested 5M 6-mer arm by 0.13. | [`ablations/L16_5M_no_rc_augment.yaml`](../configs/ablations/L16_5M_no_rc_augment.yaml) |
| Attention pooling at *k* = 13 | A cost, not a gain: −0.41 closed set, −1.32 Track A. | [`13mer/exact_1L_50M_attnpool.yaml`](../configs/13mer/exact_1L_50M_attnpool.yaml) |

The conv front-end is off by default and every arm above except that one ran without
it. A GENERanno backbone was also trained at three data budgets and is not reported in
the manuscript; its configs are in [`configs/extra/`](../configs/extra/).

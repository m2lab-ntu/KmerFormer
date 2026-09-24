# Track B — novel-species OOD ladder (E38–E43)

Genus-level classification when the **species** in the test read was never in
training, extending Track A (which only guaranteed unseen *accessions*, and so
mixed same-species strains together with genuinely novel species).

**Results:** [`docs/RESULTS.md`](../../docs/RESULTS.md#the-ani-ladder-track-b) — the ANI
ladder, measured from the stored predictions and regenerable with
[`scripts/eval/track_b_retention.py`](../eval/track_b_retention.py).

Working directory: `$KF_TRACK_B` — read pools, `preds.npz` and Kraken 2 output,
all far too large for git. The scripts live here in `scripts/track_b/` and write
there. Track A's pipeline is the same shape, in `scripts/track_a/`.

## `--models` is a hardcoded list, and the steps now say what they skipped

`tb_07`, `tb_09` and `tb_14` each take `--models` with a default naming the arms
that existed when the step was written. Add an arm later and it is excluded from
every level with no message — the only evidence is a row missing from the output.

That cost a session. The manuscript cites the 29-layer KmerFormer at *r* = 0.522 on
the L2_lognormal pool; no such row was in the stored summary, because `L29` was not
in `tb_09`'s default while its predictions sat in `out/L2_lognormal/L29/` the whole
time. Re-scoring with it added gives 0.522 exactly.

Each step now lists what is on disk against what it scored:

```
  scored 5 of 6 model directories under .../out/L2_lognormal
  NOT SCORED: L29
    Add them to --models, or pass --allow-unscored to say the omission is deliberate.
```

It warns rather than fails, because scoring a subset on purpose is normal; the point
is that the choice appears in the run's own output. `--allow-unscored` drops the
advice and keeps the count, so the fact stays visible either way.

Against what is on disk today the defaults skip 10 of 17 model directories per ladder
level in `tb_07`, 12 of 17 in `tb_09` and 13 of 17 in `tb_14`. Most of that is fine —
the ladder figures in [`docs/RESULTS.md`](../../docs/RESULTS.md#the-ani-ladder-track-b)
come from [`scripts/eval/track_b_retention.py`](../eval/track_b_retention.py), which
discovers arms by listing the directory instead of naming them — but deliberate and
forgotten looked identical until now.

## Why ANI and not species names

The training catalogue is 1,535 references, but only 219 are RefSeq (`GCF_*`)
with real binomial names. 1,179 are UMGS MAGs and 137 are HGR genomes, and
`taxonomy_umgs.tab` / `taxonomy_hgr.tab` only resolve them **down to genus** —
there is no species name to compare against. "Same genus, different species"
therefore cannot be decided by name matching.

Instead every candidate genome is compared by ANI (skani) against **all 1,535
training genomes**, and the 95% ANI species boundary defines the ladder:

| Level | max ANI to same-genus training genome | Meaning |
|---|---|---|
| `L0_dup` | ≥ 99% | near-duplicate of a training genome → **dropped** |
| `L1_strain` | 95–99% | same species, unseen strain (Track A's regime) |
| `L2_species` | 80–95% | **novel species, known genus** ← main target |
| `L2_far` | no hit / < 80% | novel species, distant from anything in training |

ANI is kept as a continuous value so E40 can plot accuracy against it rather
than reporting a single OOD number.

Two guards, both because a wrong call here silently invalidates every accuracy
downstream:

- **Cross-genus leak**: a candidate ≥99% ANI to a training genome labelled as a
  *different* genus has an ambiguous ground-truth label and is dropped.
- **Prediction alignment**: `tb_07` refuses to report metrics unless the labels
  stored inside each `preds.npz` match the pool label table position by position.

## Pipeline

```bash
# art_illumina, skani, kraken2/bracken and NCBI datasets must be on PATH
cd "$KF_TRACK_B"

# 1. shortlist candidate RefSeq genomes per training genus (no downloads)
python scripts/tb_01_survey_candidates.py \
    --genus_map assets/genus_map.tsv \
    --training_species assets/training_species_by_genus.tsv \
    --out_dir . --max_species_per_genus 8

# 2. download the shortlist
python scripts/tb_02_download_genomes.py \
    --survey candidate_survey.tsv --out_dir genomes

# 3. ANI-stratify into the ladder
python scripts/tb_03_ani_stratify.py \
    --survey candidate_survey.tsv \
    --training_species assets/training_species_by_genus.tsv \
    --genomes_dir genomes --out_dir . --threads 16

# 4. simulate reads (Track A's ART settings) and build balanced pools
python scripts/tb_04_simulate_and_build.py \
    --ani_summary ani_summary.tsv --genomes_dir genomes \
    --genus_map assets/genus_map.tsv --out_dir . \
    --reads_per_genome 5000 --max_reads_per_genus 10000

# 5. neural inference, per level  (E39 / E42)
bash scripts/tb_05_run_inference.sh L1_strain
bash scripts/tb_05_run_inference.sh L2_species
bash scripts/tb_05_run_inference.sh L2_far

# 6. Kraken2 (matched 1,535 DB) on the same pools  (E41)
for L in L1_strain L2_species L2_far; do
  python scripts/tb_06_run_kraken2.py --level $L \
      --pool_fa test_data/$L.fa --pool_labels test_data/${L}_labels.tsv \
      --out_dir out/$L/kraken2
done

# 7. ladder + ANI curve + per-genus  (E39 / E40 / E42 / E45)
python scripts/tb_07_metrics.py --track_b_dir . --out_dir results

# 8-9. non-uniform community, then abundance metrics  (E43)
python scripts/tb_08_build_lognormal_community.py --source_level L2_species
bash scripts/tb_05_run_inference.sh L2_lognormal
python scripts/tb_06_run_kraken2.py --level L2_lognormal \
    --pool_fa test_data/L2_lognormal.fa \
    --pool_labels test_data/L2_lognormal_labels.tsv \
    --out_dir out/L2_lognormal/kraken2
python scripts/tb_09_abundance_metrics.py --level L2_lognormal --out_dir results
```

## Models

All inference is forward-only except NT-v2 (RC-TTA), matching Track A. Nothing
is retrained, so this whole track runs on the local 4090.

| Key | Model | Checkpoint |
|---|---|---|
| `mt_250M` | MT 13-mer, 250M reads | `track_a/assets/mt_13mer_250M/` |
| `mt_50M` | MT 13-mer, 50M reads | `track_a/assets/mt_13mer_50M/` |
| `mt_6mer` | MT 6-mer stride 1, 50M | `mt6mer_assets/mt_6mer_stride1_50M_887333/` |
| `nt_v9` | NT-v2 + LoRA, 50M (RC-TTA) | `track_a/assets/nt_v9/` |
| `kraken2` | Kraken2 + Bracken, 1,535-genome DB | `$KRAKEN2_DB` — see the warning below |

> **The index must have genus nodes.** `scripts/baselines/build_kraken2_db_1535.py`
> writes 1,535 species nodes parented directly to root and no genus nodes, which is
> **not** the index the reported Kraken 2 numbers come from. Without genus nodes a
> read ambiguous between two species of one genus has nowhere to stop, climbs to
> root and is scored unclassified — 20.01% of the closed-set pool, costing 19.09
> points (79.97% against 99.06%) and dropping raw abundance from *r* = 0.9999 to
> 0.8311. The genus-aware builder has not been recovered
> ([`docs/MISSING_ASSETS.md`](../../docs/MISSING_ASSETS.md) has the recipe).
> [`scripts/eval/kraken2_genus_calls.py`](../eval/kraken2_genus_calls.py) prints the
> rank histogram of whichever index it is given, so check before scoring.

## Genus-name fixes applied in tb_01

- `Bacillus` — the bare name is an exact match for two NCBI taxids (walking
  sticks 55087 and the firmicute 1386); the query is pinned to **1386**.
- `Massiliomicrobiota` — training-set spelling; NCBI's current genus is
  `Massilimicrobiota`. `Massiliimicrobium` is a *different* genus and is
  rejected. Only contig-level assemblies exist, so the N50 floor is relaxed
  for this genus (recorded in `candidate_survey.tsv`).

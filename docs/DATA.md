# Data

No reads, genomes or vocabularies are in this repository. Every path in `configs/` is
written against three environment variables, so one config runs unchanged on any
machine:

| Variable | Holds |
|---|---|
| `KF_DATA` | read pools and evaluation sets |
| `KF_OUT` | where a run writes its checkpoints and metrics |
| `KF_VOCAB` | the directory holding `vocab_13mer.txt` (exact 13-mer arms only) |

An unset variable is left verbatim rather than replaced by an empty string, so a
missing one fails as `no such file: $KF_DATA/reads_50M.fa` instead of silently reading
from the filesystem root.

```bash
export KF_DATA=/path/to/kmerformer-data
export KF_OUT=/path/to/runs
export KF_VOCAB=$KF_DATA/vocab
```

## Expected layout

```
$KF_DATA/
├── balanced_500000/   reads.fa        labels.tsv          # Part 1 data-scale curve
├── balanced_1000000/  reads.fa        labels.tsv
├── balanced_5M/       reads_5M.fa     labels_5M.tsv
├── balanced_50M/      reads_50M.fa    labels_50M.tsv      # the main training pool
├── balanced_250M/     reads_250M.fa   labels_250M.tsv     # MetaTransformer's budget
├── val_100K/          reads_100K_val.fa  labels_100K_val.tsv   # closed-set test set
├── track_a/           newgenome_test.fa  newgenome_test_labels.tsv
└── track_b/           L1_strain.fa    L1_strain_labels.tsv
                       L2_species.fa   L2_species_labels.tsv
                       L2_far.fa       L2_far_labels.tsv
                       L2_lognormal.fa L2_lognormal_expected_abundance.tsv
$KF_VOCAB/
└── vocab_13mer.txt                        # 33,545,099 lines, 448 MiB
    vocab_13mer.txt.table_k13.npy          # built on first use, 256 MiB, cached
```

Sizes: `reads_50M.fa` is 8.9 GiB and `labels_50M.tsv` 3.6 GiB; the 250M pool is five
times that. Plan for ~80 GiB if you want every arm.

## File formats

Reads are plain FASTA, one 150 bp record per read, with the label encoded in the
header so a read and its truth travel together:

```
>lbl|397|UMGS1075|34|Collinsella-225/1
CCGCCGTGTCGGTTGTCGGCAAGGACGAGTACGGAGACAAGATGAGAGAGGCGCTGGCC...
```

The header fields are `lbl | species_class | assembly | genus_class | genus_name -
read_index / mate`. Labels are a TSV carrying the same information in columns, which is
what the loader actually reads:

```
idx  seq_id                                  species_class  genus_class  genus_name    species_name
0    lbl|397|UMGS1075|34|Collinsella-225/1   397            34           Collinsella   UMGS1075
```

`data.task: genus` selects `genus_class` (120 classes); `species` selects
`species_class` (1,535). Every result in [`RESULTS.md`](RESULTS.md) is genus-level.

`data.lazy: true` streams reads from the FASTA through a byte-offset index instead of
holding the pool in RAM. It is recommended for new training on 50M and 250M pools. Historical eager
configurations are preserved; keep their explicit split strategy when changing
the storage path. See [TRAINING.md](TRAINING.md).

## Provenance

**Gut catalogue** — 120 genera, 1,535 species, one representative genome per species,
drawn from the HGR/UMGS reference genomes of Almeida et al. 2019. Identifiers are a
mixture: RefSeq assemblies with binomial names (`GCF_*`) alongside MAG bins that resolve
only to genus (`UMGS*`, and numeric HGR bin ids such as `11861_6_55`). That mixture is
why the ANI ladder is defined by sequence distance rather than by species names — only
219 of the 1,535 carry a binomial.

### There is no "build the catalogue" script, because the labels *are* the catalogue

This is worth stating plainly, since its absence looks like a gap and is not one.
**The `species_name` column is the assembly's filename**, minus the `.fa.gz`. Nothing
has to be inferred: every row of every label file names the genome it came from, so the
species → genome mapping is read off directly.

```bash
python scripts/data/catalogue_manifest.py \
    --labels  $KF_DATA/val_100K/labels_100K_val.tsv \
    --genomes /path/to/reference_genomes \
    --out     catalogue_manifest.tsv
```

Verified against the original reference directory, which holds **2,505 candidate
assemblies** (1,952 `UMGS*`, 327 `GCF_*`, 226 numeric HGR bins):

- the labels select **1,535** of those 2,505, and **all 1,535 resolve by name** to an
  assembly on disk — zero misses;
- as a second, independent check, `species_class` also equals that assembly's index in
  the *sorted* file list, for all 1,535. Class ids were assigned in sorted-filename
  order, which follows from the first point rather than adding to it — but it means a
  manifest built by name and one built by index agree, and `catalogue_manifest.py`
  checks both so a mismatch would be caught;
- the selection is therefore a *sparse subset* of `0..2504`, not a renumbering, which
  is why the ids run to 2503 while numbering only 1,535 species. A species-level model
  reusing these ids needs **2,504** output classes; renumbering densely to `0..1534` is
  perfectly fine and saves the unused rows, but it makes the labels incompatible with
  any existing checkpoint, so nothing here does it;
- `genus_class` is dense, `0..119`. Every result in [`RESULTS.md`](RESULTS.md) is
  genus-level and uses those 120 classes.

Downstream of that, everything is here: `scripts/baselines/build_kraken2_db_1535.py`
takes the same `--labels` and `--genomes` pair to build the matched Kraken 2 index, and
`scripts/data/subsample_balanced.py` builds the read pools.

The one upstream step **not** in this project is which 1,535 of the 2,505 candidates were
chosen and in what order the class ids were assigned; that came from the original
MetaTransformer data preparation. It does not block reproduction, because the ids it
produced are recorded in the label files that ship with the data — and it is why
Almeida et al. 2019 is the correct citation for the genomes rather than this repo.

**Soil catalogue** — 309 genera from RefSoil (Choi et al. 2017), a curated collection of
soil isolate genomes. Reads are simulated identically and the same balanced pools built
at 5M and 50M, so the only thing that changes between environments is the catalogue.

**Reads** — 150 bp, simulated with ART (Huang et al. 2012): HiSeq 2500, paired-end,
insert 400±50, seed 42.

Those settings are not just documented, they are executable. The **evaluation** pools are
rebuilt from RefSeq assemblies by two pipelines in this repository —
[`scripts/track_a/`](../scripts/track_a/) (6 steps) and
[`scripts/track_b/`](../scripts/track_b/) (14 steps) — which download the genomes,
simulate the reads, run every arm and score them.
[`scripts/track_a/README.md`](../scripts/track_a/README.md) is the entry point, including
the external tools needed (`art_illumina`, `skani`, NCBI `datasets`, `kraken2`).

The **training** pool is the exception. Its 258.67M reads were simulated upstream, in the
data preparation this project inherited, and that driver is not here — the same boundary
as the species-selection step above. Everything downstream of it is:
`scripts/data/subsample_balanced.py` builds the balanced pools, and the two pipelines
build everything the models are tested on. The ART parameters are identical across all of
them, so a from-scratch regeneration has the settings it needs.

**Track A** — 595,000 reads from RefSeq genomes absent from the training reference,
filtered to 580,000 reads and 116 genera after removing three training-overlapping
accessions (`GCF_000010185`, `GCF_000158275`, `GCF_000312005`). It simulates one
assembly per genus, 119 in all, at 5,000 reads each. It guarantees unseen *assemblies*
only, and is a mixture of distances — see the caveat in [`RESULTS.md`](RESULTS.md#track-a-caveat).

**Track B** — an ANI-stratified ladder built with skani against all 1,535 training
genomes at the 95% species boundary GTDB uses. Of 601 candidates, 37 duplicates and
four cross-genus-ambiguous assemblies are dropped, leaving 560 genomes: 93 the same
species as a reference genome, 139 a different species of a represented genus, 328 with
no reliable same-genus hit under skani screening and alignment-fraction criteria,
and none above 99.9% ANI. The far rung does not establish measured ANI below 80%.
The construction uses `-s 70`, `-c 125` and minimum AF 15%; the cross-genus
leak exclusion threshold is 95% ANI. Fourteen genera are held present at
every rung so a line through them measures distance rather than class count.

**Real mocks** — ZymoBIOMICS D6331 (SRR33710519 and SRR33710518, ~135 bp) and the
Mori cell-mix (PRJDB10817, DRR466867, 150 bp; Mori et al. 2023,
https://doi.org/10.1093/dnares/dsad010). The historical `kim` directory name is an
internal alias. Mori contains 18 equal-cell bacterial strains; expected input DNA
is weighted by the published genome masses, with 17 in-catalogue genera covering
93.79% of expected DNA. These expectations are not post-extraction read-level truth.
Only composition metrics, not dummy-label read accuracy, are evaluated on the mocks.

## Duplication, and what it means for the accuracies

The earlier 42.64M/17.76M/41.94M counts counted read identifiers, not DNA
sequences, and do not measure sequence duplication. A complete 50M-read scan
on 2026-09-08 found 49,640,347 distinct sequence hashes (99.28%), using 128-bit
BLAKE2b on the sequence strings while preserving strand orientation. The corresponding
250M and 258.67M sequence-uniqueness counts have not been recomputed. These data do
not justify attributing the observed scaling plateau to duplicate sequence coverage.

ART numbers reads per reference genome, so identifiers collide across
independent simulation runs without denoting the same read: 174 of the closed-set test
set's identifiers appear among the 50M training reads, and at all 1,110 of their
occurrences the sequence differs. Those are collisions of name, not shared reads.

Sequence identity is the quantity that does bear on the results. Of the test set's
99,997 distinct sequences, 1,345 (1.345%) occur among the 50M training reads. They are
easier by a wide margin — the 6-mer depth series scores 80.4/86.0/88.4/89.7% on them
against 53.92/65.44/67.94/69.52% overall — but they are too few to matter in aggregate.
Excluding them shifts the four arms by −0.36, −0.28, −0.28 and −0.28 points, nearly
uniformly, preserving the ordering. The following scripts audit identifiers and
leftover coverage, not the sequence-overlap claim itself:
[`scripts/data/profile_leftover_coverage.py`](../scripts/data/profile_leftover_coverage.py)
and [`scripts/data/count_unique_seqids.py`](../scripts/data/count_unique_seqids.py).

## The 13-mer vocabulary

`vocab_13mer.txt` is a fixed-width plain-text list of the 33,545,099 canonical 13-mers
observed in the gut catalogue, one per line, the line number being the token id. It is
MetaTransformer's vocabulary, used unchanged so that the *k* = 13 head-to-head is matched
on vocabulary as well as on tokenizer.

It is **canonical**: it holds only one member of each reverse-complement pair, covering
49.99% of 4^13, with 0.00% of its *k*-mers having their RC also present, every entry
being the `min()` of its pair. `ExactKmerTokenizer` detects this by sampling and then
canonicalises every query code to `min(code, rc(code))` before lookup — without that,
about half of all real *k*-mers would hash to `UNK`. One consequence worth knowing: with a
canonical vocabulary a read and its reverse complement produce the *same* token ids in
reversed order, so RC augmentation becomes an order permutation rather than a content
change.

On first use the tokenizer builds an `int32` lookup table indexed by the 2-bit-packed
code — 4^13 × 4 bytes = 256 MiB, O(1), no Python dict — and caches it next to the vocab
file. Keep that cache on local disk. It is loaded fully into RAM on purpose: with
`mmap_mode="r"` on NFS every random *k*-mer lookup becomes a page fault, measured at
719 reads/s end-to-end against a 3,158 reads/s model ceiling. DataLoader workers fork
after the load, so copy-on-write shares one copy.

The hashed arms need none of this — `HashedKmerTokenizer` is a pure function of the
*k*-mer, seeded from the config, with no vocabulary file at all.

They also do **not** canonicalise, where the exact path does. In the hashed arms a
*k*-mer and its reverse complement occupy unrelated buckets; in the exact arms they
share one row. So the exact-vs-hashed comparison in
[`RESULTS.md §3`](RESULTS.md#3-part-3--at-13-mer-the-vocabulary-stops-being-free) varies
canonicalisation and embedding updates alongside the vocabulary. Exact embeddings
use SparseAdam without embedding weight decay/clipping; hashed embeddings use dense
AdamW. The 6.37-point difference is a complete-configuration trade-off, not a guaranteed
hashing-cost bound. At matched d64 it reduces the table from 8 GiB to 1 GiB;
the 2 GiB hashed table instead uses d128. The manuscript discloses these differences. It is recorded here
rather than corrected, because the arms were trained this way and changing the code now
would document a model that was never run.

## Availability

> **TODO** — Zenodo DOI for the read pools, evaluation sets and vocabulary, once the
> record is published. Model weights go in a separate record; see
> [`weights/README.md`](../weights/README.md).

[`MISSING_ASSETS.md`](MISSING_ASSETS.md) distinguishes bundled evidence, local-only
assets and unresolved provenance. On the machine
the runs were done on, `scripts/fetch/assemble_local_data.sh` builds the tree above
from where the pieces actually live.

The balanced pools can be built with
[`scripts/data/subsample_balanced.py`](../scripts/data/subsample_balanced.py) from ART
simulations over the catalogue. [`build_clean_test.py`](../scripts/data/build_clean_test.py)
and [`build_clean_pool.py`](../scripts/data/build_clean_pool.py) construct
identifier-filtered leftover pools from the source reads. They are **not** the
generator of the manuscript's independently simulated 100,000-read closed set.
That exact pool must be obtained with the frozen evaluation assets. The Nano4
audit recovered the original 99,742-read filter: excluding the union of 174
(50M) and 258 (250M) read IDs reconstructs the subset byte for byte. The former
list is a subset of the latter. An occurrence-level manifest, exclusion lists
and checksums accompany the manuscript under
`supplementary_data/nt_99742_provenance/`; do not substitute a newly generated
leftover pool or collapse repeated IDs.

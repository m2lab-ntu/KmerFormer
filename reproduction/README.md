# Frozen evidence catalogue

This directory packages existing prediction arrays, catalogue records and
recovered configurations. `manifest.json` records source and delivered hashes,
transformations, sizes and array accuracies. Packaging performs no training or
new model inference.

| Result | Configuration / record | Evidence |
|---|---|---|
| Gut primary closed-set comparison | [Result manifest](../docs/results_manifest.json) | [Primary arrays](../docs/assets/) |
| Tokenizer ranking at distance | [ANI-ladder catalogue](catalogues/track_b_ani_ladder.csv) | [Ladder arrays and tables](ladder/) |
| Soil supervised-data crossover | [Recovered soil configs](soil/configs/) | [Soil predictions](soil/predictions/) |
| NT-v2 50M/250M continuation | [Current config mapping](../docs/REPRODUCE.md) | [99,742-read series](nt_99742/) |
| Real mocks | [Mori composition](catalogues/mori_mock_expected.csv) | [Archived rescoring summaries](summaries/) |
| Sequence uniqueness | [Sequence recount](catalogues/sequence_uniqueness_50M.json) | 49,640,347 distinct strings; source record retained |

## Interpretation

The NT-v2 scaling pool contains 99,742 reads after the 258-read identifier
filter. It is separate from the primary 100K arrays and from the 1,345-sequence
overlap sensitivity analysis. Keep 64.84%, 67.12% and 67.29% associated with their
respective training recipes and archived evaluation series.

Ladder arrays retain their original row order and labels. The accompanying
catalogue and per-genus tables identify the shared fourteen-genus analysis.
Probability/logit matrices may be omitted from packaged arrays; the manifest
records that transformation. Source-family protocols must be retained when
comparing archives; a filename alone does not establish RC-TTA or precision.

Soil MetaTransformer controls are the **6-mer e128** arms at 5M and 50M.
Their configuration files retain the third-party MetaTransformer schema and
must be run with that implementation, not `kmerformer-train`. KmerFormer/NT-v2
soil configs use the package schema with normalized data paths. The recovered
source and delivered hashes record those path changes.

The Mori et al. 2023 mock is DRR466867, with pre-extraction DNA-mass truth.
It is separate from the D6331 mock records. These summaries preserve the
existing analysis and are not newly validated abundance estimates.

## Data delivery

Full training pools, reference genomes and their construction tools are
external assets. The catalogue, membership records and checksums identify the
inputs; they do not recover an absent upstream species-selection or simulation
driver. Use [DATA.md](../docs/DATA.md) for expected layouts and
[MISSING_ASSETS.md](../docs/MISSING_ASSETS.md) for remaining delivery limits.

Maintainers can repeat packaging with `scripts/weights/collect_evidence.py`,
passing explicit manuscript, experiment and backup roots. That collector never
modifies its sources.

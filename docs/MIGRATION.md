# Versions and historical experiments

The maintained `0.2` interface adds bundles, unlabelled prediction, streaming
independent evaluation and explicit splits. The preceding source snapshot is
`5a9870d46b614d05a121ca3309ef5400fa24cd7c`; snapshot
`9255f5ea3b7eb1080d5dd2524c3a7487f426d04e` predates the runtime repairs.
Neither is asserted to be the exact source of every historical run.

## Compatibility

- `shallow_transformer` configs and checkpoint tensor keys are retained;
  `kmerformer` is the public model-type name.
- Historical numbers, hyperparameters, data-loading mode and split algorithm
  are retained. Explicit split and unseeded-initialization fields describe them.
- DDP now consumes the configured split strategy. Historical DDP runs may have
  used the former unconditional random split.
- Independent evaluation streams by default. `--legacy-eager` retains the
  older evaluator for historical comparisons and experimental routing modes.
- Bundles use safetensors and checked tokenizer/class maps. Training recovery
  checkpoints retain PyTorch optimizer state.
- The default factory constructs KmerFormer. Foundation-model configs specify
  `model.type: gfm` and require the `baselines` extra.

## Reproducing a reported number

Use the [evidence catalogue](../reproduction/README.md) to identify checkpoint,
pool, config and series. Preserve precision, RC-TTA, aggregation, and the
100K versus 99,742-read distinction. Loading equivalence and software tests do
not replace these protocol requirements.

[history/ENVIRONMENT.md](history/ENVIRONMENT.md) preserves the old environment;
the supported installation is defined by `pyproject.toml`.

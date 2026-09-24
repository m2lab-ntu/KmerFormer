# Training KmerFormer

Use `model.type: kmerformer`, labelled FASTA/TSV inputs and a dedicated run
directory. `shallow_transformer` remains a historical alias.
See [data formats](DATA.md) and [examples](../examples/README.md).

## Split membership

`data.split_strategy` selects `stratified` or `random`. Both training entry
points call the same partition function. They create `split.npz` in the output
directory unless `data.split_manifest` names another path. It records ordered
train/validation indices, task, seed, ratio and full input-file hashes.
Input or protocol mismatches are rejected on reuse.

```yaml
data:
  lazy: true
  split_strategy: stratified
  split_manifest: /path/to/shared-split.npz
  seed: 42
training:
  seed: 123
```

The indexed loader reads multiline labelled FASTA on demand. Header class IDs
must agree with TSV rows in FASTA order. The in-memory loader also accepts
arbitrary FASTA identifiers joined through `seq_id`. Prediction and independent
evaluation support compressed FASTA/FASTQ; indexed training uses plain FASTA.

Historical configs retain their original strategies: stratified for the former
eager path and random for the former lazy path. Sharing a source pool does not
imply that historical comparisons shared partition membership.

## Class IDs

IDs must be nonnegative integers with one name per ID. Training preserves IDs
and allocates `max(class_id) + 1` outputs. Historical species IDs ending at 2503
therefore need 2,504 outputs despite representing 1,535 distinct species.
A new dataset can use a documented dense mapping. Independent evaluation
obtains output width from the checkpoint and retains that width for subsets.

## Seeds and recovery

`training.seed` controls Python, NumPy and PyTorch initialization. Worker seeds
derive from PyTorch. Historical configurations retain `training.seed: null`
because initialization was unseeded. A split seed alone does not seed training.
`run_provenance.json` records dependencies, source revision, config hash,
world size and seed.

`last.pt` saves optimizer, scheduler, AMP, early stopping and RNG state; DDP
records each rank's RNG. Epoch-boundary resume uses the same world size,
software, hardware and worker setup. Incomplete-epoch checkpoints restart the
epoch from saved weights and optimizer state; they do not reproduce an
uninterrupted trajectory. `--init_from` starts a new run from weights.

```bash
kmerformer-train --config your-config.yaml
kmerformer-train --config your-config.yaml --resume /path/to/run/last.pt
torchrun --nproc_per_node=2 -m kmerformer.train_ddp --config your-config.yaml
```

DDP supports genus classification with dense gradients. Sparse exact 13-mer
embeddings use the single-process trainer. DDP batch size is per rank;
identical partitions do not equate effective batches or optimization trajectories.

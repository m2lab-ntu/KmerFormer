# Examples

`reads.fa` contains the first 16 records in the primary gut closed-set FASTA.
`provenance.json` records source and example checksums. Selection preceded
inference. `labels.tsv` provides original truth for optional evaluation;
prediction requires only the FASTA.

```bash
kmerformer-predict --model models/6mer-l29 --reads examples/reads.fa \
  --output /tmp/kmerformer-predictions.tsv --device cpu
```

`expected_predictions.tsv` records FP32 CPU mean-logit RC-TTA on the 6-mer L29
model. Compare class IDs and use tolerance for scores across numerical
environments. This example is not a representative accuracy estimate.

For an installation-only synthetic training/checkpoint check:

```bash
python scripts/smoke_cpu.py
```

That check downloads no model and is separate from the trained demonstration.

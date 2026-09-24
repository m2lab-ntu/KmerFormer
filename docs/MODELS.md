# Choosing a KmerFormer model

| Model ID | Use case | FP32 weight size | Vocabulary |
|---|---|---:|---|
| `kmerformer-6mer-l29-50m` | Small starting model; stronger retention at distance in the ANI ladder | 25 MB | Fixed non-overlapping 6-mers |
| `kmerformer-exact13-1l-50m` | High closed-set accuracy with close catalogue coverage | 8 GiB | Exact canonical 13-mers |
| `kmerformer-exact13-1l-250m` | Exact model at the larger supervised-read budget | 8 GiB | Exact canonical 13-mers |
| `kmerformer-hashed13-d128-1l-50m` | Reduced vocabulary storage relative to exact 13-mers | 2 GiB | 2^22 hashed buckets |

Sizes exclude tokenizers, activations and runtime. Exact bundles also need their
vocabulary. Training adds optimizer and gradient storage; weight size is not a
training-memory estimate. `kmerformer-model list` reports bundle availability.

## Scope and score interpretation

The gut classifiers were trained on 150 bp simulated reads and assign one of
120 genus classes. `class_id` identifies the training catalogue, not NCBI taxonomy.
The label map is part of each bundle. Species training preserves a full raw-ID
output space; released genus models do not provide species classifications.

Prediction follows the bundle's RC-TTA setting and mean-logit aggregation.
The CLI defaults to FP32 with the unfused encoder and highest matrix-multiply
precision as its reference path. Faster BF16/FP16 inference is explicit.
Paper evaluation precision and throughput remain separate historical records.
Scores are softmax values over the catalogue.
Unknown-genus rejection and probability calibration have not been established.

The ANI ladder compares new genomes within the trained genus space. Its far
rung means no reliable same-genus hit under skani `-s 70 -c 125 --min-af 15`,
not ANI <80%. Track A mixes distances and is interpreted separately.

## Comparisons

Exact and hashed 13-mer configurations also differ in RC folding and embedding
updates. Their difference measures the complete-configuration trade-off, not a
bound on hashing alone. Compare using the recorded configuration and test pool.

Published model-only throughput measures forward computation at a specified
device, batch and precision. The prediction CLI times input parsing,
tokenization, inference and TSV writing, excluding model loading.

## Model cards

Bundles contain `manifest.json`, `config.json`, `labels.json`, tokenizer assets
and `model.safetensors`. Manifests record source checkpoint checksums, weight
licensing and inference protocol. The four weight files use Apache-2.0;
`WEIGHTS_LICENSE.txt` and `TOKENIZER_LICENSE.txt` keep weight and tokenizer
terms separate. The [weight guide](../weights/README.md)
records identities and the [evidence catalogue](../reproduction/README.md)
connects them to the paper results.

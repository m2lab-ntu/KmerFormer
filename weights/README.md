# Model bundles

KmerFormer inference bundles contain safetensors weights, a fixed tokenizer,
the full output-class map and a versioned inference protocol. They load through
`ModelBundle` and `kmerformer-predict` without downloading a foundation model.

## Availability

The release registry is [models.json](../kmerformer/assets/models.json).
`kmerformer-model list` adds an `availability` block to each entry: whether a
download is registered at all, and whether your credentials can use it.
`kmerformer-model list --check` goes further and asks the host, reporting the
HTTP status rather than inferring.

**The trained weights are not currently available for public download.** The
registry lists the four models with their checksums and licences but no download
parts, so `list` reports each as `unregistered` and `download` exits with that
explanation. The weights will be released upon acceptance of the paper and are
available to reviewers on request in the meantime.

They were briefly publicly downloadable, from Hugging Face between 2026-09-21
and 2026-09-25. Copies obtained then remain under the licences below; the
withdrawal stops further public distribution and does not revoke those grants. Bundles have been prepared
from the existing checkpoints. All four listed KmerFormer weight files are
licensed under Apache-2.0; tokenizer assets retain the terms listed below.

| Model ID | Source checkpoint | Weights | Closed set | Track A |
|---|---|---:|---:|---:|
| `kmerformer-6mer-l29-50m` | `kmerformer_6mer_L29_50M.pt` | 25 MB | 69.52% | 28.90% |
| `kmerformer-exact13-1l-50m` | `kmerformer_exact13mer_1L_50M.pt` | 8 GiB | 91.15% | 27.44% |
| `kmerformer-exact13-1l-250m` | `kmerformer_exact13mer_1L_250M.pt` | 8 GiB | 99.16% | 31.92% |
| `kmerformer-hashed13-d128-1l-50m` | hashed d128, one-layer 50M `best.pt` | 2 GiB | 86.80% | 23.37% |

Accuracies are existing paper results, not new estimates from bundle conversion.
Track A mixes reference distances and budgets; use the ANI ladder for the
matched retention comparison. Configurations and source records are linked in
[RESULTS](../docs/RESULTS.md) and [the evidence catalogue](../reproduction/README.md).
The [model guide](../docs/MODELS.md) explains model choice and scope.

## Download and predict

```bash
kmerformer-model download kmerformer-6mer-l29-50m --output models/6mer-l29
kmerformer-model verify models/6mer-l29
kmerformer-predict --model models/6mer-l29 --reads sample.fastq.gz \
  --output predictions.tsv --device cpu
```

The commands above apply once download parts are registered again. For a
GitHub asset that requires access, the downloader can use
`GH_TOKEN`/`GITHUB_TOKEN` or the existing GitHub CLI login; credentials are never
written into the registry or forwarded to redirected storage hosts. Other hosts
are reported as `unverified` until `list --check` confirms them.
A missing download entry is reported explicitly. Inference from a local bundle
uses no network access.

Each bundle defines 120 genus outputs, its token IDs, max token length and
mean-logit RC-TTA. The prediction CLI defaults to FP32. Paper precision and
aggregation distinctions remain in [REPRODUCE](../docs/REPRODUCE.md).
Output class IDs belong to the training catalogue, not the NCBI taxon namespace.

## Loading from Python

```python
from kmerformer import ModelBundle, iter_reads

model = ModelBundle.from_pretrained("models/6mer-l29", device="cpu")
for row in model.predict(iter_reads("sample.fa")):
    print(row)
```

Hashes are checked before loading. A mismatched tokenizer, label map, model
width or file checksum raises an error. Exact 13-mer bundles carry their
vocabulary; hashed bundles record bucket count and hash seed.

## Convert an existing checkpoint

Maintainers can package a trusted local KmerFormer checkpoint:

```bash
kmerformer-model export --checkpoint /path/to/best.pt \
  --config configs/6mer/L29_50M.yaml --labels /path/to/training-labels.tsv \
  --model-id kmerformer-6mer-l29-50m --output /path/to/new-bundle \
  --licence Apache-2.0
```

Supply the full trained class catalogue, not an evaluation subset. Existing
output directories are protected. Conversion retains tensor values and tensor
keys, removes optimizer state, normalizes paths and records the source checksum.
`scripts/weights/package_bundles.py` creates bounded download parts, with
checksums for every part and the complete archive. See [RELEASE.md](../RELEASE.md).
The example selects the licence approved for the four maintained bundles.
When exporting other checkpoints, select terms you are authorised to grant;
omitting `--licence` leaves their weight terms as `LicenseRef-Pending`.

For an existing verified bundle, `scripts/weights/license_bundles.py --licence
Apache-2.0 /path/to/bundle` adds the full licence texts and updates only the
manifest. Repackage its download parts afterwards so their hashes cover the
new metadata and licence files.

A checkpoint's stored best-validation accuracy can differ from its independent
closed-set score. The 6-mer source records 0.6928 and the 50M exact source 0.9111;
the table reports 69.52% and 91.15% under their separate test protocols.

## Licensing and attribution

The four maintained bundles license `model.safetensors` under **Apache-2.0**.
Each contains the full text in `WEIGHTS_LICENSE.txt`, and its manifest records
`weights_licence_scope: model.safetensors`. The repository's code remains MIT.
The NT-v2 vocabulary retains its upstream
[CC BY-NC-SA 4.0 attribution](../kmerformer/assets/NOTICE.md). Exact-vocabulary
provenance and [CC BY 4.0 terms](../kmerformer/assets/EXACT13_NOTICE.md) accompany
the exact model bundles. The upstream compressed-file checksum and decompressed
vocabulary SHA-256 were verified against Zenodo record 7594864.

`TOKENIZER_LICENSE.txt` contains the applicable tokenizer terms: CC BY-NC-SA
4.0 for the fixed NT-v2 asset, CC BY 4.0 for the exact vocabulary, and MIT for
the KmerFormer hashed tokenizer implementation. The 6-mer bundle includes a
NonCommercial tokenizer asset; the weight licence grants no additional rights
in that asset. [Licence sources and scope](../kmerformer/assets/licences/README.md)
are included in the source and installed wheel.

Foundation-model and MetaTransformer baseline weights belong to their source
projects and are separate from the KmerFormer bundles.

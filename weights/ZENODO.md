# Zenodo delivery for KmerFormer 0.2.0rc1

The delivery contains one source snapshot, its matching wheel, four runnable
model bundles, and supporting manifests. Prepare two linked Zenodo records:
`code/` for software and recorded evidence, and `weights/` for trained models.
Each record has its own citation metadata, inventory and checksums. The source ZIP already includes the
109 recovered evidence files, primary prediction arrays, training configs,
tests and the fixed 16-read example. Full raw training pools and reference
genomes remain external; see [evidence scope](../reproduction/README.md).

## Upload inventory

| File | Contents | Approximate size |
|---|---|---:|
| `kmerformer-code-<commit>.zip` | Every tracked file from the named clean commit, with file hashes and modes | 100 MB |
| `kmerformer-0.2.0rc1-py3-none-any.whl` | Installable library, CLIs, fixed tokenizer and current model registry | 0.1 MB |
| `kmerformer-6mer-l29-50m.tar` | 6-mer L29 model, tokenizer, 120-class map and inference protocol | 26 MB |
| `kmerformer-exact13-1l-50m.tar` | Exact 13-mer 50M model with its vocabulary and metadata | 9.06 GB |
| `kmerformer-exact13-1l-250m.tar` | Exact 13-mer 250M model with its vocabulary and metadata | 9.06 GB |
| `kmerformer-hashed13-d128-1l-50m.tar` | Hashed 13-mer model with tokenizer settings and metadata | 2.15 GB |
| `README.txt`, `VALIDATION.md`, `CITATION.cff` | Loading instructions, executed checks and citation metadata | Small |
| `DEPOSIT_MANIFEST.json`, `SHA256SUMS` | Source identity, asset sizes, licences and SHA-256 checksums | Small |

The complete delivery is approximately 20.4 GB. The separate-record layout has
seven files in `code/` and nine in `weights/`, plus a local delivery index.
Without `--separate-records`, the preparation tool retains the flat 11-file
layout for a single combined record. Sizes use decimal GB. Zenodo's documented
default limit per record is 50,000,000,000 bytes and 100 files, checked on 2026-09-18
([official file guidance](https://help.zenodo.org/docs/deposit/manage-files/)).
Standalone TAR files avoid requiring users to concatenate the 1 GiB parts used
for GitHub release assets. The reconstructed bytes retain the same archive hashes.

The three old `.pt` checkpoints alone are superseded as the delivery format.
Keep the original checkpoints as source provenance; users receive safetensors,
configuration, labels and tokenizer assets together. Printed paper results and
historical evaluation protocols are retained.

## Build from one source revision

After committing the intended source revision, build the wheel and code ZIP
from that same clean checkout. Use a fresh wheel output directory:

```bash
python -m pip wheel . --no-deps --wheel-dir /path/to/wheel-output
python scripts/package_reviewer.py --release --output /path/to/kmerformer-code-COMMIT.zip
python scripts/weights/prepare_zenodo.py \
  --code /path/to/kmerformer-code-COMMIT.zip \
  --wheel /path/to/wheel-output/kmerformer-0.2.0rc1-py3-none-any.whl \
  --parts /path/to/verified-github-model-parts \
  --output /path/to/new-zenodo-delivery --separate-records
```

`prepare_zenodo.py` refuses a stale code snapshot or wheel, checks each model
part and complete TAR hash, and creates the upload directory atomically.
It does not upload, reserve a DOI, publish a record or select a licence.
Install the wheel outside the checkout and run the trained example before
uploading. `sha256sum -c SHA256SUMS` verifies the delivery files after transfer.
Run it inside each record directory. Upload the contents of `code/` and
`weights/` into separate Zenodo drafts; the top-level `DELIVERY_MANIFEST.json`
is a local index. Link the records through Related identifiers after reserving
their real DOIs. Their manifests retain the compatible source commit.

## Metadata and publication state

The generated manifest describes software, trained model bundles and recorded
evidence. It copies creator metadata from `CITATION.cff`. The authors confirmed
the latest manuscript order on 2026-09-18: Ming-Ju Yang, TING-YU YEN, Chien-Yu
Chen and Joyce Tzu-Yu Liu. Only supplied affiliations and identifiers are included.
No tool identity is added as an author. No grant identifier, Zenodo DOI, record
number or publication date is invented.
The code record's `CITATION.cff` retains MIT for software; the weights record's
copy identifies the trained model collection and its Apache-2.0 weight terms.
Both preserve the four authors and the manuscript citation. Tokenizer terms
remain explicit in the record descriptions and model bundles.

Code retains MIT terms. The bundled NT-v2 vocabulary retains CC BY-NC-SA 4.0;
the MetaTransformer exact vocabulary retains CC BY 4.0. Their notices and full
licence texts remain inside the corresponding bundles. The four KmerFormer
weight files use **Apache-2.0**, recorded in `WEIGHTS_LICENSE.txt` and scoped to
`model.safetensors` in each manifest. Tokenizer terms are in
`TOKENIZER_LICENSE.txt`; the 6-mer asset retains its NonCommercial restriction.
The record's access setting and real DOI are selected in Zenodo, not
inferred from the private GitHub draft.

Before publishing, verify the uploaded file inventory against the local
`DEPOSIT_MANIFEST.json`. Local preparation and GitHub uploads do not establish
that a Zenodo record exists.

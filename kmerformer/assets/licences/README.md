# Licence texts for model delivery

The four KmerFormer release bundles license `model.safetensors` under
Apache-2.0. `WEIGHTS_LICENSE.txt` contains those terms, and the bundle manifest
records their scope. This licence applies to the KmerFormer weights only.

Tokenizer assets keep their own terms, with the full text in
`TOKENIZER_LICENSE.txt`:

- The fixed NT-v2 vocabulary retains CC BY-NC-SA 4.0 and its upstream attribution.
- The MetaTransformer exact 13-mer vocabulary retains CC BY 4.0 and its source record.
- The KmerFormer hashed tokenizer implementation retains the repository's MIT terms.

The 6-mer bundle therefore contains a NonCommercial tokenizer asset. The
Apache-2.0 weight licence grants no additional rights in that asset.
The repository's code remains MIT. Custom checkpoint export requires an
explicit weight licence; its default remains `LicenseRef-Pending`.

These unmodified texts were retrieved on 2026-09-18:

| File | Source |
|---|---|
| `Apache-2.0.txt` | https://www.apache.org/licenses/LICENSE-2.0.txt |
| `CC-BY-NC-SA-4.0.txt` | https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt |
| `CC-BY-4.0.txt` | https://creativecommons.org/licenses/by/4.0/legalcode.txt |
| `MIT.txt` | Copy of the repository's root `LICENSE` |

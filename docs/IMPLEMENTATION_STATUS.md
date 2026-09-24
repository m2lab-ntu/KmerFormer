# Usable release implementation

Work starts from `5a9870d46b614d05a121ca3309ef5400fa24cd7c`.
The manuscript and historical prediction arrays remain source evidence.

| Review item | Deliverable | Status |
|---|---|---|
| 1 | Unlabelled FASTA/FASTQ prediction CLI and API | Implemented; actual CPU/GPU example verified |
| 2 | Verified, portable model bundles and download registry | Four Apache-2.0 weight bundles with separate tokenizer terms; all 22 part hashes and URLs verified; primary download/inference verified |
| 3 | Shared, explicit single-process/DDP split contract | Implemented; both partition strategies tested |
| 4 | Bounded-memory evaluation and prediction | Implemented; metrics and saved-array equivalence tested |
| 5 | Validated class-ID space for genus and species | Raw ID gaps preserved; checkpoint-derived evaluation width tested |
| 6 | Consistent result records and executable examples | Canonical README records and 16-read reference supplied |
| 7 | Supported core environment; isolated historical baselines | CPU PyTorch 2.14 validated; NT-v2 and LoRA on Transformers 5.17 validated |
| 8 | User-first README and real-model demonstration | Implemented; wheel and downloaded primary bundle exercised |
| 9 | Model selection guide and model cards | Guide, per-model configs, provenance, labels and protocol manifests supplied |
| 10 | Standalone KmerFormer API and fixed tokenizer assets | Implemented; 1,004 tokenizer comparisons matched the original tokenizer |
| 11 | Training seeds, RNG checkpoints and run provenance | Implemented; single-process and two-rank CPU resume tested |
| 12 | Claim/config/asset manifests and recovered evidence | 109 evidence files recovered and verified; full upstream generation remains external |
| 13 | Historical snapshot and maintained-version migration guide | Source revisions, environment and two published snapshot tags preserved |
| 14 | Current configuration documentation and maintainer separation | Historical hyperparameters preserved; obsolete headers removed |
| 15 | CI, packaged-install checks and versioned release candidate | Remote CI passed; draft `v0.2.0rc1` assets, wheel, reviewer archive and download registry updated |

Availability and licensing statements describe actual supplied assets. Unknown
licences and archive identifiers remain unresolved until the authors provide them.

[Validation record](validation/0.2.0rc1.md) states execution scope and remaining
release work. Local software checks do not constitute a full training rerun.

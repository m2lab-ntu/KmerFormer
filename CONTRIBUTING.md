# Development

Install `.[review,plots]` in a fresh environment and run:

```bash
python -m pytest tests -q
python scripts/check_doc_links.py
python scripts/review_predictions.py
python scripts/update_result_tables.py --check
```

Supported interfaces are `kmerformer-predict`, `kmerformer-model`,
`kmerformer-train`, `kmerformer-eval`, and Python `ModelBundle`.
CI checks an installed wheel outside the source tree. Optional baseline and
CUDA tests report execution or skips explicitly.

| Area | Purpose |
|---|---|
| `kmerformer/` | Supported library and commands |
| `examples/` | Public examples and expected outputs |
| `configs/` | Paper configurations and recorded results |
| `reproduction/` | Evidence, provenance and source configs |
| `scripts/data`, `scripts/eval`, `scripts/track_*`, `scripts/baselines`, `scripts/bench` | Paper pipelines and comparison tools |
| `scripts/fetch`, `scripts/weights`, `scripts/package_reviewer.py` | Maintainer acquisition and delivery |

Machine-specific fetch tools need the source systems; downloaded bundles do
not depend on them. New usage examples should use portable paths and explicit
metadata. Current config headers describe present contents; Git preserves
historical research notes.

## Release

Retain source snapshots, validate a clean installation, install a built wheel,
run the actual model example and verify downloaded checksums.
[RELEASE.md](RELEASE.md) describes code, model, evidence and access checks.
Keep result values associated with their pool and protocol; update canonical
records and regenerate linked tables together.

# Release procedure

The source revision, wheel, model bundle and evidence collection each have an
identity. Keep them linked in the release notes and registry. A software version
does not silently replace a historical paper protocol.

1. Run the CPU suite, optional baseline tests and available CUDA tests. Record
   the actual device and distinguish CPU Gloo from multi-GPU NCCL coverage.
2. Build and install the wheel outside the checkout; run command help, a small
   checkpoint round trip and the trained 6-mer example.
3. Verify model manifests, source checkpoint hashes, class maps and fixed
   tokenizer IDs. Tensor-only bundles contain no training optimizer state.
4. Assemble model archives and record the checksum of every part. Large exact
   models may use multiple download parts; the downloader verifies both the
   parts and reconstructed archive before extraction.
5. Verify access and recorded licensing. Keep an unresolved weight licence or
   archive identifier explicit. Private repository access remains private unless
   the owner changes it; a local model directory is not a download URL.
6. Commit reviewed source changes, retain historical snapshot references and
   tag the release candidate. Publish the tested wheel and model assets with
   their matching registry. Run the model smoke workflow against downloaded assets.
7. Extract the reviewer code archive and run its checks. Its source must match
   the release commit. State the available evidence and any remaining upstream
   data-generation gaps in the release notes.

Use the [implementation record](docs/IMPLEMENTATION_STATUS.md) and
[evidence catalogue](reproduction/README.md) to track delivery status. The old
environment is preserved as [historical documentation](docs/history/ENVIRONMENT.md),
not as the supported install specification.

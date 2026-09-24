#!/usr/bin/env bash
# Assemble the release weight bundle in weights/.
#
# The three arms below are the ones every headline number in the paper rests on.
# They are copied, not moved: the originals stay where the runs left them.
#
#   usage: scripts/weights/collect_weights.sh [DEST]
#          DEST defaults to ./weights
#
# Override any source with an environment variable of the same name if your
# checkpoints live elsewhere:
#
#   L29_6MER=/path/best.pt scripts/weights/collect_weights.sh
#
# Total is ~16 GiB, which fits one Zenodo record (50 GB). The two 13-mer files
# are 8 GiB each because 99.99% of their parameters are the embedding table;
# there is no smaller form that can still run inference, since the table *is*
# the model's knowledge of 13-mers.
set -euo pipefail

DEST="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/weights}"

# Machine-specific source locations are not kept in the repository. Put them in
# scripts/local_paths.sh (git-ignored) as plain assignments, or export them.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$_here/../local_paths.sh" ]] && source "$_here/../local_paths.sh"

# arm name -> source checkpoint; unset means "not configured" and is reported
: "${L29_6MER:=}"
: "${EXACT13_1L_50M:=}"
: "${EXACT13_1L_250M:=}"

# release filename | source | config it belongs to
ARMS=(
  "kmerformer_6mer_L29_50M.pt|${L29_6MER}|configs/6mer/L29_50M.yaml"
  "kmerformer_exact13mer_1L_50M.pt|${EXACT13_1L_50M}|configs/13mer/exact_1L_50M_meanpool.yaml"
  "kmerformer_exact13mer_1L_250M.pt|${EXACT13_1L_250M}|configs/13mer/exact_1L_250M_meanpool.yaml"
)

mkdir -p "$DEST"
echo "destination: $DEST"
echo

missing=0
for arm in "${ARMS[@]}"; do
  IFS='|' read -r name src cfg <<<"$arm"
  if [[ ! -f "$src" ]]; then
    echo "MISSING  $name"
    echo "         expected at: ${src:-(not configured -- set it in scripts/local_paths.sh)}"
    missing=1
    continue
  fi
  echo "copying  $name  <- $src"
  # -c so a re-run that finds an identical file does no I/O; this matters when
  # the source is on NFS and the file is 8 GiB.
  rsync -h --progress --checksum "$src" "$DEST/$name"
  echo "         config: $cfg"
done

if [[ $missing -ne 0 ]]; then
  echo
  echo "One or more checkpoints were not found. Set the matching variable and re-run."
  exit 1
fi

echo
echo "Writing checksums (this reads ~16 GiB and takes a few minutes)..."
( cd "$DEST" && sha256sum ./*.pt > SHA256SUMS )

echo
echo "Bundle contents:"
ls -lh "$DEST"
echo
echo "Verify what you have:"
echo "  python scripts/weights/inspect_checkpoint.py $DEST/*.pt"
echo "  (cd $DEST && sha256sum -c SHA256SUMS)"

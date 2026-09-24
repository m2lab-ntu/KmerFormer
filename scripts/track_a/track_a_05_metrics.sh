#!/bin/bash
# Track A – Step 5: Compute per-read Top-1 and sample-level abundance metrics.
#
# Usage (from repo root):
#   bash local_realworld_eval/track_a_05_metrics.sh
#
# Prints a comparison table of new-genome vs closed-set accuracy.
# Writes abundance scatter plots to out/<model>/sample/

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATA_ROOT="${KF_TRACK_A:-${DATA_ROOT:-${KF_OUT:?set KF_OUT or KF_TRACK_A}/track_a}}"
OUT="${DATA_ROOT}/out"
SCRIPTS="scripts/eval"
PYTHON="${KF_PYTHON:-python}"

# ── helper: per-read Top-1 ────────────────────────────────────────────────────
top1() {
    local npz="$1"
    "$PYTHON" -c "
import sys, numpy as np
d = np.load(sys.argv[1], allow_pickle=False)
preds  = d.get('preds',  d.get('predictions', None))
labels = d.get('labels', d.get('targets',     None))
if preds is None or labels is None:
    print('keys:', list(d.keys()))
else:
    print(f'{(preds==labels).mean()*100:.2f}%  ({(preds==labels).sum():,}/{len(labels):,})')
" "$npz"
}

# ── per-read accuracy ─────────────────────────────────────────────────────────
echo "================================================================"
echo "  Per-read Top-1 accuracy on NEW genomes (Track A)"
echo "================================================================"
echo ""
printf "  %-25s  %s\n" "Model" "New-genome Top-1"
printf "  %-25s  %s\n" "─────────────────────────" "────────────────"

for tag in mt_250M mt_50M nt_v9; do
    npz="${OUT}/${tag}/preds.npz"
    if [[ ! -f "${npz}" ]]; then
        printf "  %-25s  %s\n" "${tag}" "preds.npz not found – run step 4 first"
    else
        acc=$(top1 "${npz}")
        printf "  %-25s  %s\n" "${tag}" "${acc}"
    fi
done

echo ""
echo "  Closed-set reference (same-genome, from THESIS_NUMBERS.md):"
echo "    MT 13-mer 250M  →  98.7%  (clean_common)"
echo "    MT 13-mer 50M   →  87.5%"
echo "    NT-v2 v9 50M    →  67.1%  (RC-TTA)"
echo ""

# ── sample-level abundance ────────────────────────────────────────────────────
echo "================================================================"
echo "  Sample-level abundance (Pearson r / Bray-Curtis)"
echo "================================================================"
echo ""
for tag in mt_250M mt_50M nt_v9; do
    npz="${OUT}/${tag}/preds.npz"
    if [[ ! -f "${npz}" ]]; then
        echo "  [${tag}] skipped (no preds.npz)"
        continue
    fi
    echo "  [${tag}]"
    "$PYTHON" "${SCRIPTS}/evaluate_sample.py" \
        --predictions "${npz}" \
        --out_dir     "${OUT}/${tag}/sample" \
        --exp_name    "${tag}_newgenome" \
        --reads_per_sample 10000 \
        --n_partition_samples 50 \
        --n_sparse_samples 100
    echo ""
done

echo "Scatter plots → ${OUT}/<model>/sample/abundance_scatter.png"

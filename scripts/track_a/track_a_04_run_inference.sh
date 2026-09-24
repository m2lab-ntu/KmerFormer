#!/bin/bash
# Track A – Step 4: Run inference with all three model families.
# Run from the repo root:
#   cd <this repository>
#   bash local_realworld_eval/track_a_04_run_inference.sh
#
# Prerequisites (all under local_realworld_eval/):
#   assets/mt_13mer_250M/{checkpoints/classification_transformer_ckpt_best.pt, config.yaml}
#   assets/mt_13mer_50M/{checkpoints/classification_transformer_ckpt_best.pt, config.yaml}
#   assets/nt_v9/nt_token_genus_v9_50M_best.pt
#   assets/vocab_13mer.txt
#   assets/MetaTransformer_src/
#   assets/labels_250M.tsv
#   test_data/newgenome_test.fa          (from step 3)
#   test_data/newgenome_test_labels.tsv  (from step 3)
#   test_data/val_dir/newgenome_test.fa  (symlink; used by MT --val_dir)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Inherit DATA_ROOT from parent (run_track_a.sh) or use default
DATA_ROOT="${KF_TRACK_A:-${DATA_ROOT:-${KF_OUT:?set KF_OUT or KF_TRACK_A}/track_a}}"
ASSETS="${DATA_ROOT}/assets"
DATA="${DATA_ROOT}/test_data"
OUT="${DATA_ROOT}/out"
SCRIPTS="scripts/eval"

TEST_FA="${DATA}/newgenome_test.fa"
TEST_TSV="${DATA}/newgenome_test_labels.tsv"
TRAIN_TSV="${ASSETS}/genus_map.tsv"
VAL_DIR="${DATA}/val_dir"
VOCAB="${ASSETS}/vocab_13mer.txt"
MT_SRC="${ASSETS}/MetaTransformer_src"

# ── sanity checks ────────────────────────────────────────────────────────────
check_file() { [[ -f "$1" ]] || { echo "MISSING: $1"; exit 1; }; }
check_file "${TEST_FA}"
check_file "${TEST_TSV}"
check_file "${TRAIN_TSV}"   # genus_map.tsv (120 rows, fast)
check_file "${VOCAB}"
check_file "${ASSETS}/mt_13mer_250M/checkpoints/classification_transformer_ckpt_best.pt"
check_file "${ASSETS}/mt_13mer_50M/checkpoints/classification_transformer_ckpt_best.pt"
check_file "${ASSETS}/nt_v9/nt_token_genus_v9_50M_best.pt"

# MT inference uses the MetaTransformer conda env (has all transitive deps).
# NT-v2 inference uses gfm-local.
MT_PYTHON="${MT_PYTHON:-python}"   # MetaTransformer runs in its own env
NT_PYTHON="${KF_PYTHON:-python}"

# ── MT 13-mer 250M ───────────────────────────────────────────────────────────
echo "================================================================"
echo " [1/3]  MT 13-mer 250M inference"
echo "================================================================"
mkdir -p "${OUT}/mt_250M"
PYTHONPATH="${MT_SRC}" ${MT_PYTHON} "${SCRIPTS}/extract_mt_predictions.py" \
    --exp_dir  "${ASSETS}/mt_13mer_250M" \
    --val_dir  "${VAL_DIR}" \
    --vocab    "${VOCAB}" \
    --out      "${OUT}/mt_250M/preds.npz" \
    --class_indices 3 \
    --batch_size 1024
echo ""

# ── MT 13-mer 50M ────────────────────────────────────────────────────────────
echo "================================================================"
echo " [2/3]  MT 13-mer 50M inference"
echo "================================================================"
mkdir -p "${OUT}/mt_50M"
PYTHONPATH="${MT_SRC}" ${MT_PYTHON} "${SCRIPTS}/extract_mt_predictions.py" \
    --exp_dir  "${ASSETS}/mt_13mer_50M" \
    --val_dir  "${VAL_DIR}" \
    --vocab    "${VOCAB}" \
    --out      "${OUT}/mt_50M/preds.npz" \
    --class_indices 3 \
    --batch_size 1024
echo ""

# ── NT-v2 genus v9 (RC-TTA) ──────────────────────────────────────────────────
echo "================================================================"
echo " [3/3]  NT-v2 v9 50M inference (RC-TTA)"
echo "================================================================"
mkdir -p "${OUT}/nt_v9"
${NT_PYTHON} "${SCRIPTS}/run_genus_rctta.py" \
    --config      configs/baselines/ntv2_lora_50M.yaml \
    --checkpoint  "${ASSETS}/nt_v9/nt_token_genus_v9_50M_best.pt" \
    --test_fasta  "${TEST_FA}" \
    --test_labels "${TEST_TSV}" \
    --train_labels "${TRAIN_TSV}" \
    --out_dir      "${OUT}/nt_v9" \
    --batch_size 256
ln -sfn rctta.npz "${OUT}/nt_v9/preds.npz"
echo ""

echo "================================================================"
echo " All inference done.  Results in: ${OUT}/"
echo "================================================================"

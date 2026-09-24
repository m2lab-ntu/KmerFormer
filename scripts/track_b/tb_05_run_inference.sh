#!/bin/bash
# Track B – Step 5: run all four neural models on one ladder level.
#
#   bash tb_05_run_inference.sh L2_species
#
# Reuses Track A's assets and the existing inference scripts unchanged, so the
# Track A / Track B numbers stay protocol-comparable.  Kraken2 is handled
# separately by tb_06.
#
# Models that already produced predictions are skipped, so an interrupted run
# can be resumed without redoing hours of work (this box also hosts a
# long-running depth_sweep training job and gets memory-tight).  FORCE=1 reruns.
set -euo pipefail

LEVEL="${1:?usage: tb_05_run_inference.sh <level>   e.g. L2_species}"

REPO="${KF_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TB="${KF_TRACK_B:?set KF_TRACK_B}"
ASSETS="${KF_TRACK_A:?set KF_TRACK_A}/assets"
MT6_EXP="${MT6_EXP:?set MT6_EXP to the MetaTransformer 6-mer experiment dir}"
VOCAB6="${MT6_VOCAB:?set MT6_VOCAB to the MetaTransformer vocab_6mer.txt}"

DATA=$TB/test_data
OUT=$TB/out/$LEVEL
LOGS=$TB/logs
mkdir -p "$OUT" "$LOGS"

TEST_FA=$DATA/$LEVEL.fa
TEST_TSV=$DATA/${LEVEL}_labels.tsv
VAL_DIR=$DATA/$LEVEL/val_dir

MT_PY="${MT_PYTHON:-python}"
NT_PY="${KF_PYTHON:-python}"

for f in "$TEST_FA" "$TEST_TSV" "$ASSETS/genus_map.tsv" "$ASSETS/vocab_13mer.txt" "$VOCAB6"; do
    [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }
done
[[ -d "$VAL_DIR" ]] || { echo "MISSING: $VAL_DIR"; exit 1; }

cd "$REPO"
echo "### Track B inference — level=$LEVEL  reads=$(grep -c '^>' "$TEST_FA")"

skip() {
    if [[ "${FORCE:-0}" != "1" && -s "$OUT/$1/preds.npz" ]]; then
        echo "===== skip $1 (preds.npz exists) ====="
        return 0
    fi
    return 1
}

run_mt() {   # run_mt <name> <exp_dir> <vocab> <logsuffix>
    mkdir -p "$OUT/$1"
    PYTHONPATH="$ASSETS/MetaTransformer_src" "$MT_PY" scripts/eval/extract_mt_predictions.py \
        --exp_dir "$2" --val_dir "$VAL_DIR" --vocab "$3" \
        --out "$OUT/$1/preds.npz" --class_indices 3 --batch_size 1024 \
        2>&1 | tee "$LOGS/${LEVEL}_$4.log"
}

skip mt_250M || { echo "===== [1/4] MT 13-mer 250M ====="
    run_mt mt_250M "$ASSETS/mt_13mer_250M" "$ASSETS/vocab_13mer.txt" mt250M; }

skip mt_50M || { echo "===== [2/4] MT 13-mer 50M ====="
    run_mt mt_50M "$ASSETS/mt_13mer_50M" "$ASSETS/vocab_13mer.txt" mt50M; }

skip mt_6mer || { echo "===== [3/4] MT 6-mer 50M (stride 1) ====="
    run_mt mt_6mer "$MT6_EXP" "$VOCAB6" mt6mer; }

skip nt_v9 || { echo "===== [4/4] NT-v2 v9 50M (RC-TTA) ====="
    mkdir -p "$OUT/nt_v9"
    "$NT_PY" scripts/eval/run_genus_rctta.py \
        --config configs/baselines/ntv2_lora_50M.yaml \
        --checkpoint "$ASSETS/nt_v9/nt_token_genus_v9_50M_best.pt" \
        --test_fasta "$TEST_FA" --test_labels "$TEST_TSV" \
        --train_labels "$ASSETS/genus_map.tsv" \
        --out_dir "$OUT/nt_v9" --batch_size 256 \
        2>&1 | tee "$LOGS/${LEVEL}_ntv9.log"
    # run_genus_rctta.py names its output rctta.npz; expose it under the same
    # preds.npz name the others use so tb_07 needs no per-model special case.
    ln -sf rctta.npz "$OUT/nt_v9/preds.npz"; }

echo "### done — $OUT"

#!/bin/bash
# Track B – sequential driver for the core experiments (E39-E43).
#
# Everything runs one job at a time on purpose: this box also hosts a
# long-running depth_sweep training job holding ~17 GB, and Kraken2's 1,535
# genome DB needs ~9 GB resident.  Running the neural models and Kraken2
# concurrently drove the machine into IO-wait thrashing, so they are serialised.
#
#   bash tb_run_all.sh              # neural -> kraken2 -> metrics -> figures
#   SKIP_NB=0 bash tb_run_all.sh    # also run the k-mer NB baselines (slow)
set -euo pipefail

TB="${KF_TRACK_B:?set KF_TRACK_B}"
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${KF_PYTHON:-python}"
# kraken2 and bracken are env binaries, not importable from $PY alone.
# art_illumina, skani, kraken2 and NCBI datasets must be on PATH
LEVELS="${LEVELS:-L1_strain L2_species L2_far}"
SKIP_NB="${SKIP_NB:-1}"

cd "$TB"
mkdir -p logs results figures

for L in $LEVELS; do
    echo "################ neural: $L ################"
    bash "$SCRIPTS/tb_05_run_inference.sh" "$L"
done

for L in $LEVELS; do
    echo "################ kraken2: $L ################"
    # Gate on the stats file, not preds.npz: a run that died after writing
    # predictions but before the summary is not a finished run.
    if [[ -s "out/$L/kraken2/kraken_stats.json" ]]; then
        echo "skip $L (preds.npz exists)"
        continue
    fi
    "$PY" "$SCRIPTS/tb_06_run_kraken2.py" --level "$L" \
        --pool_fa "test_data/$L.fa" --pool_labels "test_data/${L}_labels.tsv" \
        --out_dir "out/$L/kraken2" --threads 8 \
        2>&1 | tee "logs/tb_06_kraken2_$L.log"
done

if [[ "$SKIP_NB" != "1" ]]; then
    # One pass over the 50M reference pool per k, scoring every level at once.
    for K in 13 6; do
        echo "################ ${K}-mer NB (all levels) ################"
        TEST_ARGS=()
        for L in $LEVELS; do TEST_ARGS+=(--test "$L=test_data/$L.fa"); done
        # 20K reads/level: the 13-mer count table is (distinct test k-mers x 120)
        # float32, and only ~8 GB of RAM is free next to the depth_sweep job.
        "$PY" "$SCRIPTS/tb_10_kmer_nb.py" --k "$K" "${TEST_ARGS[@]}" \
            --out_dir out --subsample 20000 \
            2>&1 | tee "logs/tb_10_nb_${K}mer.log"
    done
fi

echo "################ metrics ################"
"$PY" "$SCRIPTS/tb_07_metrics.py" --track_b_dir . --out_dir results \
    --levels "$(echo $LEVELS | tr ' ' ',')" \
    2>&1 | tee logs/tb_07_metrics.log

echo "################ figures ################"
"$PY" "$SCRIPTS/tb_11_plots.py" --results_dir results --out_dir figures

echo "################ done ################"

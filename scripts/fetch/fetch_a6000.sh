#!/usr/bin/env bash
# Pull the artefacts that exist only on the A6000 box.
#
#   scripts/fetch/fetch_a6000.sh                      # dry run: report what is there
#   scripts/fetch/fetch_a6000.sh --apply              # metrics + histories, a few MiB
#   scripts/fetch/fetch_a6000.sh --apply --with-predictions   # + predictions.npz (~2 GiB)
#   scripts/fetch/fetch_a6000.sh --apply --with-250m-pool     # + the 67 GiB read pool
#
# What is here and why: the two headline 13-mer numbers (91.15% closed set, 99.16%
# at 250M) were produced on this box, and their eval metrics and training curves
# were never copied down. The checkpoints were -- see weights/README.md -- so this
# is corroboration, not a dependency.
#
# Default is a dry run. Nothing large is fetched without an explicit flag, because
# the 250M pool is 67 GiB and is needed only to RETRAIN a 250M arm; the trained
# checkpoint is present locally; reviewer download access is still pending.
set -uo pipefail

APPLY=0; PREDS=0; POOL=0
for a in "$@"; do
  case "$a" in
    --apply) APPLY=1 ;;
    --with-predictions) PREDS=1 ;;
    --with-250m-pool) POOL=1 ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

: "${A6000:=a6000}"          # ssh alias, override with A6000=<host>
: "${REMOTE_OUT:=depth_sweep_out}"
: "${REMOTE_DATA:=data}"
: "${DEST:=${KF_OUT:-./fetched}}"

RUNS=(
  "shallow1_exact13mer_50M_meanpool|exact 13-mer 1L 50M -- the 91.15% arm"
  "shallow1_exact13mer_250M_meanpool|exact 13-mer 1L 250M -- the 99.16% arm"
  "shallow1_exact13mer_50M|exact 13-mer 1L 50M, attention pooling -- 90.74%"
  "L16_5M_meanpool|k=6 pooling ablation, 54.29% -- no local output directory exists"
  "L29|6-mer 29 layers, 69.52% -- also local; fetch to cross-check"
)

echo "remote      : ${A6000}:~/${REMOTE_OUT}"
echo "destination : ${DEST}"
echo "mode        : $([[ $APPLY -eq 1 ]] && echo applying || echo 'DRY RUN -- pass --apply')"
echo

if ! ssh -o ConnectTimeout=15 -o BatchMode=yes "$A6000" true 2>/dev/null; then
  cat >&2 <<EOF
Cannot reach '$A6000' over ssh without a password.

The runs used an ~/.ssh/config alias 'a6000' reached through a jump host. If you
do not have it, ask whoever does to push the artefacts out instead -- see
docs/MISSING_ASSETS.md for exactly which files are wanted and which to skip.
Override the alias with A6000=<host> if yours is named differently.
EOF
  exit 1
fi

# Small artefacts: everything except the checkpoints and predictions.
SMALL=(--include='*/' --include='training_history.csv' --include='config.yaml'
       --include='classification_report.csv' --include='metrics.json'
       --include='eval_metrics*.json' --include='resource_report.json'
       --exclude='*')

for run in "${RUNS[@]}"; do
  IFS='|' read -r name note <<<"$run"
  remote_path="\$HOME/${REMOTE_OUT}/${name}"
  if ! ssh "$A6000" "test -d $remote_path" 2>/dev/null; then
    printf "  [absent  ] %-36s not on the remote\n" "$name"; continue
  fi
  size=$(ssh "$A6000" "du -sh $remote_path 2>/dev/null | cut -f1")
  printf "  [ found  ] %-36s %6s  %s\n" "$name" "$size" "$note"
  if [[ $APPLY -eq 1 ]]; then
    mkdir -p "$DEST/$name"
    rsync -az --info=progress2 "${SMALL[@]}" \
      "$A6000:${REMOTE_OUT}/${name}/" "$DEST/$name/"
    if [[ $PREDS -eq 1 ]]; then
      rsync -az --info=progress2 --include='*/' --include='predictions.npz' \
        --exclude='*' "$A6000:${REMOTE_OUT}/${name}/" "$DEST/$name/"
    fi
  fi
done

echo
if [[ $POOL -eq 1 ]]; then
  echo "250M read pool (67 GiB) -> ${KF_DATA:?set KF_DATA for the pool}/balanced_250M"
  if [[ $APPLY -eq 1 ]]; then
    mkdir -p "$KF_DATA/balanced_250M"
    rsync -az --info=progress2 --partial \
      "$A6000:${REMOTE_DATA}/balanced_250M/" "$KF_DATA/balanced_250M/"
  fi
else
  echo "250M read pool: skipped (67 GiB). Add --with-250m-pool only if retraining;"
  echo "                the trained 250M checkpoint is local; download access is pending."
fi
[[ $PREDS -eq 0 ]] && echo "predictions.npz: skipped (~2 GiB). Add --with-predictions to include."
[[ $APPLY -eq 0 ]] && { echo; echo "Re-run with --apply to fetch."; }

if [[ $APPLY -eq 1 ]]; then
  cat <<'EOF'

------------------------------------------------------------------------------
BEFORE COMPARING THESE AGAINST docs/RESULTS.md -- two traps that make a correct
number look wrong. Details and the full table: docs/MISSING_ASSETS.md.

1. eval_track_a/ was run on the RAW 595,000-read / 119-genome pool. The paper
   reports the 580,000 / 116-genome pool, after removing the three training-
   overlapping accessions, so the 13-mer arms read ~1.5 points HIGHER here.
   Multiply micro_accuracy by each candidate pool size: only the right one
   gives an integer read count.
   Do NOT equate that ~1.5 with the paper's 1.13/1.75 near-identical-genome
   figures -- those measure a DISJOINT set of six other genomes. Same
   mechanism, different subset; the magnitudes must not be reconciled.

2. The attention-pooling arm has TWO closed-set eval dirs. eval_final_rctta is
   the reported one (90.74%); eval_clean_common_rctta is an earlier evaluation
   (90.12%). Check checkpoint_epoch in the JSON.

3. metrics.json and resource_report.json are ABSENT for nine runs, and their
   train.log does not name the best epoch. They were never written: an earlier
   train.py crashed building the summary. Read the peak val_acc row of
   training_history.csv, or checkpoint_epoch in the eval JSON, instead. Full
   list and explanation in docs/MISSING_ASSETS.md.
------------------------------------------------------------------------------
EOF
fi

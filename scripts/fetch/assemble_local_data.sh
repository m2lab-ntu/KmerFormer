#!/usr/bin/env bash
# Build a single $KF_DATA tree from the places these assets actually live on this
# machine. Symlinks by default -- the read pools are ~13 GiB and there is no reason
# to duplicate them.
#
#   scripts/fetch/assemble_local_data.sh                 # dry run: report only
#   KF_DATA=/path scripts/fetch/assemble_local_data.sh --apply
#   KF_DATA=/path scripts/fetch/assemble_local_data.sh --apply --copy
#
# Dry run is the default on purpose: running it to see what is present costs
# nothing and touches nothing. Sources can each be overridden by an env var of the
# name shown in the table below, so this works on a machine laid out differently.
set -uo pipefail

APPLY=0; MODE=symlink
for a in "$@"; do
  case "$a" in
    --apply) APPLY=1 ;;
    --copy)  MODE=copy ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

: "${KF_DATA:?set KF_DATA to the destination data root}"

# Machine-specific source locations are not kept in the repository. Put them in
# scripts/local_paths.sh (git-ignored) as plain assignments, or export them.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$_here/../local_paths.sh" ]] && source "$_here/../local_paths.sh"

# Unset sources are reported as not configured rather than guessed.
: "${SRC_50M:=}" "${SRC_5M:=}" "${SRC_1M:=}" "${SRC_500K:=}" "${SRC_VAL100K:=}"
: "${SRC_TRACK_A:=}" "${SRC_TRACK_B:=}" "${SRC_VOCAB_DIR:=}" "${SRC_GENOMES:=}"

# dest-relative | variable holding the source | note
ITEMS=(
  "balanced_50M|SRC_50M|main training pool (reads_50M.fa, labels_50M.tsv)"
  "balanced_5M|SRC_5M|5M pool -- the data-scale crossover"
  "balanced_1000000|SRC_1M|1M pool"
  "balanced_500000|SRC_500K|500K pool"
  "val_100K|SRC_VAL100K|closed-set test set, 100,000 reads"
  "track_a|SRC_TRACK_A|out-of-genome pool, 580,000 reads"
  "track_b|SRC_TRACK_B|ANI ladder: L1_strain, L2_species, L2_far, L2_lognormal"
  "reference_genomes|SRC_GENOMES|2,505 candidate assemblies (Kraken 2 index, catalogue manifest)"
)

echo "destination : $KF_DATA"
echo "mode        : $MODE $([[ $APPLY -eq 1 ]] && echo '(applying)' || echo '(DRY RUN -- pass --apply)')"
echo

missing=0; ready=0; todo=0
for item in "${ITEMS[@]}"; do
  IFS='|' read -r dest var note <<<"$item"
  src="${!var}"
  target="$KF_DATA/$dest"
  if [[ -e "$target" ]]; then
    printf "  [in place] %-20s %s\n" "$dest" "$note"; ready=$((ready+1)); continue
  fi
  if [[ ! -d "$src" ]]; then
    printf "  [MISSING ] %-20s source not found: %s\n" "$dest" "${src:-(not configured)}"
    # The variable name travels with the item; deriving it from the directory
    # name produced SRC_BALANCEDM for balanced_50M, which sets nothing.
    printf "             set %s=<path> (export it, or in scripts/local_paths.sh)\n" "$var"
    missing=$((missing+1)); continue
  fi
  printf "  [ to do  ] %-20s <- %s\n" "$dest" "$src"; todo=$((todo+1))
  if [[ $APPLY -eq 1 ]]; then
    mkdir -p "$KF_DATA"
    if [[ $MODE == symlink ]]; then ln -s "$src" "$target"
    else cp -r --reflink=auto "$src" "$target"; fi
  fi
done

# The 13-mer vocabulary is two files rather than a directory.
vdest="$KF_DATA/vocab"
for f in vocab_13mer.txt vocab_13mer.txt.table_k13.npy; do
  if [[ -e "$vdest/$f" ]]; then
    printf "  [in place] %-20s %s\n" "vocab/$f" "13-mer vocabulary"; ready=$((ready+1))
  elif [[ -f "$SRC_VOCAB_DIR/$f" ]]; then
    printf "  [ to do  ] %-20s <- %s\n" "vocab/$f" "$SRC_VOCAB_DIR/$f"; todo=$((todo+1))
    if [[ $APPLY -eq 1 ]]; then mkdir -p "$vdest"; ln -s "$SRC_VOCAB_DIR/$f" "$vdest/$f"; fi
  else
    # the cached table is rebuilt automatically on first use; the text file is not
    if [[ $f == *.npy ]]; then
      printf "  [ skip   ] %-20s not found; rebuilt automatically on first use\n" "vocab/$f"
    else
      printf "  [MISSING ] %-20s not found under %s\n" "vocab/$f" "$SRC_VOCAB_DIR"
      missing=$((missing+1))
    fi
  fi
done

echo
echo "in place: $ready   to do: $todo   missing: $missing"
[[ $APPLY -eq 0 && $todo -gt 0 ]] && echo "Re-run with --apply to create them."
if [[ $missing -gt 0 ]]; then
  echo
  echo "For anything missing, see docs/MISSING_ASSETS.md -- it says which machine"
  echo "each asset is on and whether you actually need it."
fi
echo
echo "Not covered here (only on the A6000): the 250M read pool and the 13-mer run"
echo "outputs. See scripts/fetch/fetch_a6000.sh."
exit $(( missing > 0 ? 1 : 0 ))

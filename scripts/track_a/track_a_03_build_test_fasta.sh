#!/bin/bash
# Track A – Step 3: merge per-genus reads → single test FASTA.
#
# Key design: ONE output redirect ">" for the entire loop.
# Multiple "tee -a" calls on NFS cause write reordering / corruption.
#
# Usage:
#   bash local_realworld_eval/track_a_03_build_test_fasta.sh \
#       [reads_dir] [out_dir] [max_per_genus]

set -euo pipefail

READS_DIR="${1:-${KF_TRACK_A:?set KF_TRACK_A}/reads}"
OUT_DIR="${2:-${KF_TRACK_A:?set KF_TRACK_A}/test_data}"
MAX_PER_GENUS="${3:-5000}"
TAKE_LINES=$(( MAX_PER_GENUS * 2 ))  # 2 lines per read (single-line FASTA)

mkdir -p "${OUT_DIR}/val_dir"
OUT_FA="${OUT_DIR}/newgenome_test.fa"
OUT_TSV="${OUT_DIR}/newgenome_test_labels.tsv"

echo "Reads dir  : ${READS_DIR}"
echo "Out FASTA  : ${OUT_FA}"
echo "Max/genus  : ${MAX_PER_GENUS} reads"
echo ""

# ── Single-redirect loop: NFS-safe ───────────────────────────────────────────
# Each genus: take first MAX_PER_GENUS reads from its first .fa file.
# With 10× coverage, even the smallest genome (>1 Mb) yields >60K reads >> 5000.
{
    for genus_dir in "${READS_DIR}"/*/; do
        [[ -d "${genus_dir}" ]] || continue
        for fa in "${genus_dir}"*.fa; do
            [[ -f "${fa}" ]] || break   # no .fa → skip genus
            head -n "${TAKE_LINES}" "${fa}"
            break   # one file per genus is enough
        done
    done
} > "${OUT_FA}"

# ── Quick validation & counts ─────────────────────────────────────────────────
echo "Counting …"
stats=$(awk '
    /^>/ {
        headers++
        n = split(substr($0,2), f, "|")
        if (n >= 4) classes[f[4]]++
    }
    END { print headers, length(classes) }
' "${OUT_FA}")
total=$(echo "$stats" | awk '{print $1}')
genera=$(echo "$stats" | awk '{print $2}')
echo "Total: ${total} reads from ${genera} genera"

# ── Labels TSV ────────────────────────────────────────────────────────────────
echo "Building labels TSV …"
printf "seq_id\tgenus_class\tgenus_name\n" > "${OUT_TSV}"
awk '/^>/{
    hdr = substr($0, 2)
    n = split(hdr, f, "|")
    if (n < 5) next
    gn = f[5]; sub(/-[0-9]+$/, "", gn)
    print hdr "\t" f[4] "\t" gn
}' "${OUT_FA}" >> "${OUT_TSV}"
echo "Labels: $(( $(wc -l < "${OUT_TSV}") - 1 )) rows"

# ── Verify no malformed headers ───────────────────────────────────────────────
bad=$(awk 'BEGIN{bad=0} /^>/{n=split(substr($0,2),f,"|"); if(n<4) bad++} END{print bad}' "${OUT_FA}")
echo "Malformed headers: ${bad}"
if [[ "${bad}" -gt 0 ]]; then
    echo "ERROR: malformed headers found — check NFS write integrity"
    exit 1
fi

# ── Symlink for MT --val_dir ──────────────────────────────────────────────────
ln -sf "$(realpath "${OUT_FA}")" "${OUT_DIR}/val_dir/newgenome_test.fa" 2>/dev/null || \
  cp "${OUT_FA}" "${OUT_DIR}/val_dir/newgenome_test.fa"

echo ""
echo "Done.  FASTA: $(du -sh "${OUT_FA}" | cut -f1)"

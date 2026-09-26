#!/usr/bin/env bash
# Full pipeline: extract (BM25 retrieval + features, cached to disk) -> fit (+ tuning on the exact
# leaderboard metric) -> analyze (error report) -> predict (submission TSVs) -> validate.
# Every step logs to $OUT/pipeline.log. Re-running skips extract steps that already finished.
#
#   DATA=~/ber_project/dataset OUT=~/ber_project/output_v3 JOBS=40 bash run_pipeline.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${DATA:?set DATA to the folder that contains train/ and test/}"
OUT="${OUT:?set OUT to a NEW output folder (caches from older code versions are not compatible)}"
JOBS="${JOBS:-32}"
MAX_DF="${MAX_DF:-0.05}"
US_SAMPLE="${US_SAMPLE:-300000}"
mkdir -p "$OUT"
LOG="$OUT/pipeline.log"
FM=(python "$HERE/src/fast_match.py" --data-dir "$DATA" --out-dir "$OUT" --n-jobs "$JOBS")

run() {
    echo "===== $(date '+%F %T')  $*" | tee -a "$LOG"
    nice -n 10 "${FM[@]}" "$@" 2>&1 | tee -a "$LOG"
}

run --stage extract --split train --countries India --max-df "$MAX_DF"
run --stage extract --split train --countries US --sample "$US_SAMPLE" --max-df "$MAX_DF"
run --stage extract --split test --max-df "$MAX_DF"
run --stage fit
run --stage analyze
run --stage predict
python "$HERE/../../utils/validate_submission.py" -m "$OUT/matching_results.tsv" \
    -c "$OUT/candidate_pairs.tsv" -t "$DATA/test" 2>&1 | tee -a "$LOG"
echo "===== $(date '+%F %T')  done. Submission: $OUT/matching_results.tsv + $OUT/candidate_pairs.tsv" | tee -a "$LOG"

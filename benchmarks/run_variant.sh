#!/usr/bin/env bash
# Run one CellScoper configuration variant for the ablation study.
#
#   run_variant.sh DATA_DIR RUNS_DIR NAME GPU [BASE_RUN]
#
# RUNS_DIR/overlays/NAME.yaml is merged over the packaged params.yaml. With
# BASE_RUN, upstream checkpoints are copied from RUNS_DIR/BASE_RUN and only the
# annotation stage is rerun, so variants share identical MetaSpots/MetaDomains.
set -euo pipefail
DATA=$1; RUNS=$2; NAME=$3; GPU=$4; BASE=${5:-}
PY=${CELLSCOPER_PYTHON:-python}
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$RUNS/$NAME
EXTRA=()
if [[ -n "$BASE" ]]; then
  mkdir -p "$OUT"
  ( cd "$RUNS/$BASE" && find . -mindepth 1 -maxdepth 1 \
      ! -name 'final_data*' ! -name 'adata*.h5ad' ! -name 'run_manifest.json' \
      ! -name '*.log' ! -name '.cellscoper.run.lock' -exec cp -r {} "$OUT/" \; )
  EXTRA=(--resume-from annotate)
fi
cd "$REPO"
PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=$GPU /usr/bin/time -v "$PY" -m cellscope.cli \
  --spatial "$DATA/test_transcripts_df.csv" \
  --cell-boundaries "$DATA/test_cell_boundaries_df.csv" \
  --nucleus-boundaries "$DATA/test_nucleus_boundaries_df.csv" \
  --out-dir "$OUT" --config "$RUNS/overlays/$NAME.yaml" --no-show-sample \
  "${EXTRA[@]}" > "$RUNS/$NAME.log" 2>&1

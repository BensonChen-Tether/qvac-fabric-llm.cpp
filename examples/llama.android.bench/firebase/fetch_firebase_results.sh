#!/usr/bin/env bash
set -euo pipefail

FIREBASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: fetch_firebase_results.sh GCS_OR_LOCAL_PATH [LOCAL_DIR]

Download Firebase Test Lab artifacts from GCS (or use an existing local folder),
extract llama-bench JSON from logcat, and generate an Excel summary.

Arguments:
  GCS_OR_LOCAL_PATH   gs://bucket/bench/MODEL/RUN_ID  OR  local testlab_results folder
  LOCAL_DIR           Optional destination (default: firebase/testlab_results/<basename>)

Examples:
  ./firebase/fetch_firebase_results.sh gs://qvac-test-ftl-results/bench/qwen3-0.6B_Qwen3-0.6B-TQ2_0_Tether.gguf/20260604_124047

  ./firebase/fetch_firebase_results.sh firebase/testlab_results/my_run_folder

Environment (optional, from firebase/config.env):
  RESULTS_BUCKET
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

if [[ -f "$FIREBASE_DIR/config.env" ]]; then
  # shellcheck disable=SC1091
  source "$FIREBASE_DIR/config.env"
fi

SOURCE="$1"
if [[ $# -ge 2 ]]; then
  LOCAL_DIR="$2"
else
  LOCAL_DIR="$FIREBASE_DIR/testlab_results/$(basename "${SOURCE%/}")"
fi

if [[ "$SOURCE" == gs://* ]]; then
  if ! command -v gsutil >/dev/null 2>&1; then
    echo "gsutil not found. Install Google Cloud SDK first." >&2
    exit 1
  fi
  echo "Downloading Test Lab artifacts..."
  echo "  From: $SOURCE"
  echo "  To:   $LOCAL_DIR"
  mkdir -p "$LOCAL_DIR"
  gsutil -m cp -r "${SOURCE%/}/*" "$LOCAL_DIR/" || true
elif [[ -d "$SOURCE" ]]; then
  LOCAL_DIR="$(cd "$SOURCE" && pwd)"
  echo "Using local results: $LOCAL_DIR"
else
  echo "Path not found (expected gs://... or existing directory): $SOURCE" >&2
  exit 1
fi

EXCEL_PATH="$LOCAL_DIR/benchmark.xlsx"
echo "Extracting benchmark JSON and generating Excel report..."
python3 "$FIREBASE_DIR/benchmark_firebase.py" \
  --results-dir "$LOCAL_DIR" \
  --output "$EXCEL_PATH" \
  --extract-json

echo
echo "Retrieved benchmark output:"
echo "  Local folder:  $LOCAL_DIR"
echo "  JSON extracts: $LOCAL_DIR/extracted/"
echo "  Excel report:  $EXCEL_PATH"
echo
echo "Per-device logcat (raw):"
find "$LOCAL_DIR" -name logcat -print 2>/dev/null || true

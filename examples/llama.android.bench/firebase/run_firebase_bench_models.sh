#!/usr/bin/env bash
set -euo pipefail

FIREBASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$FIREBASE_DIR/.." && pwd)"
RUN_BENCH="$FIREBASE_DIR/run_firebase_bench.sh"

usage() {
  cat <<'EOF'
Usage: run_firebase_bench_models.sh [OPTIONS] [MODEL_PATH...]

Run each model in a separate Firebase Test Lab session (one gcloud matrix per model).
Builds APKs once, then invokes run_firebase_bench.sh independently for each model.

Options:
  -f, --models-file FILE   Text file with one model path per line
      --cpu                Run on CPU only (llama-bench -ngl 0)
      --stop-on-error      Stop after the first failed model (default: continue)
  -h, --help               Show this help

Examples:
  ./firebase/run_firebase_bench_models.sh \
    qwen3-0.6B/Qwen3-0.6B-TQ2_0_Tether.gguf \
    qwen3-0.6B/Qwen3-0.6B-Q4_K_M.gguf \
    qwen3-1.7B/Qwen3-1.7B-TQ2_0.gguf

  ./firebase/run_firebase_bench_models.sh -f firebase/model-matrix.txt

Each model gets its own Test Lab run ID, GCS results folder, and local report.
EOF
}

MODELS=()
MODELS_FILE=""
CONTINUE_ON_ERROR=1
N_GPU_LAYERS="${N_GPU_LAYERS:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -f|--models-file)
      MODELS_FILE="$2"
      shift 2
      ;;
    --cpu)
      N_GPU_LAYERS=0
      shift
      ;;
    --stop-on-error)
      CONTINUE_ON_ERROR=0
      shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      MODELS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$MODELS_FILE" ]]; then
  if [[ ! -f "$MODELS_FILE" ]]; then
    echo "Models file not found: $MODELS_FILE" >&2
    exit 1
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="$(echo "$line" | xargs)"
    [[ -n "$line" ]] || continue
    MODELS+=("$line")
  done < "$MODELS_FILE"
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "No models specified. Pass model paths or use --models-file." >&2
  usage >&2
  exit 1
fi

if [[ -n "$N_GPU_LAYERS" ]]; then
  export N_GPU_LAYERS
fi

BACKEND_LABEL="GPU"
if [[ "${N_GPU_LAYERS:-999}" == "0" ]]; then
  BACKEND_LABEL="CPU"
fi

echo "Building APKs once for ${#MODELS[@]} model session(s) (${BACKEND_LABEL})..."
cd "$ROOT_DIR"
./gradlew assembleDebug assembleDebugAndroidTest

SESSION=0
FAILED_MODELS=()
PASSED_MODELS=()

for model in "${MODELS[@]}"; do
  SESSION=$((SESSION + 1))
  echo
  echo "============================================================"
  echo "Test Lab session ${SESSION}/${#MODELS[@]}: ${model} (${BACKEND_LABEL})"
  echo "============================================================"

  if "$RUN_BENCH" --skip-build "$model"; then
    PASSED_MODELS+=("$model")
  else
    FAILED_MODELS+=("$model")
    echo "Session failed: $model" >&2
    if [[ "$CONTINUE_ON_ERROR" == "0" ]]; then
      break
    fi
  fi
done

echo
echo "Batch summary (${#PASSED_MODELS[@]} passed, ${#FAILED_MODELS[@]} failed):"
if [[ ${#PASSED_MODELS[@]} -gt 0 ]]; then
  echo "  Passed:"
  for model in "${PASSED_MODELS[@]}"; do
    echo "    - $model"
  done
fi
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
  echo "  Failed:"
  for model in "${FAILED_MODELS[@]}"; do
    echo "    - $model"
  done
  exit 1
fi

echo "All model sessions completed successfully."

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIREBASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: run_firebase_bench.sh [OPTIONS] [MODEL_PATH]

Run llama-bench on Firebase Test Lab devices and generate an Excel report.

Options:
  -d, --devices FILE     JSON device matrix (default: firebase/device-matrix.json)
  -D, --device SPEC      Single device spec, repeatable. Example:
                           --device model=oriole,version=33,locale=en,orientation=portrait
  -h, --help             Show this help

Environment / config.env:
  DEVICE_MATRIX          Same as --devices
  GCP_PROJECT_ID         Firebase/GCP project
  RESULTS_BUCKET         gs:// bucket for Test Lab output
  DEFAULT_MODEL_PATH     Model when MODEL_PATH is omitted
  DEFAULT_REPETITIONS    llama-bench -r value (default: 5)
  TEST_TIMEOUT           Per-device timeout (default: 45m)

Examples:
  ./firebase/run_firebase_bench.sh
  ./firebase/run_firebase_bench.sh qwen3-1.7B/Qwen3-1.7B-TQ2_0.gguf
  ./firebase/run_firebase_bench.sh -d firebase/device-matrix-pixel7.json
  ./firebase/run_firebase_bench.sh -D model=panther,version=34,locale=en,orientation=portrait
  DEVICE_MATRIX=firebase/my-devices.json ./firebase/run_firebase_bench.sh

Device matrix JSON format (one object per device):
  [
    { "model": "oriole", "version": "33", "locale": "en", "orientation": "portrait" }
  ]

Find device IDs: https://firebase.google.com/docs/test-lab/android/available-testing-devices
  or: gcloud firebase test android models list
EOF
}

if [[ -f "$FIREBASE_DIR/config.env" ]]; then
  # shellcheck disable=SC1091
  source "$FIREBASE_DIR/config.env"
fi

MODEL_PATH="${DEFAULT_MODEL_PATH:-qwen3-1.7B/Qwen3-1.7B-Q4_K_M.gguf}"
REPETITIONS="${REPETITIONS:-${DEFAULT_REPETITIONS:-5}}"
TEST_TIMEOUT="${TEST_TIMEOUT:-45m}"
DEVICE_MATRIX="${DEVICE_MATRIX:-$FIREBASE_DIR/device-matrix.json}"
RESULTS_BUCKET="${RESULTS_BUCKET:-}"
GCP_PROJECT_ID="${GCP_PROJECT_ID:-}"
DOWNLOAD_RESULTS="${DOWNLOAD_RESULTS:-1}"
DEVICE_SPECS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -d|--devices)
      DEVICE_MATRIX="$2"
      shift 2
      ;;
    -D|--device)
      DEVICE_SPECS+=("$2")
      shift 2
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      MODEL_PATH="$1"
      shift
      ;;
  esac
done

if [[ -z "$GCP_PROJECT_ID" ]]; then
  echo "Set GCP_PROJECT_ID in firebase/config.env (copy from config.env.example)" >&2
  exit 1
fi

if [[ -z "$RESULTS_BUCKET" ]]; then
  echo "Set RESULTS_BUCKET in firebase/config.env" >&2
  exit 1
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud CLI not found. Install Google Cloud SDK first." >&2
  exit 1
fi

APP_APK="$ROOT_DIR/app/build/outputs/apk/debug/app-debug.apk"
TEST_APK="$ROOT_DIR/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
SAFE_MODEL_PATH="${MODEL_PATH//\//_}"
RESULTS_DIR="$FIREBASE_DIR/testlab_results/${SAFE_MODEL_PATH}_${RUN_ID}"
GCS_RESULTS_DIR="${RESULTS_BUCKET%/}/bench/${SAFE_MODEL_PATH}/${RUN_ID}"

GCLOUD_DEVICE_ARGS=()
load_devices_from_matrix() {
  python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    devices = json.load(handle)

if not isinstance(devices, list) or not devices:
    raise SystemExit("Device matrix must be a non-empty JSON array")

for device in devices:
    if not isinstance(device, dict) or "model" not in device:
        raise SystemExit("Each device entry must be an object with at least 'model'")
    spec = ",".join(f"{key}={device[key]}" for key in ("model", "version", "locale", "orientation") if key in device)
    print(spec)
PY
}

if [[ ${#DEVICE_SPECS[@]} -gt 0 ]]; then
  for spec in "${DEVICE_SPECS[@]}"; do
    GCLOUD_DEVICE_ARGS+=(--device "$spec")
  done
elif [[ -f "$DEVICE_MATRIX" ]]; then
  while IFS= read -r spec; do
    [[ -n "$spec" ]] || continue
    DEVICE_SPECS+=("$spec")
    GCLOUD_DEVICE_ARGS+=(--device "$spec")
  done < <(load_devices_from_matrix "$DEVICE_MATRIX")
  if [[ ${#GCLOUD_DEVICE_ARGS[@]} -eq 0 ]]; then
    echo "No devices found in matrix: $DEVICE_MATRIX" >&2
    exit 1
  fi
else
  echo "Device matrix not found: $DEVICE_MATRIX" >&2
  echo "Create the file or pass --device model=...,version=..." >&2
  exit 1
fi

echo "Building APKs..."
cd "$ROOT_DIR"
./gradlew assembleDebug assembleDebugAndroidTest

echo "Running Firebase Test Lab..."
echo "  Project:     $GCP_PROJECT_ID"
echo "  Model:       $MODEL_PATH"
echo "  Repetitions: $REPETITIONS"
if [[ ${#DEVICE_SPECS[@]} -gt 0 ]]; then
  echo "  Devices:"
  for spec in "${DEVICE_SPECS[@]}"; do
    echo "    - $spec"
  done
fi
echo "  Results:     $GCS_RESULTS_DIR"

gcloud firebase test android run \
  --project "$GCP_PROJECT_ID" \
  --type instrumentation \
  --app "$APP_APK" \
  --test "$TEST_APK" \
  "${GCLOUD_DEVICE_ARGS[@]}" \
  --timeout "$TEST_TIMEOUT" \
  --environment-variables "model_path=${MODEL_PATH},repetitions=${REPETITIONS},skip_download=false" \
  --results-bucket "${RESULTS_BUCKET#gs://}" \
  --results-dir "bench/${SAFE_MODEL_PATH}/${RUN_ID}"

if [[ "$DOWNLOAD_RESULTS" == "1" ]]; then
  if ! command -v gsutil >/dev/null 2>&1; then
    echo "gsutil not found; skipping download. Results are at $GCS_RESULTS_DIR"
  else
    echo "Downloading Test Lab artifacts..."
    mkdir -p "$RESULTS_DIR"
    gsutil -m cp -r "${GCS_RESULTS_DIR}/*" "$RESULTS_DIR/"

    EXCEL_PATH="$RESULTS_DIR/benchmark.xlsx"
    echo "Generating Excel report..."
    python3 "$FIREBASE_DIR/benchmark_firebase.py" \
      --results-dir "$RESULTS_DIR" \
      --output "$EXCEL_PATH" \
      --extract-json

    echo
    echo "Benchmark output retrieved:"
    echo "  Local artifacts: $RESULTS_DIR"
    echo "  JSON extracts:   $RESULTS_DIR/extracted/"
    echo "  Excel report:    $EXCEL_PATH"
    echo "  GCS (cloud):     $GCS_RESULTS_DIR"
    echo
    echo "Re-fetch later without re-running Test Lab:"
    echo "  ./firebase/fetch_firebase_results.sh ${GCS_RESULTS_DIR}"
  fi
else
  echo "Skipped local download. Fetch results later with:"
  echo "  ./firebase/fetch_firebase_results.sh ${GCS_RESULTS_DIR}"
fi

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${1:-qwen3-1.7B/Qwen3-1.7B-Q4_K_M.gguf}"
REPETITIONS="${2:-5}"
SKIP_DOWNLOAD="${3:-true}"

cd "$ROOT_DIR"

echo "Building and running connected instrumentation test..."
echo "  Model:       $MODEL_PATH"
echo "  Repetitions: $REPETITIONS"
echo "  Skip download if cached: $SKIP_DOWNLOAD"

./gradlew assembleDebug assembleDebugAndroidTest connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.model_path="$MODEL_PATH" \
  -Pandroid.testInstrumentationRunnerArguments.repetitions="$REPETITIONS" \
  -Pandroid.testInstrumentationRunnerArguments.skip_download="$SKIP_DOWNLOAD"

echo
echo "To generate Excel from device logcat manually:"
echo "  adb logcat -d | grep -E 'LLAMA_BENCH_(RESULT|META)' > firebase/testlab_results/local_logcat.txt"
echo "  python3 firebase/benchmark_firebase.py --results-dir firebase/testlab_results"

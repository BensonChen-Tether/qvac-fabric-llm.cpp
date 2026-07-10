#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICEFARM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT_DIR/build/devicefarm"
BENCH_CONFIG_JSON="$DEVICEFARM_DIR/devicefarm_bench.json"
ASSETS_DIR="$ROOT_DIR/app/src/main/assets"

usage() {
  cat <<'EOF'
Usage: build_devicefarm.sh

Build debug app APK + androidTest APK for AWS Device Farm.
Writes devicefarm_bench.json into app assets when MODEL_DOWNLOAD_URL is set.

Environment:
  MODEL_DOWNLOAD_URL   Presigned S3 model URL
  MODEL_PATH           Model filename (e.g. qwen3_1p7b-epoch01-TQ2_0.gguf)
  N_GPU_LAYERS         999=GPU Vulkan, 0=CPU
  REPETITIONS          Benchmark repetitions (default: 5)

Outputs:
  build/devicefarm/app-debug.apk
  build/devicefarm/app-debug-androidTest.apk
EOF
}

if [[ -f "$DEVICEFARM_DIR/config.env" ]]; then
  # shellcheck disable=SC1091
  source "$DEVICEFARM_DIR/config.env"
fi

REPETITIONS="${REPETITIONS:-${DEFAULT_REPETITIONS:-5}}"

write_bench_config() {
  if [[ -z "${MODEL_DOWNLOAD_URL:-}" ]]; then
    echo "MODEL_DOWNLOAD_URL is required to bake Device Farm bench config" >&2
    exit 1
  fi
  python3 - <<'PY' > "$BENCH_CONFIG_JSON"
import json
import os

print(json.dumps({
    "automation": True,
    "model_path": os.environ["MODEL_PATH"],
    "model_download_url": os.environ["MODEL_DOWNLOAD_URL"],
    "n_gpu_layers": int(os.environ["N_GPU_LAYERS"]),
    "repetitions": int(os.environ["REPETITIONS"]),
    "skip_download": False,
}, indent=2))
PY
  echo "Wrote bench config: $BENCH_CONFIG_JSON" >&2
}

inject_bench_config() {
  mkdir -p "$ASSETS_DIR"
  cp "$BENCH_CONFIG_JSON" "$ASSETS_DIR/devicefarm_bench.json"
  echo "Injected devicefarm_bench.json into $ASSETS_DIR" >&2
}

write_bench_config
inject_bench_config

mkdir -p "$BUILD_DIR"
cd "$ROOT_DIR"
./gradlew assembleDebug assembleDebugAndroidTest

APP_APK="$ROOT_DIR/app/build/outputs/apk/debug/app-debug.apk"
TEST_APK="$ROOT_DIR/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"

if [[ ! -f "$APP_APK" || ! -f "$TEST_APK" ]]; then
  echo "Missing APK outputs after Gradle build" >&2
  exit 1
fi

cp "$APP_APK" "$BUILD_DIR/app-debug.apk"
cp "$TEST_APK" "$BUILD_DIR/app-debug-androidTest.apk"

echo "Built:"
echo "  $BUILD_DIR/app-debug.apk"
echo "  $BUILD_DIR/app-debug-androidTest.apk"

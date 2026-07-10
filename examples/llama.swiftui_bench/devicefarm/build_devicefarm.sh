#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QVAC_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
DEVICEFARM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT_DIR/build/devicefarm"
BENCH_CONFIG_JSON="$DEVICEFARM_DIR/devicefarm_bench.json"

usage() {
  cat <<'EOF'
Usage: build_devicefarm.sh [OPTIONS]

Build Release iOS app (.ipa) and XCUITest bundle (.zip) for AWS Device Farm.

When MODEL_DOWNLOAD_URL is set, writes devicefarm_bench.json into the app bundle
with the presigned URL and benchmark settings baked in.

Options:
  --skip-xcframework   Skip rebuilding llama/prism xcframeworks
  --tests-only         Rebuild only the XCUITest zip (reuse existing IPA)
  -h, --help

Environment (for baked bench config):
  MODEL_DOWNLOAD_URL   Presigned S3 model URL (required for Device Farm app)
  MODEL_PATH           Model filename (e.g. qwen3_1p7b-epoch01-TQ2_0.gguf)
  N_GPU_LAYERS         99=GPU, 0=CPU
  REPETITIONS          Benchmark repetitions (default: 5)

Outputs:
  build/devicefarm/llama.swiftui_bench.ipa
  build/devicefarm/llama.swiftui_benchUITests.zip
EOF
}

SKIP_XCFRAMEWORK=0
TESTS_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-xcframework) SKIP_XCFRAMEWORK=1; shift ;;
    --tests-only) TESTS_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -f "$DEVICEFARM_DIR/config.env" ]]; then
  # shellcheck disable=SC1091
  source "$DEVICEFARM_DIR/config.env"
fi

TEAM="${DEVELOPMENT_TEAM:-3LT8Z8ZRCG}"
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
  local app_path="$1"
  cp "$BENCH_CONFIG_JSON" "$app_path/devicefarm_bench.json"
  echo "Injected devicefarm_bench.json into $app_path" >&2
}

sign_app_bundle() {
  local app_path="$1"
  echo "Ad-hoc signing embedded frameworks in $app_path"
  if [[ -d "$app_path/Frameworks" ]]; then
    for framework in "$app_path/Frameworks/"*.framework; do
      [[ -d "$framework" ]] || continue
      local binary_name
      binary_name="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$framework/Info.plist" 2>/dev/null || basename "$framework" .framework)"
      if [[ -f "$framework/$binary_name" ]]; then
        codesign --force --sign - --timestamp=none "$framework/$binary_name"
      fi
      codesign --force --sign - --timestamp=none "$framework"
    done
  fi
  codesign --force --sign - --deep --timestamp=none "$app_path"
}

mkdir -p "$BUILD_DIR"

if [[ "$TESTS_ONLY" -eq 1 ]]; then
  IPA_PATH="$BUILD_DIR/llama.swiftui_bench.ipa"
  if [[ ! -f "$IPA_PATH" ]]; then
    echo "Missing $IPA_PATH for --tests-only" >&2
    exit 1
  fi
else
  write_bench_config

  if [[ "$SKIP_XCFRAMEWORK" -eq 0 ]]; then
    echo "Building llama.xcframework (qvac)..."
    (cd "$QVAC_ROOT" && ./build-xcframework.sh)
    echo "Building prism_llama.xcframework (Bonsai)..."
    (cd "$ROOT_DIR" && ./scripts/build-prism-xcframework.sh)
  fi

  echo "Building app + UI tests for iOS device..."
  xcodebuild build-for-testing \
    -project "$ROOT_DIR/llama.swiftui_bench.xcodeproj" \
    -scheme llama.swiftui_bench \
    -configuration Release \
    -destination 'generic/platform=iOS' \
    -derivedDataPath "$BUILD_DIR/DerivedData" \
    -allowProvisioningUpdates \
    CODE_SIGNING_ALLOWED=NO \
    CODE_SIGNING_REQUIRED=NO \
    DEVELOPMENT_TEAM="$TEAM" \
    CODE_SIGN_STYLE=Automatic \
    | tee "$BUILD_DIR/xcodebuild.log" | tail -30

  APP_PATH="$(find "$BUILD_DIR/DerivedData/Build/Products" -path '*Release-iphoneos/llama.swiftui_bench.app' -type d | head -1)"
  RUNNER_PATH="$(find "$BUILD_DIR/DerivedData/Build/Products" -path '*Release-iphoneos/*-Runner.app' -type d | head -1)"

  if [[ -z "$APP_PATH" || -z "$RUNNER_PATH" ]]; then
    echo "Could not locate build products. See $BUILD_DIR/xcodebuild.log" >&2
    exit 1
  fi

  inject_bench_config "$APP_PATH"
  sign_app_bundle "$APP_PATH"

  echo "App:    $APP_PATH"
  echo "Runner: $RUNNER_PATH"

  IPA_PATH="$BUILD_DIR/llama.swiftui_bench.ipa"
  rm -rf "$BUILD_DIR/Payload" "$IPA_PATH"
  mkdir -p "$BUILD_DIR/Payload"
  cp -R "$APP_PATH" "$BUILD_DIR/Payload/"
  ( cd "$BUILD_DIR" && zip -qr "$(basename "$IPA_PATH")" Payload )
  rm -rf "$BUILD_DIR/Payload"
fi

if [[ "$TESTS_ONLY" -eq 0 ]]; then
  RUNNER_PATH="$(find "$BUILD_DIR/DerivedData/Build/Products" -path '*Release-iphoneos/*-Runner.app' -type d | head -1)"
fi

TEST_ZIP="$BUILD_DIR/llama.swiftui_benchUITests.zip"
if [[ -n "${RUNNER_PATH:-}" && -d "$RUNNER_PATH" ]]; then
  ( cd "$(dirname "$RUNNER_PATH")" && zip -qr "$TEST_ZIP" "$(basename "$RUNNER_PATH")" )
else
  echo "Warning: XCUITest runner not rebuilt; reusing existing $TEST_ZIP" >&2
fi

echo "Built:"
echo "  $IPA_PATH"
echo "  $TEST_ZIP"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QVAC_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
DEVICEFARM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT_DIR/build/devicefarm"

usage() {
  cat <<'EOF'
Usage: build_devicefarm.sh [--skip-xcframework]

Build Release iOS app (.ipa) and XCUITest bundle (.zip) for AWS Device Farm.

Outputs:
  build/devicefarm/llama.swiftui_bench.ipa
  build/devicefarm/llama.swiftui_benchUITests.zip
EOF
}

SKIP_XCFRAMEWORK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-xcframework) SKIP_XCFRAMEWORK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -f "$DEVICEFARM_DIR/config.env" ]]; then
  # shellcheck disable=SC1091
  source "$DEVICEFARM_DIR/config.env"
fi

TEAM="${DEVELOPMENT_TEAM:-3LT8Z8ZRCG}"

if [[ "$SKIP_XCFRAMEWORK" -eq 0 ]]; then
  echo "Building llama.xcframework (qvac)..."
  (cd "$QVAC_ROOT" && ./build-xcframework.sh)
  echo "Building prism_llama.xcframework (Bonsai)..."
  (cd "$ROOT_DIR" && ./scripts/build-prism-xcframework.sh)
fi

mkdir -p "$BUILD_DIR"

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

sign_app_bundle "$APP_PATH"

echo "App:    $APP_PATH"
echo "Runner: $RUNNER_PATH"

IPA_PATH="$BUILD_DIR/llama.swiftui_bench.ipa"
TEST_ZIP="$BUILD_DIR/llama.swiftui_benchUITests.zip"

rm -rf "$BUILD_DIR/Payload" "$IPA_PATH" "$TEST_ZIP"
mkdir -p "$BUILD_DIR/Payload"
cp -R "$APP_PATH" "$BUILD_DIR/Payload/"
( cd "$BUILD_DIR" && zip -qr "$(basename "$IPA_PATH")" Payload )
rm -rf "$BUILD_DIR/Payload"

( cd "$(dirname "$RUNNER_PATH")" && zip -qr "$TEST_ZIP" "$(basename "$RUNNER_PATH")" )

echo "Built:"
echo "  $IPA_PATH"
echo "  $TEST_ZIP"

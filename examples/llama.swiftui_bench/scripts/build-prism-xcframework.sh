#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TETHER_ROOT="$(cd "$BENCH_ROOT/../../.." && pwd)"
PRISM_ROOT="$TETHER_ROOT/prism-llama.cpp"
SRC_XCFRAMEWORK="$PRISM_ROOT/build-apple/llama.xcframework"
OUT_XCFRAMEWORK="$PRISM_ROOT/build-apple/prism_llama.xcframework"

rename_framework_slice() {
  local slice_dir="$1"
  local old_fw="$slice_dir/llama.framework"
  local new_fw="$slice_dir/prism_llama.framework"

  if [[ ! -d "$old_fw" ]]; then
    echo "Skipping missing framework slice: $old_fw" >&2
    return 0
  fi

  rm -rf "$new_fw"
  mv "$old_fw" "$new_fw"

  local old_bin="$new_fw/llama"
  local new_bin="$new_fw/prism_llama"
  if [[ -f "$old_bin" ]]; then
    mv "$old_bin" "$new_bin"
    install_name_tool -id "@rpath/prism_llama.framework/prism_llama" "$new_bin"
  elif [[ -L "$new_fw/llama" ]]; then
    rm "$new_fw/llama"
    ln -sf prism_llama "$new_fw/prism_llama"
  fi

  if [[ -f "$new_fw/Info.plist" ]]; then
    /usr/libexec/PlistBuddy -c "Set :CFBundleName prism_llama" "$new_fw/Info.plist" 2>/dev/null || true
    /usr/libexec/PlistBuddy -c "Set :CFBundleExecutable prism_llama" "$new_fw/Info.plist" 2>/dev/null || true
    /usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier org.ggml.prism-llama" "$new_fw/Info.plist" 2>/dev/null || true
  fi

  if [[ -d "$new_fw/Versions/A" ]]; then
    local mac_bin="$new_fw/Versions/A/prism_llama"
    if [[ -f "$new_fw/Versions/A/llama" ]]; then
      mv "$new_fw/Versions/A/llama" "$mac_bin"
      install_name_tool -id "@rpath/prism_llama.framework/Versions/Current/prism_llama" "$mac_bin"
    fi
    rm -f "$new_fw/prism_llama"
    ln -sf Versions/Current/prism_llama "$new_fw/prism_llama"
  fi
}

echo "Building prism-llama.xcframework..."
(cd "$PRISM_ROOT" && ./build-xcframework.sh)

rm -rf "$OUT_XCFRAMEWORK"
cp -R "$SRC_XCFRAMEWORK" "$OUT_XCFRAMEWORK"

while IFS= read -r -d '' slice; do
  rename_framework_slice "$slice"
  if [[ -d "$slice/dSYMs/llama.dSYM" ]]; then
    mv "$slice/dSYMs/llama.dSYM" "$slice/dSYMs/prism_llama.dSYM"
  fi
done < <(find "$OUT_XCFRAMEWORK" -mindepth 1 -maxdepth 1 -type d -print0)

python3 - <<'PY' "$OUT_XCFRAMEWORK/Info.plist"
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
text = text.replace("llama.framework/llama", "prism_llama.framework/prism_llama")
text = text.replace("llama.framework/Versions/A/llama", "prism_llama.framework/Versions/A/prism_llama")
text = text.replace("<string>llama.framework</string>", "<string>prism_llama.framework</string>")
path.write_text(text)
PY

echo "Renamed prism xcframework: $OUT_XCFRAMEWORK"

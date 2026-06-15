# LlamaBench (Android)

Android benchmark app for running `llama-bench` on device. The app bundles a per-ABI `llama-bench` binary (in APK assets) and links **prism-llama.cpp** with Vulkan GPU offload.

- **Package ID:** `com.example.llama.bench`
- **Gradle project name:** `LlamaBench`
- **minSdk:** 33 (Android 13+)
- **targetSdk / compileSdk:** 36
- **ABIs:** `arm64-v8a`, `x86_64`

## Repository layout

This project lives inside the [Tether](https://github.com/Benson-Chen/Tether) monorepo and expects **prism-llama.cpp** as a sibling directory:

```
Tether/
├── prism-llama.cpp/                          # required — Bonsai + Vulkan backend
└── qvac-fabric-llm.cpp/
    └── examples/
        └── llama.android.bench/              # this project
            ├── app/                          # Android application
            ├── lib/                          # native CMake (llama-bench + GGML)
            └── firebase/                     # Firebase Test Lab scripts
```

The native build in `lib/src/main/cpp/CMakeLists.txt` resolves `prism-llama.cpp` relative to the Tether root. If that directory is missing, CMake fails.

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Android SDK** | API 36; set `ANDROID_HOME` or create `local.properties` with `sdk.dir=...` |
| **Android NDK** | **29.0.13113456** (pinned in `lib/build.gradle.kts`) |
| **Java 17+** | Required by AGP 8.x / Kotlin toolchain |
| **Vulkan SDK** | Host headers (`vulkan.hpp`) and `glslc` on PATH; or set `VULKAN_SDK` |
| **prism-llama.cpp** | Cloned at `Tether/prism-llama.cpp` |

Install the NDK via Android Studio SDK Manager or:

```bash
sdkmanager "ndk;29.0.13113456"
```

On macOS with Homebrew Vulkan:

```bash
brew install vulkan-headers vulkan-loader molten-vk   # or full LunarG Vulkan SDK
which glslc   # must resolve
```

## Build

### Option A — build script (from Tether root)

```bash
# Debug APK (default) → copied to Tether root as llama.android.bench-debug.apk
./.cursor/skills/build-llama-android-bench/scripts/build_apk.sh

# Release APK
./.cursor/skills/build-llama-android-bench/scripts/build_apk.sh --release
```

### Option B — Gradle directly

```bash
cd qvac-fabric-llm.cpp/examples/llama.android.bench

# Create local.properties if needed
echo "sdk.dir=$ANDROID_HOME" > local.properties

# Debug APK
./gradlew assembleDebug

# Release APK
./gradlew assembleRelease
```

### Build outputs

| Variant | APK path |
|---------|----------|
| Debug | `app/build/outputs/apk/debug/app-debug.apk` |
| Release | `app/build/outputs/apk/release/app-release.apk` |

For instrumentation / Firebase Test Lab, also build the test APK:

```bash
./gradlew assembleDebug assembleDebugAndroidTest
```

### What the build does

1. **`:lib`** — CMake compiles **prism-llama.cpp** (Vulkan + CPU backends) for each ABI.
2. **`llama-bench`** — built per ABI and copied into `app/src/main/assets/bin/<abi>/llama-bench`.
3. **`:app`** — packages native `.so` libraries, assets, and Kotlin UI into the APK.

The first build compiles all of prism-llama.cpp and can take several minutes. Later builds are incremental.

> **Note:** Debug builds are intentionally unobfuscated (`isMinifyEnabled = false`) for profiling and Firebase Test Lab. Release builds enable R8 shrinking.

## Install on a device

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Or install the copy at the Tether root:

```bash
adb install -r ../../../llama.android.bench-debug.apk
```

## Run benchmarks

### In the app

Launch **LlamaBench**, pick or download a GGUF model, and run a benchmark from the UI.

Models are downloaded from Hugging Face (`Benson-Chen/tether-gguf-models`) at runtime unless already cached on device.

### Headless instrumentation (USB device)

```bash
./firebase/run_local_instrumented_test.sh \
  qwen3-1.7B/Qwen3-1.7B-Q4_K_M.gguf 5 true
```

Arguments: `MODEL_PATH`, `REPETITIONS`, `SKIP_DOWNLOAD_IF_CACHED`.

### Firebase Test Lab

1. Copy and edit config:

   ```bash
   cp firebase/config.env.example firebase/config.env
   ```

2. Run a single model:

   ```bash
   ./firebase/run_firebase_bench.sh qwen3-1.7B/Qwen3-1.7B-TQ2_0.gguf
   ```

3. Run a model matrix (CPU + GPU):

   ```bash
   ./firebase/run_firebase_bench.sh -f firebase/model-matrix-bonsai.txt --cpu-and-gpu
   ```

See `firebase/run_firebase_bench.sh --help` for device matrix options and environment variables.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `prism-llama.cpp` / CMake path errors | Clone `prism-llama.cpp` next to `qvac-fabric-llm.cpp` under the Tether root |
| `sdk.dir` / SDK not found | Set `ANDROID_HOME` and write `local.properties`, or open the project once in Android Studio |
| NDK version mismatch | Install NDK **29.0.13113456** exactly |
| `glslc` or Vulkan headers missing | Install Vulkan SDK; export `VULKAN_SDK` or use Homebrew `vulkan-headers` |
| `mergeDebugAssets` fails / missing `llama-bench` in assets | Ensure `:lib:externalNativeBuildRelease` completed; run `./gradlew clean` and rebuild |
| Large APK (~200 MB) | Expected — includes Vulkan/CPU native libs plus per-ABI `llama-bench` binaries in assets |
| Bonsai model load fails | This app uses **prism-llama.cpp** (not qvac) specifically for Bonsai Q1/Q2 quants |

## Related

- **Tether build script:** `build_apk.sh` in `.cursor/skills/build-llama-android-bench/`
- **CLI benchmarks (adb push):** `benchmark_models.py` + `build.py -qvac -a -v -p` at the Tether root
- **iOS counterpart:** `qvac-fabric-llm.cpp/examples/llama.swiftui_bench`

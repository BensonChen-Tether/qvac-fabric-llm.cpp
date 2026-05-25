# llama.android.bench

Android sample that runs **llama-bench** on a user-selected GGUF file.

Based on [llama.android](../llama.android). The native layer links `tools/llama-bench/llama-bench.cpp` and invokes it via JNI with fixed arguments:

```text
llama-bench -m <model> -p 64 -n 32 -ngl 99 -r 20
```

## Backend

The default build uses **Vulkan GPU** (`llamaBench.backend=vulkan` in `gradle.properties`), matching the project's `qvac-android-vulkan` setup. `-ngl 99` offloads all layers to the GPU.

Requirements for GPU builds:

- **Physical arm64 device** with Vulkan support (Adreno, etc.)
- **Vulkan SDK** on the build machine (`VULKAN_SDK` env var, or Homebrew headers at `/opt/homebrew/include`)
- **`glslc`** on PATH (`$VULKAN_SDK/bin/glslc` or `/usr/local/bin/glslc`)

To build CPU-only instead, set in `gradle.properties`:

```properties
llamaBench.backend=cpu
```

## Usage

1. Open the project in Android Studio (`examples/llama.android.bench`).
2. Build and run on a **physical arm64 device** (Vulkan) or emulator (CPU-only build).
3. Tap the folder button and pick a `.gguf` file.
4. The app copies the model into app storage, runs the benchmark, and shows the markdown table output.

## Modules

- **app** — UI for GGUF selection and benchmark results.
- **lib** — JNI wrapper (`llama-bench` shared library) and GGUF metadata reader.

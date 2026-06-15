# qvac-fabric-llm.cpp/examples/llama.swiftui_bench

Benchmark / finetune variant of the llama.cpp SwiftUI sample for local inference on iPhone.
This is a separate Xcode project from `llama.swiftui` so builds, DerivedData, and signing
do not collide with the upstream example.

For usage instructions and performance stats for the base app, see:
https://github.com/ggml-org/llama.cpp/discussions/4508

### Building

Build both XCFrameworks used by the app:

```console
# Qwen / TQ models (linked at build time)
$ cd qvac-fabric-llm.cpp && ./build-xcframework.sh

# Bonsai models (embedded and loaded at runtime)
$ cd qvac-fabric-llm.cpp/examples/llama.swiftui_bench && ./scripts/build-prism-xcframework.sh
```

Open `llama.swiftui_bench.xcodeproj` in Xcode and build the **llama.swiftui_bench** scheme on a
simulator or device. (Do not use `llama.swiftui.xcodeproj` — that name is from the upstream sample
and is not part of this bench example.)

Bonsai GGUF files use custom Q1_0/Q2_0 quant types supported only by **prism-llama.cpp**. The app
detects `bonsai` in the model filename and dynamically loads `prism_llama.framework` for inference
and benchmarking; all other models use the qvac `llama.xcframework`.

Bundle identifier: `llama-collabora-bench` (distinct from the sample app's `llama-collabora`).

Built-in model downloads come from [Benson-Chen/tether-gguf-models](https://huggingface.co/Benson-Chen/tether-gguf-models) on Hugging Face (Qwen3 0.6B/1.7B and Bonsai 1.7B/4B/8B quantizations).

To use the framework with a different project, add `build-apple/llama.xcframework` via drag-and-drop
or in "Frameworks, Libraries, and Embedded Content" in project settings.

import Foundation
import UIKit

enum BenchmarkAutomationError: LocalizedError {
    case deviceFarmModelAccess(String)

    var errorDescription: String? {
        switch self {
        case .deviceFarmModelAccess(let detail):
            return "Cannot access model due to device farm error: \(detail)"
        }
    }
}

enum BenchmarkAutomation {
    static let resultLogTag = "LLAMA_BENCH_RESULT"
    static let metaLogTag = "LLAMA_BENCH_META"
    static let errorLogTag = "LLAMA_BENCH_ERROR"

    struct RunResult {
        let modelPathInRepo: String
        let modelFile: URL
        let benchJSON: String
    }

    static func shouldRunOnLaunch() -> Bool {
        if ProcessInfo.processInfo.environment["BENCHMARK_AUTOMATION"] == "1" {
            return true
        }
        if DeviceFarmBenchConfig.loadFromBundle()?.automation == true {
            return true
        }
        return false
    }

    static func runFromLaunchEnvironment() async throws -> RunResult {
        let env = ProcessInfo.processInfo.environment
        let bundled = DeviceFarmBenchConfig.loadFromBundle()

        let modelPath = bundled?.modelPath
            ?? env["model_path"]
            ?? "qwen3_1p7b-epoch01-TQ2_0.gguf"
        let repetitions = bundled?.repetitions
            ?? Int(env["repetitions"] ?? "")
            ?? BenchmarkConfig.automationRepetitions
        let nGpuLayers = Int32(
            bundled?.nGpuLayers
                ?? Int(env["n_gpu_layers"] ?? "")
                ?? BenchmarkConfig.defaultNGpuLayers
        )
        let skipDownloadIfCached: Bool
        if let bundledSkip = bundled?.skipDownload {
            skipDownloadIfCached = bundledSkip
        } else {
            skipDownloadIfCached = (env["skip_download"] ?? "true").lowercased() != "false"
        }
        let downloadURL = bundled?.modelDownloadURL ?? env["model_download_url"]
        return try await run(
            modelPathInRepo: modelPath,
            downloadURL: downloadURL,
            repetitions: repetitions,
            nGpuLayers: nGpuLayers,
            skipDownloadIfCached: skipDownloadIfCached
        )
    }

    static func run(
        modelPathInRepo: String,
        downloadURL: String? = nil,
        repetitions: Int = BenchmarkConfig.automationRepetitions,
        nGpuLayers: Int32 = Int32(BenchmarkConfig.defaultNGpuLayers),
        skipDownloadIfCached: Bool = true
    ) async throws -> RunResult {
        let modelFile = try await ensureModel(
            pathInRepo: modelPathInRepo,
            downloadURL: downloadURL,
            skipDownloadIfCached: skipDownloadIfCached
        )

        let options = LlamaRuntimeOptions(
            contextLength: 2048,
            nGpuLayers: nGpuLayers,
            seed: 42,
            temperature: 0,
            topP: 0.95,
            topK: 40,
            flashAttention: false
        )

        let context = try LlamaContext.create_context(path: modelFile.path, options: options)

        _ = await context.bench(
            pp: 8,
            tg: 4,
            pl: 1,
            nr: 1
        )

        let benchJSON = await context.benchJSON(
            pp: BenchmarkConfig.promptTokens,
            tg: BenchmarkConfig.genTokens,
            pl: 1,
            nr: repetitions
        )

        let meta = buildMeta(
            modelPathInRepo: modelPathInRepo,
            modelFile: modelFile,
            repetitions: repetitions,
            nGpuLayers: Int(nGpuLayers),
            downloadURL: downloadURL
        )

        writeArtifacts(meta: meta, benchJSON: benchJSON)

        NSLog("%@ %@", metaLogTag, meta)
        NSLog("%@ %@", resultLogTag, benchJSON)

        return RunResult(
            modelPathInRepo: modelPathInRepo,
            modelFile: modelFile,
            benchJSON: benchJSON
        )
    }

    private static func modelsDirectory() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("models", isDirectory: true)
    }

    private static func ensureModel(
        pathInRepo: String,
        downloadURL: String?,
        skipDownloadIfCached: Bool
    ) async throws -> URL {
        let modelsDir = modelsDirectory()
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        let destination = modelsDir.appendingPathComponent(BenchmarkConfig.localFileName(pathInRepo: pathInRepo))

        if skipDownloadIfCached,
           FileManager.default.fileExists(atPath: destination.path),
           let attrs = try? FileManager.default.attributesOfItem(atPath: destination.path),
           let size = attrs[.size] as? NSNumber,
           size.intValue > 0 {
            return destination
        }

        let url = BenchmarkConfig.downloadURL(pathInRepo: pathInRepo, presignedURL: downloadURL)
        let usingPresignedURL = downloadURL != nil && !(downloadURL?.isEmpty ?? true)

        let (tempURL, response): (URL, URLResponse)
        do {
            (tempURL, response) = try await URLSession.shared.download(from: url)
        } catch {
            if usingPresignedURL {
                throw BenchmarkAutomationError.deviceFarmModelAccess(error.localizedDescription)
            }
            throw error
        }
        defer { try? FileManager.default.removeItem(at: tempURL) }

        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            if usingPresignedURL {
                throw BenchmarkAutomationError.deviceFarmModelAccess("model download failed with HTTP \(statusCode)")
            }
            throw URLError(.badServerResponse)
        }

        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
        return destination
    }

    private static func buildMeta(
        modelPathInRepo: String,
        modelFile: URL,
        repetitions: Int,
        nGpuLayers: Int,
        downloadURL: String?
    ) -> String {
        let attrs = (try? FileManager.default.attributesOfItem(atPath: modelFile.path)) ?? [:]
        let bytes = (attrs[.size] as? NSNumber)?.intValue ?? 0
        let payload: [String: Any] = [
            "model_path": modelPathInRepo,
            "model_file": modelFile.lastPathComponent,
            "model_bytes": bytes,
            "repetitions": repetitions,
            "prompt_tokens": BenchmarkConfig.promptTokens,
            "gen_tokens": BenchmarkConfig.genTokens,
            "n_gpu_layers": nGpuLayers,
            "repo_id": BenchmarkConfig.repoId,
            "download_source": BenchmarkConfig.downloadSource(presignedURL: downloadURL),
            "s3_bucket": BenchmarkConfig.s3Bucket,
            "s3_prefix": BenchmarkConfig.s3ModelPrefix,
            "device_model": UIDevice.current.model,
            "manufacturer": "Apple",
            "ios_version": UIDevice.current.systemVersion,
            "bundle_id": Bundle.main.bundleIdentifier ?? "",
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: []),
              let json = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return json
    }

    private static func writeArtifacts(meta: String, benchJSON: String) {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        try? meta.write(to: docs.appendingPathComponent("benchmark_meta.json"), atomically: true, encoding: .utf8)
        try? benchJSON.write(to: docs.appendingPathComponent("benchmark_result.json"), atomically: true, encoding: .utf8)
    }
}

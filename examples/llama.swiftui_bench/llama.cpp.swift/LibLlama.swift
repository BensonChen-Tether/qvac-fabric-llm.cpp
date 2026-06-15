import Foundation
import llama

enum LlamaError: Error, LocalizedError {
    case couldNotInitializeContext(reason: String)

    var errorDescription: String? {
        switch self {
        case .couldNotInitializeContext(let reason):
            return reason
        }
    }
}

struct LlamaRuntimeOptions {
    var contextLength: Int32
    var nGpuLayers: Int32
    var seed: UInt32
    var temperature: Float
    var topP: Float
    var topK: Int32
    var flashAttention: Bool
}

func llama_batch_clear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
}

func llama_batch_add(_ batch: inout llama_batch, _ id: llama_token, _ pos: llama_pos, _ seq_ids: [llama_seq_id], _ logits: Bool) {
    batch.token   [Int(batch.n_tokens)] = id
    batch.pos     [Int(batch.n_tokens)] = pos
    batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
    for i in 0..<seq_ids.count {
        batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
    }
    batch.logits  [Int(batch.n_tokens)] = logits ? 1 : 0

    batch.n_tokens += 1
}

actor LlamaContext {
    private var model: OpaquePointer
    private var context: OpaquePointer
    private var vocab: OpaquePointer
    private var sampling: UnsafeMutablePointer<llama_sampler>?
    private var batch: llama_batch
    private var tokens_list: [llama_token]
    private var runtimeOptions: LlamaRuntimeOptions
    private let backend: LlamaModelBackend
    var is_done: Bool = false

    /// This variable is used to store temporarily invalid cchars
    private var temporary_invalid_cchars: [CChar]

    var n_len: Int32 = 1024
    var n_cur: Int32 = 0

    var n_decode: Int32 = 0

    init(model: OpaquePointer, context: OpaquePointer, options: LlamaRuntimeOptions, backend: LlamaModelBackend) {
        self.model = model
        self.context = context
        self.backend = backend
        self.tokens_list = []
        self.batch = RuntimeLlama.batchInit(max(Int32(512), options.contextLength), 0, 1)
        self.temporary_invalid_cchars = []
        self.runtimeOptions = options
        self.n_len = options.contextLength
        vocab = RuntimeLlama.modelGetVocab(model)!

        let chainParams = RuntimeLlama.samplerChainDefaultParams()
        let initialChain = RuntimeLlama.samplerChainInit(chainParams)

        if options.topK > 0 {
            RuntimeLlama.samplerChainAdd(initialChain, RuntimeLlama.samplerInitTopK(options.topK))
        }

        let clampedTopP = max(0.0, min(Double(options.topP), 1.0))
        RuntimeLlama.samplerChainAdd(initialChain, RuntimeLlama.samplerInitTopP(Float(clampedTopP), 1))

        let clampedTemp = max(0.0, Double(options.temperature))
        RuntimeLlama.samplerChainAdd(initialChain, RuntimeLlama.samplerInitTemp(Float(clampedTemp)))

        let seed = options.seed == 0 ? UInt32.max : options.seed
        RuntimeLlama.samplerChainAdd(initialChain, RuntimeLlama.samplerInitDist(seed))

        sampling = initialChain
    }

    deinit {
        if let sampling {
            RuntimeLlama.samplerFree(sampling)
        }
        RuntimeLlama.batchFree(batch)
        RuntimeLlama.modelFree(model, backend: backend)
        RuntimeLlama.free(context, backend: backend)
        RuntimeLlama.backendFree(for: backend)
    }

    private func rebuildSamplerChain() {
        let chainParams = RuntimeLlama.samplerChainDefaultParams()
        let newChain = RuntimeLlama.samplerChainInit(chainParams)

        if runtimeOptions.topK > 0 {
            RuntimeLlama.samplerChainAdd(newChain, RuntimeLlama.samplerInitTopK(runtimeOptions.topK))
        }

        let clampedTopP = max(0.0, min(runtimeOptions.topP, 1.0))
        RuntimeLlama.samplerChainAdd(newChain, RuntimeLlama.samplerInitTopP(clampedTopP, 1))

        let clampedTemp = max(0.0, Double(runtimeOptions.temperature))
        RuntimeLlama.samplerChainAdd(newChain, RuntimeLlama.samplerInitTemp(Float(clampedTemp)))

        let seed = runtimeOptions.seed == 0 ? UInt32.max : runtimeOptions.seed
        RuntimeLlama.samplerChainAdd(newChain, RuntimeLlama.samplerInitDist(seed))

        if let sampling {
            RuntimeLlama.samplerFree(sampling)
        }

        sampling = newChain
    }

    private static func validateModelFile(path: String) throws {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory), !isDirectory.boolValue else {
            throw LlamaError.couldNotInitializeContext(reason: "Model file not found at \(path)")
        }

        let attrs = try FileManager.default.attributesOfItem(atPath: path)
        let size = (attrs[.size] as? NSNumber)?.int64Value ?? 0
        guard size > 32 else {
            throw LlamaError.couldNotInitializeContext(reason: "Model file at \(path) is empty or truncated")
        }

        let handle = FileHandle(forReadingAtPath: path)
        defer { try? handle?.close() }
        guard let data = try handle?.read(upToCount: 4), data == Data("GGUF".utf8) else {
            throw LlamaError.couldNotInitializeContext(reason: "Model file at \(path) is not a valid GGUF archive")
        }

        if LlamaModelBackend.forModel(path: path) == .prism {
            try validateBonsaiArchitecture(path: path)
        }
    }

    private static func validateBonsaiArchitecture(path: String) throws {
        let handle = FileHandle(forReadingAtPath: path)
        defer { try? handle?.close() }
        guard let chunk = try handle?.read(upToCount: 4_000_000), !chunk.isEmpty else {
            return
        }

        guard let text = String(data: chunk, encoding: .utf8) else {
            return
        }

        guard text.contains("general.architecture") else {
            return
        }

        if text.contains("qwen3") {
            return
        }

        if text.contains("clip") {
            throw LlamaError.couldNotInitializeContext(
                reason: "Downloaded file looks like a CLIP/mmproj model, not a Bonsai qwen3 model. Delete it and re-download."
            )
        }

        throw LlamaError.couldNotInitializeContext(
            reason: "Bonsai models must use general.architecture=qwen3. This file appears to use a different architecture."
        )
    }

    private static func summarizeLoadLog(_ text: String) -> String {
        let lines = text
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if lines.isEmpty {
            return ""
        }
        let interesting = lines.filter {
            $0.localizedCaseInsensitiveContains("error")
                || $0.localizedCaseInsensitiveContains("failed")
                || $0.localizedCaseInsensitiveContains("invalid")
                || $0.localizedCaseInsensitiveContains("no backends")
        }
        let chosen = interesting.isEmpty ? Array(lines.suffix(3)) : Array(interesting.suffix(3))
        return chosen.joined(separator: " ")
    }

    static func create_context(path: String, options: LlamaRuntimeOptions) throws -> LlamaContext {
        try validateModelFile(path: path)
        try RuntimeLlama.prepare(for: path)

        let backend = LlamaModelBackend.forModel(path: path)
        print("Using \(backend == .prism ? "prism-llama.cpp" : "qvac-fabric-llm.cpp") backend")
        if backend == .prism {
            print("prism backend_reg_count=\(RuntimeLlama.prismBackendRegCount())")
        }

        RuntimeLlama.backendInit()

#if targetEnvironment(simulator)
        let defaultNgl: Int32 = 0
        print("Running on simulator, force use n_gpu_layers = 0")
#else
        let defaultNgl: Int32 = options.nGpuLayers >= 0 ? options.nGpuLayers : -1
#endif

        var loadAttempts: [(label: String, nGpuLayers: Int32, useMmap: Bool)] = [
            ("default", defaultNgl, true)
        ]

        if backend == .prism {
            loadAttempts.append(("cpu", 0, true))
            loadAttempts.append(("no-mmap", defaultNgl, false))
            loadAttempts.append(("cpu-no-mmap", 0, false))
        }

        var model: OpaquePointer?
        var lastDetail = ""
        for attempt in loadAttempts {
            if let loaded = RuntimeLlama.modelLoadWithOptions(
                path,
                nGpuLayers: attempt.nGpuLayers,
                useMmap: attempt.useMmap,
                backend: backend
            ) {
                model = loaded
                if attempt.label != "default" {
                    print("Loaded Bonsai model using \(attempt.label) fallback")
                }
                break
            }
            lastDetail = RuntimeLlama.lastModelLoadDetail(for: backend)
            if lastDetail.isEmpty {
                lastDetail = "Model load failed using \(attempt.label) settings"
            }
        }

        guard let model else {
            print("Could not load model at \(path)")
            let backendName = backend == .prism ? "prism-llama.cpp" : "qvac-fabric-llm.cpp"
            let summary = summarizeLoadLog(lastDetail)
            let reason = summary.isEmpty
                ? "Model load failed using \(backendName) at \(path)"
                : "Model load failed using \(backendName): \(summary)"
            throw LlamaError.couldNotInitializeContext(reason: reason)
        }

        let n_threads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        print("Using \(n_threads) threads")

        let context = RuntimeLlama.initFromModel(
            model,
            nCtx: UInt32(options.contextLength),
            nThreads: Int32(n_threads),
            flashAttention: options.flashAttention,
            backend: backend
        )
        guard let context else {
            RuntimeLlama.modelFree(model, backend: backend)
            print("Could not load context!")
            throw LlamaError.couldNotInitializeContext(reason: "Failed to create llama context")
        }

        return LlamaContext(model: model, context: context, options: options, backend: backend)
    }

    func updateSampler(options: LlamaRuntimeOptions) {
        runtimeOptions = options
        n_len = options.contextLength
        rebuildSamplerChain()
    }

    func model_info() -> String {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 256)
        result.initialize(repeating: Int8(0), count: 256)
        defer {
            result.deallocate()
        }

        let nChars = RuntimeLlama.modelDesc(model, result, 256)
        let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nChars))

        var SwiftString = ""
        for char in bufferPointer {
            SwiftString.append(Character(UnicodeScalar(UInt8(char))))
        }

        return SwiftString
    }

    func get_n_tokens() -> Int32 {
        return batch.n_tokens;
    }

    func completion_init(text: String) {
        print("attempting to complete \"\(text)\"")

        tokens_list = tokenize(text: text, add_bos: true)
        temporary_invalid_cchars = []

        let n_ctx = RuntimeLlama.nCtx(context)
        let n_kv_req = tokens_list.count + (Int(n_len) - tokens_list.count)

        print("\n n_len = \(n_len), n_ctx = \(n_ctx), n_kv_req = \(n_kv_req)")

        if n_kv_req > n_ctx {
            print("error: n_kv_req > n_ctx, the required KV cache size is not big enough")
        }

        for id in tokens_list {
            print(String(cString: token_to_piece(token: id) + [0]))
        }

        llama_batch_clear(&batch)

        for i1 in 0..<tokens_list.count {
            let i = Int(i1)
            llama_batch_add(&batch, tokens_list[i], Int32(i), [0], false)
        }
        batch.logits[Int(batch.n_tokens) - 1] = 1 // true

        if RuntimeLlama.decode(context, batch) != 0 {
            print("llama_decode() failed")
        }

        n_cur = batch.n_tokens
    }

    func completion_loop() -> String {
        var new_token_id: llama_token = 0

        guard let sampling else {
            return ""
        }

        new_token_id = RuntimeLlama.samplerSample(sampling, context, batch.n_tokens - 1)

        if RuntimeLlama.vocabIsEog(vocab, new_token_id) || n_cur == n_len {
            print("\n")
            is_done = true
            let new_token_str = String(cString: temporary_invalid_cchars + [0])
            temporary_invalid_cchars.removeAll()
            return new_token_str
        }

        let new_token_cchars = token_to_piece(token: new_token_id)
        temporary_invalid_cchars.append(contentsOf: new_token_cchars)
        let new_token_str: String
        if let string = String(validatingUTF8: temporary_invalid_cchars + [0]) {
            temporary_invalid_cchars.removeAll()
            new_token_str = string
        } else if (0 ..< temporary_invalid_cchars.count).contains(where: {$0 != 0 && String(validatingUTF8: Array(temporary_invalid_cchars.suffix($0)) + [0]) != nil}) {
            let string = String(cString: temporary_invalid_cchars + [0])
            temporary_invalid_cchars.removeAll()
            new_token_str = string
        } else {
            new_token_str = ""
        }
        print(new_token_str)

        llama_batch_clear(&batch)
        llama_batch_add(&batch, new_token_id, n_cur, [0], true)

        n_decode += 1
        n_cur    += 1

        if RuntimeLlama.decode(context, batch) != 0 {
            print("failed to evaluate llama!")
        }

        return new_token_str
    }

    func bench(pp: Int, tg: Int, pl: Int, nr: Int = 1) -> String {
        var pp_avg: Double = 0
        var tg_avg: Double = 0

        var pp_std: Double = 0
        var tg_std: Double = 0

        for _ in 0..<nr {
            llama_batch_clear(&batch)

            let n_tokens = pp

            for i in 0..<n_tokens {
                llama_batch_add(&batch, 0, Int32(i), [0], false)
            }
            batch.logits[Int(batch.n_tokens) - 1] = 1 // true

            RuntimeLlama.memoryClear(RuntimeLlama.getMemory(context), false)

            let t_pp_start = DispatchTime.now().uptimeNanoseconds / 1000;

            if RuntimeLlama.decode(context, batch) != 0 {
                print("llama_decode() failed during prompt")
            }
            RuntimeLlama.synchronize(context)

            let t_pp_end = DispatchTime.now().uptimeNanoseconds / 1000;

            RuntimeLlama.memoryClear(RuntimeLlama.getMemory(context), false)

            let t_tg_start = DispatchTime.now().uptimeNanoseconds / 1000;

            for i in 0..<tg {
                llama_batch_clear(&batch)

                for j in 0..<pl {
                    llama_batch_add(&batch, 0, Int32(i), [Int32(j)], true)
                }

                if RuntimeLlama.decode(context, batch) != 0 {
                    print("llama_decode() failed during text generation")
                }
                RuntimeLlama.synchronize(context)
            }

            let t_tg_end = DispatchTime.now().uptimeNanoseconds / 1000;

            RuntimeLlama.memoryClear(RuntimeLlama.getMemory(context), false)

            let t_pp = Double(t_pp_end - t_pp_start) / 1000000.0
            let t_tg = Double(t_tg_end - t_tg_start) / 1000000.0

            let speed_pp = Double(pp)    / t_pp
            let speed_tg = Double(pl*tg) / t_tg

            pp_avg += speed_pp
            tg_avg += speed_tg

            pp_std += speed_pp * speed_pp
            tg_std += speed_tg * speed_tg

            print("pp \(speed_pp) t/s, tg \(speed_tg) t/s")
        }

        pp_avg /= Double(nr)
        tg_avg /= Double(nr)

        if nr > 1 {
            pp_std = sqrt(pp_std / Double(nr - 1) - pp_avg * pp_avg * Double(nr) / Double(nr - 1))
            tg_std = sqrt(tg_std / Double(nr - 1) - tg_avg * tg_avg * Double(nr) / Double(nr - 1))
        } else {
            pp_std = 0
            tg_std = 0
        }

        let model_desc     = model_info();
        let model_size     = String(format: "%.2f GiB", Double(RuntimeLlama.modelSize(model)) / 1024.0 / 1024.0 / 1024.0);
        let model_n_params = String(format: "%.2f B", Double(RuntimeLlama.modelNParams(model)) / 1e9);
        let nGpu           = Int(runtimeOptions.nGpuLayers >= 0 ? runtimeOptions.nGpuLayers : 99)
        let backend        = nGpu == 0 ? "CPU" : "Metal";
        let pp_avg_str     = String(format: "%.2f", pp_avg);
        let tg_avg_str     = String(format: "%.2f", tg_avg);
        let pp_std_str     = String(format: "%.2f", pp_std);
        let tg_std_str     = String(format: "%.2f", tg_std);

        var result = ""

        result += String("| model | size | params | backend | test | t/s |\n")
        result += String("| --- | --- | --- | --- | --- | --- |\n")
        result += String("| \(model_desc) | \(model_size) | \(model_n_params) | \(backend) | pp \(pp) | \(pp_avg_str) ± \(pp_std_str) |\n")
        result += String("| \(model_desc) | \(model_size) | \(model_n_params) | \(backend) | tg \(tg) | \(tg_avg_str) ± \(tg_std_str) |\n")

        return result;
    }

    /// llama-bench compatible JSON array (pp + tg entries) for automation / Device Farm reporting.
    func benchJSON(pp: Int, tg: Int, pl: Int, nr: Int = 1) -> String {
        var ppAvgNs: Double = 0
        var tgAvgNs: Double = 0

        for _ in 0..<nr {
            llama_batch_clear(&batch)

            for i in 0..<pp {
                llama_batch_add(&batch, 0, Int32(i), [0], false)
            }
            batch.logits[Int(batch.n_tokens) - 1] = 1

            RuntimeLlama.memoryClear(RuntimeLlama.getMemory(context), false)

            let tPpStart = DispatchTime.now().uptimeNanoseconds
            if RuntimeLlama.decode(context, batch) != 0 {
                print("llama_decode() failed during prompt")
            }
            RuntimeLlama.synchronize(context)
            let tPpEnd = DispatchTime.now().uptimeNanoseconds

            RuntimeLlama.memoryClear(RuntimeLlama.getMemory(context), false)

            let tTgStart = DispatchTime.now().uptimeNanoseconds
            for i in 0..<tg {
                llama_batch_clear(&batch)
                for j in 0..<pl {
                    llama_batch_add(&batch, 0, Int32(i), [Int32(j)], true)
                }
                if RuntimeLlama.decode(context, batch) != 0 {
                    print("llama_decode() failed during text generation")
                }
                RuntimeLlama.synchronize(context)
            }
            let tTgEnd = DispatchTime.now().uptimeNanoseconds

            RuntimeLlama.memoryClear(RuntimeLlama.getMemory(context), false)

            ppAvgNs += Double(tPpEnd - tPpStart) / Double(nr)
            tgAvgNs += Double(tTgEnd - tTgStart) / Double(nr)
        }

        let nGpu = Int(runtimeOptions.nGpuLayers >= 0 ? runtimeOptions.nGpuLayers : 99)
        let backend = nGpu == 0 ? "CPU" : "Metal"
        let modelSize = Double(RuntimeLlama.modelSize(model))
        let ppTs = ppAvgNs > 0 ? Double(pp) / (ppAvgNs / 1e9) : 0
        let tgTs = tgAvgNs > 0 ? Double(pl * tg) / (tgAvgNs / 1e9) : 0

        let ppEntry: [String: Any] = [
            "n_prompt": pp,
            "n_gen": 0,
            "n_batch": pl,
            "n_threads": Int(RuntimeLlama.nThreads(context)),
            "avg_ns": Int(ppAvgNs),
            "stddev_ns": 0,
            "avg_ts": ppTs,
            "stddev_ts": 0,
            "n_gpu_layers": nGpu,
            "backend": backend,
            "model_size": modelSize,
        ]
        let tgEntry: [String: Any] = [
            "n_prompt": 0,
            "n_gen": tg,
            "n_batch": pl,
            "n_threads": Int(RuntimeLlama.nThreads(context)),
            "avg_ns": Int(tgAvgNs),
            "stddev_ns": 0,
            "avg_ts": tgTs,
            "stddev_ts": 0,
            "n_gpu_layers": nGpu,
            "backend": backend,
            "model_size": modelSize,
        ]

        guard let data = try? JSONSerialization.data(withJSONObject: [ppEntry, tgEntry], options: []),
              let json = String(data: data, encoding: .utf8) else {
            return "[]"
        }
        return json
    }

    func clear() {
        tokens_list.removeAll()
        temporary_invalid_cchars.removeAll()
        RuntimeLlama.memoryClear(RuntimeLlama.getMemory(context), true)
    }

    private func tokenize(text: String, add_bos: Bool) -> [llama_token] {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (add_bos ? 1 : 0) + 1
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: n_tokens)
        let tokenCount = RuntimeLlama.tokenize(vocab, text, Int32(utf8Count), tokens, Int32(n_tokens), add_bos, false)

        var swiftTokens: [llama_token] = []
        for i in 0..<tokenCount {
            swiftTokens.append(tokens[Int(i)])
        }

        tokens.deallocate()

        return swiftTokens
    }

    /// - note: The result does not contain null-terminator
    private func token_to_piece(token: llama_token) -> [CChar] {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 8)
        result.initialize(repeating: Int8(0), count: 8)
        defer {
            result.deallocate()
        }
        let nTokens = RuntimeLlama.tokenToPiece(vocab, token, result, 8, 0, false)

        if nTokens < 0 {
            let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: Int(-nTokens))
            newResult.initialize(repeating: Int8(0), count: Int(-nTokens))
            defer {
                newResult.deallocate()
            }
            let nNewTokens = RuntimeLlama.tokenToPiece(vocab, token, newResult, -nTokens, 0, false)
            let bufferPointer = UnsafeBufferPointer(start: newResult, count: Int(nNewTokens))
            return Array(bufferPointer)
        } else {
            let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nTokens))
            return Array(bufferPointer)
        }
    }
}

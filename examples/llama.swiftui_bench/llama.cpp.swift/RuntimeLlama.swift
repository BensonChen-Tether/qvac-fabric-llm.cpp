import Foundation
import llama

enum RuntimeLlama {
    private static var activeBackend: LlamaModelBackend = .qvac

    static func prepare(for path: String) throws {
        let backend = LlamaModelBackend.forModel(path: path)
        if backend == activeBackend {
            return
        }

        teardown()

        switch backend {
        case .qvac:
            activeBackend = .qvac
        case .prism:
            guard prism_llama_activate() else {
                let detail = String(cString: prism_llama_last_error())
                throw LlamaError.couldNotInitializeContext(
                    reason: detail.isEmpty
                        ? "Failed to load prism-llama.cpp backend for Bonsai models."
                        : detail
                )
            }
            activeBackend = .prism
        }
    }

    static func teardown() {
        if activeBackend == .prism, prism_llama_is_active() {
            prism_llama_deactivate()
        }
        activeBackend = .qvac
    }

    static var usesPrism: Bool {
        activeBackend == .prism
    }

    static func backendInit() {
        if usesPrism {
            prism_llama_backend_init()
        } else {
            llama_backend_init()
        }
    }

    static func backendFree() {
        if usesPrism {
            prism_llama_backend_free()
        } else {
            llama_backend_free()
        }
    }

    static func modelDefaultParams() -> llama_model_params {
        usesPrism ? prism_llama_model_default_params() : llama_model_default_params()
    }

    static func modelLoadWithOptions(
        _ path: String,
        nGpuLayers: Int32,
        useMmap: Bool,
        backend: LlamaModelBackend
    ) -> OpaquePointer? {
        if backend == .prism {
            return prism_llama_model_load_with_options(path, nGpuLayers, useMmap)
        }
        var params = llama_model_default_params()
        params.n_gpu_layers = nGpuLayers
        params.use_mmap = useMmap
        return llama_model_load_from_file(path, params)
    }

    static func lastModelLoadDetail(for backend: LlamaModelBackend) -> String {
        guard backend == .prism else { return "" }
        let err = String(cString: prism_llama_last_error())
        let log = String(cString: prism_llama_last_model_load_log())
        if !err.isEmpty {
            return err
        }
        return log
    }

    static func prismBackendRegCount() -> Int {
        Int(prism_llama_backend_reg_count())
    }

    static func modelFree(_ model: OpaquePointer?, backend: LlamaModelBackend? = nil) {
        guard let model else { return }
        let usePrism = backend == .prism || (backend == nil && usesPrism)
        if usePrism {
            prism_llama_model_free(model)
        } else {
            llama_model_free(model)
        }
    }

    static func modelDesc(_ model: OpaquePointer?, _ buf: UnsafeMutablePointer<Int8>, _ size: Int) -> Int32 {
        if usesPrism {
            return prism_llama_model_desc(model, buf, size)
        }
        return llama_model_desc(model, buf, size)
    }

    static func modelGetVocab(_ model: OpaquePointer?) -> OpaquePointer? {
        if usesPrism {
            return prism_llama_model_get_vocab(model)
        }
        return llama_model_get_vocab(model)
    }

    static func modelSize(_ model: OpaquePointer?) -> UInt64 {
        if usesPrism {
            return prism_llama_model_size(model)
        }
        return llama_model_size(model)
    }

    static func modelNParams(_ model: OpaquePointer?) -> UInt64 {
        if usesPrism {
            return prism_llama_model_n_params(model)
        }
        return llama_model_n_params(model)
    }

    static func contextDefaultParams() -> llama_context_params {
        usesPrism ? prism_llama_context_default_params() : llama_context_default_params()
    }

    static func initFromModel(
        _ model: OpaquePointer?,
        nCtx: UInt32,
        nThreads: Int32,
        flashAttention: Bool,
        backend: LlamaModelBackend
    ) -> OpaquePointer? {
        if backend == .prism {
            return prism_llama_init_context_with_options(model, nCtx, nThreads, flashAttention)
        }
        var params = llama_context_default_params()
        params.n_ctx = nCtx
        params.n_threads = nThreads
        params.n_threads_batch = nThreads
        params.flash_attn_type = flashAttention ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED
        return llama_init_from_model(model, params)
    }

    static func free(_ context: OpaquePointer?, backend: LlamaModelBackend? = nil) {
        guard let context else { return }
        let usePrism = backend == .prism || (backend == nil && usesPrism)
        if usePrism {
            prism_llama_free(context)
        } else {
            llama_free(context)
        }
    }

    static func backendFree(for backend: LlamaModelBackend) {
        if backend == .prism {
            prism_llama_backend_free()
        } else {
            llama_backend_free()
        }
    }

    static func nCtx(_ context: OpaquePointer?) -> UInt32 {
        if usesPrism {
            return prism_llama_n_ctx(context)
        }
        return llama_n_ctx(context)
    }

    static func nThreads(_ context: OpaquePointer?) -> Int32 {
        if usesPrism {
            return prism_llama_n_threads(context)
        }
        return llama_n_threads(context)
    }

    static func batchInit(_ nTokens: Int32, _ embd: Int32, _ nSeqMax: Int32) -> llama_batch {
        if usesPrism {
            return prism_llama_batch_init(nTokens, embd, nSeqMax)
        }
        return llama_batch_init(nTokens, embd, nSeqMax)
    }

    static func batchFree(_ batch: llama_batch) {
        if usesPrism {
            prism_llama_batch_free(batch)
        } else {
            llama_batch_free(batch)
        }
    }

    static func samplerChainDefaultParams() -> llama_sampler_chain_params {
        if usesPrism {
            return prism_llama_sampler_chain_default_params()
        }
        return llama_sampler_chain_default_params()
    }

    static func samplerChainInit(_ params: llama_sampler_chain_params) -> UnsafeMutablePointer<llama_sampler>? {
        if usesPrism {
            return prism_llama_sampler_chain_init(params)
        }
        return llama_sampler_chain_init(params)
    }

    static func samplerChainAdd(_ chain: UnsafeMutablePointer<llama_sampler>?, _ smpl: UnsafeMutablePointer<llama_sampler>?) {
        if usesPrism {
            prism_llama_sampler_chain_add(chain, smpl)
        } else {
            llama_sampler_chain_add(chain, smpl)
        }
    }

    static func samplerInitTopK(_ k: Int32) -> UnsafeMutablePointer<llama_sampler>? {
        if usesPrism {
            return prism_llama_sampler_init_top_k(k)
        }
        return llama_sampler_init_top_k(k)
    }

    static func samplerInitTopP(_ p: Float, _ minKeep: Int) -> UnsafeMutablePointer<llama_sampler>? {
        if usesPrism {
            return prism_llama_sampler_init_top_p(p, minKeep)
        }
        return llama_sampler_init_top_p(p, minKeep)
    }

    static func samplerInitTemp(_ t: Float) -> UnsafeMutablePointer<llama_sampler>? {
        if usesPrism {
            return prism_llama_sampler_init_temp(t)
        }
        return llama_sampler_init_temp(t)
    }

    static func samplerInitDist(_ seed: UInt32) -> UnsafeMutablePointer<llama_sampler>? {
        if usesPrism {
            return prism_llama_sampler_init_dist(seed)
        }
        return llama_sampler_init_dist(seed)
    }

    static func samplerFree(_ smpl: UnsafeMutablePointer<llama_sampler>?) {
        guard let smpl else { return }
        if usesPrism {
            prism_llama_sampler_free(smpl)
        } else {
            llama_sampler_free(smpl)
        }
    }

    static func samplerSample(_ smpl: UnsafeMutablePointer<llama_sampler>?, _ context: OpaquePointer?, _ idx: Int32) -> llama_token {
        if usesPrism {
            return prism_llama_sampler_sample(smpl, context, idx)
        }
        return llama_sampler_sample(smpl, context, idx)
    }

    static func decode(_ context: OpaquePointer?, _ batch: llama_batch) -> Int32 {
        if usesPrism {
            return prism_llama_decode(context, batch)
        }
        return llama_decode(context, batch)
    }

    static func synchronize(_ context: OpaquePointer?) {
        if usesPrism {
            prism_llama_synchronize(context)
        } else {
            llama_synchronize(context)
        }
    }

    static func getMemory(_ context: OpaquePointer?) -> llama_memory_t? {
        if usesPrism {
            return prism_llama_get_memory(context)
        }
        return llama_get_memory(context)
    }

    static func memoryClear(_ mem: llama_memory_t?, _ data: Bool) {
        if usesPrism {
            prism_llama_memory_clear(mem, data)
        } else {
            llama_memory_clear(mem, data)
        }
    }

    static func tokenize(
        _ vocab: OpaquePointer?,
        _ text: String,
        _ textLen: Int32,
        _ tokens: UnsafeMutablePointer<llama_token>,
        _ nTokensMax: Int32,
        _ addSpecial: Bool,
        _ parseSpecial: Bool
    ) -> Int32 {
        if usesPrism {
            return prism_llama_tokenize(vocab, text, textLen, tokens, nTokensMax, addSpecial, parseSpecial)
        }
        return llama_tokenize(vocab, text, textLen, tokens, nTokensMax, addSpecial, parseSpecial)
    }

    static func vocabIsEog(_ vocab: OpaquePointer?, _ token: llama_token) -> Bool {
        if usesPrism {
            return prism_llama_vocab_is_eog(vocab, token)
        }
        return llama_vocab_is_eog(vocab, token)
    }

    static func tokenToPiece(
        _ vocab: OpaquePointer?,
        _ token: llama_token,
        _ buf: UnsafeMutablePointer<Int8>,
        _ length: Int32,
        _ lstrip: Int32,
        _ special: Bool
    ) -> Int32 {
        if usesPrism {
            return prism_llama_token_to_piece(vocab, token, buf, length, lstrip, special)
        }
        return llama_token_to_piece(vocab, token, buf, length, lstrip, special)
    }
}

#import "PrismLlamaBridge.h"

#import <Foundation/Foundation.h>

#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace {

void * g_prism_handle = nullptr;
NSBundle * g_prism_bundle = nil;
std::string g_last_error;
std::string g_model_load_log;

static void set_last_error(NSString * message) {
    g_last_error = message.UTF8String ? message.UTF8String : "unknown prism_llama error";
    NSLog(@"prism_llama: %@", message);
}

#define PRISM_DECLARE_FN(name) static decltype(&name) p_##name = nullptr
#define PRISM_LOAD_FN(name) \
    do { \
        p_##name = reinterpret_cast<decltype(&name)>(dlsym(g_prism_handle, #name)); \
        if (!p_##name) { \
            const char * dl_error = dlerror(); \
            set_last_error([NSString stringWithFormat:@"Missing symbol %s (%s)", #name, dl_error ? dl_error : "unknown"]); \
            return false; \
        } \
    } while (0)

PRISM_DECLARE_FN(llama_backend_init);
PRISM_DECLARE_FN(llama_backend_free);
PRISM_DECLARE_FN(llama_log_set);
PRISM_DECLARE_FN(ggml_backend_load_all);
PRISM_DECLARE_FN(ggml_backend_reg_count);
PRISM_DECLARE_FN(llama_model_default_params);
PRISM_DECLARE_FN(llama_model_load_from_file);
PRISM_DECLARE_FN(llama_model_free);
PRISM_DECLARE_FN(llama_model_desc);
PRISM_DECLARE_FN(llama_model_get_vocab);
PRISM_DECLARE_FN(llama_model_size);
PRISM_DECLARE_FN(llama_model_n_params);
PRISM_DECLARE_FN(llama_context_default_params);
PRISM_DECLARE_FN(llama_init_from_model);
PRISM_DECLARE_FN(llama_free);
PRISM_DECLARE_FN(llama_n_ctx);
PRISM_DECLARE_FN(llama_n_threads);
PRISM_DECLARE_FN(llama_batch_init);
PRISM_DECLARE_FN(llama_batch_free);
PRISM_DECLARE_FN(llama_sampler_chain_default_params);
PRISM_DECLARE_FN(llama_sampler_chain_init);
PRISM_DECLARE_FN(llama_sampler_chain_add);
PRISM_DECLARE_FN(llama_sampler_init_top_k);
PRISM_DECLARE_FN(llama_sampler_init_top_p);
PRISM_DECLARE_FN(llama_sampler_init_temp);
PRISM_DECLARE_FN(llama_sampler_init_dist);
PRISM_DECLARE_FN(llama_sampler_free);
PRISM_DECLARE_FN(llama_sampler_sample);
PRISM_DECLARE_FN(llama_decode);
PRISM_DECLARE_FN(llama_synchronize);
PRISM_DECLARE_FN(llama_get_memory);
PRISM_DECLARE_FN(llama_memory_clear);
PRISM_DECLARE_FN(llama_tokenize);
PRISM_DECLARE_FN(llama_vocab_is_eog);
PRISM_DECLARE_FN(llama_token_to_piece);

static void reset_symbols() {
    p_llama_backend_init = nullptr;
    p_llama_backend_free = nullptr;
    p_llama_log_set = nullptr;
    p_ggml_backend_load_all = nullptr;
    p_ggml_backend_reg_count = nullptr;
    p_llama_model_default_params = nullptr;
    p_llama_model_load_from_file = nullptr;
    p_llama_model_free = nullptr;
    p_llama_model_desc = nullptr;
    p_llama_model_get_vocab = nullptr;
    p_llama_model_size = nullptr;
    p_llama_model_n_params = nullptr;
    p_llama_context_default_params = nullptr;
    p_llama_init_from_model = nullptr;
    p_llama_free = nullptr;
    p_llama_n_ctx = nullptr;
    p_llama_n_threads = nullptr;
    p_llama_batch_init = nullptr;
    p_llama_batch_free = nullptr;
    p_llama_sampler_chain_default_params = nullptr;
    p_llama_sampler_chain_init = nullptr;
    p_llama_sampler_chain_add = nullptr;
    p_llama_sampler_init_top_k = nullptr;
    p_llama_sampler_init_top_p = nullptr;
    p_llama_sampler_init_temp = nullptr;
    p_llama_sampler_init_dist = nullptr;
    p_llama_sampler_free = nullptr;
    p_llama_sampler_sample = nullptr;
    p_llama_decode = nullptr;
    p_llama_synchronize = nullptr;
    p_llama_get_memory = nullptr;
    p_llama_memory_clear = nullptr;
    p_llama_tokenize = nullptr;
    p_llama_vocab_is_eog = nullptr;
    p_llama_token_to_piece = nullptr;
}

static bool load_symbols() {
    PRISM_LOAD_FN(llama_backend_init);
    PRISM_LOAD_FN(llama_backend_free);
    PRISM_LOAD_FN(llama_log_set);
    PRISM_LOAD_FN(ggml_backend_load_all);
    PRISM_LOAD_FN(ggml_backend_reg_count);
    PRISM_LOAD_FN(llama_model_default_params);
    PRISM_LOAD_FN(llama_model_load_from_file);
    PRISM_LOAD_FN(llama_model_free);
    PRISM_LOAD_FN(llama_model_desc);
    PRISM_LOAD_FN(llama_model_get_vocab);
    PRISM_LOAD_FN(llama_model_size);
    PRISM_LOAD_FN(llama_model_n_params);
    PRISM_LOAD_FN(llama_context_default_params);
    PRISM_LOAD_FN(llama_init_from_model);
    PRISM_LOAD_FN(llama_free);
    PRISM_LOAD_FN(llama_n_ctx);
    PRISM_LOAD_FN(llama_n_threads);
    PRISM_LOAD_FN(llama_batch_init);
    PRISM_LOAD_FN(llama_batch_free);
    PRISM_LOAD_FN(llama_sampler_chain_default_params);
    PRISM_LOAD_FN(llama_sampler_chain_init);
    PRISM_LOAD_FN(llama_sampler_chain_add);
    PRISM_LOAD_FN(llama_sampler_init_top_k);
    PRISM_LOAD_FN(llama_sampler_init_top_p);
    PRISM_LOAD_FN(llama_sampler_init_temp);
    PRISM_LOAD_FN(llama_sampler_init_dist);
    PRISM_LOAD_FN(llama_sampler_free);
    PRISM_LOAD_FN(llama_sampler_sample);
    PRISM_LOAD_FN(llama_decode);
    PRISM_LOAD_FN(llama_synchronize);
    PRISM_LOAD_FN(llama_get_memory);
    PRISM_LOAD_FN(llama_memory_clear);
    PRISM_LOAD_FN(llama_tokenize);
    PRISM_LOAD_FN(llama_vocab_is_eog);
    PRISM_LOAD_FN(llama_token_to_piece);
    return true;
}

static NSString * prism_framework_path() {
    NSBundle * main_bundle = [NSBundle mainBundle];
    NSString * framework_path = [main_bundle pathForResource:@"prism_llama"
                                                      ofType:@"framework"
                                                 inDirectory:@"Frameworks"];
    if (framework_path) {
        return framework_path;
    }

    framework_path = [main_bundle pathForResource:@"prism_llama" ofType:@"framework"];
    if (framework_path) {
        return framework_path;
    }

    NSString * frameworks_dir = [[main_bundle bundlePath] stringByAppendingPathComponent:@"Frameworks/prism_llama.framework"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:frameworks_dir]) {
        return frameworks_dir;
    }

    return nil;
}

static void prism_log_callback(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (!text) {
        return;
    }
    g_model_load_log += text;
    if (g_model_load_log.size() > 8192) {
        g_model_load_log.erase(0, g_model_load_log.size() - 8192);
    }
    NSLog(@"prism_llama[%d]: %s", static_cast<int>(level), text);
}

static void ensure_prism_backends_ready() {
    if (!p_ggml_backend_reg_count) {
        return;
    }

    size_t count = p_ggml_backend_reg_count();
    NSLog(@"prism_llama: backend_reg_count=%zu", count);
    if (count == 0 && p_ggml_backend_load_all) {
        p_ggml_backend_load_all();
        count = p_ggml_backend_reg_count();
        NSLog(@"prism_llama: backend_reg_count after ggml_backend_load_all=%zu", count);
        if (count == 0) {
            set_last_error(@"No GGML backends available in prism-llama.cpp");
        }
    }
}

static NSString * prism_binary_path(NSString * framework_path) {
    NSBundle * framework_bundle = [NSBundle bundleWithPath:framework_path];
    if (!framework_bundle) {
        return [framework_path stringByAppendingPathComponent:@"prism_llama"];
    }

    NSString * executable_path = [framework_bundle executablePath];
    if (executable_path.length > 0) {
        return executable_path;
    }

    return [framework_path stringByAppendingPathComponent:@"prism_llama"];
}

}  // namespace

const char * prism_llama_last_error(void) {
    return g_last_error.c_str();
}

const char * prism_llama_last_model_load_log(void) {
    return g_model_load_log.c_str();
}

size_t prism_llama_backend_reg_count(void) {
    if (!p_ggml_backend_reg_count) {
        return 0;
    }
    return p_ggml_backend_reg_count();
}

bool prism_llama_is_active(void) {
    return g_prism_handle != nullptr;
}

bool prism_llama_activate(void) {
    if (g_prism_handle) {
        return true;
    }

    NSString * framework_path = prism_framework_path();
    if (!framework_path) {
        set_last_error(@"prism_llama.framework not found in app bundle");
        return false;
    }

    g_prism_bundle = [NSBundle bundleWithPath:framework_path];
    if (!g_prism_bundle) {
        set_last_error([NSString stringWithFormat:@"Failed to create bundle for %@", framework_path]);
        return false;
    }

    NSError * bundle_error = nil;
    if (![g_prism_bundle loadAndReturnError:&bundle_error]) {
        NSString * message = bundle_error.localizedDescription ?: @"NSBundle load failed";
        set_last_error([NSString stringWithFormat:@"Failed to load %@: %@", framework_path, message]);
        g_prism_bundle = nil;
        return false;
    }

    NSString * binary_path = prism_binary_path(framework_path);
    g_prism_handle = dlopen(binary_path.UTF8String, RTLD_NOW | RTLD_LOCAL);
    if (!g_prism_handle) {
        const char * dl_error = dlerror();
        set_last_error([NSString stringWithFormat:@"dlopen failed for %@ (%s)", binary_path, dl_error ? dl_error : "unknown"]);
        [g_prism_bundle unload];
        g_prism_bundle = nil;
        return false;
    }

    if (!load_symbols()) {
        dlclose(g_prism_handle);
        g_prism_handle = nullptr;
        [g_prism_bundle unload];
        g_prism_bundle = nil;
        reset_symbols();
        if (g_last_error.empty()) {
            set_last_error(@"Failed to resolve prism llama symbols");
        }
        return false;
    }

    g_last_error.clear();
    return true;
}

void prism_llama_deactivate(void) {
    if (!g_prism_handle) {
        return;
    }
    dlclose(g_prism_handle);
    g_prism_handle = nullptr;
    if (g_prism_bundle) {
        [g_prism_bundle unload];
        g_prism_bundle = nil;
    }
    reset_symbols();
}

#define PRISM_DISPATCH0(ret, name) \
    ret prism_##name(void) { \
        return p_##name(); \
    }

#define PRISM_DISPATCH1(ret, name, t1, a1) \
    ret prism_##name(t1 a1) { \
        return p_##name(a1); \
    }

#define PRISM_DISPATCH2(ret, name, t1, a1, t2, a2) \
    ret prism_##name(t1 a1, t2 a2) { \
        return p_##name(a1, a2); \
    }

#define PRISM_DISPATCH3(ret, name, t1, a1, t2, a2, t3, a3) \
    ret prism_##name(t1 a1, t2 a2, t3 a3) { \
        return p_##name(a1, a2, a3); \
    }

#define PRISM_DISPATCH4(ret, name, t1, a1, t2, a2, t3, a3, t4, a4) \
    ret prism_##name(t1 a1, t2 a2, t3 a3, t4 a4) { \
        return p_##name(a1, a2, a3, a4); \
    }

#define PRISM_DISPATCH6(ret, name, t1, a1, t2, a2, t3, a3, t4, a4, t5, a5, t6, a6) \
    ret prism_##name(t1 a1, t2 a2, t3 a3, t4 a4, t5 a5, t6 a6) { \
        return p_##name(a1, a2, a3, a4, a5, a6); \
    }

#define PRISM_DISPATCH7(ret, name, t1, a1, t2, a2, t3, a3, t4, a4, t5, a5, t6, a6, t7, a7) \
    ret prism_##name(t1 a1, t2 a2, t3 a3, t4 a4, t5 a5, t6 a6, t7 a7) { \
        return p_##name(a1, a2, a3, a4, a5, a6, a7); \
    }

void prism_llama_backend_init(void) {
    p_llama_backend_init();
    ensure_prism_backends_ready();
}

PRISM_DISPATCH0(void, llama_backend_free);
PRISM_DISPATCH0(struct llama_model_params, llama_model_default_params);

static struct llama_model * prism_model_load_internal(const char * path, struct llama_model_params params) {
    g_model_load_log.clear();
    if (p_llama_log_set) {
        p_llama_log_set(prism_log_callback, nullptr);
    }
    ensure_prism_backends_ready();

    if (p_ggml_backend_reg_count && p_ggml_backend_reg_count() == 0) {
        set_last_error(@"prism-llama.cpp has no loaded GGML backends");
        return nullptr;
    }

    struct llama_model * model = nullptr;
    try {
        model = p_llama_model_load_from_file(path, params);
    } catch (const std::exception & err) {
        set_last_error([NSString stringWithFormat:@"Model load exception: %s", err.what()]);
        return nullptr;
    } catch (...) {
        set_last_error(@"Model load threw an unknown exception");
        return nullptr;
    }

    if (!model) {
        NSString * detail = nil;
        if (!g_model_load_log.empty()) {
            detail = [NSString stringWithUTF8String:g_model_load_log.c_str()];
        }
        if (detail.length == 0) {
            detail = @"llama_model_load_from_file returned null";
        }
        set_last_error(detail);
    }

    return model;
}

struct llama_model * prism_llama_model_load_with_options(
        const char * path,
        int32_t n_gpu_layers,
        bool use_mmap) {
    struct llama_model_params params = p_llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    params.use_mmap = use_mmap;
    return prism_model_load_internal(path, params);
}
PRISM_DISPATCH1(void, llama_model_free, struct llama_model *, model);
PRISM_DISPATCH3(int32_t, llama_model_desc, const struct llama_model *, model, char *, buf, size_t, buf_size);
PRISM_DISPATCH1(const struct llama_vocab *, llama_model_get_vocab, const struct llama_model *, model);
PRISM_DISPATCH1(uint64_t, llama_model_size, const struct llama_model *, model);
PRISM_DISPATCH1(uint64_t, llama_model_n_params, const struct llama_model *, model);
PRISM_DISPATCH0(struct llama_context_params, llama_context_default_params);

struct llama_context * prism_llama_init_context_with_options(
        struct llama_model * model,
        uint32_t n_ctx,
        int32_t n_threads,
        bool flash_attention) {
    struct llama_context_params params = p_llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_threads = n_threads;
    params.n_threads_batch = n_threads;
    params.flash_attn_type = flash_attention
        ? LLAMA_FLASH_ATTN_TYPE_ENABLED
        : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    return p_llama_init_from_model(model, params);
}
PRISM_DISPATCH1(void, llama_free, struct llama_context *, ctx);
PRISM_DISPATCH1(uint32_t, llama_n_ctx, const struct llama_context *, ctx);
PRISM_DISPATCH1(int32_t, llama_n_threads, struct llama_context *, ctx);
PRISM_DISPATCH3(struct llama_batch, llama_batch_init, int32_t, n_tokens, int32_t, embd, int32_t, n_seq_max);
PRISM_DISPATCH1(void, llama_batch_free, struct llama_batch, batch);
PRISM_DISPATCH0(struct llama_sampler_chain_params, llama_sampler_chain_default_params);
PRISM_DISPATCH1(struct llama_sampler *, llama_sampler_chain_init, struct llama_sampler_chain_params, params);
PRISM_DISPATCH2(void, llama_sampler_chain_add, struct llama_sampler *, chain, struct llama_sampler *, smpl);
PRISM_DISPATCH1(struct llama_sampler *, llama_sampler_init_top_k, int32_t, k);
PRISM_DISPATCH2(struct llama_sampler *, llama_sampler_init_top_p, float, p, size_t, min_keep);
PRISM_DISPATCH1(struct llama_sampler *, llama_sampler_init_temp, float, t);
PRISM_DISPATCH1(struct llama_sampler *, llama_sampler_init_dist, uint32_t, seed);
PRISM_DISPATCH1(void, llama_sampler_free, struct llama_sampler *, smpl);
PRISM_DISPATCH3(llama_token, llama_sampler_sample, struct llama_sampler *, smpl, struct llama_context *, ctx, int32_t, idx);
PRISM_DISPATCH2(int32_t, llama_decode, struct llama_context *, ctx, struct llama_batch, batch);
PRISM_DISPATCH1(void, llama_synchronize, struct llama_context *, ctx);
PRISM_DISPATCH1(llama_memory_t, llama_get_memory, const struct llama_context *, ctx);
PRISM_DISPATCH2(void, llama_memory_clear, llama_memory_t, mem, bool, data);
PRISM_DISPATCH7(int32_t, llama_tokenize, const struct llama_vocab *, vocab, const char *, text, int32_t, text_len, llama_token *, tokens, int32_t, n_tokens_max, bool, add_special, bool, parse_special);
PRISM_DISPATCH2(bool, llama_vocab_is_eog, const struct llama_vocab *, vocab, llama_token, token);
PRISM_DISPATCH6(int32_t, llama_token_to_piece, const struct llama_vocab *, vocab, llama_token, token, char *, buf, int32_t, length, int32_t, lstrip, bool, special);

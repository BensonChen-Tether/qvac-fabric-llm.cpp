#pragma once

#include <stdbool.h>
#include <stdint.h>

#include <llama/llama.h>

#ifdef __cplusplus
extern "C" {
#endif

bool prism_llama_is_active(void);
bool prism_llama_activate(void);
void prism_llama_deactivate(void);
const char * prism_llama_last_error(void);
const char * prism_llama_last_model_load_log(void);
size_t prism_llama_backend_reg_count(void);

void prism_llama_backend_init(void);
void prism_llama_backend_free(void);

struct llama_model_params prism_llama_model_default_params(void);
struct llama_model * prism_llama_model_load_with_options(
    const char * path,
    int32_t n_gpu_layers,
    bool use_mmap);
void prism_llama_model_free(struct llama_model * model);
int32_t prism_llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size);
const struct llama_vocab * prism_llama_model_get_vocab(const struct llama_model * model);
uint64_t prism_llama_model_size(const struct llama_model * model);
uint64_t prism_llama_model_n_params(const struct llama_model * model);

struct llama_context_params prism_llama_context_default_params(void);
struct llama_context * prism_llama_init_context_with_options(
    struct llama_model * model,
    uint32_t n_ctx,
    int32_t n_threads,
    bool flash_attention);
void prism_llama_free(struct llama_context * ctx);
uint32_t prism_llama_n_ctx(const struct llama_context * ctx);
int32_t prism_llama_n_threads(struct llama_context * ctx);

struct llama_batch prism_llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
void prism_llama_batch_free(struct llama_batch batch);

struct llama_sampler_chain_params prism_llama_sampler_chain_default_params(void);
struct llama_sampler * prism_llama_sampler_chain_init(struct llama_sampler_chain_params params);
void prism_llama_sampler_chain_add(struct llama_sampler * chain, struct llama_sampler * smpl);
struct llama_sampler * prism_llama_sampler_init_top_k(int32_t k);
struct llama_sampler * prism_llama_sampler_init_top_p(float p, size_t min_keep);
struct llama_sampler * prism_llama_sampler_init_temp(float t);
struct llama_sampler * prism_llama_sampler_init_dist(uint32_t seed);
void prism_llama_sampler_free(struct llama_sampler * smpl);
llama_token prism_llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx);

int32_t prism_llama_decode(struct llama_context * ctx, struct llama_batch batch);
void prism_llama_synchronize(struct llama_context * ctx);

llama_memory_t prism_llama_get_memory(const struct llama_context * ctx);
void prism_llama_memory_clear(llama_memory_t mem, bool data);

int32_t prism_llama_tokenize(
    const struct llama_vocab * vocab,
    const char * text,
    int32_t text_len,
    llama_token * tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special);

bool prism_llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token);

int32_t prism_llama_token_to_piece(
    const struct llama_vocab * vocab,
    llama_token token,
    char * buf,
    int32_t length,
    int32_t lstrip,
    bool special);

#ifdef __cplusplus
}
#endif

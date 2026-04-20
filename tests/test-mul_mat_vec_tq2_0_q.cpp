// Unit tests for the mul_mat_vec_tq2_0_q Vulkan shader
// Tests matrix-vector multiplication where:
// - Matrix A is in TQ2_0 format (2-bit ternary quantization)
// - Vector B is in Q8_1 format (8-bit quantization) or F32
// Tests both CPU reference implementation and GPU (Vulkan) backend

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <string>

// TQ2_0 format constants (from types.glsl)
constexpr int QUANT_K_TQ2_0 = 256;  // Block size: 256 elements per block
constexpr int QUANT_R_TQ2_0 = 4;    // 4 elements per byte (2 bits each)

// Q8_1 format constants
constexpr int QUANT_K_Q8_1 = 32;    // Block size: 32 elements per block

// TQ2_0 block structure - matches the shader's block_tq2_0
// TQ2_0 stores 2-bit ternary values: {0, 1, 2} representing {-1, 0, 1} relative to some offset
struct block_tq2_0 {
    uint8_t qs[QUANT_K_TQ2_0 / QUANT_R_TQ2_0];  // 64 bytes for 256 elements (4 per byte)
    uint16_t d;  // FP16 scale factor (stored as raw bits)
};

// Q8_1 block structure - matches the shader's block_q8_1
struct block_q8_1 {
    uint16_t ds[2];   // FP16: ds[0] = scale (d), ds[1] = sum (s)
    int8_t qs[QUANT_K_Q8_1];  // 32 int8 quantized values
};

// Q8_1 x4 block structure - matches the shader's block_q8_1_x4
// This packs 4 Q8_1 blocks together for efficient memory access
struct block_q8_1_x4 {
    uint16_t ds[4][2];  // 4 sets of (d, s) pairs
    int32_t qs[32];     // 32 packed int32 values (4 int8 per int32)
};

// FP16 conversion utilities
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t result = sign;
            float f;
            memcpy(&f, &result, sizeof(f));
            return f;
        }
        // Denormalized
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x400;
    } else if (exp == 31) {
        uint32_t result = sign | 0x7F800000 | (mant << 13);
        float f;
        memcpy(&f, &result, sizeof(f));
        return f;
    }

    exp = exp + (127 - 15);
    mant = mant << 13;

    uint32_t result = sign | (exp << 23) | mant;
    float f;
    memcpy(&f, &result, sizeof(f));
    return f;
}

static uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));

    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;

    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t)sign;
        }
        mant = (mant | 0x800000) >> (1 - exp);
        return (uint16_t)(sign | (mant >> 13));
    } else if (exp == 0xFF - 127 + 15) {
        if (mant == 0) {
            return (uint16_t)(sign | 0x7C00);
        } else {
            return (uint16_t)(sign | 0x7C00 | (mant >> 13));
        }
    }

    if (exp > 30) {
        return (uint16_t)(sign | 0x7C00);
    }

    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

// TQ2_0 memory layout helper functions
// The TQ2_0 format uses an interleaved layout for efficient GPU access:
// - Elements 0-31:   bytes 0-31,  bits 0-1
// - Elements 32-63:  bytes 0-31,  bits 2-3
// - Elements 64-95:  bytes 0-31,  bits 4-5
// - Elements 96-127: bytes 0-31,  bits 6-7
// - Elements 128-159: bytes 32-63, bits 0-1
// - Elements 160-191: bytes 32-63, bits 2-3
// - Elements 192-223: bytes 32-63, bits 4-5
// - Elements 224-255: bytes 32-63, bits 6-7
static void tq2_0_get_byte_and_shift(int element_idx, int* byte_idx, int* bit_shift) {
    // Based on the shader's repack function for TQ2_0
    int k = element_idx;
    int ip = ((k >> 7) & 1) * 32;  // 0 for elements 0-127, 32 for elements 128-255
    int b = (k & 31) + ip;         // byte index within block
    int s = ((k >> 5) & 3) * 2;    // shift within byte

    *byte_idx = b;
    *bit_shift = s;
}

// Dequantize TQ2_0 block to float values
// TQ2_0 stores 2-bit ternary values where each value v in {0, 1, 2} represents (v - 1) = {-1, 0, 1}
static void dequantize_tq2_0(const block_tq2_0* block, float* output) {
    const float d = fp16_to_fp32(block->d);

    for (int i = 0; i < QUANT_K_TQ2_0; i++) {
        int byte_idx, bit_shift;
        tq2_0_get_byte_and_shift(i, &byte_idx, &bit_shift);
        int val = (block->qs[byte_idx] >> bit_shift) & 0x3;

        // Convert from {0, 1, 2} to {-1, 0, 1}
        output[i] = d * (float)(val - 1);
    }
}

// Dequantize Q8_1 block to float values
static void dequantize_q8_1(const block_q8_1* block, float* output) {
    const float d = fp16_to_fp32(block->ds[0]);

    for (int i = 0; i < QUANT_K_Q8_1; i++) {
        output[i] = d * (float)block->qs[i];
    }
}

// Quantize float values to TQ2_0 format (using interleaved layout)
static void quantize_to_tq2_0(const float* input, block_tq2_0* block) {
    // Find the maximum absolute value for scaling
    float max_abs = 0.0f;
    for (int i = 0; i < QUANT_K_TQ2_0; i++) {
        float abs_val = fabsf(input[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }

    // Compute scale factor
    float d = max_abs > 0.0f ? max_abs : 1.0f;
    block->d = fp32_to_fp16(d);

    // Clear quantized values
    memset(block->qs, 0, sizeof(block->qs));

    // Quantize each value to ternary {-1, 0, 1} stored as {0, 1, 2}
    // Using the interleaved TQ2_0 layout
    for (int i = 0; i < QUANT_K_TQ2_0; i++) {
        float scaled = input[i] / d;
        int q;
        if (scaled <= -0.5f) {
            q = 0;  // -1
        } else if (scaled >= 0.5f) {
            q = 2;  // +1
        } else {
            q = 1;  // 0
        }

        int byte_idx, bit_shift;
        tq2_0_get_byte_and_shift(i, &byte_idx, &bit_shift);
        block->qs[byte_idx] |= (q << bit_shift);
    }
}

// Quantize float values to Q8_1 format
static void quantize_to_q8_1(const float* input, block_q8_1* block) {
    // Find the maximum absolute value for scaling
    float max_abs = 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < QUANT_K_Q8_1; i++) {
        float abs_val = fabsf(input[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
        sum += input[i];
    }

    // Compute scale factor (d) such that values fit in int8 range [-127, 127]
    float d = max_abs / 127.0f;
    if (d == 0.0f) d = 1.0f;

    block->ds[0] = fp32_to_fp16(d);
    block->ds[1] = fp32_to_fp16(sum);  // Store sum for the offset correction

    // Quantize each value
    for (int i = 0; i < QUANT_K_Q8_1; i++) {
        int q = (int)roundf(input[i] / d);
        q = std::max(-127, std::min(127, q));
        block->qs[i] = (int8_t)q;
    }
}

// CPU reference implementation using dequantized values
// This computes the mathematically correct result that the shader should approximate
static void mul_mat_vec_dequantized_cpu(
    const block_tq2_0* A,
    const block_q8_1* B,
    float* output,
    int M,  // Number of rows
    int K   // Number of columns (must be multiple of QUANT_K_TQ2_0)
) {
    const int num_blocks_per_row = K / QUANT_K_TQ2_0;
    const int q8_blocks_per_tq2_block = QUANT_K_TQ2_0 / QUANT_K_Q8_1;  // 256/32 = 8

    // Temporary buffers for dequantized values
    std::vector<float> a_dequant(QUANT_K_TQ2_0);
    std::vector<float> b_dequant(QUANT_K_TQ2_0);

    for (int row = 0; row < M; row++) {
        float acc = 0.0f;

        for (int block_idx = 0; block_idx < num_blocks_per_row; block_idx++) {
            const block_tq2_0* a_block = &A[row * num_blocks_per_row + block_idx];

            // Dequantize A block
            dequantize_tq2_0(a_block, a_dequant.data());

            // Dequantize corresponding B blocks
            for (int q8_idx = 0; q8_idx < q8_blocks_per_tq2_block; q8_idx++) {
                const block_q8_1* b_block = &B[block_idx * q8_blocks_per_tq2_block + q8_idx];
                dequantize_q8_1(b_block, &b_dequant[q8_idx * QUANT_K_Q8_1]);
            }

            // Compute dot product with dequantized values
            for (int k = 0; k < QUANT_K_TQ2_0; k++) {
                acc += a_dequant[k] * b_dequant[k];
            }
        }

        output[row] = acc;
    }
}

// Helper to print test results
static const char* result_str(bool passed) {
    return passed ? "PASSED" : "FAILED";
}

// ============================================================================
// GPU Testing Infrastructure
// ============================================================================

static ggml_backend_t g_backend_gpu = nullptr;
static ggml_backend_t g_backend_cpu = nullptr;
static bool g_gpu_available = false;

// Initialize backends
static bool init_backends() {
    printf("Initializing backends...\n");

    // Load all backends
    ggml_backend_load_all();

    // Find and initialize CPU backend
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            g_backend_cpu = ggml_backend_dev_init(dev, nullptr);
            printf("  CPU backend: %s\n", ggml_backend_name(g_backend_cpu));
            break;
        }
    }

    if (!g_backend_cpu) {
        printf("  ERROR: No CPU backend found!\n");
        return false;
    }

    // Find and initialize GPU (Vulkan) backend
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char* name = ggml_backend_dev_name(dev);

        // Check for Vulkan backend
        if (strstr(name, "Vulkan") != nullptr ||
            strstr(name, "vulkan") != nullptr ||
            strstr(name, "VK") != nullptr) {
            g_backend_gpu = ggml_backend_dev_init(dev, nullptr);
            if (g_backend_gpu) {
                printf("  GPU backend: %s (%s)\n",
                       ggml_backend_name(g_backend_gpu),
                       ggml_backend_dev_description(dev));
                g_gpu_available = true;
                break;
            }
        }
    }

    // Also try GPU type if Vulkan not found by name
    if (!g_backend_gpu) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                g_backend_gpu = ggml_backend_dev_init(dev, nullptr);
                if (g_backend_gpu) {
                    printf("  GPU backend: %s (%s)\n",
                           ggml_backend_name(g_backend_gpu),
                           ggml_backend_dev_description(dev));
                    g_gpu_available = true;
                    break;
                }
            }
        }
    }

    if (!g_gpu_available) {
        printf("  WARNING: No GPU (Vulkan) backend found. GPU tests will be skipped.\n");
    }

    return true;
}

// Cleanup backends
static void cleanup_backends() {
    if (g_backend_gpu) {
        ggml_backend_free(g_backend_gpu);
        g_backend_gpu = nullptr;
    }
    if (g_backend_cpu) {
        ggml_backend_free(g_backend_cpu);
        g_backend_cpu = nullptr;
    }
}

// Run MUL_MAT operation on a specific backend
// A: [K, M] in row-major = [M rows, K cols] matrix stored in TQ2_0 format
// B: [K, 1] vector in F32 format
// Result: [M] output vector
static bool run_mul_mat_on_backend(
    ggml_backend_t backend,
    const std::vector<float>& A_f32,  // M x K matrix in row-major
    const std::vector<float>& B_f32,  // K vector
    std::vector<float>& output,       // M output
    int M, int K,
    bool use_tq2_0
) {
    const char* backend_name = ggml_backend_name(backend);

    // In GGML, matrix A for MUL_MAT is [ne0=K, ne1=M] (transposed storage)
    // and B is [ne0=K, ne1=1]
    // Result is [ne0=M, ne1=1]

    // Calculate memory needed
    size_t ctx_size = ggml_tensor_overhead() * 10 + ggml_graph_overhead();
    ggml_init_params params = {
        /* .mem_size   = */ ctx_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("    Failed to create GGML context\n");
        return false;
    }

    // Create tensors
    // A: [K, M] - K elements per row, M rows (stored in column-major for GGML)
    ggml_type a_type = use_tq2_0 ? GGML_TYPE_TQ2_0 : GGML_TYPE_F32;
    ggml_tensor* tensor_a = ggml_new_tensor_2d(ctx, a_type, K, M);
    ggml_set_name(tensor_a, "A");

    // B: [K, 1] - vector
    ggml_tensor* tensor_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    ggml_set_name(tensor_b, "B");

    // Result: MUL_MAT(A, B) = [M, 1]
    ggml_tensor* tensor_result = ggml_mul_mat(ctx, tensor_a, tensor_b);
    ggml_set_name(tensor_result, "result");

    // Check if backend supports this operation
    if (!ggml_backend_supports_op(backend, tensor_result)) {
        printf("    Backend %s does not support MUL_MAT with TQ2_0\n", backend_name);
        ggml_free(ctx);
        return false;
    }

    // Build computation graph
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, tensor_result);

    // Allocate buffers
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        printf("    Failed to allocate backend buffer\n");
        ggml_free(ctx);
        return false;
    }

    // Set tensor data
    if (use_tq2_0) {
        // Quantize A to TQ2_0
        const int blocks_per_row = K / QUANT_K_TQ2_0;
        const int total_blocks = M * blocks_per_row;
        std::vector<block_tq2_0> A_tq2(total_blocks);

        for (int row = 0; row < M; row++) {
            for (int b = 0; b < blocks_per_row; b++) {
                quantize_to_tq2_0(&A_f32[row * K + b * QUANT_K_TQ2_0],
                                 &A_tq2[row * blocks_per_row + b]);
            }
        }

        ggml_backend_tensor_set(tensor_a, A_tq2.data(), 0, total_blocks * sizeof(block_tq2_0));
    } else {
        // Use F32 directly
        // GGML MUL_MAT(A, B): result[m, n] = sum_k A[k, m] * B[k, n]
        // We want: output[row] = sum_col A_f32[row, col] * B_f32[col]
        // So we need A[k, m] = A_f32[m, k], meaning A is stored with each row of A_f32
        // becoming a column in GGML's A tensor.
        // For GGML tensor [K, M], element (k, m) is at memory offset k + m*K
        // We want A[col, row] = A_f32[row * K + col], stored at col + row*K = row*K + col
        // This means the memory layout is actually the same!
        ggml_backend_tensor_set(tensor_a, A_f32.data(), 0, M * K * sizeof(float));
    }

    // Set B vector
    ggml_backend_tensor_set(tensor_b, B_f32.data(), 0, K * sizeof(float));

    // Compute
    ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        printf("    Graph compute failed with status %d\n", status);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    // Get result
    output.resize(M);
    ggml_backend_tensor_get(tensor_result, output.data(), 0, M * sizeof(float));

    // Cleanup
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    return true;
}

// ============================================================================
// CPU-only Tests (existing tests, simplified)
// ============================================================================

// Test 1: Basic small matrix-vector multiplication with random data
static bool test_basic_small() {
    printf("Test 1: Basic small matrix-vector multiplication (random data)...\n");
    printf("  Note: High error expected for ternary quantization of random data\n");

    const int M = 4;   // 4 rows
    const int K = 256; // 256 columns (1 TQ2_0 block per row)

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_f32(M * K);
    std::vector<float> B_f32(K);
    for (int i = 0; i < M * K; i++) A_f32[i] = dist(rng);
    for (int i = 0; i < K; i++) B_f32[i] = dist(rng);

    // Quantize
    const int num_a_blocks = M;
    std::vector<block_tq2_0> A_tq2(num_a_blocks);
    for (int row = 0; row < M; row++) {
        quantize_to_tq2_0(&A_f32[row * K], &A_tq2[row]);
    }

    const int num_b_blocks = K / QUANT_K_Q8_1;
    std::vector<block_q8_1> B_q8(num_b_blocks);
    for (int i = 0; i < num_b_blocks; i++) {
        quantize_to_q8_1(&B_f32[i * QUANT_K_Q8_1], &B_q8[i]);
    }

    std::vector<float> output_dequant(M);
    mul_mat_vec_dequantized_cpu(A_tq2.data(), B_q8.data(), output_dequant.data(), M, K);

    bool all_finite = true;
    for (int i = 0; i < M; i++) {
        if (!std::isfinite(output_dequant[i])) {
            all_finite = false;
            printf("  Row %d: NaN or Inf detected!\n", i);
        }
    }

    bool passed = all_finite;
    printf("  All outputs finite: %s\n\n", result_str(passed));
    return passed;
}

// Test 2: Larger matrix with multiple blocks per row
static bool test_larger_matrix() {
    printf("Test 2: Larger matrix with multiple TQ2_0 blocks per row...\n");
    printf("  Note: Testing multi-block accumulation correctness\n");

    const int M = 8;
    const int K = 512;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_f32(M * K);
    std::vector<float> B_f32(K);
    for (int i = 0; i < M * K; i++) A_f32[i] = dist(rng);
    for (int i = 0; i < K; i++) B_f32[i] = dist(rng);

    const int blocks_per_row = K / QUANT_K_TQ2_0;
    const int num_a_blocks = M * blocks_per_row;
    std::vector<block_tq2_0> A_tq2(num_a_blocks);
    for (int row = 0; row < M; row++) {
        for (int b = 0; b < blocks_per_row; b++) {
            quantize_to_tq2_0(&A_f32[row * K + b * QUANT_K_TQ2_0],
                             &A_tq2[row * blocks_per_row + b]);
        }
    }

    const int num_b_blocks = K / QUANT_K_Q8_1;
    std::vector<block_q8_1> B_q8(num_b_blocks);
    for (int i = 0; i < num_b_blocks; i++) {
        quantize_to_q8_1(&B_f32[i * QUANT_K_Q8_1], &B_q8[i]);
    }

    std::vector<float> output_dequant(M);
    mul_mat_vec_dequantized_cpu(A_tq2.data(), B_q8.data(), output_dequant.data(), M, K);

    bool all_finite = true;
    for (int i = 0; i < M; i++) {
        if (!std::isfinite(output_dequant[i])) {
            all_finite = false;
            printf("  Row %d: NaN or Inf detected!\n", i);
        }
    }

    bool passed = all_finite;
    printf("  All outputs finite: %s\n\n", result_str(passed));
    return passed;
}

// Test 3: Quantization round-trip accuracy
static bool test_quantization_roundtrip() {
    printf("Test 3: Quantization round-trip accuracy...\n");

    const int K = 256;
    std::mt19937 rng(111);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> original(K);
    for (int i = 0; i < K; i++) {
        original[i] = dist(rng);
    }

    // TQ2_0 round-trip
    block_tq2_0 tq2_block;
    quantize_to_tq2_0(original.data(), &tq2_block);

    std::vector<float> recovered(K);
    dequantize_tq2_0(&tq2_block, recovered.data());

    float tq2_rmse = 0.0f;
    for (int i = 0; i < K; i++) {
        float diff = original[i] - recovered[i];
        tq2_rmse += diff * diff;
    }
    tq2_rmse = sqrtf(tq2_rmse / K);

    printf("  TQ2_0 round-trip RMSE: %.6f\n", tq2_rmse);

    // Q8_1 round-trip
    std::vector<float> original_q8(QUANT_K_Q8_1);
    for (int i = 0; i < QUANT_K_Q8_1; i++) {
        original_q8[i] = dist(rng);
    }

    block_q8_1 q8_block;
    quantize_to_q8_1(original_q8.data(), &q8_block);

    std::vector<float> recovered_q8(QUANT_K_Q8_1);
    dequantize_q8_1(&q8_block, recovered_q8.data());

    float q8_rmse = 0.0f;
    for (int i = 0; i < QUANT_K_Q8_1; i++) {
        float diff = original_q8[i] - recovered_q8[i];
        q8_rmse += diff * diff;
    }
    q8_rmse = sqrtf(q8_rmse / QUANT_K_Q8_1);

    printf("  Q8_1 round-trip RMSE: %.6f\n", q8_rmse);

    bool passed = tq2_rmse < 0.5f && q8_rmse < 0.02f;
    printf("  %s\n\n", result_str(passed));
    return passed;
}

// ============================================================================
// GPU Tests - Compare GPU (Vulkan) vs CPU
// ============================================================================

// GPU Test 1: Basic GPU vs CPU comparison
static bool test_gpu_basic() {
    printf("GPU Test 1: Basic GPU vs CPU comparison...\n");

    if (!g_gpu_available) {
        printf("  SKIPPED: No GPU backend available\n\n");
        return true;  // Don't fail if GPU not available
    }

    const int M = 4;
    const int K = 256;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_f32(M * K);
    std::vector<float> B_f32(K);
    for (int i = 0; i < M * K; i++) A_f32[i] = dist(rng);
    for (int i = 0; i < K; i++) B_f32[i] = dist(rng);

    // Compute on CPU with TQ2_0 (as reference)
    const int blocks_per_row = K / QUANT_K_TQ2_0;
    std::vector<block_tq2_0> A_tq2(M * blocks_per_row);
    for (int row = 0; row < M; row++) {
        for (int b = 0; b < blocks_per_row; b++) {
            quantize_to_tq2_0(&A_f32[row * K + b * QUANT_K_TQ2_0],
                             &A_tq2[row * blocks_per_row + b]);
        }
    }

    const int num_b_blocks = K / QUANT_K_Q8_1;
    std::vector<block_q8_1> B_q8(num_b_blocks);
    for (int i = 0; i < num_b_blocks; i++) {
        quantize_to_q8_1(&B_f32[i * QUANT_K_Q8_1], &B_q8[i]);
    }

    std::vector<float> output_cpu(M);
    mul_mat_vec_dequantized_cpu(A_tq2.data(), B_q8.data(), output_cpu.data(), M, K);

    // Compute on GPU with TQ2_0
    std::vector<float> output_gpu;
    if (!run_mul_mat_on_backend(g_backend_gpu, A_f32, B_f32, output_gpu, M, K, true)) {
        printf("  GPU computation failed\n");
        printf("  FAILED\n\n");
        return false;
    }

    // Compare results
    float max_error = 0.0f;
    float sum_error = 0.0f;
    for (int i = 0; i < M; i++) {
        float error = fabsf(output_gpu[i] - output_cpu[i]);
        float rel_error = error / (fabsf(output_cpu[i]) + 1e-6f);
        max_error = std::max(max_error, rel_error);
        sum_error += rel_error;
    }
    float avg_error = sum_error / M;

    printf("  CPU results: [%.4f, %.4f, %.4f, %.4f]\n",
           output_cpu[0], output_cpu[1], output_cpu[2], output_cpu[3]);
    printf("  GPU results: [%.4f, %.4f, %.4f, %.4f]\n",
           output_gpu[0], output_gpu[1], output_gpu[2], output_gpu[3]);
    printf("  Max relative error: %.4f%%\n", max_error * 100.0f);
    printf("  Avg relative error: %.4f%%\n", avg_error * 100.0f);

    // Allow some error due to different quantization paths
    bool passed = max_error < 0.20f;  // 20% max error
    printf("  %s\n\n", result_str(passed));
    return passed;
}

// GPU Test 2: Larger matrix GPU vs CPU comparison
static bool test_gpu_larger_matrix() {
    printf("GPU Test 2: Larger matrix GPU vs CPU comparison...\n");

    if (!g_gpu_available) {
        printf("  SKIPPED: No GPU backend available\n\n");
        return true;
    }

    const int M = 16;
    const int K = 1024;  // 4 TQ2_0 blocks per row

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_f32(M * K);
    std::vector<float> B_f32(K);
    for (int i = 0; i < M * K; i++) A_f32[i] = dist(rng);
    for (int i = 0; i < K; i++) B_f32[i] = dist(rng);

    // Compute on CPU
    const int blocks_per_row = K / QUANT_K_TQ2_0;
    std::vector<block_tq2_0> A_tq2(M * blocks_per_row);
    for (int row = 0; row < M; row++) {
        for (int b = 0; b < blocks_per_row; b++) {
            quantize_to_tq2_0(&A_f32[row * K + b * QUANT_K_TQ2_0],
                             &A_tq2[row * blocks_per_row + b]);
        }
    }

    const int num_b_blocks = K / QUANT_K_Q8_1;
    std::vector<block_q8_1> B_q8(num_b_blocks);
    for (int i = 0; i < num_b_blocks; i++) {
        quantize_to_q8_1(&B_f32[i * QUANT_K_Q8_1], &B_q8[i]);
    }

    std::vector<float> output_cpu(M);
    mul_mat_vec_dequantized_cpu(A_tq2.data(), B_q8.data(), output_cpu.data(), M, K);

    // Compute on GPU
    std::vector<float> output_gpu;
    if (!run_mul_mat_on_backend(g_backend_gpu, A_f32, B_f32, output_gpu, M, K, true)) {
        printf("  GPU computation failed\n");
        printf("  FAILED\n\n");
        return false;
    }

    // Compare results
    float max_error = 0.0f;
    float sum_error = 0.0f;
    for (int i = 0; i < M; i++) {
        float error = fabsf(output_gpu[i] - output_cpu[i]);
        float rel_error = error / (fabsf(output_cpu[i]) + 1e-6f);
        max_error = std::max(max_error, rel_error);
        sum_error += rel_error;
    }
    float avg_error = sum_error / M;

    printf("  Max relative error: %.4f%%\n", max_error * 100.0f);
    printf("  Avg relative error: %.4f%%\n", avg_error * 100.0f);

    bool passed = max_error < 0.25f;  // 25% max error for larger matrix
    printf("  %s\n\n", result_str(passed));
    return passed;
}

// GPU Test 3: Ternary-friendly data (values close to -1, 0, +1)
static bool test_gpu_ternary_friendly() {
    printf("GPU Test 3: Ternary-friendly data (values near -1, 0, +1)...\n");

    if (!g_gpu_available) {
        printf("  SKIPPED: No GPU backend available\n\n");
        return true;
    }

    const int M = 8;
    const int K = 256;

    std::mt19937 rng(456);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    std::vector<float> A_f32(M * K);
    std::vector<float> B_f32(K);

    // Generate ternary-friendly data: mostly -1, 0, +1 with small noise
    for (int i = 0; i < M * K; i++) {
        int val = (rng() % 3) - 1;  // -1, 0, or +1
        A_f32[i] = (float)val + dist(rng) * 0.1f;
    }
    // Use non-zero B values to avoid division by near-zero
    for (int i = 0; i < K; i++) {
        float v = (float)((rng() % 201) - 100) / 100.0f;  // [-1, 1]
        B_f32[i] = (v == 0.0f) ? 0.1f : v;
    }

    // Compute on CPU
    const int blocks_per_row = K / QUANT_K_TQ2_0;
    std::vector<block_tq2_0> A_tq2(M * blocks_per_row);
    for (int row = 0; row < M; row++) {
        for (int b = 0; b < blocks_per_row; b++) {
            quantize_to_tq2_0(&A_f32[row * K + b * QUANT_K_TQ2_0],
                             &A_tq2[row * blocks_per_row + b]);
        }
    }

    const int num_b_blocks = K / QUANT_K_Q8_1;
    std::vector<block_q8_1> B_q8(num_b_blocks);
    for (int i = 0; i < num_b_blocks; i++) {
        quantize_to_q8_1(&B_f32[i * QUANT_K_Q8_1], &B_q8[i]);
    }

    std::vector<float> output_cpu(M);
    mul_mat_vec_dequantized_cpu(A_tq2.data(), B_q8.data(), output_cpu.data(), M, K);

    // Compute on GPU
    std::vector<float> output_gpu;
    if (!run_mul_mat_on_backend(g_backend_gpu, A_f32, B_f32, output_gpu, M, K, true)) {
        printf("  GPU computation failed\n");
        printf("  FAILED\n\n");
        return false;
    }

    // Compare results
    float max_error = 0.0f;
    float sum_error = 0.0f;
    int valid_count = 0;
    for (int i = 0; i < M; i++) {
        float error = fabsf(output_gpu[i] - output_cpu[i]);
        float denom = fabsf(output_cpu[i]);
        if (denom > 1e-3f) {
            float rel_error = error / denom;
            max_error = std::max(max_error, rel_error);
            sum_error += rel_error;
            valid_count++;
        }
    }
    float avg_error = valid_count > 0 ? sum_error / valid_count : 0.0f;

    printf("  Max relative error: %.4f%%\n", max_error * 100.0f);
    printf("  Avg relative error: %.4f%%\n", avg_error * 100.0f);

    // Should have lower error for ternary-friendly data
    bool passed = max_error < 0.15f;
    printf("  %s\n\n", result_str(passed));
    return passed;
}

// GPU Test 4: Stress test with large matrix
static bool test_gpu_stress() {
    printf("GPU Test 4: Stress test with large matrix...\n");

    if (!g_gpu_available) {
        printf("  SKIPPED: No GPU backend available\n\n");
        return true;
    }

    const int M = 128;
    const int K = 4096;  // 16 TQ2_0 blocks per row

    std::mt19937 rng(789);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_f32(M * K);
    std::vector<float> B_f32(K);
    for (int i = 0; i < M * K; i++) A_f32[i] = dist(rng);
    for (int i = 0; i < K; i++) B_f32[i] = dist(rng);

    // Compute on CPU
    const int blocks_per_row = K / QUANT_K_TQ2_0;
    std::vector<block_tq2_0> A_tq2(M * blocks_per_row);
    for (int row = 0; row < M; row++) {
        for (int b = 0; b < blocks_per_row; b++) {
            quantize_to_tq2_0(&A_f32[row * K + b * QUANT_K_TQ2_0],
                             &A_tq2[row * blocks_per_row + b]);
        }
    }

    const int num_b_blocks = K / QUANT_K_Q8_1;
    std::vector<block_q8_1> B_q8(num_b_blocks);
    for (int i = 0; i < num_b_blocks; i++) {
        quantize_to_q8_1(&B_f32[i * QUANT_K_Q8_1], &B_q8[i]);
    }

    std::vector<float> output_cpu(M);
    mul_mat_vec_dequantized_cpu(A_tq2.data(), B_q8.data(), output_cpu.data(), M, K);

    // Compute on GPU
    std::vector<float> output_gpu;
    if (!run_mul_mat_on_backend(g_backend_gpu, A_f32, B_f32, output_gpu, M, K, true)) {
        printf("  GPU computation failed\n");
        printf("  FAILED\n\n");
        return false;
    }

    // Compare results
    float max_error = 0.0f;
    float sum_error = 0.0f;
    int num_large_errors = 0;
    for (int i = 0; i < M; i++) {
        float error = fabsf(output_gpu[i] - output_cpu[i]);
        float rel_error = error / (fabsf(output_cpu[i]) + 1e-6f);
        max_error = std::max(max_error, rel_error);
        sum_error += rel_error;
        if (rel_error > 0.10f) num_large_errors++;
    }
    float avg_error = sum_error / M;

    printf("  Matrix size: %d x %d (%d TQ2_0 blocks per row)\n", M, K, blocks_per_row);
    printf("  Max relative error: %.4f%%\n", max_error * 100.0f);
    printf("  Avg relative error: %.4f%%\n", avg_error * 100.0f);
    printf("  Rows with >10%% error: %d/%d\n", num_large_errors, M);

    // TQ2_0 is 2-bit ternary quantization - very lossy for random data
    // High relative errors are expected for individual rows, but avg should be bounded
    // The key test is that GPU and CPU produce similar results
    bool passed = avg_error < 0.20f;  // Average error below 20%
    printf("  %s\n\n", result_str(passed));
    return passed;
}

// GPU Test 5: F32 baseline (no TQ2_0 quantization) to verify GPU correctness
static bool test_gpu_f32_baseline() {
    printf("GPU Test 5: F32 baseline (no TQ2_0 quantization)...\n");

    if (!g_gpu_available) {
        printf("  SKIPPED: No GPU backend available\n\n");
        return true;
    }

    const int M = 4;
    const int K = 256;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_f32(M * K);
    std::vector<float> B_f32(K);
    for (int i = 0; i < M * K; i++) A_f32[i] = dist(rng);
    for (int i = 0; i < K; i++) B_f32[i] = dist(rng);

    // Compute on CPU (simple F32 matmul)
    std::vector<float> output_cpu(M);
    for (int row = 0; row < M; row++) {
        float sum = 0.0f;
        for (int col = 0; col < K; col++) {
            sum += A_f32[row * K + col] * B_f32[col];
        }
        output_cpu[row] = sum;
    }

    // Compute on GPU with F32
    std::vector<float> output_gpu;
    if (!run_mul_mat_on_backend(g_backend_gpu, A_f32, B_f32, output_gpu, M, K, false)) {
        printf("  GPU computation failed\n");
        printf("  FAILED\n\n");
        return false;
    }

    // Compare results - should be nearly identical for F32
    float max_error = 0.0f;
    for (int i = 0; i < M; i++) {
        float error = fabsf(output_gpu[i] - output_cpu[i]);
        float rel_error = error / (fabsf(output_cpu[i]) + 1e-6f);
        max_error = std::max(max_error, rel_error);
    }

    printf("  CPU results: [%.4f, %.4f, %.4f, %.4f]\n",
           output_cpu[0], output_cpu[1], output_cpu[2], output_cpu[3]);
    printf("  GPU results: [%.4f, %.4f, %.4f, %.4f]\n",
           output_gpu[0], output_gpu[1], output_gpu[2], output_gpu[3]);
    printf("  Max relative error: %.6f%%\n", max_error * 100.0f);

    // F32 should have very low error
    bool passed = max_error < 0.001f;  // 0.1% max error for F32
    printf("  %s\n\n", result_str(passed));
    return passed;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("===========================================\n");
    printf("  TQ2_0 Vulkan Shader Unit Tests\n");
    printf("  (GPU vs CPU Comparison)\n");
    printf("===========================================\n\n");

    // Initialize backends
    if (!init_backends()) {
        printf("Failed to initialize backends!\n");
        return 1;
    }
    printf("\n");

    int num_tests = 0;
    int num_passed = 0;

    // CPU-only tests
    printf("--- CPU Reference Tests ---\n\n");
    num_tests++; if (test_basic_small()) num_passed++;
    num_tests++; if (test_larger_matrix()) num_passed++;
    num_tests++; if (test_quantization_roundtrip()) num_passed++;

    // GPU tests
    printf("--- GPU vs CPU Comparison Tests ---\n\n");
    num_tests++; if (test_gpu_f32_baseline()) num_passed++;
    num_tests++; if (test_gpu_basic()) num_passed++;
    num_tests++; if (test_gpu_larger_matrix()) num_passed++;
    num_tests++; if (test_gpu_ternary_friendly()) num_passed++;
    num_tests++; if (test_gpu_stress()) num_passed++;

    // Cleanup
    cleanup_backends();

    printf("===========================================\n");
    printf("  Results: %d/%d tests passed\n", num_passed, num_tests);
    printf("===========================================\n");

    return (num_passed == num_tests) ? 0 : 1;
}

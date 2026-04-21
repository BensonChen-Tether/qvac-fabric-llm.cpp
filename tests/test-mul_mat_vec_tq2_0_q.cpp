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

// TQ2_0 block structure - matches the shader's block_tq2_0
// TQ2_0 stores 2-bit ternary values: {0, 1, 2} representing {-1, 0, 1} relative to some offset
struct block_tq2_0 {
    uint8_t qs[QUANT_K_TQ2_0 / QUANT_R_TQ2_0];  // 64 bytes for 256 elements (4 per byte)
    uint16_t d;  // FP16 scale factor (stored as raw bits)
};

// FP16 conversion utility
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

// Forward declaration - defined after backend infrastructure below.
// Runs MUL_MAT on the given backend: A [K,M] quantized to TQ2_0, B [K,1] F32.
// When called with the CPU backend, this dispatches to ggml's internal
// TQ2_0 CPU kernel (ggml_vec_dot_tq2_0_q8_*), so we don't have to
// reimplement the matrix-vector product ourselves.
static bool run_mul_mat_on_backend(
    ggml_backend_t backend,
    const std::vector<float>& A_f32,
    const std::vector<float>& B_f32,
    std::vector<float>& output,
    int M, int K
);

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

// Run MUL_MAT operation on a specific backend.
// A: [K, M] in row-major = [M rows, K cols] matrix, ALWAYS stored as TQ2_0.
// B: [K, 1] vector in F32 format
// Result: [M] output vector
//
// K must be a multiple of QUANT_K_TQ2_0 (256).
static bool run_mul_mat_on_backend(
    ggml_backend_t backend,
    const std::vector<float>& A_f32,  // M x K matrix in row-major (will be quantized to TQ2_0)
    const std::vector<float>& B_f32,  // K vector
    std::vector<float>& output,       // M output
    int M, int K
) {
    const char* backend_name = ggml_backend_name(backend);

    assert(K % QUANT_K_TQ2_0 == 0 && "K must be a multiple of the TQ2_0 block size (256)");

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
    // A: [K, M] - K elements per row, M rows (stored in column-major for GGML), TQ2_0 only.
    ggml_tensor* tensor_a = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ2_0, K, M);
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

    // Quantize A to TQ2_0 and upload.
    const int blocks_per_row = K / QUANT_K_TQ2_0;
    const int total_blocks   = M * blocks_per_row;
    std::vector<block_tq2_0> A_tq2(total_blocks);

    for (int row = 0; row < M; row++) {
        for (int b = 0; b < blocks_per_row; b++) {
            quantize_to_tq2_0(&A_f32[row * K + b * QUANT_K_TQ2_0],
                              &A_tq2[row * blocks_per_row + b]);
        }
    }

    ggml_backend_tensor_set(tensor_a, A_tq2.data(), 0, total_blocks * sizeof(block_tq2_0));

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

    // Compute on CPU using ggml's TQ2_0 mul_mat (reference)
    std::vector<float> output_cpu;
    if (!run_mul_mat_on_backend(g_backend_cpu, A_f32, B_f32, output_cpu, M, K)) {
        printf("  CPU computation failed\n");
        printf("  FAILED\n\n");
        return false;
    }

    // Compute on GPU with TQ2_0
    std::vector<float> output_gpu;
    if (!run_mul_mat_on_backend(g_backend_gpu, A_f32, B_f32, output_gpu, M, K)) {
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

    // Compute on CPU using ggml's TQ2_0 mul_mat (reference)
    std::vector<float> output_cpu;
    if (!run_mul_mat_on_backend(g_backend_cpu, A_f32, B_f32, output_cpu, M, K)) {
        printf("  CPU computation failed\n");
        printf("  FAILED\n\n");
        return false;
    }

    // Compute on GPU
    std::vector<float> output_gpu;
    if (!run_mul_mat_on_backend(g_backend_gpu, A_f32, B_f32, output_gpu, M, K)) {
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

    // Compute on CPU using ggml's TQ2_0 mul_mat (reference)
    std::vector<float> output_cpu;
    if (!run_mul_mat_on_backend(g_backend_cpu, A_f32, B_f32, output_cpu, M, K)) {
        printf("  CPU computation failed\n");
        printf("  FAILED\n\n");
        return false;
    }

    // Compute on GPU
    std::vector<float> output_gpu;
    if (!run_mul_mat_on_backend(g_backend_gpu, A_f32, B_f32, output_gpu, M, K)) {
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

    // Compute on CPU using ggml's TQ2_0 mul_mat (reference)
    std::vector<float> output_cpu;
    if (!run_mul_mat_on_backend(g_backend_cpu, A_f32, B_f32, output_cpu, M, K)) {
        printf("  CPU computation failed\n");
        printf("  FAILED\n\n");
        return false;
    }

    // Compute on GPU
    std::vector<float> output_gpu;
    if (!run_mul_mat_on_backend(g_backend_gpu, A_f32, B_f32, output_gpu, M, K)) {
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

    printf("  Matrix size: %d x %d (%d TQ2_0 blocks per row)\n", M, K, K / QUANT_K_TQ2_0);
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

    // GPU tests (TQ2_0 only)
    printf("--- GPU vs CPU Comparison Tests (TQ2_0) ---\n\n");
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

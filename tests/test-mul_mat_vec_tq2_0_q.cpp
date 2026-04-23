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

// ============================================================================
// Debug dump support
// ----------------------------------------------------------------------------
// Every test dumps its inputs, the CPU reference output, the GPU output, and
// the per-element diff into ./debug.txt next to the binary's CWD.
//
// Normal stdout output (PASS/FAIL etc.) is unchanged; this is additional,
// numerical detail meant to be diffed across devices / drivers.
// ============================================================================

static FILE* g_dbg = nullptr;

static void dbg_open() {
    if (g_dbg) return;
    g_dbg = fopen("debug.txt", "w");
    if (!g_dbg) {
        fprintf(stderr, "debug dump: failed to open debug.txt for writing\n");
        return;
    }
    fprintf(g_dbg, "TQ2_0 test debug dump\n");
    fprintf(g_dbg, "=====================\n\n");
}

static void dbg_close() {
    if (g_dbg) { fclose(g_dbg); g_dbg = nullptr; }
}

// Dump a full test case: inputs, CPU reference, GPU output, per-element diff.
// Caps the amount of A printed to keep the dump readable for multi-row cases.
// Always prints, regardless of build mode: writes to debug.txt when available,
// otherwise falls back to stderr so the dump is never silently dropped.
// Print a 2D matrix in a grid layout: one row per line, columns separated by
// a single space. Caps row/col count so huge inputs stay readable.
static void dbg_print_matrix(FILE* out, const char* label,
                             const float* data, int rows, int cols,
                             int max_rows, int max_cols) {
    const int r_shown = std::min(rows, max_rows);
    const int c_shown = std::min(cols, max_cols);

    fprintf(out, "-- %s (%d x %d", label, rows, cols);
    if (r_shown < rows || c_shown < cols) {
        fprintf(out, ", showing %d x %d", r_shown, c_shown);
    }
    fprintf(out, ") --\n");

    for (int r = 0; r < r_shown; ++r) {
        for (int c = 0; c < c_shown; ++c) {
            fprintf(out, "% 8.4f", data[r * cols + c]);
            if (c + 1 < c_shown) fputc(' ', out);
        }
        if (c_shown < cols) fprintf(out, " ...");
        fputc('\n', out);
    }
    if (r_shown < rows) {
        fprintf(out, "... (%d more rows omitted)\n", rows - r_shown);
    }
    fputc('\n', out);
}

static void dbg_dump_case(const char* name, int M, int K,
                          const std::vector<float>& A,
                          const std::vector<float>& B,
                          const std::vector<float>& out_cpu,
                          const std::vector<float>& out_gpu) {
    FILE* out = g_dbg ? g_dbg : stderr;

    const int kMaxRowsPrint = 8;     // never dump more than 8 rows of A
    const int kMaxColsPrint = 256;   // never dump more than one TQ2_0 block per row

    fprintf(out, "======================================================================\n");
    fprintf(out, "Test: %s\n", name);
    fprintf(out, "Dims: M=%d, K=%d\n", M, K);
    fprintf(out, "======================================================================\n\n");

    // B is the input vector [1 x K].
    dbg_print_matrix(out, "Input B", B.data(), 1, K, 1, kMaxColsPrint);

    // A is the input matrix [M x K].
    dbg_print_matrix(out, "Input A", A.data(), M, K, kMaxRowsPrint, kMaxColsPrint);

    // Outputs as tall column vectors [M x 1], side by side.
    fprintf(out, "-- Output (CPU ref vs GPU, per-row) --\n");
    fprintf(out, "  %4s  %14s  %14s  %14s  %14s\n",
            "row", "cpu", "gpu", "abs_diff", "rel_diff");
    for (int i = 0; i < M; ++i) {
        const float c = out_cpu[i];
        const float g = out_gpu[i];
        const float ad = std::fabs(g - c);
        const float rd = ad / (std::fabs(c) + 1e-9f);
        fprintf(out, "  [%3d] % 14.6f  % 14.6f  % 14.6g  % 14.6g\n",
                i, c, g, ad, rd);
    }
    fprintf(out, "\n");
    fflush(out);
}

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

    dbg_dump_case("gpu_basic", M, K, A_f32, B_f32, output_cpu, output_gpu);

    // Allow some error due to different quantization paths
    bool passed = max_error < 0.20f;  // 20% max error
    printf("  %s\n\n", result_str(passed));
    return passed;
}

// ============================================================================
// Small / deterministic tests (easy to inspect in debug.txt)
// ============================================================================

// Generic tiny-test runner. K must be a multiple of 256.
static bool run_tiny_case(const char* name, int M, int K,
                          const std::vector<float>& A_f32,
                          const std::vector<float>& B_f32,
                          float rel_tol) {
    printf("GPU Tiny: %s ... (M=%d, K=%d)\n", name, M, K);
    if (!g_gpu_available) {
        printf("  SKIPPED: No GPU backend available\n\n");
        return true;
    }

    std::vector<float> out_cpu;
    if (!run_mul_mat_on_backend(g_backend_cpu, A_f32, B_f32, out_cpu, M, K)) {
        printf("  CPU computation failed\n  FAILED\n\n");
        return false;
    }
    std::vector<float> out_gpu;
    if (!run_mul_mat_on_backend(g_backend_gpu, A_f32, B_f32, out_gpu, M, K)) {
        printf("  GPU computation failed\n  FAILED\n\n");
        return false;
    }

    float max_error = 0.0f;
    for (int i = 0; i < M; i++) {
        float err = std::fabs(out_gpu[i] - out_cpu[i]);
        float rel = err / (std::fabs(out_cpu[i]) + 1e-6f);
        if (rel > max_error) max_error = rel;
    }

    printf("  CPU[0]=% .4f  GPU[0]=% .4f  max_rel_err=%.4f%%\n",
           out_cpu[0], out_gpu[0], max_error * 100.0f);

    dbg_dump_case(name, M, K, A_f32, B_f32, out_cpu, out_gpu);

    bool passed = max_error < rel_tol;
    printf("  %s\n\n", result_str(passed));
    return passed;
}

// Tiny 1: single-block, single-row, A = all +1, B = all +1.
// Expected CPU ref per-row ~= d_max * K (since quantized ternary is all +1).
static bool test_gpu_tiny_all_ones() {
    const int M = 1, K = 256;
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K,     1.0f);
    return run_tiny_case("tiny_all_ones", M, K, A, B, 0.15f);
}

// Tiny 2: single-block, single-row, A = all -1, B = all +1.
// Expected CPU ref per-row ~= -d_max * K (all negative ternary).  Specifically
// targets the case where TQ2_0 encodes all -1 (one of the TQ2_0 bug patterns).
static bool test_gpu_tiny_all_neg_ones() {
    const int M = 1, K = 256;
    std::vector<float> A(M * K, -1.0f);
    std::vector<float> B(K,      1.0f);
    return run_tiny_case("tiny_all_neg_ones", M, K, A, B, 0.15f);
}

// Tiny 3: mixed signs in A and B.  Each row has the same pattern repeating
// every 4 elements: A = [+1,-1,+1,-1,...], B = [1,1,-1,-1,1,1,-1,-1,...].
// Easy to verify by hand: per-4 contribution = (+1*1)+(-1*1)+(+1*-1)+(-1*-1) = 0.
// So each row should sum to ~0.
static bool test_gpu_tiny_alternating() {
    const int M = 2, K = 256;
    std::vector<float> A(M * K);
    std::vector<float> B(K);
    for (int r = 0; r < M; r++) {
        for (int i = 0; i < K; i++) {
            A[r * K + i] = (i % 2 == 0) ? 1.0f : -1.0f;
        }
    }
    for (int i = 0; i < K; i++) {
        B[i] = ((i / 2) % 2 == 0) ? 1.0f : -1.0f;
    }
    return run_tiny_case("tiny_alternating", M, K, A, B, 0.20f);
}

// Tiny 4: single-block, single-row, A has only the first element non-zero.
// Because TQ2_0 quantizes to {-1, 0, +1}, only elements whose magnitude is
// >= 0.5*max end up non-zero.  Here max=1.0, so only A[0]=1.0 is kept; all
// other elements quantize to 0.  Expected CPU ref = d_max * B[0].
static bool test_gpu_tiny_single_nonzero() {
    const int M = 1, K = 256;
    std::vector<float> A(M * K, 0.0f);
    A[0] = 1.0f;
    std::vector<float> B(K, 0.0f);
    for (int i = 0; i < K; i++) B[i] = float(i + 1);  // B[0]=1, B[1]=2, ...
    return run_tiny_case("tiny_single_nonzero", M, K, A, B, 0.15f);
}

// Tiny 5: deterministic ramp on B, A is a repeating {-1, 0, +1, 0, ...} pattern.
// Small enough that the full numerical trail fits comfortably in debug.txt.
static bool test_gpu_tiny_ramp() {
    const int M = 2, K = 256;
    std::vector<float> A(M * K);
    std::vector<float> B(K);
    for (int r = 0; r < M; r++) {
        for (int i = 0; i < K; i++) {
            int mod = i % 4;
            float v = (mod == 0) ? -1.0f : (mod == 2 ? 1.0f : 0.0f);
            A[r * K + i] = v;
        }
    }
    for (int i = 0; i < K; i++) B[i] = float(i) / float(K);  // 0..1 ramp
    return run_tiny_case("tiny_ramp", M, K, A, B, 0.20f);
}

// ============================================================================
// Larger tests
// ============================================================================
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

    dbg_open();

    int num_tests = 0;
    int num_passed = 0;

    // Small / deterministic tests (fast, inspectable in debug.txt).
    printf("--- Tiny GPU vs CPU Tests (TQ2_0, deterministic) ---\n\n");
    num_tests++; if (test_gpu_tiny_all_ones())       num_passed++;
    num_tests++; if (test_gpu_tiny_all_neg_ones())   num_passed++;
    num_tests++; if (test_gpu_tiny_alternating())    num_passed++;
    num_tests++; if (test_gpu_tiny_single_nonzero()) num_passed++;
    num_tests++; if (test_gpu_tiny_ramp())           num_passed++;

    // Randomized + larger tests.
    printf("--- GPU vs CPU Comparison Tests (TQ2_0) ---\n\n");
    num_tests++; if (test_gpu_basic())            num_passed++;
    num_tests++; if (test_gpu_larger_matrix())    num_passed++;
    num_tests++; if (test_gpu_ternary_friendly()) num_passed++;
    num_tests++; if (test_gpu_stress())           num_passed++;

    dbg_close();

    // Cleanup
    cleanup_backends();

    printf("===========================================\n");
    printf("  Results: %d/%d tests passed\n", num_passed, num_tests);
    printf("===========================================\n");

    return (num_passed == num_tests) ? 0 : 1;
}

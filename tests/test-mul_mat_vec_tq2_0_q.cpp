// Unit tests for the mul_mat_vec_tq2_0_q Vulkan shader
// Tests matrix-vector multiplication where:
// - Matrix A is in TQ2_0 format (2-bit ternary quantization)
// - Vector B is in F32 (quantized internally by the GPU shader to Q8_1)
//
// The reference (CPU) implementation is a hand-written, inlined matrix-vector
// product that operates directly on the TQ2_0-packed matrix. It avoids the
// ggml CPU backend entirely: no graph setup, no tensor allocation, no
// dispatch overhead. This keeps the tests fast even for large matrices.

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

static float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t(h & 0x8000u) << 16);
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
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

static void quantize_matrix_to_tq2_0(
    const std::vector<float>& A_f32,
    std::vector<block_tq2_0>& A_tq2,
    int M,
    int K
) {
    assert(K % QUANT_K_TQ2_0 == 0 && "K must be a multiple of the TQ2_0 block size (256)");

    const int blocks_per_row = K / QUANT_K_TQ2_0;
    A_tq2.resize(M * blocks_per_row);

    for (int row = 0; row < M; ++row) {
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            quantize_to_tq2_0(
                &A_f32[row * K + block_idx * QUANT_K_TQ2_0],
                &A_tq2[row * blocks_per_row + block_idx]
            );
        }
    }
}

static void cpu_ref_mul_mat_tq2_0(
    const std::vector<block_tq2_0>& A_tq2,
    const std::vector<float>& B_f32,
    std::vector<float>& output,
    int M,
    int K
) {
    assert(K % QUANT_K_TQ2_0 == 0 && "K must be a multiple of the TQ2_0 block size (256)");

    const int blocks_per_row = K / QUANT_K_TQ2_0;
    output.assign(M, 0.0f);

    for (int row = 0; row < M; ++row) {
        const block_tq2_0* row_blocks = &A_tq2[row * blocks_per_row];
        float acc = 0.0f;

        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            const block_tq2_0& block = row_blocks[block_idx];
            const float d = fp16_to_fp32(block.d);
            const float* b = B_f32.data() + block_idx * QUANT_K_TQ2_0;
            const uint8_t* qs = block.qs;

            float block_acc0 = 0.0f;
            float block_acc1 = 0.0f;

            for (int byte_idx = 0; byte_idx < 64; byte_idx += 2) {
                const uint8_t packed0 = qs[byte_idx + 0];
                const uint8_t packed1 = qs[byte_idx + 1];

                const int base0 = (byte_idx & 31) + ((byte_idx >> 5) << 7);
                const int base1 = ((byte_idx + 1) & 31) + (((byte_idx + 1) >> 5) << 7);

                block_acc0 += float(int((packed0 >> 0) & 0x3u) - 1) * b[base0 + 0];
                block_acc0 += float(int((packed0 >> 2) & 0x3u) - 1) * b[base0 + 32];
                block_acc0 += float(int((packed0 >> 4) & 0x3u) - 1) * b[base0 + 64];
                block_acc0 += float(int((packed0 >> 6) & 0x3u) - 1) * b[base0 + 96];

                block_acc1 += float(int((packed1 >> 0) & 0x3u) - 1) * b[base1 + 0];
                block_acc1 += float(int((packed1 >> 2) & 0x3u) - 1) * b[base1 + 32];
                block_acc1 += float(int((packed1 >> 4) & 0x3u) - 1) * b[base1 + 64];
                block_acc1 += float(int((packed1 >> 6) & 0x3u) - 1) * b[base1 + 96];
            }

            acc += d * (block_acc0 + block_acc1);
        }

        output[row] = acc;
    }
}

// Forward declaration - defined after backend infrastructure below.
// Runs MUL_MAT on the given backend using an already-packed TQ2_0 matrix.
static bool run_mul_mat_on_backend(
    ggml_backend_t backend,
    const std::vector<block_tq2_0>& A_tq2,
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
static bool g_gpu_available = false;

// Initialize backends
static bool init_backends() {
    printf("Initializing backends...\n");

    // Load all backends
    ggml_backend_load_all();

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
}

// Run MUL_MAT operation on a specific backend.
// A: [K, M] in row-major = [M rows, K cols] matrix, ALWAYS stored as TQ2_0.
// B: [K, 1] vector in F32 format
// Result: [M] output vector
//
// K must be a multiple of QUANT_K_TQ2_0 (256).
static bool run_mul_mat_on_backend(
    ggml_backend_t backend,
    const std::vector<block_tq2_0>& A_tq2,
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

    const int blocks_per_row = K / QUANT_K_TQ2_0;
    const int total_blocks   = M * blocks_per_row;
    assert((int)A_tq2.size() == total_blocks);

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

static bool run_test_case(const char* name, int M, int K,
                          const std::vector<float>& A_f32,
                          const std::vector<float>& B_f32,
                          float max_rel_tol,
                          float avg_rel_tol = -1.0f,
                          float ref_abs_floor = 1e-6f) {
    printf("Test: %s ... (M=%d, K=%d)\n", name, M, K);
    if (!g_gpu_available) {
        printf("  SKIPPED: No GPU backend available\n\n");
        return true;
    }

    std::vector<block_tq2_0> A_tq2;
    quantize_matrix_to_tq2_0(A_f32, A_tq2, M, K);

    std::vector<float> out_cpu;
    cpu_ref_mul_mat_tq2_0(A_tq2, B_f32, out_cpu, M, K);

    std::vector<float> out_gpu;
    if (!run_mul_mat_on_backend(g_backend_gpu, A_tq2, B_f32, out_gpu, M, K)) {
        printf("  GPU computation failed\n");
        printf("  FAILED\n\n");
        return false;
    }

    float max_error = 0.0f;
    float sum_error = 0.0f;
    int valid_count = 0;
    int num_large_errors = 0;
    for (int i = 0; i < M; i++) {
        const float error = fabsf(out_gpu[i] - out_cpu[i]);
        const float denom = fabsf(out_cpu[i]);
        if (denom > ref_abs_floor) {
            const float rel_error = error / denom;
            max_error = std::max(max_error, rel_error);
            sum_error += rel_error;
            valid_count++;
            if (rel_error > 0.10f) num_large_errors++;
        }
    }
    const float avg_error = valid_count > 0 ? sum_error / valid_count : 0.0f;

    printf("  CPU[0]=% .4f  GPU[0]=% .4f\n", out_cpu[0], out_gpu[0]);
    printf("  Max relative error: %.4f%%\n", max_error * 100.0f);
    printf("  Avg relative error: %.4f%%\n", avg_error * 100.0f);
    if (M > 1) {
        if (valid_count > 0) {
            printf("  Rows with >10%% error: %d/%d\n", num_large_errors, valid_count);
        } else {
            printf("  Rows with >10%% error: 0/0 (all reference outputs below threshold)\n");
        }
    }

    dbg_dump_case(name, M, K, A_f32, B_f32, out_cpu, out_gpu);

    bool passed = true;
    if (max_rel_tol >= 0.0f) {
        passed = passed && (max_error < max_rel_tol);
    }
    if (avg_rel_tol >= 0.0f) {
        passed = passed && (avg_error < avg_rel_tol);
    }
    printf("  %s\n\n", result_str(passed));
    return passed;
}

// GPU Test 1: Basic GPU vs CPU comparison
static bool test_gpu_basic() {
    const int M = 4;
    const int K = 256;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_f32(M * K);
    std::vector<float> B_f32(K);
    for (int i = 0; i < M * K; i++) A_f32[i] = dist(rng);
    for (int i = 0; i < K; i++) B_f32[i] = dist(rng);

    return run_test_case("gpu_basic", M, K, A_f32, B_f32, 0.20f);
}

// ============================================================================
// Small / deterministic tests (easy to inspect in debug.txt)
// ============================================================================

// Tiny 1: single-block, single-row, A = all +1, B = all +1.
// Expected CPU ref per-row ~= d_max * K (since quantized ternary is all +1).
static bool test_gpu_tiny_all_ones() {
    const int M = 1, K = 256;
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K,     1.0f);
    return run_test_case("tiny_all_ones", M, K, A, B, 0.15f);
}

// Tiny 2: single-block, single-row, A = all -1, B = all +1.
// Expected CPU ref per-row ~= -d_max * K (all negative ternary).  Specifically
// targets the case where TQ2_0 encodes all -1 (one of the TQ2_0 bug patterns).
static bool test_gpu_tiny_all_neg_ones() {
    const int M = 1, K = 256;
    std::vector<float> A(M * K, -1.0f);
    std::vector<float> B(K,      1.0f);
    return run_test_case("tiny_all_neg_ones", M, K, A, B, 0.15f);
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
    return run_test_case("tiny_alternating", M, K, A, B, 0.20f);
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
    return run_test_case("tiny_ramp", M, K, A, B, 0.20f);
}

// ============================================================================
// Larger tests
// ============================================================================
// GPU Test 2: Larger matrix GPU vs CPU comparison
static bool test_gpu_larger_matrix() {
    const int M = 16;
    const int K = 1024;  // 4 TQ2_0 blocks per row

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_f32(M * K);
    std::vector<float> B_f32(K);
    for (int i = 0; i < M * K; i++) A_f32[i] = dist(rng);
    for (int i = 0; i < K; i++) B_f32[i] = dist(rng);

    return run_test_case("gpu_larger_matrix", M, K, A_f32, B_f32, 0.25f);
}

// GPU Test 3: Ternary-friendly data (values close to -1, 0, +1)
static bool test_gpu_ternary_friendly() {
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

    return run_test_case("gpu_ternary_friendly", M, K, A_f32, B_f32, 0.15f, -1.0f, 1e-3f);
}

// GPU Test 4: Stress test with large matrix
static bool test_gpu_stress() {
    const int M = 128;
    const int K = 4096;  // 16 TQ2_0 blocks per row

    std::mt19937 rng(789);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_f32(M * K);
    std::vector<float> B_f32(K);
    for (int i = 0; i < M * K; i++) A_f32[i] = dist(rng);
    for (int i = 0; i < K; i++) B_f32[i] = dist(rng);

    return run_test_case("gpu_stress", M, K, A_f32, B_f32, -1.0f, 0.20f);
}

// ============================================================================
// Production-shape tests
// ----------------------------------------------------------------------------
// These mirror the real matmul shapes observed when running TQ2_0 models
// (see profiling/TQ2_0_perf.txt). All are matrix x vector:
//     [K x M] (tq2_0) * [K x 1] (f32) -> [M x 1] (f32)
// Random values in [-1, 1] on both sides. Max-relative-error is disabled
// (too easy to get spurious spikes on rows whose expected magnitude is
// near zero for large M); instead we check average relative error.
// ============================================================================

// Shape: [1024x1024] * [1024x1] -> [1024x1]
static bool test_gpu_prod_1024x1024() {
    const int M = 1024;
    const int K = 1024;

    std::mt19937 rng(1111);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_f32(M * K);
    std::vector<float> B_f32(K);
    for (int i = 0; i < M * K; i++) A_f32[i] = dist(rng);
    for (int i = 0; i < K; i++)     B_f32[i] = dist(rng);

    return run_test_case("prod_1024x1024", M, K, A_f32, B_f32, -1.0f, 0.20f);
}

// Shape: [1024x2048] * [1024x1] -> [2048x1]
static bool test_gpu_prod_1024x2048() {
    const int M = 2048;
    const int K = 1024;

    std::mt19937 rng(2222);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_f32(M * K);
    std::vector<float> B_f32(K);
    for (int i = 0; i < M * K; i++) A_f32[i] = dist(rng);
    for (int i = 0; i < K; i++)     B_f32[i] = dist(rng);

    return run_test_case("prod_1024x2048", M, K, A_f32, B_f32, -1.0f, 0.20f);
}

// Shape: [1024x3072] * [1024x1] -> [3072x1]
static bool test_gpu_prod_1024x3072() {
    const int M = 3072;
    const int K = 1024;

    std::mt19937 rng(3333);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_f32(M * K);
    std::vector<float> B_f32(K);
    for (int i = 0; i < M * K; i++) A_f32[i] = dist(rng);
    for (int i = 0; i < K; i++)     B_f32[i] = dist(rng);

    return run_test_case("prod_1024x3072", M, K, A_f32, B_f32, -1.0f, 0.20f);
}

// Shape: [2048x1024] * [2048x1] -> [1024x1]
static bool test_gpu_prod_2048x1024() {
    const int M = 1024;
    const int K = 2048;

    std::mt19937 rng(4444);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_f32(M * K);
    std::vector<float> B_f32(K);
    for (int i = 0; i < M * K; i++) A_f32[i] = dist(rng);
    for (int i = 0; i < K; i++)     B_f32[i] = dist(rng);

    return run_test_case("prod_2048x1024", M, K, A_f32, B_f32, -1.0f, 0.20f);
}

// Shape: [3072x1024] * [3072x1] -> [1024x1]
static bool test_gpu_prod_3072x1024() {
    const int M = 1024;
    const int K = 3072;

    std::mt19937 rng(5555);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_f32(M * K);
    std::vector<float> B_f32(K);
    for (int i = 0; i < M * K; i++) A_f32[i] = dist(rng);
    for (int i = 0; i < K; i++)     B_f32[i] = dist(rng);

    return run_test_case("prod_3072x1024", M, K, A_f32, B_f32, -1.0f, 0.20f);
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
    num_tests++; if (test_gpu_tiny_ramp())           num_passed++;

    // Randomized + larger tests.
    printf("--- GPU vs CPU Comparison Tests (TQ2_0) ---\n\n");
    num_tests++; if (test_gpu_basic())            num_passed++;
    num_tests++; if (test_gpu_larger_matrix())    num_passed++;
    num_tests++; if (test_gpu_ternary_friendly()) num_passed++;
    num_tests++; if (test_gpu_stress())           num_passed++;

    // Production-shape tests (from profiling/TQ2_0_perf.txt).
    printf("--- Production-shape Tests (TQ2_0) ---\n\n");
    num_tests++; if (test_gpu_prod_1024x1024()) num_passed++;
    num_tests++; if (test_gpu_prod_1024x2048()) num_passed++;
    num_tests++; if (test_gpu_prod_1024x3072()) num_passed++;
    num_tests++; if (test_gpu_prod_2048x1024()) num_passed++;
    num_tests++; if (test_gpu_prod_3072x1024()) num_passed++;

    dbg_close();

    // Cleanup
    cleanup_backends();

    printf("===========================================\n");
    printf("  Results: %d/%d tests passed\n", num_passed, num_tests);
    printf("===========================================\n");

    return (num_passed == num_tests) ? 0 : 1;
}

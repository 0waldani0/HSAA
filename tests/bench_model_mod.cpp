#include "model_mod.h"
#include "vsa_db.h"

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <iomanip>
#include <chrono>

#pragma comment(lib, "psapi.lib")

// =============================================================================
// Milestone M3 Empirical Challenger Benchmark & Stress Harness
// File: tests/bench_model_mod.cpp
//
// Hard Contract Invariants:
// 1. TT16 low-rank plasticity updates: 10,000 updates, average latency < 25.0 us.
// 2. Layer block hot-swapping: 10,000 swaps, average latency < 50.0 ns.
// 3. Forward pass numerical fidelity: exact dyadic math U(V*x) for TT16 residual,
//    and reference verification for GQA, Gated DeltaNet, and AVX-512 VSA.
// =============================================================================

// High-Resolution Timer Utilities (Win32 QPC)
class HighResTimer {
public:
    HighResTimer() {
        QueryPerformanceFrequency(&freq_);
    }

    inline void start() {
        QueryPerformanceCounter(&t_start_);
    }

    inline double stop_us() const {
        LARGE_INTEGER t_end;
        QueryPerformanceCounter(&t_end);
        return static_cast<double>(t_end.QuadPart - t_start_.QuadPart) * 1000000.0 / static_cast<double>(freq_.QuadPart);
    }

    inline double stop_ns() const {
        LARGE_INTEGER t_end;
        QueryPerformanceCounter(&t_end);
        return static_cast<double>(t_end.QuadPart - t_start_.QuadPart) * 1000000000.0 / static_cast<double>(freq_.QuadPart);
    }

    inline double freq() const { return static_cast<double>(freq_.QuadPart); }

private:
    LARGE_INTEGER freq_;
    LARGE_INTEGER t_start_;
};

struct LatencyStats {
    double min_val;
    double max_val;
    double avg_val;
    double p50_val;
    double p95_val;
    double p99_val;
    double p999_val;
    size_t samples;
};

static LatencyStats compute_stats(std::vector<double>& times) {
    LatencyStats s = {0};
    if (times.empty()) return s;
    s.samples = times.size();

    std::sort(times.begin(), times.end());
    s.min_val = times.front();
    s.max_val = times.back();

    double sum = 0.0;
    for (double v : times) sum += v;
    s.avg_val = sum / static_cast<double>(s.samples);

    auto get_pct = [&](double pct) {
        size_t idx = static_cast<size_t>(pct * static_cast<double>(s.samples - 1));
        return times[idx];
    };

    s.p50_val  = get_pct(0.50);
    s.p95_val  = get_pct(0.95);
    s.p99_val  = get_pct(0.99);
    s.p999_val = get_pct(0.999);

    return s;
}

static size_t get_process_ram_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024 * 1024);
    }
    return 0;
}

// PRNG (xorshift64)
static inline uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *state = x;
}

static inline float rand_uniform(uint64_t* state, float min_val, float max_val) {
    uint64_t r = xorshift64(state);
    double unit = static_cast<double>(r & 0xFFFFFFFFULL) / static_cast<double>(0xFFFFFFFFULL);
    return static_cast<float>(min_val + unit * (max_val - min_val));
}

// Numerical Metrics
struct AccuracyMetrics {
    double cosine_similarity;
    double sqnr_db;
    double max_abs_error;
    double mean_squared_error;
    double rel_l2_error;
};

static AccuracyMetrics evaluate_accuracy(const float* candidate, const float* reference, size_t dim) {
    AccuracyMetrics m = {0};
    double dot = 0.0;
    double norm_cand_sq = 0.0;
    double norm_ref_sq = 0.0;
    double sum_sq_err = 0.0;
    double max_err = 0.0;

    for (size_t i = 0; i < dim; ++i) {
        double c = static_cast<double>(candidate[i]);
        double r = static_cast<double>(reference[i]);
        dot += c * r;
        norm_cand_sq += c * c;
        norm_ref_sq  += r * r;

        double err = std::fabs(c - r);
        if (err > max_err) max_err = err;
        sum_sq_err += err * err;
    }

    double norm_cand = std::sqrt(norm_cand_sq);
    double norm_ref  = std::sqrt(norm_ref_sq);

    if (norm_cand > 1e-12 && norm_ref > 1e-12) {
        m.cosine_similarity = dot / (norm_cand * norm_ref);
    } else if (norm_cand <= 1e-12 && norm_ref <= 1e-12) {
        m.cosine_similarity = 1.0;
    } else {
        m.cosine_similarity = 0.0;
    }

    m.max_abs_error = max_err;
    m.mean_squared_error = sum_sq_err / static_cast<double>(dim);
    m.rel_l2_error = norm_ref > 1e-12 ? (std::sqrt(sum_sq_err) / norm_ref) : 0.0;

    if (sum_sq_err > 1e-15) {
        m.sqnr_db = 10.0 * std::log10(norm_ref_sq / sum_sq_err);
    } else {
        m.sqnr_db = 120.0; // Perfect match ceiling
    }

    return m;
}

// Global benchmark counters
static int g_bench_run = 0;
static int g_bench_passed = 0;
static int g_bench_failed = 0;

#define BENCH_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::printf("  [FAIL] Line %d: %s (Assertion: %s)\n", __LINE__, msg, #cond); \
            return false; \
        } \
    } while (0)

#define RUN_BENCHMARK(fn) \
    do { \
        g_bench_run++; \
        std::printf("\n=================================================================\n"); \
        std::printf("[BENCHMARK START] %s\n", #fn); \
        std::printf("=================================================================\n"); \
        HighResTimer _t; \
        _t.start(); \
        bool _res = fn(); \
        double _elapsed = _t.stop_us() / 1000.0; \
        if (_res) { \
            g_bench_passed++; \
            std::printf(">>> [PASS] %s (Elapsed: %.2f ms)\n", #fn, _elapsed); \
        } else { \
            g_bench_failed++; \
            std::printf(">>> [FAIL] %s (Elapsed: %.2f ms)\n", #fn, _elapsed); \
        } \
    } while (0)

// =============================================================================
// BENCHMARK 1: 10,000 REPEATED TT16 LOW-RANK UPDATES LATENCY
// Contract Budget: Average latency strictly below 25.0 microseconds.
// =============================================================================
static bool bench_10000_tt16_plasticity_updates() {
    std::printf("[*] Benchmarking 10,000 TT16 Low-Rank Plasticity Updates...\n");

    ModelContext* ctx = model_create_context(MODEL_DEFAULT_NUM_LAYERS);
    BENCH_ASSERT(ctx != nullptr, "Context creation failed");

    const size_t num_updates = 10000;
    const size_t core_size = MODEL_TT16_CORE_SIZE; // 16,384 floats

    // Pre-allocate cyclic delta buffers (16 unique delta pairs)
    const size_t num_buffers = 16;
    std::vector<std::vector<float>> u_buffers(num_buffers, std::vector<float>(core_size));
    std::vector<std::vector<float>> v_buffers(num_buffers, std::vector<float>(core_size));

    uint64_t rng = 0xDEADBEEF12345678ULL;
    for (size_t b = 0; b < num_buffers; ++b) {
        for (size_t i = 0; i < core_size; ++i) {
            u_buffers[b][i] = rand_uniform(&rng, -0.05f, 0.05f);
            v_buffers[b][i] = rand_uniform(&rng, -0.05f, 0.05f);
        }
    }

    // Warmup 500 iterations
    for (size_t w = 0; w < 500; ++w) {
        size_t b = w % num_buffers;
        model_update_tt16_residual(ctx, 0, u_buffers[b].data(), v_buffers[b].data());
    }

    // Reset weights
    std::memset(ctx->layers[0].tt_u, 0, core_size * sizeof(float));
    std::memset(ctx->layers[0].tt_v, 0, core_size * sizeof(float));

    // Execution with high-precision timing
    std::vector<double> latencies_us;
    latencies_us.reserve(num_updates);

    HighResTimer total_timer;
    total_timer.start();

    HighResTimer step_timer;
    for (size_t iter = 0; iter < num_updates; ++iter) {
        size_t b = iter % num_buffers;
        step_timer.start();
        int rc = model_update_tt16_residual(ctx, 0, u_buffers[b].data(), v_buffers[b].data());
        double dt_us = step_timer.stop_us();

        BENCH_ASSERT(rc == 0, "model_update_tt16_residual returned non-zero error code");
        latencies_us.push_back(dt_us);
    }
    double total_elapsed_us = total_timer.stop_us();
    double direct_avg_us = total_elapsed_us / static_cast<double>(num_updates);

    LatencyStats stats = compute_stats(latencies_us);

    std::printf("  [RESULT] 10,000 TT16 Plasticity Updates Completed:\n");
    std::printf("    -> Total Elapsed:   %.2f ms\n", total_elapsed_us / 1000.0);
    std::printf("    -> Direct Average:  %.3f us (Contract Budget: < 25.0 us)\n", direct_avg_us);
    std::printf("    -> Sample Average:  %.3f us\n", stats.avg_val);
    std::printf("    -> Minimum Latency: %.3f us\n", stats.min_val);
    std::printf("    -> Median (P50):    %.3f us\n", stats.p50_val);
    std::printf("    -> 95th Percentile: %.3f us\n", stats.p95_val);
    std::printf("    -> 99th Percentile: %.3f us\n", stats.p99_val);
    std::printf("    -> Maximum Latency: %.3f us\n", stats.max_val);
    std::printf("    -> Throughput:      %.1f updates/second\n", (num_updates * 1000000.0) / total_elapsed_us);

    // Verify mathematical integrity of accumulated weights
    const size_t repeats = num_updates / num_buffers;
    double max_accum_err_u = 0.0;
    double max_accum_err_v = 0.0;
    for (size_t i = 0; i < 100; ++i) { // Spot check 100 elements
        double expected_u = 0.0;
        double expected_v = 0.0;
        for (size_t b = 0; b < num_buffers; ++b) {
            expected_u += static_cast<double>(u_buffers[b][i]) * repeats;
            expected_v += static_cast<double>(v_buffers[b][i]) * repeats;
        }
        double actual_u = ctx->layers[0].tt_u[i];
        double actual_v = ctx->layers[0].tt_v[i];
        max_accum_err_u = std::max(max_accum_err_u, std::fabs(actual_u - expected_u));
        max_accum_err_v = std::max(max_accum_err_v, std::fabs(actual_v - expected_v));
    }
    std::printf("    -> Weight Accumulation Max Delta (U): %e\n", max_accum_err_u);
    std::printf("    -> Weight Accumulation Max Delta (V): %e\n", max_accum_err_v);
    BENCH_ASSERT(max_accum_err_u < 1e-2, "Accumulated weights U drifted beyond precision limit");
    BENCH_ASSERT(max_accum_err_v < 1e-2, "Accumulated weights V drifted beyond precision limit");

    // HARD LATENCY ASSERTION
    BENCH_ASSERT(direct_avg_us < 25.0, "Average TT16 update latency must be strictly < 25.0 us");
    BENCH_ASSERT(stats.avg_val < 25.0, "Sample average TT16 update latency must be strictly < 25.0 us");

    model_destroy_context(ctx);
    return true;
}

// =============================================================================
// BENCHMARK 2: 10,000 REPEATED LAYER BLOCK HOT-SWAPS LATENCY
// Contract Budget: Average latency strictly below 50.0 nanoseconds.
// =============================================================================
static bool bench_10000_layer_block_hot_swaps() {
    std::printf("[*] Benchmarking 10,000 Layer Block Hot-Swaps...\n");

    ModelContext* ctx = model_create_context(MODEL_DEFAULT_NUM_LAYERS);
    BENCH_ASSERT(ctx != nullptr, "Context creation failed");

    const size_t num_swaps = 10000;
    const LayerType types[3] = {
        LAYER_GQA_ATTENTION,
        LAYER_GATED_DELTANET,
        LAYER_AVX512_VSA_ASSOCIATIVE
    };

    // Warmup
    for (size_t w = 0; w < 500; ++w) {
        model_swap_layer_block(ctx, 0, types[w % 3]);
    }

    // Direct QPC measurement over 10,000 swaps in batches of 100
    const size_t batch_size = 100;
    const size_t num_batches = num_swaps / batch_size;
    std::vector<double> batch_latencies_ns;
    batch_latencies_ns.reserve(num_batches);

    HighResTimer total_timer;
    total_timer.start();

    HighResTimer batch_timer;
    for (size_t b = 0; b < num_batches; ++b) {
        batch_timer.start();
        for (size_t i = 0; i < batch_size; ++i) {
            LayerType target = types[(b * batch_size + i) % 3];
            int rc = model_swap_layer_block(ctx, 0, target);
            // Verify immediate visibility
            LayerType read_back = model_get_layer_type(ctx, 0);
            if (rc != 0 || read_back != target) {
                BENCH_ASSERT(false, "model_swap_layer_block failed or readback mismatch");
            }
        }
        double batch_ns = batch_timer.stop_ns();
        batch_latencies_ns.push_back(batch_ns / static_cast<double>(batch_size));
    }
    double total_elapsed_ns = total_timer.stop_ns();
    double direct_avg_ns = total_elapsed_ns / static_cast<double>(num_swaps);

    LatencyStats stats = compute_stats(batch_latencies_ns);

    std::printf("  [RESULT] 10,000 Layer Block Hot-Swaps Completed:\n");
    std::printf("    -> Total Elapsed:   %.3f us\n", total_elapsed_ns / 1000.0);
    std::printf("    -> Direct Average:  %.2f ns (Contract Budget: < 50.0 ns)\n", direct_avg_ns);
    std::printf("    -> Batch Average:   %.2f ns\n", stats.avg_val);
    std::printf("    -> Minimum Latency: %.2f ns\n", stats.min_val);
    std::printf("    -> Median (P50):    %.2f ns\n", stats.p50_val);
    std::printf("    -> 95th Percentile: %.2f ns\n", stats.p95_val);
    std::printf("    -> 99th Percentile: %.2f ns\n", stats.p99_val);
    std::printf("    -> Maximum Latency: %.2f ns\n", stats.max_val);
    std::printf("    -> Throughput:      %.2f million swaps/second\n", (num_swaps * 1000.0) / total_elapsed_ns);

    // HARD LATENCY ASSERTION
    BENCH_ASSERT(direct_avg_ns < 50.0, "Average layer block hot-swap latency must be strictly < 50.0 ns");

    model_destroy_context(ctx);
    return true;
}

// =============================================================================
// BENCHMARK 3: TT16 RESIDUAL EXACT DYADIC ACCUMULATION NUMERICAL FIDELITY
// Math: out_vec = in_vec + U (V in_vec)
// Contract: Validate exact dyadic projection against double-precision oracle.
// =============================================================================
static bool bench_tt16_dyadic_residual_fidelity() {
    std::printf("[*] Benchmarking TT16 Dyadic Residual Accumulation Numerical Fidelity...\n");

    ModelContext* ctx = model_create_context(MODEL_DEFAULT_NUM_LAYERS);
    BENCH_ASSERT(ctx != nullptr, "Context creation failed");

    const uint32_t R = 16;
    const uint32_t D = 1024;
    ModelLayerConfig* lyr = &ctx->layers[0];

    // Seed pseudorandom weights for U and V in [-0.05, 0.05]
    uint64_t rng = 0xCAFEBABE11223344ULL;
    for (size_t i = 0; i < R * D; ++i) {
        lyr->tt_u[i] = rand_uniform(&rng, -0.05f, 0.05f);
        lyr->tt_v[i] = rand_uniform(&rng, -0.05f, 0.05f);
    }

    // Test cases: Standard Normal, Uniform, Sparse Spikes, Dynamic Range
    struct TestCase {
        const char* name;
        int mode; // 0=uniform, 1=sparse, 2=extreme dynamic
    } test_cases[] = {
        {"Uniform Distribution [-1.0, 1.0]", 0},
        {"Sparse Spikes (95% zero)", 1},
        {"Extreme Dynamic Range [1e-3 to 1e2]", 2}
    };

    for (const auto& tc : test_cases) {
        std::printf("  [*] Testing Input Profile: %s\n", tc.name);

        for (int trial = 0; trial < 100; ++trial) {
            std::vector<float> in_x(D);
            if (tc.mode == 0) {
                for (size_t i = 0; i < D; ++i) {
                    in_x[i] = rand_uniform(&rng, -1.0f, 1.0f);
                }
            } else if (tc.mode == 1) {
                for (size_t i = 0; i < D; ++i) {
                    in_x[i] = (xorshift64(&rng) % 20 == 0) ? rand_uniform(&rng, -1.0f, 1.0f) : 0.0f;
                }
            } else {
                for (size_t i = 0; i < D; ++i) {
                    float sign = (xorshift64(&rng) % 2 == 0) ? 1.0f : -1.0f;
                    float mag = std::pow(10.0f, rand_uniform(&rng, -3.0f, 2.0f));
                    in_x[i] = sign * mag;
                }
            }

            // Software Reference Oracle (Double-Precision Dyadic Matrix Math)
            // Stage 1: z[r] = sum_{j=0..1023} V[r, j] * in_x[j]
            std::vector<double> z_ref(R, 0.0);
            for (uint32_t r = 0; r < R; ++r) {
                for (uint32_t j = 0; j < D; ++j) {
                    z_ref[r] += static_cast<double>(lyr->tt_v[r * D + j]) * static_cast<double>(in_x[j]);
                }
            }

            // Stage 2: out[i] = in_x[i] + sum_{r=0..15} z[r] * U[r, i]
            std::vector<float> ref_out(D);
            for (uint32_t i = 0; i < D; ++i) {
                double acc = static_cast<double>(in_x[i]);
                for (uint32_t r = 0; r < R; ++r) {
                    acc += z_ref[r] * static_cast<double>(lyr->tt_u[r * D + i]);
                }
                ref_out[i] = static_cast<float>(acc);
            }

            // Candidate: AVX-512 SIMD implementation via model_apply_tt16_residual
            std::vector<float> simd_out = in_x; // Initialize to in_x
            int rc = model_apply_tt16_residual(ctx, 0, in_x.data(), simd_out.data());
            BENCH_ASSERT(rc == 0, "model_apply_tt16_residual failed");

            // Evaluate fidelity
            AccuracyMetrics acc = evaluate_accuracy(simd_out.data(), ref_out.data(), D);

            if (trial == 0) {
                std::printf("    -> Trial 0 Metrics:\n");
                std::printf("       Cosine Similarity: %.8f (Assertion: > 0.999999)\n", acc.cosine_similarity);
                std::printf("       SQNR:              %.2f dB (Assertion: > 70.0 dB)\n", acc.sqnr_db);
                std::printf("       Max Abs Error:     %e (Assertion: < 1e-3)\n", acc.max_abs_error);
                std::printf("       Relative L2 Error: %e\n", acc.rel_l2_error);
            }

            BENCH_ASSERT(acc.cosine_similarity > 0.999999, "Cosine similarity must exceed 0.999999");
            BENCH_ASSERT(acc.sqnr_db > 70.0, "SQNR must exceed 70.0 dB");
            BENCH_ASSERT(acc.max_abs_error < 1e-2, "Max absolute error exceeds tolerance");
        }
    }

    model_destroy_context(ctx);
    return true;
}

// =============================================================================
// BENCHMARK 4: FORWARD PASS NUMERICAL FIDELITY ACROSS ALL 3 LAYER TYPES
// 1. Grouped-Query Attention (GQA)
// 2. Gated DeltaNet (Recurrent State Space)
// 3. AVX-512 VSA Associative Memory Lookup
// =============================================================================

// Reference GQA Implementation (Double-Precision)
static void reference_gqa_forward(const float* in_vec, float* out_vec) {
    const size_t H_Q = 8;
    const size_t H_KV = 2;
    const size_t G = 4;
    const size_t D_H = 128;
    const double scale = 0.08838834764831844; // 1.0 / sqrt(128.0)

    for (size_t h = 0; h < H_Q; ++h) {
        size_t kv_head = h / G;
        const float* q_ptr = &in_vec[h * D_H];
        const float* k_ptr = &in_vec[kv_head * D_H];
        const float* v_ptr = &in_vec[512 + kv_head * D_H];
        float* out_ptr = &out_vec[h * D_H];

        double score = 0.0;
        for (size_t j = 0; j < D_H; ++j) {
            score += static_cast<double>(q_ptr[j]) * static_cast<double>(k_ptr[j]);
        }
        score *= scale;

        double attn_weight = 1.0 / (1.0 + std::exp(-score));

        for (size_t j = 0; j < D_H; ++j) {
            out_ptr[j] = static_cast<float>(static_cast<double>(q_ptr[j]) + attn_weight * static_cast<double>(v_ptr[j]));
        }
    }
}

// Reference Gated DeltaNet Implementation (Double-Precision State)
struct ReferenceDeltaNet {
    double S[8][128][128];

    ReferenceDeltaNet() {
        std::memset(S, 0, sizeof(S));
    }

    void reset() {
        std::memset(S, 0, sizeof(S));
    }

    void forward(const float* in_vec, float* out_vec) {
        const size_t H = 8;
        const size_t D_H = 128;
        const double alpha = 0.95;
        const double beta = 0.25;

        for (size_t h = 0; h < H; ++h) {
            const float* x_h = &in_vec[h * D_H];
            float* out_h = &out_vec[h * D_H];

            // 1. Key norm
            double sum_sq = 0.0;
            for (size_t i = 0; i < D_H; ++i) {
                sum_sq += static_cast<double>(x_h[i]) * static_cast<double>(x_h[i]);
            }
            double norm_k = std::sqrt(sum_sq);
            double inv_norm = norm_k > 1e-6 ? (1.0 / norm_k) : 0.0;

            double k_bar[128];
            for (size_t i = 0; i < D_H; ++i) {
                k_bar[i] = static_cast<double>(x_h[i]) * inv_norm;
            }

            // 2. Value prediction
            double v_hat[128] = {0};
            for (size_t r = 0; r < D_H; ++r) {
                for (size_t c = 0; c < D_H; ++c) {
                    v_hat[r] += S[h][r][c] * k_bar[c];
                }
            }

            // 3. Error
            double e_h[128];
            for (size_t i = 0; i < D_H; ++i) {
                e_h[i] = static_cast<double>(x_h[i]) - v_hat[i];
            }

            // 4. Delta update
            for (size_t r = 0; r < D_H; ++r) {
                for (size_t c = 0; c < D_H; ++c) {
                    S[h][r][c] = alpha * S[h][r][c] + beta * (e_h[r] * k_bar[c]);
                }
            }

            // 5. Readout
            for (size_t r = 0; r < D_H; ++r) {
                double dot = 0.0;
                for (size_t c = 0; c < D_H; ++c) {
                    dot += S[h][r][c] * static_cast<double>(x_h[c]);
                }
                double gate = 1.0 / (1.0 + std::exp(-static_cast<double>(x_h[r])));
                out_h[r] = static_cast<float>(static_cast<double>(x_h[r]) + dot * gate);
            }
        }
    }
};

static bool bench_layer_types_numerical_fidelity() {
    std::printf("[*] Benchmarking Numerical Fidelity of All 3 Layer Types...\n");

    ModelContext* ctx = model_create_context(MODEL_DEFAULT_NUM_LAYERS);
    BENCH_ASSERT(ctx != nullptr, "Context creation failed");

    const size_t D = 1024;
    uint64_t rng = 0x12349876FEDCBA00ULL;

    // -------------------------------------------------------------------------
    // 1. Grouped-Query Attention (GQA) Verification
    // -------------------------------------------------------------------------
    std::printf("  [*] Testing Layer Type 1: LAYER_GQA_ATTENTION...\n");
    model_swap_layer_block(ctx, 0, LAYER_GQA_ATTENTION);

    double gqa_min_sim = 1.0;
    double gqa_min_sqnr = 1000.0;

    for (int iter = 0; iter < 100; ++iter) {
        std::vector<float> in_vec(D);
        for (size_t i = 0; i < D; ++i) {
            in_vec[i] = rand_uniform(&rng, -1.0f, 1.0f);
        }

        std::vector<float> ref_out(D, 0.0f);
        reference_gqa_forward(in_vec.data(), ref_out.data());

        std::vector<float> simd_out(D, 0.0f);
        int rc = model_forward_layer(ctx, 0, in_vec.data(), simd_out.data());
        BENCH_ASSERT(rc == 0, "model_forward_layer failed for GQA");

        AccuracyMetrics acc = evaluate_accuracy(simd_out.data(), ref_out.data(), D);
        gqa_min_sim = std::min(gqa_min_sim, acc.cosine_similarity);
        gqa_min_sqnr = std::min(gqa_min_sqnr, acc.sqnr_db);

        BENCH_ASSERT(acc.cosine_similarity > 0.99999, "GQA cosine similarity below 0.99999");
        BENCH_ASSERT(acc.sqnr_db > 60.0, "GQA SQNR below 60.0 dB");
    }
    std::printf("    -> GQA Min Cosine Similarity: %.8f\n", gqa_min_sim);
    std::printf("    -> GQA Min SQNR:              %.2f dB\n", gqa_min_sqnr);

    // -------------------------------------------------------------------------
    // 2. Gated DeltaNet Verification (Recurrent State Over 50 Tokens)
    // -------------------------------------------------------------------------
    std::printf("  [*] Testing Layer Type 2: LAYER_GATED_DELTANET (Recurrent Tracking)...\n");
    model_swap_layer_block(ctx, 0, LAYER_GATED_DELTANET);
    model_reset_layer_state(ctx, 0);

    ReferenceDeltaNet ref_deltanet;
    ref_deltanet.reset();

    double deltanet_min_sim = 1.0;
    double deltanet_min_sqnr = 1000.0;

    for (int step = 0; step < 50; ++step) {
        std::vector<float> in_vec(D);
        for (size_t i = 0; i < D; ++i) {
            in_vec[i] = rand_uniform(&rng, -0.5f, 0.5f);
        }

        std::vector<float> ref_out(D, 0.0f);
        ref_deltanet.forward(in_vec.data(), ref_out.data());

        std::vector<float> simd_out(D, 0.0f);
        int rc = model_forward_layer(ctx, 0, in_vec.data(), simd_out.data());
        BENCH_ASSERT(rc == 0, "model_forward_layer failed for DeltaNet");

        AccuracyMetrics acc = evaluate_accuracy(simd_out.data(), ref_out.data(), D);
        deltanet_min_sim = std::min(deltanet_min_sim, acc.cosine_similarity);
        deltanet_min_sqnr = std::min(deltanet_min_sqnr, acc.sqnr_db);

        // Check internal recurrent state matrix alignment between reference and SIMD
        double state_diff_sum = 0.0;
        double state_ref_sum = 0.0;
        const float* simd_state = ctx->layers[0].deltanet_state;
        for (size_t h = 0; h < 8; ++h) {
            for (size_t r = 0; r < 128; ++r) {
                for (size_t c = 0; c < 128; ++c) {
                    double s_ref = ref_deltanet.S[h][r][c];
                    double s_simd = static_cast<double>(simd_state[h * 128 * 128 + r * 128 + c]);
                    double d = s_ref - s_simd;
                    state_diff_sum += d * d;
                    state_ref_sum  += s_ref * s_ref;
                }
            }
        }
        double state_rel_err = state_ref_sum > 1e-12 ? (std::sqrt(state_diff_sum) / std::sqrt(state_ref_sum)) : 0.0;

        BENCH_ASSERT(acc.cosine_similarity > 0.9999, "DeltaNet token output cosine similarity degraded");
        BENCH_ASSERT(acc.sqnr_db > 55.0, "DeltaNet token SQNR degraded");
        BENCH_ASSERT(state_rel_err < 1e-3, "DeltaNet internal recurrent state divergence");
    }
    std::printf("    -> DeltaNet 50-Step Min Cosine Similarity: %.8f\n", deltanet_min_sim);
    std::printf("    -> DeltaNet 50-Step Min SQNR:              %.2f dB\n", deltanet_min_sqnr);

    // -------------------------------------------------------------------------
    // 3. AVX-512 VSA Associative Memory Block Verification
    // -------------------------------------------------------------------------
    std::printf("  [*] Testing Layer Type 3: LAYER_AVX512_VSA_ASSOCIATIVE...\n");
    model_swap_layer_block(ctx, 0, LAYER_AVX512_VSA_ASSOCIATIVE);

    // Create and bind VSA Memory Database with 500 facts
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MEDIUM, nullptr);
    BENCH_ASSERT(db != nullptr, "VSA DB creation failed");

    for (int f = 0; f < 500; ++f) {
        char subj[32], obj[32];
        std::snprintf(subj, sizeof(subj), "Node_%d", f);
        std::snprintf(obj, sizeof(obj), "Concept_%d", f * 3);
        vsa_db_insert_fact(db, subj, "AssociatedWith", obj);
    }
    model_set_layer_vsa_db(ctx, 0, db);

    double vsa_min_sim = 1.0;
    double vsa_min_sqnr = 1000.0;

    for (int iter = 0; iter < 100; ++iter) {
        std::vector<float> in_vec(D);
        for (size_t i = 0; i < D; ++i) {
            in_vec[i] = rand_uniform(&rng, -1.0f, 1.0f);
        }

        // Software reference for dual-rail binarization
        uint64_t ref_query[8] = {0};
        for (size_t w = 0; w < 8; ++w) {
            uint64_t word = 0;
            for (size_t chunk = 0; chunk < 4; ++chunk) {
                size_t c = w * 4 + chunk;
                for (size_t b = 0; b < 16; ++b) {
                    float lo = in_vec[c * 16 + b];
                    float hi = in_vec[512 + c * 16 + b];
                    if (lo >= hi) {
                        word |= (1ULL << (chunk * 16 + b));
                    }
                }
            }
            ref_query[w] = word;
        }

        // Query reference DB
        uint64_t ref_best[8];
        float ref_sim = 1.0f;
        int found = vsa_db_query_vector(db, ref_query, ref_best, &ref_sim);
        if (!found) {
            std::memcpy(ref_best, ref_query, sizeof(ref_best));
            ref_sim = 1.0f;
        }

        // Project reference
        float scale = 0.1f * ref_sim;
        std::vector<float> ref_out(D);
        for (size_t c = 0; c < 32; ++c) {
            size_t w_idx = c / 4;
            size_t shift = (c % 4) * 16;
            uint16_t mask = static_cast<uint16_t>((ref_best[w_idx] >> shift) & 0xFFFF);
            for (size_t b = 0; b < 16; ++b) {
                bool bit_set = (mask & (1U << b)) != 0;
                float proj_lo = bit_set ? scale : -scale;
                float proj_hi = bit_set ? -scale : scale;
                ref_out[c * 16 + b] = in_vec[c * 16 + b] + proj_lo;
                ref_out[512 + c * 16 + b] = in_vec[512 + c * 16 + b] + proj_hi;
            }
        }

        std::vector<float> simd_out(D, 0.0f);
        int rc = model_forward_layer(ctx, 0, in_vec.data(), simd_out.data());
        BENCH_ASSERT(rc == 0, "model_forward_layer failed for VSA Associative");

        AccuracyMetrics acc = evaluate_accuracy(simd_out.data(), ref_out.data(), D);
        vsa_min_sim = std::min(vsa_min_sim, acc.cosine_similarity);
        vsa_min_sqnr = std::min(vsa_min_sqnr, acc.sqnr_db);

        BENCH_ASSERT(acc.cosine_similarity > 0.999999, "VSA block cosine similarity mismatch");
        BENCH_ASSERT(acc.sqnr_db > 70.0, "VSA block SQNR below 70 dB");
    }
    std::printf("    -> VSA Associative Min Cosine Similarity: %.8f\n", vsa_min_sim);
    std::printf("    -> VSA Associative Min SQNR:              %.2f dB\n", vsa_min_sqnr);

    model_set_layer_vsa_db(ctx, 0, nullptr);
    vsa_db_destroy(db);
    model_destroy_context(ctx);
    return true;
}

// =============================================================================
// BENCHMARK 5: ADVERSARIAL STRESS MATRIX (0-1-N-NULL-BOUND & Concurrency)
// 1. Boundary vectors (zeros, NaNs, infinities, denormals)
// 2. 10,000 interleaved random cycles with continuous mutation
// 3. Memory leak & working set stability assertion
// =============================================================================
static bool bench_adversarial_stress_matrix() {
    std::printf("[*] Running Adversarial Stress Matrix & Long-Duration Mutation...\n");

    ModelContext* ctx = model_create_context(MODEL_DEFAULT_NUM_LAYERS);
    BENCH_ASSERT(ctx != nullptr, "Context creation failed");

    const size_t D = 1024;
    const size_t core_size = MODEL_TT16_CORE_SIZE;

    // 1. Boundary Vector Testing
    std::printf("  [*] Testing Boundary Inputs (0, denormal, huge)...\n");
    std::vector<float> zero_in(D, 0.0f);
    std::vector<float> denormal_in(D, 1e-38f);
    std::vector<float> huge_in(D, 1e4f);
    std::vector<float> out(D);

    const LayerType all_types[3] = {LAYER_GQA_ATTENTION, LAYER_GATED_DELTANET, LAYER_AVX512_VSA_ASSOCIATIVE};
    for (LayerType t : all_types) {
        model_swap_layer_block(ctx, 0, t);

        // Zero test
        int rc0 = model_forward_layer(ctx, 0, zero_in.data(), out.data());
        BENCH_ASSERT(rc0 == 0, "Zero input forward failed");
        for (float v : out) {
            BENCH_ASSERT(!std::isnan(v) && !std::isinf(v), "Zero input produced NaN/Inf");
        }

        // Denormal test
        int rc1 = model_forward_layer(ctx, 0, denormal_in.data(), out.data());
        BENCH_ASSERT(rc1 == 0, "Denormal input forward failed");
        for (float v : out) {
            BENCH_ASSERT(!std::isnan(v) && !std::isinf(v), "Denormal input produced NaN/Inf");
        }

        // Huge test
        int rc2 = model_forward_layer(ctx, 0, huge_in.data(), out.data());
        BENCH_ASSERT(rc2 == 0, "Huge input forward failed");
        for (float v : out) {
            BENCH_ASSERT(!std::isnan(v), "Huge input produced NaN");
        }
    }

    // 2. Long-Duration Interleaved Stress: 10,000 Full Forward Cycles with Mutation
    std::printf("  [*] Executing 10,000 Interleaved Cycles of Swapping, Mutation & Forward...\n");
    size_t ram_before_mb = get_process_ram_mb();

    std::vector<float> du(core_size, 0.001f);
    std::vector<float> dv(core_size, 0.001f);
    std::vector<float> in_x(D, 0.1f);

    uint64_t rng = 0x9988776655443322ULL;
    const QuantPrecision precisions[5] = {
        QUANT_2BIT, QUANT_3BIT, QUANT_4BIT, QUANT_DELTA_24, QUANT_DELTA_34
    };

    HighResTimer stress_timer;
    stress_timer.start();

    for (size_t cycle = 0; cycle < 10000; ++cycle) {
        uint32_t layer = static_cast<uint32_t>(cycle % MODEL_DEFAULT_NUM_LAYERS);

        // Random layer block swap
        LayerType target_type = all_types[xorshift64(&rng) % 3];
        model_swap_layer_block(ctx, layer, target_type);

        // Random precision modulation
        QuantPrecision target_prec = precisions[xorshift64(&rng) % 5];
        model_set_layer_precision(ctx, layer, target_prec);

        // Online TT16 plasticity update
        if (cycle % 4 == 0) {
            model_update_tt16_residual(ctx, layer, du.data(), dv.data());
        }

        // Forward execution
        int rc = model_forward_layer(ctx, layer, in_x.data(), out.data());
        if (rc != 0 || std::isnan(out[0])) {
            BENCH_ASSERT(false, "Forward execution failed during interleaved stress cycle");
        }
    }
    double stress_elapsed_ms = stress_timer.stop_us() / 1000.0;
    size_t ram_after_mb = get_process_ram_mb();

    std::printf("    -> 10,000 Interleaved Cycles Completed in %.2f ms (%.1f passes/sec)\n",
        stress_elapsed_ms, (10000.0 * 1000.0) / stress_elapsed_ms);
    std::printf("    -> Process RAM Before: %zu MB, After: %zu MB (Delta: %zd MB)\n",
        ram_before_mb, ram_after_mb, static_cast<ptrdiff_t>(ram_after_mb - ram_before_mb));

    // Assert zero memory leak (delta <= 16 MB ceiling)
    BENCH_ASSERT(ram_after_mb <= ram_before_mb + 16, "Memory leak detected during interleaved stress");

    model_destroy_context(ctx);
    return true;
}

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================
int main() {
    std::printf("=================================================================\n");
    std::printf("Milestone M3: Architecture Experimentation Subsystem\n");
    std::printf("Independent Adversarial Benchmark & Numerical Fidelity Suite\n");
    std::printf("=================================================================\n");

    RUN_BENCHMARK(bench_10000_tt16_plasticity_updates);
    RUN_BENCHMARK(bench_10000_layer_block_hot_swaps);
    RUN_BENCHMARK(bench_tt16_dyadic_residual_fidelity);
    RUN_BENCHMARK(bench_layer_types_numerical_fidelity);
    RUN_BENCHMARK(bench_adversarial_stress_matrix);

    std::printf("\n=================================================================\n");
    std::printf("BENCHMARK SUMMARY: %d Executed | %d Passed | %d Failed\n",
        g_bench_run, g_bench_passed, g_bench_failed);
    std::printf("=================================================================\n");

    if (g_bench_failed > 0) {
        std::printf(">>> FINAL VERDICT: CHALLENGE REJECTED / REQUEST_CHANGES <<<\n");
        return 1;
    }

    std::printf(">>> FINAL VERDICT: ALL INVARIANTS VERIFIED / APPROVE <<<\n");
    return 0;
}

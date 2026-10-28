#include "bench_runner.h"
#include "model_mod.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <thread>
#include <chrono>
#endif
#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>

static int g_run = 0;
static int g_passed = 0;
static int g_failed = 0;

#define TEST_CHECK(name, expr) \
    do { \
        g_run++; \
        if (expr) { \
            g_passed++; \
            std::cout << "  [PASS] " << name << "\n"; \
        } else { \
            g_failed++; \
            std::cerr << "  [FAIL] " << name << " (Line " << __LINE__ << ": " #expr ")\n"; \
        } \
    } while (0)

int main() {
    std::cout << "===============================================================================\n";
    std::cout << "  BENCH RUNNER UNIT & ADVERSARIAL TEST HARNESS\n";
    std::cout << "===============================================================================\n\n";

    // 1. QPC Timer Resolution and Monotonicity
    std::cout << ">>> 1. High-Resolution QPC Timer Tests <<<\n";
    {
        double t0 = bench_get_qpc_us();
#if defined(_WIN32)
        Sleep(5);
#else
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
#endif
        double t1 = bench_get_qpc_us();
        TEST_CHECK("QPC elapsed time positive and >= 3000 us for 5ms sleep", (t1 - t0) >= 3000.0);

        bool monotonic = true;
        double prev = bench_get_qpc_us();
        for (int i = 0; i < 1000; ++i) {
            double cur = bench_get_qpc_us();
            if (cur < prev) {
                monotonic = false;
                break;
            }
            prev = cur;
        }
        TEST_CHECK("QPC monotonicity across 1,000 successive queries", monotonic);
    }

    // 2. Host RAM and GPU VRAM Queries
    std::cout << "\n>>> 2. Hardware Resource Tracking Tests <<<\n";
    {
        size_t ram_mb = bench_get_host_ram_mb();
        TEST_CHECK("Host RAM working set > 0 MB", ram_mb > 0);
        TEST_CHECK("Host RAM working set <= 32,768 MB workstation ceiling", ram_mb <= MAX_WORKSTATION_RAM_MB);

        size_t vram_mb = bench_get_gpu_vram_mb();
        TEST_CHECK("GPU VRAM query valid (0 on CPU-only, >0 with CUDA)", vram_mb <= MAX_WORKSTATION_VRAM_MB);
        TEST_CHECK("GPU VRAM query <= 4,096 MB workstation ceiling", vram_mb <= MAX_WORKSTATION_VRAM_MB);
    }

    // 3. Hardware Bounds Assertion
    std::cout << "\n>>> 3. Hardware Bounds Assertion Tests <<<\n";
    {
        TEST_CHECK("Hardware assertion with 0 RAM limit returns -1", bench_assert_hardware_bounds(0, 4096) == -1);
        TEST_CHECK("Hardware assertion with 0 VRAM limit returns -1", bench_assert_hardware_bounds(32768, 0) == -1);
        TEST_CHECK("Hardware assertion with (0, 0) returns -1", bench_assert_hardware_bounds(0, 0) == -1);
        TEST_CHECK("Hardware assertion with workstation specs (32GB, 4GB) passes (0)",
                   bench_assert_hardware_bounds(MAX_WORKSTATION_RAM_MB, MAX_WORKSTATION_VRAM_MB) == 0);
        TEST_CHECK("Hardware assertion fails (-2) if max_ram is set to 1 MB", bench_assert_hardware_bounds(1, MAX_WORKSTATION_VRAM_MB) == -2);
    }

    // 4. Mathematical Accuracy: Cosine Similarity
    std::cout << "\n>>> 4. Cosine Similarity Numerical Accuracy Tests <<<\n";
    {
        float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float cos_ident = bench_compute_cosine_similarity(a, a, 4);
        TEST_CHECK("Cosine similarity of identical vectors == 1.0f", std::fabs(cos_ident - 1.0f) < 1e-5f);

        float b[2] = {1.0f, 0.0f};
        float c[2] = {0.0f, 1.0f};
        float cos_ortho = bench_compute_cosine_similarity(b, c, 2);
        TEST_CHECK("Cosine similarity of orthogonal vectors == 0.0f", std::fabs(cos_ortho) < 1e-5f);

        float d[3] = {-1.0f, -2.0f, -3.0f};
        float cos_anti = bench_compute_cosine_similarity(a, d, 3);
        TEST_CHECK("Cosine similarity of anti-parallel vectors == -1.0f", std::fabs(cos_anti - (-1.0f)) < 1e-5f);

        float zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float cos_zeros = bench_compute_cosine_similarity(zeros, a, 4);
        TEST_CHECK("Cosine similarity with all-zero vector == 0.0f (no NaN)", cos_zeros == 0.0f);

        float suba[2] = {1e-20f, 2e-20f};
        float subb[2] = {1e-20f, 2e-20f};
        float cos_sub = bench_compute_cosine_similarity(suba, subb, 2);
        TEST_CHECK("Cosine similarity of subnormal floats == 1.0f", std::fabs(cos_sub - 1.0f) < 1e-4f);

        float clamp_vec[1] = {1.00001f};
        float cos_clamp = bench_compute_cosine_similarity(clamp_vec, clamp_vec, 1);
        TEST_CHECK("Cosine similarity strictly clamped to <= 1.0f", cos_clamp <= 1.0f && cos_clamp >= -1.0f);

        TEST_CHECK("Null pointer rejected in cosine similarity (0.0f)", bench_compute_cosine_similarity(nullptr, a, 4) == 0.0f);
        TEST_CHECK("Zero length rejected in cosine similarity (0.0f)", bench_compute_cosine_similarity(a, a, 0) == 0.0f);
    }

    // 5. Mathematical Accuracy: SQNR dB
    std::cout << "\n>>> 5. SQNR dB Numerical Accuracy Tests <<<\n";
    {
        float sig[4] = {10.0f, 10.0f, 10.0f, 10.0f};
        float noisy[4] = {9.9f, 10.1f, 9.9f, 10.1f};
        float sqnr = bench_compute_sqnr_db(sig, noisy, 4);
        TEST_CHECK("SQNR of ~40 dB error signal calculates accurately (~40 dB)", std::fabs(sqnr - 40.0f) < 0.5f);

        float sqnr_ident = bench_compute_sqnr_db(sig, sig, 4);
        TEST_CHECK("SQNR of identical signal returns reconstruction ceiling (>= 100 dB)", sqnr_ident >= 100.0f);

        TEST_CHECK("Null pointer rejected in SQNR (0.0f)", bench_compute_sqnr_db(nullptr, sig, 4) == 0.0f);
        TEST_CHECK("Zero length rejected in SQNR (0.0f)", bench_compute_sqnr_db(sig, sig, 0) == 0.0f);
    }

    // 6. Step Profiler Execution
    std::cout << "\n>>> 6. Step Profiler Execution Tests <<<\n";
    {
        TEST_CHECK("bench_run_step_profile with null out_metrics returns -1", bench_run_step_profile(nullptr, nullptr) == -1);

        BenchmarkMetrics m;
        int res_null = bench_run_step_profile(nullptr, &m);
        TEST_CHECK("bench_run_step_profile with null ctx succeeds (0)", res_null == 0);
        TEST_CHECK("m.latency_us > 0.0", m.latency_us > 0.0);
        TEST_CHECK("m.tokens_per_sec > 0.0", m.tokens_per_sec > 0.0);
        TEST_CHECK("m.host_ram_working_set_mb > 0", m.host_ram_working_set_mb > 0);
        TEST_CHECK("m.gpu_vram_used_mb <= MAX_WORKSTATION_VRAM_MB", m.gpu_vram_used_mb <= MAX_WORKSTATION_VRAM_MB);
        // The profiled layer is FP32 and has no quantized reference. The old
        // assertions pinned numbers that bench_run_step_profile produced by
        // adding deterministic noise to its own output; the honest contract is
        // N/A, not cosine ~= 1 / SQNR ~= 120 dB.
        TEST_CHECK("m.accuracy_available == false", !m.accuracy_available);
        TEST_CHECK("m.cosine_similarity is NaN (N/A)", std::isnan(m.cosine_similarity));
        TEST_CHECK("m.sqnr_db is NaN (N/A)", std::isnan(m.sqnr_db));

        ModelContext* ctx = model_create_context(4);
        TEST_CHECK("ModelContext created (4 layers)", ctx != nullptr);
        if (ctx) {
            BenchmarkMetrics m_ctx;
            int res_ctx = bench_run_step_profile(ctx, &m_ctx);
            TEST_CHECK("bench_run_step_profile with ModelContext succeeds (0)", res_ctx == 0);
            TEST_CHECK("m_ctx.latency_us > 0.0", m_ctx.latency_us > 0.0);
            TEST_CHECK("m_ctx.tokens_per_sec > 0.0", m_ctx.tokens_per_sec > 0.0);
            model_destroy_context(ctx);
        }
    }

    // 7. Adversarial Matrix and Mutation Flip
    std::cout << "\n>>> 7. Adversarial Stress Matrix & Mutation Flip Tests <<<\n";
    {
        int adv_res = bench_run_adversarial_matrix();
        TEST_CHECK("bench_run_adversarial_matrix returns 0 (All 0-1-N-NULL tests passed)", adv_res == 0);

        // Explicit MUTATION FLIP check
        float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float x_mut[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float cos_orig = bench_compute_cosine_similarity(x, x_mut, 4);
        TEST_CHECK("Base similarity == 1.0f", std::fabs(cos_orig - 1.0f) < 1e-5f);

        x_mut[0] = -x_mut[0]; // Sign flip mutation
        float cos_mut = bench_compute_cosine_similarity(x, x_mut, 4);
        TEST_CHECK("Mutation flip condition detected (cos_mut < 0.95f)", cos_mut < 0.95f);
        TEST_CHECK("Falsification check: cos_mut is NOT equal to 1.0f", std::fabs(cos_mut - 1.0f) > 0.05f);
    }

    std::cout << "\n===============================================================================\n";
    std::cout << "TEST SUMMARY: " << g_run << " Ran, " << g_passed << " Passed, " << g_failed << " Failed\n";
    std::cout << "===============================================================================\n\n";

    return (g_failed == 0) ? 0 : 1;
}

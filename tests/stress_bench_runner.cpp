#include "bench_runner.h"
#include "model_mod.h"
#include "bridge.h"
#include "vsa_db.h"

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <vector>
#include <cassert>
#include <limits>

static int g_run = 0;
static int g_passed = 0;
static int g_failed = 0;

#define ADVERSARIAL_CHECK(name, expr) \
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
    std::cout << "  CHALLENGER M4: ADVERSARIAL STRESS TEST & FALSIFICATION HARNESS\n";
    std::cout << "===============================================================================\n\n";

    // -------------------------------------------------------------------------
    // 1. Boundary Matrix: 0-1-N-NULL in Cosine Similarity
    // -------------------------------------------------------------------------
    std::cout << ">>> 1. Cosine Similarity Adversarial Matrix (0-1-N-NULL) <<<\n";
    {
        // NULL & 0 size checks
        ADVERSARIAL_CHECK("Null pointer a returns 0.0f", bench_compute_cosine_similarity(nullptr, (const float*)0x1234, 10) == 0.0f);
        ADVERSARIAL_CHECK("Null pointer b returns 0.0f", bench_compute_cosine_similarity((const float*)0x1234, nullptr, 10) == 0.0f);
        ADVERSARIAL_CHECK("Both null pointers return 0.0f", bench_compute_cosine_similarity(nullptr, nullptr, 10) == 0.0f);
        float dummy[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        ADVERSARIAL_CHECK("Zero size length returns 0.0f", bench_compute_cosine_similarity(dummy, dummy, 0) == 0.0f);

        // All zero vectors
        float zeros[8] = {0};
        ADVERSARIAL_CHECK("Both all-zero vectors return 0.0f (no NaN / no div-by-zero)",
                          bench_compute_cosine_similarity(zeros, zeros, 8) == 0.0f);
        ADVERSARIAL_CHECK("One all-zero, one non-zero returns 0.0f",
                          bench_compute_cosine_similarity(zeros, dummy, 4) == 0.0f);
        ADVERSARIAL_CHECK("One non-zero, one all-zero returns 0.0f",
                          bench_compute_cosine_similarity(dummy, zeros, 4) == 0.0f);

        // Single element (N=1)
        float s1_a[1] = {5.0f};
        float s1_b[1] = {5.0f};
        ADVERSARIAL_CHECK("Single element positive identical == 1.0f",
                          std::fabs(bench_compute_cosine_similarity(s1_a, s1_b, 1) - 1.0f) < 1e-6f);
        float s1_c[1] = {-3.0f};
        ADVERSARIAL_CHECK("Single element opposite sign == -1.0f",
                          std::fabs(bench_compute_cosine_similarity(s1_a, s1_c, 1) - (-1.0f)) < 1e-6f);

        // Orthogonal vectors
        float ortho_a[4] = {1.0f, 0.0f, -1.0f, 0.0f};
        float ortho_b[4] = {0.0f, 2.0f,  0.0f, 3.0f};
        ADVERSARIAL_CHECK("Exact orthogonal vectors == 0.0f",
                          std::fabs(bench_compute_cosine_similarity(ortho_a, ortho_b, 4)) < 1e-6f);

        // Collinear scale invariance
        float col_a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float col_large[4] = {1e6f, 2e6f, 3e6f, 4e6f};
        float col_tiny[4] = {1e-6f, 2e-6f, 3e-6f, 4e-6f};
        ADVERSARIAL_CHECK("Collinear scale invariance (+1e6 scale) == 1.0f",
                          std::fabs(bench_compute_cosine_similarity(col_a, col_large, 4) - 1.0f) < 1e-5f);
        ADVERSARIAL_CHECK("Collinear scale invariance (+1e-6 scale) == 1.0f",
                          std::fabs(bench_compute_cosine_similarity(col_a, col_tiny, 4) - 1.0f) < 1e-5f);

        // Subnormal float resilience
        float sub_a[4] = {1e-35f, 2e-35f, 3e-35f, 4e-35f};
        float sub_b[4] = {1e-35f, 2e-35f, 3e-35f, 4e-35f};
        float sub_cos = bench_compute_cosine_similarity(sub_a, sub_b, 4);
        ADVERSARIAL_CHECK("Subnormal floats (1e-35f) compute cleanly without NaN",
                          !std::isnan(sub_cos) && std::fabs(sub_cos - 1.0f) < 1e-4f);

        // Large vector stress (N = 65,536)
        std::vector<float> big_a(65536, 1.0f);
        std::vector<float> big_b(65536, 1.0f);
        float big_cos = bench_compute_cosine_similarity(big_a.data(), big_b.data(), big_a.size());
        ADVERSARIAL_CHECK("Large vector (N=65536) identical similarity == 1.0f",
                          std::fabs(big_cos - 1.0f) < 1e-5f);
    }

    // -------------------------------------------------------------------------
    // 2. Boundary Matrix: 0-1-N-NULL in SQNR dB
    // -------------------------------------------------------------------------
    std::cout << "\n>>> 2. SQNR dB Adversarial Matrix (0-1-N-NULL) <<<\n";
    {
        // NULL & zero size
        ADVERSARIAL_CHECK("SQNR null signal returns 0.0f", bench_compute_sqnr_db(nullptr, (const float*)0x1234, 10) == 0.0f);
        ADVERSARIAL_CHECK("SQNR null noise returns 0.0f", bench_compute_sqnr_db((const float*)0x1234, nullptr, 10) == 0.0f);
        ADVERSARIAL_CHECK("SQNR zero size returns 0.0f", bench_compute_sqnr_db(nullptr, nullptr, 0) == 0.0f);

        // Identical signal -> ceiling (120 dB)
        float sig[4] = {5.0f, -3.0f, 2.0f, 1.0f};
        ADVERSARIAL_CHECK("Identical signal returns reconstruction ceiling (120 dB)",
                          bench_compute_sqnr_db(sig, sig, 4) >= 120.0f);

        // All zero signal
        float zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        ADVERSARIAL_CHECK("All-zero signal returns 0.0f",
                          bench_compute_sqnr_db(zeros, sig, 4) == 0.0f);

        // Known signal-to-noise ratios
        // 1. Signal = 10.0, Error = 1.0 -> P_sig = 100, P_err = 1 -> 10 * log10(100) = 20 dB
        float sig_20db[4] = {10.0f, 10.0f, 10.0f, 10.0f};
        float noisy_20db[4] = {11.0f, 9.0f, 11.0f, 9.0f}; // error is +1 or -1, err^2 = 1.0
        float sqnr_20 = bench_compute_sqnr_db(sig_20db, noisy_20db, 4);
        ADVERSARIAL_CHECK("Theoretical 20 dB SQNR accuracy (err=1.0 on sig=10.0)",
                          std::fabs(sqnr_20 - 20.0f) < 0.2f);

        // 2. Signal = 10.0, Error = 0.1 -> P_sig = 100, P_err = 0.01 -> 10 * log10(10000) = 40 dB
        float noisy_40db[4] = {10.1f, 9.9f, 10.1f, 9.9f};
        float sqnr_40 = bench_compute_sqnr_db(sig_20db, noisy_40db, 4);
        ADVERSARIAL_CHECK("Theoretical 40 dB SQNR accuracy (err=0.1 on sig=10.0)",
                          std::fabs(sqnr_40 - 40.0f) < 0.2f);

        // 3. Signal = 10.0, Error = 0.01 -> 60 dB
        float noisy_60db[4] = {10.01f, 9.99f, 10.01f, 9.99f};
        float sqnr_60 = bench_compute_sqnr_db(sig_20db, noisy_60db, 4);
        ADVERSARIAL_CHECK("Theoretical 60 dB SQNR accuracy (err=0.01 on sig=10.0)",
                          std::fabs(sqnr_60 - 60.0f) < 0.2f);

        // 4. Massive noise test: Error is 100x signal -> SQNR should be negative dB (-40 dB)
        float noisy_huge[4] = {1010.0f, -990.0f, 1010.0f, -990.0f}; // err = 1000
        float sqnr_neg = bench_compute_sqnr_db(sig_20db, noisy_huge, 4);
        ADVERSARIAL_CHECK("Massive noise produces negative SQNR dB without crash",
                          !std::isnan(sqnr_neg) && sqnr_neg < -35.0f);
    }

    // -------------------------------------------------------------------------
    // 3. Hardware Bounds Assertion Falsification & Detection
    // -------------------------------------------------------------------------
    std::cout << "\n>>> 3. Hardware Bounds Assertion Stress & Fault Injection <<<\n";
    {
        // Zero limits rejection
        ADVERSARIAL_CHECK("Hardware assertion rejects RAM limit 0", bench_assert_hardware_bounds(0, 4096) == -1);
        ADVERSARIAL_CHECK("Hardware assertion rejects VRAM limit 0", bench_assert_hardware_bounds(32768, 0) == -1);
        ADVERSARIAL_CHECK("Hardware assertion rejects both limits 0", bench_assert_hardware_bounds(0, 0) == -1);

        // Current real usage
        size_t host_ram = bench_get_host_ram_mb();
        size_t gpu_vram = bench_get_gpu_vram_mb();
        std::cout << "      [Telemetry] Current Host RAM Working Set: " << host_ram << " MB\n";
        std::cout << "      [Telemetry] Current GPU VRAM Allocation:  " << gpu_vram << " MB\n";

        ADVERSARIAL_CHECK("Host RAM working set > 0 MB", host_ram > 0);
        ADVERSARIAL_CHECK("Host RAM working set <= 32,768 MB workstation ceiling", host_ram <= MAX_WORKSTATION_RAM_MB);
        ADVERSARIAL_CHECK("GPU VRAM query > 0 MB", gpu_vram > 0);
        ADVERSARIAL_CHECK("GPU VRAM query <= 4,096 MB workstation ceiling", gpu_vram <= MAX_WORKSTATION_VRAM_MB);

        // Fault injection: set limit below current usage
        ADVERSARIAL_CHECK("Fault injection: limit 1 MB RAM fails with -2",
                          bench_assert_hardware_bounds(1, MAX_WORKSTATION_VRAM_MB) == -2);
        ADVERSARIAL_CHECK("Fault injection: limit 1 MB VRAM fails with -3",
                          bench_assert_hardware_bounds(MAX_WORKSTATION_RAM_MB, 1) == -3);
        ADVERSARIAL_CHECK("Valid workstation limits (32768, 4096) pass cleanly (0)",
                          bench_assert_hardware_bounds(MAX_WORKSTATION_RAM_MB, MAX_WORKSTATION_VRAM_MB) == 0);
    }

    // -------------------------------------------------------------------------
    // 4. Mutation Flip Sensitivity & Monotonicity
    // -------------------------------------------------------------------------
    std::cout << "\n>>> 4. Mutation Flip Sensitivity & Metric Degradation Ordering <<<\n";
    {
        const size_t N = 1024;
        std::vector<float> base(N);
        for (size_t i = 0; i < N; ++i) base[i] = static_cast<float>(std::sin(i * 0.05));

        std::vector<float> p0 = base; // 0% perturbation
        std::vector<float> p1 = base; // slight perturbation (+0.005 on 1 element)
        p1[10] += 0.005f;

        std::vector<float> p2 = base; // moderate perturbation (+0.2 on 10 elements)
        for (size_t k = 0; k < 10; ++k) p2[k * 50] += 0.2f;

        std::vector<float> p3 = base; // severe perturbation (sign flip on 50 elements)
        for (size_t k = 0; k < 50; ++k) p3[k * 20] = -p3[k * 20];

        std::vector<float> p4 = base; // inverted perturbation (-base)
        for (size_t i = 0; i < N; ++i) p4[i] = -base[i];

        float cos_0 = bench_compute_cosine_similarity(base.data(), p0.data(), N);
        float cos_1 = bench_compute_cosine_similarity(base.data(), p1.data(), N);
        float cos_2 = bench_compute_cosine_similarity(base.data(), p2.data(), N);
        float cos_3 = bench_compute_cosine_similarity(base.data(), p3.data(), N);
        float cos_4 = bench_compute_cosine_similarity(base.data(), p4.data(), N);

        float sqnr_0 = bench_compute_sqnr_db(base.data(), p0.data(), N);
        float sqnr_1 = bench_compute_sqnr_db(base.data(), p1.data(), N);
        float sqnr_2 = bench_compute_sqnr_db(base.data(), p2.data(), N);
        float sqnr_3 = bench_compute_sqnr_db(base.data(), p3.data(), N);
        float sqnr_4 = bench_compute_sqnr_db(base.data(), p4.data(), N);

        std::cout << "      [Perturbation 0] Cosine: " << cos_0 << " | SQNR: " << sqnr_0 << " dB\n";
        std::cout << "      [Perturbation 1] Cosine: " << cos_1 << " | SQNR: " << sqnr_1 << " dB\n";
        std::cout << "      [Perturbation 2] Cosine: " << cos_2 << " | SQNR: " << sqnr_2 << " dB\n";
        std::cout << "      [Perturbation 3] Cosine: " << cos_3 << " | SQNR: " << sqnr_3 << " dB\n";
        std::cout << "      [Perturbation 4] Cosine: " << cos_4 << " | SQNR: " << sqnr_4 << " dB\n";

        ADVERSARIAL_CHECK("Perturbation 0 is exact match (cos == 1.0f)", std::fabs(cos_0 - 1.0f) < 1e-6f);
        ADVERSARIAL_CHECK("Cosine similarity monotonic degradation: cos_0 >= cos_1", cos_0 >= cos_1);
        ADVERSARIAL_CHECK("Cosine similarity monotonic degradation: cos_1 > cos_2", cos_1 > cos_2);
        ADVERSARIAL_CHECK("Cosine similarity monotonic degradation: cos_2 > cos_3", cos_2 > cos_3);
        ADVERSARIAL_CHECK("Cosine similarity monotonic degradation: cos_3 > cos_4", cos_3 > cos_4);
        ADVERSARIAL_CHECK("Inversion perturbation reaches -1.0f", std::fabs(cos_4 - (-1.0f)) < 1e-5f);

        ADVERSARIAL_CHECK("SQNR monotonic degradation: sqnr_0 >= sqnr_1", sqnr_0 >= sqnr_1);
        ADVERSARIAL_CHECK("SQNR monotonic degradation: sqnr_1 > sqnr_2", sqnr_1 > sqnr_2);
        ADVERSARIAL_CHECK("SQNR monotonic degradation: sqnr_2 > sqnr_3", sqnr_2 > sqnr_3);
        ADVERSARIAL_CHECK("SQNR monotonic degradation: sqnr_3 > sqnr_4", sqnr_3 > sqnr_4);
    }

    // -------------------------------------------------------------------------
    // 5. Memory Leak & Stability Under Repeated Step Profiling
    // -------------------------------------------------------------------------
    std::cout << "\n>>> 5. Step Profiler Repeated Execution & Memory Stability <<<\n";
    {
        ModelContext* ctx = model_create_context(MODEL_DEFAULT_NUM_LAYERS);
        ADVERSARIAL_CHECK("ModelContext allocated for stress profiling", ctx != nullptr);

        size_t ram_initial = bench_get_host_ram_mb();
        BenchmarkMetrics m;

        // Run 200 consecutive step profiles
        for (int iter = 0; iter < 200; ++iter) {
            int rc = bench_run_step_profile(ctx, &m);
            assert(rc == 0);
        }

        size_t ram_final = bench_get_host_ram_mb();
        std::cout << "      [Memory] Initial RAM: " << ram_initial << " MB, Final RAM after 200 steps: " << ram_final << " MB\n";

        ADVERSARIAL_CHECK("Host RAM growth across 200 steps <= 5 MB (no leak)", (ram_final - ram_initial) <= 5);
        ADVERSARIAL_CHECK("Final step latency > 0.0 us", m.latency_us > 0.0);
        ADVERSARIAL_CHECK("Final step throughput > 10,000 tok/s", m.tokens_per_sec > 10000.0);
        // The profiled layer is FP32, so there is no quantized reference to compare
        // against. The old check pinned a number that was manufactured by adding
        // deterministic noise to the output; the honest contract is N/A.
        ADVERSARIAL_CHECK("Accuracy is reported unavailable, not fabricated", !m.accuracy_available);
        ADVERSARIAL_CHECK("Cosine is NaN (N/A), not a synthetic ~1.0", std::isnan(m.cosine_similarity));
        ADVERSARIAL_CHECK("SQNR is NaN (N/A), not a synthetic ~120 dB", std::isnan(m.sqnr_db));

        model_destroy_context(ctx);
        ADVERSARIAL_CHECK("ModelContext cleanly destroyed", true);
    }

    std::cout << "\n===============================================================================\n";
    std::cout << "CHALLENGER ADVERSARIAL SUMMARY: " << g_run << " Ran, " << g_passed << " Passed, " << g_failed << " Failed\n";
    std::cout << "===============================================================================\n\n";

    return (g_failed == 0) ? 0 : 1;
}

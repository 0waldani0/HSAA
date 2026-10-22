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
#include <iostream>
#include <iomanip>
#include <limits>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>

#pragma comment(lib, "psapi.lib")

// =============================================================================
// Challenger M3.1: Adversarial Stress Test Harness
// Target: include/model_mod.h, src/model_mod.cpp
// =============================================================================

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define STRESS_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::printf("  [FAIL] Line %d: %s (%s)\n", __LINE__, msg, #cond); \
            return false; \
        } \
    } while (0)

#define RUN_STRESS_TEST(fn) \
    do { \
        g_tests_run++; \
        std::printf("[STRESS-RUN] %s...\n", #fn); \
        auto t0 = std::chrono::high_resolution_clock::now(); \
        bool ok = fn(); \
        auto t1 = std::chrono::high_resolution_clock::now(); \
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); \
        if (ok) { \
            g_tests_passed++; \
            std::printf("  [PASS] %s (%.2f ms)\n", #fn, elapsed_ms); \
        } else { \
            g_tests_failed++; \
            std::printf("  [FAILED] %s (%.2f ms)\n", #fn, elapsed_ms); \
        } \
    } while (0)

static size_t get_process_private_mb() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.PrivateUsage / (1024 * 1024);
    }
    return 0;
}

static size_t get_process_private_bytes() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.PrivateUsage;
    }
    return 0;
}

static DWORD get_process_handle_count() {
    DWORD count = 0;
    GetProcessHandleCount(GetCurrentProcess(), &count);
    return count;
}

// =============================================================================
// Test 1: Extreme Delta Values (10^6, -10^6, NaN, Inf, Subnormals)
// =============================================================================
static bool test_extreme_delta_values() {
    ModelContext* ctx = model_create_context(MODEL_DEFAULT_NUM_LAYERS);
    STRESS_ASSERT(ctx != nullptr, "Context creation failed");

    const size_t core_sz = MODEL_TT16_CORE_SIZE; // 16,384 floats
    std::vector<float> u_extreme(core_sz, 1.0e6f);
    std::vector<float> v_extreme(core_sz, -1.0e6f);

    // 1. Extreme large magnitude (+10^6, -10^6)
    int rc = model_update_tt16_residual(ctx, 0, u_extreme.data(), v_extreme.data());
    STRESS_ASSERT(rc == 0, "Extreme update must succeed");
    for (size_t i = 0; i < core_sz; ++i) {
        STRESS_ASSERT(ctx->layers[0].tt_u[i] == 1.0e6f, "tt_u element must be +10^6");
        STRESS_ASSERT(ctx->layers[0].tt_v[i] == -1.0e6f, "tt_v element must be -10^6");
    }

    // Forward evaluation with extreme residual
    std::vector<float> in_x(1024, 0.001f);
    std::vector<float> out_y(1024, 0.0f);
    rc = model_apply_tt16_residual(ctx, 0, in_x.data(), out_y.data());
    STRESS_ASSERT(rc == 0, "apply_tt16_residual with extreme delta must succeed");
    for (size_t i = 0; i < 1024; ++i) {
        STRESS_ASSERT(!std::isnan(out_y[i]), "Residual output must not be NaN");
        STRESS_ASSERT(!std::isinf(out_y[i]), "Residual output must not be Inf");
    }

    // 2. Near overflow magnitude (+1.0e18)
    std::vector<float> u_huge(core_sz, 1.0e18f);
    std::vector<float> v_huge(core_sz, 1.0e18f);
    rc = model_update_tt16_residual(ctx, 0, u_huge.data(), v_huge.data());
    STRESS_ASSERT(rc == 0, "Huge update must succeed without crash");

    // 3. Subnormal / near underflow magnitude (1.0e-38)
    std::vector<float> u_subnormal(core_sz, 1.0e-38f);
    std::vector<float> v_subnormal(core_sz, 1.0e-38f);
    rc = model_update_tt16_residual(ctx, 0, u_subnormal.data(), v_subnormal.data());
    STRESS_ASSERT(rc == 0, "Subnormal update must succeed without crash");

    // 4. NaN and Inf handling
    std::vector<float> u_nan(core_sz, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> v_inf(core_sz, std::numeric_limits<float>::infinity());
    rc = model_update_tt16_residual(ctx, 0, u_nan.data(), v_inf.data());
    STRESS_ASSERT(rc == 0, "NaN/Inf update must execute without unhandled trap");

    // Verify all 3 forward operators handle NaN/Inf in state/residuals safely without SEH crash
    LayerType types[3] = {LAYER_GQA_ATTENTION, LAYER_GATED_DELTANET, LAYER_AVX512_VSA_ASSOCIATIVE};
    for (int t = 0; t < 3; ++t) {
        model_swap_layer_block(ctx, 0, types[t]);
        rc = model_forward_layer(ctx, 0, in_x.data(), out_y.data());
        STRESS_ASSERT(rc == 0, "Forward pass with NaN/Inf residual must not crash");
    }

    // 5. Self-healing / State recovery: verify reset clears recurrent state
    rc = model_reset_layer_state(ctx, 0);
    STRESS_ASSERT(rc == 0, "Layer state reset must succeed");
    for (size_t i = 0; i < MODEL_DELTANET_HEADS * MODEL_DELTANET_HEAD_DIM * MODEL_DELTANET_HEAD_DIM; ++i) {
        STRESS_ASSERT(ctx->layers[0].deltanet_state[i] == 0.0f, "DeltaNet state must be zeroed");
    }

    // Zero out TT16 buffers directly to test recovery to clean operation
    std::memset(ctx->layers[0].tt_u, 0, core_sz * sizeof(float));
    std::memset(ctx->layers[0].tt_v, 0, core_sz * sizeof(float));
    model_swap_layer_block(ctx, 0, LAYER_GQA_ATTENTION);
    std::fill(in_x.begin(), in_x.end(), 0.1f);
    rc = model_forward_layer(ctx, 0, in_x.data(), out_y.data());
    STRESS_ASSERT(rc == 0, "Post-recovery forward must succeed");
    for (size_t i = 0; i < 1024; ++i) {
        STRESS_ASSERT(!std::isnan(out_y[i]), "Post-recovery output must be clean finite");
        STRESS_ASSERT(!std::isinf(out_y[i]), "Post-recovery output must not be Inf");
    }

    model_destroy_context(ctx);
    return true;
}

// =============================================================================
// Test 2: Consecutive Cancelling Updates (+1.0 followed by -1.0 returning to 0.0)
// =============================================================================
static bool test_consecutive_cancelling_updates() {
    ModelContext* ctx = model_create_context(MODEL_DEFAULT_NUM_LAYERS);
    STRESS_ASSERT(ctx != nullptr, "Context creation failed");

    const size_t core_sz = MODEL_TT16_CORE_SIZE;
    std::vector<float> plus_one(core_sz, 1.0f);
    std::vector<float> minus_one(core_sz, -1.0f);

    // Initial state must be strictly 0.0f
    for (size_t i = 0; i < core_sz; ++i) {
        STRESS_ASSERT(ctx->layers[0].tt_u[i] == 0.0f, "Initial tt_u must be 0");
        STRESS_ASSERT(ctx->layers[0].tt_v[i] == 0.0f, "Initial tt_v must be 0");
    }

    // Step A: Single large cancelling cycle (+1.0 then -1.0)
    int rc1 = model_update_tt16_residual(ctx, 0, plus_one.data(), plus_one.data());
    int rc2 = model_update_tt16_residual(ctx, 0, minus_one.data(), minus_one.data());
    STRESS_ASSERT(rc1 == 0 && rc2 == 0, "Single cancelling update failed");

    for (size_t i = 0; i < core_sz; ++i) {
        STRESS_ASSERT(ctx->layers[0].tt_u[i] == 0.0f, "tt_u must return exactly to 0.0f");
        STRESS_ASSERT(ctx->layers[0].tt_v[i] == 0.0f, "tt_v must return exactly to 0.0f");
    }

    // Step B: High-frequency churn (10,000 consecutive cancelling cycles)
    const int CHURN_CYCLES = 10000;
    for (int c = 0; c < CHURN_CYCLES; ++c) {
        model_update_tt16_residual(ctx, 0, plus_one.data(), plus_one.data());
        model_update_tt16_residual(ctx, 0, minus_one.data(), minus_one.data());
    }

    float max_drift_u = 0.0f;
    float max_drift_v = 0.0f;
    for (size_t i = 0; i < core_sz; ++i) {
        float du = std::abs(ctx->layers[0].tt_u[i]);
        float dv = std::abs(ctx->layers[0].tt_v[i]);
        if (du > max_drift_u) max_drift_u = du;
        if (dv > max_drift_v) max_drift_v = dv;
    }
    std::printf("    -> 10,000 Cancelling Cycles Max Absolute Drift: U = %.8f, V = %.8f\n", max_drift_u, max_drift_v);
    STRESS_ASSERT(max_drift_u == 0.0f, "IEEE-754 exact cancellation invariant: U drift must be 0.0f");
    STRESS_ASSERT(max_drift_v == 0.0f, "IEEE-754 exact cancellation invariant: V drift must be 0.0f");

    // Step C: Fractional cancellation with non-powers-of-two (1,000 cycles)
    std::vector<float> plus_frac(core_sz, 0.1234567f);
    std::vector<float> minus_frac(core_sz, -0.1234567f);
    for (int c = 0; c < 1000; ++c) {
        model_update_tt16_residual(ctx, 0, plus_frac.data(), plus_frac.data());
        model_update_tt16_residual(ctx, 0, minus_frac.data(), minus_frac.data());
    }
    float frac_drift = 0.0f;
    for (size_t i = 0; i < core_sz; ++i) {
        float d = std::abs(ctx->layers[0].tt_u[i]);
        if (d > frac_drift) frac_drift = d;
    }
    std::printf("    -> 1,000 Fractional Cancelling Cycles Max Drift: %.2e (Threshold < 1e-4)\n", frac_drift);
    STRESS_ASSERT(frac_drift < 1.0e-4f, "Fractional cancellation drift must be bounded");

    // Step D: Zero-effect verification on residual forward
    std::memset(ctx->layers[0].tt_u, 0, core_sz * sizeof(float));
    std::memset(ctx->layers[0].tt_v, 0, core_sz * sizeof(float));
    std::vector<float> in_x(1024, 0.5f);
    std::vector<float> out_y(1024, 1.234f);
    std::vector<float> expected_y = out_y;

    model_apply_tt16_residual(ctx, 0, in_x.data(), out_y.data());
    for (size_t i = 0; i < 1024; ++i) {
        STRESS_ASSERT(out_y[i] == expected_y[i], "Zero residual must leave output completely unchanged");
    }

    model_destroy_context(ctx);
    return true;
}

// =============================================================================
// Test 3: Rapid Precision Cycling Across All 5 Modes (10,000 Times)
// Modes: QUANT_2BIT, QUANT_3BIT, QUANT_4BIT, QUANT_DELTA_24, QUANT_DELTA_34
// =============================================================================
static bool test_rapid_precision_cycling_10k() {
    ModelContext* ctx = model_create_context(MODEL_DEFAULT_NUM_LAYERS);
    STRESS_ASSERT(ctx != nullptr, "Context creation failed");

    const QuantPrecision modes[5] = {
        QUANT_2BIT,
        QUANT_3BIT,
        QUANT_4BIT,
        QUANT_DELTA_24,
        QUANT_DELTA_34
    };

    size_t mem_before_bytes = get_process_private_bytes();

    const int TOTAL_CYCLES = 10000;
    for (int i = 0; i < TOTAL_CYCLES; ++i) {
        QuantPrecision target_prec = modes[i % 5];
        // Cycle on multiple layers simultaneously
        uint32_t layers_to_test[3] = {0, 13, 27};
        for (uint32_t lyr : layers_to_test) {
            int rc = model_set_layer_precision(ctx, lyr, target_prec);
            STRESS_ASSERT(rc == 0, "Precision change must succeed");
            QuantPrecision read_prec = model_get_layer_precision(ctx, lyr);
            STRESS_ASSERT(read_prec == target_prec, "Readback precision must match set precision");
        }
    }

    size_t mem_after_bytes = get_process_private_bytes();
    ptrdiff_t mem_delta = static_cast<ptrdiff_t>(mem_after_bytes) - static_cast<ptrdiff_t>(mem_before_bytes);
    std::printf("    -> 10,000 Precision Cycles Memory Delta: %zd bytes (Target 0 bytes leak)\n", mem_delta);

    // Precision cycling must perform zero allocations
    STRESS_ASSERT(mem_delta <= 1024 * 1024, "Memory delta must be near-zero (no memory leak)");

    // Verify all layer invariant properties remain intact and uncorrupted
    for (uint32_t l = 0; l < ctx->num_layers; ++l) {
        ModelLayerConfig* lyr = &ctx->layers[l];
        STRESS_ASSERT(lyr->M == 1024 && lyr->N == 1024 && lyr->R == 16, "Dimensions corrupted");
        STRESS_ASSERT(lyr->tt_u != nullptr && lyr->tt_v != nullptr && lyr->deltanet_state != nullptr, "Pointers corrupted");
        STRESS_ASSERT((reinterpret_cast<uintptr_t>(lyr->tt_u) % 64) == 0, "tt_u alignment lost");
        STRESS_ASSERT((reinterpret_cast<uintptr_t>(lyr->tt_v) % 64) == 0, "tt_v alignment lost");
        STRESS_ASSERT((reinterpret_cast<uintptr_t>(lyr->deltanet_state) % 64) == 0, "deltanet alignment lost");
    }

    model_destroy_context(ctx);
    return true;
}

static int safe_fuzz_step(ModelContext* ctx, uint32_t lyr, int enum_val, float* in_ptr, float* out_ptr) {
    __try {
        model_set_layer_precision(ctx, lyr, (QuantPrecision)enum_val);
        model_get_layer_precision(ctx, lyr);
        model_swap_layer_block(ctx, lyr, (LayerType)enum_val);
        model_get_layer_type(ctx, lyr);
        model_update_tt16_residual(ctx, lyr, in_ptr, out_ptr);
        model_apply_tt16_residual(ctx, lyr, in_ptr, out_ptr);
        model_forward_layer(ctx, lyr, in_ptr, out_ptr);
        model_reset_layer_state(ctx, lyr);
        return 0;
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        return -1;
    }
}

// =============================================================================
// Test 4: Out-of-Bounds Layer Indices and NULL Pointer Fuzzing
// =============================================================================
static bool test_null_and_oob_fuzzing() {
    ModelContext* ctx = model_create_context(MODEL_DEFAULT_NUM_LAYERS);
    STRESS_ASSERT(ctx != nullptr, "Context creation failed");

    std::vector<float> dummy_vec(MODEL_TT16_CORE_SIZE, 0.0f);
    std::vector<float> out_vec(MODEL_TT16_CORE_SIZE, 0.0f);

    // 1. NULL Context Fuzzing across entire API
    STRESS_ASSERT(model_get_layer_count(nullptr) == 0, "get_layer_count null");
    STRESS_ASSERT(model_set_layer_precision(nullptr, 0, QUANT_4BIT) == -1, "set_layer_precision null");
    STRESS_ASSERT(model_get_layer_precision(nullptr, 0) == QUANT_4BIT, "get_layer_precision null default");
    STRESS_ASSERT(model_update_tt16_residual(nullptr, 0, dummy_vec.data(), dummy_vec.data()) == -1, "update_tt16 null");
    STRESS_ASSERT(model_apply_tt16_residual(nullptr, 0, dummy_vec.data(), out_vec.data()) == -1, "apply_tt16 null");
    STRESS_ASSERT(model_swap_layer_block(nullptr, 0, LAYER_GQA_ATTENTION) == -1, "swap_layer null");
    STRESS_ASSERT(model_get_layer_type(nullptr, 0) == LAYER_GQA_ATTENTION, "get_layer_type null default");
    STRESS_ASSERT(model_forward_layer(nullptr, 0, dummy_vec.data(), out_vec.data()) == -1, "forward_layer null");
    STRESS_ASSERT(model_reset_layer_state(nullptr, 0) == -1, "reset_layer null");
    STRESS_ASSERT(model_reset_all_states(nullptr) == -1, "reset_all null");
    STRESS_ASSERT(model_set_layer_vsa_db(nullptr, 0, nullptr) == -1, "set_layer_vsa null");
    STRESS_ASSERT(model_get_layer_vsa_db(nullptr, 0) == nullptr, "get_layer_vsa null");
    STRESS_ASSERT(model_set_global_vsa_db(nullptr, nullptr) == -1, "set_global_vsa null");
    STRESS_ASSERT(model_get_global_vsa_db(nullptr) == nullptr, "get_global_vsa null");
    model_destroy_context(nullptr); // Must be a safe no-op
    model_free_context(nullptr);    // Must be a safe no-op

    // 2. NULL Parameter Pointer Fuzzing with valid context
    STRESS_ASSERT(model_update_tt16_residual(ctx, 0, nullptr, dummy_vec.data()) == -1, "update_tt16 u null");
    STRESS_ASSERT(model_update_tt16_residual(ctx, 0, dummy_vec.data(), nullptr) == -1, "update_tt16 v null");
    STRESS_ASSERT(model_update_tt16_residual(ctx, 0, nullptr, nullptr) == -1, "update_tt16 both null");
    STRESS_ASSERT(model_apply_tt16_residual(ctx, 0, nullptr, out_vec.data()) == -1, "apply_tt16 in null");
    STRESS_ASSERT(model_apply_tt16_residual(ctx, 0, dummy_vec.data(), nullptr) == -1, "apply_tt16 out null");
    STRESS_ASSERT(model_forward_layer(ctx, 0, nullptr, out_vec.data()) == -1, "forward in null");
    STRESS_ASSERT(model_forward_layer(ctx, 0, dummy_vec.data(), nullptr) == -1, "forward out null");

    // 3. Out-of-bounds layer indices fuzzing
    const uint32_t oob_indices[] = {
        28, 29, 30, 100, 1024, 65535, 65536,
        0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFFFFFE
    };

    for (uint32_t oob : oob_indices) {
        STRESS_ASSERT(model_set_layer_precision(ctx, oob, QUANT_4BIT) == -2, "set_precision OOB");
        STRESS_ASSERT(model_get_layer_precision(ctx, oob) == QUANT_4BIT, "get_precision OOB default");
        STRESS_ASSERT(model_update_tt16_residual(ctx, oob, dummy_vec.data(), dummy_vec.data()) == -2, "update_tt16 OOB");
        STRESS_ASSERT(model_apply_tt16_residual(ctx, oob, dummy_vec.data(), out_vec.data()) == -2, "apply_tt16 OOB");
        STRESS_ASSERT(model_swap_layer_block(ctx, oob, LAYER_GQA_ATTENTION) == -2, "swap_layer OOB");
        STRESS_ASSERT(model_get_layer_type(ctx, oob) == LAYER_GQA_ATTENTION, "get_layer_type OOB default");
        STRESS_ASSERT(model_forward_layer(ctx, oob, dummy_vec.data(), out_vec.data()) == -2, "forward_layer OOB");
        STRESS_ASSERT(model_reset_layer_state(ctx, oob) == -2, "reset_layer OOB");
        STRESS_ASSERT(model_set_layer_vsa_db(ctx, oob, nullptr) == -2, "set_layer_vsa OOB");
        STRESS_ASSERT(model_get_layer_vsa_db(ctx, oob) == nullptr, "get_layer_vsa OOB");
    }

    // 4. Invalid enum values fuzzing
    const int invalid_precisions[] = {-999, -1, 0, 1, 5, 6, 23, 25, 33, 35, 100, 9999};
    for (int inv_p : invalid_precisions) {
        STRESS_ASSERT(model_set_layer_precision(ctx, 0, (QuantPrecision)inv_p) == -3, "invalid precision enum");
    }

    const int invalid_layer_types[] = {-10, -1, 3, 4, 5, 42, 999};
    for (int inv_t : invalid_layer_types) {
        STRESS_ASSERT(model_swap_layer_block(ctx, 0, (LayerType)inv_t) == -3, "invalid layer type enum");
    }

    // 5. High-volume randomized combinatorial fuzzing (20,000 iterations)
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist_layer(0, 1000);
    std::uniform_int_distribution<int> dist_enum(-100, 100);
    std::uniform_int_distribution<int> dist_null(0, 3);

    for (int iter = 0; iter < 20000; ++iter) {
        uint32_t lyr = dist_layer(rng);
        int enum_val = dist_enum(rng);
        int null_mode = dist_null(rng);

        float* in_ptr = (null_mode == 1 || null_mode == 3) ? nullptr : dummy_vec.data();
        float* out_ptr = (null_mode == 2 || null_mode == 3) ? nullptr : out_vec.data();

        // Exercise all APIs under randomized fuzzing inside safe SEH helper
        if (safe_fuzz_step(ctx, lyr, enum_val, in_ptr, out_ptr) != 0) {
            std::printf("  [FATAL] Access violation caught in fuzzing iteration %d\n", iter);
            model_destroy_context(ctx);
            return false;
        }
    }

    model_destroy_context(ctx);
    return true;
}

// =============================================================================
// Test 5: Rapid Context Creation and Destruction Churn (1,000 Cycles)
// =============================================================================
static bool test_rapid_context_churn_1000() {
    const int CHURN_CYCLES = 1000;
    std::vector<float> in_vec(1024, 0.05f);
    std::vector<float> out_vec(1024, 0.0f);
    std::vector<float> du(MODEL_TT16_CORE_SIZE, 0.001f);
    std::vector<float> dv(MODEL_TT16_CORE_SIZE, 0.001f);

    // Warmup memory / handle state (prime Windows LFH bucket cache to steady-state)
    for (int w = 0; w < 250; ++w) {
        ModelContext* warmup = model_create_context(MODEL_DEFAULT_NUM_LAYERS);
        model_set_layer_precision(warmup, 0, QUANT_DELTA_34);
        model_swap_layer_block(warmup, 0, (w % 2 == 0) ? LAYER_GATED_DELTANET : LAYER_GQA_ATTENTION);
        model_update_tt16_residual(warmup, 0, du.data(), dv.data());
        model_forward_layer(warmup, 0, in_vec.data(), out_vec.data());
        model_destroy_context(warmup);
    }

    size_t mem_start_mb = get_process_private_mb();
    size_t mem_start_bytes = get_process_private_bytes();
    DWORD handles_start = get_process_handle_count();

    std::vector<size_t> mem_checkpoints;
    for (int i = 0; i < CHURN_CYCLES; ++i) {
        // Create context
        ModelContext* ctx = model_create_context(MODEL_DEFAULT_NUM_LAYERS);
        if (!ctx) {
            std::printf("  [FAIL] Context allocation failed at cycle %d\n", i);
            return false;
        }

        // Perform representative workload
        model_set_layer_precision(ctx, 0, QUANT_DELTA_34);
        model_swap_layer_block(ctx, 0, (i % 2 == 0) ? LAYER_GATED_DELTANET : LAYER_GQA_ATTENTION);
        model_update_tt16_residual(ctx, 0, du.data(), dv.data());
        model_forward_layer(ctx, 0, in_vec.data(), out_vec.data());

        // Alternate destruction methods
        if (i % 2 == 0) {
            model_destroy_context(ctx);
        } else {
            model_free_context(ctx);
        }

        if ((i + 1) % 100 == 0) {
            mem_checkpoints.push_back(get_process_private_bytes());
        }
    }

    size_t mem_end_mb = get_process_private_mb();
    size_t mem_end_bytes = get_process_private_bytes();
    DWORD handles_end = get_process_handle_count();

    std::printf("    -> Memory progression per 100 cycles:\n");
    for (size_t c = 0; c < mem_checkpoints.size(); ++c) {
        std::printf("       Cycle %3zu: %zu bytes (%.2f MB)\n",
            (c + 1) * 100, mem_checkpoints[c], (double)mem_checkpoints[c] / (1024.0 * 1024.0));
    }

    ptrdiff_t mem_delta_bytes = static_cast<ptrdiff_t>(mem_end_bytes) - static_cast<ptrdiff_t>(mem_start_bytes);
    ptrdiff_t mem_delta_mb = static_cast<ptrdiff_t>(mem_end_mb) - static_cast<ptrdiff_t>(mem_start_mb);
    long handle_delta = static_cast<long>(handles_end) - static_cast<long>(handles_start);
    std::printf("       OS Handle Delta: %ld handles (Start: %u, End: %u)\n", handle_delta, handles_start, handles_end);

    // Assert handle leak invariant
    STRESS_ASSERT(handle_delta == 0, "Handle count must remain strictly invariant (0 leak)");

    // Assert memory leak invariant (must be <= 2 MB allocator headroom)
    STRESS_ASSERT(mem_delta_mb <= 2, "Memory leak detected during context lifecycle churn");

    return true;
}

// =============================================================================
// Test 6: Dynamic Layer Capacity and Zero-Layer Edge Cases
// =============================================================================
static bool test_dynamic_capacity_and_edge_cases() {
    // 1. Zero layer request defaults to MODEL_DEFAULT_NUM_LAYERS (28)
    ModelContext* ctx0 = model_create_context(0);
    STRESS_ASSERT(ctx0 != nullptr, "Context 0 layers creation");
    STRESS_ASSERT(model_get_layer_count(ctx0) == MODEL_DEFAULT_NUM_LAYERS, "0 layers must default to 28");
    model_destroy_context(ctx0);

    // 2. Single layer model (minimal edge)
    ModelContext* ctx1 = model_create_context(1);
    STRESS_ASSERT(ctx1 != nullptr, "Context 1 layer creation");
    STRESS_ASSERT(model_get_layer_count(ctx1) == 1, "Layer count must be 1");
    STRESS_ASSERT(model_set_layer_precision(ctx1, 0, QUANT_2BIT) == 0, "Layer 0 precision");
    STRESS_ASSERT(model_set_layer_precision(ctx1, 1, QUANT_2BIT) == -2, "Layer 1 OOB");
    model_destroy_context(ctx1);

    // 3. Deep 64-layer model (stress scaling)
    ModelContext* ctx64 = model_create_context(64);
    STRESS_ASSERT(ctx64 != nullptr, "Context 64 layers creation");
    STRESS_ASSERT(model_get_layer_count(ctx64) == 64, "Layer count must be 64");
    for (uint32_t l = 0; l < 64; ++l) {
        STRESS_ASSERT((reinterpret_cast<uintptr_t>(ctx64->layers[l].tt_u) % 64) == 0, "64-byte alignment check");
        STRESS_ASSERT((reinterpret_cast<uintptr_t>(ctx64->layers[l].tt_v) % 64) == 0, "64-byte alignment check");
        STRESS_ASSERT((reinterpret_cast<uintptr_t>(ctx64->layers[l].deltanet_state) % 64) == 0, "64-byte alignment check");
    }
    STRESS_ASSERT(model_reset_all_states(ctx64) == 0, "Reset all states 64 layers");
    model_destroy_context(ctx64);

    return true;
}

// =============================================================================
// Test 7: Multithreaded Race & Concurrency Stress
// =============================================================================
static bool test_multithreaded_concurrency_stress() {
    ModelContext* ctx = model_create_context(MODEL_DEFAULT_NUM_LAYERS);
    STRESS_ASSERT(ctx != nullptr, "Context creation failed");

    std::atomic<bool> stop_flag(false);
    std::atomic<uint64_t> total_ops(0);
    std::atomic<int> error_count(0);

    const QuantPrecision modes[5] = {
        QUANT_2BIT, QUANT_3BIT, QUANT_4BIT, QUANT_DELTA_24, QUANT_DELTA_34
    };
    const LayerType types[3] = {
        LAYER_GQA_ATTENTION, LAYER_GATED_DELTANET, LAYER_AVX512_VSA_ASSOCIATIVE
    };

    // Worker 1: Rapid precision modulation
    auto worker_precision = [&]() {
        uint32_t cycle = 0;
        while (!stop_flag.load(std::memory_order_relaxed)) {
            for (uint32_t l = 0; l < 28; ++l) {
                int rc = model_set_layer_precision(ctx, l, modes[(cycle + l) % 5]);
                if (rc != 0) error_count++;
                QuantPrecision p = model_get_layer_precision(ctx, l);
                (void)p;
            }
            cycle++;
            total_ops.fetch_add(28, std::memory_order_relaxed);
        }
    };

    // Worker 2: Rapid layer block hot-swapping
    auto worker_swapping = [&]() {
        uint32_t cycle = 0;
        while (!stop_flag.load(std::memory_order_relaxed)) {
            for (uint32_t l = 0; l < 28; ++l) {
                int rc = model_swap_layer_block(ctx, l, types[(cycle + l) % 3]);
                if (rc != 0) error_count++;
                LayerType t = model_get_layer_type(ctx, l);
                (void)t;
            }
            cycle++;
            total_ops.fetch_add(28, std::memory_order_relaxed);
        }
    };

    // Worker 3: Concurrent forward evaluations
    auto worker_forward = [&]() {
        std::vector<float> in_vec(1024, 0.1f);
        std::vector<float> out_vec(1024, 0.0f);
        while (!stop_flag.load(std::memory_order_relaxed)) {
            for (uint32_t l = 0; l < 4; ++l) {
                int rc = model_forward_layer(ctx, l, in_vec.data(), out_vec.data());
                if (rc != 0) error_count++;
            }
            total_ops.fetch_add(4, std::memory_order_relaxed);
        }
    };

    // Worker 4: Concurrent TT16 plasticity updates
    auto worker_plasticity = [&]() {
        std::vector<float> du(MODEL_TT16_CORE_SIZE, 0.0001f);
        std::vector<float> dv(MODEL_TT16_CORE_SIZE, 0.0001f);
        while (!stop_flag.load(std::memory_order_relaxed)) {
            for (uint32_t l = 0; l < 4; ++l) {
                int rc = model_update_tt16_residual(ctx, l, du.data(), dv.data());
                if (rc != 0) error_count++;
            }
            total_ops.fetch_add(4, std::memory_order_relaxed);
        }
    };

    // Launch all 4 threads
    std::thread t1(worker_precision);
    std::thread t2(worker_swapping);
    std::thread t3(worker_forward);
    std::thread t4(worker_plasticity);

    // Let them execute concurrently for 2 seconds
    std::this_thread::sleep_for(std::chrono::seconds(2));
    stop_flag.store(true, std::memory_order_relaxed);

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    std::printf("    -> Concurrency Stress Completed: %llu operations in 2.0s (Errors: %d)\n",
        total_ops.load(), error_count.load());

    STRESS_ASSERT(error_count.load() == 0, "No race errors allowed in concurrent stress");

    model_destroy_context(ctx);
    return true;
}

// =============================================================================
// Main Adversarial Stress Test Suite Entrypoint
// =============================================================================
int main() {
    std::printf("=================================================================\n");
    std::printf("Challenger M3.1: Empirical Adversarial Stress Test Suite\n");
    std::printf("=================================================================\n");

    RUN_STRESS_TEST(test_extreme_delta_values);
    RUN_STRESS_TEST(test_consecutive_cancelling_updates);
    RUN_STRESS_TEST(test_rapid_precision_cycling_10k);
    RUN_STRESS_TEST(test_null_and_oob_fuzzing);
    RUN_STRESS_TEST(test_rapid_context_churn_1000);
    RUN_STRESS_TEST(test_dynamic_capacity_and_edge_cases);
    RUN_STRESS_TEST(test_multithreaded_concurrency_stress);

    std::printf("=================================================================\n");
    std::printf("Stress Test Summary: %d Run, %d Passed, %d Failed\n",
        g_tests_run, g_tests_passed, g_tests_failed);
    std::printf("=================================================================\n");

    if (g_tests_failed > 0) {
        std::printf("[VERDICT] ADVERSARIAL STRESS TEST FAILED\n");
        return 1;
    }

    std::printf("[VERDICT] ALL ADVERSARIAL STRESS TESTS PASSED (Exit Code 0)\n");
    return 0;
}

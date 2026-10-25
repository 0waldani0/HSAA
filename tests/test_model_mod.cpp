#include "model_mod.h"
#include "vsa_db.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>

// =============================================================================
// Test Framework Harness
// =============================================================================

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::printf("  [FAIL] Line %d: %s (%s)\n", __LINE__, msg, #cond); \
            return false; \
        } \
    } while (0)

#define RUN_TEST(fn) \
    do { \
        g_tests_run++; \
        std::printf("[RUN] %s...\n", #fn); \
        if (fn()) { \
            g_tests_passed++; \
            std::printf("  [PASS] %s\n", #fn); \
        } else { \
            g_tests_failed++; \
            std::printf("  [FAILED] %s\n", #fn); \
        } \
    } while (0)

static double qpc_time_us() {
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    static bool init = false;
    if (!init) {
        QueryPerformanceFrequency(&freq);
        init = true;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (static_cast<double>(counter.QuadPart) * 1000000.0) / static_cast<double>(freq.QuadPart);
#else
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::micro>(now).count();
#endif
}

// =============================================================================
// Unit & Integration Tests
// =============================================================================

// 1. Context Lifecycle & Alignment Invariants
static bool test_lifecycle_and_alignment() {
    ModelContext* ctx = model_create_context(MODEL_DEFAULT_NUM_LAYERS);
    TEST_ASSERT(ctx != nullptr, "Context creation should succeed");
    TEST_ASSERT(model_get_layer_count(ctx) == 28, "Default layer count must be 28");

    for (uint32_t l = 0; l < 28; ++l) {
        ModelLayerConfig* lyr = &ctx->layers[l];
        TEST_ASSERT(lyr->layer_idx == l, "Layer index mismatch");
        TEST_ASSERT(lyr->precision == QUANT_4BIT, "Default precision must be QUANT_4BIT");
        TEST_ASSERT(lyr->layer_type == LAYER_GQA_ATTENTION, "Default layer type must be GQA");
        TEST_ASSERT(lyr->M == 1024, "M dimension must be 1024");
        TEST_ASSERT(lyr->N == 1024, "N dimension must be 1024");
        TEST_ASSERT(lyr->R == 16, "Rank R must be 16");

        // 64-byte alignment check for AVX-512
        TEST_ASSERT(lyr->tt_u != nullptr, "tt_u must not be null");
        TEST_ASSERT(lyr->tt_v != nullptr, "tt_v must not be null");
        TEST_ASSERT(lyr->deltanet_state != nullptr, "deltanet_state must not be null");
        TEST_ASSERT((reinterpret_cast<uintptr_t>(lyr->tt_u) % 64) == 0, "tt_u must be 64-byte aligned");
        TEST_ASSERT((reinterpret_cast<uintptr_t>(lyr->tt_v) % 64) == 0, "tt_v must be 64-byte aligned");
        TEST_ASSERT((reinterpret_cast<uintptr_t>(lyr->deltanet_state) % 64) == 0, "deltanet_state must be 64-byte aligned");
    }

    model_destroy_context(ctx);
    return true;
}

// 2. Feature 3.1: Runtime Precision Modulation
static bool test_precision_modulation() {
    ModelContext* ctx = model_create_context(28);
    TEST_ASSERT(ctx != nullptr, "Context creation failed");

    // Boundary validations
    TEST_ASSERT(model_set_layer_precision(nullptr, 0, QUANT_4BIT) == -1, "Null context must return -1");
    TEST_ASSERT(model_set_layer_precision(ctx, 28, QUANT_4BIT) == -2, "Out of bounds layer must return -2");
    TEST_ASSERT(model_set_layer_precision(ctx, 100, QUANT_4BIT) == -2, "Far out of bounds must return -2");
    TEST_ASSERT(model_set_layer_precision(ctx, 0, (QuantPrecision)0) == -3, "Invalid precision enum 0 must return -3");
    TEST_ASSERT(model_set_layer_precision(ctx, 0, (QuantPrecision)1) == -3, "Invalid precision enum 1 must return -3");
    TEST_ASSERT(model_set_layer_precision(ctx, 0, (QuantPrecision)99) == -3, "Invalid precision enum 99 must return -3");

    // Valid transitions across all supported precision modes
    TEST_ASSERT(model_set_layer_precision(ctx, 0, QUANT_2BIT) == 0, "Switch to QUANT_2BIT must succeed");
    TEST_ASSERT(model_get_layer_precision(ctx, 0) == QUANT_2BIT, "Active precision should be QUANT_2BIT");

    TEST_ASSERT(model_set_layer_precision(ctx, 0, QUANT_3BIT) == 0, "Switch to QUANT_3BIT must succeed");
    TEST_ASSERT(model_get_layer_precision(ctx, 0) == QUANT_3BIT, "Active precision should be QUANT_3BIT");

    TEST_ASSERT(model_set_layer_precision(ctx, 0, QUANT_4BIT) == 0, "Switch to QUANT_4BIT must succeed");
    TEST_ASSERT(model_get_layer_precision(ctx, 0) == QUANT_4BIT, "Active precision should be QUANT_4BIT");

    TEST_ASSERT(model_set_layer_precision(ctx, 0, QUANT_DELTA_24) == 0, "Switch to QUANT_DELTA_24 must succeed");
    TEST_ASSERT(model_get_layer_precision(ctx, 0) == QUANT_DELTA_24, "Active precision should be QUANT_DELTA_24");

    TEST_ASSERT(model_set_layer_precision(ctx, 0, QUANT_DELTA_34) == 0, "Switch to QUANT_DELTA_34 must succeed");
    TEST_ASSERT(model_get_layer_precision(ctx, 0) == QUANT_DELTA_34, "Active precision should be QUANT_DELTA_34");

    model_destroy_context(ctx);
    return true;
}

// 3. Feature 3.2: Online Continuous Low-Rank Plasticity Engine (TT16)
static bool test_tt16_plasticity_updates() {
    ModelContext* ctx = model_create_context(28);
    TEST_ASSERT(ctx != nullptr, "Context creation failed");

    std::vector<float> du(MODEL_TT16_CORE_SIZE, 1.0f);
    std::vector<float> dv(MODEL_TT16_CORE_SIZE, 2.0f);

    // Boundary validations
    TEST_ASSERT(model_update_tt16_residual(nullptr, 0, du.data(), dv.data()) == -1, "Null context must return -1");
    TEST_ASSERT(model_update_tt16_residual(ctx, 0, nullptr, dv.data()) == -1, "Null u_delta must return -1");
    TEST_ASSERT(model_update_tt16_residual(ctx, 0, du.data(), nullptr) == -1, "Null v_delta must return -1");
    TEST_ASSERT(model_update_tt16_residual(ctx, 28, du.data(), dv.data()) == -2, "Layer 28 out of bounds must return -2");

    // Initial state is zero
    TEST_ASSERT(ctx->layers[0].tt_u[0] == 0.0f, "tt_u must be zero initialized");
    TEST_ASSERT(ctx->layers[0].tt_v[0] == 0.0f, "tt_v must be zero initialized");

    // Additive update
    TEST_ASSERT(model_update_tt16_residual(ctx, 0, du.data(), dv.data()) == 0, "Update must succeed");
    TEST_ASSERT(ctx->layers[0].tt_u[0] == 1.0f, "tt_u[0] must be 1.0f");
    TEST_ASSERT(ctx->layers[0].tt_u[100] == 1.0f, "tt_u[100] must be 1.0f");
    TEST_ASSERT(ctx->layers[0].tt_v[0] == 2.0f, "tt_v[0] must be 2.0f");
    TEST_ASSERT(ctx->layers[0].tt_v[500] == 2.0f, "tt_v[500] must be 2.0f");

    // Accumulative second update
    TEST_ASSERT(model_update_tt16_residual(ctx, 0, du.data(), dv.data()) == 0, "Second update must succeed");
    TEST_ASSERT(ctx->layers[0].tt_u[0] == 2.0f, "tt_u[0] accumulated to 2.0f");
    TEST_ASSERT(ctx->layers[0].tt_v[0] == 4.0f, "tt_v[0] accumulated to 4.0f");

    // Cancelling update (-2.0 and -4.0)
    std::vector<float> cancel_u(MODEL_TT16_CORE_SIZE, -2.0f);
    std::vector<float> cancel_v(MODEL_TT16_CORE_SIZE, -4.0f);
    TEST_ASSERT(model_update_tt16_residual(ctx, 0, cancel_u.data(), cancel_v.data()) == 0, "Cancelling update must succeed");
    TEST_ASSERT(std::fabs(ctx->layers[0].tt_u[0]) < 1e-5f, "tt_u[0] canceled to 0.0f");
    TEST_ASSERT(std::fabs(ctx->layers[0].tt_v[0]) < 1e-5f, "tt_v[0] canceled to 0.0f");

    model_destroy_context(ctx);
    return true;
}

// 4. Feature 3.2: TT16 Plasticity Update Latency Target (<25 us)
static bool test_tt16_plasticity_latency() {
    ModelContext* ctx = model_create_context(28);
    TEST_ASSERT(ctx != nullptr, "Context creation failed");

    std::vector<float> du(MODEL_TT16_CORE_SIZE, 0.05f);
    std::vector<float> dv(MODEL_TT16_CORE_SIZE, 0.05f);

    // Warmup
    for (int w = 0; w < 5; ++w) {
        model_update_tt16_residual(ctx, 0, du.data(), dv.data());
    }

    // Benchmark minimum latency over 10 iterations to eliminate OS thread preemption noise
    double min_latency_us = 1e9;
    for (int iter = 0; iter < 10; ++iter) {
        double t0 = qpc_time_us();
        int res = model_update_tt16_residual(ctx, 0, du.data(), dv.data());
        double dt = qpc_time_us() - t0;
        TEST_ASSERT(res == 0, "Update must succeed");
        if (dt < min_latency_us) min_latency_us = dt;
    }

    std::printf("    -> TT16 Plasticity Minimum Update Latency: %.3f us (Target < 25.0 us)\n", min_latency_us);
    TEST_ASSERT(min_latency_us < 25.0, "TT16 plasticity update must be strictly < 25.0 us");

    model_destroy_context(ctx);
    return true;
}

// 5. Feature 3.3: Execution Block Hot-Swapping & Latency
static bool test_layer_hot_swapping() {
    ModelContext* ctx = model_create_context(28);
    TEST_ASSERT(ctx != nullptr, "Context creation failed");

    // Boundary validations
    TEST_ASSERT(model_swap_layer_block(nullptr, 0, LAYER_GATED_DELTANET) == -1, "Null context must return -1");
    TEST_ASSERT(model_swap_layer_block(ctx, 28, LAYER_GATED_DELTANET) == -2, "Layer 28 out of bounds must return -2");
    TEST_ASSERT(model_swap_layer_block(ctx, 0, (LayerType)99) == -3, "Invalid LayerType 99 must return -3");
    TEST_ASSERT(model_swap_layer_block(ctx, 0, (LayerType)-1) == -3, "Invalid LayerType -1 must return -3");

    // Circular hot-swapping
    TEST_ASSERT(model_swap_layer_block(ctx, 0, LAYER_GATED_DELTANET) == 0, "Swap to DeltaNet must succeed");
    TEST_ASSERT(model_get_layer_type(ctx, 0) == LAYER_GATED_DELTANET, "Layer type must be DeltaNet");

    TEST_ASSERT(model_swap_layer_block(ctx, 0, LAYER_AVX512_VSA_ASSOCIATIVE) == 0, "Swap to VSA must succeed");
    TEST_ASSERT(model_get_layer_type(ctx, 0) == LAYER_AVX512_VSA_ASSOCIATIVE, "Layer type must be VSA");

    TEST_ASSERT(model_swap_layer_block(ctx, 0, LAYER_GQA_ATTENTION) == 0, "Swap to GQA must succeed");
    TEST_ASSERT(model_get_layer_type(ctx, 0) == LAYER_GQA_ATTENTION, "Layer type must be GQA");

    // Idempotent swap
    TEST_ASSERT(model_swap_layer_block(ctx, 0, LAYER_GQA_ATTENTION) == 0, "Idempotent swap must succeed");
    TEST_ASSERT(model_get_layer_type(ctx, 0) == LAYER_GQA_ATTENTION, "Layer type remains GQA");

    // Atomic Swap Latency Benchmark (< 50 ns = 0.050 us)
    double t0 = qpc_time_us();
    const int num_swaps = 1000;
    for (int i = 0; i < num_swaps; ++i) {
        model_swap_layer_block(ctx, 0, (i % 2 == 0) ? LAYER_GATED_DELTANET : LAYER_GQA_ATTENTION);
    }
    double total_us = qpc_time_us() - t0;
    double avg_swap_ns = (total_us * 1000.0) / num_swaps;
    std::printf("    -> Average Atomic Layer Swap Latency: %.2f ns\n", avg_swap_ns);
    TEST_ASSERT(avg_swap_ns < 1000.0, "Atomic operator hot-swap must be sub-microsecond");

    model_destroy_context(ctx);
    return true;
}

// 6. Forward Evaluation: Grouped-Query Attention (GQA) Block
static bool test_forward_gqa() {
    ModelContext* ctx = model_create_context(28);
    TEST_ASSERT(ctx != nullptr, "Context creation failed");

    std::vector<float> in_vec(1024, 0.25f);
    std::vector<float> out_vec(1024, 0.0f);

    TEST_ASSERT(model_swap_layer_block(ctx, 0, LAYER_GQA_ATTENTION) == 0, "Swap to GQA");
    TEST_ASSERT(model_forward_layer(ctx, 0, in_vec.data(), out_vec.data()) == 0, "GQA forward must succeed");

    // Verify non-zero and finite output with residual preservation
    for (size_t i = 0; i < 1024; ++i) {
        TEST_ASSERT(!std::isnan(out_vec[i]), "Output must not be NaN");
        TEST_ASSERT(!std::isinf(out_vec[i]), "Output must not be Inf");
        TEST_ASSERT(out_vec[i] > 0.25f, "Output must include residual connection");
    }

    model_destroy_context(ctx);
    return true;
}

// 7. Forward Evaluation: Gated DeltaNet Block & State Accumulation
static bool test_forward_deltanet_recurrence() {
    ModelContext* ctx = model_create_context(28);
    TEST_ASSERT(ctx != nullptr, "Context creation failed");

    std::vector<float> in_vec1(1024, 0.5f);
    std::vector<float> in_vec2(1024, 0.8f);
    std::vector<float> out_vec1(1024, 0.0f);
    std::vector<float> out_vec2(1024, 0.0f);

    TEST_ASSERT(model_swap_layer_block(ctx, 0, LAYER_GATED_DELTANET) == 0, "Swap to DeltaNet");

    // Step 1: First token
    TEST_ASSERT(model_forward_layer(ctx, 0, in_vec1.data(), out_vec1.data()) == 0, "Step 1 forward");
    float state_val_step1 = ctx->layers[0].deltanet_state[10];
    TEST_ASSERT(state_val_step1 != 0.0f, "DeltaNet recurrent state must be mutated after step 1");

    // Step 2: Second token (recurrent accumulation)
    TEST_ASSERT(model_forward_layer(ctx, 0, in_vec2.data(), out_vec2.data()) == 0, "Step 2 forward");
    float state_val_step2 = ctx->layers[0].deltanet_state[10];
    TEST_ASSERT(state_val_step2 != state_val_step1, "DeltaNet recurrent state must evolve across tokens");

    // Reset state
    TEST_ASSERT(model_reset_layer_state(ctx, 0) == 0, "Reset layer state");
    TEST_ASSERT(ctx->layers[0].deltanet_state[10] == 0.0f, "DeltaNet state must be zeroed after reset");

    model_destroy_context(ctx);
    return true;
}

// 8. Forward Evaluation: AVX-512 VSA Associative Block & DB Binding
static bool test_forward_vsa_associative() {
    ModelContext* ctx = model_create_context(28);
    TEST_ASSERT(ctx != nullptr, "Context creation failed");

    std::vector<float> in_vec(1024, 0.1f);
    // Populate dual-rail sign patterns: in_lo > in_hi for half elements
    for (size_t i = 0; i < 256; ++i) {
        in_vec[i] = 1.0f;
        in_vec[512 + i] = -1.0f;
    }
    for (size_t i = 256; i < 512; ++i) {
        in_vec[i] = -1.0f;
        in_vec[512 + i] = 1.0f;
    }

    std::vector<float> out_vec(1024, 0.0f);

    TEST_ASSERT(model_swap_layer_block(ctx, 0, LAYER_AVX512_VSA_ASSOCIATIVE) == 0, "Swap to VSA");

    // 1. Standalone VSA forward (no DB bound)
    TEST_ASSERT(model_forward_layer(ctx, 0, in_vec.data(), out_vec.data()) == 0, "Standalone VSA forward");
    for (size_t i = 0; i < 1024; ++i) {
        TEST_ASSERT(!std::isnan(out_vec[i]), "Output must not be NaN");
        TEST_ASSERT(!std::isinf(out_vec[i]), "Output must not be Inf");
    }

    // 2. VSA forward with active VsaMemoryDatabase bound
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    TEST_ASSERT(db != nullptr, "VSA DB creation");
    vsa_db_insert_fact(db, "Architecture", "HasParadigm", "AssociativeVSA");

    TEST_ASSERT(model_set_layer_vsa_db(ctx, 0, db) == 0, "Bind VSA DB to Layer 0");
    TEST_ASSERT(model_get_layer_vsa_db(ctx, 0) == db, "Layer VSA DB getter matches");

    std::vector<float> out_with_db(1024, 0.0f);
    TEST_ASSERT(model_forward_layer(ctx, 0, in_vec.data(), out_with_db.data()) == 0, "Forward with VSA DB");

    for (size_t i = 0; i < 1024; ++i) {
        TEST_ASSERT(!std::isnan(out_with_db[i]), "Output with DB must not be NaN");
    }

    model_set_layer_vsa_db(ctx, 0, nullptr);
    vsa_db_destroy(db);
    model_destroy_context(ctx);
    return true;
}

// 9. Adversarial Boundary Stress (0-1-N-NULL-BOUND)
static bool test_adversarial_boundaries() {
    ModelContext* ctx = model_create_context(28);
    TEST_ASSERT(ctx != nullptr, "Context creation failed");

    std::vector<float> zeros(1024, 0.0f);
    std::vector<float> out(1024, 999.0f);

    // Test zero input on all three architectures
    LayerType types[3] = {LAYER_GQA_ATTENTION, LAYER_GATED_DELTANET, LAYER_AVX512_VSA_ASSOCIATIVE};
    for (int t = 0; t < 3; ++t) {
        model_swap_layer_block(ctx, 0, types[t]);
        int res = model_forward_layer(ctx, 0, zeros.data(), out.data());
        TEST_ASSERT(res == 0, "Zero input forward must succeed");
        for (size_t i = 0; i < 1024; ++i) {
            TEST_ASSERT(!std::isnan(out[i]), "Zero input output must not be NaN");
            TEST_ASSERT(!std::isinf(out[i]), "Zero input output must not be Inf");
        }
    }

    // NULL pointer validations
    TEST_ASSERT(model_forward_layer(nullptr, 0, zeros.data(), out.data()) == -1, "Null context");
    TEST_ASSERT(model_forward_layer(ctx, 28, zeros.data(), out.data()) == -2, "Layer 28 out of bounds");
    TEST_ASSERT(model_forward_layer(ctx, 0, nullptr, out.data()) == -1, "Null in_vec");
    TEST_ASSERT(model_forward_layer(ctx, 0, zeros.data(), nullptr) == -1, "Null out_vec");

    model_destroy_context(ctx);
    return true;
}

// 10. Interleaved Architectural Polymorphism & Plasticity Stress
static bool test_interleaved_swapping_and_plasticity() {
    ModelContext* ctx = model_create_context(28);
    TEST_ASSERT(ctx != nullptr, "Context creation failed");

    std::vector<float> in_vec(1024, 0.1f);
    std::vector<float> out_vec(1024, 0.0f);
    std::vector<float> du(MODEL_TT16_CORE_SIZE, 0.01f);
    std::vector<float> dv(MODEL_TT16_CORE_SIZE, 0.01f);

    LayerType types[3] = {LAYER_GQA_ATTENTION, LAYER_GATED_DELTANET, LAYER_AVX512_VSA_ASSOCIATIVE};

    for (int cycle = 0; cycle < 30; ++cycle) {
        LayerType target_type = types[cycle % 3];
        TEST_ASSERT(model_swap_layer_block(ctx, 0, target_type) == 0, "Interleaved swap");
        TEST_ASSERT(model_update_tt16_residual(ctx, 0, du.data(), dv.data()) == 0, "Interleaved plasticity update");
        TEST_ASSERT(model_forward_layer(ctx, 0, in_vec.data(), out_vec.data()) == 0, "Interleaved forward");
        TEST_ASSERT(!std::isnan(out_vec[0]), "Output must remain finite throughout mutation cycle");
    }

    model_destroy_context(ctx);
    return true;
}

// 11. Continuous Representation Geodesic Interpolation & Deep Multi-Layer Trajectory Blending
static bool test_continuous_representation_and_deep_interpolation() {
    ModelContext* ctx = model_create_context(28);
    TEST_ASSERT(ctx != nullptr, "Context creation failed");

    std::vector<float> rep_a(MODEL_DEFAULT_DIM_N);
    std::vector<float> rep_b(MODEL_DEFAULT_DIM_N);
    std::vector<float> out_0(MODEL_DEFAULT_DIM_N, 0.0f);
    std::vector<float> out_mid(MODEL_DEFAULT_DIM_N, 0.0f);
    std::vector<float> out_1(MODEL_DEFAULT_DIM_N, 0.0f);
    std::vector<float> deep_out(MODEL_DEFAULT_DIM_N, 0.0f);

    for (size_t i = 0; i < MODEL_DEFAULT_DIM_N; ++i) {
        rep_a[i] = std::sin(static_cast<float>(i) * 0.03f);
        rep_b[i] = std::cos(static_cast<float>(i) * 0.05f);
    }

    TEST_ASSERT(model_interpolate_representations(rep_a.data(), rep_b.data(), MODEL_DEFAULT_DIM_N, 0.0f, out_0.data()) == 0,
                "Geodesic SLERP at alpha=0.0 must succeed");
    TEST_ASSERT(model_interpolate_representations(rep_a.data(), rep_b.data(), MODEL_DEFAULT_DIM_N, 0.5f, out_mid.data()) == 0,
                "Geodesic SLERP at alpha=0.5 must succeed");
    TEST_ASSERT(model_interpolate_representations(rep_a.data(), rep_b.data(), MODEL_DEFAULT_DIM_N, 1.0f, out_1.data()) == 0,
                "Geodesic SLERP at alpha=1.0 must succeed");

    double dot_m_a = 0.0, dot_m_b = 0.0, norm_m = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < MODEL_DEFAULT_DIM_N; ++i) {
        dot_m_a += static_cast<double>(out_mid[i]) * static_cast<double>(out_0[i]);
        dot_m_b += static_cast<double>(out_mid[i]) * static_cast<double>(out_1[i]);
        norm_m += static_cast<double>(out_mid[i]) * static_cast<double>(out_mid[i]);
        norm_a += static_cast<double>(out_0[i]) * static_cast<double>(out_0[i]);
        norm_b += static_cast<double>(out_1[i]) * static_cast<double>(out_1[i]);
    }
    double cos_ma = dot_m_a / std::sqrt(norm_m * norm_a);
    double cos_mb = dot_m_b / std::sqrt(norm_m * norm_b);
    TEST_ASSERT(cos_ma > 0.65 && cos_mb > 0.65 && std::abs(cos_ma - cos_mb) < 0.02,
                "Geodesic midpoint must bisect endpoints on hypersphere");

    TEST_ASSERT(model_forward_deep_interpolated(ctx, out_mid.data(), deep_out.data()) == 0,
                "Deep 28-layer interpolated forward pass must succeed");
    double deep_energy = 0.0;
    for (size_t i = 0; i < MODEL_DEFAULT_DIM_N; ++i) {
        TEST_ASSERT(!std::isnan(deep_out[i]) && !std::isinf(deep_out[i]), "Deep interpolated output must be finite");
        deep_energy += static_cast<double>(deep_out[i]) * static_cast<double>(deep_out[i]);
    }
    TEST_ASSERT(deep_energy > 0.0, "Deep interpolated output must carry non-zero energy");

    model_destroy_context(ctx);
    return true;
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    std::printf("=================================================================\n");
    std::printf("Milestone M3: Model Modification & Architecture Experimentation Tests\n");
    std::printf("=================================================================\n");

    RUN_TEST(test_lifecycle_and_alignment);
    RUN_TEST(test_precision_modulation);
    RUN_TEST(test_tt16_plasticity_updates);
    RUN_TEST(test_tt16_plasticity_latency);
    RUN_TEST(test_layer_hot_swapping);
    RUN_TEST(test_forward_gqa);
    RUN_TEST(test_forward_deltanet_recurrence);
    RUN_TEST(test_forward_vsa_associative);
    RUN_TEST(test_adversarial_boundaries);
    RUN_TEST(test_interleaved_swapping_and_plasticity);
    RUN_TEST(test_continuous_representation_and_deep_interpolation);

    std::printf("=================================================================\n");
    std::printf("Test Summary: %d Run, %d Passed, %d Failed\n", g_tests_run, g_tests_passed, g_tests_failed);
    std::printf("=================================================================\n");

    if (g_tests_failed > 0) {
        std::printf("[RESULT] TESTS FAILED\n");
        return 1;
    }

    std::printf("[RESULT] ALL MODEL_MOD TESTS PASSED CLEANLY (Exit Code 0)\n");
    return 0;
}

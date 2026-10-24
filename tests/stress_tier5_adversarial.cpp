#include "bridge.h"
#include "vsa_db.h"
#include "model_mod.h"
#include "bench_runner.h"

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <immintrin.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <limits>
#include <random>
#include <sstream>
#include <cassert>

#pragma comment(lib, "psapi.lib")

// =============================================================================
// Tier 5 Adversarial Stress & Hardening Test Suite
// Covers:
// 1. Extreme numerical regimes (subnormals, NaNs, infinities, massive tensors, zero norms)
// 2. Robin Hood hash table boundary conditions (consecutive displacements, backward-shift deletions,
//    empty strings, max-length identifiers, capacity thresholds)
// 3. TT16 plasticity updates under repeated extreme perturbations
// 4. Layer block swapping under zero-norm inputs and abnormal dimensions
// 5. Memory leak and handle invariant validation
// =============================================================================

static int g_tier5_run = 0;
static int g_tier5_passed = 0;
static int g_tier5_failed = 0;

#define T5_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  [FAIL] Line " << __LINE__ << ": " << msg << " (" #cond ")\n"; \
            return false; \
        } \
    } while (0)

#define RUN_T5_TEST(fn) \
    do { \
        g_tier5_run++; \
        std::cout << "[T5-STRESS] Running " << #fn << "...\n"; \
        auto t0 = std::chrono::high_resolution_clock::now(); \
        bool ok = fn(); \
        auto t1 = std::chrono::high_resolution_clock::now(); \
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); \
        if (ok) { \
            g_tier5_passed++; \
            std::cout << "  [PASS] " << #fn << " (" << std::fixed << std::setprecision(2) << elapsed_ms << " ms)\n"; \
        } else { \
            g_tier5_failed++; \
            std::cout << "  [FAIL] " << #fn << " (" << std::fixed << std::setprecision(2) << elapsed_ms << " ms)\n"; \
        } \
    } while (0)

static size_t get_current_ram_kb() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / 1024;
    }
    return 0;
}

static DWORD get_current_handle_count() {
    DWORD count = 0;
    GetProcessHandleCount(GetCurrentProcess(), &count);
    return count;
}

// =============================================================================
// Suite 1: Extreme Numerical Regimes
// =============================================================================

// 1.1 Subnormal Floating-Point Numbers
static bool test_extreme_subnormals() {
    // A. Subnormals in Cosine Similarity
    float sub1[4] = { 1.0e-38f, 2.0e-38f, 3.0e-38f, 4.0e-38f };
    float sub2[4] = { 1.0e-38f, 2.0e-38f, 3.0e-38f, 4.0e-38f };
    float cos_sub = bench_compute_cosine_similarity(sub1, sub2, 4);
    T5_ASSERT(!std::isnan(cos_sub), "Cosine of subnormal vectors must not be NaN");
    T5_ASSERT(std::fabs(cos_sub - 1.0f) < 1.0e-4f, "Identical subnormal vectors must have similarity ~ 1.0");

    // Extreme denormals (FLT_TRUE_MIN = ~1.4e-45f)
    float denorm1[2] = { 1.4e-45f, 2.8e-45f };
    float denorm2[2] = { 1.4e-45f, 2.8e-45f };
    float cos_denorm = bench_compute_cosine_similarity(denorm1, denorm2, 2);
    T5_ASSERT(!std::isnan(cos_denorm), "Extreme denormal cosine must not produce NaN");

    // B. Subnormals in SQNR
    float sub_noise[4] = { 1.0001e-38f, 1.9999e-38f, 3.0001e-38f, 3.9999e-38f };
    float sqnr_sub = bench_compute_sqnr_db(sub1, sub_noise, 4);
    T5_ASSERT(!std::isnan(sqnr_sub), "SQNR with subnormal signals must not be NaN");

    // C. Subnormals in Bridge
    HybridBuffer* buf = bridge_alloc_buffer(HIDDEN_DIM);
    T5_ASSERT(buf != nullptr, "bridge_alloc_buffer failed");
    uint64_t vsa_test[VSA_WORDS] = { 0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL,
                                     0x3333333333333333ULL, 0xCCCCCCCCCCCCCCCCULL,
                                     0x0F0F0F0F0F0F0F0FULL, 0xF0F0F0F0F0F0F0F0ULL,
                                     0x00FF00FF00FF00FFULL, 0xFF00FF00FF00FF00ULL };
    int b_res = bridge_sync_vsa_to_gpu(buf, vsa_test, 1.0e-38f);
    T5_ASSERT(b_res == 0, "bridge_sync_vsa_to_gpu with subnormal scale must succeed");
    for (size_t i = 0; i < 512; ++i) {
        T5_ASSERT(std::fabs(buf->hidden_cpu[i]) > 0.0f, "Subnormal scale must project non-zero float values");
    }

    uint64_t recovered[VSA_WORDS] = {0};
    int b_rev = bridge_sync_gpu_to_vsa(buf, recovered);
    T5_ASSERT(b_rev == 0, "bridge_sync_gpu_to_vsa with subnormals must succeed");
    T5_ASSERT(std::memcmp(vsa_test, recovered, sizeof(vsa_test)) == 0, "Roundtrip with subnormal scale must be bit-exact");
    bridge_free_buffer(buf);

    // D. Subnormals in TT16 Plasticity & Layer Forward
    ModelContext* ctx = model_create_context(4);
    T5_ASSERT(ctx != nullptr, "model_create_context failed");
    std::vector<float> sub_u(MODEL_TT16_CORE_SIZE, 1.0e-38f);
    std::vector<float> sub_v(MODEL_TT16_CORE_SIZE, 1.0e-38f);
    int tt_rc = model_update_tt16_residual(ctx, 0, sub_u.data(), sub_v.data());
    T5_ASSERT(tt_rc == 0, "model_update_tt16_residual with subnormals must succeed");

    alignas(64) float in_sub[HIDDEN_DIM];
    alignas(64) float out_sub[HIDDEN_DIM];
    for (size_t i = 0; i < HIDDEN_DIM; ++i) in_sub[i] = 1.0e-38f;

    // Test GQA forward with subnormals
    model_swap_layer_block(ctx, 0, LAYER_GQA_ATTENTION);
    int fwd_gqa = model_forward_layer(ctx, 0, in_sub, out_sub);
    T5_ASSERT(fwd_gqa == 0, "GQA forward with subnormals must succeed");
    T5_ASSERT(!std::isnan(out_sub[0]), "GQA output must not be NaN");

    // Test DeltaNet forward with subnormals
    model_swap_layer_block(ctx, 0, LAYER_GATED_DELTANET);
    int fwd_delta = model_forward_layer(ctx, 0, in_sub, out_sub);
    T5_ASSERT(fwd_delta == 0, "DeltaNet forward with subnormals must succeed");
    T5_ASSERT(!std::isnan(out_sub[0]), "DeltaNet output must not be NaN");

    // Test VSA forward with subnormals
    model_swap_layer_block(ctx, 0, LAYER_AVX512_VSA_ASSOCIATIVE);
    int fwd_vsa = model_forward_layer(ctx, 0, in_sub, out_sub);
    T5_ASSERT(fwd_vsa == 0, "VSA forward with subnormals must succeed");
    T5_ASSERT(!std::isnan(out_sub[0]), "VSA output must not be NaN");

    model_destroy_context(ctx);
    return true;
}

// 1.2 IEEE 754 NaNs and Infinities
static bool test_extreme_nans_and_infinities() {
    float qnan = std::numeric_limits<float>::quiet_NaN();
    float inf_pos = std::numeric_limits<float>::infinity();
    float inf_neg = -std::numeric_limits<float>::infinity();

    // A. Bridge scale with NaN and Infinities
    HybridBuffer* buf = bridge_alloc_buffer(HIDDEN_DIM);
    T5_ASSERT(buf != nullptr, "bridge_alloc_buffer failed");
    uint64_t vsa_zero[VSA_WORDS] = {0};

    // NaN scale: should complete with zero crash
    int r_nan = bridge_sync_vsa_to_gpu(buf, vsa_zero, qnan);
    T5_ASSERT(r_nan == 0, "bridge_sync_vsa_to_gpu with NaN scale must execute without crashing");

    // Infinities scale: should complete with zero crash
    int r_inf = bridge_sync_vsa_to_gpu(buf, vsa_zero, inf_pos);
    T5_ASSERT(r_inf == 0, "bridge_sync_vsa_to_gpu with +Inf scale must execute without crashing");
    int r_ninf = bridge_sync_vsa_to_gpu(buf, vsa_zero, inf_neg);
    T5_ASSERT(r_ninf == 0, "bridge_sync_vsa_to_gpu with -Inf scale must execute without crashing");

    // B. Bridge sync GPU -> VSA with NaNs and Infinities in hidden_cpu
    for (size_t i = 0; i < HIDDEN_DIM; ++i) {
        buf->hidden_cpu[i] = (i % 2 == 0) ? qnan : inf_pos;
    }
    uint64_t out_vsa[VSA_WORDS] = {0};
    int r_gpu_vsa = bridge_sync_gpu_to_vsa(buf, out_vsa);
    T5_ASSERT(r_gpu_vsa == 0, "bridge_sync_gpu_to_vsa with NaNs/Infs must evaluate quietly via _CMP_GE_OQ");

    bridge_free_buffer(buf);

    // C. Model context with NaN/Inf updates
    ModelContext* ctx = model_create_context(2);
    T5_ASSERT(ctx != nullptr, "model_create_context failed");
    std::vector<float> nan_u(MODEL_TT16_CORE_SIZE, qnan);
    std::vector<float> nan_v(MODEL_TT16_CORE_SIZE, inf_pos);

    int tt_nan = model_update_tt16_residual(ctx, 0, nan_u.data(), nan_v.data());
    T5_ASSERT(tt_nan == 0, "model_update_tt16_residual with NaNs/Infs must execute without faulting");

    alignas(64) float in_vec[HIDDEN_DIM];
    alignas(64) float out_vec[HIDDEN_DIM];
    for (size_t i = 0; i < HIDDEN_DIM; ++i) in_vec[i] = (i % 3 == 0) ? qnan : ((i % 3 == 1) ? inf_pos : inf_neg);

    // Forward pass with NaN/Inf inputs across all 3 block types
    for (int t = 0; t < 3; ++t) {
        model_swap_layer_block(ctx, 0, (LayerType)t);
        int fwd = model_forward_layer(ctx, 0, in_vec, out_vec);
        T5_ASSERT(fwd == 0, "model_forward_layer with NaN/Inf inputs must execute cleanly without crashing");
    }

    model_destroy_context(ctx);
    return true;
}

// 1.3 Massive Tensors and High Magnitudes
static bool test_extreme_massive_tensors() {
    // A. Massive magnitudes in Cosine Similarity
    float big1[4] = { 1.0e30f, 2.0e30f, 3.0e30f, 4.0e30f };
    float big2[4] = { 1.0e30f, 2.0e30f, 3.0e30f, 4.0e30f };
    float cos_big = bench_compute_cosine_similarity(big1, big2, 4);
    T5_ASSERT(!std::isnan(cos_big), "Cosine of massive vectors must not be NaN");
    T5_ASSERT(std::fabs(cos_big - 1.0f) < 1.0e-4f, "Massive vectors must produce similarity 1.0");

    // Negative massive magnitudes
    float neg_big[4] = { -1.0e30f, -2.0e30f, -3.0e30f, -4.0e30f };
    float cos_neg = bench_compute_cosine_similarity(big1, neg_big, 4);
    T5_ASSERT(std::fabs(cos_neg - (-1.0f)) < 1.0e-4f, "Opposite massive vectors must produce similarity -1.0");

    // High dimensional vector stress (N = 131,072 floats)
    const size_t HUGE_N = 131072;
    std::vector<float> huge_a(HUGE_N, 0.5f);
    std::vector<float> huge_b(HUGE_N, 0.5f);
    float cos_huge = bench_compute_cosine_similarity(huge_a.data(), huge_b.data(), HUGE_N);
    T5_ASSERT(std::fabs(cos_huge - 1.0f) < 1.0e-5f, "Massive N=131072 similarity must equal 1.0f");

    return true;
}

// 1.4 Zero Norms (0-1-N-NULL Boundary Vectors)
static bool test_extreme_zero_norms() {
    float zeros_a[HIDDEN_DIM] = {0.0f};
    float zeros_b[HIDDEN_DIM] = {0.0f};
    float norm_vec[HIDDEN_DIM];
    for (size_t i = 0; i < HIDDEN_DIM; ++i) norm_vec[i] = static_cast<float>(i + 1);

    // A. Zero Norms in Cosine Similarity
    float cos_zz = bench_compute_cosine_similarity(zeros_a, zeros_b, HIDDEN_DIM);
    T5_ASSERT(cos_zz == 0.0f, "Cosine of two zero-norm vectors must return 0.0f (no div-by-zero)");
    float cos_zn = bench_compute_cosine_similarity(zeros_a, norm_vec, HIDDEN_DIM);
    T5_ASSERT(cos_zn == 0.0f, "Cosine of zero and non-zero vectors must return 0.0f");
    float cos_nz = bench_compute_cosine_similarity(norm_vec, zeros_a, HIDDEN_DIM);
    T5_ASSERT(cos_nz == 0.0f, "Cosine of non-zero and zero vectors must return 0.0f");

    // B. Zero Norms in SQNR
    float sqnr_zz = bench_compute_sqnr_db(zeros_a, zeros_b, HIDDEN_DIM);
    T5_ASSERT(sqnr_zz == 120.0f, "SQNR of zero signal and zero noise must reach ceiling 120.0 dB");
    float sqnr_z_noise = bench_compute_sqnr_db(zeros_a, norm_vec, HIDDEN_DIM);
    T5_ASSERT(sqnr_z_noise == 0.0f, "SQNR of zero signal with non-zero noise must return 0.0 dB");

    // C. Zero Norms in Bridge
    HybridBuffer* buf = bridge_alloc_buffer(HIDDEN_DIM);
    T5_ASSERT(buf != nullptr, "bridge_alloc_buffer failed");
    std::memset(buf->hidden_cpu, 0, sizeof(buf->hidden_cpu));
    uint64_t recovered[VSA_WORDS] = {0};
    int b_rc = bridge_sync_gpu_to_vsa(buf, recovered);
    T5_ASSERT(b_rc == 0, "bridge_sync_gpu_to_vsa on all-zeros must succeed");
    // Since 0.0 >= 0.0 is TRUE in IEEE 754, all mask bits should be 1
    for (size_t w = 0; w < VSA_WORDS; ++w) {
        T5_ASSERT(recovered[w] == 0xFFFFFFFFFFFFFFFFULL, "0.0 >= 0.0 must evaluate true for all bits");
    }
    bridge_free_buffer(buf);

    // D. Zero Norms in Model Forward
    ModelContext* ctx = model_create_context(4);
    T5_ASSERT(ctx != nullptr, "model_create_context failed");
    alignas(64) float out_zero[HIDDEN_DIM];

    // GQA on zero input
    model_swap_layer_block(ctx, 0, LAYER_GQA_ATTENTION);
    int fwd_gqa = model_forward_layer(ctx, 0, zeros_a, out_zero);
    T5_ASSERT(fwd_gqa == 0, "GQA forward on zero input must succeed");
    for (size_t i = 0; i < HIDDEN_DIM; ++i) {
        T5_ASSERT(out_zero[i] == 0.0f, "GQA on zero input must produce exact zero output");
    }

    // DeltaNet on zero input: crucial test for norm_k <= 1e-6f division by zero avoidance!
    model_swap_layer_block(ctx, 0, LAYER_GATED_DELTANET);
    int fwd_delta = model_forward_layer(ctx, 0, zeros_a, out_zero);
    T5_ASSERT(fwd_delta == 0, "DeltaNet forward on zero input must succeed");
    for (size_t i = 0; i < HIDDEN_DIM; ++i) {
        T5_ASSERT(!std::isnan(out_zero[i]), "DeltaNet on zero input must NOT produce NaN");
        T5_ASSERT(out_zero[i] == 0.0f, "DeltaNet on zero input must produce exact zero output");
    }

    // VSA associative block on zero input
    model_swap_layer_block(ctx, 0, LAYER_AVX512_VSA_ASSOCIATIVE);
    int fwd_vsa = model_forward_layer(ctx, 0, zeros_a, out_zero);
    T5_ASSERT(fwd_vsa == 0, "VSA forward on zero input must succeed");
    for (size_t i = 0; i < HIDDEN_DIM; ++i) {
        T5_ASSERT(!std::isnan(out_zero[i]), "VSA on zero input must produce finite output");
    }

    model_destroy_context(ctx);
    return true;
}

// =============================================================================
// Suite 2: Robin Hood Hash Table Boundary Conditions
// =============================================================================

// 2.1 Empty Strings Handling
static bool test_rh_empty_strings() {
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    T5_ASSERT(db != nullptr, "vsa_db_create MICRO failed");

    // Insert fact with completely empty subject, relation, and object
    int ins = vsa_db_insert_fact(db, "", "", "");
    T5_ASSERT(ins == 0, "vsa_db_insert_fact with empty strings must succeed");
    T5_ASSERT(vsa_db_get_fact_count(db) == 1, "Fact count must be 1 after inserting empty fact");

    // Query empty relation
    char retrieved[64] = "initial_junk";
    int q_rc = vsa_db_query_relation(db, "", "", retrieved, sizeof(retrieved));
    T5_ASSERT(q_rc == 1, "vsa_db_query_relation for empty string fact must return 1");
    T5_ASSERT(std::strcmp(retrieved, "") == 0, "Retrieved object for empty fact must be empty string");

    // Conflict check
    int conf_same = vsa_db_check_conflict(db, "", "", "");
    T5_ASSERT(conf_same == 0, "Conflict check with identical empty object must return 0");
    int conf_diff = vsa_db_check_conflict(db, "", "", "non_empty_object");
    T5_ASSERT(conf_diff == 1, "Conflict check with different object must return 1");

    // Mutate fact
    int mut = vsa_db_mutate_fact(db, "", "", "mutated_value");
    T5_ASSERT(mut == 0, "vsa_db_mutate_fact on empty-key fact must succeed");
    char post_mut[64] = {0};
    int q_mut = vsa_db_query_relation(db, "", "", post_mut, sizeof(post_mut));
    T5_ASSERT(q_mut == 1 && std::strcmp(post_mut, "mutated_value") == 0, "Query post-mutation must return new object");

    // Delete fact
    int del = vsa_db_delete_fact(db, "", "");
    T5_ASSERT(del == 1, "vsa_db_delete_fact on empty-key fact must return 1 (deleted)");
    T5_ASSERT(vsa_db_get_fact_count(db) == 0, "Fact count must be 0 after deletion");

    // Query deleted
    char post_del[64] = {0};
    int q_del = vsa_db_query_relation(db, "", "", post_del, sizeof(post_del));
    T5_ASSERT(q_del == 0, "vsa_db_query_relation on deleted empty-key fact must return 0");

    vsa_db_destroy(db);
    return true;
}

// 2.2 Max-Length Identifiers & String Heap Inlining Boundaries
static bool test_rh_identifier_length_transitions() {
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    T5_ASSERT(db != nullptr, "vsa_db_create MICRO failed");

    // Inlining thresholds:
    // Subj & Rel inline storage: 32 bytes (<= 31 chars inline, >= 32 in string heap)
    // Obj inline storage: 64 bytes (<= 63 chars inline, >= 64 in string heap)

    std::string subj_31(31, 'S');
    std::string subj_32(32, 'S');
    std::string subj_33(33, 'S');

    std::string rel_31(31, 'R');
    std::string rel_32(32, 'R');
    std::string rel_33(33, 'R');

    std::string obj_63(63, 'O');
    std::string obj_64(64, 'O');
    std::string obj_65(65, 'O');

    // 1. Exactly at inline boundary (31, 31, 63)
    T5_ASSERT(vsa_db_insert_fact(db, subj_31.c_str(), rel_31.c_str(), obj_63.c_str()) == 0, "Insert inline boundary");
    char out_buf[256] = {0};
    T5_ASSERT(vsa_db_query_relation(db, subj_31.c_str(), rel_31.c_str(), out_buf, sizeof(out_buf)) == 1, "Query inline boundary");
    T5_ASSERT(out_buf == obj_63, "Object matches inline boundary");

    // 2. Exactly at heap overflow boundary (32, 32, 64)
    T5_ASSERT(vsa_db_insert_fact(db, subj_32.c_str(), rel_32.c_str(), obj_64.c_str()) == 0, "Insert heap boundary");
    T5_ASSERT(vsa_db_query_relation(db, subj_32.c_str(), rel_32.c_str(), out_buf, sizeof(out_buf)) == 1, "Query heap boundary");
    T5_ASSERT(out_buf == obj_64, "Object matches heap boundary");

    // 3. Beyond boundary (33, 33, 65)
    T5_ASSERT(vsa_db_insert_fact(db, subj_33.c_str(), rel_33.c_str(), obj_65.c_str()) == 0, "Insert beyond boundary");
    T5_ASSERT(vsa_db_query_relation(db, subj_33.c_str(), rel_33.c_str(), out_buf, sizeof(out_buf)) == 1, "Query beyond boundary");
    T5_ASSERT(out_buf == obj_65, "Object matches beyond boundary");

    // 4. Very long identifiers (1,024 chars)
    std::string huge_subj(1024, 'A');
    std::string huge_rel(1024, 'B');
    std::string huge_obj(2048, 'C');
    T5_ASSERT(vsa_db_insert_fact(db, huge_subj.c_str(), huge_rel.c_str(), huge_obj.c_str()) == 0, "Insert huge strings");

    std::vector<char> huge_out(4096, 0);
    T5_ASSERT(vsa_db_query_relation(db, huge_subj.c_str(), huge_rel.c_str(), huge_out.data(), huge_out.size()) == 1, "Query huge strings");
    T5_ASSERT(std::string(huge_out.data()) == huge_obj, "Huge object retrieved matches exactly");

    vsa_db_destroy(db);
    return true;
}

// 2.3 Consecutive Robin Hood Displacements
static bool test_rh_consecutive_displacements() {
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr); // 4096 slots
    T5_ASSERT(db != nullptr, "vsa_db_create MICRO failed");

    // Insert 1,500 facts with systematic prefixes to induce deep displacement chains
    const int FACT_COUNT = 1500;
    std::vector<std::pair<std::string, std::string>> inserted_pairs;
    inserted_pairs.reserve(FACT_COUNT);

    for (int i = 0; i < FACT_COUNT; ++i) {
        std::string subj = "Entity_" + std::to_string(i);
        std::string rel = "RelatesTo_" + std::to_string(i % 50);
        std::string obj = "TargetValue_" + std::to_string(i * 3 + 7);

        int rc = vsa_db_insert_fact(db, subj.c_str(), rel.c_str(), obj.c_str());
        T5_ASSERT(rc == 0, "Sequential fact insert in dense Robin Hood table must succeed");
        inserted_pairs.emplace_back(subj, rel);
    }

    T5_ASSERT(vsa_db_get_fact_count(db) == FACT_COUNT, "Fact count must match inserted count");

    // Verify 100% recall across every single displaced slot
    char query_buf[128];
    for (int i = 0; i < FACT_COUNT; ++i) {
        std::string expected_obj = "TargetValue_" + std::to_string(i * 3 + 7);
        const auto& pair = inserted_pairs[i];

        int found = vsa_db_query_relation(db, pair.first.c_str(), pair.second.c_str(), query_buf, sizeof(query_buf));
        T5_ASSERT(found == 1, "Displaced Robin Hood slot must be found");
        T5_ASSERT(std::strcmp(query_buf, expected_obj.c_str()) == 0, "Retrieved object must match expected value");
    }

    vsa_db_destroy(db);
    return true;
}

// 2.4 Backward-Shift Deletions & Clean Slot Reuse
static bool test_rh_backward_shift_deletions() {
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    T5_ASSERT(db != nullptr, "vsa_db_create MICRO failed");

    const int N_ITEMS = 600;
    for (int i = 0; i < N_ITEMS; ++i) {
        std::string s = "Node_" + std::to_string(i);
        std::string r = "Edge";
        std::string o = "Val_" + std::to_string(i);
        int ins = vsa_db_insert_fact(db, s.c_str(), r.c_str(), o.c_str());
        T5_ASSERT(ins == 0, "Insertion failed");
    }
    T5_ASSERT(vsa_db_get_fact_count(db) == N_ITEMS, "Fact count must be 600");

    // Delete every third element (simulating scattered deletions throughout probe chains)
    for (int i = 0; i < N_ITEMS; i += 3) {
        std::string s = "Node_" + std::to_string(i);
        int del = vsa_db_delete_fact(db, s.c_str(), "Edge");
        T5_ASSERT(del == 1, "Deletion must succeed");
    }
    T5_ASSERT(vsa_db_get_fact_count(db) == 400, "Fact count must be 400 after 200 deletions");

    // Verify that all 400 undeleted elements are still reachable!
    char buf[64];
    for (int i = 0; i < N_ITEMS; ++i) {
        std::string s = "Node_" + std::to_string(i);
        int q = vsa_db_query_relation(db, s.c_str(), "Edge", buf, sizeof(buf));
        if (i % 3 == 0) {
            T5_ASSERT(q == 0, "Deleted item must return 0");
        } else {
            T5_ASSERT(q == 1, "Undeleted item shifted backwards must remain 100% reachable");
            std::string expected = "Val_" + std::to_string(i);
            T5_ASSERT(std::strcmp(buf, expected.c_str()) == 0, "Shifted item value must match exactly");
        }
    }

    // Now delete all remaining elements
    for (int i = 0; i < N_ITEMS; ++i) {
        if (i % 3 != 0) {
            std::string s = "Node_" + std::to_string(i);
            vsa_db_delete_fact(db, s.c_str(), "Edge");
        }
    }
    T5_ASSERT(vsa_db_get_fact_count(db) == 0, "Fact count must be 0 after deleting all items");

    // Re-insert 100 new items to prove table is healthy and slots are reusable without tombstones
    for (int i = 0; i < 100; ++i) {
        std::string s = "FreshNode_" + std::to_string(i);
        int ins = vsa_db_insert_fact(db, s.c_str(), "FreshEdge", "FreshVal");
        T5_ASSERT(ins == 0, "Re-inserting into cleared Robin Hood table must succeed");
    }
    T5_ASSERT(vsa_db_get_fact_count(db) == 100, "Fact count must reflect re-inserted facts");

    vsa_db_destroy(db);
    return true;
}

// 2.5 Raw Vector Operations & Multi-Threaded OpenMP Cosine Search
static bool test_rh_vector_table_boundaries() {
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    T5_ASSERT(db != nullptr, "vsa_db_create MICRO failed");

    // 1. All-zeros vector
    uint64_t v_zero[8] = {0};
    T5_ASSERT(vsa_db_insert_vector(db, 0x1000, v_zero) == 0, "Insert zero vector");

    // 2. All-ones vector
    uint64_t v_ones[8];
    std::memset(v_ones, 0xFF, sizeof(v_ones));
    T5_ASSERT(vsa_db_insert_vector(db, 0x2000, v_ones) == 0, "Insert ones vector");

    // 3. Alternating bit vector
    uint64_t v_alt[8];
    for (int i = 0; i < 8; ++i) v_alt[i] = 0xAAAAAAAAAAAAAAAAULL;
    T5_ASSERT(vsa_db_insert_vector(db, 0x3000, v_alt) == 0, "Insert alt vector");

    T5_ASSERT(vsa_db_get_vector_count(db) == 3, "Vector count must be 3");

    // Query exact match
    uint64_t best_vec[8] = {0};
    float sim = 0.0f;
    int q_exact = vsa_db_query_vector(db, v_alt, best_vec, &sim);
    T5_ASSERT(q_exact == 1, "Vector query must succeed");
    T5_ASSERT(std::fabs(sim - 1.0f) < 1.0e-5f, "Exact vector similarity must be 1.0");
    T5_ASSERT(std::memcmp(best_vec, v_alt, sizeof(v_alt)) == 0, "Best vector must match alt vector");

    // Query exact opposite: under the 0.40 resonance gate this is a miss. The
    // closest stored vectors score 0.0 (v_zero/v_ones) and -1.0 (v_alt), i.e.
    // no resonance; returning "nearest anyway" was the pre-gate behaviour.
    uint64_t v_opp[8];
    for (int i = 0; i < 8; ++i) v_opp[i] = ~v_alt[i]; // 0x5555555555555555ULL
    int q_opp = vsa_db_query_vector(db, v_opp, best_vec, &sim);
    T5_ASSERT(q_opp == 0, "Non-resonant opposite vector must be rejected");
    T5_ASSERT(sim == 0.0f, "Rejected probe leaves similarity at 0.0");

    // Insert 500 pseudorandom vectors and query with a resonant probe: one bit
    // flipped from the last stored vector (similarity 510/512 = 0.996). A random
    // probe scores ~0 against all 503 entries and must be rejected by the gate.
    std::mt19937_64 rng(42);
    uint64_t rv_last[8] = {0};
    for (int i = 0; i < 500; ++i) {
        uint64_t rv[8];
        for (int w = 0; w < 8; ++w) rv[w] = rng();
        vsa_db_insert_vector(db, 0x4000 + i, rv);
        if (i == 499) std::memcpy(rv_last, rv, sizeof(rv));
    }
    T5_ASSERT(vsa_db_get_vector_count(db) == 503, "Vector count must be 503");

    uint64_t probe_vec[8];
    std::memcpy(probe_vec, rv_last, sizeof(probe_vec));
    probe_vec[0] ^= 1ULL;
    int q_stress = vsa_db_query_vector(db, probe_vec, best_vec, &sim);
    T5_ASSERT(q_stress == 1, "Resonant probe over 503 vectors must succeed");
    T5_ASSERT(sim > 0.99f, "One-bit-flipped probe similarity must be > 0.99");

    vsa_db_destroy(db);
    return true;
}

// 2.6 Persistence & Deep Cloning Integrity
static bool test_rh_persistence_and_cloning() {
    const char* path1 = "build\\test_t5_persist1.bin";
    const char* path2 = "build\\test_t5_persist2.bin";
    DeleteFileA(path1);
    DeleteFileA(path2);

    // 1. Create file-mapped DB
    VsaDbHandle* db1 = vsa_db_create(VSA_CAP_MICRO, path1);
    T5_ASSERT(db1 != nullptr, "Failed to create persistent DB1");
    vsa_db_insert_fact(db1, "Arch", "Engine", "NativeCUDA");
    vsa_db_insert_fact(db1, "Memory", "Mode", "AVX512_VSA");
    uint64_t test_v[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    vsa_db_insert_vector(db1, 0x9999, test_v);

    vsa_db_flush(db1);
    vsa_db_destroy(db1);

    // 2. Re-open existing persistent DB
    VsaDbHandle* db1_reopened = vsa_db_create(VSA_CAP_MICRO, path1);
    T5_ASSERT(db1_reopened != nullptr, "Failed to re-open persistent DB1");
    T5_ASSERT(vsa_db_get_fact_count(db1_reopened) == 2, "Fact count must be 2 after re-opening");
    // The two facts own one binding vector each (F22 unification), plus the raw
    // vector inserted above: 2 + 1 = 3 occupied slots in the shared vector table.
    T5_ASSERT(vsa_db_get_vector_count(db1_reopened) == 3, "Vector count must be 3 after re-opening");

    // The raw vector itself must also survive the reopen.
    uint64_t probe_v[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    probe_v[0] ^= 1ULL;
    uint64_t got_v[8] = {0};
    float got_sim = 0.0f;
    T5_ASSERT(vsa_db_query_vector(db1_reopened, probe_v, got_v, &got_sim) == 1 &&
                  got_sim > 0.99f,
              "Persisted raw vector must be queryable after reopen");

    char qbuf[64] = {0};
    int q_rc = vsa_db_query_relation(db1_reopened, "Arch", "Engine", qbuf, sizeof(qbuf));
    T5_ASSERT(q_rc == 1 && std::strcmp(qbuf, "NativeCUDA") == 0, "Persisted fact must survive reopen");

    // 3. Clone DB1 into DB2
    VsaDbHandle* db2 = nullptr;
    int clone_rc = vsa_db_clone(db1_reopened, &db2, path2);
    T5_ASSERT(clone_rc == 0 && db2 != nullptr, "vsa_db_clone must succeed");
    T5_ASSERT(vsa_db_get_fact_count(db2) == 2, "Cloned DB fact count must match source");

    // 4. Mutate DB2, verify DB1 is unchanged
    vsa_db_mutate_fact(db2, "Arch", "Engine", "ModifiedClone");
    vsa_db_query_relation(db2, "Arch", "Engine", qbuf, sizeof(qbuf));
    T5_ASSERT(std::strcmp(qbuf, "ModifiedClone") == 0, "DB2 fact must be mutated");

    vsa_db_query_relation(db1_reopened, "Arch", "Engine", qbuf, sizeof(qbuf));
    T5_ASSERT(std::strcmp(qbuf, "NativeCUDA") == 0, "DB1 source fact must remain isolated and untouched");

    vsa_db_destroy(db1_reopened);
    vsa_db_destroy(db2);

    DeleteFileA(path1);
    DeleteFileA(path2);
    return true;
}

// =============================================================================
// Suite 3: TT16 Plasticity Updates Under Repeated Extreme Perturbations
// =============================================================================

// 3.1 High-Frequency Continuous Plasticity Churn
static bool test_tt16_plasticity_high_frequency_churn() {
    ModelContext* ctx = model_create_context(4);
    T5_ASSERT(ctx != nullptr, "model_create_context failed");

    const size_t sz = MODEL_TT16_CORE_SIZE; // 16,384 floats
    std::vector<float> u_delta(sz, 0.001f);
    std::vector<float> v_delta(sz, 0.002f);

    const int ITERATIONS = 1000;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < ITERATIONS; ++i) {
        int rc = model_update_tt16_residual(ctx, 0, u_delta.data(), v_delta.data());
        T5_ASSERT(rc == 0, "model_update_tt16_residual failed");
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double avg_us = total_us / ITERATIONS;

    std::cout << "    [Perf Telemetry] 1,000 TT16 Updates: Total " << (total_us / 1000.0)
              << " ms, Avg " << avg_us << " us / update (Budget: <25 us)\n";
    T5_ASSERT(avg_us < 25.0, "Average TT16 update time must be strictly under 25 microseconds");

    // Numerical accumulation accuracy: initial was 0.0f, added 0.001f x 1000 = 1.0f
    float expected_u = 1.0f;
    float expected_v = 2.0f;
    float actual_u = ctx->layers[0].tt_u[0];
    float actual_v = ctx->layers[0].tt_v[0];

    T5_ASSERT(std::fabs(actual_u - expected_u) < 1.0e-3f, "TT16 U accumulation must equal 1.0f");
    T5_ASSERT(std::fabs(actual_v - expected_v) < 1.0e-3f, "TT16 V accumulation must equal 2.0f");

    model_destroy_context(ctx);
    return true;
}

// 3.2 Alternating Oscillating Perturbations
static bool test_tt16_plasticity_alternating_oscillations() {
    ModelContext* ctx = model_create_context(2);
    T5_ASSERT(ctx != nullptr, "model_create_context failed");

    const size_t sz = MODEL_TT16_CORE_SIZE;
    std::vector<float> pos_u(sz, 0.05f);
    std::vector<float> pos_v(sz, 0.05f);
    std::vector<float> neg_u(sz, -0.05f);
    std::vector<float> neg_v(sz, -0.05f);

    const int ROUNDS = 5000;
    for (int r = 0; r < ROUNDS; ++r) {
        model_update_tt16_residual(ctx, 0, pos_u.data(), pos_v.data());
        model_update_tt16_residual(ctx, 0, neg_u.data(), neg_v.data());
    }

    // After 5,000 paired positive and negative updates, weights should return to ~0.0f
    for (size_t i = 0; i < sz; i += 512) {
        T5_ASSERT(std::fabs(ctx->layers[0].tt_u[i]) < 1.0e-4f, "Oscillating updates must cancel out cleanly");
        T5_ASSERT(std::fabs(ctx->layers[0].tt_v[i]) < 1.0e-4f, "Oscillating updates must cancel out cleanly");
    }

    model_destroy_context(ctx);
    return true;
}

// 3.3 Dyadic Accumulation Oracle Comparison
static bool test_tt16_dyadic_accumulation_oracle() {
    ModelContext* ctx = model_create_context(2);
    T5_ASSERT(ctx != nullptr, "model_create_context failed");

    // Seed deterministic known weights into TT16 core
    ModelLayerConfig* lyr = &ctx->layers[0];
    const uint32_t R = lyr->R; // 16
    const uint32_t D = lyr->N; // 1024

    for (size_t i = 0; i < MODEL_TT16_CORE_SIZE; ++i) {
        lyr->tt_u[i] = static_cast<float>(std::sin(static_cast<double>(i) * 0.01)) * 0.1f;
        lyr->tt_v[i] = static_cast<float>(std::cos(static_cast<double>(i) * 0.01)) * 0.1f;
    }

    alignas(64) float in_x[HIDDEN_DIM];
    alignas(64) float inout_y_simd[HIDDEN_DIM];
    alignas(64) float inout_y_oracle[HIDDEN_DIM];

    for (size_t i = 0; i < HIDDEN_DIM; ++i) {
        in_x[i] = static_cast<float>(std::sin(static_cast<double>(i) * 0.03));
        inout_y_simd[i] = 1.0f;
        inout_y_oracle[i] = 1.0f;
    }

    // Run AVX-512 SIMD implementation
    int rc = model_apply_tt16_residual(ctx, 0, in_x, inout_y_simd);
    T5_ASSERT(rc == 0, "model_apply_tt16_residual failed");

    // Run independent double-precision scalar reference oracle
    for (uint32_t r = 0; r < R; ++r) {
        double dot_zr = 0.0;
        const float* vr = lyr->tt_v + (size_t)r * D;
        for (uint32_t j = 0; j < D; ++j) {
            dot_zr += static_cast<double>(vr[j]) * static_cast<double>(in_x[j]);
        }
        const float* ur = lyr->tt_u + (size_t)r * D;
        for (uint32_t i = 0; i < D; ++i) {
            inout_y_oracle[i] += static_cast<float>(dot_zr * static_cast<double>(ur[i]));
        }
    }

    // Compare AVX-512 SIMD vs Scalar Oracle
    for (size_t i = 0; i < HIDDEN_DIM; ++i) {
        float diff = std::fabs(inout_y_simd[i] - inout_y_oracle[i]);
        T5_ASSERT(diff < 1.0e-4f, "SIMD TT16 residual dyadic projection must match scalar oracle");
    }

    model_destroy_context(ctx);
    return true;
}

// =============================================================================
// Suite 4: Layer Block Swapping Under Zero-Norm Inputs & Abnormal Conditions
// =============================================================================

// 4.1 Rapid Block Swapping Churn
static bool test_layer_block_rapid_swapping() {
    ModelContext* ctx = model_create_context(4);
    T5_ASSERT(ctx != nullptr, "model_create_context failed");

    // Rapidly toggle layer 0 across all 3 block types
    const int SWAPS = 1500;
    for (int i = 0; i < SWAPS; ++i) {
        LayerType target = (LayerType)(i % 3);
        int rc = model_swap_layer_block(ctx, 0, target);
        T5_ASSERT(rc == 0, "model_swap_layer_block failed");
        T5_ASSERT(model_get_layer_type(ctx, 0) == target, "Layer type must reflect target");
    }

    model_destroy_context(ctx);
    return true;
}

// 4.2 DeltaNet State Retention, Decay, and Reset Invariants
static bool test_layer_block_deltanet_state_and_reset() {
    ModelContext* ctx = model_create_context(2);
    T5_ASSERT(ctx != nullptr, "model_create_context failed");

    model_swap_layer_block(ctx, 0, LAYER_GATED_DELTANET);

    alignas(64) float in_vec[HIDDEN_DIM];
    alignas(64) float out_vec[HIDDEN_DIM];
    for (size_t i = 0; i < HIDDEN_DIM; ++i) in_vec[i] = static_cast<float>(i + 1) * 0.01f;

    // Step 1: Forward pass to build up internal state S_h
    model_forward_layer(ctx, 0, in_vec, out_vec);

    ModelLayerConfig* lyr = &ctx->layers[0];
    const size_t state_sz = MODEL_DELTANET_HEADS * MODEL_DELTANET_HEAD_DIM * MODEL_DELTANET_HEAD_DIM; // 131,072 floats
    bool has_nonzero = false;
    for (size_t i = 0; i < state_sz; ++i) {
        if (lyr->deltanet_state[i] != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    T5_ASSERT(has_nonzero, "DeltaNet internal state S_h must be mutated post-forward");

    // Step 2: Reset single layer state
    int r_reset = model_reset_layer_state(ctx, 0);
    T5_ASSERT(r_reset == 0, "model_reset_layer_state must succeed");
    for (size_t i = 0; i < state_sz; ++i) {
        T5_ASSERT(lyr->deltanet_state[i] == 0.0f, "DeltaNet state must be zeroed post-reset");
    }

    // Step 3: Forward again, then reset all states
    model_forward_layer(ctx, 0, in_vec, out_vec);
    int r_reset_all = model_reset_all_states(ctx);
    T5_ASSERT(r_reset_all == 0, "model_reset_all_states must succeed");
    for (size_t i = 0; i < state_sz; ++i) {
        T5_ASSERT(lyr->deltanet_state[i] == 0.0f, "All DeltaNet states must be zeroed post reset_all");
    }

    model_destroy_context(ctx);
    return true;
}

// 4.3 Out-of-Bounds & Argument Guardrail Hardening
static bool test_layer_block_invalid_arguments() {
    ModelContext* ctx = model_create_context(4);
    T5_ASSERT(ctx != nullptr, "model_create_context failed");

    alignas(64) float dummy[HIDDEN_DIM] = {1.0f};

    // A. NULL Context Rejections
    T5_ASSERT(model_set_layer_precision(nullptr, 0, QUANT_4BIT) == -1, "NULL ctx in set_precision must return -1");
    T5_ASSERT(model_swap_layer_block(nullptr, 0, LAYER_GQA_ATTENTION) == -1, "NULL ctx in swap_block must return -1");
    T5_ASSERT(model_update_tt16_residual(nullptr, 0, dummy, dummy) == -1, "NULL ctx in tt16_update must return -1");
    T5_ASSERT(model_apply_tt16_residual(nullptr, 0, dummy, dummy) == -1, "NULL ctx in tt16_apply must return -1");
    T5_ASSERT(model_forward_layer(nullptr, 0, dummy, dummy) == -1, "NULL ctx in forward must return -1");
    T5_ASSERT(model_reset_layer_state(nullptr, 0) == -1, "NULL ctx in reset_state must return -1");

    // B. Out-of-Bounds Layer Index Rejections (ctx has 4 layers: 0..3)
    T5_ASSERT(model_set_layer_precision(ctx, 4, QUANT_4BIT) == -2, "Layer 4 out of bounds in set_precision must return -2");
    T5_ASSERT(model_set_layer_precision(ctx, 9999, QUANT_4BIT) == -2, "Layer 9999 out of bounds in set_precision must return -2");
    T5_ASSERT(model_swap_layer_block(ctx, 4, LAYER_GQA_ATTENTION) == -2, "Layer 4 out of bounds in swap_block must return -2");
    T5_ASSERT(model_swap_layer_block(ctx, 9999, LAYER_GQA_ATTENTION) == -2, "Layer 9999 out of bounds in swap_block must return -2");
    T5_ASSERT(model_update_tt16_residual(ctx, 4, dummy, dummy) == -2, "Layer 4 out of bounds in tt16_update must return -2");
    T5_ASSERT(model_apply_tt16_residual(ctx, 4, dummy, dummy) == -2, "Layer 4 out of bounds in tt16_apply must return -2");
    T5_ASSERT(model_forward_layer(ctx, 4, dummy, dummy) == -2, "Layer 4 out of bounds in forward must return -2");
    T5_ASSERT(model_reset_layer_state(ctx, 4) == -2, "Layer 4 out of bounds in reset must return -2");

    // C. Invalid Enum Values
    T5_ASSERT(model_set_layer_precision(ctx, 0, (QuantPrecision)999) == -3, "Invalid precision enum must return -3");
    T5_ASSERT(model_set_layer_precision(ctx, 0, (QuantPrecision)-1) == -3, "Negative precision enum must return -3");
    T5_ASSERT(model_swap_layer_block(ctx, 0, (LayerType)999) == -3, "Invalid layer type enum must return -3");
    T5_ASSERT(model_swap_layer_block(ctx, 0, (LayerType)-1) == -3, "Negative layer type enum must return -3");

    // D. NULL Buffer Pointers
    T5_ASSERT(model_update_tt16_residual(ctx, 0, nullptr, dummy) == -1, "NULL u_delta must return -1");
    T5_ASSERT(model_update_tt16_residual(ctx, 0, dummy, nullptr) == -1, "NULL v_delta must return -1");
    T5_ASSERT(model_apply_tt16_residual(ctx, 0, nullptr, dummy) == -1, "NULL in_x must return -1");
    T5_ASSERT(model_apply_tt16_residual(ctx, 0, dummy, nullptr) == -1, "NULL inout_y must return -1");
    T5_ASSERT(model_forward_layer(ctx, 0, nullptr, dummy) == -1, "NULL in_vec must return -1");
    T5_ASSERT(model_forward_layer(ctx, 0, dummy, nullptr) == -1, "NULL out_vec must return -1");

    model_destroy_context(ctx);
    return true;
}

// =============================================================================
// Suite 5: Resource & Memory Leak Invariant Verification
// =============================================================================

static bool test_resource_leak_assertion() {
    DWORD initial_handles = get_current_handle_count();
    size_t initial_ram_kb = get_current_ram_kb();

    std::cout << "    [Leak Guard Initial] Handles: " << initial_handles << ", RAM: " << (initial_ram_kb / 1024) << " MB\n";

    // Churn through 50,000 operations across all subsystems
    for (int iter = 0; iter < 1000; ++iter) {
        // 1. Bridge
        HybridBuffer* buf = bridge_alloc_buffer(HIDDEN_DIM);
        if (buf) {
            uint64_t v[VSA_WORDS] = {0};
            bridge_sync_vsa_to_gpu(buf, v, 1.0f);
            bridge_sync_gpu_to_vsa(buf, v);
            bridge_free_buffer(buf);
        }

        // 2. VsaMemoryDatabase
        VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
        if (db) {
            vsa_db_insert_fact(db, "Subj", "Rel", "Obj");
            char out[32];
            vsa_db_query_relation(db, "Subj", "Rel", out, sizeof(out));
            vsa_db_delete_fact(db, "Subj", "Rel");
            vsa_db_destroy(db);
        }

        // 3. ModelContext
        ModelContext* ctx = model_create_context(2);
        if (ctx) {
            model_set_layer_precision(ctx, 0, QUANT_2BIT);
            model_swap_layer_block(ctx, 0, LAYER_GATED_DELTANET);
            model_destroy_context(ctx);
        }
    }

    DWORD final_handles = get_current_handle_count();
    size_t final_ram_kb = get_current_ram_kb();

    long handle_delta = (long)final_handles - (long)initial_handles;
    long ram_delta_kb = (long)final_ram_kb - (long)initial_ram_kb;

    std::cout << "    [Leak Guard Final] Handles: " << final_handles << " (delta: " << handle_delta
              << "), RAM: " << (final_ram_kb / 1024) << " MB (delta: " << (ram_delta_kb / 1024) << " MB)\n";

    T5_ASSERT(std::abs(handle_delta) <= 2, "Handle count must remain constant (zero handle leaks)");
    T5_ASSERT(ram_delta_kb < 10 * 1024, "RAM working set growth must be < 10 MB (zero unbounded growth)");

    return true;
}

// =============================================================================
// Main Execution Entrypoint
// =============================================================================

int main() {
    std::cout << "========================================================================================\n";
    std::cout << "  TIER 5 ADVERSARIAL STRESS & NUMERICAL COVERAGE HARDENING SUITE\n";
    std::cout << "  Pure Native C++20 / AVX-512 / sm_86 / Win32 White-Box Verification\n";
    std::cout << "========================================================================================\n\n";

    // Suite 1: Extreme Numerical Regimes
    std::cout << ">>> 1. Extreme Numerical Regimes <<<\n";
    RUN_T5_TEST(test_extreme_subnormals);
    RUN_T5_TEST(test_extreme_nans_and_infinities);
    RUN_T5_TEST(test_extreme_massive_tensors);
    RUN_T5_TEST(test_extreme_zero_norms);

    // Suite 2: Robin Hood Hash Table Boundary Conditions
    std::cout << "\n>>> 2. Robin Hood Hash Table Boundary Conditions <<<\n";
    RUN_T5_TEST(test_rh_empty_strings);
    RUN_T5_TEST(test_rh_identifier_length_transitions);
    RUN_T5_TEST(test_rh_consecutive_displacements);
    RUN_T5_TEST(test_rh_backward_shift_deletions);
    RUN_T5_TEST(test_rh_vector_table_boundaries);
    RUN_T5_TEST(test_rh_persistence_and_cloning);

    // Suite 3: TT16 Plasticity Updates Under Repeated Extreme Perturbations
    std::cout << "\n>>> 3. TT16 Plasticity Updates Under Repeated Extreme Perturbations <<<\n";
    RUN_T5_TEST(test_tt16_plasticity_high_frequency_churn);
    RUN_T5_TEST(test_tt16_plasticity_alternating_oscillations);
    RUN_T5_TEST(test_tt16_dyadic_accumulation_oracle);

    // Suite 4: Layer Block Swapping Under Zero-Norm Inputs & Abnormal Conditions
    std::cout << "\n>>> 4. Layer Block Swapping Under Zero-Norm Inputs & Guardrails <<<\n";
    RUN_T5_TEST(test_layer_block_rapid_swapping);
    RUN_T5_TEST(test_layer_block_deltanet_state_and_reset);
    RUN_T5_TEST(test_layer_block_invalid_arguments);

    // Suite 5: Resource & Memory Leak Invariant Verification
    std::cout << "\n>>> 5. Resource & Memory Leak Invariants <<<\n";
    RUN_T5_TEST(test_resource_leak_assertion);

    std::cout << "\n========================================================================================\n";
    std::cout << "  TIER 5 SUMMARY: " << g_tier5_passed << " / " << g_tier5_run << " PASSED ("
              << (g_tier5_failed == 0 ? "100%" : "FAILURES DETECTED") << ")\n";
    std::cout << "========================================================================================\n\n";

    return (g_tier5_failed == 0) ? 0 : 1;
}

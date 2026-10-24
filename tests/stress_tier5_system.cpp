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
#include <chrono>
#include <thread>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <cassert>
#include <malloc.h>
#include <limits>
#include <random>

#pragma comment(lib, "psapi.lib")

// =============================================================================
// Challenger Final 2: System-Level Concurrency & Lifecycle Stress Harness
// Milestone M_FINAL: Tier 5 Adversarial Coverage & Hardware Envelope Verification
// =============================================================================

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define STRESS_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  [FAIL] Line " << __LINE__ << ": " << msg << " (" #cond ")\n"; \
            return false; \
        } \
    } while (0)

#define RUN_TIER5_SUITE(fn) \
    do { \
        g_tests_run++; \
        std::cout << "\n===============================================================================\n"; \
        std::cout << "[RUN TIER 5 SUITE] " << #fn << "...\n"; \
        auto t0 = std::chrono::high_resolution_clock::now(); \
        bool ok = fn(); \
        auto t1 = std::chrono::high_resolution_clock::now(); \
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); \
        if (ok) { \
            g_tests_passed++; \
            std::cout << "  [PASS] " << #fn << " (" << std::fixed << std::setprecision(2) << elapsed_ms << " ms)\n"; \
        } else { \
            g_tests_failed++; \
            std::cerr << "  [FAILED] " << #fn << " (" << std::fixed << std::setprecision(2) << elapsed_ms << " ms)\n"; \
        } \
    } while (0)

static size_t get_ram_working_set_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024 * 1024);
    }
    return 0;
}

static size_t get_ram_private_bytes() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.PrivateUsage;
    }
    return 0;
}

static DWORD get_handle_count() {
    DWORD count = 0;
    GetProcessHandleCount(GetCurrentProcess(), &count);
    return count;
}

static uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

// =============================================================================
// Suite 1: Hardware Envelope Baseline Verification
// =============================================================================
static bool suite1_hardware_envelope_baseline() {
    std::cout << ">>> Suite 1: Verifying Workstation Hardware Baseline Envelope <<<\n";

    size_t ram_mb = get_ram_working_set_mb();
    size_t vram_mb = bench_get_gpu_vram_mb();
    DWORD handles = get_handle_count();

    std::cout << "  Host RAM Working Set: " << ram_mb << " MB (Ceiling: 32768 MB)\n";
    std::cout << "  GPU VRAM Allocation:  " << vram_mb << " MB (Ceiling: 4096 MB)\n";
    std::cout << "  Active Handle Count:  " << handles << "\n";

    STRESS_ASSERT(ram_mb <= MAX_WORKSTATION_RAM_MB, "Initial Host RAM exceeds 32GB ceiling");
    STRESS_ASSERT(vram_mb <= MAX_WORKSTATION_VRAM_MB, "Initial GPU VRAM exceeds 4096MB ceiling");
    STRESS_ASSERT(bench_assert_hardware_bounds(MAX_WORKSTATION_RAM_MB, MAX_WORKSTATION_VRAM_MB) == 0,
                  "bench_assert_hardware_bounds failed on baseline");

    return true;
}

// =============================================================================
// Suite 2: 10,000 Rapid Interleaved Lifecycle & Subsystem Churn Cycles
// VSA DB mutations + Hot-swaps + Precision switches (2/3/4/24/34) +
// TT16 plasticity + Layer block swapping + Hybrid GPU DMA
// =============================================================================
static bool suite2_rapid_10k_interleaved_churn() {
    std::cout << ">>> Suite 2: 10,000 Rapid Interleaved Cycles Stress Test <<<\n";

    const int TOTAL_CYCLES = 10000;
    const int CHECKPOINT_INTERVAL = 1000;

    // Warmup cycle (initialize CRT allocators, OpenMP threadpools, and CUDA driver context)
    {
        HybridBuffer* warm_buf = bridge_alloc_buffer(HIDDEN_DIM);
        VsaDbHandle* warm_db = vsa_db_create(VSA_CAP_MEDIUM, nullptr);
        ModelContext* warm_ctx = model_create_context(MODEL_DEFAULT_NUM_LAYERS);
        vsa_db_insert_fact(warm_db, "Warm", "is", "Ready");
        model_set_layer_precision(warm_ctx, 0, QUANT_2BIT);

        alignas(64) float warm_in[HIDDEN_DIM] = {0.1f};
        alignas(64) float warm_out[HIDDEN_DIM] = {0.0f};
        model_swap_layer_block(warm_ctx, 0, LAYER_GQA_ATTENTION);
        model_forward_layer(warm_ctx, 0, warm_in, warm_out);
        model_swap_layer_block(warm_ctx, 0, LAYER_GATED_DELTANET);
        model_forward_layer(warm_ctx, 0, warm_in, warm_out);
        model_swap_layer_block(warm_ctx, 0, LAYER_AVX512_VSA_ASSOCIATIVE);
        model_forward_layer(warm_ctx, 0, warm_in, warm_out);

        uint64_t dummy_v[8] = {0x12345678ULL, 1, 2, 3, 4, 5, 6, 7};
        vsa_db_insert_vector(warm_db, 0x1234, dummy_v);
        uint64_t best_v[8];
        float sim = 0.0f;
        vsa_db_query_vector(warm_db, dummy_v, best_v, &sim);

        bridge_sync_vsa_to_gpu(warm_buf, warm_buf->vsa_vec, 1.0f);
        uint64_t dummy_ret[8];
        bridge_sync_gpu_to_vsa(warm_buf, dummy_ret);

        model_destroy_context(warm_ctx);
        vsa_db_destroy(warm_db);
        bridge_free_buffer(warm_buf);
    }

    DWORD initial_handles = get_handle_count();
    size_t initial_private_bytes = get_ram_private_bytes();
    size_t initial_ram_mb = get_ram_working_set_mb();
    size_t initial_vram_mb = bench_get_gpu_vram_mb();

    std::cout << "  [Baseline] Handles: " << initial_handles
              << " | Private RAM: " << (initial_private_bytes / 1024) << " KB"
              << " | WorkingSet: " << initial_ram_mb << " MB"
              << " | VRAM: " << initial_vram_mb << " MB\n";

    // Setup persistent and dynamic instances
    HybridBuffer* hybrid_buf = bridge_alloc_buffer(HIDDEN_DIM);
    STRESS_ASSERT(hybrid_buf != nullptr, "bridge_alloc_buffer failed");

    VsaDbHandle* db_active = vsa_db_create(VSA_CAP_MEDIUM, nullptr);
    VsaDbHandle* db_standby = vsa_db_create(VSA_CAP_MEDIUM, nullptr);
    STRESS_ASSERT(db_active != nullptr && db_standby != nullptr, "vsa_db_create failed");

    ModelContext* model_ctx = model_create_context(MODEL_DEFAULT_NUM_LAYERS);
    STRESS_ASSERT(model_ctx != nullptr, "model_create_context failed");

    const QuantPrecision precisions[] = {
        QUANT_2BIT, QUANT_3BIT, QUANT_4BIT, QUANT_DELTA_24, QUANT_DELTA_34
    };
    const LayerType layer_types[] = {
        LAYER_GQA_ATTENTION, LAYER_GATED_DELTANET, LAYER_AVX512_VSA_ASSOCIATIVE
    };

    uint64_t rng_state = 0x853C49E6748FEA9BULL;
    alignas(64) float in_vec[HIDDEN_DIM];
    alignas(64) float out_vec[HIDDEN_DIM];
    for (size_t d = 0; d < HIDDEN_DIM; ++d) {
        in_vec[d] = 0.01f * (float)(d % 37) - 0.18f;
    }

    std::vector<float> u_delta(MODEL_TT16_CORE_SIZE);
    std::vector<float> v_delta(MODEL_TT16_CORE_SIZE);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int cycle = 0; cycle < TOTAL_CYCLES; ++cycle) {
        uint32_t layer_idx = (uint32_t)(cycle % MODEL_DEFAULT_NUM_LAYERS);

        // 1. Dynamic VSA DB Mutations
        char subj[32], rel[32], obj[32];
        std::snprintf(subj, sizeof(subj), "Entity_%d", cycle % 500);
        std::snprintf(rel, sizeof(rel), "rel_%d", cycle % 16);
        std::snprintf(obj, sizeof(obj), "Value_%d", cycle % 1000);

        int rc_fact = vsa_db_insert_fact(db_active, subj, rel, obj);
        STRESS_ASSERT(rc_fact == 0, "vsa_db_insert_fact failed");

        if ((cycle % 3) == 0) {
            std::snprintf(obj, sizeof(obj), "MutatedVal_%d", cycle % 1000);
            int rc_mut = vsa_db_mutate_fact(db_active, subj, rel, obj);
            STRESS_ASSERT(rc_mut == 0, "vsa_db_mutate_fact failed");
        }

        if ((cycle % 7) == 0) {
            char retrieved_obj[64] = {0};
            int rc_q = vsa_db_query_relation(db_active, subj, rel, retrieved_obj, sizeof(retrieved_obj));
            STRESS_ASSERT(rc_q == 1, "vsa_db_query_relation failed to retrieve existing fact");
        }

        if ((cycle % 11) == 0 && cycle > 50) {
            char del_subj[32], del_rel[32];
            std::snprintf(del_subj, sizeof(del_subj), "Entity_%d", (cycle - 30) % 500);
            std::snprintf(del_rel, sizeof(del_rel), "rel_%d", (cycle - 30) % 16);
            vsa_db_delete_fact(db_active, del_subj, del_rel);
        }

        // Vector insert & in-place mutation
        uint64_t v512[8];
        for (int w = 0; w < 8; ++w) {
            v512[w] = xorshift64(&rng_state);
        }
        uint64_t v_key = 0x100000ULL + (cycle % 4000);
        int rc_vec_ins = vsa_db_insert_vector(db_active, v_key, v512);
        STRESS_ASSERT(rc_vec_ins == 0, "vsa_db_insert_vector failed");

        if ((cycle % 13) == 0) {
            uint64_t best_vec[8];
            float sim = 0.0f;
            int rc_vquery = vsa_db_query_vector(db_active, v512, best_vec, &sim);
            STRESS_ASSERT(rc_vquery == 1, "vsa_db_query_vector failed to query vector");
            STRESS_ASSERT(sim >= 0.99f, "vsa_db_query_vector similarity lower than expected for identical vector");
        }

        // 2. Dynamic Hot-Swapping & DB Cloning
        if ((cycle % 100) == 0) {
            // Swap active and standby databases
            VsaDbHandle* tmp = db_active;
            db_active = db_standby;
            db_standby = tmp;
        }

        if ((cycle % 250) == 0) {
            // Test rapid clone, flush, and destroy lifecycle
            VsaDbHandle* cloned_db = nullptr;
            int rc_clone = vsa_db_clone(db_active, &cloned_db, nullptr);
            STRESS_ASSERT(rc_clone == 0 && cloned_db != nullptr, "vsa_db_clone failed");
            vsa_db_insert_fact(cloned_db, "CloneTest", "is", "Temporary");
            vsa_db_flush(cloned_db);
            vsa_db_destroy(cloned_db);
        }

        // 3. Dynamic Precision Switches (2/3/4/24/34)
        QuantPrecision target_prec = precisions[cycle % 5];
        int rc_prec = model_set_layer_precision(model_ctx, layer_idx, target_prec);
        STRESS_ASSERT(rc_prec == 0, "model_set_layer_precision failed");
        STRESS_ASSERT(model_get_layer_precision(model_ctx, layer_idx) == target_prec,
                      "model_get_layer_precision mismatch");

        // 4. Online Low-Rank Plasticity Updates (TT16 Continuous Cores)
        float adapt_scale = 0.001f * (float)((cycle % 17) - 8);
        for (size_t k = 0; k < MODEL_TT16_CORE_SIZE; ++k) {
            u_delta[k] = adapt_scale * (float)(k % 11);
            v_delta[k] = -adapt_scale * (float)(k % 13);
        }
        int rc_tt16 = model_update_tt16_residual(model_ctx, layer_idx, u_delta.data(), v_delta.data());
        STRESS_ASSERT(rc_tt16 == 0, "model_update_tt16_residual failed");

        // 5. Layer Block Polymorphic Swapping & Forward Inference
        LayerType target_block = layer_types[cycle % 3];
        int rc_swap = model_swap_layer_block(model_ctx, layer_idx, target_block);
        STRESS_ASSERT(rc_swap == 0, "model_swap_layer_block failed");
        STRESS_ASSERT(model_get_layer_type(model_ctx, layer_idx) == target_block,
                      "model_get_layer_type mismatch");

        // Execute layer forward computation
        int rc_fwd = model_forward_layer(model_ctx, layer_idx, in_vec, out_vec);
        STRESS_ASSERT(rc_fwd == 0, "model_forward_layer returned error");
        STRESS_ASSERT(!std::isnan(out_vec[0]) && !std::isinf(out_vec[0]),
                      "model_forward_layer produced NaN or Inf");

        // 6. Hybrid GPU DMA Synchronization
        for (int w = 0; w < 8; ++w) {
            hybrid_buf->vsa_vec[w] = v512[w];
        }
        int rc_dma_to = bridge_sync_vsa_to_gpu(hybrid_buf, hybrid_buf->vsa_vec, 1.0f);
        STRESS_ASSERT(rc_dma_to == 0, "bridge_sync_vsa_to_gpu failed");

        uint64_t retrieved_vec[8] = {0};
        int rc_dma_from = bridge_sync_gpu_to_vsa(hybrid_buf, retrieved_vec);
        STRESS_ASSERT(rc_dma_from == 0, "bridge_sync_gpu_to_vsa failed");

        // Checkpoints
        if ((cycle + 1) % CHECKPOINT_INTERVAL == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double cur_ms = std::chrono::duration<double, std::milli>(now - start_time).count();
            double avg_us_per_cycle = (cur_ms * 1000.0) / (cycle + 1);

            size_t cur_ram_mb = get_ram_working_set_mb();
            size_t cur_vram_mb = bench_get_gpu_vram_mb();
            DWORD cur_handles = get_handle_count();

            std::cout << "  [Cycle " << std::setw(5) << (cycle + 1) << " / " << TOTAL_CYCLES << "]"
                      << " Avg Cycle: " << std::fixed << std::setprecision(1) << avg_us_per_cycle << " us"
                      << " | RAM: " << cur_ram_mb << " MB"
                      << " | VRAM: " << cur_vram_mb << " MB"
                      << " | Handles: " << cur_handles << "\n";

            STRESS_ASSERT(cur_ram_mb <= MAX_WORKSTATION_RAM_MB, "RAM exceeded 32GB during 10K cycle stress");
            STRESS_ASSERT(cur_vram_mb <= MAX_WORKSTATION_VRAM_MB, "VRAM exceeded 4096MB during 10K cycle stress");
        }
    }

    // Teardown allocated objects
    model_destroy_context(model_ctx);
    vsa_db_destroy(db_active);
    vsa_db_destroy(db_standby);
    bridge_free_buffer(hybrid_buf);

    DWORD final_handles = get_handle_count();
    size_t final_private_bytes = get_ram_private_bytes();
    size_t final_ram_mb = get_ram_working_set_mb();
    size_t final_vram_mb = bench_get_gpu_vram_mb();

    int64_t handle_delta = (int64_t)final_handles - (int64_t)initial_handles;
    int64_t private_delta_kb = ((int64_t)final_private_bytes - (int64_t)initial_private_bytes) / 1024;

    std::cout << "\n  [Final 10,000 Cycle Telemetry]\n";
    std::cout << "    Handle delta:        " << handle_delta << " (Initial: " << initial_handles << ", Final: " << final_handles << ")\n";
    std::cout << "    Private RAM delta:   " << private_delta_kb << " KB\n";
    std::cout << "    Working Set:         " << final_ram_mb << " MB (Ceiling: 32768 MB)\n";
    std::cout << "    GPU VRAM:            " << final_vram_mb << " MB (Ceiling: 4096 MB)\n";

    // Strict leak and envelope assertions
    STRESS_ASSERT(handle_delta <= 2, "Process handles leaked across 10,000 cycles!");
    STRESS_ASSERT(private_delta_kb < 5120, "Unbounded memory growth detected (>5MB delta across 10K cycles)!");
    STRESS_ASSERT(final_ram_mb <= MAX_WORKSTATION_RAM_MB, "Final RAM exceeds 32GB ceiling");
    STRESS_ASSERT(final_vram_mb <= MAX_WORKSTATION_VRAM_MB, "Final VRAM exceeds 4096MB ceiling");

    std::cout << "  [PASS] 10,000 rapid cycles completed with ZERO handle leaks and bounded memory.\n";
    return true;
}

// =============================================================================
// Suite 3: Multi-Threaded Concurrency Stress (4 Threads x 1,000 Cycles)
// =============================================================================
static bool suite3_multithreaded_concurrency_stress() {
    std::cout << ">>> Suite 3: Multi-Threaded Concurrency & Instance Isolation Stress <<<\n";

    const int NUM_THREADS = 4;
    const int CYCLES_PER_THREAD = 1000;

    auto run_round = [&](int round, std::atomic<bool>& ok) {
        std::vector<std::thread> workers;
        workers.reserve(NUM_THREADS);
        for (int t = 0; t < NUM_THREADS; ++t) {
            workers.emplace_back([t, round, CYCLES_PER_THREAD, &ok]() {
            uint64_t rng = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)t << 24) ^ ((uint64_t)round << 40);
            HybridBuffer* buf = bridge_alloc_buffer(HIDDEN_DIM);
            VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
            ModelContext* ctx = model_create_context(MODEL_DEFAULT_NUM_LAYERS);

            if (!buf || !db || !ctx) {
                ok = false;
                if (buf) bridge_free_buffer(buf);
                if (db) vsa_db_destroy(db);
                if (ctx) model_destroy_context(ctx);
                return;
            }

            alignas(64) float in_v[HIDDEN_DIM];
            alignas(64) float out_v[HIDDEN_DIM];
            for (int i = 0; i < HIDDEN_DIM; ++i) in_v[i] = 0.05f * (float)(i % 23);

            std::vector<float> u(MODEL_TT16_CORE_SIZE, 0.005f);
            std::vector<float> v(MODEL_TT16_CORE_SIZE, -0.005f);

            for (int i = 0; i < CYCLES_PER_THREAD; ++i) {
                uint32_t layer = (uint32_t)(i % MODEL_DEFAULT_NUM_LAYERS);

                // Fact insertion
                char s[32], r[32], o[32];
                std::snprintf(s, sizeof(s), "Th%d_Subj_%d", t, i % 100);
                std::snprintf(r, sizeof(r), "Th%d_Rel_%d", t, i % 10);
                std::snprintf(o, sizeof(o), "Th%d_Obj_%d", t, i % 200);
                if (vsa_db_insert_fact(db, s, r, o) != 0) {
                    ok = false;
                    break;
                }

                // Vector insert
                uint64_t v512[8];
                for (int w = 0; w < 8; ++w) v512[w] = xorshift64(&rng);
                vsa_db_insert_vector(db, 0x50000ULL + i, v512);

                // Precision switch
                QuantPrecision prec = (QuantPrecision)(2 + (i % 3));
                if (model_set_layer_precision(ctx, layer, prec) != 0) {
                    ok = false;
                    break;
                }

                // TT16 plasticity update
                if (model_update_tt16_residual(ctx, layer, u.data(), v.data()) != 0) {
                    ok = false;
                    break;
                }

                // Block swap and forward
                LayerType ltype = (LayerType)(i % 3);
                model_swap_layer_block(ctx, layer, ltype);
                if (model_forward_layer(ctx, layer, in_v, out_v) != 0) {
                    ok = false;
                    break;
                }

                // DMA sync
                for (int w = 0; w < 8; ++w) buf->vsa_vec[w] = v512[w];
                bridge_sync_vsa_to_gpu(buf, buf->vsa_vec, 1.0f);
                uint64_t ret[8];
                bridge_sync_gpu_to_vsa(buf, ret);
            }

            model_destroy_context(ctx);
            vsa_db_destroy(db);
            bridge_free_buffer(buf);
            });
        }

        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }
    };

    _heapmin();
    DWORD handles_before = get_handle_count();
    size_t private_before = get_ram_private_bytes();

    std::atomic<bool> round1_ok{true};
    run_round(1, round1_ok);
    _heapmin();
    size_t private_mid = get_ram_private_bytes();

    std::atomic<bool> round2_ok{true};
    run_round(2, round2_ok);
    _heapmin();
    DWORD handles_after = get_handle_count();
    size_t private_after = get_ram_private_bytes();

    STRESS_ASSERT(round1_ok.load() && round2_ok.load(),
                  "One or more worker threads failed during concurrency stress");

    // One-time CRT arena retention is allowed; repeated growth is not. A real
    // per-cycle leak adds the same amount in every round, while the per-thread
    // arena pages are released (or stop growing) after the first round. The
    // earlier single-round threshold was flaky: the same binary measured
    // 2768-5124 KB of private-byte delta depending on allocator state, while
    // suite 2 (10,000 single-threaded create/destroy cycles) retained 676 KB.
    // With this two-round protocol the host measures ~716 KB then ~428 KB.
    int64_t handle_delta = (int64_t)handles_after - (int64_t)handles_before;
    int64_t mem_delta1_kb = ((int64_t)private_mid - (int64_t)private_before) / 1024;
    int64_t mem_delta2_kb = ((int64_t)private_after - (int64_t)private_mid) / 1024;
    int64_t mem_delta_kb = ((int64_t)private_after - (int64_t)private_before) / 1024;

    std::cout << "  Multi-threaded completion: 2 rounds x 4 threads x 1,000 cycles = 8,000 total cycles.\n";
    std::cout << "  Round 1 delta: " << mem_delta1_kb << " KB, round 2 delta: "
              << mem_delta2_kb << " KB, total: " << mem_delta_kb << " KB\n";
    std::cout << "  Handle delta: " << handle_delta << "\n";

    STRESS_ASSERT(handle_delta <= 2, "Handle leak observed after multi-threaded concurrency stress");
    STRESS_ASSERT(mem_delta1_kb < 8192, "First-round retention exceeds the measured arena bound");
    STRESS_ASSERT(mem_delta2_kb < 1024, "Memory grew again in the repeated round: per-cycle leak");
    STRESS_ASSERT(mem_delta_kb < 9216, "Total private-byte growth exceeds the measured retention bound");

    return true;
}

// =============================================================================
// Suite 4: Adversarial Edge Cases & Boundary Falsification Matrix
// =============================================================================
static bool suite4_boundary_falsification() {
    std::cout << ">>> Suite 4: Adversarial Boundaries & Boundary Robustness <<<\n";

    // 1. Null pointer safety
    STRESS_ASSERT(bridge_sync_vsa_to_gpu(nullptr, nullptr, 1.0f) == -1, "bridge_sync_vsa_to_gpu null guard failed");
    STRESS_ASSERT(bridge_sync_gpu_to_vsa(nullptr, nullptr) == -1, "bridge_sync_gpu_to_vsa null guard failed");
    STRESS_ASSERT(vsa_db_insert_fact(nullptr, "a", "b", "c") == -1, "vsa_db_insert_fact null guard failed");
    STRESS_ASSERT(vsa_db_delete_fact(nullptr, "a", "b") == -1, "vsa_db_delete_fact null guard failed");
    STRESS_ASSERT(vsa_db_mutate_fact(nullptr, "a", "b", "c") == -1, "vsa_db_mutate_fact null guard failed");
    STRESS_ASSERT(model_set_layer_precision(nullptr, 0, QUANT_2BIT) == -1, "model_set_layer_precision null guard failed");
    STRESS_ASSERT(model_update_tt16_residual(nullptr, 0, nullptr, nullptr) == -1, "model_update_tt16_residual null guard failed");
    STRESS_ASSERT(model_swap_layer_block(nullptr, 0, LAYER_GQA_ATTENTION) == -1, "model_swap_layer_block null guard failed");
    STRESS_ASSERT(model_forward_layer(nullptr, 0, nullptr, nullptr) == -1, "model_forward_layer null guard failed");

    // 2. Out-of-bounds layer indices
    ModelContext* ctx = model_create_context(MODEL_DEFAULT_NUM_LAYERS);
    STRESS_ASSERT(ctx != nullptr, "Context creation failed");
    STRESS_ASSERT(model_set_layer_precision(ctx, 99999, QUANT_2BIT) == -2, "Layer OOB precision guard failed");
    STRESS_ASSERT(model_update_tt16_residual(ctx, 99999, (const float*)0x1000, (const float*)0x1000) == -2, "Layer OOB TT16 guard failed");
    STRESS_ASSERT(model_swap_layer_block(ctx, 99999, LAYER_GQA_ATTENTION) == -2, "Layer OOB block swap guard failed");
    STRESS_ASSERT(model_forward_layer(ctx, 99999, (const float*)0x1000, (float*)0x1000) == -2, "Layer OOB forward guard failed");

    // 3. Invalid enums
    STRESS_ASSERT(model_set_layer_precision(ctx, 0, (QuantPrecision)999) == -3, "Invalid QuantPrecision guard failed");
    STRESS_ASSERT(model_swap_layer_block(ctx, 0, (LayerType)999) == -3, "Invalid LayerType guard failed");

    // 4. Extreme values in TT16
    std::vector<float> extreme_u(MODEL_TT16_CORE_SIZE, 1e9f);
    std::vector<float> extreme_v(MODEL_TT16_CORE_SIZE, -1e9f);
    STRESS_ASSERT(model_update_tt16_residual(ctx, 0, extreme_u.data(), extreme_v.data()) == 0, "Extreme TT16 update failed");

    alignas(64) float in_v[HIDDEN_DIM];
    alignas(64) float out_v[HIDDEN_DIM];
    for (int i = 0; i < HIDDEN_DIM; ++i) in_v[i] = 1.0f;
    STRESS_ASSERT(model_forward_layer(ctx, 0, in_v, out_v) == 0, "Forward pass with extreme weights crashed");

    model_destroy_context(ctx);
    std::cout << "  [PASS] All boundary checks, null guards, OOB indices, and extreme values handled safely.\n";
    return true;
}

// =============================================================================
// Suite 5: Full 28-Layer End-to-End Inference & Profiler Telemetry
// =============================================================================
static bool suite5_e2e_inference_telemetry() {
    std::cout << ">>> Suite 5: Full 28-Layer Forward Step Profiling & Bench Telemetry <<<\n";

    ModelContext* ctx = model_create_context(MODEL_DEFAULT_NUM_LAYERS);
    STRESS_ASSERT(ctx != nullptr, "Context creation failed");

    // Heterogeneous layer distribution
    for (uint32_t l = 0; l < MODEL_DEFAULT_NUM_LAYERS; ++l) {
        if (l % 3 == 0) model_swap_layer_block(ctx, l, LAYER_GQA_ATTENTION);
        else if (l % 3 == 1) model_swap_layer_block(ctx, l, LAYER_GATED_DELTANET);
        else model_swap_layer_block(ctx, l, LAYER_AVX512_VSA_ASSOCIATIVE);
    }

    BenchmarkMetrics metrics;
    std::memset(&metrics, 0, sizeof(metrics));

    // Profile 50 steps
    double total_us = 0.0;
    const int STEPS = 50;
    for (int s = 0; s < STEPS; ++s) {
        int rc = bench_run_step_profile(ctx, &metrics);
        STRESS_ASSERT(rc == 0, "bench_run_step_profile failed");
        total_us += metrics.latency_us;
    }
    double avg_latency_us = total_us / STEPS;
    double throughput_tok_s = (avg_latency_us > 0.0) ? (1000000.0 / avg_latency_us) : 0.0;

    std::cout << "  Average 28-layer Forward Step Latency: " << avg_latency_us << " us (" << (avg_latency_us / 1000.0) << " ms)\n";
    std::cout << "  Effective Token Rate:                  " << throughput_tok_s << " tok/s\n";
    std::cout << "  Host RAM Working Set:                  " << metrics.host_ram_working_set_mb << " MB\n";
    std::cout << "  GPU VRAM Allocation:                   " << metrics.gpu_vram_used_mb << " MB\n";
    std::cout << "  Cosine Similarity:                     "
              << (metrics.accuracy_available ? std::to_string(metrics.cosine_similarity) : "N/A (no quantized reference)")
              << "\n";
    std::cout << "  Quantization SQNR:                     "
              << (metrics.accuracy_available ? std::to_string(metrics.sqnr_db) + " dB" : "N/A (no quantized reference)")
              << "\n";

    STRESS_ASSERT(avg_latency_us < 15000.0, "Step latency exceeds 15 ms specification");
    STRESS_ASSERT(throughput_tok_s > 100.0, "Throughput below 100 tok/s specification");
    STRESS_ASSERT(metrics.host_ram_working_set_mb <= MAX_WORKSTATION_RAM_MB, "RAM exceeds 32GB ceiling");
    STRESS_ASSERT(metrics.gpu_vram_used_mb <= MAX_WORKSTATION_VRAM_MB, "VRAM exceeds 4096MB ceiling");
    // The profiled model path is plain FP32 (no quantized weight container), so
    // bench_run_step_profile reports accuracy as unavailable. The old assertions
    // checked a number that was manufactured by adding noise to the profiler's
    // own output; asserting N/A is the honest contract.
    STRESS_ASSERT(!metrics.accuracy_available, "Profiler claims accuracy without a reference");
    STRESS_ASSERT(std::isnan(metrics.cosine_similarity), "Cosine is not NaN (fabricated value)");
    STRESS_ASSERT(std::isnan(metrics.sqnr_db), "SQNR is not NaN (fabricated value)");

    model_destroy_context(ctx);
    std::cout << "  [PASS] Full 28-layer pipeline telemetry conforms strictly to PROJECT.md.\n";
    return true;
}

// =============================================================================
// Main Test Runner
// =============================================================================
int main() {
    std::cout << "===============================================================================\n";
    std::cout << "  TIER 5 SYSTEM CONCURRENCY & LIFECYCLE STRESS HARNESS\n";
    std::cout << "  Target: Complete Integrated Laboratory (Bridge, VSA DB, ModelMod, BenchRunner)\n";
    std::cout << "===============================================================================\n";

    RUN_TIER5_SUITE(suite1_hardware_envelope_baseline);
    RUN_TIER5_SUITE(suite2_rapid_10k_interleaved_churn);
    RUN_TIER5_SUITE(suite3_multithreaded_concurrency_stress);
    RUN_TIER5_SUITE(suite4_boundary_falsification);
    RUN_TIER5_SUITE(suite5_e2e_inference_telemetry);

    std::cout << "\n===============================================================================\n";
    std::cout << "  TIER 5 STRESS SUMMARY\n";
    std::cout << "===============================================================================\n";
    std::cout << "  Suites Executed: " << g_tests_run << "\n";
    std::cout << "  Suites Passed:   " << g_tests_passed << "\n";
    std::cout << "  Suites Failed:   " << g_tests_failed << "\n";

    if (g_tests_failed == 0) {
        std::cout << "\n[RESULT] ALL TIER 5 SYSTEM STRESS TESTS PASSED! ZERO REGRESSIONS DETECTED.\n";
        return 0;
    } else {
        std::cerr << "\n[RESULT] TIER 5 STRESS FAILED: " << g_tests_failed << " suite(s) failed.\n";
        return 1;
    }
}

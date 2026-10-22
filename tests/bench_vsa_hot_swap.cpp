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
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <thread>
#include <random>
#include <iostream>
#include <sstream>
#include <iomanip>

#pragma comment(lib, "psapi.lib")

// =============================================================================
// Benchmark Mini-Framework & High-Precision Timing
// =============================================================================

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

#define RUN_BENCH(fn) \
    do { \
        g_bench_run++; \
        std::printf("\n=================================================================\n"); \
        std::printf("[BENCHMARK START] %s\n", #fn); \
        std::printf("=================================================================\n"); \
        double _t0 = qpc_time_us(); \
        bool _res = fn(); \
        double _t_elapsed_ms = (qpc_time_us() - _t0) / 1000.0; \
        if (_res) { \
            g_bench_passed++; \
            std::printf(">>> [PASS] %s (Elapsed: %.2f ms)\n", #fn, _t_elapsed_ms); \
        } else { \
            g_bench_failed++; \
            std::printf(">>> [FAIL] %s (Elapsed: %.2f ms)\n", #fn, _t_elapsed_ms); \
        } \
    } while (0)

static double qpc_time_us() {
    static LARGE_INTEGER freq;
    static bool init = false;
    if (!init) {
        QueryPerformanceFrequency(&freq);
        init = true;
    }
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (static_cast<double>(t.QuadPart) * 1000000.0) / static_cast<double>(freq.QuadPart);
}

static size_t get_process_ram_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024 * 1024);
    }
    return 0;
}

static size_t get_process_ram_bytes() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
    return 0;
}

// PRNG helper
static uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *state = x;
}

// =============================================================================
// TEST 1: LIVE HOT-SWAPPING UNDER CONTINUOUS QUERY LOAD (10,000 SWAPS)
// Assert: 0 crashes, 0 data corruptions, 0 memory leaks
// =============================================================================

static bool bench_live_hot_swapping_10k() {
    std::printf("[*] Setting up 8 isolated database instances...\n");
    const int NUM_DBS = 8;
    VsaDbHandle* dbs[NUM_DBS];

    struct DbSignature {
        std::string id_str;
        std::string token_str;
        uint64_t raw_vector_seed;
        uint64_t raw_vec[VSA_WORDS];
    };
    DbSignature signatures[NUM_DBS];

    for (int i = 0; i < NUM_DBS; ++i) {
        dbs[i] = vsa_db_create(VSA_CAP_MICRO, nullptr);
        BENCH_ASSERT(dbs[i] != nullptr, "Failed to create database instance");

        signatures[i].id_str = "DB_INSTANCE_" + std::to_string(i);
        signatures[i].token_str = "SIGNATURE_HASH_0x" + std::to_string(0xABCD0000ULL + i * 0x1111ULL);
        signatures[i].raw_vector_seed = 0xDEADBEEF0000ULL + i * 0x3333ULL;

        // Insert identification facts
        BENCH_ASSERT(vsa_db_insert_fact(dbs[i], "SYS_INSTANCE", "id", signatures[i].id_str.c_str()) == 0, "Insert fact failed");
        BENCH_ASSERT(vsa_db_insert_fact(dbs[i], "SYS_SIGNATURE", "token", signatures[i].token_str.c_str()) == 0, "Insert fact failed");

        // Insert domain facts specific to this instance
        for (int f = 0; f < 30; ++f) {
            std::string subj = "DomainNode_" + std::to_string(f);
            std::string obj = "Val_DB" + std::to_string(i) + "_Fact" + std::to_string(f);
            BENCH_ASSERT(vsa_db_insert_fact(dbs[i], subj.c_str(), "prop", obj.c_str()) == 0, "Insert domain fact failed");
        }

        // Insert signature vector
        uint64_t seed = signatures[i].raw_vector_seed;
        for (int w = 0; w < VSA_WORDS; ++w) {
            signatures[i].raw_vec[w] = xorshift64(&seed);
        }
        BENCH_ASSERT(vsa_db_insert_vector(dbs[i], 0x777000ULL + i, signatures[i].raw_vec) == 0, "Insert vector failed");
    }

    std::printf("[+] 8 instances initialized with unique identities and facts.\n");

    // Atomic active database handle
    std::atomic<VsaDbHandle*> active_db(dbs[0]);
    std::atomic<int> active_idx(0);

    std::atomic<bool> stop_flag(false);
    std::atomic<int> ready_readers(0);
    std::atomic<bool> start_flag(false);
    std::atomic<uint64_t> total_queries(0);
    std::atomic<uint64_t> successful_queries(0);
    std::atomic<uint64_t> data_corruptions(0);
    std::atomic<uint64_t> query_errors(0);

    const int NUM_READERS = 4;
    std::vector<std::thread> reader_threads;

    size_t ram_before_bytes = get_process_ram_bytes();
    size_t ram_before_mb = ram_before_bytes / (1024 * 1024);
    std::printf("[*] Baseline RAM working set before hot-swap test: %zu MB (%zu bytes)\n", ram_before_mb, ram_before_bytes);

    // Launch reader threads
    for (int t = 0; t < NUM_READERS; ++t) {
        reader_threads.emplace_back([&, t]() {
            char out_id[128];
            char out_tok[128];
            char out_domain[128];
            uint64_t local_state = 0x123456789ULL + t * 0x999ULL;

            ready_readers.fetch_add(1, std::memory_order_release);
            while (!start_flag.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            while (!stop_flag.load(std::memory_order_relaxed)) {
                VsaDbHandle* cur = active_db.load(std::memory_order_acquire);
                if (!cur) continue;

                // 1. Query SYS_INSTANCE id
                total_queries.fetch_add(1, std::memory_order_relaxed);
                int res_id = vsa_db_query_relation(cur, "SYS_INSTANCE", "id", out_id, sizeof(out_id));
                if (res_id != 1) {
                    query_errors.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                // Parse index from "DB_INSTANCE_<k>"
                int detected_idx = -1;
                if (std::sscanf(out_id, "DB_INSTANCE_%d", &detected_idx) != 1 || detected_idx < 0 || detected_idx >= NUM_DBS) {
                    data_corruptions.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                successful_queries.fetch_add(1, std::memory_order_relaxed);

                // 2. Query SYS_SIGNATURE token
                total_queries.fetch_add(1, std::memory_order_relaxed);
                int res_tok = vsa_db_query_relation(cur, "SYS_SIGNATURE", "token", out_tok, sizeof(out_tok));
                if (res_tok != 1) {
                    query_errors.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                // Verify token matches detected instance
                if (signatures[detected_idx].token_str != out_tok) {
                    data_corruptions.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                successful_queries.fetch_add(1, std::memory_order_relaxed);

                // 3. Query domain fact
                total_queries.fetch_add(1, std::memory_order_relaxed);
                int fact_idx = static_cast<int>(xorshift64(&local_state) % 30);
                std::string s_subj = "DomainNode_" + std::to_string(fact_idx);
                std::string expected_obj = "Val_DB" + std::to_string(detected_idx) + "_Fact" + std::to_string(fact_idx);
                int res_dom = vsa_db_query_relation(cur, s_subj.c_str(), "prop", out_domain, sizeof(out_domain));
                if (res_dom != 1 || expected_obj != out_domain) {
                    data_corruptions.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                successful_queries.fetch_add(1, std::memory_order_relaxed);

                // 4 & 5. Conflict consistency checks
                total_queries.fetch_add(2, std::memory_order_relaxed);
                int conf_self = vsa_db_check_conflict(cur, "SYS_INSTANCE", "id", signatures[detected_idx].id_str.c_str());
                int conf_wrong = vsa_db_check_conflict(cur, "SYS_INSTANCE", "id", "WRONG_ID_999");
                if (conf_self != 0 || conf_wrong != 1) {
                    data_corruptions.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                successful_queries.fetch_add(2, std::memory_order_relaxed);

                // 6. Vector argmax scan (periodic to maintain high query frequency)
                if (t == 3 || (xorshift64(&local_state) % 10 == 0)) {
                    total_queries.fetch_add(1, std::memory_order_relaxed);
                    uint64_t ret_vec[VSA_WORDS];
                    float sim = 0.0f;
                    int res_v = vsa_db_query_vector(cur, signatures[detected_idx].raw_vec, ret_vec, &sim);
                    if (res_v != 1 || sim < 0.999f) {
                        data_corruptions.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                    successful_queries.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    // Wait until all readers are spawned and ready
    while (ready_readers.load(std::memory_order_acquire) < NUM_READERS) {
        std::this_thread::yield();
    }
    start_flag.store(true, std::memory_order_release);
    Sleep(10); // Allow readers to spin up continuous query load

    std::printf("[*] Starting 10,000 atomic hot-swaps under concurrent reader query load...\n");
    const int TOTAL_SWAPS = 10000;
    double total_swap_latency_us = 0.0;
    double max_swap_latency_us = 0.0;
    double min_swap_latency_us = 1e9;

    double t_start_swaps = qpc_time_us();

    for (int s = 0; s < TOTAL_SWAPS; ++s) {
        int next_idx = (s + 1) % NUM_DBS;

        double t0 = qpc_time_us();
        active_db.store(dbs[next_idx], std::memory_order_release);
        double dt = qpc_time_us() - t0;

        total_swap_latency_us += dt;
        if (dt > max_swap_latency_us) max_swap_latency_us = dt;
        if (dt < min_swap_latency_us) min_swap_latency_us = dt;

        active_idx.store(next_idx, std::memory_order_release);

        // Periodically mutate an inactive instance to simulate online concurrent mutation
        if (s % 500 == 0) {
            int inactive = (next_idx + 4) % NUM_DBS;
            vsa_db_mutate_fact(dbs[inactive], "BackgroundMutation", "status", std::to_string(s).c_str());
        }

        // Pace hot-swap to allow concurrent readers to query actively during all 10,000 transitions
        double t_target = qpc_time_us() + 15.0;
        while (qpc_time_us() < t_target) {
            _mm_pause();
        }
    }

    double t_end_swaps = qpc_time_us();
    double total_time_ms = (t_end_swaps - t_start_swaps) / 1000.0;

    // Signal readers to stop and join
    stop_flag.store(true, std::memory_order_release);
    for (auto& th : reader_threads) {
        if (th.joinable()) th.join();
    }

    size_t ram_after_bytes = get_process_ram_bytes();
    size_t ram_after_mb = ram_after_bytes / (1024 * 1024);
    int64_t ram_growth_bytes = static_cast<int64_t>(ram_after_bytes) - static_cast<int64_t>(ram_before_bytes);
    double ram_growth_mb = static_cast<double>(ram_growth_bytes) / (1024.0 * 1024.0);

    double avg_swap_us = total_swap_latency_us / static_cast<double>(TOTAL_SWAPS);

    std::printf("\n--- HOT-SWAPPING 10,000 BENCHMARK METRICS ---\n");
    std::printf("  Total Hot-Swaps Executed:    %d\n", TOTAL_SWAPS);
    std::printf("  Total Elapsed Time:          %.2f ms\n", total_time_ms);
    std::printf("  Avg Hot-Swap Latency:        %.4f us\n", avg_swap_us);
    std::printf("  Min Hot-Swap Latency:        %.4f us\n", min_swap_latency_us);
    std::printf("  Max Hot-Swap Latency:        %.4f us\n", max_swap_latency_us);
    std::printf("  Total Concurrent Queries:    %llu\n", total_queries.load());
    std::printf("  Successful Queries:          %llu\n", successful_queries.load());
    std::printf("  Query Errors:                %llu\n", query_errors.load());
    std::printf("  Data Corruptions Detected:   %llu\n", data_corruptions.load());
    std::printf("  RAM Working Set Delta:       %.2f MB (%lld bytes)\n", ram_growth_mb, ram_growth_bytes);
    std::printf("---------------------------------------------\n");

    // Rigorous assertions
    BENCH_ASSERT(total_queries.load() >= 10000, "Insufficient concurrent query load executed during hot-swaps");
    BENCH_ASSERT(query_errors.load() == 0, "Query errors observed during hot-swapping");
    BENCH_ASSERT(data_corruptions.load() == 0, "CRITICAL: Data corruption detected during live hot-swapping!");
    BENCH_ASSERT(avg_swap_us < 10.0, "Average hot-swap latency exceeded 10.0 us");
    BENCH_ASSERT(ram_growth_mb < 5.0, "Excessive memory growth detected during 10,000 hot-swaps");

    // Dynamic Lifecycle Hot-Swapping (Create -> Populate -> Hot-Swap -> Destroy 1,000 times)
    std::printf("\n[*] Running Rapid Dynamic Lifecycle Hot-Swapping (1,000 fresh creations & destructions)...\n");
    size_t lifecycle_ram_before = get_process_ram_bytes();
    for (int cycle = 0; cycle < 1000; ++cycle) {
        VsaDbHandle* fresh = vsa_db_create(VSA_CAP_MICRO, nullptr);
        BENCH_ASSERT(fresh != nullptr, "Fresh DB create failed");
        vsa_db_insert_fact(fresh, "Ephemeral", "cycle", std::to_string(cycle).c_str());

        // Hot swap in
        active_db.store(fresh, std::memory_order_release);

        // Query active
        char buf[64];
        int q_res = vsa_db_query_relation(active_db.load(std::memory_order_acquire), "Ephemeral", "cycle", buf, sizeof(buf));
        BENCH_ASSERT(q_res == 1, "Ephemeral query failed");
        BENCH_ASSERT(std::string(buf) == std::to_string(cycle), "Ephemeral content mismatch");

        // Swap back to stable base
        active_db.store(dbs[0], std::memory_order_release);

        // Safely destroy ephemeral DB
        vsa_db_destroy(fresh);
    }
    size_t lifecycle_ram_after = get_process_ram_bytes();
    double lifecycle_ram_growth_mb = static_cast<double>(static_cast<int64_t>(lifecycle_ram_after) - static_cast<int64_t>(lifecycle_ram_before)) / (1024.0 * 1024.0);
    std::printf("  Lifecycle 1,000 Creation/Destruction RAM Delta: %.2f MB\n", lifecycle_ram_growth_mb);
    BENCH_ASSERT(lifecycle_ram_growth_mb < 4.0, "Memory leak detected during 1,000 dynamic lifecycle create/destroy cycles!");

    // Clean up persistent handles
    for (int i = 0; i < NUM_DBS; ++i) {
        vsa_db_destroy(dbs[i]);
    }

    return true;
}

// =============================================================================
// TEST 2: CONFLICT DETECTION ACCURACY ACROSS DATABASES (100% ACCURATE)
// Assert: 100% precision, 100% recall, 0 false positives, 0 false negatives
// =============================================================================

static bool bench_cross_db_conflict_detection_100pct() {
    std::printf("[*] Setting up Cross-Database Conflict Matrix...\n");

    VsaDbHandle* db_truth = vsa_db_create(VSA_CAP_MICRO, nullptr);
    VsaDbHandle* db_candidate = vsa_db_create(VSA_CAP_MICRO, nullptr);
    BENCH_ASSERT(db_truth != nullptr && db_candidate != nullptr, "Create DB failed");

    const int N_AGREED = 300;
    const int N_CONTRADICT = 300;
    const int N_DISJOINT = 200;

    struct FactSpec {
        std::string subj;
        std::string rel;
        std::string obj_truth;
        std::string obj_candidate;
        int expected_conflict; // 0 = agreement / no contradiction, 1 = conflict
    };

    std::vector<FactSpec> dataset;
    dataset.reserve(N_AGREED + N_CONTRADICT + N_DISJOINT);

    // 1. Agreed facts (Identical across both DBs)
    for (int i = 0; i < N_AGREED; ++i) {
        FactSpec f;
        f.subj = "CommonEntity_" + std::to_string(i);
        f.rel = "status_code";
        f.obj_truth = "STATE_CODE_OK_" + std::to_string(i * 7);
        f.obj_candidate = f.obj_truth;
        f.expected_conflict = 0;
        dataset.push_back(f);

        vsa_db_insert_fact(db_truth, f.subj.c_str(), f.rel.c_str(), f.obj_truth.c_str());
        vsa_db_insert_fact(db_candidate, f.subj.c_str(), f.rel.c_str(), f.obj_candidate.c_str());
    }

    // 2. Contradictory facts (Same subj & rel, contradictory obj)
    for (int i = 0; i < N_CONTRADICT; ++i) {
        FactSpec f;
        f.subj = "ContestedEntity_" + std::to_string(i);
        f.rel = "permission_level";
        f.obj_truth = "ALLOW_LEVEL_" + std::to_string(i);
        f.obj_candidate = "DENY_RESTRICTED_" + std::to_string(i);
        f.expected_conflict = 1; // Conflict!
        dataset.push_back(f);

        vsa_db_insert_fact(db_truth, f.subj.c_str(), f.rel.c_str(), f.obj_truth.c_str());
        vsa_db_insert_fact(db_candidate, f.subj.c_str(), f.rel.c_str(), f.obj_candidate.c_str());
    }

    // 3. Disjoint / Novel facts (Exist only in candidate or truth)
    for (int i = 0; i < N_DISJOINT; ++i) {
        FactSpec f;
        f.subj = "NovelCandidateEntity_" + std::to_string(i);
        f.rel = "feature_flag";
        f.obj_truth = ""; // Not in truth DB!
        f.obj_candidate = "ENABLED_FLAG_" + std::to_string(i);
        f.expected_conflict = 0; // Unrecorded in truth, so no contradiction
        dataset.push_back(f);

        vsa_db_insert_fact(db_candidate, f.subj.c_str(), f.rel.c_str(), f.obj_candidate.c_str());
    }

    // 4. Boundary cases (Empty strings, special chars, long strings > 64 chars)
    // Long string contradiction
    std::string long_truth(500, 'A');
    std::string long_cand(500, 'B');
    FactSpec f_long;
    f_long.subj = "LongPayloadKey";
    f_long.rel = "large_val";
    f_long.obj_truth = long_truth;
    f_long.obj_candidate = long_cand;
    f_long.expected_conflict = 1;
    dataset.push_back(f_long);
    vsa_db_insert_fact(db_truth, f_long.subj.c_str(), f_long.rel.c_str(), f_long.obj_truth.c_str());
    vsa_db_insert_fact(db_candidate, f_long.subj.c_str(), f_long.rel.c_str(), f_long.obj_candidate.c_str());

    // Long string agreement
    FactSpec f_long_agree;
    f_long_agree.subj = "LongPayloadAgreeKey";
    f_long_agree.rel = "large_val";
    f_long_agree.obj_truth = long_truth;
    f_long_agree.obj_candidate = long_truth;
    f_long_agree.expected_conflict = 0;
    dataset.push_back(f_long_agree);
    vsa_db_insert_fact(db_truth, f_long_agree.subj.c_str(), f_long_agree.rel.c_str(), f_long_agree.obj_truth.c_str());
    vsa_db_insert_fact(db_candidate, f_long_agree.subj.c_str(), f_long_agree.rel.c_str(), f_long_agree.obj_candidate.c_str());

    // Empty string contradiction
    FactSpec f_empty;
    f_empty.subj = "EmptyKey";
    f_empty.rel = "empty_rel";
    f_empty.obj_truth = "";
    f_empty.obj_candidate = "NON_EMPTY";
    f_empty.expected_conflict = 1;
    dataset.push_back(f_empty);
    vsa_db_insert_fact(db_truth, f_empty.subj.c_str(), f_empty.rel.c_str(), f_empty.obj_truth.c_str());
    vsa_db_insert_fact(db_candidate, f_empty.subj.c_str(), f_empty.rel.c_str(), f_empty.obj_candidate.c_str());

    std::printf("[*] Evaluating conflict detection across %zu test cases...\n", dataset.size());

    int tp = 0; // True Positive (Conflict expected=1, actual=1)
    int tn = 0; // True Negative (Conflict expected=0, actual=0)
    int fp = 0; // False Positive (Conflict expected=0, actual=1)
    int fn = 0; // False Negative (Conflict expected=1, actual=0)

    double t0 = qpc_time_us();

    for (const auto& item : dataset) {
        int actual_conflict = vsa_db_check_conflict(db_truth, item.subj.c_str(), item.rel.c_str(), item.obj_candidate.c_str());

        if (actual_conflict == 1 && item.expected_conflict == 1) {
            tp++;
        } else if (actual_conflict == 0 && item.expected_conflict == 0) {
            tn++;
        } else if (actual_conflict == 1 && item.expected_conflict == 0) {
            fp++;
            std::printf("  [FP] Subj: %s, Rel: %s, CandidateObj: %s (Expected 0, Got 1)\n",
                item.subj.c_str(), item.rel.c_str(), item.obj_candidate.c_str());
        } else if (actual_conflict == 0 && item.expected_conflict == 1) {
            fn++;
            std::printf("  [FN] Subj: %s, Rel: %s, CandidateObj: %s (Expected 1, Got 0)\n",
                item.subj.c_str(), item.rel.c_str(), item.obj_candidate.c_str());
        } else {
            BENCH_ASSERT(false, "Invalid conflict return code");
        }
    }

    double elapsed_us = qpc_time_us() - t0;
    double avg_us = elapsed_us / static_cast<double>(dataset.size());

    double accuracy = static_cast<double>(tp + tn) / static_cast<double>(dataset.size()) * 100.0;
    double precision = (tp + fp > 0) ? (static_cast<double>(tp) / static_cast<double>(tp + fp) * 100.0) : 0.0;
    double recall = (tp + fn > 0) ? (static_cast<double>(tp) / static_cast<double>(tp + fn) * 100.0) : 0.0;

    std::printf("\n--- CONFLICT DETECTION ACCURACY METRICS ---\n");
    std::printf("  Total Test Facts Evaluated:  %zu\n", dataset.size());
    std::printf("  True Positives (Conflicts):  %d\n", tp);
    std::printf("  True Negatives (Agreed/New): %d\n", tn);
    std::printf("  False Positives (Spurious):  %d\n", fp);
    std::printf("  False Negatives (Missed):    %d\n", fn);
    std::printf("  Total Latency:               %.2f us (Avg: %.3f us/check)\n", elapsed_us, avg_us);
    std::printf("  Calculated Accuracy:         %.4f %%\n", accuracy);
    std::printf("  Calculated Precision:        %.4f %%\n", precision);
    std::printf("  Calculated Recall:           %.4f %%\n", recall);
    std::printf("--------------------------------------------\n");

    BENCH_ASSERT(fp == 0, "False positives detected in conflict checking!");
    BENCH_ASSERT(fn == 0, "False negatives detected in conflict checking!");
    BENCH_ASSERT(accuracy == 100.0, "Conflict detection accuracy must be strictly 100.0%");
    BENCH_ASSERT(precision == 100.0, "Conflict detection precision must be strictly 100.0%");
    BENCH_ASSERT(recall == 100.0, "Conflict detection recall must be strictly 100.0%");

    vsa_db_destroy(db_truth);
    vsa_db_destroy(db_candidate);
    return true;
}

// =============================================================================
// TEST 3: RAW VECTOR COSINE ARGMAX SCAN & NOISE TOLERANCE
// Assert: Correct argmax recovery, precise cosine similarity math, noise tolerance
// =============================================================================

static bool bench_raw_vector_cosine_argmax_scan() {
    std::printf("[*] Setting up Raw Vector Database with 500 distinct 512-bit hypervectors...\n");

    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    BENCH_ASSERT(db != nullptr, "Create DB failed");

    const int NUM_VECTORS = 500;
    std::vector<std::vector<uint64_t>> base_vectors(NUM_VECTORS, std::vector<uint64_t>(VSA_WORDS));

    uint64_t prng_state = 0xCAFEBABE12345678ULL;
    for (int i = 0; i < NUM_VECTORS; ++i) {
        for (int w = 0; w < VSA_WORDS; ++w) {
            base_vectors[i][w] = xorshift64(&prng_state);
        }
        BENCH_ASSERT(vsa_db_insert_vector(db, 0x100000ULL + i, base_vectors[i].data()) == 0, "Insert vector failed");
    }

    std::printf("[+] Stored %d 512-bit vectors in database.\n", NUM_VECTORS);

    // 1. Exact Argmax Scan Test
    std::printf("[*] Verifying exact match recovery across 100 random targets...\n");
    int exact_matches_ok = 0;
    for (int t = 0; t < 100; ++t) {
        int target_idx = (t * 5) % NUM_VECTORS;
        uint64_t ret_vec[VSA_WORDS];
        float sim = 0.0f;

        int res = vsa_db_query_vector(db, base_vectors[target_idx].data(), ret_vec, &sim);
        BENCH_ASSERT(res == 1, "Vector query failed");
        BENCH_ASSERT(std::memcmp(ret_vec, base_vectors[target_idx].data(), sizeof(uint64_t) * VSA_WORDS) == 0, "Vector content mismatch on exact query");
        BENCH_ASSERT(std::abs(sim - 1.0f) < 1e-5f, "Exact cosine similarity must be 1.000000f");
        exact_matches_ok++;
    }
    std::printf("[+] Exact recovery: %d / 100 targets perfectly matched (100%%).\n", exact_matches_ok);

    // 2. Controlled Noise Bit-Flip Tolerance & Cosine Math Verification
    // Noise levels: 1, 2, 4, 8, 16, 32, 64 bits flipped
    const std::vector<int> noise_levels = {1, 2, 4, 8, 16, 32, 64};

    std::printf("\n[*] Testing Noise Tolerance & Cosine Math across %zu noise levels...\n", noise_levels.size());
    std::printf("  %-12s | %-15s | %-15s | %-15s | %-12s\n",
        "Bits Flipped", "Expected Sim", "Actual Sim", "Max Sim Error", "Argmax Match");
    std::printf("--------------------------------------------------------------------------------\n");

    for (int flips : noise_levels) {
        // Analytical similarity:
        // Matching bits M = 512 - flips
        // sim = (2 * M - 512) / 512.0 = (1024 - 2*flips - 512) / 512 = (512 - 2*flips) / 512 = 1.0 - (2*flips)/512.0
        float expected_sim = 1.0f - (2.0f * static_cast<float>(flips)) / 512.0f;

        int correct_argmax_count = 0;
        float max_error = 0.0f;
        const int NUM_TRIALS = 50;

        for (int trial = 0; trial < NUM_TRIALS; ++trial) {
            int target_idx = (trial * 7) % NUM_VECTORS;
            std::vector<uint64_t> noisy_vec = base_vectors[target_idx];

            // Introduce exactly `flips` bit flips in distinct bit positions
            std::vector<int> bit_positions(512);
            for (int b = 0; b < 512; ++b) bit_positions[b] = b;
            for (int b = 0; b < flips; ++b) {
                int swap_pos = b + static_cast<int>(xorshift64(&prng_state) % (512 - b));
                std::swap(bit_positions[b], bit_positions[swap_pos]);
                int bit_idx = bit_positions[b];
                int word_idx = bit_idx / 64;
                int bit_in_word = bit_idx % 64;
                noisy_vec[word_idx] ^= (1ULL << bit_in_word);
            }

            uint64_t ret_vec[VSA_WORDS];
            float actual_sim = 0.0f;
            int res = vsa_db_query_vector(db, noisy_vec.data(), ret_vec, &actual_sim);
            BENCH_ASSERT(res == 1, "Noisy query failed");

            // Verify argmax selects original target
            if (std::memcmp(ret_vec, base_vectors[target_idx].data(), sizeof(uint64_t) * VSA_WORDS) == 0) {
                correct_argmax_count++;
            }

            float err = std::abs(actual_sim - expected_sim);
            if (err > max_error) max_error = err;
        }

        std::printf("  %-12d | %-15.6f | %-15.6f | %-15.6e | %d / %d (%s)\n",
            flips, expected_sim, expected_sim, max_error,
            correct_argmax_count, NUM_TRIALS,
            (correct_argmax_count == NUM_TRIALS) ? "PERFECT" : "DEGRADED");

        BENCH_ASSERT(max_error < 1e-5f, "Cosine similarity calculation error exceeds tolerance!");
        BENCH_ASSERT(correct_argmax_count == NUM_TRIALS, "Argmax failed to retrieve the correct target under noise!");
    }

    // 3. Bitwise Inverse Query Test (Negative Cosine Similarity: -1.0)
    std::printf("\n[*] Testing Bitwise Opposite Vector (Expected Sim: -1.0)...\n");
    VsaDbHandle* db_single = vsa_db_create(VSA_CAP_MICRO, nullptr);
    BENCH_ASSERT(db_single != nullptr, "Create DB failed");

    uint64_t v_norm[VSA_WORDS] = {0x0123456789ABCDEFULL, 0x1111222233334444ULL, 0, 0, 0, 0, 0, 0};
    uint64_t v_inv[VSA_WORDS];
    for (int w = 0; w < VSA_WORDS; ++w) v_inv[w] = ~v_norm[w];

    vsa_db_insert_vector(db_single, 1, v_norm);

    uint64_t ret_inv[VSA_WORDS];
    float sim_inv = 0.0f;
    int res_inv = vsa_db_query_vector(db_single, v_inv, ret_inv, &sim_inv);
    BENCH_ASSERT(res_inv == 1, "Inverse vector query failed");
    std::printf("  Sim for exact bitwise NOT: %.6f (Expected: -1.000000)\n", sim_inv);
    BENCH_ASSERT(std::abs(sim_inv - (-1.0f)) < 1e-5f, "Inverse vector similarity must be exactly -1.0f");
    vsa_db_destroy(db_single);

    // 4. Large-Scale Argmax Throughput & Latency Profiling
    std::printf("\n[*] Profiling Argmax Scan Latency across 1,000 queries over %d stored vectors...\n", NUM_VECTORS);
    const int BENCH_QUERIES = 1000;
    double t_start_q = qpc_time_us();

    for (int q = 0; q < BENCH_QUERIES; ++q) {
        int target = q % NUM_VECTORS;
        uint64_t out_v[VSA_WORDS];
        float out_s = 0.0f;
        vsa_db_query_vector(db, base_vectors[target].data(), out_v, &out_s);
    }

    double t_end_q = qpc_time_us();
    double total_us = t_end_q - t_start_q;
    double avg_q_us = total_us / static_cast<double>(BENCH_QUERIES);
    double qps = (static_cast<double>(BENCH_QUERIES) / total_us) * 1000000.0;

    std::printf("--- ARGMAX SCAN PERFORMANCE METRICS ---\n");
    std::printf("  Total Scan Queries:          %d\n", BENCH_QUERIES);
    std::printf("  Database Stored Vectors:     %d\n", NUM_VECTORS);
    std::printf("  Total Benchmark Time:        %.2f ms\n", total_us / 1000.0);
    std::printf("  Average Latency per Query:   %.2f us\n", avg_q_us);
    std::printf("  Scan Throughput:             %.0f QPS\n", qps);
    std::printf("----------------------------------------\n");

    BENCH_ASSERT(avg_q_us < 500.0, "Argmax scan latency exceeded 500 us target!");

    vsa_db_destroy(db);
    return true;
}

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

int main(int argc, char** argv) {
    std::printf("=================================================================\n");
    std::printf("CHALLENGER M2.2: ADVERSARIAL BENCHMARK & VERIFICATION HARNESS\n");
    std::printf("Milestone: M2 (Dynamic Non-Fixed Memory & Knowledge Base)\n");
    std::printf("Verification Scope: Hot-Swapping, Conflict Detection, Raw Vectors\n");
    std::printf("=================================================================\n");

    RUN_BENCH(bench_live_hot_swapping_10k);
    RUN_BENCH(bench_cross_db_conflict_detection_100pct);
    RUN_BENCH(bench_raw_vector_cosine_argmax_scan);

    std::printf("\n=================================================================\n");
    std::printf("BENCHMARK SUMMARY: %d Executed, %d Passed, %d Failed\n",
        g_bench_run, g_bench_passed, g_bench_failed);
    std::printf("=================================================================\n");

    if (g_bench_failed > 0) {
        std::printf("[VERDICT] REQUEST_CHANGES — Verification harness failed with %d failure(s).\n", g_bench_failed);
        return 1;
    }

    std::printf("[VERDICT] APPROVE — All adversarial verification benchmarks passed with 0 errors.\n");
    return 0;
}

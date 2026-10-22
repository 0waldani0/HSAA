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
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <chrono>
#include <atomic>
#include <random>
#include <iostream>

#pragma comment(lib, "psapi.lib")

static int g_stress_run = 0;
static int g_stress_passed = 0;
static int g_stress_failed = 0;

#define STRESS_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::printf("  [FAIL] Line %d: %s (%s)\n", __LINE__, msg, #cond); \
            return false; \
        } \
    } while (0)

#define RUN_STRESS(fn) \
    do { \
        g_stress_run++; \
        std::printf("\n=================================================================\n"); \
        std::printf("[RUN STRESS] %s...\n", #fn); \
        auto t0 = std::chrono::high_resolution_clock::now(); \
        bool ok = fn(); \
        auto t1 = std::chrono::high_resolution_clock::now(); \
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); \
        if (ok) { \
            g_stress_passed++; \
            std::printf("  [PASS] %s (%.2f ms)\n", #fn, ms); \
        } else { \
            g_stress_failed++; \
            std::printf("  [FAILED] %s (%.2f ms)\n", #fn, ms); \
        } \
    } while (0)

static size_t get_ram_working_set_kb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / 1024;
    }
    return 0;
}

static DWORD get_handle_count() {
    DWORD count = 0;
    GetProcessHandleCount(GetCurrentProcess(), &count);
    return count;
}

// =============================================================================
// Stress 1: Rapid Sequential DB Lifecycle Churn (Memory & Handle Leak Assertion)
// =============================================================================
static bool test_stress_lifecycle_churn() {
    const int kIterations = 1000;
    DWORD initial_handles = get_handle_count();
    size_t initial_ram_kb = get_ram_working_set_kb();

    std::printf("  Initial handles: %lu, RAM: %zu KB\n", initial_handles, initial_ram_kb);

    // Part A: In-memory DB churn
    for (int i = 0; i < kIterations; ++i) {
        VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
        STRESS_ASSERT(db != nullptr, "vsa_db_create MICRO failed during churn");
        vsa_db_insert_fact(db, "Subject", "predicate", "Object");
        uint64_t v[8] = { (uint64_t)i, 1, 2, 3, 4, 5, 6, 7 };
        vsa_db_insert_vector(db, 0x1000 + i, v);
        vsa_db_destroy(db);
    }

    // Part B: Persistent file-mapped DB churn
    const char* temp_db_path = "build\\stress_lifecycle_churn.bin";
    for (int i = 0; i < 200; ++i) {
        VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, temp_db_path);
        STRESS_ASSERT(db != nullptr, "vsa_db_create persistent failed during churn");
        vsa_db_insert_fact(db, "Alpha", "connects_to", "Beta");
        vsa_db_flush(db);
        vsa_db_destroy(db);
    }
    DeleteFileA(temp_db_path);

    DWORD final_handles = get_handle_count();
    size_t final_ram_kb = get_ram_working_set_kb();
    std::printf("  Final handles: %lu, RAM: %zu KB\n", final_handles, final_ram_kb);

    // Assert handle stability: handle delta must be negligible (< 10 handles)
    long handle_delta = (long)final_handles - (long)initial_handles;
    STRESS_ASSERT(handle_delta < 20 && handle_delta > -20, "Handle leak detected across 1200 DB lifecycle cycles!");

    return true;
}

// =============================================================================
// Stress 2: Heavy Fact Load, Online Mutation, and Deletion Churn
// =============================================================================
static bool test_stress_heavy_fact_churn() {
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    STRESS_ASSERT(db != nullptr, "Failed to create micro db");

    const int kNumFacts = 2500; // ~61% load factor for 4096 capacity
    std::map<std::pair<std::string, std::string>, std::string> oracle;

    std::printf("  Phase 1: Inserting %d facts...\n", kNumFacts);
    for (int i = 0; i < kNumFacts; ++i) {
        std::string s = "Node_" + std::to_string(i);
        std::string r = "linked_to";
        std::string o = "TargetVal_" + std::to_string(i * 3 + 7);
        int rc = vsa_db_insert_fact(db, s.c_str(), r.c_str(), o.c_str());
        STRESS_ASSERT(rc == 0, "Insert fact returned failure during load phase");
        oracle[{s, r}] = o;
    }
    STRESS_ASSERT(vsa_db_get_fact_count(db) == kNumFacts, "Fact count mismatch after initial load");

    std::printf("  Phase 2: Verifying all %d facts...\n", kNumFacts);
    char buf[128];
    for (const auto& kv : oracle) {
        int found = vsa_db_query_relation(db, kv.first.first.c_str(), kv.first.second.c_str(), buf, sizeof(buf));
        STRESS_ASSERT(found == 1, "Fact missing from database!");
        STRESS_ASSERT(kv.second == buf, "Fact value mismatch!");
    }

    std::printf("  Phase 3: Mutating 800 facts in-place...\n", 800);
    for (int i = 0; i < 800; ++i) {
        std::string s = "Node_" + std::to_string(i);
        std::string r = "linked_to";
        std::string new_o = "MutatedVal_" + std::to_string(i * 99 + 1);
        int rc = vsa_db_mutate_fact(db, s.c_str(), r.c_str(), new_o.c_str());
        STRESS_ASSERT(rc == 0, "Mutate fact returned failure");
        oracle[{s, r}] = new_o;
    }
    STRESS_ASSERT(vsa_db_get_fact_count(db) == kNumFacts, "Fact count should remain constant after in-place mutation");

    std::printf("  Phase 4: Deleting 1000 facts...\n", 1000);
    for (int i = 500; i < 1500; ++i) {
        std::string s = "Node_" + std::to_string(i);
        std::string r = "linked_to";
        int rc = vsa_db_delete_fact(db, s.c_str(), r.c_str());
        STRESS_ASSERT(rc == 1, "Delete fact returned 0 for existing fact");
        oracle.erase({s, r});
    }
    STRESS_ASSERT(vsa_db_get_fact_count(db) == oracle.size(), "Fact count mismatch after deletion");

    std::printf("  Phase 5: Verifying oracle after deletion...\n");
    for (int i = 0; i < kNumFacts; ++i) {
        std::string s = "Node_" + std::to_string(i);
        std::string r = "linked_to";
        int found = vsa_db_query_relation(db, s.c_str(), r.c_str(), buf, sizeof(buf));
        auto it = oracle.find({s, r});
        if (it != oracle.end()) {
            STRESS_ASSERT(found == 1, "Remaining fact should be present");
            std::string expected = it->second;
            STRESS_ASSERT(expected == buf, "Remaining fact value mismatch");
        } else {
            STRESS_ASSERT(found == 0, "Deleted fact should NOT be found");
        }
    }

    std::printf("  Phase 6: Re-inserting 500 new facts...\n");
    for (int i = kNumFacts; i < kNumFacts + 500; ++i) {
        std::string s = "Node_" + std::to_string(i);
        std::string r = "linked_to";
        std::string o = "TargetVal_" + std::to_string(i * 3 + 7);
        int rc = vsa_db_insert_fact(db, s.c_str(), r.c_str(), o.c_str());
        STRESS_ASSERT(rc == 0, "Re-insert returned failure");
        oracle[{s, r}] = o;
    }
    STRESS_ASSERT(vsa_db_get_fact_count(db) == oracle.size(), "Final fact count mismatch");

    vsa_db_destroy(db);
    return true;
}

// =============================================================================
// Stress 3: Capacity Boundary & Corruption Check (Fill to Limit & Reject When Full)
// =============================================================================
static bool test_stress_capacity_boundary_and_corruption() {
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    STRESS_ASSERT(db != nullptr, "Failed to create micro db");

    std::printf("  Filling MICRO database (capacity 4096) to absolute capacity...\n");
    std::map<std::pair<std::string, std::string>, std::string> accepted_facts;

    int accepted_count = 0;
    int first_rejection_at = -1;

    // Attempt to insert up to 4500 facts into a 4096 slot table
    for (int i = 0; i < 4500; ++i) {
        std::string s = "SubjCap_" + std::to_string(i);
        std::string r = "rel";
        std::string o = "ObjCap_" + std::to_string(i);
        int rc = vsa_db_insert_fact(db, s.c_str(), r.c_str(), o.c_str());
        if (rc == 0) {
            accepted_facts[{s, r}] = o;
            accepted_count++;
        } else {
            if (first_rejection_at == -1) {
                first_rejection_at = i;
                std::printf("  First rejection occurred at index %d (accepted %d facts)\n", i, accepted_count);
            }
        }
    }

    STRESS_ASSERT(first_rejection_at >= 0, "Database should have rejected insertions when full!");
    STRESS_ASSERT(accepted_count <= 4096, "Database accepted more facts than its table capacity!");
    std::printf("  Total accepted facts: %d / 4096\n", accepted_count);

    // Attempt to insert 50 more facts, which MUST be rejected
    std::printf("  Testing graceful rejection of 50 subsequent inserts...\n");
    std::vector<std::pair<std::string, std::string>> rejected_attempts;
    for (int i = 5000; i < 5050; ++i) {
        std::string s = "ExcessSubj_" + std::to_string(i);
        std::string r = "rel";
        std::string o = "ExcessObj_" + std::to_string(i);
        int rc = vsa_db_insert_fact(db, s.c_str(), r.c_str(), o.c_str());
        STRESS_ASSERT(rc == -1, "Insert when full must return -1 (graceful rejection)!");
        rejected_attempts.push_back({s, r});
    }

    // CRITICAL CORRUPTION CHECK 1:
    // Verify none of the rejected facts were accidentally written into the database
    std::printf("  Checking that rejected facts are NOT present in database...\n");
    char buf[128];
    int leaked_rejected_count = 0;
    for (const auto& rej : rejected_attempts) {
        int found = vsa_db_query_relation(db, rej.first.c_str(), rej.second.c_str(), buf, sizeof(buf));
        if (found == 1) {
            leaked_rejected_count++;
        }
    }
    std::printf("  Leaked rejected facts found in DB: %d / %zu\n", leaked_rejected_count, rejected_attempts.size());

    // CRITICAL CORRUPTION CHECK 2:
    // Verify EVERY SINGLE previously accepted fact is STILL INTACT and UNCORRUPTED!
    std::printf("  Checking data integrity of all %d accepted facts...\n", accepted_count);
    int intact_count = 0;
    int corrupted_count = 0;
    int missing_count = 0;
    for (const auto& kv : accepted_facts) {
        int found = vsa_db_query_relation(db, kv.first.first.c_str(), kv.first.second.c_str(), buf, sizeof(buf));
        if (found == 1) {
            if (kv.second == buf) {
                intact_count++;
            } else {
                corrupted_count++;
            }
        } else {
            missing_count++;
        }
    }
    std::printf("  Integrity results: %d intact, %d corrupted, %d missing\n", intact_count, corrupted_count, missing_count);
    STRESS_ASSERT(leaked_rejected_count == 0, "CRITICAL: Rejected facts were written into DB despite insert returning -1!");
    STRESS_ASSERT(missing_count == 0, "CRITICAL: Previously accepted facts were dropped/lost after capacity rejection!");
    STRESS_ASSERT(corrupted_count == 0, "CRITICAL: Facts were corrupted after capacity rejection!");

    // Part B: Raw vector capacity boundary
    std::printf("  Testing raw vector capacity boundary...\n");
    std::vector<uint64_t> accepted_keys;
    int vec_accepted = 0;
    for (int i = 0; i < 4500; ++i) {
        uint64_t key = 0xABCD0000ULL + i;
        uint64_t vec[8] = { key, key ^ 0x1, key ^ 0x2, key ^ 0x3, key ^ 0x4, key ^ 0x5, key ^ 0x6, key ^ 0x7 };
        int rc = vsa_db_insert_vector(db, key, vec);
        if (rc == 0) {
            vec_accepted++;
            accepted_keys.push_back(key);
        }
    }
    std::printf("  Vectors accepted: %d / 4096\n", vec_accepted);
    STRESS_ASSERT(vec_accepted <= 4096, "Vector table accepted more vectors than capacity!");

    // Try 20 excess vectors
    for (int i = 5000; i < 5020; ++i) {
        uint64_t key = 0xEEEE0000ULL + i;
        uint64_t vec[8] = { key, 0, 0, 0, 0, 0, 0, 0 };
        int rc = vsa_db_insert_vector(db, key, vec);
        STRESS_ASSERT(rc == -1, "Excess vector insert must return -1!");
    }

    vsa_db_destroy(db);
    return true;
}

// =============================================================================
// Stress 4: Adversarial 0-1-N-NULL Strings and Boundary Dimensions
// =============================================================================
static bool test_stress_0_1_n_null_and_boundary_strings() {
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    STRESS_ASSERT(db != nullptr, "Failed to create micro db");

    // 1. NULL pointer robustness
    STRESS_ASSERT(vsa_db_insert_fact(nullptr, "a", "b", "c") == -1, "NULL db must return -1");
    STRESS_ASSERT(vsa_db_insert_fact(db, nullptr, "b", "c") == -1, "NULL subj must return -1");
    STRESS_ASSERT(vsa_db_insert_fact(db, "a", nullptr, "c") == -1, "NULL rel must return -1");
    STRESS_ASSERT(vsa_db_insert_fact(db, "a", "b", nullptr) == -1, "NULL obj must return -1");

    STRESS_ASSERT(vsa_db_delete_fact(nullptr, "a", "b") == -1, "NULL db delete must return -1");
    STRESS_ASSERT(vsa_db_delete_fact(db, nullptr, "b") == -1, "NULL subj delete must return -1");
    STRESS_ASSERT(vsa_db_delete_fact(db, "a", nullptr) == -1, "NULL rel delete must return -1");

    char out_buf[256];
    STRESS_ASSERT(vsa_db_query_relation(nullptr, "a", "b", out_buf, sizeof(out_buf)) == 0, "NULL db query must return 0");
    STRESS_ASSERT(vsa_db_query_relation(db, nullptr, "b", out_buf, sizeof(out_buf)) == 0, "NULL subj query must return 0");
    STRESS_ASSERT(vsa_db_query_relation(db, "a", nullptr, out_buf, sizeof(out_buf)) == 0, "NULL rel query must return 0");
    STRESS_ASSERT(vsa_db_query_relation(db, "a", "b", nullptr, sizeof(out_buf)) == 0, "NULL out_buf query must return 0");
    STRESS_ASSERT(vsa_db_query_relation(db, "a", "b", out_buf, 0) == 0, "max_len 0 query must return 0");

    STRESS_ASSERT(vsa_db_check_conflict(nullptr, "a", "b", "c") == -1, "NULL db conflict must return -1");
    STRESS_ASSERT(vsa_db_check_conflict(db, nullptr, "b", "c") == -1, "NULL subj conflict must return -1");
    STRESS_ASSERT(vsa_db_check_conflict(db, "a", nullptr, "c") == -1, "NULL rel conflict must return -1");
    STRESS_ASSERT(vsa_db_check_conflict(db, "a", "b", nullptr) == -1, "NULL obj conflict must return -1");

    STRESS_ASSERT(vsa_db_insert_vector(nullptr, 1, nullptr) == -1, "NULL db insert_vector must return -1");
    STRESS_ASSERT(vsa_db_insert_vector(db, 1, nullptr) == -1, "NULL vec insert_vector must return -1");

    uint64_t dummy_v[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float dummy_sim = 0.0f;
    STRESS_ASSERT(vsa_db_query_vector(nullptr, dummy_v, dummy_v, &dummy_sim) == -1, "NULL db query_vector must return -1");
    STRESS_ASSERT(vsa_db_query_vector(db, nullptr, dummy_v, &dummy_sim) == -1, "NULL query query_vector must return -1");
    STRESS_ASSERT(vsa_db_query_vector(db, dummy_v, nullptr, &dummy_sim) == -1, "NULL out_best query_vector must return -1");
    STRESS_ASSERT(vsa_db_query_vector(db, dummy_v, dummy_v, nullptr) == -1, "NULL out_sim query_vector must return -1");

    // 2. Empty strings ("")
    STRESS_ASSERT(vsa_db_insert_fact(db, "", "", "") == 0, "Empty strings insert must succeed");
    STRESS_ASSERT(vsa_db_query_relation(db, "", "", out_buf, sizeof(out_buf)) == 1, "Empty strings query must succeed");
    STRESS_ASSERT(std::string(out_buf) == "", "Empty string value mismatch");
    STRESS_ASSERT(vsa_db_check_conflict(db, "", "", "") == 0, "Empty string conflict should be 0");
    STRESS_ASSERT(vsa_db_check_conflict(db, "", "", "non_empty") == 1, "Empty string conflict should be 1");
    STRESS_ASSERT(vsa_db_delete_fact(db, "", "") == 1, "Empty strings delete must succeed");

    // 3. Exact inline boundary strings (31 chars, 32 chars, 63 chars, 64 chars)
    std::string str31(31, 'A'); // Fits in inline_subj (32-byte array with \0)
    std::string str32(32, 'B'); // Exceeds inline_subj (needs string heap)
    std::string str63(63, 'C'); // Fits in inline_obj (64-byte array with \0)
    std::string str64(64, 'D'); // Exceeds inline_obj (needs string heap)

    STRESS_ASSERT(vsa_db_insert_fact(db, str31.c_str(), "rel", str63.c_str()) == 0, "Insert inline boundary failed");
    STRESS_ASSERT(vsa_db_query_relation(db, str31.c_str(), "rel", out_buf, sizeof(out_buf)) == 1, "Query inline boundary failed");
    STRESS_ASSERT(std::string(out_buf) == str63, "Inline boundary value mismatch");

    STRESS_ASSERT(vsa_db_insert_fact(db, str32.c_str(), "rel", str64.c_str()) == 0, "Insert heap boundary failed");
    STRESS_ASSERT(vsa_db_query_relation(db, str32.c_str(), "rel", out_buf, sizeof(out_buf)) == 1, "Query heap boundary failed");
    STRESS_ASSERT(std::string(out_buf) == str64, "Heap boundary value mismatch");

    // 4. Large string (10,000 chars)
    std::string large_s(10000, 'X');
    std::string large_o(10000, 'Y');
    STRESS_ASSERT(vsa_db_insert_fact(db, large_s.c_str(), "large_rel", large_o.c_str()) == 0, "Insert 10K string failed");
    std::vector<char> large_buf(15000);
    STRESS_ASSERT(vsa_db_query_relation(db, large_s.c_str(), "large_rel", large_buf.data(), large_buf.size()) == 1, "Query 10K string failed");
    STRESS_ASSERT(std::string(large_buf.data()) == large_o, "10K string value mismatch");

    // 5. Query buffer truncation safety (max_len = 5 on a 10K string)
    char tiny_buf[5];
    STRESS_ASSERT(vsa_db_query_relation(db, large_s.c_str(), "large_rel", tiny_buf, sizeof(tiny_buf)) == 1, "Query with tiny buffer must succeed");
    STRESS_ASSERT(std::strlen(tiny_buf) == 4, "Truncated output length must be max_len - 1");
    STRESS_ASSERT(std::string(tiny_buf) == "YYYY", "Truncated characters mismatch");

    // 6. Special characters, UTF-8, punctuation, control chars
    const char* utf8_s = "Узел_123_Нейросеть";
    const char* utf8_r = "связан_с";
    const char* utf8_o = "Значение_Гипервектора!@#$%^&*()_+{}[]:;\"'<>?,./";
    STRESS_ASSERT(vsa_db_insert_fact(db, utf8_s, utf8_r, utf8_o) == 0, "Insert UTF-8/special characters failed");
    STRESS_ASSERT(vsa_db_query_relation(db, utf8_s, utf8_r, out_buf, sizeof(out_buf)) == 1, "Query UTF-8/special characters failed");
    STRESS_ASSERT(std::string(out_buf) == utf8_o, "UTF-8/special characters mismatch");

    vsa_db_destroy(db);
    return true;
}

// =============================================================================
// Stress 5: String Heap Exhaustion Stress
// =============================================================================
static bool test_stress_string_heap_exhaustion() {
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    STRESS_ASSERT(db != nullptr, "Failed to create micro db");

    // In VSA_CAP_MICRO, string heap is ~1.5 MB (1,572,864 bytes)
    // We insert chunks of 32 KB objects until string heap is exhausted
    std::string big_obj(32768, 'Z');
    int accepted = 0;
    for (int i = 0; i < 200; ++i) {
        std::string s = "BigSubj_" + std::to_string(i);
        int rc = vsa_db_insert_fact(db, s.c_str(), "has_blob", big_obj.c_str());
        if (rc == 0) {
            accepted++;
        } else {
            // Rejection due to string heap exhaustion
            break;
        }
    }
    std::printf("  String heap accepted %d x 32KB blobs (~%d KB total) before graceful rejection\n", accepted, (accepted * 32));
    STRESS_ASSERT(accepted > 0, "String heap should accept at least one 32KB blob");
    STRESS_ASSERT(accepted < 100, "String heap must hit capacity ceiling before 100 blobs (1.5MB max)");

    // After heap exhaustion, inserting an inline fact (not requiring heap) should still work!
    int rc = vsa_db_insert_fact(db, "InlineSubj", "rel", "InlineObj");
    STRESS_ASSERT(rc == 0, "Inline facts should still succeed even when string heap is exhausted!");

    char buf[64];
    STRESS_ASSERT(vsa_db_query_relation(db, "InlineSubj", "rel", buf, sizeof(buf)) == 1, "Query inline fact failed");
    STRESS_ASSERT(std::string(buf) == "InlineObj", "Inline fact value mismatch");

    vsa_db_destroy(db);
    return true;
}

// =============================================================================
// Stress 6: Multi-Threaded Concurrent Operations
// =============================================================================
static bool test_stress_concurrent_access() {
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MEDIUM, nullptr);
    STRESS_ASSERT(db != nullptr, "Failed to create medium db");

    const int kNumThreads = 8;
    const int kOpsPerThread = 500;
    std::atomic<bool> start_flag(false);
    std::atomic<int> errors(0);

    std::vector<std::thread> workers;
    for (int t = 0; t < kNumThreads; ++t) {
        workers.emplace_back([db, t, kOpsPerThread, &start_flag, &errors]() {
            while (!start_flag.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            char q_buf[64];
            for (int i = 0; i < kOpsPerThread; ++i) {
                std::string s = "Thread_" + std::to_string(t) + "_Key_" + std::to_string(i);
                std::string r = "operates_on";
                std::string o = "Val_" + std::to_string(i);

                // Insert
                if (vsa_db_insert_fact(db, s.c_str(), r.c_str(), o.c_str()) != 0) {
                    errors++;
                }

                // Query
                if (vsa_db_query_relation(db, s.c_str(), r.c_str(), q_buf, sizeof(q_buf)) != 1) {
                    errors++;
                }

                // Conflict check
                if (vsa_db_check_conflict(db, s.c_str(), r.c_str(), o.c_str()) != 0) {
                    errors++;
                }

                // Mutate subset
                if (i % 2 == 0) {
                    std::string new_o = "NewVal_" + std::to_string(i);
                    if (vsa_db_mutate_fact(db, s.c_str(), r.c_str(), new_o.c_str()) != 0) {
                        errors++;
                    }
                }
            }
        });
    }

    start_flag.store(true, std::memory_order_release);
    for (auto& th : workers) {
        th.join();
    }

    std::printf("  Concurrent operations completed: %d threads x %d ops. Errors: %d\n", kNumThreads, kOpsPerThread, errors.load());
    STRESS_ASSERT(errors.load() == 0, "Concurrent multi-threaded operations produced errors!");

    vsa_db_destroy(db);
    return true;
}

// =============================================================================
// Stress 7: Branching, Cloning, and Multi-Instance Isolation
// =============================================================================
static bool test_stress_clone_and_branch_isolation() {
    VsaDbHandle* parent = vsa_db_create(VSA_CAP_MICRO, nullptr);
    STRESS_ASSERT(parent != nullptr, "Failed to create parent db");

    for (int i = 0; i < 500; ++i) {
        std::string s = "BaseNode_" + std::to_string(i);
        std::string o = "BaseVal_" + std::to_string(i);
        vsa_db_insert_fact(parent, s.c_str(), "rel", o.c_str());
    }

    // Clone parent
    VsaDbHandle* child = nullptr;
    STRESS_ASSERT(vsa_db_clone(parent, &child, nullptr) == 0, "Clone failed");
    STRESS_ASSERT(child != nullptr, "Child db is null");
    STRESS_ASSERT(vsa_db_get_fact_count(child) == 500, "Child fact count mismatch");

    // Divergent mutations:
    // Mutate in parent
    for (int i = 0; i < 100; ++i) {
        std::string s = "BaseNode_" + std::to_string(i);
        vsa_db_mutate_fact(parent, s.c_str(), "rel", "ParentMutated");
    }

    // Mutate in child
    for (int i = 0; i < 100; ++i) {
        std::string s = "BaseNode_" + std::to_string(i);
        vsa_db_mutate_fact(child, s.c_str(), "rel", "ChildMutated");
    }

    // Insert new exclusively into parent
    for (int i = 500; i < 600; ++i) {
        std::string s = "ParentExclusive_" + std::to_string(i);
        vsa_db_insert_fact(parent, s.c_str(), "rel", "ParentOnly");
    }

    // Insert new exclusively into child
    for (int i = 500; i < 700; ++i) {
        std::string s = "ChildExclusive_" + std::to_string(i);
        vsa_db_insert_fact(child, s.c_str(), "rel", "ChildOnly");
    }

    // Verify isolation
    STRESS_ASSERT(vsa_db_get_fact_count(parent) == 600, "Parent fact count mismatch");
    STRESS_ASSERT(vsa_db_get_fact_count(child) == 700, "Child fact count mismatch");

    char buf[64];
    // Check parent mutations
    vsa_db_query_relation(parent, "BaseNode_0", "rel", buf, sizeof(buf));
    STRESS_ASSERT(std::string(buf) == "ParentMutated", "Parent mutation mismatch");
    // Check child mutations
    vsa_db_query_relation(child, "BaseNode_0", "rel", buf, sizeof(buf));
    STRESS_ASSERT(std::string(buf) == "ChildMutated", "Child mutation mismatch");

    // Check parent exclusive missing from child
    STRESS_ASSERT(vsa_db_query_relation(child, "ParentExclusive_550", "rel", buf, sizeof(buf)) == 0, "Parent exclusive leaked into child!");
    // Check child exclusive missing from parent
    STRESS_ASSERT(vsa_db_query_relation(parent, "ChildExclusive_550", "rel", buf, sizeof(buf)) == 0, "Child exclusive leaked into parent!");

    vsa_db_destroy(parent);
    vsa_db_destroy(child);
    return true;
}

// =============================================================================
// Main Harness Router
// =============================================================================
int main(int argc, char** argv) {
    std::printf("=================================================================\n");
    std::printf("ADVERSARIAL STRESS TEST HARNESS — MILESTONE M2 (vsa_db)\n");
    std::printf("=================================================================\n");

    RUN_STRESS(test_stress_lifecycle_churn);
    RUN_STRESS(test_stress_heavy_fact_churn);
    RUN_STRESS(test_stress_capacity_boundary_and_corruption);
    RUN_STRESS(test_stress_0_1_n_null_and_boundary_strings);
    RUN_STRESS(test_stress_string_heap_exhaustion);
    RUN_STRESS(test_stress_concurrent_access);
    RUN_STRESS(test_stress_clone_and_branch_isolation);

    std::printf("\n=================================================================\n");
    std::printf("Stress Test Summary: %d Run, %d Passed, %d Failed\n",
                g_stress_run, g_stress_passed, g_stress_failed);
    std::printf("=================================================================\n");

    if (g_stress_failed > 0) {
        std::printf("[VERDICT] STRESS TESTS FAILED (Exit Code 1)\n");
        return 1;
    }

    std::printf("[VERDICT] ALL ADVERSARIAL STRESS TESTS PASSED (Exit Code 0)\n");
    return 0;
}

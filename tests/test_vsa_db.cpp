#include "vsa_db.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#else
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/resource.h>
#ifndef MAX_PATH
#define MAX_PATH 4096
#endif
typedef uint32_t DWORD;
#define INVALID_FILE_ATTRIBUTES ((DWORD)-1)
static inline void GetTempPathA(DWORD buf_len, char* buf) {
    const char* t = "/tmp/";
    std::strncpy(buf, t, buf_len);
    buf[buf_len - 1] = '\0';
}
static inline void strcat_s(char* dst, size_t dst_sz, const char* src) {
    size_t cur = std::strlen(dst);
    if (cur + 1 < dst_sz) {
        std::strncat(dst, src, dst_sz - cur - 1);
    }
}
static inline void DeleteFileA(const char* path) {
    if (path) unlink(path);
}
static inline DWORD GetFileAttributesA(const char* path) {
    struct stat st;
    return (path && stat(path, &st) == 0) ? 1u : INVALID_FILE_ATTRIBUTES;
}
#endif

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
#include <iostream>

// =============================================================================
// Test Harness Mini-Framework
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
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (static_cast<double>(t.QuadPart) * 1000000.0) / static_cast<double>(freq.QuadPart);
#else
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::micro>(now).count();
#endif
}

static size_t get_process_ram_mb() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024 * 1024);
    }
    return 0;
#else
    FILE* fp = std::fopen("/proc/self/statm", "r");
    if (fp) {
        unsigned long size_pages = 0, resident_pages = 0;
        if (std::fscanf(fp, "%lu %lu", &size_pages, &resident_pages) == 2) {
            std::fclose(fp);
            long page_sz = sysconf(_SC_PAGESIZE);
            return (static_cast<size_t>(resident_pages) * static_cast<size_t>(page_sz)) / (1024 * 1024);
        }
        std::fclose(fp);
    }
    return 0;
#endif
}

// =============================================================================
// Test 1: Capacity Levels & Allocations
// =============================================================================

static bool test_capacity_levels() {
    // 1. Micro Capacity (1024 pages = 4MB)
    VsaDbHandle* db_micro = vsa_db_create(VSA_CAP_MICRO, nullptr);
    TEST_ASSERT(db_micro != nullptr, "vsa_db_create MICRO failed");
    TEST_ASSERT(vsa_db_get_capacity(db_micro) == VSA_CAP_MICRO, "Capacity mismatch MICRO");
    TEST_ASSERT(vsa_db_get_active_pages(db_micro) >= 1, "Active pages should be >= 1");
    vsa_db_destroy(db_micro);

    // 2. Medium Capacity (16384 pages = 64MB)
    VsaDbHandle* db_medium = vsa_db_create(VSA_CAP_MEDIUM, nullptr);
    TEST_ASSERT(db_medium != nullptr, "vsa_db_create MEDIUM failed");
    TEST_ASSERT(vsa_db_get_capacity(db_medium) == VSA_CAP_MEDIUM, "Capacity mismatch MEDIUM");
    vsa_db_destroy(db_medium);

    // 3. Large Capacity (262144 pages = 1GB)
    VsaDbHandle* db_large = vsa_db_create(VSA_CAP_LARGE, nullptr);
    TEST_ASSERT(db_large != nullptr, "vsa_db_create LARGE failed");
    TEST_ASSERT(vsa_db_get_capacity(db_large) == VSA_CAP_LARGE, "Capacity mismatch LARGE");
    vsa_db_destroy(db_large);

    // 4. Invalid capacities must return NULL
    TEST_ASSERT(vsa_db_create(static_cast<VsaDbCapacity>(0), nullptr) == nullptr, "Capacity 0 should be rejected");
    TEST_ASSERT(vsa_db_create(static_cast<VsaDbCapacity>(999), nullptr) == nullptr, "Capacity 999 should be rejected");
    TEST_ASSERT(vsa_db_create(static_cast<VsaDbCapacity>(12345), nullptr) == nullptr, "Capacity 12345 should be rejected");

    return true;
}

// =============================================================================
// Test 2: Safe Disk Persistence Across Restarts
// =============================================================================

static bool test_persistence_lifecycle() {
    // Absolute path in the OS temp directory: the suite must pass regardless of the
    // caller's working directory (build_tests.bat does not pin one).
    static char path[MAX_PATH] = {0};
    if (path[0] == '\0') {
        GetTempPathA(MAX_PATH, path);
        strcat_s(path, sizeof(path), "project_upgrade_vsa_db_test.bin");
    }
    DeleteFileA(path);

    // Step A: Create and populate persistent DB
    VsaDbHandle* db1 = vsa_db_create(VSA_CAP_MICRO, path);
    TEST_ASSERT(db1 != nullptr, "Failed to create persistent DB");
    TEST_ASSERT(std::string(vsa_db_get_persist_path(db1)) == path, "Persist path mismatch");

    for (int i = 0; i < 25; ++i) {
        std::string s = "Entity_" + std::to_string(i);
        std::string r = "connects_to";
        std::string o = "TargetNode_" + std::to_string(i * 10);
        TEST_ASSERT(vsa_db_insert_fact(db1, s.c_str(), r.c_str(), o.c_str()) == 0, "Insert fact failed");
    }

    uint64_t raw_vec[VSA_WORDS] = {0xAA55AA55AA55AA55ULL, 1, 2, 3, 4, 5, 6, 7};
    TEST_ASSERT(vsa_db_insert_vector(db1, 0x1337, raw_vec) == 0, "Insert raw vector failed");

    TEST_ASSERT(vsa_db_flush(db1) == 0, "vsa_db_flush failed");
    vsa_db_destroy(db1);

    // Step B: Verify file exists on disk (destructor must NOT delete it!)
    DWORD attrib = GetFileAttributesA(path);
    TEST_ASSERT(attrib != INVALID_FILE_ATTRIBUTES, "Database file was deleted on destroy!");

    // Step C: Reopen existing DB and verify data integrity
    VsaDbHandle* db2 = vsa_db_create(VSA_CAP_MICRO, path);
    TEST_ASSERT(db2 != nullptr, "Failed to reopen existing DB from disk");
    TEST_ASSERT(vsa_db_get_fact_count(db2) == 25, "Fact count mismatch after reopen");
    // Contract after F22: every stored fact also registers its 512-bit binding
    // in the unified vector table, so 25 facts + 1 raw vector = 26. The point of
    // the assertion is that the vector index survives persistence, not the old
    // count -- an unindexed fact is exactly the defect this pins shut.
    TEST_ASSERT(vsa_db_get_vector_count(db2) == 26, "Vector count mismatch after reopen");

    for (int i = 0; i < 25; ++i) {
        std::string s = "Entity_" + std::to_string(i);
        std::string expected_o = "TargetNode_" + std::to_string(i * 10);
        char out[64];
        TEST_ASSERT(vsa_db_query_relation(db2, s.c_str(), "connects_to", out, sizeof(out)) == 1, "Fact missing after reopen");
        TEST_ASSERT(expected_o == out, "Object mismatch after reopen");
    }

    uint64_t ret_vec[VSA_WORDS];
    float sim = 0.0f;
    TEST_ASSERT(vsa_db_query_vector(db2, raw_vec, ret_vec, &sim) == 1, "Raw vector query failed after reopen");
    TEST_ASSERT(std::abs(sim - 1.0f) < 1e-4f, "Vector sim should be 1.0");

    // Step D: Mutate online and verify persistence update
    TEST_ASSERT(vsa_db_mutate_fact(db2, "Entity_5", "connects_to", "MutatedTarget_999") == 0, "Mutation failed");
    TEST_ASSERT(vsa_db_flush(db2) == 0, "Flush failed");
    vsa_db_destroy(db2);

    VsaDbHandle* db3 = vsa_db_create(VSA_CAP_MICRO, path);
    TEST_ASSERT(db3 != nullptr, "Reopen 3 failed");
    char out_mut[64];
    TEST_ASSERT(vsa_db_query_relation(db3, "Entity_5", "connects_to", out_mut, sizeof(out_mut)) == 1, "Query failed");
    TEST_ASSERT(std::string("MutatedTarget_999") == out_mut, "Mutated value did not persist");
    vsa_db_destroy(db3);

    DeleteFileA(path);
    return true;
}

// =============================================================================
// Test 3: Non-Destructive Cloning & Branching
// =============================================================================

static bool test_cloning_and_branching() {
    VsaDbHandle* parent = vsa_db_create(VSA_CAP_MICRO, nullptr);
    TEST_ASSERT(parent != nullptr, "Create parent failed");

    vsa_db_insert_fact(parent, "Root", "config", "Default");
    vsa_db_insert_fact(parent, "Subsystem", "state", "Active");

    VsaDbHandle* child = nullptr;
    TEST_ASSERT(vsa_db_clone(parent, &child, nullptr) == 0, "Clone failed");
    TEST_ASSERT(child != nullptr, "Child is null");

    // Mutate child without touching parent
    vsa_db_mutate_fact(child, "Root", "config", "Experimental");

    char p_out[64], c_out[64];
    TEST_ASSERT(vsa_db_query_relation(parent, "Root", "config", p_out, sizeof(p_out)) == 1, "Query parent failed");
    TEST_ASSERT(vsa_db_query_relation(child, "Root", "config", c_out, sizeof(c_out)) == 1, "Query child failed");

    TEST_ASSERT(std::string("Default") == p_out, "Parent was corrupted by child mutation");
    TEST_ASSERT(std::string("Experimental") == c_out, "Child was not updated");

    // Destroy parent and verify child remains fully intact
    vsa_db_destroy(parent);

    char c_out2[64];
    TEST_ASSERT(vsa_db_query_relation(child, "Subsystem", "state", c_out2, sizeof(c_out2)) == 1, "Child query failed after parent destroyed");
    TEST_ASSERT(std::string("Active") == c_out2, "Child state corrupted");

    vsa_db_destroy(child);
    return true;
}

// =============================================================================
// Test 4: Online Mutation & Backward-Shift Deletion
// =============================================================================

static bool test_online_mutation_and_deletion() {
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    TEST_ASSERT(db != nullptr, "Create failed");

    // 1. Insert 50 facts
    for (int i = 0; i < 50; ++i) {
        std::string s = "Item_" + std::to_string(i);
        vsa_db_insert_fact(db, s.c_str(), "has_id", std::to_string(i * 100).c_str());
    }
    TEST_ASSERT(vsa_db_get_fact_count(db) == 50, "Fact count should be 50");
    // Each fact owns one binding in the unified vector table (F22).
    TEST_ASSERT(vsa_db_get_vector_count(db) == 50, "Each fact must register a vector binding");

    // 2. Mutate 10 facts online
    for (int i = 0; i < 10; ++i) {
        std::string s = "Item_" + std::to_string(i);
        vsa_db_mutate_fact(db, s.c_str(), "has_id", "NEW_VAL");
    }

    char out[64];
    vsa_db_query_relation(db, "Item_3", "has_id", out, sizeof(out));
    TEST_ASSERT(std::string("NEW_VAL") == out, "Mutation query mismatch");

    // 3. Delete facts with backward-shift
    TEST_ASSERT(vsa_db_delete_fact(db, "Item_3", "has_id") == 1, "Delete existing fact should return 1");
    TEST_ASSERT(vsa_db_query_relation(db, "Item_3", "has_id", out, sizeof(out)) == 0, "Deleted fact should not be found");
    TEST_ASSERT(vsa_db_delete_fact(db, "Item_3", "has_id") == 0, "Repeated delete should return 0");
    TEST_ASSERT(vsa_db_get_fact_count(db) == 49, "Fact count should decrease to 49");
    TEST_ASSERT(vsa_db_get_vector_count(db) == 49, "Deleting a fact must drop its vector binding");

    // Verify remaining facts are still queryable despite backward shifting
    vsa_db_query_relation(db, "Item_4", "has_id", out, sizeof(out));
    TEST_ASSERT(std::string("NEW_VAL") == out, "Subsequent fact should remain intact");

    vsa_db_query_relation(db, "Item_49", "has_id", out, sizeof(out));
    TEST_ASSERT(std::string("4900") == out, "Tail fact should remain intact");

    // 4. Mutating non-existent fact inserts it predictably
    TEST_ASSERT(vsa_db_mutate_fact(db, "FreshKey", "prop", "FreshVal") == 0, "Mutate non-existent should insert");
    TEST_ASSERT(vsa_db_query_relation(db, "FreshKey", "prop", out, sizeof(out)) == 1, "FreshKey query failed");
    TEST_ASSERT(std::string("FreshVal") == out, "FreshKey val mismatch");

    // 5. Rapid 1000 in-place mutations without leak
    size_t ram_before = get_process_ram_mb();
    for (int i = 0; i < 1000; ++i) {
        std::string v = (i % 2 == 0) ? "EVEN_CYCLE" : "ODD_CYCLE";
        vsa_db_mutate_fact(db, "ToggleKey", "phase", v.c_str());
    }
    size_t ram_after = get_process_ram_mb();
    TEST_ASSERT(ram_after - ram_before <= 2, "Memory leak detected during 1000 rapid mutations");

    vsa_db_query_relation(db, "ToggleKey", "phase", out, sizeof(out));
    TEST_ASSERT(std::string("ODD_CYCLE") == out, "Toggle final state mismatch");

    vsa_db_destroy(db);
    return true;
}

// =============================================================================
// Test 5: Conflict Checking Against Contradictory Facts
// =============================================================================

static bool test_conflict_checking() {
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    TEST_ASSERT(db != nullptr, "Create failed");

    vsa_db_insert_fact(db, "Processor", "arch", "x86_64_AVX512");
    vsa_db_insert_fact(db, "GPU", "vram_mb", "4096");

    // Consistent facts
    TEST_ASSERT(vsa_db_check_conflict(db, "Processor", "arch", "x86_64_AVX512") == 0, "Identical fact should not conflict");
    TEST_ASSERT(vsa_db_check_conflict(db, "GPU", "vram_mb", "4096") == 0, "Identical fact should not conflict");

    // Contradictions
    TEST_ASSERT(vsa_db_check_conflict(db, "Processor", "arch", "ARM64_Neon") == 1, "Contradictory arch should trigger conflict");
    TEST_ASSERT(vsa_db_check_conflict(db, "GPU", "vram_mb", "8192") == 1, "Contradictory vram should trigger conflict");

    // Unrecorded relations (no conflict)
    TEST_ASSERT(vsa_db_check_conflict(db, "Processor", "clock_ghz", "4.6") == 0, "Unrecorded relation has no conflict");

    // Invalid parameters
    TEST_ASSERT(vsa_db_check_conflict(db, nullptr, "arch", "val") == -1, "Null subj should return -1");
    TEST_ASSERT(vsa_db_check_conflict(db, "Processor", nullptr, "val") == -1, "Null rel should return -1");
    TEST_ASSERT(vsa_db_check_conflict(db, "Processor", "arch", nullptr) == -1, "Null obj should return -1");

    vsa_db_destroy(db);
    return true;
}

// =============================================================================
// Test 6: Raw Vector Insertion & AVX-512 Cosine Argmax Scan
// =============================================================================

static bool test_raw_vector_argmax_scan() {
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    TEST_ASSERT(db != nullptr, "Create failed");

    // Base vectors
    uint64_t v_target[VSA_WORDS] = {
        0x1234567890ABCDEFULL, 0xFEDCBA0987654321ULL,
        0x0F0F0F0F0F0F0F0FULL, 0xF0F0F0F0F0F0F0F0ULL,
        0xAAAAAAAA55555555ULL, 0x55555555AAAAAAAAULL,
        0xCCCCCCCC33333333ULL, 0x33333333CCCCCCCCULL
    };

    uint64_t v_distractor[VSA_WORDS];
    for (int i = 0; i < VSA_WORDS; ++i) v_distractor[i] = ~v_target[i];

    vsa_db_insert_vector(db, 101, v_target);
    vsa_db_insert_vector(db, 102, v_distractor);

    // 1. Exact query
    uint64_t retrieved[VSA_WORDS];
    float sim = 0.0f;
    TEST_ASSERT(vsa_db_query_vector(db, v_target, retrieved, &sim) == 1, "Exact query failed");
    TEST_ASSERT(std::abs(sim - 1.0f) < 1e-4f, "Exact similarity must be 1.0");
    TEST_ASSERT(std::memcmp(retrieved, v_target, sizeof(v_target)) == 0, "Exact vector mismatch");

    // 2. Query with flipped bits
    uint64_t v_noisy[VSA_WORDS];
    std::memcpy(v_noisy, v_target, sizeof(v_target));
    v_noisy[0] ^= 0x01ULL; // 1 bit flip
    TEST_ASSERT(vsa_db_query_vector(db, v_noisy, retrieved, &sim) == 1, "Noisy query failed");
    // 511 matches out of 512: (2*511 - 512) / 512 = 510/512 = 0.99609375
    TEST_ASSERT(sim > 0.99f, "Noisy query similarity should be > 0.99");
    TEST_ASSERT(std::memcmp(retrieved, v_target, sizeof(v_target)) == 0, "Argmax should select target vector");

    // 3. Batch 200 random vectors and profile argmax scan latency
    for (uint64_t i = 1; i <= 200; ++i) {
        uint64_t rv[VSA_WORDS];
        for (int w = 0; w < VSA_WORDS; ++w) rv[w] = (i * 0x9e3779b97f4a7c15ULL) ^ w;
        vsa_db_insert_vector(db, i + 1000, rv);
    }

    double t0 = qpc_time_us();
    int res = vsa_db_query_vector(db, v_target, retrieved, &sim);
    double elapsed_us = qpc_time_us() - t0;

    TEST_ASSERT(res == 1, "Argmax search failed");
    TEST_ASSERT(std::abs(sim - 1.0f) < 1e-4f, "Target should still match exactly");
    TEST_ASSERT(elapsed_us < 500.0, "Argmax scan should complete in under 500 us");

    // 4. A non-resonant probe must be a miss. The all-zero vector is not
    //    stored and agrees with a random 512-bit vector on ~50% of bits; before
    //    the resonance gate the function answered 1 for any probe as long as
    //    the table was non-empty.
    //    The zero probe is deliberately the hard case: one of the 200 vectors
    //    above has only 180 ones, so zero agrees with it on 332/512 bits
    //    (similarity 0.297) -- above the old 0.25 gate. 0.40 rejects it.
    uint64_t v_unrelated[VSA_WORDS] = {0, 0, 0, 0, 0, 0, 0, 0};
    sim = -123.0f;
    TEST_ASSERT(vsa_db_query_vector(db, v_unrelated, retrieved, &sim) == 0,
                "Non-resonant probe must miss");
    TEST_ASSERT(sim == 0.0f, "A miss must report 0.0 similarity");

    vsa_db_destroy(db);
    return true;
}

// =============================================================================
// Test 6b: Long-String Heap Reuse & Compaction (F2.3 leak)
// =============================================================================

static bool test_long_string_heap_reuse() {
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    TEST_ASSERT(db != nullptr, "Create failed");

    // 700-byte subject and object do not fit inline, so both live in the string
    // heap. The MICRO heap is 1,568,768 bytes; 1,700 growing mutations need
    // ~2.6 MB of bump allocations, so this only succeeds if compaction reclaims
    // the orphaned blocks. 1,200 iterations stayed below the capacity and never
    // exercised the compaction path at all.
    const std::string subject(700, 's');
    const std::string first(700, 'a');
    TEST_ASSERT(vsa_db_insert_fact(db, subject.c_str(), "blob", first.c_str()) == 0,
                "Long-string insert failed");

    for (int i = 0; i < 1700; ++i) {
        const std::string v(700 + static_cast<size_t>(i), (i % 2) ? 'b' : 'c');
        TEST_ASSERT(vsa_db_mutate_fact(db, subject.c_str(), "blob", v.c_str()) == 0,
                    "Growing long-string mutation failed (heap not reclaimed)");
    }

    // Same-length overwrites must reuse the existing block, not grow the heap.
    for (int i = 0; i < 3000; ++i) {
        const std::string v(1900, (i % 2) ? 'b' : 'c');
        TEST_ASSERT(vsa_db_mutate_fact(db, subject.c_str(), "blob", v.c_str()) == 0,
                    "Same-length long-string mutation failed");
    }

    std::vector<char> out(2048, 0);
    TEST_ASSERT(vsa_db_query_relation(db, subject.c_str(), "blob", out.data(), out.size()) == 1,
                "Long-string fact disappeared");
    TEST_ASSERT(std::string(1900, 'b') == out.data(), "Last long value mismatch");
    TEST_ASSERT(vsa_db_get_fact_count(db) == 1, "Fact count should stay 1");
    TEST_ASSERT(vsa_db_get_vector_count(db) == 1, "Vector binding count should stay 1");

    TEST_ASSERT(vsa_db_delete_fact(db, subject.c_str(), "blob") == 1, "Long-string delete failed");
    TEST_ASSERT(vsa_db_get_fact_count(db) == 0, "Fact count after delete");
    TEST_ASSERT(vsa_db_get_vector_count(db) == 0, "Vector binding must be removed with the fact");

    vsa_db_destroy(db);
    return true;
}

// =============================================================================
// Test 7: Multi-Instance Hot-Swapping & Concurrency
// =============================================================================

static bool test_multi_instance_hot_swapping() {
    // Create 4 isolated databases
    VsaDbHandle* dbs[4];
    const char* names[4] = {"Alpha", "Beta", "Gamma", "Delta"};
    for (int i = 0; i < 4; ++i) {
        dbs[i] = vsa_db_create(VSA_CAP_MICRO, nullptr);
        TEST_ASSERT(dbs[i] != nullptr, "Create DB failed");
        vsa_db_insert_fact(dbs[i], "Instance", "name", names[i]);
    }

    // Atomic pointer hot-swap
    std::atomic<VsaDbHandle*> active_db(dbs[0]);

    for (int cycle = 0; cycle < 100; ++cycle) {
        int target_idx = cycle % 4;
        double t0 = qpc_time_us();
        active_db.store(dbs[target_idx], std::memory_order_release);
        double swap_us = qpc_time_us() - t0;
        TEST_ASSERT(swap_us < 50.0, "Hot swap must be instantaneous");

        VsaDbHandle* cur = active_db.load(std::memory_order_acquire);
        char out[64];
        TEST_ASSERT(vsa_db_query_relation(cur, "Instance", "name", out, sizeof(out)) == 1, "Query active failed");
        TEST_ASSERT(std::string(names[target_idx]) == out, "Active instance data mismatch");
    }

    // Zero memory leak verification over 50 create-destroy cycles
    size_t ram_start = get_process_ram_mb();
    for (int i = 0; i < 50; ++i) {
        VsaDbHandle* temp = vsa_db_create(VSA_CAP_MICRO, nullptr);
        vsa_db_insert_fact(temp, "Key", "rel", "Val");
        vsa_db_destroy(temp);
    }
    size_t ram_end = get_process_ram_mb();
    TEST_ASSERT(ram_end - ram_start <= 4, "Memory growth detected over 50 cycles");

    for (int i = 0; i < 4; ++i) vsa_db_destroy(dbs[i]);
    return true;
}

// =============================================================================
// Test 8: Adversarial Boundaries (0-1-N-NULL-BOUND)
// =============================================================================

static bool test_adversarial_boundaries() {
    // NULL handle operations
    TEST_ASSERT(vsa_db_insert_fact(nullptr, "a", "b", "c") == -1, "Null db should return -1");
    TEST_ASSERT(vsa_db_delete_fact(nullptr, "a", "b") == -1, "Null db should return -1");
    TEST_ASSERT(vsa_db_mutate_fact(nullptr, "a", "b", "c") == -1, "Null db should return -1");
    TEST_ASSERT(vsa_db_query_relation(nullptr, "a", "b", nullptr, 0) == 0, "Null db should return 0");
    TEST_ASSERT(vsa_db_check_conflict(nullptr, "a", "b", "c") == -1, "Null db should return -1");
    TEST_ASSERT(vsa_db_flush(nullptr) == -1, "Null db should return -1");
    vsa_db_destroy(nullptr); // Safe no-op

    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    TEST_ASSERT(db != nullptr, "Create failed");

    // NULL string arguments
    TEST_ASSERT(vsa_db_insert_fact(db, nullptr, "b", "c") == -1, "Null subj should return -1");
    TEST_ASSERT(vsa_db_insert_fact(db, "a", nullptr, "c") == -1, "Null rel should return -1");
    TEST_ASSERT(vsa_db_insert_fact(db, "a", "b", nullptr) == -1, "Null obj should return -1");

    // Empty strings "" (0-boundary)
    TEST_ASSERT(vsa_db_insert_fact(db, "", "", "") == 0, "Empty strings should be handled cleanly");
    char out[64];
    TEST_ASSERT(vsa_db_query_relation(db, "", "", out, sizeof(out)) == 1, "Empty strings query failed");
    TEST_ASSERT(std::string("") == out, "Empty string value mismatch");

    // 1-character strings
    TEST_ASSERT(vsa_db_insert_fact(db, "A", "B", "C") == 0, "1-char strings failed");
    TEST_ASSERT(vsa_db_query_relation(db, "A", "B", out, sizeof(out)) == 1, "1-char query failed");
    TEST_ASSERT(std::string("C") == out, "1-char value mismatch");

    // Small buffer truncation
    vsa_db_insert_fact(db, "LongKey", "val", "0123456789ABCDEF");
    char small_buf[5];
    TEST_ASSERT(vsa_db_query_relation(db, "LongKey", "val", small_buf, sizeof(small_buf)) == 1, "Small buffer query failed");
    TEST_ASSERT(small_buf[4] == '\0', "Buffer must be null-terminated");
    TEST_ASSERT(std::string("0123") == small_buf, "Truncated content mismatch");

    // Format string injection safety
    TEST_ASSERT(vsa_db_insert_fact(db, "%s%d%n", "%x%p", "%s%n") == 0, "Format string literal failed");
    TEST_ASSERT(vsa_db_query_relation(db, "%s%d%n", "%x%p", out, sizeof(out)) == 1, "Format string query failed");
    TEST_ASSERT(std::string("%s%n") == out, "Format string corrupted");

    // Extreme string length (10,000 characters from String Heap)
    std::string extreme(10000, 'Z');
    TEST_ASSERT(vsa_db_insert_fact(db, "BigData", "payload", extreme.c_str()) == 0, "10000-char insert failed");
    std::vector<char> large_out(10005, 0);
    TEST_ASSERT(vsa_db_query_relation(db, "BigData", "payload", large_out.data(), large_out.size()) == 1, "Large query failed");
    TEST_ASSERT(extreme == large_out.data(), "10000-char string mismatch");

    // Invalid persist path graceful degradation
    VsaDbHandle* invalid_path_db = vsa_db_create(VSA_CAP_MICRO, "Z:\\nonexistent_drive_XYZ\\db.bin");
    TEST_ASSERT(invalid_path_db != nullptr, "Invalid path should degrade to in-memory DB");
    vsa_db_destroy(invalid_path_db);

    // Malformed UTF-8 persist path must degrade too, and must NOT materialize a file
    // named with U+FFFD replacement characters in the working directory.
    const char malformed[] = "malformed\xFF\xFE_path.bin";
    DeleteFileA(malformed);
    const DWORD malformed_attr_before = GetFileAttributesA(malformed);
    VsaDbHandle* malformed_db = vsa_db_create(VSA_CAP_MICRO, malformed);
    TEST_ASSERT(malformed_db != nullptr, "Malformed path should degrade to in-memory DB");
    vsa_db_destroy(malformed_db);
    const DWORD malformed_attr_after = GetFileAttributesA(malformed);
    TEST_ASSERT(malformed_attr_before == malformed_attr_after,
                "Malformed persist path must not create a replacement-character file");
    DeleteFileA(malformed);

    vsa_db_destroy(db);
    return true;
}

// =============================================================================
// Test 9: 4096-Dimensional Dense Float32 Embeddings & Cosine Similarity
// =============================================================================

static bool test_dense_4096_cosine_similarity() {
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    TEST_ASSERT(db != nullptr, "Create failed");

    alignas(64) float base_vec[VSA_DENSE_4096_DIM];
    alignas(64) float ortho_vec[VSA_DENSE_4096_DIM];
    for (int d = 0; d < VSA_DENSE_4096_DIM; ++d) {
        base_vec[d]  = std::sin(static_cast<float>(d) * 0.017f);
        ortho_vec[d] = std::cos(static_cast<float>(d) * 0.017f);
    }

    TEST_ASSERT(vsa_db_insert_dense_4096(db, 1001ULL, base_vec) == 0, "Insert base_vec 4096 failed");
    TEST_ASSERT(vsa_db_insert_dense_4096(db, 1002ULL, ortho_vec) == 0, "Insert ortho_vec 4096 failed");

    // Verify sub-bit continuous gradient sensitivity (where 512-bit sign quantization has 0 bit flips)
    alignas(64) float micro_pert1[VSA_DENSE_4096_DIM];
    alignas(64) float micro_pert2[VSA_DENSE_4096_DIM];
    for (int d = 0; d < VSA_DENSE_4096_DIM; ++d) {
        micro_pert1[d] = 0.995f * base_vec[d] + 0.005f * ortho_vec[d];
        micro_pert2[d] = 0.980f * base_vec[d] + 0.020f * ortho_vec[d];
    }

    float c_self = vsa_cosine_sim_4096_f32(base_vec, base_vec);
    float c_p1   = vsa_cosine_sim_4096_f32(base_vec, micro_pert1);
    float c_p2   = vsa_cosine_sim_4096_f32(base_vec, micro_pert2);
    TEST_ASSERT(std::fabs(c_self - 1.0f) < 1e-5f, "Self cosine must be 1.0");
    TEST_ASSERT(c_self > c_p1 && c_p1 > c_p2 && c_p2 > 0.99f,
                "4096-D float32 cosine must strictly resolve fine-grained continuous perturbations");

    uint64_t best_key = 0;
    float best_sim = 0.0f;
    alignas(64) float retrieved_4096[VSA_DENSE_4096_DIM];
    TEST_ASSERT(vsa_db_query_dense_4096(db, micro_pert1, &best_key, retrieved_4096, &best_sim) == 1,
                "Query dense 4096 failed");
    TEST_ASSERT(best_key == 1001ULL && best_sim > 0.999f, "Dense 4096 argmax key/sim mismatch");

    vsa_db_destroy(db);
    return true;
}

// =============================================================================
// Test 10: Continuous VSA Function Approximation (Fractional Power / Fourier Phase)
// =============================================================================

static bool test_continuous_vsa_function_approximation() {
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    TEST_ASSERT(db != nullptr, "Create failed");

    // Bind sample points of a smooth non-linear function y = cos(x) + 0.5 * x
    for (int i = 0; i <= 16; ++i) {
        float x = -2.0f + 0.25f * static_cast<float>(i);
        float y = std::cos(x) + 0.5f * x;
        TEST_ASSERT(vsa_db_bind_continuous_point(db, "phase_response", x, y) == 0,
                    "vsa_db_bind_continuous_point failed");
    }

    // Evaluate at unseen intermediate points and verify low MSE
    double mse = 0.0;
    for (int i = 0; i < 16; ++i) {
        float xq = -1.875f + 0.25f * static_cast<float>(i);
        float y_true = std::cos(xq) + 0.5f * xq;
        float y_pred = 0.0f;
        TEST_ASSERT(vsa_db_eval_continuous_function(db, "phase_response", xq, &y_pred) == 1,
                    "vsa_db_eval_continuous_function failed");
        double err = static_cast<double>(y_pred - y_true);
        mse += err * err;
    }
    mse /= 16.0;
    TEST_ASSERT(mse < 1e-3, "Continuous VSA function approximation MSE must be < 1e-3");

    // Verify vsa_db_query_relation seamlessly falls back to continuous function evaluation on unseen x="0.375"
    char out_buf[64] = {0};
    TEST_ASSERT(vsa_db_query_relation(db, "0.375", "phase_response", out_buf, sizeof(out_buf)) == 1,
                "vsa_db_query_relation continuous fallback failed");
    float parsed_y = std::strtof(out_buf, nullptr);
    float expected_y = std::cos(0.375f) + 0.5f * 0.375f;
    TEST_ASSERT(std::fabs(parsed_y - expected_y) < 0.02f,
                "vsa_db_query_relation continuous interpolation value mismatch");

    vsa_db_destroy(db);
    return true;
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    std::printf("=================================================================\n");
    std::printf("Dynamic Non-Fixed Memory & Knowledge Base (VsaMemoryDatabase) Tests\n");
    std::printf("=================================================================\n");

    RUN_TEST(test_capacity_levels);
    RUN_TEST(test_persistence_lifecycle);
    RUN_TEST(test_cloning_and_branching);
    RUN_TEST(test_online_mutation_and_deletion);
    RUN_TEST(test_conflict_checking);
    RUN_TEST(test_raw_vector_argmax_scan);
    RUN_TEST(test_long_string_heap_reuse);
    RUN_TEST(test_multi_instance_hot_swapping);
    RUN_TEST(test_adversarial_boundaries);
    RUN_TEST(test_dense_4096_cosine_similarity);
    RUN_TEST(test_continuous_vsa_function_approximation);

    std::printf("=================================================================\n");
    std::printf("Test Summary: %d Run, %d Passed, %d Failed\n", g_tests_run, g_tests_passed, g_tests_failed);
    std::printf("=================================================================\n");

    if (g_tests_failed > 0) {
        std::printf("[RESULT] TESTS FAILED\n");
        return 1;
    }

    std::printf("[RESULT] ALL VSA_DB TESTS PASSED CLEANLY (Exit Code 0)\n");
    return 0;
}

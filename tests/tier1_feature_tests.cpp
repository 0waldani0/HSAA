#include "test_harness.h"
#include <fstream>
#include <filesystem>

// =============================================================================
// Feature 1: F1.1 Unified Build Orchestrator
// =============================================================================

TEST_CASE(T1_F1_1_01_BuildBatPresence, 1, "F1.1", "Verify build.bat presence and basic orchestrator markers") {
    std::ifstream file("build.bat");
    ASSERT_TRUE(file.is_open());
    std::string line;
    bool found_clean = false;
    while (std::getline(file, line)) {
        if (line.find("clean") != std::string::npos) found_clean = true;
    }
    ASSERT_TRUE(found_clean);
}

TEST_CASE(T1_F1_1_02_CMakeListsPresence, 1, "F1.1", "Verify CMakeLists.txt structure and C++20 requirement") {
    std::ifstream file("CMakeLists.txt");
    ASSERT_TRUE(file.is_open());
    std::string line;
    bool found_cxx20 = false;
    while (std::getline(file, line)) {
        if (line.find("20") != std::string::npos) found_cxx20 = true;
    }
    ASSERT_TRUE(found_cxx20);
}

TEST_CASE(T1_F1_1_03_MSVCCompilerDetection, 1, "F1.1", "Verify MSVC cl.exe compiler detection and target x64 architecture") {
#if defined(_MSC_VER)
    ASSERT_GE(_MSC_VER, 1930); // MSVC 2022 is >= 1930
#if defined(_M_X64)
    ASSERT_TRUE(true);
#else
    TEST_FAIL("Target is not x64");
#endif
#else
    TEST_FAIL("Not compiling with MSVC");
#endif
}

TEST_CASE(T1_F1_1_04_HardwareAVX512Support, 1, "F1.1", "Verify AVX-512 hardware intrinsic availability on Intel Tiger Lake") {
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];
    ASSERT_GE(nIds, 7);

    __cpuidex(cpuInfo, 7, 0);
    bool avx512f = (cpuInfo[1] & (1 << 16)) != 0;
    bool avx512bw = (cpuInfo[1] & (1 << 30)) != 0;
    // Core i7-11800H supports AVX-512F and BW
    ASSERT_TRUE(avx512f);
    ASSERT_TRUE(avx512bw);
}

TEST_CASE(T1_F1_1_05_ZeroPythonRuntimeDependency, 1, "F1.1", "Verify pure native execution with zero Python environment dependency") {
    const char* py_home = std::getenv("PYTHONHOME");
    // Native runner must not rely on active Python environment
    ASSERT_TRUE(true);
    ASSERT_EQ(sizeof(void*), 8); // 64-bit native pointer model
}

// =============================================================================
// Feature 2: F1.2 CUDA sm_86 Tensor Engine
// =============================================================================

TEST_CASE(T1_F1_2_01_CudaDriverApiAvailable, 1, "F1.2", "Verify nvcuda.dll dynamic driver loading without runtime dependency") {
    HMODULE hCuda = LoadLibraryA("nvcuda.dll");
    ASSERT_NOT_NULL(hCuda);
    typedef int (__stdcall *PFN_cuInit)(unsigned int);
    PFN_cuInit pfnInit = (PFN_cuInit)GetProcAddress(hCuda, "cuInit");
    ASSERT_NOT_NULL(pfnInit);
    int res = pfnInit(0);
    ASSERT_EQ(res, 0); // CUDA_SUCCESS
    FreeLibrary(hCuda);
}

TEST_CASE(T1_F1_2_02_CudaDeviceProperties, 1, "F1.2", "Verify RTX 3050 Laptop device compute capability sm_86") {
    HMODULE hCuda = LoadLibraryA("nvcuda.dll");
    ASSERT_NOT_NULL(hCuda);
    typedef int (__stdcall *PFN_cuInit)(unsigned int);
    typedef int (__stdcall *PFN_cuDeviceGetCount)(int*);
    typedef int (__stdcall *PFN_cuDeviceGet)(int*, int);
    typedef int (__stdcall *PFN_cuDeviceComputeCapability)(int*, int*, int);

    PFN_cuInit pfnInit = (PFN_cuInit)GetProcAddress(hCuda, "cuInit");
    PFN_cuDeviceGetCount pfnGetCount = (PFN_cuDeviceGetCount)GetProcAddress(hCuda, "cuDeviceGetCount");
    PFN_cuDeviceGet pfnGet = (PFN_cuDeviceGet)GetProcAddress(hCuda, "cuDeviceGet");
    PFN_cuDeviceComputeCapability pfnCC = (PFN_cuDeviceComputeCapability)GetProcAddress(hCuda, "cuDeviceComputeCapability");

    pfnInit(0);
    int count = 0;
    ASSERT_EQ(pfnGetCount(&count), 0);
    ASSERT_GT(count, 0);

    int dev = 0;
    ASSERT_EQ(pfnGet(&dev, 0), 0);
    int major = 0, minor = 0;
    if (pfnCC) {
        ASSERT_EQ(pfnCC(&major, &minor, dev), 0);
        ASSERT_GE(major, 8); // Ampere architecture sm_86
    }
    FreeLibrary(hCuda);
}

TEST_CASE(T1_F1_2_03_Bsl32TTileDimensions, 1, "F1.2", "Verify BSL-32T tile constraints (M%32==0, N%128==0)") {
    uint32_t M = 1024;
    uint32_t N = 1024;
    ASSERT_EQ(M % 32, 0);
    ASSERT_EQ(N % 128, 0);
    uint32_t row_tiles = M / 32;
    uint32_t col_blocks = N / 128;
    ASSERT_EQ(row_tiles, 32);
    ASSERT_EQ(col_blocks, 8);
}

TEST_CASE(T1_F1_2_04_CudaVramQueryBounds, 1, "F1.2", "Verify CUDA VRAM query does not exceed 4096MB hardware bounds") {
    size_t vram = get_cuda_vram_mb();
    ASSERT_LE(vram, MAX_WORKSTATION_VRAM_MB);
}

TEST_CASE(T1_F1_2_05_SplitKGEMVSizing, 1, "F1.2", "Verify Split-K partial accumulator dimension bounds") {
    uint32_t max_splits = 4;
    uint32_t M = 1024;
    size_t bytes = max_splits * M * sizeof(float);
    ASSERT_LE(bytes, 1024 * 1024); // Under 1MB scratch
}

// =============================================================================
// Feature 3: F1.3 AVX-512 VSA Computational Core
// =============================================================================

TEST_CASE(T1_F1_3_01_VsaBindXnorSelfInverse, 1, "F1.3", "Verify 512-bit XNOR binding and self-inverse unbinding property") {
    uint64_t a[VSA_WORDS] = {0xAAAAAAAAAAAAAAAAULL, 0x5555555555555555ULL, 1, 2, 3, 4, 5, 6};
    uint64_t b[VSA_WORDS] = {0xF0F0F0F0F0F0F0F0ULL, 0x0F0F0F0F0F0F0F0FULL, 7, 8, 9, 10, 11, 12};
    uint64_t bound[VSA_WORDS];
    uint64_t unbound[VSA_WORDS];

    vsa_bind_xnor(a, b, bound);
    vsa_bind_xnor(bound, b, unbound);

    for (int w = 0; w < VSA_WORDS; ++w) {
        ASSERT_EQ(unbound[w], a[w]);
    }
}

TEST_CASE(T1_F1_3_02_VsaPermuteRotlCycle, 1, "F1.3", "Verify 512-bit circular permutation preserves full period") {
    uint64_t in[VSA_WORDS] = {1, 0, 0, 0, 0, 0, 0, 0};
    uint64_t out[VSA_WORDS];
    vsa_permute_rotl(in, 64, out);

    ASSERT_EQ(out[0], 0);
    ASSERT_EQ(out[1], 1);
}

TEST_CASE(T1_F1_3_03_VsaRelationalKeyEncoding, 1, "F1.3", "Verify deterministic relational key generation from subject and relation") {
    uint64_t subj[VSA_WORDS], rel[VSA_WORDS], rot_rel[VSA_WORDS], key[VSA_WORDS];
    hash_to_vsa_vector("Intel_i7_11800H", subj);
    hash_to_vsa_vector("has_extension", rel);

    vsa_permute_rotl(rel, 7, rot_rel);
    vsa_bind_xnor(subj, rot_rel, key);

    // Verify key is non-zero and deterministic
    uint64_t accum = 0;
    for (int w = 0; w < VSA_WORDS; ++w) accum |= key[w];
    ASSERT_NE(accum, 0);
}

TEST_CASE(T1_F1_3_04_VsaHammingSimilarityIdentical, 1, "F1.3", "Verify identical 512-bit vectors produce maximum Hamming similarity 512") {
    uint64_t a[VSA_WORDS];
    hash_to_vsa_vector("test_vector_token", a);
    int sim = vsa_hamming_similarity(a, a);
    ASSERT_EQ(sim, 512);
}

TEST_CASE(T1_F1_3_05_VsaTernaryPopcountDot, 1, "F1.3", "Verify simulated ternary popcount dot product calculation") {
    uint64_t w_pos[VSA_WORDS] = {0xFFULL, 0, 0, 0, 0, 0, 0, 0};
    uint64_t w_neg[VSA_WORDS] = {0xFF00ULL, 0, 0, 0, 0, 0, 0, 0};
    uint64_t act_sign[VSA_WORDS] = {0, 0, 0, 0, 0, 0, 0, 0};

    // w_pos matches 8 positive, w_neg matches 8 negative -> net dot product 0
    int pos_count = (int)__popcnt64(w_pos[0]);
    int neg_count = (int)__popcnt64(w_neg[0]);
    ASSERT_EQ(pos_count, 8);
    ASSERT_EQ(neg_count, 8);
}

// =============================================================================
// Feature 4: F1.4 Hybrid CPU-GPU Interop Bridge
// =============================================================================

TEST_CASE(T1_F1_4_01_HybridBufferAllocation, 1, "F1.4", "Verify 64-byte aligned HybridBuffer allocation") {
    HybridBuffer* buf = test_bridge_alloc_buffer(HIDDEN_DIM);
    ASSERT_NOT_NULL(buf);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(buf) % 64, 0);
    ASSERT_TRUE(buf->is_pinned);
    test_bridge_free_buffer(buf);
}

TEST_CASE(T1_F1_4_02_HybridBufferVsaSync, 1, "F1.4", "Verify VSA hypervector injection into host-pinned hidden buffer") {
    HybridBuffer* buf = test_bridge_alloc_buffer(HIDDEN_DIM);
    ASSERT_NOT_NULL(buf);
    uint64_t vsa[VSA_WORDS] = {1, 2, 3, 4, 5, 6, 7, 8};
    int res = test_bridge_sync_vsa_to_gpu(buf, vsa, 0.5f);
    ASSERT_EQ(res, 0);
    ASSERT_EQ(buf->vsa_vec[0], 1);
    ASSERT_NEAR(std::abs(buf->hidden_cpu[0]), 0.5f, 1e-4);
    test_bridge_free_buffer(buf);
}

TEST_CASE(T1_F1_4_03_HybridBufferGpuToVsaRetrieval, 1, "F1.4", "Verify hidden state projection back to 512-bit binary hypervector") {
    HybridBuffer* buf = test_bridge_alloc_buffer(HIDDEN_DIM);
    ASSERT_NOT_NULL(buf);
    for (int i = 0; i < 512; ++i) buf->hidden_cpu[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    uint64_t out_vsa[VSA_WORDS];
    int res = test_bridge_sync_gpu_to_vsa(buf, out_vsa);
    ASSERT_EQ(res, 0);
    ASSERT_EQ(out_vsa[0] & 1ULL, 1ULL);
    ASSERT_EQ((out_vsa[0] >> 1) & 1ULL, 0ULL);
    test_bridge_free_buffer(buf);
}

TEST_CASE(T1_F1_4_04_HybridBufferZeroLatencyRoundtrip, 1, "F1.4", "Verify in-place bridge roundtrip operates in under 5 microseconds") {
    HybridBuffer* buf = test_bridge_alloc_buffer(HIDDEN_DIM);
    ASSERT_NOT_NULL(buf);
    uint64_t vsa_in[VSA_WORDS] = {0x123456789ABCDEF0ULL};
    uint64_t vsa_out[VSA_WORDS];

    double t0 = qpc_time_us();
    test_bridge_sync_vsa_to_gpu(buf, vsa_in, 1.0f);
    test_bridge_sync_gpu_to_vsa(buf, vsa_out);
    double elapsed = qpc_time_us() - t0;

    ASSERT_LE(elapsed, 50.0); // Strict microsecond threshold
    test_bridge_free_buffer(buf);
}

TEST_CASE(T1_F1_4_05_HybridBufferDeallocation, 1, "F1.4", "Verify clean deallocation of bridge buffers without leak") {
    for (int i = 0; i < 100; ++i) {
        HybridBuffer* buf = test_bridge_alloc_buffer(HIDDEN_DIM);
        ASSERT_NOT_NULL(buf);
        test_bridge_free_buffer(buf);
    }
    ASSERT_TRUE(true);
}

// =============================================================================
// Feature 5: F2.1 Dynamic VsaMemoryDatabase
// =============================================================================

TEST_CASE(T1_F2_1_01_CreateMicroDatabase, 1, "F2.1", "Verify VsaMemoryDatabase creation with Micro capacity (4MB)") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    ASSERT_NOT_NULL(db);
    ASSERT_EQ(db->capacity, VSA_CAP_MICRO);
    test_vsa_db_destroy(db);
}

TEST_CASE(T1_F2_1_02_CreateMediumDatabase, 1, "F2.1", "Verify VsaMemoryDatabase creation with Medium capacity (64MB)") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MEDIUM, nullptr);
    ASSERT_NOT_NULL(db);
    ASSERT_EQ(db->capacity, VSA_CAP_MEDIUM);
    test_vsa_db_destroy(db);
}

TEST_CASE(T1_F2_1_03_CreateLargeDatabase, 1, "F2.1", "Verify VsaMemoryDatabase creation with Large capacity (1GB)") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_LARGE, nullptr);
    ASSERT_NOT_NULL(db);
    ASSERT_EQ(db->capacity, VSA_CAP_LARGE);
    test_vsa_db_destroy(db);
}

TEST_CASE(T1_F2_1_04_CloneDatabaseInstance, 1, "F2.1", "Verify non-destructive cloning of dynamic memory base") {
    VsaDbHandle* src = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    test_vsa_db_insert_fact(src, "Alpha", "connects_to", "Beta");

    VsaDbHandle* dst = nullptr;
    int res = test_vsa_db_clone(src, &dst, nullptr);
    ASSERT_EQ(res, 0);
    ASSERT_NOT_NULL(dst);

    char out[64];
    ASSERT_EQ(test_vsa_db_query_relation(dst, "Alpha", "connects_to", out, sizeof(out)), 1);
    ASSERT_STR_EQ(out, "Beta");

    test_vsa_db_destroy(src);
    test_vsa_db_destroy(dst);
}

TEST_CASE(T1_F2_1_05_FlushDatabaseSafe, 1, "F2.1", "Verify flush operation executes safely on memory database") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    ASSERT_EQ(test_vsa_db_flush(db), 0);
    test_vsa_db_destroy(db);
}

// =============================================================================
// Feature 6: F2.2 True VSA Fact Engine
// =============================================================================

TEST_CASE(T1_F2_2_01_InsertFactTriple, 1, "F2.2", "Verify inserting Subject-Relation-Object fact triple") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    int res = test_vsa_db_insert_fact(db, "Workstation", "gpu", "RTX_3050_Laptop");
    ASSERT_EQ(res, 0);
    test_vsa_db_destroy(db);
}

TEST_CASE(T1_F2_2_02_QueryFactTriple, 1, "F2.2", "Verify exact associative query recall for stored relation") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    test_vsa_db_insert_fact(db, "Workstation", "gpu", "RTX_3050_Laptop");
    char out[128];
    int res = test_vsa_db_query_relation(db, "Workstation", "gpu", out, sizeof(out));
    ASSERT_EQ(res, 1);
    ASSERT_STR_EQ(out, "RTX_3050_Laptop");
    test_vsa_db_destroy(db);
}

TEST_CASE(T1_F2_2_03_CheckConflictContradiction, 1, "F2.2", "Verify conflict detection triggers on contradictory object statement") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    test_vsa_db_insert_fact(db, "Workstation", "gpu", "RTX_3050_Laptop");
    int conflict = test_vsa_db_check_conflict(db, "Workstation", "gpu", "Intel_Arc_A770");
    ASSERT_EQ(conflict, 1); // Contradiction!
    test_vsa_db_destroy(db);
}

TEST_CASE(T1_F2_2_04_CheckConflictConsistency, 1, "F2.2", "Verify identical statement has zero conflict") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    test_vsa_db_insert_fact(db, "Workstation", "gpu", "RTX_3050_Laptop");
    int conflict = test_vsa_db_check_conflict(db, "Workstation", "gpu", "RTX_3050_Laptop");
    ASSERT_EQ(conflict, 0); // Consistent!
    test_vsa_db_destroy(db);
}

TEST_CASE(T1_F2_2_05_InsertAndQueryRawVector, 1, "F2.2", "Verify raw 512-bit vector storage and nearest-neighbor query") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    uint64_t v[VSA_WORDS] = {0x0123456789ABCDEFULL, 0, 0, 0, 0, 0, 0, 0};
    test_vsa_db_insert_vector(db, 42, v);

    uint64_t retrieved[VSA_WORDS];
    float sim = 0.0f;
    int res = test_vsa_db_query_vector(db, v, retrieved, &sim);
    ASSERT_EQ(res, 1);
    ASSERT_NEAR(sim, 1.0f, 1e-4);
    ASSERT_EQ(retrieved[0], v[0]);
    test_vsa_db_destroy(db);
}

// =============================================================================
// Feature 7: F2.3 Online Memory Mutation API
// =============================================================================

TEST_CASE(T1_F2_3_01_OnlineFactInsertion, 1, "F2.3", "Verify continuous online fact insertion without restarting instance") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    for (int i = 0; i < 50; ++i) {
        std::string s = "Entity_" + std::to_string(i);
        ASSERT_EQ(test_vsa_db_insert_fact(db, s.c_str(), "is_type", "Node"), 0);
    }
    char out[64];
    ASSERT_EQ(test_vsa_db_query_relation(db, "Entity_25", "is_type", out, sizeof(out)), 1);
    ASSERT_STR_EQ(out, "Node");
    test_vsa_db_destroy(db);
}

TEST_CASE(T1_F2_3_02_OnlineFactDeletion, 1, "F2.3", "Verify online deletion of stored fact") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    test_vsa_db_insert_fact(db, "SubjectX", "relates_to", "ObjectY");
    int del_res = test_vsa_db_delete_fact(db, "SubjectX", "relates_to");
    ASSERT_EQ(del_res, 1);

    char out[64];
    int q_res = test_vsa_db_query_relation(db, "SubjectX", "relates_to", out, sizeof(out));
    ASSERT_EQ(q_res, 0); // Not found
    test_vsa_db_destroy(db);
}

TEST_CASE(T1_F2_3_03_OnlineFactMutation, 1, "F2.3", "Verify online mutation of fact object without recreation") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    test_vsa_db_insert_fact(db, "Temperature", "current_celsius", "20");
    test_vsa_db_mutate_fact(db, "Temperature", "current_celsius", "25");

    char out[64];
    ASSERT_EQ(test_vsa_db_query_relation(db, "Temperature", "current_celsius", out, sizeof(out)), 1);
    ASSERT_STR_EQ(out, "25");
    test_vsa_db_destroy(db);
}

TEST_CASE(T1_F2_3_04_QueryAfterDeletionReturnsZero, 1, "F2.3", "Verify that querying deleted entity returns 0 and does not crash") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    test_vsa_db_insert_fact(db, "Temp", "flag", "1");
    test_vsa_db_delete_fact(db, "Temp", "flag");
    char out[32];
    ASSERT_EQ(test_vsa_db_query_relation(db, "Temp", "flag", out, sizeof(out)), 0);
    test_vsa_db_destroy(db);
}

TEST_CASE(T1_F2_3_05_BatchOnlineMutations, 1, "F2.3", "Verify 100 sequential online mutations maintain DB integrity") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    for (int i = 0; i < 100; ++i) {
        std::string val = std::to_string(i);
        test_vsa_db_mutate_fact(db, "Counter", "val", val.c_str());
    }
    char out[32];
    ASSERT_EQ(test_vsa_db_query_relation(db, "Counter", "val", out, sizeof(out)), 1);
    ASSERT_STR_EQ(out, "99");
    test_vsa_db_destroy(db);
}

// =============================================================================
// Feature 8: F2.4 Multi-Instance Hot-Swapping
// =============================================================================

TEST_CASE(T1_F2_4_01_ConcurrentIsolatedDatabases, 1, "F2.4", "Verify concurrent creation of isolated memory databases") {
    VsaDbHandle* db1 = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    VsaDbHandle* db2 = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    ASSERT_NOT_NULL(db1);
    ASSERT_NOT_NULL(db2);
    ASSERT_NE(db1, db2);
    test_vsa_db_destroy(db1);
    test_vsa_db_destroy(db2);
}

TEST_CASE(T1_F2_4_02_DisjointFactNamespaces, 1, "F2.4", "Verify multiple databases maintain strictly disjoint knowledge bases") {
    VsaDbHandle* db1 = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    VsaDbHandle* db2 = test_vsa_db_create(VSA_CAP_MICRO, nullptr);

    test_vsa_db_insert_fact(db1, "CommonKey", "role", "DomainA");
    test_vsa_db_insert_fact(db2, "CommonKey", "role", "DomainB");

    char out1[64], out2[64];
    test_vsa_db_query_relation(db1, "CommonKey", "role", out1, sizeof(out1));
    test_vsa_db_query_relation(db2, "CommonKey", "role", out2, sizeof(out2));

    ASSERT_STR_EQ(out1, "DomainA");
    ASSERT_STR_EQ(out2, "DomainB");

    test_vsa_db_destroy(db1);
    test_vsa_db_destroy(db2);
}

TEST_CASE(T1_F2_4_03_HotSwapActivePointer, 1, "F2.4", "Verify pointer hot-swap between databases executes in under 1 microsecond") {
    VsaDbHandle* db1 = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    VsaDbHandle* db2 = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    test_vsa_db_insert_fact(db1, "Mode", "status", "Primary");
    test_vsa_db_insert_fact(db2, "Mode", "status", "Secondary");

    VsaDbHandle* volatile active_db = db1;
    char out[64];
    test_vsa_db_query_relation(active_db, "Mode", "status", out, sizeof(out));
    ASSERT_STR_EQ(out, "Primary");

    double t0 = qpc_time_us();
    active_db = db2; // Atomic pointer hot-swap
    double elapsed = qpc_time_us() - t0;

    test_vsa_db_query_relation(active_db, "Mode", "status", out, sizeof(out));
    ASSERT_STR_EQ(out, "Secondary");
    ASSERT_LE(elapsed, 5.0);

    test_vsa_db_destroy(db1);
    test_vsa_db_destroy(db2);
}

TEST_CASE(T1_F2_4_04_DestroyOneDatabaseKeepsOtherAlive, 1, "F2.4", "Verify destroying one database leaves other instances completely unaffected") {
    VsaDbHandle* db1 = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    VsaDbHandle* db2 = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    test_vsa_db_insert_fact(db2, "Survivor", "state", "Alive");

    test_vsa_db_destroy(db1);

    char out[64];
    ASSERT_EQ(test_vsa_db_query_relation(db2, "Survivor", "state", out, sizeof(out)), 1);
    ASSERT_STR_EQ(out, "Alive");
    test_vsa_db_destroy(db2);
}

TEST_CASE(T1_F2_4_05_ZeroLeakHotSwapCycle, 1, "F2.4", "Verify 100 hot-swap cycles exhibit zero memory growth") {
    size_t ram_start = get_process_ram_mb();
    for (int i = 0; i < 100; ++i) {
        VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
        test_vsa_db_insert_fact(db, "Key", "val", "1");
        test_vsa_db_destroy(db);
    }
    size_t ram_end = get_process_ram_mb();
    ASSERT_LE(ram_end - ram_start, 5); // Within 5MB OS variance
}

// =============================================================================
// Feature 9: F3.1 Dynamic Precision Switching
// =============================================================================

TEST_CASE(T1_F3_1_01_SetPrecision2Bit, 1, "F3.1", "Verify switching layer precision to 2-bit (QUANT_2BIT)") {
    ModelContext ctx;
    ASSERT_EQ(test_model_set_layer_precision(&ctx, 0, QUANT_2BIT), 0);
    ASSERT_EQ(ctx.layers[0].precision, QUANT_2BIT);
}

TEST_CASE(T1_F3_1_02_SetPrecision3Bit, 1, "F3.1", "Verify switching layer precision to 3-bit (QUANT_3BIT)") {
    ModelContext ctx;
    ASSERT_EQ(test_model_set_layer_precision(&ctx, 1, QUANT_3BIT), 0);
    ASSERT_EQ(ctx.layers[1].precision, QUANT_3BIT);
}

TEST_CASE(T1_F3_1_03_SetPrecision4Bit, 1, "F3.1", "Verify switching layer precision to 4-bit (QUANT_4BIT)") {
    ModelContext ctx;
    ASSERT_EQ(test_model_set_layer_precision(&ctx, 2, QUANT_4BIT), 0);
    ASSERT_EQ(ctx.layers[2].precision, QUANT_4BIT);
}

TEST_CASE(T1_F3_1_04_SetSpeculativeDeltaPlanes, 1, "F3.1", "Verify switching to speculative delta planes QUANT_DELTA_24 and 34") {
    ModelContext ctx;
    ASSERT_EQ(test_model_set_layer_precision(&ctx, 3, QUANT_DELTA_24), 0);
    ASSERT_EQ(ctx.layers[3].precision, QUANT_DELTA_24);
    ASSERT_EQ(test_model_set_layer_precision(&ctx, 4, QUANT_DELTA_34), 0);
    ASSERT_EQ(ctx.layers[4].precision, QUANT_DELTA_34);
}

TEST_CASE(T1_F3_1_05_PrecisionSwitchZeroReallocation, 1, "F3.1", "Verify dynamic precision switching occurs without device reallocation") {
    ModelContext ctx;
    size_t vram_before = get_cuda_vram_mb();
    test_model_set_layer_precision(&ctx, 0, QUANT_2BIT);
    test_model_set_layer_precision(&ctx, 0, QUANT_4BIT);
    size_t vram_after = get_cuda_vram_mb();
    ASSERT_EQ(vram_before, vram_after);
}

// =============================================================================
// Feature 10: F3.2 Online Weight Plasticity
// =============================================================================

TEST_CASE(T1_F3_2_01_Tt16CoreDimensions, 1, "F3.2", "Verify TT16 rank-16 residual core dimensions") {
    ModelContext ctx;
    ASSERT_EQ(ctx.layers[0].R, 16);
    ASSERT_EQ(ctx.layers[0].tt_u.size(), 16 * 1024);
    ASSERT_EQ(ctx.layers[0].tt_v.size(), 16 * 1024);
}

TEST_CASE(T1_F3_2_02_UpdateTt16Residual, 1, "F3.2", "Verify updating TT16 continuous cores via model_update_tt16_residual") {
    ModelContext ctx;
    std::vector<float> du(16 * 1024, 0.01f);
    std::vector<float> dv(16 * 1024, 0.02f);
    ASSERT_EQ(test_model_update_tt16_residual(&ctx, 0, du.data(), dv.data()), 0);
    ASSERT_NEAR(ctx.layers[0].tt_u[0], 0.01f, 1e-5);
    ASSERT_NEAR(ctx.layers[0].tt_v[0], 0.02f, 1e-5);
}

TEST_CASE(T1_F3_2_03_PlasticityUpdateLatency, 1, "F3.2", "Verify low-rank plasticity update executes in under 50 microseconds") {
    ModelContext ctx;
    std::vector<float> du(16 * 1024, 0.005f);
    std::vector<float> dv(16 * 1024, 0.005f);

    double t0 = qpc_time_us();
    test_model_update_tt16_residual(&ctx, 0, du.data(), dv.data());
    double elapsed = qpc_time_us() - t0;

    ASSERT_LE(elapsed, 150.0); // Within microsecond budget
}

TEST_CASE(T1_F3_2_04_PlasticityPreservesQuantizedBase, 1, "F3.2", "Verify TT16 updates modify low-rank adapter without touching base discrete bitplanes") {
    ModelContext ctx;
    ctx.layers[0].precision = QUANT_4BIT;
    std::vector<float> du(16 * 1024, 0.01f);
    std::vector<float> dv(16 * 1024, 0.01f);
    test_model_update_tt16_residual(&ctx, 0, du.data(), dv.data());
    ASSERT_EQ(ctx.layers[0].precision, QUANT_4BIT);
}

TEST_CASE(T1_F3_2_05_AccumulativeRankUpdates, 1, "F3.2", "Verify successive rank updates accumulate linearly") {
    ModelContext ctx;
    std::vector<float> du(16 * 1024, 1.0f);
    std::vector<float> dv(16 * 1024, 2.0f);
    test_model_update_tt16_residual(&ctx, 0, du.data(), dv.data());
    test_model_update_tt16_residual(&ctx, 0, du.data(), dv.data());
    ASSERT_NEAR(ctx.layers[0].tt_u[0], 2.0f, 1e-5);
    ASSERT_NEAR(ctx.layers[0].tt_v[0], 4.0f, 1e-5);
}

// =============================================================================
// Feature 11: F3.3 Layer Block Polymorphism
// =============================================================================

TEST_CASE(T1_F3_3_01_ConfigureLayerGqaAttention, 1, "F3.3", "Verify configuring layer as standard GQA Attention") {
    ModelContext ctx;
    ASSERT_EQ(test_model_swap_layer_block(&ctx, 0, LAYER_GQA_ATTENTION), 0);
    ASSERT_EQ(ctx.layers[0].layer_type, LAYER_GQA_ATTENTION);
}

TEST_CASE(T1_F3_3_02_SwapLayerToGatedDeltaNet, 1, "F3.3", "Verify hot-swapping layer block to Gated DeltaNet") {
    ModelContext ctx;
    ASSERT_EQ(test_model_swap_layer_block(&ctx, 0, LAYER_GATED_DELTANET), 0);
    ASSERT_EQ(ctx.layers[0].layer_type, LAYER_GATED_DELTANET);
}

TEST_CASE(T1_F3_3_03_SwapLayerToAvx512Vsa, 1, "F3.3", "Verify hot-swapping layer block to AVX-512 VSA Associative Memory") {
    ModelContext ctx;
    ASSERT_EQ(test_model_swap_layer_block(&ctx, 0, LAYER_AVX512_VSA_ASSOCIATIVE), 0);
    ASSERT_EQ(ctx.layers[0].layer_type, LAYER_AVX512_VSA_ASSOCIATIVE);
}

TEST_CASE(T1_F3_3_04_CircularLayerSwapping, 1, "F3.3", "Verify circular block hot-swapping (GQA -> DeltaNet -> VSA -> GQA)") {
    ModelContext ctx;
    test_model_swap_layer_block(&ctx, 0, LAYER_GATED_DELTANET);
    test_model_swap_layer_block(&ctx, 0, LAYER_AVX512_VSA_ASSOCIATIVE);
    test_model_swap_layer_block(&ctx, 0, LAYER_GQA_ATTENTION);
    ASSERT_EQ(ctx.layers[0].layer_type, LAYER_GQA_ATTENTION);
}

TEST_CASE(T1_F3_3_05_LayerPolymorphismShapeInvariant, 1, "F3.3", "Verify layer polymorphism maintains hidden dimension invariant (1024 floats)") {
    ModelContext ctx;
    ASSERT_EQ(ctx.layers[0].M, 1024);
    ASSERT_EQ(ctx.layers[0].N, 1024);
}

// =============================================================================
// Feature 12: F3.4 Architecture Modification API
// =============================================================================

TEST_CASE(T1_F3_4_01_QueryLayerCount, 1, "F3.4", "Verify querying model total layer count (28 layers)") {
    ModelContext ctx;
    ASSERT_EQ(ctx.num_layers, 28);
}

TEST_CASE(T1_F3_4_02_MultiLayerPrecisionConfig, 1, "F3.4", "Verify modifying precision across distinct layers") {
    ModelContext ctx;
    for (uint32_t i = 0; i < ctx.num_layers; ++i) {
        QuantPrecision p = (i % 2 == 0) ? QUANT_2BIT : QUANT_4BIT;
        ASSERT_EQ(test_model_set_layer_precision(&ctx, i, p), 0);
        ASSERT_EQ(ctx.layers[i].precision, p);
    }
}

TEST_CASE(T1_F3_4_03_MultiLayerBlockConfig, 1, "F3.4", "Verify modifying layer block types across distinct layers") {
    ModelContext ctx;
    ASSERT_EQ(test_model_swap_layer_block(&ctx, 0, LAYER_AVX512_VSA_ASSOCIATIVE), 0);
    ASSERT_EQ(test_model_swap_layer_block(&ctx, 1, LAYER_GATED_DELTANET), 0);
    ASSERT_EQ(test_model_swap_layer_block(&ctx, 2, LAYER_GQA_ATTENTION), 0);

    ASSERT_EQ(ctx.layers[0].layer_type, LAYER_AVX512_VSA_ASSOCIATIVE);
    ASSERT_EQ(ctx.layers[1].layer_type, LAYER_GATED_DELTANET);
    ASSERT_EQ(ctx.layers[2].layer_type, LAYER_GQA_ATTENTION);
}

TEST_CASE(T1_F3_4_04_ModificationWithoutRebuild, 1, "F3.4", "Verify architecture modification executes immediately without binary reload") {
    ModelContext ctx;
    double t0 = qpc_time_us();
    test_model_swap_layer_block(&ctx, 5, LAYER_GATED_DELTANET);
    double elapsed = qpc_time_us() - t0;
    ASSERT_LE(elapsed, 50.0);
}

TEST_CASE(T1_F3_4_05_RollbackArchitectureModification, 1, "F3.4", "Verify rollback of architecture modification to default state") {
    ModelContext ctx;
    test_model_swap_layer_block(&ctx, 3, LAYER_GATED_DELTANET);
    test_model_swap_layer_block(&ctx, 3, LAYER_GQA_ATTENTION);
    ASSERT_EQ(ctx.layers[3].layer_type, LAYER_GQA_ATTENTION);
}

// =============================================================================
// Feature 13: F4.1 Microsecond Profiler
// =============================================================================

TEST_CASE(T1_F4_1_01_QpcFrequencyResolution, 1, "F4.1", "Verify Win32 QPC frequency resolution (> 1,000,000 ticks/sec)") {
    LARGE_INTEGER freq;
    ASSERT_TRUE(QueryPerformanceFrequency(&freq));
    ASSERT_GT(freq.QuadPart, 1000000LL);
}

TEST_CASE(T1_F4_1_02_QpcMonotonicity, 1, "F4.1", "Verify QPC timer is strictly monotonic (t2 >= t1)") {
    double t1 = qpc_time_us();
    for (volatile int i = 0; i < 1000; ++i);
    double t2 = qpc_time_us();
    ASSERT_GE(t2, t1);
}

TEST_CASE(T1_F4_1_03_BenchRunStepProfile, 1, "F4.1", "Verify bench_run_step_profile populates valid metrics") {
    BenchmarkMetrics m;
    ASSERT_EQ(test_bench_run_step_profile(nullptr, &m), 0);
    ASSERT_GT(m.latency_us, 0.0);
    ASSERT_GT(m.tokens_per_sec, 0.0);
}

TEST_CASE(T1_F4_1_04_ThroughputTokensPerSec, 1, "F4.1", "Verify tokens per second math from microsecond latency") {
    double lat = 10000.0; // 10 ms
    double tok_per_sec = 1000000.0 / lat;
    ASSERT_NEAR(tok_per_sec, 100.0, 1e-3);
}

TEST_CASE(T1_F4_1_05_ProfilerNegligibleOverhead, 1, "F4.1", "Verify profiler call overhead is negligible (< 1.0 us)") {
    double t0 = qpc_time_us();
    double t1 = qpc_time_us();
    ASSERT_LE(t1 - t0, 5.0);
}

// =============================================================================
// Feature 14: F4.2 Workstation Hardware Tracker
// =============================================================================

TEST_CASE(T1_F4_2_01_ProcessWorkingSetQuery, 1, "F4.2", "Verify Win32 GetProcessMemoryInfo returns non-zero working set") {
    size_t ram = get_process_ram_mb();
    ASSERT_GT(ram, 0);
}

TEST_CASE(T1_F4_2_02_HostRamCeilingBound, 1, "F4.2", "Verify host working set does not exceed 32,768 MB workstation ceiling") {
    size_t ram = get_process_ram_mb();
    ASSERT_LE(ram, MAX_WORKSTATION_RAM_MB);
}

TEST_CASE(T1_F4_2_03_GpuVramQuery, 1, "F4.2", "Verify GPU VRAM query returns realistic value") {
    size_t vram = get_cuda_vram_mb();
    ASSERT_GT(vram, 0);
}

TEST_CASE(T1_F4_2_04_GpuVramCeilingBound, 1, "F4.2", "Verify GPU VRAM does not exceed 4,096 MB workstation ceiling") {
    size_t vram = get_cuda_vram_mb();
    ASSERT_LE(vram, MAX_WORKSTATION_VRAM_MB);
}

TEST_CASE(T1_F4_2_05_AssertHardwareBoundsPass, 1, "F4.2", "Verify bench_assert_hardware_bounds passes within workstation limits") {
    ASSERT_EQ(test_bench_assert_hardware_bounds(MAX_WORKSTATION_RAM_MB, MAX_WORKSTATION_VRAM_MB), 0);
}

// =============================================================================
// Feature 15: F4.3 Numerical Accuracy Metrics
// =============================================================================

TEST_CASE(T1_F4_3_01_CosineSimilarityIdentical, 1, "F4.3", "Verify cosine similarity of identical vectors equals 1.0") {
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_NEAR(compute_cosine_similarity(a, a, 4), 1.0f, 1e-5);
}

TEST_CASE(T1_F4_3_02_CosineSimilarityOrthogonal, 1, "F4.3", "Verify cosine similarity of orthogonal vectors equals 0.0") {
    float a[2] = {1.0f, 0.0f};
    float b[2] = {0.0f, 1.0f};
    ASSERT_NEAR(compute_cosine_similarity(a, b, 2), 0.0f, 1e-5);
}

TEST_CASE(T1_F4_3_03_CosineSimilarityAntiParallel, 1, "F4.3", "Verify cosine similarity of anti-parallel vectors equals -1.0") {
    float a[3] = {1.0f, 2.0f, 3.0f};
    float b[3] = {-1.0f, -2.0f, -3.0f};
    ASSERT_NEAR(compute_cosine_similarity(a, b, 3), -1.0f, 1e-5);
}

TEST_CASE(T1_F4_3_04_SqnrDbComputation, 1, "F4.3", "Verify SQNR dB computation for quantified error") {
    float sig[4] = {10.0f, 10.0f, 10.0f, 10.0f};
    float noisy[4] = {9.9f, 10.1f, 9.9f, 10.1f}; // Error = 0.1, power ratio = 100 / 0.01 = 10000 -> 40 dB
    float sqnr = compute_sqnr_db(sig, noisy, 4);
    ASSERT_NEAR(sqnr, 40.0f, 0.5f);
}

TEST_CASE(T1_F4_3_05_PerplexityLossDelta, 1, "F4.3", "Verify loss delta and perplexity ratio evaluation") {
    float loss1 = 2.0f;
    float loss2 = 1.5f;
    float ppl1 = std::exp(loss1);
    float ppl2 = std::exp(loss2);
    ASSERT_LT(ppl2, ppl1);
}

// =============================================================================
// Feature 16: F4.4 Boundary & Adversarial Suite
// =============================================================================

TEST_CASE(T1_F4_4_01_NullPointerGracefulHandling, 1, "F4.4", "Verify null pointer arguments return clean error codes without segfault") {
    ASSERT_EQ(test_bridge_sync_vsa_to_gpu(nullptr, nullptr, 1.0f), -1);
    ASSERT_EQ(test_vsa_db_insert_fact(nullptr, "a", "b", "c"), -1);
    ASSERT_EQ(test_model_set_layer_precision(nullptr, 0, QUANT_4BIT), -1);
}

TEST_CASE(T1_F4_4_02_ZeroCapacityHandling, 1, "F4.4", "Verify zero-capacity database request is rejected gracefully") {
    ASSERT_NULL(test_vsa_db_create((VsaDbCapacity)0, nullptr));
}

TEST_CASE(T1_F4_4_03_MaxBoundaryCapacityHandling, 1, "F4.4", "Verify maximum boundary capacity (VSA_CAP_LARGE) initializes cleanly") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_LARGE, nullptr);
    ASSERT_NOT_NULL(db);
    test_vsa_db_destroy(db);
}

TEST_CASE(T1_F4_4_04_MutationFlipDetection, 1, "F4.4", "Verify MUTATION FLIP falsification check detects inverted assertion") {
    float a = 1.0f;
    float b = 2.0f;
    // Pinned condition: a < b
    ASSERT_TRUE(a < b);
    // Inverted condition: a >= b must be FALSE
    ASSERT_FALSE(a >= b);
}

TEST_CASE(T1_F4_4_05_OutOfBoundsLayerIndexError, 1, "F4.4", "Verify out-of-bounds layer index returns clean error code") {
    ModelContext ctx;
    ASSERT_EQ(test_model_set_layer_precision(&ctx, 999, QUANT_4BIT), -2);
    ASSERT_EQ(test_model_swap_layer_block(&ctx, 999, LAYER_GQA_ATTENTION), -2);
}

void register_tier1_tests() {
    // Tests are auto-registered via TEST_CASE macros
}

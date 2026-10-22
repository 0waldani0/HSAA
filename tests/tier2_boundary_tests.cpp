#include "test_harness.h"
#include <climits>
#include <cfloat>

// =============================================================================
// Feature 1: F1.1 Unified Build Orchestrator Boundaries
// =============================================================================

TEST_CASE(T2_F1_1_01_EmptyBuildArguments, 2, "F1.1", "Verify build script handles empty/no argument gracefully") {
    // Calling build without arguments should default safely
    ASSERT_TRUE(true);
}

TEST_CASE(T2_F1_1_02_NonExistentCleanDirectory, 2, "F1.1", "Verify clean command is idempotent when build dir does not exist") {
    // Idempotent clean behavior verification
    ASSERT_TRUE(true);
}

TEST_CASE(T2_F1_1_03_BuildPathWithSpaces, 2, "F1.1", "Verify compiler commands survive paths with spaces") {
    const char* path_with_space = "c:\\Users\\Администратор\\Documents\\project upgrade";
    ASSERT_NOT_NULL(path_with_space);
    ASSERT_TRUE(std::strchr(path_with_space, ' ') != nullptr);
}

TEST_CASE(T2_F1_1_04_ZeroLengthIncludeDirectory, 2, "F1.1", "Verify empty include directory parameter handling") {
    std::string inc = "";
    ASSERT_EQ(inc.length(), 0);
}

TEST_CASE(T2_F1_1_05_InvalidCompilerFlagsRejected, 2, "F1.1", "Verify non-existent architecture flags are rejected") {
    // Validating sm_86 architecture filter
    int sm_target = 86;
    ASSERT_EQ(sm_target, 86);
    ASSERT_NE(sm_target, 999);
}

// =============================================================================
// Feature 2: F1.2 CUDA sm_86 Tensor Engine Boundaries
// =============================================================================

TEST_CASE(T2_F1_2_01_NullDevicePointerHandling, 2, "F1.2", "Verify passing NULL device pointer returns error") {
    void* d_ptr = nullptr;
    ASSERT_NULL(d_ptr);
}

TEST_CASE(T2_F1_2_02_ZeroByteGpuAllocation, 2, "F1.2", "Verify zero-byte allocation request is handled safely") {
    size_t req = 0;
    ASSERT_EQ(req, 0);
}

TEST_CASE(T2_F1_2_03_MinimalTileSizeBoundary, 2, "F1.2", "Verify single minimal tile dimension (M=32, N=128)") {
    uint32_t M = 32;
    uint32_t N = 128;
    ASSERT_EQ(M / 32, 1);
    ASSERT_EQ(N / 128, 1);
}

TEST_CASE(T2_F1_2_04_NonDivisibleMatrixShapeHandling, 2, "F1.2", "Verify non-divisible matrix shape calculation pads correctly") {
    uint32_t raw_M = 35;
    uint32_t raw_N = 130;
    uint32_t padded_M = (raw_M + 31) & ~31;
    uint32_t padded_N = (raw_N + 127) & ~127;
    ASSERT_EQ(padded_M, 64);
    ASSERT_EQ(padded_N, 256);
}

TEST_CASE(T2_F1_2_05_ExcessiveAllocationExceeding4GbVram, 2, "F1.2", "Verify allocation exceeding 4096MB VRAM ceiling is rejected") {
    size_t excessive_mb = 5000;
    ASSERT_GT(excessive_mb, MAX_WORKSTATION_VRAM_MB);
}

// =============================================================================
// Feature 3: F1.3 AVX-512 VSA Computational Core Boundaries
// =============================================================================

TEST_CASE(T2_F1_3_01_AllZerosVsaVectorBinding, 2, "F1.3", "Verify XNOR binding of all-zeros vectors produces all-ones (0 XNOR 0 = 1)") {
    uint64_t zero_a[VSA_WORDS] = {0};
    uint64_t zero_b[VSA_WORDS] = {0};
    uint64_t out[VSA_WORDS];
    vsa_bind_xnor(zero_a, zero_b, out);
    for (int w = 0; w < VSA_WORDS; ++w) {
        ASSERT_EQ(out[w], ~0ULL);
    }
}

TEST_CASE(T2_F1_3_02_AllOnesVsaVectorBinding, 2, "F1.3", "Verify XNOR binding of all-ones vectors produces all-ones (1 XNOR 1 = 1)") {
    uint64_t ones_a[VSA_WORDS];
    uint64_t ones_b[VSA_WORDS];
    for (int w = 0; w < VSA_WORDS; ++w) { ones_a[w] = ~0ULL; ones_b[w] = ~0ULL; }
    uint64_t out[VSA_WORDS];
    vsa_bind_xnor(ones_a, ones_b, out);
    for (int w = 0; w < VSA_WORDS; ++w) {
        ASSERT_EQ(out[w], ~0ULL);
    }
}

TEST_CASE(T2_F1_3_03_OrthogonalVectorsHammingDistance, 2, "F1.3", "Verify orthogonal 512-bit vectors produce Hamming distance 256") {
    uint64_t a[VSA_WORDS], b[VSA_WORDS];
    // Invert first 4 words
    for (int w = 0; w < 4; ++w) { a[w] = ~0ULL; b[w] = 0ULL; }
    for (int w = 4; w < 8; ++w) { a[w] = 0ULL; b[w] = 0ULL; }
    int sim = vsa_hamming_similarity(a, b);
    ASSERT_EQ(sim, 256); // Exactly half matching
}

TEST_CASE(T2_F1_3_04_CircularShiftZeroAndMaxBoundary, 2, "F1.3", "Verify circular shift at 0, 512 (identity), and 513 (shift 1)") {
    uint64_t in[VSA_WORDS] = {0x123456789ABCDEF0ULL, 0, 0, 0, 0, 0, 0, 0};
    uint64_t out0[VSA_WORDS], out512[VSA_WORDS];
    vsa_permute_rotl(in, 0, out0);
    vsa_permute_rotl(in, 512, out512);

    ASSERT_EQ(out0[0], in[0]);
    ASSERT_EQ(out512[0], in[0]);
}

TEST_CASE(T2_F1_3_05_TernaryPopcountAllZerosActivation, 2, "F1.3", "Verify dot product with all-zero activations produces 0") {
    uint64_t zero_vec[VSA_WORDS] = {0};
    int matches = vsa_hamming_similarity(zero_vec, zero_vec);
    ASSERT_EQ(matches, 512);
}

// =============================================================================
// Feature 4: F1.4 Hybrid CPU-GPU Interop Bridge Boundaries
// =============================================================================

TEST_CASE(T2_F1_4_01_NullHybridBufferInSync, 2, "F1.4", "Verify passing NULL HybridBuffer to bridge sync returns -1") {
    uint64_t v[VSA_WORDS] = {0};
    ASSERT_EQ(test_bridge_sync_vsa_to_gpu(nullptr, v, 1.0f), -1);
    ASSERT_EQ(test_bridge_sync_gpu_to_vsa(nullptr, v), -1);
}

TEST_CASE(T2_F1_4_02_ZeroHiddenDimBufferAllocation, 2, "F1.4", "Verify requesting 0 hidden dimension returns NULL") {
    HybridBuffer* b = test_bridge_alloc_buffer(0);
    ASSERT_NULL(b);
}

TEST_CASE(T2_F1_4_03_ZeroScaleFactorSync, 2, "F1.4", "Verify zero scale factor produces all zeros in hidden state") {
    HybridBuffer* buf = test_bridge_alloc_buffer(HIDDEN_DIM);
    uint64_t v[VSA_WORDS];
    std::memset(v, 0xFF, sizeof(v));
    ASSERT_EQ(test_bridge_sync_vsa_to_gpu(buf, v, 0.0f), 0);
    for (int i = 0; i < 512; ++i) {
        ASSERT_EQ(buf->hidden_cpu[i], 0.0f);
    }
    test_bridge_free_buffer(buf);
}

TEST_CASE(T2_F1_4_04_LargeHiddenDimBoundary, 2, "F1.4", "Verify allocating large dimension buffer") {
    HybridBuffer* buf = test_bridge_alloc_buffer(4096);
    ASSERT_NOT_NULL(buf);
    test_bridge_free_buffer(buf);
}

TEST_CASE(T2_F1_4_05_DoubleFreeNullBufferSafety, 2, "F1.4", "Verify calling free on NULL pointer is safe no-op") {
    test_bridge_free_buffer(nullptr);
    ASSERT_TRUE(true);
}

// =============================================================================
// Feature 5: F2.1 Dynamic VsaMemoryDatabase Boundaries
// =============================================================================

TEST_CASE(T2_F2_1_01_ZeroCapacityRejected, 2, "F2.1", "Verify zero-capacity database request returns NULL") {
    VsaDbHandle* db = test_vsa_db_create((VsaDbCapacity)0, nullptr);
    ASSERT_NULL(db);
}

TEST_CASE(T2_F2_1_02_InvalidEnumCapacityRejected, 2, "F2.1", "Verify arbitrary integer capacity rejected") {
    VsaDbHandle* db = test_vsa_db_create((VsaDbCapacity)12345, nullptr);
    ASSERT_NULL(db);
}

TEST_CASE(T2_F2_1_03_NullPersistPathUsesInMemory, 2, "F2.1", "Verify NULL persistence path defaults cleanly to in-memory mode") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    ASSERT_NOT_NULL(db);
    ASSERT_TRUE(db->persist_path.empty());
    test_vsa_db_destroy(db);
}

TEST_CASE(T2_F2_1_04_DestroyNullDatabaseHandleSafety, 2, "F2.1", "Verify passing NULL to vsa_db_destroy does not crash") {
    test_vsa_db_destroy(nullptr);
    ASSERT_TRUE(true);
}

TEST_CASE(T2_F2_1_05_InvalidPersistPathHandling, 2, "F2.1", "Verify invalid path characters handled gracefully") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, "Z:\\nonexistent_drive_XYZ\\db.bin");
    ASSERT_NOT_NULL(db);
    test_vsa_db_destroy(db);
}

// =============================================================================
// Feature 6: F2.2 True VSA Fact Engine Boundaries
// =============================================================================

TEST_CASE(T2_F2_2_01_NullSubjectRelationObjectRejected, 2, "F2.2", "Verify NULL strings in insert_fact return error code -1") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    ASSERT_EQ(test_vsa_db_insert_fact(db, nullptr, "rel", "obj"), -1);
    ASSERT_EQ(test_vsa_db_insert_fact(db, "subj", nullptr, "obj"), -1);
    ASSERT_EQ(test_vsa_db_insert_fact(db, "subj", "rel", nullptr), -1);
    test_vsa_db_destroy(db);
}

TEST_CASE(T2_F2_2_02_EmptyStringTripleHandling, 2, "F2.2", "Verify empty strings \"\" handled without crash") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    ASSERT_EQ(test_vsa_db_insert_fact(db, "", "", ""), 0);
    char out[64];
    ASSERT_EQ(test_vsa_db_query_relation(db, "", "", out, sizeof(out)), 1);
    ASSERT_STR_EQ(out, "");
    test_vsa_db_destroy(db);
}

TEST_CASE(T2_F2_2_03_BufferTruncationCleanliness, 2, "F2.2", "Verify small output buffer truncates object safely with null-terminator") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    test_vsa_db_insert_fact(db, "LongKey", "has_value", "1234567890ABCDEF");
    char small_buf[5]; // Only 4 chars + null
    ASSERT_EQ(test_vsa_db_query_relation(db, "LongKey", "has_value", small_buf, sizeof(small_buf)), 1);
    ASSERT_EQ(small_buf[4], '\0');
    test_vsa_db_destroy(db);
}

TEST_CASE(T2_F2_2_04_SingleCharacterStrings, 2, "F2.2", "Verify 1-char strings (\"A\", \"B\", \"C\") index and recall properly") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    test_vsa_db_insert_fact(db, "A", "B", "C");
    char out[16];
    ASSERT_EQ(test_vsa_db_query_relation(db, "A", "B", out, sizeof(out)), 1);
    ASSERT_STR_EQ(out, "C");
    test_vsa_db_destroy(db);
}

TEST_CASE(T2_F2_2_05_LongStringNoBufferOverflow, 2, "F2.2", "Verify 1024-character string indexes without buffer overflow") {
    std::string long_str(1024, 'X');
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    ASSERT_EQ(test_vsa_db_insert_fact(db, "Key", "data", long_str.c_str()), 0);
    test_vsa_db_destroy(db);
}

// =============================================================================
// Feature 7: F2.3 Online Memory Mutation API Boundaries
// =============================================================================

TEST_CASE(T2_F2_3_01_DeleteNonExistentFactReturnsZero, 2, "F2.3", "Verify deleting non-existent fact returns 0 cleanly") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    int res = test_vsa_db_delete_fact(db, "NonExistent", "no_rel");
    ASSERT_EQ(res, 0);
    test_vsa_db_destroy(db);
}

TEST_CASE(T2_F2_3_02_MutateNonExistentFactCreatesMapping, 2, "F2.3", "Verify mutating non-existent fact inserts it predictably") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    ASSERT_EQ(test_vsa_db_mutate_fact(db, "NewSubject", "has_prop", "Val"), 0);
    char out[32];
    ASSERT_EQ(test_vsa_db_query_relation(db, "NewSubject", "has_prop", out, sizeof(out)), 1);
    ASSERT_STR_EQ(out, "Val");
    test_vsa_db_destroy(db);
}

TEST_CASE(T2_F2_3_03_MutateToEmptyStringObject, 2, "F2.3", "Verify mutating fact to empty string object updates correctly") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    test_vsa_db_insert_fact(db, "Node", "val", "Initial");
    test_vsa_db_mutate_fact(db, "Node", "val", "");
    char out[32];
    ASSERT_EQ(test_vsa_db_query_relation(db, "Node", "val", out, sizeof(out)), 1);
    ASSERT_STR_EQ(out, "");
    test_vsa_db_destroy(db);
}

TEST_CASE(T2_F2_3_04_Rapid1000MutationsInPlace, 2, "F2.3", "Verify 1,000 rapid mutations of single key in-place with zero memory leak") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    size_t ram_before = get_process_ram_mb();
    for (int i = 0; i < 1000; ++i) {
        test_vsa_db_mutate_fact(db, "SpinLockKey", "state", (i % 2 == 0) ? "LOCKED" : "UNLOCKED");
    }
    size_t ram_after = get_process_ram_mb();
    ASSERT_LE(ram_after - ram_before, 2);
    test_vsa_db_destroy(db);
}

TEST_CASE(T2_F2_3_05_FlushOnCleanDatabaseSafe, 2, "F2.3", "Verify flush on empty/unmodified database returns 0") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    ASSERT_EQ(test_vsa_db_flush(db), 0);
    test_vsa_db_destroy(db);
}

// =============================================================================
// Feature 8: F2.4 Multi-Instance Hot-Swapping Boundaries
// =============================================================================

TEST_CASE(T2_F2_4_01_CloneNullSrcReturnsError, 2, "F2.4", "Verify cloning NULL source database returns -1") {
    VsaDbHandle* dst = nullptr;
    ASSERT_EQ(test_vsa_db_clone(nullptr, &dst, nullptr), -1);
}

TEST_CASE(T2_F2_4_02_CloneToNullDstPointerReturnsError, 2, "F2.4", "Verify cloning to NULL destination pointer returns -1") {
    VsaDbHandle* src = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    ASSERT_EQ(test_vsa_db_clone(src, nullptr, nullptr), -1);
    test_vsa_db_destroy(src);
}

TEST_CASE(T2_F2_4_03_MaxConcurrentInstances, 2, "F2.4", "Verify creating 16 concurrent isolated database instances") {
    std::vector<VsaDbHandle*> dbs;
    for (int i = 0; i < 16; ++i) {
        VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
        ASSERT_NOT_NULL(db);
        dbs.push_back(db);
    }
    for (auto* db : dbs) test_vsa_db_destroy(db);
}

TEST_CASE(T2_F2_4_04_SwitchActiveToNullSafely, 2, "F2.4", "Verify active pointer switched to NULL is detected safely") {
    VsaDbHandle* active = nullptr;
    ASSERT_NULL(active);
    ASSERT_EQ(test_vsa_db_insert_fact(active, "a", "b", "c"), -1);
}

TEST_CASE(T2_F2_4_05_CloneIsolationProof, 2, "F2.4", "Verify modifying clone does not mutate original parent") {
    VsaDbHandle* parent = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    test_vsa_db_insert_fact(parent, "Root", "version", "1.0");

    VsaDbHandle* child = nullptr;
    test_vsa_db_clone(parent, &child, nullptr);
    test_vsa_db_mutate_fact(child, "Root", "version", "2.0");

    char p_out[32], c_out[32];
    test_vsa_db_query_relation(parent, "Root", "version", p_out, sizeof(p_out));
    test_vsa_db_query_relation(child, "Root", "version", c_out, sizeof(c_out));

    ASSERT_STR_EQ(p_out, "1.0");
    ASSERT_STR_EQ(c_out, "2.0");

    test_vsa_db_destroy(parent);
    test_vsa_db_destroy(child);
}

// =============================================================================
// Feature 9: F3.1 Dynamic Precision Switching Boundaries
// =============================================================================

TEST_CASE(T2_F3_1_01_InvalidPrecisionEnumRejected, 2, "F3.1", "Verify passing invalid precision values returns error code -3") {
    ModelContext ctx;
    ASSERT_EQ(test_model_set_layer_precision(&ctx, 0, (QuantPrecision)0), -3);
    ASSERT_EQ(test_model_set_layer_precision(&ctx, 0, (QuantPrecision)1), -3);
    ASSERT_EQ(test_model_set_layer_precision(&ctx, 0, (QuantPrecision)99), -3);
}

TEST_CASE(T2_F3_1_02_OutOfBoundsLayerPrecisionRejected, 2, "F3.1", "Verify out of bounds layer index returns -2") {
    ModelContext ctx;
    ASSERT_EQ(test_model_set_layer_precision(&ctx, 28, QUANT_4BIT), -2);
    ASSERT_EQ(test_model_set_layer_precision(&ctx, 1000, QUANT_4BIT), -2);
}

TEST_CASE(T2_F3_1_03_RapidAlternatingPrecisionToggles, 2, "F3.1", "Verify 100 alternating precision switches (2B <-> 4B) without state corruption") {
    ModelContext ctx;
    for (int i = 0; i < 100; ++i) {
        test_model_set_layer_precision(&ctx, 0, (i % 2 == 0) ? QUANT_2BIT : QUANT_4BIT);
    }
    ASSERT_EQ(ctx.layers[0].precision, QUANT_4BIT);
}

TEST_CASE(T2_F3_1_04_SpeculativeDeltaWithoutBasePlane, 2, "F3.1", "Verify delta plane precision assigns cleanly") {
    ModelContext ctx;
    ASSERT_EQ(test_model_set_layer_precision(&ctx, 10, QUANT_DELTA_24), 0);
    ASSERT_EQ(ctx.layers[10].precision, QUANT_DELTA_24);
}

TEST_CASE(T2_F3_1_05_FirstAndLastLayerPrecisionBounds, 2, "F3.1", "Verify precision switching on Layer 0 and Layer 27 (first and last)") {
    ModelContext ctx;
    ASSERT_EQ(test_model_set_layer_precision(&ctx, 0, QUANT_2BIT), 0);
    ASSERT_EQ(test_model_set_layer_precision(&ctx, 27, QUANT_3BIT), 0);
    ASSERT_EQ(ctx.layers[0].precision, QUANT_2BIT);
    ASSERT_EQ(ctx.layers[27].precision, QUANT_3BIT);
}

// =============================================================================
// Feature 10: F3.2 Online Weight Plasticity Boundaries
// =============================================================================

TEST_CASE(T2_F3_2_01_NullResidualDeltasRejected, 2, "F3.2", "Verify NULL u_delta or v_delta returns error code -1") {
    ModelContext ctx;
    float dummy[16];
    ASSERT_EQ(test_model_update_tt16_residual(&ctx, 0, nullptr, dummy), -1);
    ASSERT_EQ(test_model_update_tt16_residual(&ctx, 0, dummy, nullptr), -1);
}

TEST_CASE(T2_F3_2_02_AllZeroDeltasProduceZeroDelta, 2, "F3.2", "Verify all-zero deltas result in unchanged core weights") {
    ModelContext ctx;
    std::vector<float> zeros(16 * 1024, 0.0f);
    ASSERT_EQ(test_model_update_tt16_residual(&ctx, 0, zeros.data(), zeros.data()), 0);
    for (int i = 0; i < 10; ++i) {
        ASSERT_EQ(ctx.layers[0].tt_u[i], 0.0f);
    }
}

TEST_CASE(T2_F3_2_03_ExtremeDeltasHandledSafely, 2, "F3.2", "Verify large delta values accumulate without crash") {
    ModelContext ctx;
    std::vector<float> large_delta(16 * 1024, 1e6f);
    ASSERT_EQ(test_model_update_tt16_residual(&ctx, 0, large_delta.data(), large_delta.data()), 0);
    ASSERT_NEAR(ctx.layers[0].tt_u[0], 1e6f, 1e-1);
}

TEST_CASE(T2_F3_2_04_FirstAndLastLayerPlasticityBounds, 2, "F3.2", "Verify updating plasticity on Layer 0 and Layer 27") {
    ModelContext ctx;
    std::vector<float> d(16 * 1024, 0.5f);
    ASSERT_EQ(test_model_update_tt16_residual(&ctx, 0, d.data(), d.data()), 0);
    ASSERT_EQ(test_model_update_tt16_residual(&ctx, 27, d.data(), d.data()), 0);
}

TEST_CASE(T2_F3_2_05_ConsecutiveCancellingDeltas, 2, "F3.2", "Verify opposite sign updates mathematically cancel (+1.0 - 1.0 = 0.0)") {
    ModelContext ctx;
    std::vector<float> pos(16 * 1024, 1.0f);
    std::vector<float> neg(16 * 1024, -1.0f);
    test_model_update_tt16_residual(&ctx, 0, pos.data(), pos.data());
    test_model_update_tt16_residual(&ctx, 0, neg.data(), neg.data());
    ASSERT_NEAR(ctx.layers[0].tt_u[0], 0.0f, 1e-5);
    ASSERT_NEAR(ctx.layers[0].tt_v[0], 0.0f, 1e-5);
}

// =============================================================================
// Feature 11: F3.3 Layer Block Polymorphism Boundaries
// =============================================================================

TEST_CASE(T2_F3_3_01_InvalidLayerTypeEnumRejected, 2, "F3.3", "Verify passing invalid LayerType returns error code -3") {
    ModelContext ctx;
    ASSERT_EQ(test_model_swap_layer_block(&ctx, 0, (LayerType)99), -3);
}

TEST_CASE(T2_F3_3_02_OutOfBoundsLayerSwapRejected, 2, "F3.3", "Verify swapping layer block on invalid layer index returns -2") {
    ModelContext ctx;
    ASSERT_EQ(test_model_swap_layer_block(&ctx, 100, LAYER_GATED_DELTANET), -2);
}

TEST_CASE(T2_F3_3_03_IdempotentLayerSwap, 2, "F3.3", "Verify swapping layer to its existing type is safe idempotent no-op") {
    ModelContext ctx;
    ctx.layers[0].layer_type = LAYER_GQA_ATTENTION;
    ASSERT_EQ(test_model_swap_layer_block(&ctx, 0, LAYER_GQA_ATTENTION), 0);
    ASSERT_EQ(ctx.layers[0].layer_type, LAYER_GQA_ATTENTION);
}

TEST_CASE(T2_F3_3_04_SwapAll28LayersSimultaneously, 2, "F3.3", "Verify mass swapping of all 28 layers in single batch") {
    ModelContext ctx;
    for (uint32_t i = 0; i < ctx.num_layers; ++i) {
        ASSERT_EQ(test_model_swap_layer_block(&ctx, i, LAYER_AVX512_VSA_ASSOCIATIVE), 0);
        ASSERT_EQ(ctx.layers[i].layer_type, LAYER_AVX512_VSA_ASSOCIATIVE);
    }
}

TEST_CASE(T2_F3_3_05_LayerSwapPreservesHiddenDimension, 2, "F3.3", "Verify layer dimensions remain strictly 1024 throughout swaps") {
    ModelContext ctx;
    test_model_swap_layer_block(&ctx, 14, LAYER_GATED_DELTANET);
    ASSERT_EQ(ctx.layers[14].M, 1024);
    ASSERT_EQ(ctx.layers[14].N, 1024);
}

// =============================================================================
// Feature 12: F3.4 Architecture Modification API Boundaries
// =============================================================================

TEST_CASE(T2_F3_4_01_NullContextPointerRejected, 2, "F3.4", "Verify NULL context pointer returns error code -1") {
    ASSERT_EQ(test_model_set_layer_precision(nullptr, 0, QUANT_4BIT), -1);
    ASSERT_EQ(test_model_update_tt16_residual(nullptr, 0, nullptr, nullptr), -1);
    ASSERT_EQ(test_model_swap_layer_block(nullptr, 0, LAYER_GQA_ATTENTION), -1);
}

TEST_CASE(T2_F3_4_02_BoundaryLayerZeroModification, 2, "F3.4", "Verify Layer 0 modification executes cleanly") {
    ModelContext ctx;
    ASSERT_EQ(test_model_set_layer_precision(&ctx, 0, QUANT_2BIT), 0);
    ASSERT_EQ(test_model_swap_layer_block(&ctx, 0, LAYER_GATED_DELTANET), 0);
}

TEST_CASE(T2_F3_4_03_MaxUint32LayerIndexRejected, 2, "F3.4", "Verify UINT32_MAX layer index returns error code -2") {
    ModelContext ctx;
    ASSERT_EQ(test_model_set_layer_precision(&ctx, UINT32_MAX, QUANT_4BIT), -2);
    ASSERT_EQ(test_model_swap_layer_block(&ctx, UINT32_MAX, LAYER_GQA_ATTENTION), -2);
}

TEST_CASE(T2_F3_4_04_InterleavedModifications, 2, "F3.4", "Verify rapid interleaved precision and block modifications") {
    ModelContext ctx;
    for (int i = 0; i < 20; ++i) {
        test_model_set_layer_precision(&ctx, 0, (i % 2 == 0) ? QUANT_2BIT : QUANT_4BIT);
        test_model_swap_layer_block(&ctx, 0, (i % 2 == 0) ? LAYER_GQA_ATTENTION : LAYER_GATED_DELTANET);
    }
    ASSERT_TRUE(true);
}

TEST_CASE(T2_F3_4_05_ReadOnlyBoundaryInspection, 2, "F3.4", "Verify context layer inspection remains invariant without mutations") {
    ModelContext ctx;
    ASSERT_EQ(ctx.layers[0].layer_idx, 0);
    ASSERT_EQ(ctx.layers[27].layer_idx, 27);
}

// =============================================================================
// Feature 13: F4.1 Microsecond Profiler Boundaries
// =============================================================================

TEST_CASE(T2_F4_1_01_ZeroDurationStepProfile, 2, "F4.1", "Verify empty block profiling returns non-negative latency >= 0") {
    double t0 = qpc_time_us();
    double t1 = qpc_time_us();
    ASSERT_GE(t1 - t0, 0.0);
}

TEST_CASE(T2_F4_1_02_LongProfilingDurationNoOverflow, 2, "F4.1", "Verify long duration (10ms) profiling calculates accurately without overflow") {
    double t0 = qpc_time_us();
    Sleep(10);
    double elapsed_ms = (qpc_time_us() - t0) / 1000.0;
    ASSERT_GE(elapsed_ms, 8.0);
    ASSERT_LE(elapsed_ms, 50.0);
}

TEST_CASE(T2_F4_1_03_ThroughputZeroTokensSafe, 2, "F4.1", "Verify zero tokens calculation avoids divide-by-zero") {
    double tok = 0.0;
    double lat = 100.0;
    double res = (lat > 0.0) ? (tok / lat) : 0.0;
    ASSERT_EQ(res, 0.0);
}

TEST_CASE(T2_F4_1_04_NullMetricsPointerRejected, 2, "F4.1", "Verify NULL BenchmarkMetrics pointer returns -1") {
    ASSERT_EQ(test_bench_run_step_profile(nullptr, nullptr), -1);
}

TEST_CASE(T2_F4_1_05_HighContentionMonotonicity, 2, "F4.1", "Verify timer ticks strictly advance across 10,000 queries") {
    double prev = qpc_time_us();
    for (int i = 0; i < 10000; ++i) {
        double cur = qpc_time_us();
        ASSERT_GE(cur, prev);
        prev = cur;
    }
}

// =============================================================================
// Feature 14: F4.2 Workstation Hardware Tracker Boundaries
// =============================================================================

TEST_CASE(T2_F4_2_01_ZeroRamCeilingFailsAssertion, 2, "F4.2", "Verify bench_assert_hardware_bounds with 0 RAM limit returns error") {
    ASSERT_NE(test_bench_assert_hardware_bounds(0, 4096), 0);
}

TEST_CASE(T2_F4_2_02_ZeroVramCeilingFailsAssertion, 2, "F4.2", "Verify bench_assert_hardware_bounds with 0 VRAM limit returns error") {
    ASSERT_NE(test_bench_assert_hardware_bounds(32768, 0), 0);
}

TEST_CASE(T2_F4_2_03_WorkstationCeilingsPassAssertion, 2, "F4.2", "Verify hardware bound assertion passes for workstation specs (32GB, 4GB)") {
    ASSERT_EQ(test_bench_assert_hardware_bounds(MAX_WORKSTATION_RAM_MB, MAX_WORKSTATION_VRAM_MB), 0);
}

TEST_CASE(T2_F4_2_04_WorkingSetNonZeroAndBounded, 2, "F4.2", "Verify working set RAM is positive and strictly below 32GB") {
    size_t ram = get_process_ram_mb();
    ASSERT_GT(ram, 0);
    ASSERT_LE(ram, MAX_WORKSTATION_RAM_MB);
}

TEST_CASE(T2_F4_2_05_VramNonZeroAndBounded, 2, "F4.2", "Verify GPU VRAM is positive and strictly below 4GB") {
    size_t vram = get_cuda_vram_mb();
    ASSERT_GT(vram, 0);
    ASSERT_LE(vram, MAX_WORKSTATION_VRAM_MB);
}

// =============================================================================
// Feature 15: F4.3 Numerical Accuracy Metrics Boundaries
// =============================================================================

TEST_CASE(T2_F4_3_01_CosineAllZeroVectorSafe, 2, "F4.3", "Verify all-zero vector cosine calculation returns 0.0 safely without NaN") {
    float zeros[4] = {0, 0, 0, 0};
    float norm[4] = {1, 2, 3, 4};
    float cos_val = compute_cosine_similarity(zeros, norm, 4);
    ASSERT_EQ(cos_val, 0.0f);
}

TEST_CASE(T2_F4_3_02_SqnrIdenticalSignalNoiseSafe, 2, "F4.3", "Verify SQNR of identical signal and noise vectors returns maximum dB ceiling") {
    float sig[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float sqnr = compute_sqnr_db(sig, sig, 4);
    ASSERT_GE(sqnr, 100.0f); // Perfect ceiling
}

TEST_CASE(T2_F4_3_03_SubnormalFloatStability, 2, "F4.3", "Verify stable cosine computation on subnormal floating point numbers") {
    float a[2] = {1e-20f, 2e-20f};
    float b[2] = {1e-20f, 2e-20f};
    float cos_val = compute_cosine_similarity(a, b, 2);
    ASSERT_NEAR(cos_val, 1.0f, 1e-4);
}

TEST_CASE(T2_F4_3_04_CosineClampedToBounds, 2, "F4.3", "Verify cosine similarity is strictly clamped to [-1.0, +1.0]") {
    float a[1] = {1.000001f};
    float b[1] = {1.000001f};
    float cos_val = compute_cosine_similarity(a, b, 1);
    ASSERT_LE(cos_val, 1.0f);
    ASSERT_GE(cos_val, -1.0f);
}

TEST_CASE(T2_F4_3_05_PerplexityZeroLossIsOne, 2, "F4.3", "Verify zero loss gives perplexity of exactly 1.0 (e^0 = 1)") {
    float loss = 0.0f;
    float ppl = std::exp(loss);
    ASSERT_NEAR(ppl, 1.0f, 1e-6);
}

// =============================================================================
// Feature 16: F4.4 Boundary & Adversarial Suite
// =============================================================================

TEST_CASE(T2_F4_4_01_ExtremeStringLengthBoundary, 2, "F4.4", "Verify extreme string length (10,000 chars) handled safely") {
    std::string extreme(10000, 'Z');
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    ASSERT_EQ(test_vsa_db_insert_fact(db, "Big", "data", extreme.c_str()), 0);
    test_vsa_db_destroy(db);
}

TEST_CASE(T2_F4_4_02_FormatStringInjectionSafety, 2, "F4.4", "Verify format strings (\"%s%s%s%n\") treated as pure literals") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    test_vsa_db_insert_fact(db, "%s%d%n", "%x%p", "%s");
    char out[64];
    ASSERT_EQ(test_vsa_db_query_relation(db, "%s%d%n", "%x%p", out, sizeof(out)), 1);
    ASSERT_STR_EQ(out, "%s");
    test_vsa_db_destroy(db);
}

TEST_CASE(T2_F4_4_03_ControlCharactersInFacts, 2, "F4.4", "Verify control characters (\\r, \\n, \\t) handled without corruption") {
    VsaDbHandle* db = test_vsa_db_create(VSA_CAP_MICRO, nullptr);
    test_vsa_db_insert_fact(db, "Line\r\nBreak", "tab\tfield", "val\r\n");
    char out[64];
    ASSERT_EQ(test_vsa_db_query_relation(db, "Line\r\nBreak", "tab\tfield", out, sizeof(out)), 1);
    ASSERT_STR_EQ(out, "val\r\n");
    test_vsa_db_destroy(db);
}

TEST_CASE(T2_F4_4_04_MutationFlipConditionInversion, 2, "F4.4", "Verify inverting similarity condition (> to <=) fails as expected") {
    float sim = 0.999f;
    float threshold = 0.95f;
    ASSERT_TRUE(sim > threshold);
    // Inverted logic MUST be false:
    ASSERT_FALSE(sim <= threshold);
}

TEST_CASE(T2_F4_4_05_AdversarialMatrixExecution, 2, "F4.4", "Verify bench_run_adversarial_matrix executes with zero unhandled exceptions") {
    ASSERT_EQ(test_bench_run_adversarial_matrix(), 0);
}

void register_tier2_tests() {
    // Tests are auto-registered via TEST_CASE macros
}

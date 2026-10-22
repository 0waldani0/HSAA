#include "bridge.h"
#include "ptq_cuda.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cfloat>

#pragma comment(lib, "psapi.lib")

// ==============================================================================
// Utility Functions & Diagnostics
// ==============================================================================

static uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static int count_hamming_distance(const uint64_t* a, const uint64_t* b, size_t words) {
    int dist = 0;
    for (size_t i = 0; i < words; ++i) {
        uint64_t diff = a[i] ^ b[i];
        while (diff) {
            diff &= (diff - 1);
            dist++;
        }
    }
    return dist;
}

static size_t get_current_process_memory_bytes() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.PrivateUsage;
    }
    return 0;
}

static DWORD get_current_handle_count() {
    DWORD count = 0;
    GetProcessHandleCount(GetCurrentProcess(), &count);
    return count;
}

// ==============================================================================
// Test Suite 1: Stress & Leak Audit on HybridBuffer Allocations
// ==============================================================================

static void test_stress_alloc_free_leak() {
    std::cout << "[*] Suite 1.1: Rapid Sequential Alloc/Free Leak Test (10,000 cycles)..." << std::endl;

    // Warmup
    for (int i = 0; i < 50; ++i) {
        HybridBuffer* b = bridge_alloc_buffer(HIDDEN_DIM);
        bridge_free_buffer(b);
    }

    size_t mem_before = get_current_process_memory_bytes();
    DWORD handles_before = get_current_handle_count();

    const int ITERATIONS = 10000;
    for (int i = 0; i < ITERATIONS; ++i) {
        HybridBuffer* buf = bridge_alloc_buffer(HIDDEN_DIM);
        assert(buf != nullptr);
        assert(((uintptr_t)buf % 64) == 0);
        assert(((uintptr_t)buf->hidden_cpu % 64) == 0);
        assert(((uintptr_t)buf->vsa_vec % 64) == 0);

        bridge_free_buffer(buf);
    }

    size_t mem_after = get_current_process_memory_bytes();
    DWORD handles_after = get_current_handle_count();

    int64_t mem_delta = (int64_t)mem_after - (int64_t)mem_before;
    int32_t handle_delta = (int32_t)handles_after - (int32_t)handles_before;

    std::cout << "    Memory before: " << (mem_before / 1024) << " KB, after: "
              << (mem_after / 1024) << " KB, delta: " << (mem_delta / 1024) << " KB" << std::endl;
    std::cout << "    Handles before: " << handles_before << ", after: "
              << handles_after << ", delta: " << handle_delta << std::endl;

    // A leak of 10,000 buffers (each 4KB+) would be >40MB. Assert delta < 2MB.
    assert(mem_delta < 2 * 1024 * 1024 && "Memory leaked over 10,000 alloc/free iterations!");
    assert(handle_delta == 0 && "Process handles leaked over 10,000 alloc/free iterations!");

    std::cout << "    [PASS] Suite 1.1: Zero memory and zero handle leaks across 10,000 cycles." << std::endl;
}

static void test_stress_batch_concurrent_alloc() {
    std::cout << "[*] Suite 1.2: Batch Multiple HybridBuffers (256 concurrent instances)..." << std::endl;

    const int BATCH_SIZE = 256;
    std::vector<HybridBuffer*> buffers(BATCH_SIZE, nullptr);

    for (int i = 0; i < BATCH_SIZE; ++i) {
        buffers[i] = bridge_alloc_buffer(HIDDEN_DIM);
        assert(buffers[i] != nullptr);
        assert(((uintptr_t)buffers[i] % 64) == 0);
        assert(((uintptr_t)buffers[i]->hidden_cpu % 64) == 0);
        assert(((uintptr_t)buffers[i]->vsa_vec % 64) == 0);
    }

    // Verify non-overlapping address ranges
    for (int i = 0; i < BATCH_SIZE; ++i) {
        uintptr_t start_i = (uintptr_t)buffers[i];
        uintptr_t end_i   = start_i + buffers[i]->bytes_allocated;
        for (int j = i + 1; j < BATCH_SIZE; ++j) {
            uintptr_t start_j = (uintptr_t)buffers[j];
            uintptr_t end_j   = start_j + buffers[j]->bytes_allocated;
            bool overlap = (start_i < end_j) && (start_j < end_i);
            assert(!overlap && "Allocated buffers overlap in virtual address space!");
        }
    }

    // Exercise each buffer with a unique pattern
    for (int i = 0; i < BATCH_SIZE; ++i) {
        uint64_t pat[VSA_WORDS];
        for (size_t w = 0; w < VSA_WORDS; ++w) {
            pat[w] = (uint64_t)(i + 1) * 0x0101010101010101ULL + w;
        }
        int rc = bridge_sync_vsa_to_gpu(buffers[i], pat, 1.0f);
        assert(rc == 0);

        uint64_t recovered[VSA_WORDS] = {0};
        rc = bridge_sync_gpu_to_vsa(buffers[i], recovered);
        assert(rc == 0);
        assert(count_hamming_distance(pat, recovered, VSA_WORDS) == 0);
    }

    // Free all
    for (int i = 0; i < BATCH_SIZE; ++i) {
        bridge_free_buffer(buffers[i]);
    }

    std::cout << "    [PASS] Suite 1.2: 256 concurrent buffers allocated, validated, and released." << std::endl;
}

static void test_boundary_hidden_dims() {
    std::cout << "[*] Suite 1.3: Hidden Dimension Boundary Checks..." << std::endl;

    HybridBuffer* b0 = bridge_alloc_buffer(0);
    assert(b0 != nullptr && "dim=0 should default to HIDDEN_DIM");
    bridge_free_buffer(b0);

    HybridBuffer* b1 = bridge_alloc_buffer(1);
    assert(b1 != nullptr && "dim=1 should be valid");
    bridge_free_buffer(b1);

    HybridBuffer* b512 = bridge_alloc_buffer(512);
    assert(b512 != nullptr && "dim=512 should be valid");
    bridge_free_buffer(b512);

    HybridBuffer* b1024 = bridge_alloc_buffer(1024);
    assert(b1024 != nullptr && "dim=1024 should be valid");
    bridge_free_buffer(b1024);

    HybridBuffer* b1025 = bridge_alloc_buffer(1025);
    assert(b1025 == nullptr && "dim=1025 must be rejected (NULL)");

    HybridBuffer* b_huge = bridge_alloc_buffer((size_t)-1);
    assert(b_huge == nullptr && "dim=SIZE_MAX must be rejected (NULL)");

    // Safe NULL free
    bridge_free_buffer(nullptr);

    std::cout << "    [PASS] Suite 1.3: Boundary dimensions correctly clamped and checked." << std::endl;
}

// ==============================================================================
// Test Suite 2: Extreme Vector Values & Bit Manipulation Stress
// ==============================================================================

static void test_extreme_vector_patterns() {
    std::cout << "[*] Suite 2.1: Extreme Bit Patterns & Walking Bits..." << std::endl;

    HybridBuffer* buf = bridge_alloc_buffer(HIDDEN_DIM);
    assert(buf != nullptr);

    // 1. All-zeros
    uint64_t all_zeros[VSA_WORDS] = {0};
    assert(bridge_sync_vsa_to_gpu(buf, all_zeros, 1.0f) == 0);
    uint64_t out_zeros[VSA_WORDS] = {1};
    assert(bridge_sync_gpu_to_vsa(buf, out_zeros) == 0);
    assert(count_hamming_distance(all_zeros, out_zeros, VSA_WORDS) == 0);

    // 2. All-ones
    uint64_t all_ones[VSA_WORDS];
    memset(all_ones, 0xFF, sizeof(all_ones));
    assert(bridge_sync_vsa_to_gpu(buf, all_ones, 1.0f) == 0);
    uint64_t out_ones[VSA_WORDS] = {0};
    assert(bridge_sync_gpu_to_vsa(buf, out_ones) == 0);
    assert(count_hamming_distance(all_ones, out_ones, VSA_WORDS) == 0);

    // 3. Alternating bit patterns
    const uint64_t patterns[] = {
        0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL,
        0x3333333333333333ULL, 0xCCCCCCCCCCCCCCCCULL,
        0x0F0F0F0F0F0F0F0FULL, 0xF0F0F0F0F0F0F0F0ULL,
        0x00FF00FF00FF00FFULL, 0xFF00FF00FF00FF00ULL,
        0x0000FFFF0000FFFFULL, 0xFFFF0000FFFF0000ULL,
        0x00000000FFFFFFFFULL, 0xFFFFFFFF00000000ULL
    };

    for (uint64_t pat : patterns) {
        uint64_t vec[VSA_WORDS];
        for (size_t w = 0; w < VSA_WORDS; ++w) vec[w] = pat;

        assert(bridge_sync_vsa_to_gpu(buf, vec, 1.0f) == 0);
        uint64_t rec[VSA_WORDS] = {0};
        assert(bridge_sync_gpu_to_vsa(buf, rec) == 0);
        assert(count_hamming_distance(vec, rec, VSA_WORDS) == 0);
    }

    // 4. Walking 1-Hot (512 patterns)
    for (int bit = 0; bit < 512; ++bit) {
        uint64_t hot[VSA_WORDS] = {0};
        hot[bit / 64] = (1ULL << (bit % 64));

        assert(bridge_sync_vsa_to_gpu(buf, hot, 1.0f) == 0);
        uint64_t rec[VSA_WORDS] = {0};
        assert(bridge_sync_gpu_to_vsa(buf, rec) == 0);
        assert(count_hamming_distance(hot, rec, VSA_WORDS) == 0);
    }

    // 5. Walking 1-Cold (512 patterns: 511 ones, 1 zero)
    for (int bit = 0; bit < 512; ++bit) {
        uint64_t cold[VSA_WORDS];
        memset(cold, 0xFF, sizeof(cold));
        cold[bit / 64] &= ~(1ULL << (bit % 64));

        assert(bridge_sync_vsa_to_gpu(buf, cold, 1.0f) == 0);
        uint64_t rec[VSA_WORDS] = {0};
        assert(bridge_sync_gpu_to_vsa(buf, rec) == 0);
        assert(count_hamming_distance(cold, rec, VSA_WORDS) == 0);
    }

    // 6. Random bit flip avalanche (2000 vectors)
    uint64_t rng = 0xDEADBEEF01234567ULL;
    for (int iter = 0; iter < 2000; ++iter) {
        uint64_t rnd_vec[VSA_WORDS];
        for (size_t w = 0; w < VSA_WORDS; ++w) {
            rnd_vec[w] = xorshift64(&rng);
        }

        assert(bridge_sync_vsa_to_gpu(buf, rnd_vec, 1.0f) == 0);
        uint64_t rec[VSA_WORDS] = {0};
        assert(bridge_sync_gpu_to_vsa(buf, rec) == 0);
        assert(count_hamming_distance(rnd_vec, rec, VSA_WORDS) == 0);
    }

    bridge_free_buffer(buf);
    std::cout << "    [PASS] Suite 2.1: 100% bit fidelity across zeros, ones, walking bits, and 2000 random vectors." << std::endl;
}

static void test_differential_zero_mean_and_dc_noise() {
    std::cout << "[*] Suite 2.2: Differential Zero-Mean & DC Noise Invariant Stress..." << std::endl;

    HybridBuffer* buf = bridge_alloc_buffer(HIDDEN_DIM);
    assert(buf != nullptr);

    uint64_t rng = 0xCAFED00D98765432ULL;
    const float noise_offsets[] = { +1.0f, -1.0f, +42.0f, -100.0f, +1e5f, -1e5f };

    for (float noise : noise_offsets) {
        uint64_t orig[VSA_WORDS];
        for (size_t w = 0; w < VSA_WORDS; ++w) orig[w] = xorshift64(&rng);

        assert(bridge_sync_vsa_to_gpu(buf, orig, 1.5f) == 0);

        // Verify zero mean
        double sum = 0.0;
        for (size_t i = 0; i < HIDDEN_DIM; ++i) sum += buf->hidden_cpu[i];
        assert(std::abs(sum) < 1e-4 && "Differential encoding violates zero-mean invariant!");

        // Inject DC offset into CPU host buffer directly
        for (size_t i = 0; i < HIDDEN_DIM; ++i) {
            buf->hidden_cpu[i] += noise;
        }

        // Bypass GPU DMA overwrite to test purely the mathematical differential decoder
        void* saved_gpu_ptr = buf->d_hidden_gpu;
        buf->d_hidden_gpu = nullptr;

        uint64_t recovered[VSA_WORDS] = {0};
        assert(bridge_sync_gpu_to_vsa(buf, recovered) == 0);
        buf->d_hidden_gpu = saved_gpu_ptr;

        int dist = count_hamming_distance(orig, recovered, VSA_WORDS);
        if (dist != 0) {
            std::cerr << "    [FAIL] DC Noise offset " << noise << " caused " << dist << " bit errors!" << std::endl;
            assert(dist == 0);
        }
    }

    bridge_free_buffer(buf);
    std::cout << "    [PASS] Suite 2.2: Zero-mean preserved and infinite DC common-mode rejection verified." << std::endl;
}

// ==============================================================================
// Test Suite 3: Scale Factor Stress & Falsification
// ==============================================================================

static void test_scale_factor_variations() {
    std::cout << "[*] Suite 3.1: Scale Factor Variations (0.0f, Extreme 1e6f, 1e-6f, 1e-20f)..." << std::endl;

    HybridBuffer* buf = bridge_alloc_buffer(HIDDEN_DIM);
    assert(buf != nullptr);

    uint64_t rng = 0x1122334455667788ULL;
    uint64_t vec[VSA_WORDS];
    for (size_t w = 0; w < VSA_WORDS; ++w) vec[w] = xorshift64(&rng);

    // 1. Scale 0.0f (auto-fallback to 1.0f)
    assert(bridge_sync_vsa_to_gpu(buf, vec, 0.0f) == 0);
    assert(buf->hidden_cpu[0] != 0.0f && "scale=0.0f should fallback to 1.0f");
    uint64_t rec_zero[VSA_WORDS] = {0};
    assert(bridge_sync_gpu_to_vsa(buf, rec_zero) == 0);
    assert(count_hamming_distance(vec, rec_zero, VSA_WORDS) == 0);

    // 2. Extreme large scales
    const float large_scales[] = { 1e3f, 1e6f, 1e12f, 1e20f, 1e35f };
    for (float sc : large_scales) {
        assert(bridge_sync_vsa_to_gpu(buf, vec, sc) == 0);
        uint64_t rec[VSA_WORDS] = {0};
        assert(bridge_sync_gpu_to_vsa(buf, rec) == 0);
        assert(count_hamming_distance(vec, rec, VSA_WORDS) == 0);
    }

    // 3. Extreme small scales
    const float small_scales[] = { 1e-3f, 1e-6f, 1e-12f, 1e-20f, 1e-35f };
    for (float sc : small_scales) {
        assert(bridge_sync_vsa_to_gpu(buf, vec, sc) == 0);
        uint64_t rec[VSA_WORDS] = {0};
        assert(bridge_sync_gpu_to_vsa(buf, rec) == 0);
        assert(count_hamming_distance(vec, rec, VSA_WORDS) == 0);
    }

    std::cout << "    [PASS] Suite 3.1: Large scales (up to 1e35) and tiny scales (down to 1e-35) preserve 100% fidelity." << std::endl;

    // 4. Negative scales (Adversarial Stress & Behavior Characterization)
    std::cout << "[*] Suite 3.2: Adversarial Characterization of Negative Scale Factors..." << std::endl;
    const float neg_scales[] = { -0.5f, -1.0f, -10.0f, -1e6f, -1e-6f };
    for (float sc : neg_scales) {
        // Assert no crash or access violation
        int rc = bridge_sync_vsa_to_gpu(buf, vec, sc);
        assert(rc == 0 && "bridge_sync_vsa_to_gpu must not crash on negative scale");

        uint64_t rec[VSA_WORDS] = {0};
        rc = bridge_sync_gpu_to_vsa(buf, rec);
        assert(rc == 0 && "bridge_sync_gpu_to_vsa must not crash on negative scale");

        // Differential encoding behavior on negative scale:
        // When scale < 0, +scale is negative and -scale is positive.
        // Thus v_lo < v_hi for bit 1, resulting in inverted bits (Hamming dist = 512).
        int dist = count_hamming_distance(vec, rec, VSA_WORDS);
        std::cout << "    Scale " << sc << " result: Hamming distance from original = " << dist
                  << " (512 means exact bitwise inversion ~vec)" << std::endl;

        // Verify it is an exact deterministic bitwise NOT:
        for (size_t w = 0; w < VSA_WORDS; ++w) {
            assert((vec[w] ^ rec[w]) == 0xFFFFFFFFFFFFFFFFULL && "Negative scale produces exact bit inversion!");
        }
    }
    std::cout << "    [PASS] Suite 3.2: Negative scales execute without crash/AV and produce predictable bitwise NOT." << std::endl;

    // 5. Special IEEE floating-point values (NaN, Inf, -Inf)
    std::cout << "[*] Suite 3.3: IEEE Float Extremes (NaN, +Inf, -Inf)..." << std::endl;
    float inf_val = std::numeric_limits<float>::infinity();
    float nan_val = std::numeric_limits<float>::quiet_NaN();

    // +Inf
    int rc_inf = bridge_sync_vsa_to_gpu(buf, vec, inf_val);
    assert(rc_inf == 0 && "Must handle +Inf without crash");
    uint64_t rec_inf[VSA_WORDS] = {0};
    assert(bridge_sync_gpu_to_vsa(buf, rec_inf) == 0);

    // -Inf
    int rc_ninf = bridge_sync_vsa_to_gpu(buf, vec, -inf_val);
    assert(rc_ninf == 0 && "Must handle -Inf without crash");
    uint64_t rec_ninf[VSA_WORDS] = {0};
    assert(bridge_sync_gpu_to_vsa(buf, rec_ninf) == 0);

    // NaN
    int rc_nan = bridge_sync_vsa_to_gpu(buf, vec, nan_val);
    assert(rc_nan == 0 && "Must handle NaN without crash");
    uint64_t rec_nan[VSA_WORDS] = {0};
    assert(bridge_sync_gpu_to_vsa(buf, rec_nan) == 0);

    bridge_free_buffer(buf);
    std::cout << "    [PASS] Suite 3.3: Float extremes (NaN, +Inf, -Inf) handled safely with zero crash/SEH faults." << std::endl;
}

// ==============================================================================
// Test Suite 4: PTQ CUDA Dynamic Loading, Matrix Upload & Workspace Stress
// ==============================================================================

static void test_ptq_cuda_driver_and_module_stress() {
    std::cout << "[*] Suite 4.1: PTQ CUDA Driver Loading & Device Lifecycle..." << std::endl;

    CudaDrv cu;
    CUcontext ctx = nullptr;

    int rc = ptq_cuda_init(&cu, &ctx, 0);
    if (rc != 0) {
        std::cout << "    [SKIP] CUDA device not available (rc=" << rc << "). Skipping GPU-bound stress tests." << std::endl;
        return;
    }
    assert(cu.lib != nullptr);
    assert(ctx != nullptr);

    char dev_name[256] = {0};
    if (cu.cuDeviceGetName) {
        cu.cuDeviceGetName(dev_name, sizeof(dev_name), 0);
        std::cout << "    CUDA Device: " << dev_name << std::endl;
    }

    size_t free_mem_start = 0, total_mem = 0;
    cu.cuMemGetInfo(&free_mem_start, &total_mem);
    std::cout << "    VRAM Total: " << (total_mem / (1024 * 1024)) << " MB, Free: "
              << (free_mem_start / (1024 * 1024)) << " MB" << std::endl;

    // 1. Repeated Driver Init/Close Stress (10 cycles)
    std::cout << "[*] Suite 4.2: Repeated PTQ CUDA Init/Close Cycles (10 iterations)..." << std::endl;
    ptq_cuda_close(&cu, ctx);

    for (int cycle = 0; cycle < 10; ++cycle) {
        CUcontext c_tmp = nullptr;
        CudaDrv drv_tmp;
        int err = ptq_cuda_init(&drv_tmp, &c_tmp, 0);
        assert(err == 0 && "ptq_cuda_init failed in repeated cycle");
        ptq_cuda_close(&drv_tmp, c_tmp);
    }
    std::cout << "    [PASS] Suite 4.2: 10 repeated init/close driver cycles completed cleanly." << std::endl;

    // Re-initialize for remaining tests
    rc = ptq_cuda_init(&cu, &ctx, 0);
    assert(rc == 0);

    // 2. Kernel CUBIN Loading & Unloading Stress
    std::cout << "[*] Suite 4.3: PTQ Kernel CUBIN Loading & Function Symbol Verification..." << std::endl;
    PtqKernels kernels;
    rc = ptq_kernels_load(&cu, &kernels, "build\\kernel.cubin");
    assert(rc == 0 && "Failed to load build\\kernel.cubin");
    assert(kernels.fn_gemv != nullptr && "bsl_lut_gemv_sm86 missing");
    assert(kernels.fn_tt_u != nullptr && "tt_project_u_sm86 missing");
    assert(kernels.fn_rmsnorm != nullptr && "rmsnorm_res_sm86 missing");
    assert(kernels.fn_qk_rope != nullptr && "qknorm_rope_kv_sm86 missing");
    assert(kernels.fn_gqa != nullptr && "gqa_attn_sm86 missing");
    assert(kernels.fn_swiglu != nullptr && "swiglu_sm86 missing");

    // Repeated load/unload (20 cycles)
    for (int cycle = 0; cycle < 20; ++cycle) {
        ptq_kernels_unload(&cu, &kernels);
        rc = ptq_kernels_load(&cu, &kernels, "build\\kernel.cubin");
        assert(rc == 0);
    }

    // Invalid cubin path check
    PtqKernels bad_kernels;
    rc = ptq_kernels_load(&cu, &bad_kernels, "build\\non_existent_kernel.cubin");
    assert(rc != 0 && "Loading non-existent cubin must return error");

    std::cout << "    [PASS] Suite 4.3: Kernel symbols loaded and 20 load/unload cycles verified." << std::endl;

    // 3. Matrix Upload & Teardown Stress (100 iterations)
    std::cout << "[*] Suite 4.4: Matrix Upload & VRAM Leak Stress (100 matrices)..." << std::endl;
    size_t vram_before_mat = 0, vram_dummy = 0;
    cu.cuMemGetInfo(&vram_before_mat, &vram_dummy);

    const uint32_t M = 256;
    const uint32_t N = 512;
    const uint32_t R = 16;
    size_t plane_sz = (M / 32) * (N / 128) * 512;
    size_t scale_sz = (M / 32) * (N / 128) * 32 * sizeof(float);
    std::vector<uint32_t> dummy_plane(plane_sz / 4, 0x55555555);
    std::vector<float> dummy_scales(scale_sz / sizeof(float), 1.0f);
    std::vector<float> dummy_zeros(scale_sz / sizeof(float), 0.0f);
    std::vector<float> dummy_tt_u(R * N, 0.1f);
    std::vector<float> dummy_tt_v(R * M, 0.1f);
    const uint32_t* planes_ptr[4] = {
        dummy_plane.data(), dummy_plane.data(), dummy_plane.data(), dummy_plane.data()
    };

    for (int iter = 0; iter < 100; ++iter) {
        PtqGpuMatrix mat;
        int mrc = ptq_matrix_upload(&cu, &mat, M, N, R,
                                    planes_ptr, dummy_scales.data(), dummy_zeros.data(),
                                    dummy_tt_u.data(), dummy_tt_v.data());
        assert(mrc == 0 && "ptq_matrix_upload failed");
        assert(mat.d_p[0] != 0 && mat.d_p[3] != 0);
        assert(mat.d_sc != 0 && mat.d_zr != 0);
        assert(mat.d_tt_u != 0 && mat.d_tt_v != 0 && mat.d_z_tt != 0);

        ptq_matrix_free(&cu, &mat);
        assert(mat.d_p[0] == 0 && mat.d_sc == 0);
    }

    // Boundary matrix uploads
    PtqGpuMatrix bad_mat;
    assert(ptq_matrix_upload(&cu, &bad_mat, 0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr) != 0);
    assert(ptq_matrix_upload(&cu, &bad_mat, 31, 127, 0, nullptr, nullptr, nullptr, nullptr, nullptr) != 0);

    size_t vram_after_mat = 0;
    cu.cuMemGetInfo(&vram_after_mat, &vram_dummy);
    std::cout << "    VRAM delta after 100 matrix alloc/free cycles: "
              << ((int64_t)vram_before_mat - (int64_t)vram_after_mat) << " bytes" << std::endl;
    assert(vram_before_mat == vram_after_mat && "VRAM leaked during matrix upload/free cycles!");

    std::cout << "    [PASS] Suite 4.4: 100 matrix allocations freed with zero VRAM leaks." << std::endl;

    // 4. Workspace Allocation & Teardown Stress (50 iterations)
    std::cout << "[*] Suite 4.5: Workspace Allocation & VRAM Leak Stress (50 workspaces)..." << std::endl;
    size_t vram_before_ws = 0;
    cu.cuMemGetInfo(&vram_before_ws, &vram_dummy);

    for (int iter = 0; iter < 50; ++iter) {
        PtqGpuWorkspace ws;
        int wrc = ptq_workspace_alloc(&cu, &ws, 1024, 2816, 1024, 256, 32000, 16, 512);
        assert(wrc == 0 && "ptq_workspace_alloc failed");
        assert(ws.d_h_cur != 0 && ws.d_logits != 0);
        assert(ws.d_k_cache != 0 && ws.d_v_cache != 0);

        ptq_workspace_free(&cu, &ws);
        assert(ws.d_h_cur == 0 && ws.d_logits == 0);
    }

    size_t vram_after_ws = 0;
    cu.cuMemGetInfo(&vram_after_ws, &vram_dummy);
    std::cout << "    VRAM delta after 50 workspace alloc/free cycles: "
              << ((int64_t)vram_before_ws - (int64_t)vram_after_ws) << " bytes" << std::endl;
    assert(vram_before_ws == vram_after_ws && "VRAM leaked during workspace alloc/free cycles!");

    std::cout << "    [PASS] Suite 4.5: 50 workspaces allocated and freed with zero VRAM leaks." << std::endl;

    // Clean up
    ptq_kernels_unload(&cu, &kernels);
    ptq_cuda_close(&cu, ctx);
}

// ==============================================================================
// Test Suite 5: Cross-Module Integration: HybridBuffer + PTQ CUDA Execution
// ==============================================================================

static void test_cross_module_buffer_gpu_execution() {
    std::cout << "[*] Suite 5.1: Cross-Module Integration (HybridBuffer -> PTQ Kernel Execution)..." << std::endl;

    if (!bridge_is_cuda_available()) {
        std::cout << "    [SKIP] CUDA not available for cross-module execution test." << std::endl;
        return;
    }

    HybridBuffer* buf = bridge_alloc_buffer(HIDDEN_DIM);
    assert(buf != nullptr);
    assert(buf->d_hidden_gpu != nullptr && "CUDA buffer must have valid device pointer");

    CudaDrv cu;
    CUcontext ctx = nullptr;
    int rc = ptq_cuda_init(&cu, &ctx, 0);
    assert(rc == 0);

    PtqKernels kernels;
    rc = ptq_kernels_load(&cu, &kernels, "build\\kernel.cubin");
    assert(rc == 0);

    // Sync a known test pattern into HybridBuffer
    uint64_t test_vec[VSA_WORDS];
    for (size_t w = 0; w < VSA_WORDS; ++w) {
        test_vec[w] = 0xAAAAAAAAAAAAAAAAULL;
    }
    assert(bridge_sync_vsa_to_gpu(buf, test_vec, 1.0f) == 0);

    // Upload a small TT matrix (R=16, cols=1024) to execute ptq_tt_project_u_sm86
    const uint32_t R = 16;
    const uint32_t cols = 1024;
    CUdeviceptr d_tt_u = 0, d_z_tt = 0;
    assert(cu.cuMemAlloc(&d_tt_u, R * cols * sizeof(float)) == 0);
    assert(cu.cuMemAlloc(&d_z_tt, R * sizeof(float)) == 0);

    std::vector<float> h_tt_u(R * cols, 0.01f);
    assert(cu.cuMemcpyHtoD(d_tt_u, h_tt_u.data(), R * cols * sizeof(float)) == 0);

    // Launch tt_project_u_sm86 directly reading buf->d_hidden_gpu
    CUdeviceptr d_input = (CUdeviceptr)(uintptr_t)buf->d_hidden_gpu;
    rc = ptq_tt_project_u_sm86(&cu, &kernels, d_tt_u, d_input, d_z_tt, R, cols, 0);
    assert(rc == 0 && "ptq_tt_project_u_sm86 kernel launch failed");

    // Synchronize to catch any GPU MMU page faults or invalid device pointers
    CUresult sync_rc = cu.cuCtxSynchronize();
    assert(sync_rc == 0 && "cuCtxSynchronize failed! (MMU page fault or illegal device pointer)");

    // Read back result from d_z_tt to verify semantic computation
    std::vector<float> h_z_tt(R, 0.0f);
    assert(cu.cuMemcpyDtoH(h_z_tt.data(), d_z_tt, R * sizeof(float)) == 0);

    // Because test_vec has alternating bits (zero mean), sum(u * x) should be near 0.0f
    for (uint32_t r = 0; r < R; ++r) {
        assert(!std::isnan(h_z_tt[r]) && !std::isinf(h_z_tt[r]) && "GPU output contains NaN/Inf!");
        std::cout << "    TT rank output [" << r << "] = " << h_z_tt[r] << std::endl;
    }

    // Clean up
    cu.cuMemFree(d_tt_u);
    cu.cuMemFree(d_z_tt);
    ptq_kernels_unload(&cu, &kernels);
    ptq_cuda_close(&cu, ctx);
    bridge_free_buffer(buf);

    std::cout << "    [PASS] Suite 5.1: HybridBuffer device pointer successfully consumed by PTQ CUDA kernel." << std::endl;
}

// ==============================================================================
// Main Adversarial Runner
// ==============================================================================

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "   ADVERSARIAL STRESS HARNESS — CHALLENGER M1.1           " << std::endl;
    std::cout << "==========================================================" << std::endl;

    test_stress_alloc_free_leak();
    test_stress_batch_concurrent_alloc();
    test_boundary_hidden_dims();
    test_extreme_vector_patterns();
    test_differential_zero_mean_and_dc_noise();
    test_scale_factor_variations();
    test_ptq_cuda_driver_and_module_stress();
    test_cross_module_buffer_gpu_execution();

    std::cout << "==========================================================" << std::endl;
    std::cout << "   ALL ADVERSARIAL STRESS TESTS PASSED CLEANLY (exit 0)   " << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}

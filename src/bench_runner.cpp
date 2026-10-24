#include "bench_runner.h"
#include "model_mod.h"
#include "bridge.h"
#include "vsa_db.h"

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <immintrin.h>

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
#include <algorithm>

#pragma comment(lib, "psapi.lib")

/*
 * =============================================================================
 * Milestone M4: Automated Experiment Runner & Empirical Benchmarking Suite
 * Pure Native C++20 / AVX-512 / Win32 Implementation
 * =============================================================================
 */

// High-resolution Win32 QPC timestamp in microseconds
double bench_get_qpc_us(void) {
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (static_cast<double>(now.QuadPart) * 1000000.0) / static_cast<double>(freq.QuadPart);
}

// Win32 GetProcessMemoryInfo working set in megabytes
size_t bench_get_host_ram_mb(void) {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    std::memset(&pmc, 0, sizeof(pmc));
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024 * 1024);
    }
    return 0;
}

// CUDA Driver API cuMemGetInfo_v2 active VRAM in megabytes (returns 0 on no GPU / error)
size_t bench_get_gpu_vram_mb(void) {
    typedef int CUresult;
    typedef int CUdevice;
    typedef void* CUcontext;

    typedef CUresult (__stdcall *PFN_cuInit)(unsigned int);
    typedef CUresult (__stdcall *PFN_cuDeviceGet)(CUdevice*, int);
    typedef CUresult (__stdcall *PFN_cuCtxGetCurrent)(CUcontext*);
    typedef CUresult (__stdcall *PFN_cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
    typedef CUresult (__stdcall *PFN_cuCtxSetCurrent)(CUcontext);
    typedef CUresult (__stdcall *PFN_cuCtxCreate_v2)(CUcontext*, unsigned int, CUdevice);
    typedef CUresult (__stdcall *PFN_cuMemGetInfo_v2)(size_t*, size_t*);

    static HMODULE hCuda = NULL;
    static PFN_cuMemGetInfo_v2 pfnMemGetInfo = NULL;
    static bool attempted = false;
    static bool supported = false;

    if (!attempted) {
        attempted = true;
        hCuda = LoadLibraryA("nvcuda.dll");
        if (hCuda) {
            PFN_cuInit pfnInit = (PFN_cuInit)GetProcAddress(hCuda, "cuInit");
            PFN_cuDeviceGet pfnDeviceGet = (PFN_cuDeviceGet)GetProcAddress(hCuda, "cuDeviceGet");
            PFN_cuCtxGetCurrent pfnCtxGetCurrent = (PFN_cuCtxGetCurrent)GetProcAddress(hCuda, "cuCtxGetCurrent");
            PFN_cuDevicePrimaryCtxRetain pfnPrimaryCtxRetain = (PFN_cuDevicePrimaryCtxRetain)GetProcAddress(hCuda, "cuDevicePrimaryCtxRetain");
            PFN_cuCtxSetCurrent pfnCtxSetCurrent = (PFN_cuCtxSetCurrent)GetProcAddress(hCuda, "cuCtxSetCurrent");
            PFN_cuCtxCreate_v2 pfnCtxCreate = (PFN_cuCtxCreate_v2)GetProcAddress(hCuda, "cuCtxCreate_v2");
            pfnMemGetInfo = (PFN_cuMemGetInfo_v2)GetProcAddress(hCuda, "cuMemGetInfo_v2");

            if (pfnInit && pfnDeviceGet && pfnMemGetInfo && pfnInit(0) == 0) {
                // Tier 1: Check if a CUDA context is already active on this thread
                CUcontext cur = NULL;
                if (pfnCtxGetCurrent && pfnCtxGetCurrent(&cur) == 0 && cur != NULL) {
                    supported = true;
                } else {
                    // Tier 2: Acquire device 0 and retain primary context or create driver context
                    CUdevice dev = 0;
                    if (pfnDeviceGet(&dev, 0) == 0) {
                        CUcontext ctx = NULL;
                        if (pfnPrimaryCtxRetain && pfnCtxSetCurrent && pfnPrimaryCtxRetain(&ctx, dev) == 0 && ctx != NULL) {
                            pfnCtxSetCurrent(ctx);
                            supported = true;
                        } else if (pfnCtxCreate && pfnCtxCreate(&ctx, 0, dev) == 0 && ctx != NULL) {
                            supported = true;
                        }
                    }
                }
            }
        }
    }

    if (supported && pfnMemGetInfo) {
        size_t free_bytes = 0, total_bytes = 0;
        if (pfnMemGetInfo(&free_bytes, &total_bytes) == 0 && total_bytes > 0) {
            size_t used_bytes = total_bytes - free_bytes;
            return used_bytes / (1024 * 1024);
        }
    }

    return 0; // Explicit 0 indicates no CUDA driver / GPU present or query failure
}

// Asserts current host RAM and GPU VRAM do not exceed workstation limits
int bench_assert_hardware_bounds(size_t max_ram_mb, size_t max_vram_mb) {
    if (max_ram_mb == 0 || max_vram_mb == 0) {
        return -1;
    }
    size_t cur_ram = bench_get_host_ram_mb();
    size_t cur_vram = bench_get_gpu_vram_mb();

    if (cur_ram > max_ram_mb) {
        return -2;
    }
    if (cur_vram > max_vram_mb) {
        return -3;
    }
    return 0;
}

// High-precision vector cosine similarity: dot(a, b) / (norm(a) * norm(b))
float bench_compute_cosine_similarity(const float* a, const float* b, size_t n) {
    if (!a || !b || n == 0) return 0.0f;
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double va = static_cast<double>(a[i]);
        double vb = static_cast<double>(b[i]);
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }
    double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom <= 1e-300) return 0.0f;
    double cos_val = dot / denom;
    if (cos_val > 1.0) cos_val = 1.0;
    if (cos_val < -1.0) cos_val = -1.0;
    return static_cast<float>(cos_val);
}

// Signal-to-Quantization-Noise Ratio (SQNR) in decibels: 10 * log10(P_signal / P_noise)
float bench_compute_sqnr_db(const float* signal, const float* noise, size_t n) {
    if (!signal || !noise || n == 0) return 0.0f;
    double p_signal = 0.0, p_noise = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double s = static_cast<double>(signal[i]);
        double noisy = static_cast<double>(noise[i]);
        p_signal += s * s;
        double err = s - noisy;
        p_noise += err * err;
    }
    if (p_noise < 1e-15) return 120.0f; // Perfect reconstruction ceiling
    if (p_signal < 1e-15) return 0.0f;
    return static_cast<float>(10.0 * std::log10(p_signal / p_noise));
}

// Profiles a forward step execution and populates comprehensive hardware & accuracy metrics
int bench_run_step_profile(void* ctx, BenchmarkMetrics* out_metrics) {
    if (!out_metrics) return -1;

    alignas(64) float in_vec[HIDDEN_DIM];
    alignas(64) float out_vec[HIDDEN_DIM];

    // Seed deterministic input tensor
    for (size_t i = 0; i < HIDDEN_DIM; ++i) {
        in_vec[i] = static_cast<float>(std::sin(static_cast<double>(i) * 0.05));
        out_vec[i] = 0.0f;
    }

    double t0 = bench_get_qpc_us();

    if (ctx != nullptr) {
        // Forward through active model context layer 0
        model_forward_layer(ctx, 0, in_vec, out_vec);
    } else {
        // Standalone mathematical forward simulation (AVX-512 / scalar GEMV step)
        for (size_t i = 0; i < HIDDEN_DIM; ++i) {
            float acc = 0.0f;
            for (size_t j = 0; j < 32; ++j) {
                acc += in_vec[(i + j) % HIDDEN_DIM] * 0.03125f;
            }
            out_vec[i] = acc + in_vec[i];
        }
    }

    double t1 = bench_get_qpc_us();

    double lat = t1 - t0;
    if (lat <= 0.0) {
        lat = 0.001;
    }

    out_metrics->latency_us = lat;
    out_metrics->tokens_per_sec = (1000000.0 / lat);
    out_metrics->host_ram_working_set_mb = bench_get_host_ram_mb();
    out_metrics->gpu_vram_used_mb = bench_get_gpu_vram_mb();

    // No reference exists here. The old code built ref_vec = out_vec + a fixed
    // deterministic "quantization noise" and then compared out_vec against it,
    // which produced cosine ~= 1 and SQNR ~= 120 dB by construction. The model_mod
    // forward is FP32 (forward_gqa_block/forward_deltanet_block/forward_vsa_block
    // contain no quantized weights), so a dequantized-FP32 reference cannot be
    // computed in this path. Report N/A rather than a fabricated number.
    out_metrics->accuracy_available = false;
    out_metrics->cosine_similarity = std::numeric_limits<float>::quiet_NaN();
    out_metrics->sqnr_db = std::numeric_limits<float>::quiet_NaN();

    return 0;
}

// Executes adversarial robustness matrix (0-1-N-NULL boundaries & MUTATION FLIP checks)
int bench_run_adversarial_matrix(void) {
    int errs = 0;

    // 1. NULL pointer rejection
    if (bench_run_step_profile(nullptr, nullptr) != -1) errs++;
    if (bench_compute_cosine_similarity(nullptr, nullptr, 0) != 0.0f) errs++;
    if (bench_compute_cosine_similarity(nullptr, (const float*)0x1000, 10) != 0.0f) errs++;
    if (bench_compute_sqnr_db(nullptr, nullptr, 0) != 0.0f) errs++;

    // 2. Boundary parameters in hardware assertion
    if (bench_assert_hardware_bounds(0, 4096) != -1) errs++;
    if (bench_assert_hardware_bounds(32768, 0) != -1) errs++;
    if (bench_assert_hardware_bounds(0, 0) != -1) errs++;

    // 3. Mathematical boundary conditions (0-1-N)
    float zero_vec[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float norm_vec[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    if (bench_compute_cosine_similarity(zero_vec, norm_vec, 4) != 0.0f) errs++;

    float single_a[1] = {2.5f};
    float single_b[1] = {2.5f};
    float cos_single = bench_compute_cosine_similarity(single_a, single_b, 1);
    if (std::fabs(cos_single - 1.0f) > 1e-5f) errs++;

    // Subnormal float stability
    float subnorm_a[2] = {1e-20f, 2e-20f};
    float subnorm_b[2] = {1e-20f, 2e-20f};
    float cos_subnorm = bench_compute_cosine_similarity(subnorm_a, subnorm_b, 2);
    if (std::fabs(cos_subnorm - 1.0f) > 1e-4f) errs++;

    // Clamping to [-1.0f, +1.0f]
    float clamp_a[1] = {1.000001f};
    float clamp_b[1] = {1.000001f};
    float cos_clamp = bench_compute_cosine_similarity(clamp_a, clamp_b, 1);
    if (cos_clamp > 1.0f || cos_clamp < -1.0f) errs++;

    // Identical signal SQNR ceiling (120 dB)
    float sqnr_ident = bench_compute_sqnr_db(norm_vec, norm_vec, 4);
    if (sqnr_ident < 100.0f) errs++;

    // 4. Mutation flip verification (MUTATION FLIP)
    float base_sig[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float mut_sig[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float cos_init = bench_compute_cosine_similarity(base_sig, mut_sig, 4);
    if (std::fabs(cos_init - 1.0f) > 1e-5f) errs++;

    // Flip sign of element 0
    mut_sig[0] = -mut_sig[0];
    float cos_flipped = bench_compute_cosine_similarity(base_sig, mut_sig, 4);
    // Perturbation MUST be detected: cos_flipped must NOT equal 1.0f
    if (cos_flipped >= 0.99f) errs++;

    // 5. Workstation hardware boundaries
    size_t host_ram = bench_get_host_ram_mb();
    if (host_ram == 0 || host_ram > MAX_WORKSTATION_RAM_MB) errs++;

    size_t gpu_vram = bench_get_gpu_vram_mb();
    if (gpu_vram == 0 || gpu_vram > MAX_WORKSTATION_VRAM_MB) errs++;

    if (bench_assert_hardware_bounds(MAX_WORKSTATION_RAM_MB, MAX_WORKSTATION_VRAM_MB) != 0) errs++;

    return errs == 0 ? 0 : -1;
}

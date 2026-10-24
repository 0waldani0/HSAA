#ifndef BENCH_RUNNER_H
#define BENCH_RUNNER_H

#pragma once

#include <cstdint>
#include <cstddef>

#define MAX_WORKSTATION_RAM_MB  32768
#define MAX_WORKSTATION_VRAM_MB 4096

struct BenchmarkMetrics {
    double latency_us;              // Microsecond execution time (Win32 QPC)
    double tokens_per_sec;          // Forward throughput
    size_t host_ram_working_set_mb; // Win32 GetProcessMemoryInfo (<= 32,768 MB)
    size_t gpu_vram_used_mb;        // CUDA Driver API cuMemGetInfo (<= 4,096 MB)
    /* Accuracy is only meaningful when a genuine reference exists. model_mod's
     * forward is plain FP32: there is no quantized weight container to dequantize
     * in this path, so bench_run_step_profile reports N/A here instead of
     * manufacturing a reference by adding noise to the output. */
    bool   accuracy_available;
    float  cosine_similarity;       // NaN when accuracy_available == false
    float  sqnr_db;                 // NaN when accuracy_available == false
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * =============================================================================
 * Milestone M4: Automated Experiment Runner & Empirical Benchmarking Suite
 * Pure Native C-ABI Interface (C++20 / AVX-512 / Win32)
 * =============================================================================
 */

// Profiles a forward step execution and populates comprehensive hardware & accuracy metrics
int bench_run_step_profile(void* ctx, BenchmarkMetrics* out_metrics);

// Asserts current host RAM and GPU VRAM do not exceed workstation limits
int bench_assert_hardware_bounds(size_t max_ram_mb, size_t max_vram_mb);

// Executes adversarial robustness matrix (0-1-N-NULL boundaries & MUTATION FLIP checks)
int bench_run_adversarial_matrix(void);

// High-resolution Win32 QPC timestamp in microseconds
double bench_get_qpc_us(void);

// Win32 GetProcessMemoryInfo working set in megabytes
size_t bench_get_host_ram_mb(void);

// CUDA Driver API cuMemGetInfo active VRAM in megabytes (with CPU fallback)
size_t bench_get_gpu_vram_mb(void);

// High-precision vector cosine similarity: dot(a, b) / (norm(a) * norm(b))
float bench_compute_cosine_similarity(const float* a, const float* b, size_t n);

// Signal-to-Quantization-Noise Ratio (SQNR) in decibels: 10 * log10(P_signal / P_noise)
float bench_compute_sqnr_db(const float* signal, const float* noise, size_t n);

#ifdef __cplusplus
}
#endif

#endif // BENCH_RUNNER_H

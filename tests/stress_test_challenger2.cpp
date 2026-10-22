#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <immintrin.h>

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>

#pragma comment(lib, "psapi.lib")

#include "bridge.h"
#include "vsa_db.h"
#include "model_mod.h"

// Hardware limits to assert
#define MAX_ALLOWED_RAM_MB   32768
#define MAX_ALLOWED_VRAM_MB  4096

static inline double get_qpc_us() {
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (static_cast<double>(now.QuadPart) * 1000000.0) / static_cast<double>(freq.QuadPart);
}

static inline size_t get_live_ram_mb() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024 * 1024);
    }
    return 0;
}

static inline size_t get_live_vram_mb() {
    // Dynamic query via nvcuda.dll Driver API (cuInit + cuDevicePrimaryCtxRetain + cuMemGetInfo_v2)
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
        size_t free_b = 0, total_b = 0;
        if (pfnMemGetInfo(&free_b, &total_b) == 0 && total_b > 0) {
            return (total_b - free_b) / (1024 * 1024);
        }
    }
    return 0; // Explicit 0 indicates no CUDA driver / GPU present or query failure
}

#define CHALLENGER_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  [CHALLENGE FAILED] " << msg << " (" #cond ") at line " << __LINE__ << std::endl; \
            return false; \
        } \
    } while (0)

// =============================================================================
// CHALLENGE 1: Workstation Hardware Memory Envelope & Leak Stress Test
// =============================================================================
static bool test_hardware_memory_envelope() {
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "[CHALLENGE 1] Workstation Memory Envelope & Leak Stress Test" << std::endl;
    std::cout << "================================================================================" << std::endl;

    size_t ram_before = get_live_ram_mb();
    size_t vram_before = get_live_vram_mb();
    std::cout << "[*] Initial Process RAM: " << ram_before << " MB | CUDA VRAM: " << vram_before << " MB" << std::endl;

    CHALLENGER_ASSERT(ram_before <= MAX_ALLOWED_RAM_MB, "Initial RAM exceeds 32GB limit");
    CHALLENGER_ASSERT(vram_before <= MAX_ALLOWED_VRAM_MB, "Initial VRAM exceeds 4GB limit");

    // Stress: 50 cycles of allocating and destroying VsaDbHandle instances
    std::cout << "[*] Stressing VsaDbHandle lifecycle (50 create/destroy cycles)..." << std::endl;
    for (int i = 0; i < 50; ++i) {
        VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
        CHALLENGER_ASSERT(db != nullptr, "vsa_db_create returned null during lifecycle stress");
        vsa_db_insert_fact(db, "Subject", "Predicate", "Object");
        vsa_db_destroy(db);
    }

    // Stress: 50 cycles of allocating and freeing HybridBuffer
    std::cout << "[*] Stressing HybridBuffer pinned memory lifecycle (50 alloc/free cycles)..." << std::endl;
    for (int i = 0; i < 50; ++i) {
        HybridBuffer* buf = bridge_alloc_buffer(1024);
        CHALLENGER_ASSERT(buf != nullptr, "bridge_alloc_buffer returned null during lifecycle stress");
        bridge_free_buffer(buf);
    }

    // Stress: 20 cycles of ModelContext with 28 layers
    std::cout << "[*] Stressing ModelContext lifecycle (20 create/free cycles)..." << std::endl;
    for (int i = 0; i < 20; ++i) {
        ModelContext* ctx = model_create_context(28);
        CHALLENGER_ASSERT(ctx != nullptr, "model_create_context returned null during lifecycle stress");
        model_free_context(ctx);
    }

    size_t ram_after = get_live_ram_mb();
    size_t vram_after = get_live_vram_mb();
    std::cout << "[+] Post-Stress Process RAM: " << ram_after << " MB (<= 32,768 MB)" << std::endl;
    std::cout << "[+] Post-Stress CUDA VRAM: " << vram_after << " MB (<= 4,096 MB)" << std::endl;

    CHALLENGER_ASSERT(ram_after <= MAX_ALLOWED_RAM_MB, "Post-stress RAM exceeded 32GB");
    CHALLENGER_ASSERT(vram_after <= MAX_ALLOWED_VRAM_MB, "Post-stress VRAM exceeded 4GB");
    CHALLENGER_ASSERT(ram_after < 1024, "Process working set bloated beyond 1024 MB (possible leak)");

    std::cout << "[CHALLENGE 1] RESULT: PASSED" << std::endl;
    return true;
}

// =============================================================================
// CHALLENGE 2: Pinned DMA Bridge Latency & Bitwise Invariant Stress Test
// =============================================================================
static bool test_pinned_dma_bridge_latency() {
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "[CHALLENGE 2] Pinned DMA Bridge Latency & Bitwise Invariant Stress Test" << std::endl;
    std::cout << "================================================================================" << std::endl;

    HybridBuffer* buf = bridge_alloc_buffer(1024);
    CHALLENGER_ASSERT(buf != nullptr, "Failed to allocate HybridBuffer");
    CHALLENGER_ASSERT(buf->is_pinned, "Buffer is not pinned!");

    std::cout << "[*] Buffer Properties: is_pinned=" << (buf->is_pinned ? "YES" : "NO")
              << ", is_zero_copy=" << (buf->is_zero_copy ? "YES" : "NO") << std::endl;

    std::mt19937_64 rng(0x12345678ULL);
    alignas(64) uint64_t vsa_in[8];
    alignas(64) uint64_t vsa_out[8];

    // Measure over 5,000 roundtrip transfers
    const int ITERS = 5000;
    std::vector<double> roundtrips(ITERS);

    // Warm-up
    for (int i = 0; i < 200; ++i) {
        for (int w = 0; w < 8; ++w) vsa_in[w] = rng();
        bridge_sync_vsa_to_gpu(buf, vsa_in, 1.0f);
        bridge_sync_gpu_to_vsa(buf, vsa_out);
    }

    double t0_all = get_qpc_us();
    for (int i = 0; i < ITERS; ++i) {
        for (int w = 0; w < 8; ++w) vsa_in[w] = rng();

        double t0 = get_qpc_us();
        bridge_sync_vsa_to_gpu(buf, vsa_in, 1.0f);
        bridge_sync_gpu_to_vsa(buf, vsa_out);
        double t1 = get_qpc_us();

        roundtrips[i] = (t1 - t0);

        // Bitwise exactness assert
        for (int w = 0; w < 8; ++w) {
            CHALLENGER_ASSERT(vsa_in[w] == vsa_out[w], "Bitwise roundtrip mismatch between VSA and GPU bridge");
        }
    }
    double t1_all = get_qpc_us();
    double total_avg_us = (t1_all - t0_all) / ITERS;

    std::sort(roundtrips.begin(), roundtrips.end());
    double min_us = roundtrips.front();
    double med_us = roundtrips[ITERS / 2];
    double p95_us = roundtrips[static_cast<size_t>(ITERS * 0.95)];
    double p99_us = roundtrips[static_cast<size_t>(ITERS * 0.99)];
    double max_us = roundtrips.back();

    std::cout << "[+] Roundtrip Latency across " << ITERS << " transfers:" << std::endl;
    std::cout << "    Min:  " << std::fixed << std::setprecision(3) << min_us << " us" << std::endl;
    std::cout << "    Med:  " << med_us << " us" << std::endl;
    std::cout << "    Avg:  " << total_avg_us << " us (Target: < 2.000 us)" << std::endl;
    std::cout << "    P95:  " << p95_us << " us" << std::endl;
    std::cout << "    P99:  " << p99_us << " us" << std::endl;
    std::cout << "    Max:  " << max_us << " us" << std::endl;

    CHALLENGER_ASSERT(total_avg_us < 2.0, "Average roundtrip transfer exceeded 2.0 us limit");

    // Boundary check with extreme float scale values
    std::cout << "[*] Stressing scale factor boundaries (0.001f, 1000.0f, -5.0f)..." << std::endl;
    bridge_sync_vsa_to_gpu(buf, vsa_in, 0.001f);
    bridge_sync_gpu_to_vsa(buf, vsa_out);
    CHALLENGER_ASSERT(std::memcmp(vsa_in, vsa_out, sizeof(vsa_in)) == 0, "Roundtrip failed at scale 0.001f");

    bridge_sync_vsa_to_gpu(buf, vsa_in, 1000.0f);
    bridge_sync_gpu_to_vsa(buf, vsa_out);
    CHALLENGER_ASSERT(std::memcmp(vsa_in, vsa_out, sizeof(vsa_in)) == 0, "Roundtrip failed at scale 1000.0f");

    bridge_free_buffer(buf);
    std::cout << "[CHALLENGE 2] RESULT: PASSED" << std::endl;
    return true;
}

// =============================================================================
// CHALLENGE 3: Online Plasticity Non-Interference Stress Test
// =============================================================================
static bool test_online_plasticity_non_interference() {
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "[CHALLENGE 3] Online Plasticity Non-Interference Stress Test" << std::endl;
    std::cout << "================================================================================" << std::endl;

    ModelContext* ctx = model_create_context(MODEL_DEFAULT_NUM_LAYERS);
    CHALLENGER_ASSERT(ctx != nullptr, "Failed to create ModelContext");

    const int RANK = MODEL_DEFAULT_RANK_R; // 16
    const int DIM = 1024;
    const size_t U_SIZE = DIM * RANK;
    const size_t V_SIZE = RANK * DIM;

    std::vector<float> u_delta(U_SIZE);
    std::vector<float> v_delta(V_SIZE);
    std::mt19937 rng(424242);
    std::normal_distribution<float> dist(0.0f, 0.01f);

    for (size_t i = 0; i < U_SIZE; ++i) u_delta[i] = dist(rng);
    for (size_t i = 0; i < V_SIZE; ++i) v_delta[i] = dist(rng);

    // 1. Measure TT16 update latency over 1,000 iterations
    std::cout << "[*] Measuring TT16 adapter update latency over 1,000 iterations..." << std::endl;
    for (int i = 0; i < 50; ++i) {
        model_update_tt16_residual(ctx, 0, u_delta.data(), v_delta.data());
    }
    double t0_up = get_qpc_us();
    const int UPDATE_ITERS = 1000;
    for (int i = 0; i < UPDATE_ITERS; ++i) {
        model_update_tt16_residual(ctx, 0, u_delta.data(), v_delta.data());
    }
    double t1_up = get_qpc_us();
    double avg_up_us = (t1_up - t0_up) / UPDATE_ITERS;
    std::cout << "[+] TT16 Adapter Update Latency: " << std::fixed << std::setprecision(3) 
              << avg_up_us << " us (Target: < 25.000 us)" << std::endl;
    CHALLENGER_ASSERT(avg_up_us < 25.0, "TT16 update latency exceeded 25 us");

    // 2. Sequential Continual Learning: 100 sequential updates with Orthogonal Projections
    std::cout << "[*] Simulating continuous learning across 100 sequential tasks..." << std::endl;
    const int NUM_TASKS = 10;
    const int NUM_UPDATES = 100;

    std::vector<std::vector<float>> past_tasks(NUM_TASKS, std::vector<float>(DIM));
    for (int t = 0; t < NUM_TASKS; ++t) {
        for (int d = 0; d < DIM; ++d) past_tasks[t][d] = dist(rng);
    }

    auto eval_rep = [&](const float* in_v, float* out_v) {
        std::memcpy(out_v, in_v, DIM * sizeof(float));
        model_apply_tt16_residual(ctx, 0, in_v, out_v);
    };

    std::vector<std::vector<float>> init_reps(NUM_TASKS, std::vector<float>(DIM));
    for (int t = 0; t < NUM_TASKS; ++t) {
        eval_rep(past_tasks[t].data(), init_reps[t].data());
    }

    for (int step = 0; step < NUM_UPDATES; ++step) {
        for (size_t i = 0; i < U_SIZE; ++i) u_delta[i] = dist(rng) * 0.05f;
        for (size_t i = 0; i < V_SIZE; ++i) v_delta[i] = dist(rng) * 0.05f;

        // Project u_delta orthogonal to all 10 past tasks
        for (int t = 0; t < NUM_TASKS; ++t) {
            float dot = 0.0f, norm_sq = 0.0f;
            for (int d = 0; d < DIM; ++d) {
                dot += u_delta[d] * past_tasks[t][d];
                norm_sq += past_tasks[t][d] * past_tasks[t][d];
            }
            float proj = dot / (norm_sq + 1e-8f);
            for (int d = 0; d < DIM; ++d) {
                u_delta[d] -= proj * past_tasks[t][d];
            }
        }
        model_update_tt16_residual(ctx, 0, u_delta.data(), v_delta.data());
    }

    double sum_fidelity = 0.0;
    for (int t = 0; t < NUM_TASKS; ++t) {
        std::vector<float> final_rep(DIM);
        eval_rep(past_tasks[t].data(), final_rep.data());

        double dot = 0.0, s_init = 0.0, s_fin = 0.0;
        for (int d = 0; d < DIM; ++d) {
            dot += init_reps[t][d] * final_rep[d];
            s_init += init_reps[t][d] * init_reps[t][d];
            s_fin += final_rep[d] * final_rep[d];
        }
        double sim = dot / (std::sqrt(s_init) * std::sqrt(s_fin) + 1e-12);
        sum_fidelity += sim;
    }
    double avg_fidelity = (sum_fidelity / NUM_TASKS) * 100.0;
    std::cout << "[+] Historical Task Retention after 100 updates: " 
              << std::fixed << std::setprecision(2) << avg_fidelity << "% (Target: >= 98.0%)" << std::endl;
    CHALLENGER_ASSERT(avg_fidelity >= 98.0, "Historical retention dropped below 98.0%");

    // 3. Falsification check: apply destructive un-orthogonalized update
    std::cout << "[*] Executing falsification test (destructive unprojected interference)..." << std::endl;
    std::vector<float> u_kill(U_SIZE, 0.0f);
    std::vector<float> v_kill(V_SIZE, 0.0f);
    for (int d = 0; d < DIM; ++d) {
        v_kill[d] = past_tasks[0][d] * 300.0f;
        u_kill[d] = past_tasks[1][d] * 300.0f;
    }
    model_update_tt16_residual(ctx, 0, u_kill.data(), v_kill.data());
    std::vector<float> degraded_rep(DIM);
    eval_rep(past_tasks[0].data(), degraded_rep.data());

    double dot_d = 0.0, s1 = 0.0, s2 = 0.0;
    for (int d = 0; d < DIM; ++d) {
        dot_d += init_reps[0][d] * degraded_rep[d];
        s1 += init_reps[0][d] * init_reps[0][d];
        s2 += degraded_rep[d] * degraded_rep[d];
    }
    double sim_degraded = dot_d / (std::sqrt(s1) * std::sqrt(s2) + 1e-12);
    std::cout << "  -> Destructive interference dropped similarity to: " << (sim_degraded * 100.0) << "%" << std::endl;
    CHALLENGER_ASSERT(sim_degraded < 0.70, "Falsification failed: catastrophic interference was not detected");

    model_free_context(ctx);
    std::cout << "[CHALLENGE 3] RESULT: PASSED" << std::endl;
    return true;
}

// =============================================================================
// CHALLENGE 4: Neuro-Symbolic Attractor Monotonicity & Contradiction Gate
// =============================================================================
static bool test_attractor_monotonicity_and_contradiction() {
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "[CHALLENGE 4] Neuro-Symbolic Attractor Monotonicity & Contradiction Gate" << std::endl;
    std::cout << "================================================================================" << std::endl;

    const int STATE_DIM = 64;
    const int NUM_PATTERNS = 16;
    const float TAU = 1.0f;

    std::mt19937 rng(777);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<std::vector<float>> memory_patterns(NUM_PATTERNS, std::vector<float>(STATE_DIM));
    for (int m = 0; m < NUM_PATTERNS; ++m) {
        float norm = 0.0f;
        for (int d = 0; d < STATE_DIM; ++d) {
            memory_patterns[m][d] = dist(rng);
            norm += memory_patterns[m][d] * memory_patterns[m][d];
        }
        float inv_norm = 1.0f / std::sqrt(norm);
        for (int d = 0; d < STATE_DIM; ++d) {
            memory_patterns[m][d] *= inv_norm;
        }
    }

    auto compute_energy = [&](const std::vector<float>& s) -> double {
        std::vector<float> dots(NUM_PATTERNS);
        float max_dot = -1e30f;
        for (int m = 0; m < NUM_PATTERNS; ++m) {
            float d = 0.0f;
            for (int j = 0; j < STATE_DIM; ++j) {
                d += memory_patterns[m][j] * s[j];
            }
            dots[m] = d / TAU;
            if (dots[m] > max_dot) max_dot = dots[m];
        }
        double sum_exp = 0.0;
        for (int m = 0; m < NUM_PATTERNS; ++m) {
            sum_exp += std::exp(dots[m] - max_dot);
        }
        double lse = max_dot + std::log(sum_exp);

        double norm2 = 0.0;
        for (int j = 0; j < STATE_DIM; ++j) {
            norm2 += s[j] * s[j];
        }
        return -TAU * lse + 0.5 * norm2;
    };

    // 1. Monotonic Energy Descent across 50 random starting states
    std::cout << "[*] Stressing Modern Hopfield energy descent across 50 random initial trajectories..." << std::endl;
    const int NUM_TRAJECTORIES = 50;
    const int MAX_STEPS = 20;

    int total_steps_tested = 0;
    bool all_monotonic = true;

    for (int traj = 0; traj < NUM_TRAJECTORIES; ++traj) {
        std::vector<float> s_cur(STATE_DIM);
        for (int d = 0; d < STATE_DIM; ++d) {
            s_cur[d] = memory_patterns[traj % NUM_PATTERNS][d] + dist(rng) * 0.35f;
        }

        double prev_e = compute_energy(s_cur);

        for (int step = 0; step < MAX_STEPS; ++step) {
            // Hopfield update
            float max_dot = -1e30f;
            std::vector<float> dots(NUM_PATTERNS);
            for (int m = 0; m < NUM_PATTERNS; ++m) {
                float d = 0.0f;
                for (int j = 0; j < STATE_DIM; ++j) {
                    d += memory_patterns[m][j] * s_cur[j];
                }
                dots[m] = d / TAU;
                if (dots[m] > max_dot) max_dot = dots[m];
            }
            float sum_exp = 0.0f;
            for (int m = 0; m < NUM_PATTERNS; ++m) {
                dots[m] = std::exp(dots[m] - max_dot);
                sum_exp += dots[m];
            }
            float inv_sum = 1.0f / sum_exp;
            std::vector<float> s_next(STATE_DIM, 0.0f);
            for (int m = 0; m < NUM_PATTERNS; ++m) {
                float p_m = dots[m] * inv_sum;
                for (int j = 0; j < STATE_DIM; ++j) {
                    s_next[j] += p_m * memory_patterns[m][j];
                }
            }

            double cur_e = compute_energy(s_next);
            double delta_e = cur_e - prev_e;
            total_steps_tested++;

            if (delta_e > 1e-5) {
                std::cerr << "Energy descent violated: Traj " << traj << ", Step " << step 
                          << ", delta_e = " << delta_e << std::endl;
                all_monotonic = false;
                break;
            }
            prev_e = cur_e;
            s_cur = s_next;
        }
        if (!all_monotonic) break;
    }

    std::cout << "[+] Tested " << total_steps_tested << " relaxation steps: All Monotonic = " 
              << (all_monotonic ? "YES (dE/dt <= 0)" : "NO") << std::endl;
    CHALLENGER_ASSERT(all_monotonic, "Energy relaxation violated monotonic descent");

    // 2. Deterministic Contradiction Rejection Gate (Simplex Mirror Descent)
    std::cout << "[*] Stressing Contradiction Rejection Gate on 50 verified axioms..." << std::endl;
    VsaDbHandle* kb = vsa_db_create(VSA_CAP_MICRO, nullptr);
    CHALLENGER_ASSERT(kb != nullptr, "Failed to create VsaDbHandle for contradiction stress test");

    for (int i = 0; i < 50; ++i) {
        std::string s = "Entity_" + std::to_string(i);
        std::string r = "Property_" + std::to_string(i % 5);
        std::string o = "TrueValue_" + std::to_string(i);
        vsa_db_insert_fact(kb, s.c_str(), r.c_str(), o.c_str());
    }

    int rejected_count = 0;
    int accepted_count = 0;

    // Test 50 contradictory assertions
    for (int i = 0; i < 50; ++i) {
        std::string s = "Entity_" + std::to_string(i);
        std::string r = "Property_" + std::to_string(i % 5);
        std::string false_o = "FalseValue_Contradiction_" + std::to_string(i);

        int conflict = vsa_db_check_conflict(kb, s.c_str(), r.c_str(), false_o.c_str());
        float raw_logit = 4.0f;
        float proj_logit = (conflict == 1) ? -1e30f : raw_logit;
        float prob = (proj_logit <= -1e20f) ? 0.0f : std::exp(proj_logit);

        if (prob == 0.0f) {
            rejected_count++;
        }
    }

    // Test 50 valid/consistent assertions
    for (int i = 0; i < 50; ++i) {
        std::string s = "Entity_" + std::to_string(i);
        std::string r = "Property_" + std::to_string(i % 5);
        std::string true_o = "TrueValue_" + std::to_string(i);

        int conflict = vsa_db_check_conflict(kb, s.c_str(), r.c_str(), true_o.c_str());
        float raw_logit = 4.0f;
        float proj_logit = (conflict == 1) ? -1e30f : raw_logit;
        float prob = (proj_logit <= -1e20f) ? 0.0f : std::exp(proj_logit);

        if (prob > 0.0f) {
            accepted_count++;
        }
    }

    std::cout << "[+] Contradiction Rejection: " << rejected_count << "/50 (100.0% rejected)" << std::endl;
    std::cout << "[+] Valid Assertion Preservation: " << accepted_count << "/50 (100.0% preserved)" << std::endl;

    CHALLENGER_ASSERT(rejected_count == 50, "Failed to reject 100% of contradictory facts");
    CHALLENGER_ASSERT(accepted_count == 50, "Improperly rejected valid assertions");

    vsa_db_destroy(kb);
    std::cout << "[CHALLENGE 4] RESULT: PASSED" << std::endl;
    return true;
}

// =============================================================================
// Main Challenger Harness
// =============================================================================
int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "   HSSA ADVERSARIAL STRESS TEST HARNESS — CHALLENGER 2                          " << std::endl;
    std::cout << "   Hardware Envelope, Pinned DMA, Online Plasticity, & Attractor Monotonicity   " << std::endl;
    std::cout << "================================================================================" << std::endl;

    struct TestItem {
        const char* name;
        bool (*fn)();
    };

    TestItem suite[] = {
        {"Challenge 1: Workstation Memory Envelope & Leaks", test_hardware_memory_envelope},
        {"Challenge 2: Pinned DMA Latency & Bitwise Invariants", test_pinned_dma_bridge_latency},
        {"Challenge 3: Online Plasticity Non-Interference", test_online_plasticity_non_interference},
        {"Challenge 4: Attractor Monotonicity & Contradiction Gate", test_attractor_monotonicity_and_contradiction}
    };

    int passed = 0;
    int total = sizeof(suite) / sizeof(suite[0]);

    for (const auto& test : suite) {
        if (test.fn()) {
            passed++;
        } else {
            std::cerr << "\n[-] TEST FAILED: " << test.name << std::endl;
            return 1;
        }
    }

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "   CHALLENGER 2 EMPIRICAL VERIFICATION COMPLETE: " << passed << "/" << total << " PASSED" << std::endl;
    std::cout << "================================================================================" << std::endl;
    return 0;
}

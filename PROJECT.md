# Project: Native C++/CUDA Hybrid Neural Research Laboratory (`project upgrade`)

## Architecture
A pure native Windows C++/CUDA laboratory running on workstation hardware (Intel Core i7-11800H + NVIDIA RTX 3050 Laptop 4GB) with zero external Python/PyTorch dependencies. The architecture integrates:
1. **AVX-512 VSA Associative Memory Core (`second_brain`)**: Sub-microsecond 512-bit hypervector associative memory using AVX-512F/BW/DQ/VL + VPOPCNTDQ ternary dot product, XNOR binding, and circular permutation.
2. **CUDA sm_86 Quantized Tensor Engine (`legacy_ptq`)**: Sub-562MB 4-bit BSL-32T GEMV with 128B coalesced warp loads, parallel shared-memory LUT, fused Tensor Train (TT16) low-rank adapters, and dynamic precision switching (2/3/4-bit and speculative planes).
3. **Zero-Copy / Pinned DMA Hybrid Bridge (`core/bridge`)**: 64-byte aligned host-pinned memory bridge (`cuMemAllocHost` / `VirtualAlloc`) streaming intermediate hidden vectors and associative VSA memories between CPU and GPU in <2 microseconds.
4. **Dynamic Non-Fixed Knowledge Base Subsystem (`core/vsa_db`)**: Fully dynamic, multi-instance memory bases with configurable capacity (4MB to 1GB), online insertion/mutation/deletion, instant hot-swapping, and persistent storage without data loss.
5. **Architecture Experimentation Subsystem (`core/model_mod`)**: Programmable runtime API for mutating weights, altering quantization modes, and swapping execution blocks (GQA vs Gated DeltaNet vs AVX-512 VSA lookup).
6. **Automated Experiment Runner & Empirical Benchmarking Suite (`core/bench_runner`)**: High-precision evaluation harness measuring Win32 QPC latency (us), throughput (tok/s), host RAM (<= 32GB), GPU VRAM (<= 4096MB), loss/cosine deltas, and boundary conditions.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | F1.1 Unified Build Orchestrator | MSVC 2022 vcvars64 detection, NVCC cubin generation, /arch:AVX512 /openmp, zero Python | M1 | Survey |
| 2 | F1.2 CUDA sm_86 Tensor Engine | Verified cubin compilation (`kernel.cubin`), Driver API loading, BSL-32T GEMV execution | M1 | Survey |
| 3 | F1.3 AVX-512 VSA Computational Core | 64-byte aligned SIMD kernels (`ternary_popcnt_dot_512`, `binary_xnor_popcnt_512`, bit-packing) | M1 | Survey |
| 4 | F1.4 Hybrid CPU-GPU Interop Bridge | 64B pinned DMA memory bridge for continuous CPU VSA -> GPU tensor forward pass | M1 | Survey |
| 5 | F2.1 Dynamic VsaMemoryDatabase | Multi-instance memory database with configurable size (4MB-1GB) and safe persistence | M2 | Survey |
| 6 | F2.2 True VSA Fact Engine | 512-bit hypervector binding/unbinding, Robin Hood/cuckoo indexing, eliminating bypass veneer | M2 | Survey |
| 7 | F2.3 Online Memory Mutation API | Dynamic insertion, mutation, deletion/unindexing, and flushing without process restart | M2 | Survey |
| 8 | F2.4 Multi-Instance Hot-Swapping | Runtime creation, branching, and switching between isolated memory bases with zero leaks | M2 | Survey |
| 9 | F3.1 Dynamic Precision Switching | Runtime switching between 2-bit, 3-bit, 4-bit, and speculative 24/34 delta planes | M3 | Survey |
| 10 | F3.2 Online Weight Plasticity | TT16 low-rank adapter updates via SVD/SGD over PCIe in <25 us without touching base planes | M3 | Survey |
| 11 | F3.3 Layer Block Polymorphism | Hot-swapping execution blocks (GQA vs Gated DeltaNet vs AVX-512 VSA associative memory) | M3 | Survey |
| 12 | F3.4 Architecture Modification API | Programmable C/C++ API for model mutation and delta verification | M3 | Survey |
| 13 | F4.1 Microsecond Profiler | Win32 QPC step latency profiling and throughput (tok/s) measurement | M4 | Survey |
| 18 | F5.1 HSSA Autoregressive Decode | `hssa_generate` greedy/temperature decode, 1000-token working-set delta 0.0078 MB, 264 us/token (~3800 tok/s) | M3 | Empirical |
| 19 | F5.2 HSSA Contradiction Gate | Absolute veto: contradicted tokens keep p == 0.000 in every case, incl. the fully-degenerate one | M3 | Empirical |
| 20 | F5.3 HSSA Registry Growth | Assertion registry grows lazily 256 -> 16384 so an idle context pays nothing for it | M3 | Empirical |
| 21 | F5.4 CPU-First GPU Policy | sm_86 offload is opt-in (`hssa --cuda`); default is AVX-512 CPU. Rationale: a 1024x1024 GEMV is ~2.1 MFLOP so launch/sync/PCIe dominate the ~2 us of compute | M2/M3 | Empirical |
| 22 | F5.5 CLI Corpus Training | `popcnt_brain.exe --train <corpus.txt> [--epochs N] [--seed N]` teaches the STE engine from arbitrary text; reports bytes/vocab/tokens/epochs/ms/tok-per-s/errors | M3 | Empirical |
| 23 | F6.1 Honest Corpus Metrics | `live_replay_errors` measured by re-running the trained model autoregressively (in-loop `final_errors` used frozen pre-update states and under-reported by ~100 of ~800); denominator is `train_transitions`, not word tokens | M3 | Empirical |
| 24 | F6.2 Epoch-Consistent Feature Cache | `train_arbitrary_corpus_end_to_end` rebuilds the transition cache at every epoch (`collect_transitions`), so experts are keyed on states the current weights reproduce; router centroids are learned only on the first pass to avoid retiring trained pages | M3 | Empirical |
| 25 | F6.3 Encoder Unfrozen | STE gradients reach MLGRU `wc_master` on **every** epoch (was `ep == 0` only), which had frozen the encoder and plateaued the error count at 781 from 6 epochs to 96 | M3 | Empirical |
| 26 | F6.4 Relative Activation Band | `execute_sparse_micro_experts` gate is `max_h - 256` whenever a routed page is live; the old absolute floor 600 cut every neuron off whenever the achievable dot fell below it (measured `mean_max_h = -12710`) | M3 | Empirical |
| 27 | F7.1 Honest Recall Semantics | `src/hssa.cpp` and `src/model_mod.cpp` seeded recall `similarity = 1.0f` and echoed the query back as `best_vec` on every miss / empty DB, fabricating certainty and injecting a gamma-weighted projection of the query into the memory state. Both now report 0.0f on a miss; functionally proven (no-db 0.0, empty-db 0.0, miss -0.027, exact hit 0.984) | M1/M3 | Empirical |
| 28 | F7.2 Kernel Single Source of Truth | `dual_rail_binarize` is declared in `include/hssa.h` and no longer `static`; `tests/test_hssa_m1.cpp` used to carry a private byte-identical copy (`simd_psi_binarize`), which detached its latency and round-trip assertions from the shipped kernel | M1 | Structural |
| 14 | F4.2 Workstation Hardware Tracker | Win32 RAM working set (<=32GB) and CUDA Driver API VRAM (<=4096MB) ceiling assertion | M4 | Survey |
| 15 | F4.3 Numerical Accuracy Metrics | Cosine similarity, SQNR dB, and perplexity/loss delta measurement | M4 | Survey |
| 16 | F4.4 Boundary & Adversarial Suite | 0-1-N-NULL-BOUND boundary stress testing and MUTATION FLIP verification | M4 | Survey |
| 17 | F5.1 E2E Test Suite & Test Runner | Standalone test runner covering Tiers 1-4 requirements (Opaque-box) | M_TEST | Survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M_TEST | E2E Testing Track | Test infra, runner, test cases (Tiers 1-4), publishes TEST_READY.md | none | DONE |
| M1 | Native Computation Core & Bridge | Build orchestrators (build.bat, CMakeLists.txt), kernel.cubin, AVX-512 core, hybrid bridge | none | DONE |
| M2 | Dynamic Knowledge & Memory Base | VsaMemoryDatabase, true VSA indexing, online mutation, hot-swapping | M1 | DONE |
| M3 | Architecture Experimentation Subsystem | Dynamic quantization switching, TT16 plasticity, layer swapping | M1 | DONE |
| M4 | Automated Benchmark & Lab CLI | QPC profiling, RAM/VRAM tracking, empirical metrics, unified upgrade_lab CLI | M1, M2, M3 | DONE |
| M_FINAL | Final Acceptance & Adversarial Hardening | 100% E2E test pass (Tiers 1-4) + Tier 5 adversarial coverage hardening + Forensic Audit | M4, M_TEST | DONE |

## Interface Contracts

### 1. `core/bridge.h` (`second_brain` ↔ `legacy_ptq`)
```cpp
#pragma once
#include <cstdint>
#include <cstddef>

#define VSA_DIM 512
#define VSA_WORDS (VSA_DIM / 64)       // 8 x uint64_t = 64 bytes (1 cache line / 1 __m512i)
#define HIDDEN_DIM 1024
#define TENSOR_ALIGNMENT 64

struct HybridBuffer {
    alignas(64) uint64_t vsa_vec[VSA_WORDS];       // 512-bit binary hypervector
    alignas(64) float    hidden_cpu[HIDDEN_DIM];    // 1024-float pinned host buffer
    void*                d_hidden_gpu;              // CUdeviceptr on RTX 3050
    size_t               bytes_allocated;
    bool                 is_pinned;
};

// Allocate and register page-locked pinned memory for zero-copy / fast DMA
HybridBuffer* bridge_alloc_buffer(size_t hidden_dim);
void bridge_free_buffer(HybridBuffer* buf);

// Forward cycle transfer: inject AVX-512 VSA hypervector into GPU hidden state
int bridge_sync_vsa_to_gpu(HybridBuffer* buf, const uint64_t* vsa_vec, float scale);

// Retrieve GPU hidden state back to CPU for associative retrieval
int bridge_sync_gpu_to_vsa(HybridBuffer* buf, uint64_t* out_vsa_vec);
```

### 2. `core/vsa_db.h` (Dynamic Memory Base API)
```cpp
#pragma once
#include <cstdint>
#include <cstddef>

typedef struct VsaDbHandle VsaDbHandle;

enum VsaDbCapacity {
    VSA_CAP_MICRO  = 1024,    // 1K pages = 4MB (L3-cache resident)
    VSA_CAP_MEDIUM = 16384,   // 16K pages = 64MB
    VSA_CAP_LARGE  = 262144   // 262K pages = 1GB
};

// Lifecycle
VsaDbHandle* vsa_db_create(VsaDbCapacity cap, const char* persist_path);
void vsa_db_destroy(VsaDbHandle* db);
int vsa_db_flush(VsaDbHandle* db);
int vsa_db_clone(const VsaDbHandle* src, VsaDbHandle** dst, const char* new_persist_path);

// Online Mutation & Retrieval
int vsa_db_insert_fact(VsaDbHandle* db, const char* subj, const char* rel, const char* obj);
int vsa_db_delete_fact(VsaDbHandle* db, const char* subj, const char* rel);
int vsa_db_mutate_fact(VsaDbHandle* db, const char* subj, const char* rel, const char* new_obj);
int vsa_db_query_relation(VsaDbHandle* db, const char* subj, const char* rel, char* out_obj, size_t max_len);
int vsa_db_check_conflict(VsaDbHandle* db, const char* subj, const char* rel, const char* obj);

// Raw 512-bit Vector Operations
int vsa_db_insert_vector(VsaDbHandle* db, uint64_t key_hash, const uint64_t* vec_512);
int vsa_db_query_vector(VsaDbHandle* db, const uint64_t* query_512, uint64_t* out_best_vec, float* out_sim);
```

### 3. `core/model_mod.h` (Architecture Experimentation & Mutation API)
```cpp
#pragma once
#include <cstdint>

enum QuantPrecision {
    QUANT_2BIT = 2,
    QUANT_3BIT = 3,
    QUANT_4BIT = 4,
    QUANT_DELTA_24 = 24,
    QUANT_DELTA_34 = 34
};

enum LayerType {
    LAYER_GQA_ATTENTION,
    LAYER_GATED_DELTANET,
    LAYER_AVX512_VSA_ASSOCIATIVE
};

// Runtime precision modulation without VRAM reallocation
int model_set_layer_precision(void* ctx, uint32_t layer_idx, QuantPrecision prec);

// Online low-rank plasticity: update TT16 continuous cores over PCIe in <25 us
int model_update_tt16_residual(void* ctx, uint32_t layer_idx, const float* u_delta, const float* v_delta);

// Layer execution block hot-swapping
int model_swap_layer_block(void* ctx, uint32_t layer_idx, LayerType new_type);
```

### 4. `core/bench_runner.h` (Profiling & Hardware Metrics API)
```cpp
#pragma once
#include <cstdint>

struct BenchmarkMetrics {
    double latency_us;          // Microsecond execution time (Win32 QPC)
    double tokens_per_sec;      // Forward throughput
    size_t host_ram_working_set_mb; // Win32 GetProcessMemoryInfo (<= 32,768 MB)
    size_t gpu_vram_used_mb;    // CUDA Driver API cuMemGetInfo (<= 4,096 MB)
    float  cosine_similarity;   // Numerical accuracy
    float  sqnr_db;             // Signal-to-quantization-noise ratio
};

int bench_run_step_profile(void* ctx, BenchmarkMetrics* out_metrics);
int bench_assert_hardware_bounds(size_t max_ram_mb, size_t max_vram_mb);
int bench_run_adversarial_matrix();
```

## Code Layout
```
project upgrade/
├── CMakeLists.txt                 # Unified root CMake config
├── build.bat                      # Unified root build script (MSVC detection, NVCC, cl.exe)
├── PROJECT.md                     # Global project index & architectural specification
├── TEST_INFRA.md                  # E2E Test Suite specification & feature coverage matrix
├── include/
│   ├── bridge.h                   # CPU AVX-512 <-> GPU CUDA DMA bridge
│   ├── vsa_db.h                   # Dynamic memory database API
│   ├── model_mod.h                # Architecture experimentation & mutation API
│   ├── hssa.h                     # HSSA engine API (M1-M3: lifecycle, forward, simplex gate, generation)
│   └── bench_runner.h             # Benchmark runner and hardware profiler API
├── src/
│   ├── bridge.cpp                 # Memory bridge implementation
│   ├── vsa_db.cpp                 # Dynamic memory database implementation
│   ├── model_mod.cpp              # Architecture mutation implementation
│   ├── hssa.cpp                   # HSSA engine implementation (M1-M3)
│   ├── bench_runner.cpp           # Profiling and metric validation implementation
│   └── main_lab.cpp               # Unified native upgrade_lab CLI
├── legacy_ptq/                    # CUDA sm_86 kernel and BSL-32T engine
│   ├── core.h
│   ├── kernel.cu                  # Native sm_86 CUDA kernels (compiled to kernel.cubin)
│   ├── ptq_cuda.c                 # CUDA Driver API host driver (static lib)
│   ├── ptq_cuda.h                 # PTQ CUDA engine public API
│   └── engine.c                   # Legacy PTQ demo host pipeline
├── second_brain/                  # AVX-512 VSA associative memory
│   ├── popcnt_brain.h
│   ├── popcnt_brain.cpp           # 512-bit SIMD kernels
│   └── test_c_api.cpp             # C-ABI integration test
├── tests/
│   ├── e2e_test_runner.cpp        # Standalone native E2E test harness
│   ├── tier1_feature_tests.cpp    # Tier 1 tests (5 per feature = 80 tests)
│   ├── tier2_boundary_tests.cpp   # Tier 2 boundary/corner cases (80 tests)
│   ├── tier3_pairwise_tests.cpp   # Tier 3 cross-feature combinations (16 tests)
│   ├── tier4_workload_tests.cpp   # Tier 4 real-world application scenarios (8 tests)
│   ├── test_hssa.cpp             # HSSA M1 conformance suite
│   ├── test_hssa_m1.cpp           # HSSA M1 adversarial (O(1) state invariance, 0-1-N-NULL-BOUND)
│   ├── test_hssa_m2.cpp           # HSSA M2 adversarial (pinned zero-copy DMA, sm_86 GEMV)
│   ├── test_hssa_m3.cpp           # HSSA M3 adversarial (neuro-symbolic gate, autoregressive decode)
│   └── build_hssa_m*_tests.bat    # Per-milestone build/run orchestrators
└── build/                         # Output directory for bin, obj, cubin
```

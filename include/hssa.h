#ifndef HSSA_H
#define HSSA_H

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "vsa_db.h"
#include "bridge.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * =============================================================================
 * Milestone M1 & M2: Holographic State-Space Attractor (HSSA) Architecture Engine
 * Clean C-ABI / C++20 Interface (AVX-512 / sm_86 / Win32)
 * =============================================================================
 */

#define HSSA_DEFAULT_NUM_LAYERS  28
#define HSSA_HIDDEN_DIM          1024
#define HSSA_HEADS               8
#define HSSA_HEAD_DIM            128
#define HSSA_VSA_DIM             512
#define HSSA_VSA_WORDS           (HSSA_VSA_DIM / 64)       /* 8 x uint64_t = 64 bytes = 1 __m512i */
#define HSSA_RANK                16
#define HSSA_ALIGNMENT           64

#define HSSA_TT16_CORE_SIZE      (HSSA_RANK * HSSA_HIDDEN_DIM)                /* 16 * 1024 = 16,384 floats = 64 KB */
#define HSSA_DELTANET_STATE_SIZE (HSSA_HEADS * HSSA_HEAD_DIM * HSSA_HEAD_DIM) /* 8 * 128 * 128 = 131,072 floats = 512 KB */
#define HSSA_MAX_ASSERTIONS      16384
#define HSSA_MAX_PATTERNS        16

/* Configuration Descriptor for HSSA Engine */
typedef struct HssaConfig {
    uint32_t num_layers;
    uint32_t hidden_dim;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t vsa_dim;
    uint32_t rank;
} HssaConfig;

/* Per-layer HSSA Execution Descriptor */
typedef struct HssaLayer {
    uint32_t     layer_idx;
    float*       deltanet_state; /* [8 * 128 * 128] 64-byte aligned (131,072 floats = 512 KB) */
    float*       tt_u;           /* [16 * 1024] 64-byte aligned (16,384 floats = 64 KB) */
    float*       tt_v;           /* [16 * 1024] 64-byte aligned (16,384 floats = 64 KB) */
    VsaDbHandle* vsa_handle;     /* Layer-specific episodic memory or NULL */
    void*        gpu_matrix;     /* PtqGpuMatrix* on device or NULL */
    void*        reserved[3];
} HssaLayer;

/* Candidate Assertion Descriptor for Attractor Simplex Projection */
typedef struct HssaTokenAssertion {
    uint32_t token_id;
    char     subject[64];
    char     relation[64];
    char     object[64];
    bool     active;
} HssaTokenAssertion;

/* Telemetry & Introspection Metrics */
typedef struct HssaTelemetry {
    double   total_forward_us;
    double   ssm_recurrence_us;
    double   binarization_ns;
    double   vsa_recall_us;
    double   hopfield_relax_us;
    double   hopfield_initial_energy;
    double   hopfield_final_energy;
    double   hopfield_energy_delta;
    uint32_t hopfield_steps;
    uint32_t rejected_contradictions;
    size_t   state_memory_bytes;
    float    vsa_similarity;

    /* Milestone M2 CUDA DMA and GPU GEMV Telemetry */
    double   cuda_dma_fwd_us;
    double   cuda_gemv_us;
    double   cuda_dma_rev_us;
    bool     cuda_executed;
} HssaTelemetry;

/* Autoregressive Generation Telemetry & Introspection (Milestone M3) */
typedef struct HssaGenStats {
    double   total_time_ms;
    double   avg_tok_us;
    uint32_t tokens_generated;
    uint32_t contradictions_blocked;
} HssaGenStats;

/* Global HSSA Runtime Context */
typedef struct HssaContext {
    HssaConfig          config;
    HssaLayer*          layers;
    VsaDbHandle*        global_vsa_db;
    bool                owns_global_vsa_db;

    /* Hybrid CPU-GPU Zero-Copy DMA Bridge */
    HybridBuffer*       hybrid_bridge;
    bool                cuda_active;
    void*               cuda_drv;      /* CudaDrv* or NULL */
    void*               cuda_ctx;      /* CUcontext or NULL */
    void*               cuda_kernels;  /* PtqKernels* or NULL */
    void*               gpu_matrix;    /* PtqGpuMatrix* or NULL */
    uint64_t            gpu_partials;  /* CUdeviceptr for Split-K staging */
    uint64_t            gpu_tile_cnt;  /* CUdeviceptr for Split-K tile counter */
    uint64_t            gpu_out_buf;   /* CUdeviceptr for GEMV output */

    /* Preallocated 64-byte aligned hot-path scratchpads */
    float*              scratch_in;        /* [1024] */
    float*              scratch_out;       /* [1024] */
    float*              scratch_state;     /* [1024] */
    uint64_t*           scratch_vsa;       /* [8] */

    /* Modern Hopfield Attractor Memory Patterns */
    size_t              num_attractor_patterns;
    float*              attractor_patterns; /* [16 * 1024] 64-byte aligned */
    float               hopfield_tau;

    /* Registered Candidate Assertions for Simplex Projection */
    HssaTokenAssertion* assertions;
    uint32_t            assertion_count;
    uint32_t            assertion_capacity;

    /* Milestone M3: Unembedding Projection & Autoregressive Decoding */
    float*              unembed_weights;    /* [vocab_size * 1024] 64-byte aligned */
    float*              logits_buf;         /* [vocab_size] 64-byte aligned */
    size_t              vocab_size;

    /* Serializes forward passes against TT16 core updates. The scratchpads and
     * tt_u/tt_v are single-instance mutable state; concurrent forward/backward
     * calls on one context raced before this lock existed. C-ABI safe: points
     * at a C++ std::mutex owned by hssa_create. */
    void*               state_lock;

    void*               reserved[8];
} HssaContext;

/*
 * =============================================================================
 * Core Lifecycle & Forward Functions
 * =============================================================================
 */

/*
 * Create and initialize multi-layer HSSA runtime context.
 * Preallocates all 64-byte aligned recurrent states (512 KB per layer)
 * and scratchpads to guarantee zero heap allocations on the hot path.
 */
HssaContext* hssa_create(uint32_t num_layers);

/*
 * Release all resources, aligned buffers, CUDA mappings, and associated databases.
 */
void hssa_destroy(HssaContext* ctx);

/*
 * Reset all recurrent DeltaNet states to zero across all layers.
 */
int hssa_reset_state(HssaContext* ctx);

/*
 * Execute a single forward token cycle through the full HSSA engine:
 * 1. Multi-layer Gated DeltaNet state-space recurrence
 * 2. Dual-rail SIMD binarization Psi: R^1024 -> {0, 1}^512 in < 20 ns
 * 3. Episodic associative recall via vsa_db_query_vector
 * 4. Continuous float projection Phi and residual injection
 * 5. Modern Continuous Hopfield energy relaxation
 * 6. (Optional) GPU GEMV projection acceleration when CUDA is active
 *
 * Parameters:
 *   ctx:       Valid HssaContext pointer
 *   in_vec:    Input token embedding [1024]
 *   out_vec:   Output contextualized vector [1024]
 *   out_telem: Optional telemetry descriptor pointer (can be NULL)
 */
int hssa_forward_token(HssaContext* ctx, const float* in_vec, float* out_vec, HssaTelemetry* out_telem);

/*
 * Execute Attractor Simplex Mirror Descent logit projection:
 * Evaluates candidate assertions against the active VSA knowledge base;
 * sets conflicting token logits to -1e9f (contradiction nullification)
 * and performs stable log-sum-exp softmax normalization.
 *
 * Parameters:
 *   ctx:        Valid HssaContext pointer
 *   logits:     Input/Output token logits array [vocab_size]
 *   vocab_size: Number of tokens in vocabulary
 */
int hssa_project_simplex(HssaContext* ctx, float* logits, size_t vocab_size);

/*
 * Mutate or insert a verified factual assertion into the active VSA knowledge base:
 * T_fact = rho^1(v_S) XNOR rho^2(v_R) XNOR rho^3(v_O)
 * Operates in O(1) time (< 2 us) without modifying base neural weights.
 */
int hssa_mutate_knowledge(HssaContext* ctx, const char* subj, const char* rel, const char* obj);

/*
 * =============================================================================
 * Milestone M2: CUDA Acceleration APIs
 * =============================================================================
 */

/*
 * Initialize CUDA driver, CUcontext, PtqKernels, and HybridBuffer zero-copy DMA bridge.
 * Returns 0 on success, negative error code if CUDA hardware/driver unavailable.
 */
int  hssa_init_cuda(HssaContext* ctx);

/*
 * Enable or disable CUDA device offloading at runtime without reallocation.
 */
int  hssa_enable_cuda(HssaContext* ctx, bool enable);

/*
 * Query active CUDA acceleration status.
 */
bool hssa_is_cuda_active(const HssaContext* ctx);

/*
 * =============================================================================
 * Milestone M3: Continuous Neuro-Symbolic Attractor Gate & Autoregressive Decoding
 * =============================================================================
 */

/*
 * Set or upload unembedding projection weight matrix [vocab_size x 1024].
 * If unembed_weights is NULL, initializes deterministic normalized projection weights.
 */
int hssa_set_unembedding_matrix(HssaContext* ctx, const float* unembed_weights, size_t vocab_size);

/*
 * Register a candidate relational axiom (subject, relation, object) associated
 * with a specific candidate token_id for neuro-symbolic contradiction checking.
 */
int hssa_register_candidate_axiom(HssaContext* ctx, int32_t token_id, const char* subj, const char* rel, const char* obj);

/*
 * Execute continuous autoregressive token generation with Attractor Simplex projection:
 * Prefills prompt_tokens across 28-layer HSSA recurrence, then decodes up to max_gen_len
 * tokens using temperature sampling or greedy argmax (temperature <= 1e-4), projecting
 * candidate logits through Attractor Simplex Gate (blocking conflicting tokens).
 *
 * Parameters:
 *   ctx:           Valid HssaContext pointer
 *   prompt_tokens: Array of input prompt token IDs (can be NULL if prompt_len == 0)
 *   prompt_len:    Number of prompt tokens
 *   out_tokens:    Output array preallocated for at least max_gen_len token IDs
 *   max_gen_len:   Maximum number of tokens to generate
 *   temperature:   Sampling temperature (0.0f for greedy argmax)
 *   out_stats:     Optional generation statistics telemetry (can be NULL)
 */
int hssa_generate(
    HssaContext* ctx,
    const int32_t* prompt_tokens,
    size_t prompt_len,
    int32_t* out_tokens,
    size_t max_gen_len,
    float temperature,
    HssaGenStats* out_stats
);

/*
 * =============================================================================
 * Auxiliary Management & Telemetry Helpers
 * =============================================================================
 */

/* Register a token assertion for simplex contradiction checking */
int hssa_register_assertion(HssaContext* ctx, uint32_t token_id, const char* subj, const char* rel, const char* obj);

/* Clear all registered token assertions */
int hssa_clear_assertions(HssaContext* ctx);

/* Attach an external VSA database handle as global or layer-specific knowledge base */
int hssa_set_global_vsa_db(HssaContext* ctx, VsaDbHandle* db);
int hssa_set_layer_vsa_db(HssaContext* ctx, uint32_t layer_idx, VsaDbHandle* db);

/* Online TT16 low-rank continuous core adaptation: delta W = U * V^T */
int hssa_update_tt16(HssaContext* ctx, uint32_t layer_idx, const float* u_delta, const float* v_delta);

/* Introspection helpers */
HssaConfig  hssa_get_config(const HssaContext* ctx);
HssaLayer*  hssa_get_layer(HssaContext* ctx, uint32_t layer_idx);
size_t      hssa_get_state_memory_bytes(const HssaContext* ctx);

/*
 * Dual-rail binarization kernel Psi: R^1024 -> {0,1}^512.
 * Declared here so the engine and its test suites execute the same code.
 * test_hssa_m1.cpp used to carry a private copy of this kernel, which detached
 * its latency and round-trip assertions from the kernel that actually ships.
 */
void dual_rail_binarize(const float* in_vec, uint64_t* out_vsa);

#ifdef __cplusplus
}
#endif

#endif /* HSSA_H */

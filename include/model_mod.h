#ifndef MODEL_MOD_H
#define MODEL_MOD_H

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * =============================================================================
 * Milestone M3: Architecture Experimentation & Model Modification Subsystem
 * Pure Native C-ABI Interface (C++20 / AVX-512)
 * =============================================================================
 */

#ifndef QUANT_PRECISION_DEFINED
#define QUANT_PRECISION_DEFINED
typedef enum QuantPrecision {
    QUANT_2BIT = 2,
    QUANT_3BIT = 3,
    QUANT_4BIT = 4,
    QUANT_DELTA_24 = 24,
    QUANT_DELTA_34 = 34
} QuantPrecision;
#endif

#ifndef LAYER_TYPE_DEFINED
#define LAYER_TYPE_DEFINED
typedef enum LayerType {
    LAYER_GQA_ATTENTION = 0,
    LAYER_GATED_DELTANET = 1,
    LAYER_AVX512_VSA_ASSOCIATIVE = 2
} LayerType;
#endif

#define MODEL_DEFAULT_NUM_LAYERS 28
#define MODEL_DEFAULT_DIM_M      1024
#define MODEL_DEFAULT_DIM_N      1024
#define MODEL_DEFAULT_RANK_R     16
#define MODEL_DELTANET_HEADS     8
#define MODEL_DELTANET_HEAD_DIM  128
#define MODEL_TT16_CORE_SIZE     (MODEL_DEFAULT_RANK_R * MODEL_DEFAULT_DIM_M)

/* Forward declaration for VSA Database Handle */
struct VsaDbHandle;
typedef struct VsaDbHandle VsaDbHandle;

/* Per-layer architectural and quantization descriptor */
typedef struct ModelLayerConfig {
    uint32_t       layer_idx;
    QuantPrecision precision;
    LayerType      layer_type;
    uint32_t       M;              /* Rows (default 1024) */
    uint32_t       N;              /* Cols (default 1024) */
    uint32_t       R;              /* TT Rank (default 16) */
    float*         tt_u;           /* [R * N] 64-byte aligned (16,384 floats = 64 KB) */
    float*         tt_v;           /* [R * M] 64-byte aligned (16,384 floats = 64 KB) */
    float*         deltanet_state; /* [8 * 128 * 128] 64-byte aligned (131,072 floats = 512 KB) */
    VsaDbHandle*   vsa_handle;     /* Layer-specific VsaDbHandle* or NULL */
    void*          gpu_matrix;     /* PtqGpuMatrix* or NULL */
    void*          cpu_matrix;     /* BslTiledMatrix* or NULL */
    void*          reserved[4];
} ModelLayerConfig;

/* Global model experimentation runtime context */
typedef struct ModelContext {
    uint32_t          num_layers;
    ModelLayerConfig* layers;
    VsaDbHandle*      global_vsa_db; /* Global fallback VsaDbHandle* or NULL */
    void*             cuda_drv;      /* CudaDrv* */
    void*             cuda_ctx;      /* CUcontext */
    void*             kernels;       /* PtqKernels* */
    void*             workspace;     /* PtqGpuWorkspace* */
    bool              is_cuda_active;
    uint32_t          hidden_dim;
    uint32_t          intermediate_dim;
} ModelContext;

/*
 * Lifecycle Management
 */
ModelContext* model_create_context(uint32_t num_layers);
void          model_destroy_context(ModelContext* ctx);
void          model_free_context(void* ctx);
uint32_t      model_get_layer_count(const void* ctx);

/*
 * Feature 3.1: Runtime Precision Modulation
 * Switches active quantization bit-planes at runtime with ZERO VRAM reallocation.
 * Returns:
 *    0 on success
 *   -1 if ctx is NULL
 *   -2 if layer_idx is out of bounds (layer_idx >= num_layers)
 *   -3 if prec is not a valid QuantPrecision enum
 */
int            model_set_layer_precision(void* ctx, uint32_t layer_idx, QuantPrecision prec);
QuantPrecision model_get_layer_precision(const void* ctx, uint32_t layer_idx);

/*
 * Feature 3.2: Online Low-Rank Plasticity (TT16 continuous core update in <25 us)
 * Returns:
 *    0 on success
 *   -1 if ctx, u_delta, or v_delta is NULL
 *   -2 if layer_idx is out of bounds
 */
int model_update_tt16_residual(void* ctx, uint32_t layer_idx, const float* u_delta, const float* v_delta);
int model_apply_tt16_residual(const void* ctx, uint32_t layer_idx, const float* in_x, float* inout_y);

/*
 * Feature 3.3: Execution Block Hot-Swapping
 * Returns:
 *    0 on success
 *   -1 if ctx is NULL
 *   -2 if layer_idx is out of bounds
 *   -3 if new_type is not a valid LayerType enum
 */
int       model_swap_layer_block(void* ctx, uint32_t layer_idx, LayerType new_type);
LayerType model_get_layer_type(const void* ctx, uint32_t layer_idx);

/*
 * Layer Forward Dispatch & State Reset
 * Evaluates active layer according to its precision, operator type, and TT16 residual.
 */
int model_forward_layer(void* ctx, uint32_t layer_idx, const float* in_vec, float* out_vec);
int model_reset_layer_state(void* ctx, uint32_t layer_idx);
int model_reset_all_states(void* ctx);

/*
 * VSA Associative Database Binding
 */
int   model_set_layer_vsa_db(void* ctx, uint32_t layer_idx, VsaDbHandle* db);
void* model_get_layer_vsa_db(const void* ctx, uint32_t layer_idx);
int   model_set_global_vsa_db(void* ctx, VsaDbHandle* db);
void* model_get_global_vsa_db(const void* ctx);

/*
 * Feature 3.4: Continuous Representation Geodesic Interpolation & Deep Multi-Layer Interpolation
 */
int model_interpolate_representations(
    const float* rep_a,
    const float* rep_b,
    size_t dim,
    float alpha,
    float* out_rep
);

int model_forward_deep_interpolated(
    void* ctx,
    const float* in_vec,
    float* out_vec
);

/*
 * Feature 3.5: Binary Weight Checkpoint Persistence & Online Layer Training
 */
int model_save_weights(const void* ctx, const char* filepath);
int model_load_weights(void* ctx, const char* filepath);
int model_train_layer_step(
    void* ctx,
    uint32_t layer_idx,
    const float* in_vec,
    const float* target_vec,
    float lr,
    float* out_mse_loss
);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_MOD_H */


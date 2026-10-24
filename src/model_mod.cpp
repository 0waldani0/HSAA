#include "model_mod.h"
#include "vsa_db.h"

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <immintrin.h>

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <atomic>
#include <mutex>
#include <vector>

/*
 * =============================================================================
 * Milestone M3: Architecture Experimentation & Model Modification Subsystem
 * Pure Native C++20 / AVX-512 Implementation
 * =============================================================================
 */

/*
 * Forward block operators
 */

// 1. Grouped-Query Attention (GQA) Block Forward
static int forward_gqa_block(ModelLayerConfig* layer, const float* in_vec, float* out_vec) {
    const size_t H_Q = MODEL_DELTANET_HEADS;      // 8 query heads
    const size_t H_KV = 2;                        // 2 key/value heads
    const size_t G = H_Q / H_KV;                  // Group size = 4
    const size_t D_H = MODEL_DELTANET_HEAD_DIM;   // Head dimension = 128
    const float scale = 0.0883883476f;            // 1.0f / sqrtf(128.0f)

    for (size_t h = 0; h < H_Q; ++h) {
        size_t kv_head = h / G;
        const float* q_ptr = &in_vec[h * D_H];
        const float* k_ptr = &in_vec[kv_head * D_H];
        const float* v_ptr = &in_vec[512 + kv_head * D_H];
        float* out_ptr = &out_vec[h * D_H];

        // Scaled dot product score = <q, k> * scale
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        for (size_t j = 0; j < D_H; j += 32) {
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(q_ptr + j),      _mm512_loadu_ps(k_ptr + j),      acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(q_ptr + j + 16), _mm512_loadu_ps(k_ptr + j + 16), acc1);
        }
        float raw_score = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1)) * scale;

        // Sigmoid gate for single-step self-projection: 1 / (1 + exp(-score))
        float attn_weight = 1.0f / (1.0f + std::exp(-raw_score));
        __m512 v_weight = _mm512_set1_ps(attn_weight);

        // out = in + attn_weight * v
        for (size_t j = 0; j < D_H; j += 16) {
            __m512 in_sub = _mm512_loadu_ps(q_ptr + j);
            __m512 v_sub  = _mm512_loadu_ps(v_ptr + j);
            __m512 res    = _mm512_fmadd_ps(v_sub, v_weight, in_sub);
            _mm512_storeu_ps(out_ptr + j, res);
        }
    }
    return 0;
}

// 2. Gated DeltaNet Block Forward (Recurrent State-Space)
static int forward_deltanet_block(ModelLayerConfig* layer, const float* in_vec, float* out_vec) {
    const size_t H = MODEL_DELTANET_HEADS;        // 8 heads
    const size_t D_H = MODEL_DELTANET_HEAD_DIM;   // 128 dims per head
    const float alpha = 0.95f;                    // State retention decay
    const float beta  = 0.25f;                    // Fast associative learning rate

    __m512 v_alpha = _mm512_set1_ps(alpha);

    for (size_t h = 0; h < H; ++h) {
        const float* x_h = &in_vec[h * D_H];
        float* S_h = &layer->deltanet_state[h * D_H * D_H];
        float* out_h = &out_vec[h * D_H];

        // 1. Key normalization: k_bar = x_h / max(||x_h||_2, 1e-6)
        __m512 sq_acc0 = _mm512_setzero_ps();
        __m512 sq_acc1 = _mm512_setzero_ps();
        for (size_t i = 0; i < D_H; i += 32) {
            __m512 xi0 = _mm512_loadu_ps(x_h + i);
            __m512 xi1 = _mm512_loadu_ps(x_h + i + 16);
            sq_acc0 = _mm512_fmadd_ps(xi0, xi0, sq_acc0);
            sq_acc1 = _mm512_fmadd_ps(xi1, xi1, sq_acc1);
        }
        float sum_sq = _mm512_reduce_add_ps(_mm512_add_ps(sq_acc0, sq_acc1));
        float norm_k = std::sqrt(sum_sq);
        float inv_norm = norm_k > 1e-6f ? (1.0f / norm_k) : 0.0f;

        alignas(64) float k_bar[MODEL_DELTANET_HEAD_DIM];
        __m512 v_inv_norm = _mm512_set1_ps(inv_norm);
        for (size_t i = 0; i < D_H; i += 16) {
            _mm512_storeu_ps(&k_bar[i], _mm512_mul_ps(_mm512_loadu_ps(x_h + i), v_inv_norm));
        }

        // 2. Value prediction: v_hat = S_h * k_bar
        alignas(64) float v_hat[MODEL_DELTANET_HEAD_DIM];
        for (size_t r = 0; r < D_H; ++r) {
            const float* row = &S_h[r * D_H];
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            for (size_t c = 0; c < D_H; c += 32) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(row + c),      _mm512_loadu_ps(&k_bar[c]),      acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(row + c + 16), _mm512_loadu_ps(&k_bar[c + 16]), acc1);
            }
            v_hat[r] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
        }

        // 3. Error: e_h = x_h - v_hat
        alignas(64) float e_h[MODEL_DELTANET_HEAD_DIM];
        for (size_t i = 0; i < D_H; i += 16) {
            __m512 vx = _mm512_loadu_ps(x_h + i);
            __m512 vh = _mm512_loadu_ps(v_hat + i);
            _mm512_storeu_ps(&e_h[i], _mm512_sub_ps(vx, vh));
        }

        // 4. Delta Rule State Update: S_h = alpha * S_h + beta * (e_h (x) k_bar)
        for (size_t r = 0; r < D_H; ++r) {
            float* row = &S_h[r * D_H];
            float scale_e = beta * e_h[r];
            __m512 v_scale_e = _mm512_set1_ps(scale_e);
            for (size_t c = 0; c < D_H; c += 16) {
                __m512 s = _mm512_loadu_ps(row + c);
                __m512 k = _mm512_loadu_ps(&k_bar[c]);
                __m512 updated = _mm512_fmadd_ps(s, v_alpha, _mm512_mul_ps(v_scale_e, k));
                _mm512_storeu_ps(row + c, updated);
            }
        }

        // 5. Readout: u_h = S_h * x_h, Gated by Sigmoid(x_h)
        for (size_t r = 0; r < D_H; ++r) {
            const float* row = &S_h[r * D_H];
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            for (size_t c = 0; c < D_H; c += 32) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(row + c),      _mm512_loadu_ps(x_h + c),      acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(row + c + 16), _mm512_loadu_ps(x_h + c + 16), acc1);
            }
            float dot = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
            float gate = 1.0f / (1.0f + std::exp(-x_h[r]));
            out_h[r] = x_h[r] + dot * gate;
        }
    }
    return 0;
}

// 3. AVX-512 VSA Associative Block Forward
static int forward_vsa_block(ModelLayerConfig* layer, const ModelContext* ctx, const float* in_vec, float* out_vec) {
    // 1. Dual-rail binarization: R^1024 -> {0, 1}^512
    uint64_t query_512[8];
    for (size_t w = 0; w < 8; ++w) {
        uint64_t word = 0;
        for (size_t chunk = 0; chunk < 4; ++chunk) {
            size_t c = w * 4 + chunk;
            __m512 v_lo = _mm512_loadu_ps(&in_vec[c * 16]);
            __m512 v_hi = _mm512_loadu_ps(&in_vec[512 + c * 16]);
            __mmask16 m = _mm512_cmp_ps_mask(v_lo, v_hi, _CMP_GE_OQ);
            word |= ((uint64_t)(uint16_t)m) << (chunk * 16);
        }
        query_512[w] = word;
    }

    // 2. Query associative database
    VsaDbHandle* target_db = layer->vsa_handle ? layer->vsa_handle : (ctx ? ctx->global_vsa_db : nullptr);
    uint64_t best_vec[8];
    // Zero resonance until the scan says otherwise. 1.0f here claimed a perfect
    // match on paths that never queried, and echoing the query back as best_vec
    // injected a projection of the query itself into the residual state.
    float similarity = 0.0f;

    if (target_db && vsa_db_get_vector_count(target_db) > 0) {
        int found = vsa_db_query_vector(target_db, query_512, best_vec, &similarity);
        if (!found) {
            // vsa_db_query_vector leaves *out_sim at 0.0f on a miss; keep it there
            // instead of asserting a match that was never found.
            std::memset(best_vec, 0, sizeof(best_vec));
            similarity = 0.0f;
        }
    } else {
        std::memset(best_vec, 0, sizeof(best_vec));
        similarity = 0.0f;
    }

    // 3. Dual-rail float projection & residual accumulation
    float scale = 0.1f * similarity;
    __m512 v_pos = _mm512_set1_ps(scale);
    __m512 v_neg = _mm512_set1_ps(-scale);

    for (size_t c = 0; c < 32; ++c) {
        size_t word_idx = c / 4;
        size_t shift = (c % 4) * 16;
        uint16_t k = (uint16_t)((best_vec[word_idx] >> shift) & 0xFFFF);

        __m512 in_lo = _mm512_loadu_ps(&in_vec[c * 16]);
        __m512 in_hi = _mm512_loadu_ps(&in_vec[512 + c * 16]);

        __m512 proj_lo = _mm512_mask_blend_ps((__mmask16)k, v_neg, v_pos);
        __m512 proj_hi = _mm512_mask_blend_ps((__mmask16)k, v_pos, v_neg);

        _mm512_storeu_ps(&out_vec[c * 16], _mm512_add_ps(in_lo, proj_lo));
        _mm512_storeu_ps(&out_vec[512 + c * 16], _mm512_add_ps(in_hi, proj_hi));
    }

    return 0;
}

/*
 * =============================================================================
 * Lifecycle Management
 * =============================================================================
 */

ModelContext* model_create_context(uint32_t num_layers) {
    if (num_layers == 0) {
        num_layers = MODEL_DEFAULT_NUM_LAYERS;
    }

    ModelContext* ctx = (ModelContext*)std::calloc(1, sizeof(ModelContext));
    if (!ctx) return nullptr;

    ctx->num_layers = num_layers;
    ctx->hidden_dim = MODEL_DEFAULT_DIM_M;
    ctx->intermediate_dim = MODEL_DEFAULT_DIM_N;
    ctx->global_vsa_db = nullptr;
    ctx->is_cuda_active = false;

    ctx->layers = (ModelLayerConfig*)std::calloc(num_layers, sizeof(ModelLayerConfig));
    if (!ctx->layers) {
        std::free(ctx);
        return nullptr;
    }

    for (uint32_t l = 0; l < num_layers; ++l) {
        ModelLayerConfig* lyr = &ctx->layers[l];
        lyr->layer_idx = l;
        lyr->precision = QUANT_4BIT;
        lyr->layer_type = LAYER_GQA_ATTENTION;
        lyr->M = MODEL_DEFAULT_DIM_M;
        lyr->N = MODEL_DEFAULT_DIM_N;
        lyr->R = MODEL_DEFAULT_RANK_R;

        // Allocate 64-byte aligned buffers
        lyr->tt_u = (float*)_aligned_malloc(MODEL_TT16_CORE_SIZE * sizeof(float), 64);
        lyr->tt_v = (float*)_aligned_malloc(MODEL_TT16_CORE_SIZE * sizeof(float), 64);
        lyr->deltanet_state = (float*)_aligned_malloc(
            MODEL_DELTANET_HEADS * MODEL_DELTANET_HEAD_DIM * MODEL_DELTANET_HEAD_DIM * sizeof(float), 64);

        if (lyr->tt_u) std::memset(lyr->tt_u, 0, MODEL_TT16_CORE_SIZE * sizeof(float));
        if (lyr->tt_v) std::memset(lyr->tt_v, 0, MODEL_TT16_CORE_SIZE * sizeof(float));
        if (lyr->deltanet_state) {
            std::memset(lyr->deltanet_state, 0,
                MODEL_DELTANET_HEADS * MODEL_DELTANET_HEAD_DIM * MODEL_DELTANET_HEAD_DIM * sizeof(float));
        }

        lyr->vsa_handle = nullptr;
        lyr->gpu_matrix = nullptr;
        lyr->cpu_matrix = nullptr;
    }

    return ctx;
}

void model_destroy_context(ModelContext* ctx) {
    if (!ctx) return;
    if (ctx->layers) {
        for (uint32_t l = 0; l < ctx->num_layers; ++l) {
            ModelLayerConfig* lyr = &ctx->layers[l];
            if (lyr->tt_u) _aligned_free(lyr->tt_u);
            if (lyr->tt_v) _aligned_free(lyr->tt_v);
            if (lyr->deltanet_state) _aligned_free(lyr->deltanet_state);
        }
        std::free(ctx->layers);
    }
    std::free(ctx);
}

void model_free_context(void* ctx) {
    model_destroy_context((ModelContext*)ctx);
}

uint32_t model_get_layer_count(const void* ctx) {
    if (!ctx) return 0;
    return ((const ModelContext*)ctx)->num_layers;
}

/*
 * =============================================================================
 * Feature 3.1: Runtime Precision Modulation
 * =============================================================================
 */

int model_set_layer_precision(void* ctx, uint32_t layer_idx, QuantPrecision prec) {
    if (!ctx) return -1;
    ModelContext* m = (ModelContext*)ctx;
    if (layer_idx >= m->num_layers) return -2;

    if (prec != QUANT_2BIT && prec != QUANT_3BIT && prec != QUANT_4BIT &&
        prec != QUANT_DELTA_24 && prec != QUANT_DELTA_34) {
        return -3;
    }

    // $O(1)$ runtime precision toggle with zero VRAM reallocation.
    // atomic_ref is the C++20 tool for atomically accessing a plain object:
    // reinterpret_cast<std::atomic<T>*> over a non-atomic member is UB.
    std::atomic_ref<QuantPrecision>(m->layers[layer_idx].precision)
        .store(prec, std::memory_order_release);
    return 0;
}

QuantPrecision model_get_layer_precision(const void* ctx, uint32_t layer_idx) {
    if (!ctx) return QUANT_4BIT;
    const ModelContext* m = (const ModelContext*)ctx;
    if (layer_idx >= m->num_layers) return QUANT_4BIT;
    return std::atomic_ref<QuantPrecision>(
               const_cast<QuantPrecision&>(m->layers[layer_idx].precision))
        .load(std::memory_order_acquire);
}

/*
 * =============================================================================
 * Feature 3.2: Online Continuous Low-Rank Plasticity Engine (TT16)
 * =============================================================================
 */

int model_update_tt16_residual(void* ctx, uint32_t layer_idx, const float* u_delta, const float* v_delta) {
    if (!ctx || !u_delta || !v_delta) return -1;
    ModelContext* m = (ModelContext*)ctx;
    if (layer_idx >= m->num_layers) return -2;

    ModelLayerConfig* lyr = &m->layers[layer_idx];
    size_t sz_u = (size_t)lyr->R * lyr->N; // 16 * 1024 = 16,384 floats
    size_t sz_v = (size_t)lyr->R * lyr->M; // 16 * 1024 = 16,384 floats
    float* dst_u = lyr->tt_u;
    float* dst_v = lyr->tt_v;

    // 4-Way AVX-512 unrolled vector addition for U (64 floats = 256 bytes per iter)
    size_t i = 0;
    for (; i + 64 <= sz_u; i += 64) {
        __m512 u0 = _mm512_loadu_ps(dst_u + i);
        __m512 u1 = _mm512_loadu_ps(dst_u + i + 16);
        __m512 u2 = _mm512_loadu_ps(dst_u + i + 32);
        __m512 u3 = _mm512_loadu_ps(dst_u + i + 48);

        __m512 d0 = _mm512_loadu_ps(u_delta + i);
        __m512 d1 = _mm512_loadu_ps(u_delta + i + 16);
        __m512 d2 = _mm512_loadu_ps(u_delta + i + 32);
        __m512 d3 = _mm512_loadu_ps(u_delta + i + 48);

        _mm512_storeu_ps(dst_u + i,      _mm512_add_ps(u0, d0));
        _mm512_storeu_ps(dst_u + i + 16, _mm512_add_ps(u1, d1));
        _mm512_storeu_ps(dst_u + i + 32, _mm512_add_ps(u2, d2));
        _mm512_storeu_ps(dst_u + i + 48, _mm512_add_ps(u3, d3));
    }
    for (; i + 16 <= sz_u; i += 16) {
        __m512 u = _mm512_loadu_ps(dst_u + i);
        __m512 d = _mm512_loadu_ps(u_delta + i);
        _mm512_storeu_ps(dst_u + i, _mm512_add_ps(u, d));
    }
    for (; i < sz_u; ++i) dst_u[i] += u_delta[i];

    // 4-Way AVX-512 unrolled vector addition for V (64 floats = 256 bytes per iter)
    size_t j = 0;
    for (; j + 64 <= sz_v; j += 64) {
        __m512 v0 = _mm512_loadu_ps(dst_v + j);
        __m512 v1 = _mm512_loadu_ps(dst_v + j + 16);
        __m512 v2 = _mm512_loadu_ps(dst_v + j + 32);
        __m512 v3 = _mm512_loadu_ps(dst_v + j + 48);

        __m512 d0 = _mm512_loadu_ps(v_delta + j);
        __m512 d1 = _mm512_loadu_ps(v_delta + j + 16);
        __m512 d2 = _mm512_loadu_ps(v_delta + j + 32);
        __m512 d3 = _mm512_loadu_ps(v_delta + j + 48);

        _mm512_storeu_ps(dst_v + j,      _mm512_add_ps(v0, d0));
        _mm512_storeu_ps(dst_v + j + 16, _mm512_add_ps(v1, d1));
        _mm512_storeu_ps(dst_v + j + 32, _mm512_add_ps(v2, d2));
        _mm512_storeu_ps(dst_v + j + 48, _mm512_add_ps(v3, d3));
    }
    for (; j + 16 <= sz_v; j += 16) {
        __m512 v = _mm512_loadu_ps(dst_v + j);
        __m512 d = _mm512_loadu_ps(v_delta + j);
        _mm512_storeu_ps(dst_v + j, _mm512_add_ps(v, d));
    }
    for (; j < sz_v; ++j) dst_v[j] += v_delta[j];

    return 0;
}

int model_apply_tt16_residual(const void* ctx, uint32_t layer_idx, const float* in_x, float* inout_y) {
    if (!ctx || !in_x || !inout_y) return -1;
    const ModelContext* m = (const ModelContext*)ctx;
    if (layer_idx >= m->num_layers) return -2;

    const ModelLayerConfig* lyr = &m->layers[layer_idx];
    const float* v_ptr = lyr->tt_v;
    const float* u_ptr = lyr->tt_u;
    const uint32_t R = lyr->R; // 16
    const uint32_t D = lyr->N; // 1024

    // Stage 1: Bottleneck projection z[r] = <v_r, in_x> for r in [0, 15]
    alignas(64) float z[16];
    for (uint32_t r = 0; r < R; ++r) {
        const float* vr = v_ptr + (size_t)r * D;
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        for (uint32_t j = 0; j < D; j += 64) {
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(vr + j),      _mm512_loadu_ps(in_x + j),      acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(vr + j + 16), _mm512_loadu_ps(in_x + j + 16), acc1);
            acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(vr + j + 32), _mm512_loadu_ps(in_x + j + 32), acc2);
            acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(vr + j + 48), _mm512_loadu_ps(in_x + j + 48), acc3);
        }
        __m512 sum = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        z[r] = _mm512_reduce_add_ps(sum);
    }

    // Pre-broadcast all 16 scalars z[r] into vector registers
    __m512 vz[16];
    for (uint32_t r = 0; r < 16; ++r) {
        vz[r] = _mm512_set1_ps(z[r]);
    }

    // Stage 2: Dyadic accumulation inout_y[i] += sum_{r=0..15}(z[r] * u_r[i])
    for (uint32_t i = 0; i < D; i += 16) {
        __m512 y_acc = _mm512_loadu_ps(inout_y + i);
        for (uint32_t r = 0; r < 16; ++r) {
            __m512 u_vec = _mm512_loadu_ps(u_ptr + (size_t)r * D + i);
            y_acc = _mm512_fmadd_ps(u_vec, vz[r], y_acc);
        }
        _mm512_storeu_ps(inout_y + i, y_acc);
    }

    return 0;
}

/*
 * =============================================================================
 * Feature 3.3: Execution Block Hot-Swapping
 * =============================================================================
 */

int model_swap_layer_block(void* ctx, uint32_t layer_idx, LayerType new_type) {
    if (!ctx) return -1;
    ModelContext* m = (ModelContext*)ctx;
    if (layer_idx >= m->num_layers) return -2;

    if (new_type != LAYER_GQA_ATTENTION &&
        new_type != LAYER_GATED_DELTANET &&
        new_type != LAYER_AVX512_VSA_ASSOCIATIVE) {
        return -3;
    }

    // Zero-allocation, atomic release store (<50 ns)
    std::atomic_ref<LayerType>(m->layers[layer_idx].layer_type)
        .store(new_type, std::memory_order_release);
    return 0;
}

LayerType model_get_layer_type(const void* ctx, uint32_t layer_idx) {
    if (!ctx) return LAYER_GQA_ATTENTION;
    const ModelContext* m = (const ModelContext*)ctx;
    if (layer_idx >= m->num_layers) return LAYER_GQA_ATTENTION;
    return std::atomic_ref<LayerType>(
               const_cast<LayerType&>(m->layers[layer_idx].layer_type))
        .load(std::memory_order_acquire);
}

/*
 * =============================================================================
 * Layer Forward Dispatch & State Management
 * =============================================================================
 */

int model_forward_layer(void* ctx, uint32_t layer_idx, const float* in_vec, float* out_vec) {
    if (!ctx) return -1;
    ModelContext* m = (ModelContext*)ctx;
    if (layer_idx >= m->num_layers) return -2;
    if (!in_vec || !out_vec) return -1;

    ModelLayerConfig* lyr = &m->layers[layer_idx];
    LayerType current_type =
        std::atomic_ref<LayerType>(lyr->layer_type).load(std::memory_order_acquire);

    // 1. Dispatch base layer operator
    switch (current_type) {
        case LAYER_GQA_ATTENTION:
            forward_gqa_block(lyr, in_vec, out_vec);
            break;
        case LAYER_GATED_DELTANET:
            forward_deltanet_block(lyr, in_vec, out_vec);
            break;
        case LAYER_AVX512_VSA_ASSOCIATIVE:
            forward_vsa_block(lyr, m, in_vec, out_vec);
            break;
        default:
            return -3;
    }

    // 2. Continuous Online Low-Rank Plasticity (TT16 Residual)
    model_apply_tt16_residual(ctx, layer_idx, in_vec, out_vec);

    return 0;
}

int model_reset_layer_state(void* ctx, uint32_t layer_idx) {
    if (!ctx) return -1;
    ModelContext* m = (ModelContext*)ctx;
    if (layer_idx >= m->num_layers) return -2;

    ModelLayerConfig* lyr = &m->layers[layer_idx];
    if (lyr->deltanet_state) {
        std::memset(lyr->deltanet_state, 0,
            MODEL_DELTANET_HEADS * MODEL_DELTANET_HEAD_DIM * MODEL_DELTANET_HEAD_DIM * sizeof(float));
    }
    return 0;
}

int model_reset_all_states(void* ctx) {
    if (!ctx) return -1;
    ModelContext* m = (ModelContext*)ctx;
    for (uint32_t l = 0; l < m->num_layers; ++l) {
        ModelLayerConfig* lyr = &m->layers[l];
        if (lyr->deltanet_state) {
            std::memset(lyr->deltanet_state, 0,
                MODEL_DELTANET_HEADS * MODEL_DELTANET_HEAD_DIM * MODEL_DELTANET_HEAD_DIM * sizeof(float));
        }
    }
    return 0;
}

/*
 * =============================================================================
 * VSA Associative Database Binding
 * =============================================================================
 */

int model_set_layer_vsa_db(void* ctx, uint32_t layer_idx, VsaDbHandle* db) {
    if (!ctx) return -1;
    ModelContext* m = (ModelContext*)ctx;
    if (layer_idx >= m->num_layers) return -2;
    m->layers[layer_idx].vsa_handle = db;
    return 0;
}

void* model_get_layer_vsa_db(const void* ctx, uint32_t layer_idx) {
    if (!ctx) return nullptr;
    const ModelContext* m = (const ModelContext*)ctx;
    if (layer_idx >= m->num_layers) return nullptr;
    return m->layers[layer_idx].vsa_handle;
}

int model_set_global_vsa_db(void* ctx, VsaDbHandle* db) {
    if (!ctx) return -1;
    ((ModelContext*)ctx)->global_vsa_db = db;
    return 0;
}

void* model_get_global_vsa_db(const void* ctx) {
    if (!ctx) return nullptr;
    return ((const ModelContext*)ctx)->global_vsa_db;
}

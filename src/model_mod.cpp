#include "model_mod.h"
#include "vsa_db.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <stdlib.h>
static inline void* _aligned_malloc(size_t size, size_t alignment) {
    void* ptr = nullptr;
    size_t aligned_sz = (size + alignment - 1) & ~(alignment - 1);
    if (posix_memalign(&ptr, alignment, aligned_sz) != 0) return nullptr;
    return ptr;
}
static inline void _aligned_free(void* ptr) {
    free(ptr);
}
#endif
#include <immintrin.h>

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
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
 * Base layer trainable projection weights (1.58-bit ternary bitplanes + continuous STE)
 */
struct alignas(64) ModelBaseLayerWeights {
    alignas(64) float w_q_diag[MODEL_DEFAULT_DIM_M];
    alignas(64) float w_q_rot[MODEL_DEFAULT_DIM_M];
    alignas(64) float w_k_diag[MODEL_DEFAULT_DIM_M];
    alignas(64) float w_k_rot[MODEL_DEFAULT_DIM_M];
    alignas(64) float w_v_diag[MODEL_DEFAULT_DIM_M];
    alignas(64) float w_v_rot[MODEL_DEFAULT_DIM_M];
    alignas(64) float w_dn_in[MODEL_DEFAULT_DIM_M];
    alignas(64) float w_dn_gate[MODEL_DEFAULT_DIM_M];
    uint64_t q_pos[MODEL_DEFAULT_DIM_M / 64];
    uint64_t q_neg[MODEL_DEFAULT_DIM_M / 64];
    uint64_t k_pos[MODEL_DEFAULT_DIM_M / 64];
    uint64_t k_neg[MODEL_DEFAULT_DIM_M / 64];
    uint64_t v_pos[MODEL_DEFAULT_DIM_M / 64];
    uint64_t v_neg[MODEL_DEFAULT_DIM_M / 64];
    float scale_q_rot;
    float scale_k_rot;
    float scale_v_rot;
};

static inline void mmod_quantize_ternary_1024(
    const float* src,
    uint64_t* out_pos,
    uint64_t* out_neg,
    float* out_scale
) noexcept {
    float sum_abs = 0.0f;
    for (size_t i = 0; i < MODEL_DEFAULT_DIM_M; ++i) {
        sum_abs += std::fabs(src[i]);
    }
    const float mean_abs = sum_abs / static_cast<float>(MODEL_DEFAULT_DIM_M);
    const float thresh = 0.35f * mean_abs;
    std::memset(out_pos, 0, (MODEL_DEFAULT_DIM_M / 64) * sizeof(uint64_t));
    std::memset(out_neg, 0, (MODEL_DEFAULT_DIM_M / 64) * sizeof(uint64_t));
    float active_sum = 0.0f;
    uint32_t active_cnt = 0;
    for (size_t i = 0; i < MODEL_DEFAULT_DIM_M; ++i) {
        const float v = src[i];
        const size_t w = i >> 6;
        const uint64_t bit = 1ULL << (i & 63);
        if (v > thresh) {
            out_pos[w] |= bit;
            active_sum += v;
            ++active_cnt;
        } else if (v < -thresh) {
            out_neg[w] |= bit;
            active_sum -= v;
            ++active_cnt;
        }
    }
    *out_scale = (active_cnt > 0) ? (active_sum / static_cast<float>(active_cnt)) : mean_abs;
}

static inline void mmod_quantize_base_ternary(ModelBaseLayerWeights* bw) noexcept {
    if (!bw) return;
    mmod_quantize_ternary_1024(bw->w_q_rot, bw->q_pos, bw->q_neg, &bw->scale_q_rot);
    mmod_quantize_ternary_1024(bw->w_k_rot, bw->k_pos, bw->k_neg, &bw->scale_k_rot);
    mmod_quantize_ternary_1024(bw->w_v_rot, bw->v_pos, bw->v_neg, &bw->scale_v_rot);
}

static inline void mmod_init_base_layer_weights(ModelBaseLayerWeights* bw) noexcept {
    if (!bw) return;
    for (size_t i = 0; i < MODEL_DEFAULT_DIM_M; ++i) {
        bw->w_q_diag[i]  = 0.88f;
        bw->w_q_rot[i]   = 0.12f;
        bw->w_k_diag[i]  = 0.88f;
        bw->w_k_rot[i]   = -0.12f;
        bw->w_v_diag[i]  = 0.90f;
        bw->w_v_rot[i]   = 0.10f;
        bw->w_dn_in[i]   = 1.0f;
        bw->w_dn_gate[i] = 1.0f;
    }
    mmod_quantize_base_ternary(bw);
}

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
    const uint32_t l_idx = layer ? layer->layer_idx : 0;
    const ModelBaseLayerWeights* bw = layer ? static_cast<const ModelBaseLayerWeights*>(layer->cpu_matrix) : nullptr;

    for (size_t h = 0; h < H_Q; ++h) {
        size_t kv_head = h / G;
        const float* q_src = &in_vec[h * D_H];
        const float* k_src = &in_vec[kv_head * D_H];
        const float* v_src = &in_vec[512 + kv_head * D_H];
        float* out_ptr = &out_vec[h * D_H];

        // Trainable 1.58-bit ternary + continuous STE per-head Q, K, V projections
        const size_t rot_q = ((size_t)(l_idx * 5 + h * 3 + 1)) & (D_H - 1);
        const size_t rot_k = ((size_t)(l_idx * 7 + kv_head * 5 + 3)) & (D_H - 1);
        const size_t rot_v = ((size_t)(l_idx * 11 + kv_head * 7 + 5)) & (D_H - 1);
        alignas(64) float q_proj[MODEL_DELTANET_HEAD_DIM];
        alignas(64) float k_proj[MODEL_DELTANET_HEAD_DIM];
        alignas(64) float v_proj[MODEL_DELTANET_HEAD_DIM];
        if (bw) {
            const size_t base_off = h * D_H;
            for (size_t j = 0; j < D_H; ++j) {
                const size_t d_idx = base_off + j;
                const size_t w_idx = d_idx >> 6;
                const uint64_t bit = 1ULL << (d_idx & 63);
                const float tq = ((bw->q_pos[w_idx] & bit) ? 1.0f : 0.0f) - ((bw->q_neg[w_idx] & bit) ? 1.0f : 0.0f);
                const float tk = ((bw->k_pos[w_idx] & bit) ? 1.0f : 0.0f) - ((bw->k_neg[w_idx] & bit) ? 1.0f : 0.0f);
                const float tv = ((bw->v_pos[w_idx] & bit) ? 1.0f : 0.0f) - ((bw->v_neg[w_idx] & bit) ? 1.0f : 0.0f);

                const float wq_rot = 0.5f * bw->w_q_rot[d_idx] + 0.5f * bw->scale_q_rot * tq;
                const float wk_rot = 0.5f * bw->w_k_rot[d_idx] + 0.5f * bw->scale_k_rot * tk;
                const float wv_rot = 0.5f * bw->w_v_rot[d_idx] + 0.5f * bw->scale_v_rot * tv;

                q_proj[j] = bw->w_q_diag[d_idx] * q_src[j] + wq_rot * q_src[(j + rot_q) & (D_H - 1)];
                k_proj[j] = bw->w_k_diag[d_idx] * k_src[j] + wk_rot * k_src[(j + rot_k) & (D_H - 1)];
                v_proj[j] = bw->w_v_diag[d_idx] * v_src[j] + wv_rot * v_src[(j + rot_v) & (D_H - 1)];
            }
        } else {
            for (size_t j = 0; j < D_H; ++j) {
                q_proj[j] = 0.88f * q_src[j] + 0.12f * q_src[(j + rot_q) & (D_H - 1)];
                k_proj[j] = 0.88f * k_src[j] - 0.12f * k_src[(j + rot_k) & (D_H - 1)];
                v_proj[j] = 0.90f * v_src[j] + 0.10f * v_src[(j + rot_v) & (D_H - 1)];
            }
        }

        // Scaled dot product score = <q_proj, k_proj> * scale
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        for (size_t j = 0; j < D_H; j += 32) {
            acc0 = _mm512_fmadd_ps(_mm512_load_ps(q_proj + j),      _mm512_load_ps(k_proj + j),      acc0);
            acc1 = _mm512_fmadd_ps(_mm512_load_ps(q_proj + j + 16), _mm512_load_ps(k_proj + j + 16), acc1);
        }
        float raw_score = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1)) * scale;

        // Sigmoid gate for single-step self-projection: 1 / (1 + exp(-score))
        float attn_weight = 1.0f / (1.0f + std::exp(-raw_score));
        __m512 v_weight = _mm512_set1_ps(attn_weight);

        // out = in + attn_weight * v_proj
        for (size_t j = 0; j < D_H; j += 16) {
            __m512 in_sub = _mm512_loadu_ps(q_src + j);
            __m512 v_sub  = _mm512_load_ps(v_proj + j);
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
    const ModelBaseLayerWeights* bw = layer ? static_cast<const ModelBaseLayerWeights*>(layer->cpu_matrix) : nullptr;

    __m512 v_alpha = _mm512_set1_ps(alpha);

    for (size_t h = 0; h < H; ++h) {
        const float* x_raw = &in_vec[h * D_H];
        float* S_h = &layer->deltanet_state[h * D_H * D_H];
        float* out_h = &out_vec[h * D_H];

        alignas(64) float x_h[MODEL_DELTANET_HEAD_DIM];
        if (bw) {
            const size_t base_off = h * D_H;
            for (size_t i = 0; i < D_H; i += 16) {
                __m512 vx = _mm512_loadu_ps(x_raw + i);
                __m512 vw = _mm512_load_ps(&bw->w_dn_in[base_off + i]);
                _mm512_store_ps(&x_h[i], _mm512_mul_ps(vx, vw));
            }
        } else {
            std::memcpy(x_h, x_raw, D_H * sizeof(float));
        }

        // 1. Key normalization: k_bar = x_h / max(||x_h||_2, 1e-6)
        __m512 sq_acc0 = _mm512_setzero_ps();
        __m512 sq_acc1 = _mm512_setzero_ps();
        for (size_t i = 0; i < D_H; i += 32) {
            __m512 xi0 = _mm512_load_ps(x_h + i);
            __m512 xi1 = _mm512_load_ps(x_h + i + 16);
            sq_acc0 = _mm512_fmadd_ps(xi0, xi0, sq_acc0);
            sq_acc1 = _mm512_fmadd_ps(xi1, xi1, sq_acc1);
        }
        float sum_sq = _mm512_reduce_add_ps(_mm512_add_ps(sq_acc0, sq_acc1));
        float norm_k = std::sqrt(sum_sq);
        float inv_norm = norm_k > 1e-6f ? (1.0f / norm_k) : 0.0f;

        alignas(64) float k_bar[MODEL_DELTANET_HEAD_DIM];
        __m512 v_inv_norm = _mm512_set1_ps(inv_norm);
        for (size_t i = 0; i < D_H; i += 16) {
            _mm512_store_ps(&k_bar[i], _mm512_mul_ps(_mm512_load_ps(x_h + i), v_inv_norm));
        }

        // 2. Value prediction: v_hat = S_h * k_bar
        alignas(64) float v_hat[MODEL_DELTANET_HEAD_DIM];
        for (size_t r = 0; r < D_H; ++r) {
            const float* row = &S_h[r * D_H];
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            for (size_t c = 0; c < D_H; c += 32) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(row + c),      _mm512_load_ps(&k_bar[c]),      acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(row + c + 16), _mm512_load_ps(&k_bar[c + 16]), acc1);
            }
            v_hat[r] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
        }

        // 3. Error: e_h = x_h - v_hat
        alignas(64) float e_h[MODEL_DELTANET_HEAD_DIM];
        for (size_t i = 0; i < D_H; i += 16) {
            __m512 vx = _mm512_load_ps(x_h + i);
            __m512 vh = _mm512_load_ps(v_hat + i);
            _mm512_store_ps(&e_h[i], _mm512_sub_ps(vx, vh));
        }

        // 4. Delta Rule State Update: S_h = alpha * S_h + beta * (e_h (x) k_bar)
        for (size_t r = 0; r < D_H; ++r) {
            float* row = &S_h[r * D_H];
            float scale_e = beta * e_h[r];
            __m512 v_scale_e = _mm512_set1_ps(scale_e);
            for (size_t c = 0; c < D_H; c += 16) {
                __m512 s = _mm512_loadu_ps(row + c);
                __m512 k = _mm512_load_ps(&k_bar[c]);
                __m512 updated = _mm512_fmadd_ps(s, v_alpha, _mm512_mul_ps(v_scale_e, k));
                _mm512_storeu_ps(row + c, updated);
            }
        }

        // 5. Readout: u_h = S_h * x_h, Gated by Sigmoid(w_dn_gate * x_raw)
        const size_t base_off = h * D_H;
        for (size_t r = 0; r < D_H; ++r) {
            const float* row = &S_h[r * D_H];
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            for (size_t c = 0; c < D_H; c += 32) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(row + c),      _mm512_load_ps(x_h + c),      acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(row + c + 16), _mm512_load_ps(x_h + c + 16), acc1);
            }
            float dot = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
            const float g_in = bw ? (bw->w_dn_gate[base_off + r] * x_raw[r]) : x_raw[r];
            float gate = 1.0f / (1.0f + std::exp(-g_in));
            out_h[r] = x_raw[r] + dot * gate;
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
    VsaDbHandle* layer_db = std::atomic_ref<VsaDbHandle*>(layer->vsa_handle).load(std::memory_order_acquire);
    VsaDbHandle* global_db = ctx ? std::atomic_ref<VsaDbHandle*>(const_cast<VsaDbHandle*&>(ctx->global_vsa_db)).load(std::memory_order_acquire) : nullptr;
    VsaDbHandle* target_db = (layer_db && vsa_db_get_vector_count(layer_db) > 0)
        ? layer_db
        : (global_db ? global_db : layer_db);
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

        // Allocate 64-byte aligned buffers and abort cleanly if any allocation fails
        lyr->tt_u = (float*)_aligned_malloc(MODEL_TT16_CORE_SIZE * sizeof(float), 64);
        lyr->tt_v = (float*)_aligned_malloc(MODEL_TT16_CORE_SIZE * sizeof(float), 64);
        lyr->deltanet_state = (float*)_aligned_malloc(
            MODEL_DELTANET_HEADS * MODEL_DELTANET_HEAD_DIM * MODEL_DELTANET_HEAD_DIM * sizeof(float), 64);
        ModelBaseLayerWeights* bw = (ModelBaseLayerWeights*)_aligned_malloc(sizeof(ModelBaseLayerWeights), 64);

        if (!lyr->tt_u || !lyr->tt_v || !lyr->deltanet_state || !bw) {
            if (bw) _aligned_free(bw);
            model_destroy_context(ctx);
            return nullptr;
        }

        std::memset(lyr->tt_u, 0, MODEL_TT16_CORE_SIZE * sizeof(float));
        std::memset(lyr->tt_v, 0, MODEL_TT16_CORE_SIZE * sizeof(float));
        std::memset(lyr->deltanet_state, 0,
            MODEL_DELTANET_HEADS * MODEL_DELTANET_HEAD_DIM * MODEL_DELTANET_HEAD_DIM * sizeof(float));
        mmod_init_base_layer_weights(bw);

        lyr->vsa_handle = nullptr;
        lyr->gpu_matrix = nullptr;
        lyr->cpu_matrix = bw;
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
            if (lyr->cpu_matrix) {
                _aligned_free(lyr->cpu_matrix);
                lyr->cpu_matrix = nullptr;
            }
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

    // Stage input into a 64B-aligned local buffer only when in_vec == out_vec (in-place execution)
    // so GQA keys/values and TT16 residual input are never clobbered mid-layer, while keeping
    // zero copy overhead on the standard non-aliased path.
    alignas(64) float staged_in[MODEL_DEFAULT_DIM_N];
    const float* src_vec = in_vec;
    if (in_vec == out_vec) {
        std::memcpy(staged_in, in_vec, MODEL_DEFAULT_DIM_N * sizeof(float));
        src_vec = staged_in;
    }

    // Note: CPU layer operators execute FP32 reference pathways; lyr->precision
    // is preserved atomically across runtime modulation for hardware GPU dispatch.
    // 1. Dispatch base layer operator
    switch (current_type) {
        case LAYER_GQA_ATTENTION:
            forward_gqa_block(lyr, src_vec, out_vec);
            break;
        case LAYER_GATED_DELTANET:
            forward_deltanet_block(lyr, src_vec, out_vec);
            break;
        case LAYER_AVX512_VSA_ASSOCIATIVE:
            forward_vsa_block(lyr, m, src_vec, out_vec);
            break;
        default:
            return -3;
    }

    // 1b. Apply runtime precision quantization grid to the residual delta (out_vec - src_vec)
    QuantPrecision current_prec =
        std::atomic_ref<QuantPrecision>(lyr->precision).load(std::memory_order_acquire);
    if (current_prec != QUANT_4BIT) {
        const float q_levels = (current_prec == QUANT_2BIT) ? 3.0f
                             : (current_prec == QUANT_3BIT) ? 7.0f
                             : (current_prec == QUANT_DELTA_24) ? 11.0f : 13.0f;
        const float inv_q = 1.0f / q_levels;
        const __m512 v_ql = _mm512_set1_ps(q_levels);
        const __m512 v_iq = _mm512_set1_ps(0.15f * inv_q);
        const __m512 v_85 = _mm512_set1_ps(0.85f);
        for (size_t j = 0; j < MODEL_DEFAULT_DIM_N; j += 16) {
            const __m512 v_src = _mm512_loadu_ps(src_vec + j);
            const __m512 v_out = _mm512_loadu_ps(out_vec + j);
            const __m512 v_delta = _mm512_sub_ps(v_out, v_src);
            const __m512 v_q = _mm512_roundscale_ps(_mm512_mul_ps(v_delta, v_ql), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            const __m512 v_res = _mm512_fmadd_ps(v_q, v_iq, _mm512_fmadd_ps(v_85, v_delta, v_src));
            _mm512_storeu_ps(out_vec + j, v_res);
        }
    }

    // 2. Continuous Online Low-Rank Plasticity (TT16 Residual)
    model_apply_tt16_residual(ctx, layer_idx, src_vec, out_vec);

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
    std::atomic_ref<VsaDbHandle*>(m->layers[layer_idx].vsa_handle)
        .store(db, std::memory_order_release);
    return 0;
}

void* model_get_layer_vsa_db(const void* ctx, uint32_t layer_idx) {
    if (!ctx) return nullptr;
    const ModelContext* m = (const ModelContext*)ctx;
    if (layer_idx >= m->num_layers) return nullptr;
    return std::atomic_ref<VsaDbHandle*>(
               const_cast<VsaDbHandle*&>(m->layers[layer_idx].vsa_handle))
        .load(std::memory_order_acquire);
}

int model_set_global_vsa_db(void* ctx, VsaDbHandle* db) {
    if (!ctx) return -1;
    std::atomic_ref<VsaDbHandle*>(((ModelContext*)ctx)->global_vsa_db)
        .store(db, std::memory_order_release);
    return 0;
}

void* model_get_global_vsa_db(const void* ctx) {
    if (!ctx) return nullptr;
    return std::atomic_ref<VsaDbHandle*>(
               const_cast<VsaDbHandle*&>(((const ModelContext*)ctx)->global_vsa_db))
        .load(std::memory_order_acquire);
}

/*
 * =============================================================================
 * Feature 3.4: Continuous Representation Geodesic Interpolation & Deep Stack Forward
 * =============================================================================
 */

int model_interpolate_representations(
    const float* rep_a,
    const float* rep_b,
    size_t dim,
    float alpha,
    float* out_rep
) {
    if (!rep_a || !rep_b || !out_rep || dim == 0) return -1;
    const float t = (alpha < 0.0f) ? 0.0f : ((alpha > 1.0f) ? 1.0f : alpha);

    __m512 acc_aa = _mm512_setzero_ps();
    __m512 acc_bb = _mm512_setzero_ps();
    __m512 acc_ab = _mm512_setzero_ps();

    const size_t vec_end = dim & ~static_cast<size_t>(15);
    size_t i = 0;
    for (; i < vec_end; i += 16) {
        __m512 va = _mm512_loadu_ps(rep_a + i);
        __m512 vb = _mm512_loadu_ps(rep_b + i);
        acc_aa = _mm512_fmadd_ps(va, va, acc_aa);
        acc_bb = _mm512_fmadd_ps(vb, vb, acc_bb);
        acc_ab = _mm512_fmadd_ps(va, vb, acc_ab);
    }
    float sum_aa = _mm512_reduce_add_ps(acc_aa);
    float sum_bb = _mm512_reduce_add_ps(acc_bb);
    float sum_ab = _mm512_reduce_add_ps(acc_ab);
    for (size_t rem = vec_end; rem < dim; ++rem) {
        sum_aa += rep_a[rem] * rep_a[rem];
        sum_bb += rep_b[rem] * rep_b[rem];
        sum_ab += rep_a[rem] * rep_b[rem];
    }

    const float norm_a = std::sqrt(sum_aa);
    const float norm_b = std::sqrt(sum_bb);
    const float target_norm = (1.0f - t) * norm_a + t * norm_b;
    float w_a = 1.0f - t;
    float w_b = t;
    bool use_antipodal = false;
    bool rescale_to_target = false;

    if (norm_a > 1e-6f && norm_b > 1e-6f) {
        float cos_theta = sum_ab / (norm_a * norm_b);
        if (cos_theta > 1.0f) cos_theta = 1.0f;
        if (cos_theta < -1.0f) cos_theta = -1.0f;
        if (cos_theta < 0.9995f && cos_theta > -0.9995f) {
            float omega = std::acos(cos_theta);
            float sin_omega = std::sin(omega);
            if (sin_omega > 1e-6f) {
                w_a = (std::sin((1.0f - t) * omega) / sin_omega) * (target_norm / norm_a);
                w_b = (std::sin(t * omega) / sin_omega) * (target_norm / norm_b);
            }
        } else if (cos_theta <= -0.9995f && dim >= 2) {
            use_antipodal = true;
        } else {
            rescale_to_target = true;
        }
    }

    if (use_antipodal) {
        size_t min_idx = 0;
        float min_abs = std::fabs(rep_a[0]);
        for (size_t k = 1; k < dim; ++k) {
            float a_abs = std::fabs(rep_a[k]);
            if (a_abs < min_abs) {
                min_abs = a_abs;
                min_idx = k;
            }
        }
        const float inv_na = 1.0f / norm_a;
        const float neg_proj_inv_na = -(rep_a[min_idx] * inv_na) * inv_na;
        __m512 v_scale_proj = _mm512_set1_ps(neg_proj_inv_na);
        for (size_t k = 0; k < vec_end; k += 16) {
            __m512 va = _mm512_loadu_ps(rep_a + k);
            _mm512_storeu_ps(out_rep + k, _mm512_mul_ps(va, v_scale_proj));
        }
        for (size_t rem = vec_end; rem < dim; ++rem) {
            out_rep[rem] = rep_a[rem] * neg_proj_inv_na;
        }
        out_rep[min_idx] += 1.0f;

        __m512 v_orth_sq = _mm512_setzero_ps();
        for (size_t k = 0; k < vec_end; k += 16) {
            __m512 vo = _mm512_loadu_ps(out_rep + k);
            v_orth_sq = _mm512_fmadd_ps(vo, vo, v_orth_sq);
        }
        float orth_nsq = _mm512_reduce_add_ps(v_orth_sq);
        for (size_t rem = vec_end; rem < dim; ++rem) {
            orth_nsq += out_rep[rem] * out_rep[rem];
        }
        const float inv_orth = (orth_nsq > 1e-12f) ? (1.0f / std::sqrt(orth_nsq)) : 1.0f;
        const float theta = t * 3.14159265358979323846f;
        const float scale_a = std::cos(theta) * (target_norm * inv_na);
        const float scale_o = std::sin(theta) * (target_norm * inv_orth);
        __m512 vs_a = _mm512_set1_ps(scale_a);
        __m512 vs_o = _mm512_set1_ps(scale_o);
        for (size_t k = 0; k < vec_end; k += 16) {
            __m512 va = _mm512_loadu_ps(rep_a + k);
            __m512 vo = _mm512_loadu_ps(out_rep + k);
            _mm512_storeu_ps(out_rep + k, _mm512_fmadd_ps(va, vs_a, _mm512_mul_ps(vo, vs_o)));
        }
        for (size_t rem = vec_end; rem < dim; ++rem) {
            out_rep[rem] = scale_a * rep_a[rem] + scale_o * out_rep[rem];
        }
        return 0;
    }

    __m512 vw_a = _mm512_set1_ps(w_a);
    __m512 vw_b = _mm512_set1_ps(w_b);
    __m512 acc_out = _mm512_setzero_ps();
    size_t j = 0;
    for (; j < vec_end; j += 16) {
        __m512 va = _mm512_loadu_ps(rep_a + j);
        __m512 vb = _mm512_loadu_ps(rep_b + j);
        __m512 vi = _mm512_fmadd_ps(va, vw_a, _mm512_mul_ps(vb, vw_b));
        _mm512_storeu_ps(out_rep + j, vi);
        acc_out = _mm512_fmadd_ps(vi, vi, acc_out);
    }
    float out_sq = _mm512_reduce_add_ps(acc_out);
    for (size_t rem = vec_end; rem < dim; ++rem) {
        float v = w_a * rep_a[rem] + w_b * rep_b[rem];
        out_rep[rem] = v;
        out_sq += v * v;
    }
    if (rescale_to_target && out_sq > 1e-12f) {
        const float scale = target_norm / std::sqrt(out_sq);
        __m512 v_scale = _mm512_set1_ps(scale);
        for (size_t k = 0; k < vec_end; k += 16) {
            _mm512_storeu_ps(out_rep + k, _mm512_mul_ps(_mm512_loadu_ps(out_rep + k), v_scale));
        }
        for (size_t rem = vec_end; rem < dim; ++rem) {
            out_rep[rem] *= scale;
        }
    }
    return 0;
}

int model_forward_deep_interpolated(
    void* ctx,
    const float* in_vec,
    float* out_vec
) {
    if (!ctx || !in_vec || !out_vec) return -1;
    ModelContext* m = (ModelContext*)ctx;
    if (m->num_layers == 0) return -2;

    const uint32_t L = m->num_layers;
    const uint32_t s_idx = (L >= 4) ? (L / 4 - 1) : 0;
    const uint32_t m_idx = (L >= 2) ? (L / 2 - 1) : 0;

    alignas(64) float cur_a[MODEL_DEFAULT_DIM_N];
    alignas(64) float cur_b[MODEL_DEFAULT_DIM_N];
    alignas(64) float snap_s[MODEL_DEFAULT_DIM_N];
    alignas(64) float snap_m[MODEL_DEFAULT_DIM_N];

    std::memcpy(cur_a, in_vec, MODEL_DEFAULT_DIM_N * sizeof(float));
    float* src = cur_a;
    float* dst = cur_b;

    for (uint32_t l = 0; l < L; ++l) {
        int rc = model_forward_layer(ctx, l, src, dst);
        if (rc != 0) return rc;
        if (l == s_idx) std::memcpy(snap_s, dst, MODEL_DEFAULT_DIM_N * sizeof(float));
        if (l == m_idx) std::memcpy(snap_m, dst, MODEL_DEFAULT_DIM_N * sizeof(float));
        std::swap(src, dst);
    }

    // Two-stage geodesic deep interpolation across shallow, mid, and deep layer manifolds
    alignas(64) float mid_blend[MODEL_DEFAULT_DIM_N];
    model_interpolate_representations(snap_s, snap_m, MODEL_DEFAULT_DIM_N, 0.65f, mid_blend);
    model_interpolate_representations(mid_blend, src, MODEL_DEFAULT_DIM_N, 0.72f, out_vec);
    return 0;
}

namespace {

struct ModelModCheckpointHeader {
    char     magic[8];      // "MMODWGT1"
    uint32_t version;       // 1
    uint32_t num_layers;
    uint32_t hidden_dim;
    uint32_t rank;
    uint64_t payload_fnv1a;
};

struct ModelLayerDiskMeta {
    uint32_t precision;
    uint32_t layer_type;
    uint32_t M;
    uint32_t N;
    uint32_t R;
};

static uint64_t mmod_fnv1a64(uint64_t h, const void* data, size_t len) noexcept {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

} // namespace

int model_save_weights(const void* ctx, const char* filepath) {
    if (!ctx || !filepath || filepath[0] == '\0') return -1;
    const ModelContext* m = static_cast<const ModelContext*>(ctx);
    if (!m->layers || m->num_layers == 0) return -2;

    FILE* fp = std::fopen(filepath, "wb");
    if (!fp) return -3;

    const size_t tt_bytes = static_cast<size_t>(MODEL_TT16_CORE_SIZE) * sizeof(float);
    const size_t dn_bytes = static_cast<size_t>(MODEL_DELTANET_HEADS * MODEL_DELTANET_HEAD_DIM * MODEL_DELTANET_HEAD_DIM) * sizeof(float);
    const size_t bw_bytes = sizeof(ModelBaseLayerWeights);
    uint64_t csum = 14695981039346656037ULL;

    for (uint32_t l = 0; l < m->num_layers; ++l) {
        ModelLayerDiskMeta meta{
            static_cast<uint32_t>(m->layers[l].precision),
            static_cast<uint32_t>(m->layers[l].layer_type),
            m->layers[l].M,
            m->layers[l].N,
            m->layers[l].R
        };
        csum = mmod_fnv1a64(csum, &meta, sizeof(meta));
        csum = mmod_fnv1a64(csum, m->layers[l].tt_u, tt_bytes);
        csum = mmod_fnv1a64(csum, m->layers[l].tt_v, tt_bytes);
        csum = mmod_fnv1a64(csum, m->layers[l].deltanet_state, dn_bytes);
        if (m->layers[l].cpu_matrix) {
            csum = mmod_fnv1a64(csum, m->layers[l].cpu_matrix, bw_bytes);
        }
    }

    ModelModCheckpointHeader hdr{};
    std::memcpy(hdr.magic, "MMODWGT1", 8);
    hdr.version       = 2U;
    hdr.num_layers    = m->num_layers;
    hdr.hidden_dim    = m->hidden_dim;
    hdr.rank          = MODEL_DEFAULT_RANK_R;
    hdr.payload_fnv1a = csum;

    bool ok = (std::fwrite(&hdr, sizeof(hdr), 1, fp) == 1);
    for (uint32_t l = 0; ok && l < m->num_layers; ++l) {
        ModelLayerDiskMeta meta{
            static_cast<uint32_t>(m->layers[l].precision),
            static_cast<uint32_t>(m->layers[l].layer_type),
            m->layers[l].M,
            m->layers[l].N,
            m->layers[l].R
        };
        ok = ok && (std::fwrite(&meta, sizeof(meta), 1, fp) == 1);
        ok = ok && (std::fwrite(m->layers[l].tt_u, 1, tt_bytes, fp) == tt_bytes);
        ok = ok && (std::fwrite(m->layers[l].tt_v, 1, tt_bytes, fp) == tt_bytes);
        ok = ok && (std::fwrite(m->layers[l].deltanet_state, 1, dn_bytes, fp) == dn_bytes);
        if (m->layers[l].cpu_matrix) {
            ok = ok && (std::fwrite(m->layers[l].cpu_matrix, 1, bw_bytes, fp) == bw_bytes);
        }
    }
    std::fclose(fp);
    return ok ? 0 : -4;
}

int model_load_weights(void* ctx, const char* filepath) {
    if (!ctx || !filepath || filepath[0] == '\0') return -1;
    ModelContext* m = static_cast<ModelContext*>(ctx);
    if (!m->layers || m->num_layers == 0) return -2;

    FILE* fp = std::fopen(filepath, "rb");
    if (!fp) return -3;

    ModelModCheckpointHeader hdr{};
    if (std::fread(&hdr, sizeof(hdr), 1, fp) != 1) {
        std::fclose(fp);
        return -4;
    }
    if (std::memcmp(hdr.magic, "MMODWGT1", 8) != 0 ||
        (hdr.version != 1U && hdr.version != 2U) ||
        hdr.num_layers != m->num_layers ||
        hdr.hidden_dim != m->hidden_dim ||
        hdr.rank != MODEL_DEFAULT_RANK_R) {
        std::fclose(fp);
        return -5;
    }

    const size_t L = m->num_layers;
    const size_t dn_elems = MODEL_DELTANET_HEADS * MODEL_DELTANET_HEAD_DIM * MODEL_DELTANET_HEAD_DIM;
    const size_t tt_bytes = static_cast<size_t>(MODEL_TT16_CORE_SIZE) * sizeof(float);
    const size_t dn_bytes = dn_elems * sizeof(float);
    const size_t bw_bytes = sizeof(ModelBaseLayerWeights);

    std::vector<ModelLayerDiskMeta> metas(L);
    std::vector<float> tmp_u(L * MODEL_TT16_CORE_SIZE);
    std::vector<float> tmp_v(L * MODEL_TT16_CORE_SIZE);
    std::vector<float> tmp_dn(L * dn_elems);
    std::vector<ModelBaseLayerWeights> tmp_bw(L);

    uint64_t csum = 14695981039346656037ULL;
    bool ok = true;
    for (size_t l = 0; ok && l < L; ++l) {
        float* u_ptr  = &tmp_u[l * MODEL_TT16_CORE_SIZE];
        float* v_ptr  = &tmp_v[l * MODEL_TT16_CORE_SIZE];
        float* dn_ptr = &tmp_dn[l * dn_elems];
        ok = ok && (std::fread(&metas[l], sizeof(ModelLayerDiskMeta), 1, fp) == 1);
        ok = ok && (std::fread(u_ptr, 1, tt_bytes, fp) == tt_bytes);
        ok = ok && (std::fread(v_ptr, 1, tt_bytes, fp) == tt_bytes);
        ok = ok && (std::fread(dn_ptr, 1, dn_bytes, fp) == dn_bytes);
        if (ok) {
            if (metas[l].M != MODEL_DEFAULT_DIM_M ||
                metas[l].N != MODEL_DEFAULT_DIM_N ||
                metas[l].R != MODEL_DEFAULT_RANK_R ||
                (metas[l].precision != static_cast<uint32_t>(QUANT_2BIT) &&
                 metas[l].precision != static_cast<uint32_t>(QUANT_3BIT) &&
                 metas[l].precision != static_cast<uint32_t>(QUANT_4BIT) &&
                 metas[l].precision != static_cast<uint32_t>(QUANT_DELTA_24) &&
                 metas[l].precision != static_cast<uint32_t>(QUANT_DELTA_34)) ||
                (metas[l].layer_type != static_cast<uint32_t>(LAYER_GQA_ATTENTION) &&
                 metas[l].layer_type != static_cast<uint32_t>(LAYER_GATED_DELTANET) &&
                 metas[l].layer_type != static_cast<uint32_t>(LAYER_AVX512_VSA_ASSOCIATIVE))) {
                ok = false;
            } else {
                csum = mmod_fnv1a64(csum, &metas[l], sizeof(ModelLayerDiskMeta));
                csum = mmod_fnv1a64(csum, u_ptr, tt_bytes);
                csum = mmod_fnv1a64(csum, v_ptr, tt_bytes);
                csum = mmod_fnv1a64(csum, dn_ptr, dn_bytes);
            }
        }
        if (ok && hdr.version >= 2U) {
            ok = (std::fread(&tmp_bw[l], 1, bw_bytes, fp) == bw_bytes);
            if (ok) {
                csum = mmod_fnv1a64(csum, &tmp_bw[l], bw_bytes);
            }
        } else if (ok) {
            mmod_init_base_layer_weights(&tmp_bw[l]);
        }
    }
    std::fclose(fp);

    if (!ok || csum != hdr.payload_fnv1a) return -6;

    for (size_t l = 0; l < L; ++l) {
        std::atomic_ref<QuantPrecision>(m->layers[l].precision).store(
            static_cast<QuantPrecision>(metas[l].precision), std::memory_order_release);
        std::atomic_ref<LayerType>(m->layers[l].layer_type).store(
            static_cast<LayerType>(metas[l].layer_type), std::memory_order_release);
        std::memcpy(m->layers[l].tt_u, &tmp_u[l * MODEL_TT16_CORE_SIZE], tt_bytes);
        std::memcpy(m->layers[l].tt_v, &tmp_v[l * MODEL_TT16_CORE_SIZE], tt_bytes);
        std::memcpy(m->layers[l].deltanet_state, &tmp_dn[l * dn_elems], dn_bytes);
        if (m->layers[l].cpu_matrix) {
            std::memcpy(m->layers[l].cpu_matrix, &tmp_bw[l], bw_bytes);
        }
    }
    return 0;
}

int model_train_layer_step(
    void* ctx,
    uint32_t layer_idx,
    const float* in_vec,
    const float* target_vec,
    float lr,
    float* out_mse_loss
) {
    if (!ctx || !in_vec || !target_vec || !(lr > 0.0f)) return -1;
    ModelContext* m = static_cast<ModelContext*>(ctx);
    if (layer_idx >= m->num_layers) return -2;

    alignas(64) float pred_vec[MODEL_DEFAULT_DIM_M];
    int rc = model_forward_layer(ctx, layer_idx, in_vec, pred_vec);
    if (rc != 0) return rc;

    alignas(64) float err_vec[MODEL_DEFAULT_DIM_M];
    __m512 v_sq_err = _mm512_setzero_ps();
    __m512 v_sq_x   = _mm512_setzero_ps();
    for (size_t j = 0; j < MODEL_DEFAULT_DIM_M; j += 16) {
        __m512 vt = _mm512_loadu_ps(target_vec + j);
        __m512 vp = _mm512_load_ps(pred_vec + j);
        __m512 vx = _mm512_loadu_ps(in_vec + j);
        __m512 ve = _mm512_sub_ps(vt, vp);
        _mm512_store_ps(err_vec + j, ve);
        v_sq_err = _mm512_fmadd_ps(ve, ve, v_sq_err);
        v_sq_x   = _mm512_fmadd_ps(vx, vx, v_sq_x);
    }
    const float mse = _mm512_reduce_add_ps(v_sq_err) / static_cast<float>(MODEL_DEFAULT_DIM_M);
    if (out_mse_loss) *out_mse_loss = mse;

    const float x_norm_sq = std::max(1e-6f, _mm512_reduce_add_ps(v_sq_x));
    const float x_norm    = std::sqrt(x_norm_sq);
    const float inv_sqrt_r_x = 1.0f / (std::sqrt(static_cast<float>(MODEL_DEFAULT_RANK_R)) * x_norm);

    ModelLayerConfig* layer = &m->layers[layer_idx];
    const LayerType active_ltype =
        std::atomic_ref<LayerType>(layer->layer_type).load(std::memory_order_acquire);
    ModelBaseLayerWeights* bw = static_cast<ModelBaseLayerWeights*>(layer->cpu_matrix);
    if (bw) {
        const float base_lr = (0.25f * lr) / x_norm_sq;
        const size_t H_Q  = MODEL_DELTANET_HEADS;
        const size_t H_KV = 2;
        const size_t G    = H_Q / H_KV;
        const size_t D_H  = MODEL_DELTANET_HEAD_DIM;
        for (size_t h = 0; h < H_Q; ++h) {
            const size_t kv_head = h / G;
            const size_t base_off = h * D_H;
            const size_t rot_q = ((size_t)(layer_idx * 5 + h * 3 + 1)) & (D_H - 1);
            const size_t rot_k = ((size_t)(layer_idx * 7 + kv_head * 5 + 3)) & (D_H - 1);
            const size_t rot_v = ((size_t)(layer_idx * 11 + kv_head * 7 + 5)) & (D_H - 1);
            const float* q_src = &in_vec[h * D_H];
            const float* k_src = &in_vec[kv_head * D_H];
            const float* v_src = &in_vec[512 + kv_head * D_H];
            const float* e_h   = &err_vec[base_off];
            for (size_t j = 0; j < D_H; ++j) {
                const size_t d_idx = base_off + j;
                const float ej = e_h[j];
                bw->w_v_diag[d_idx]  = std::clamp(bw->w_v_diag[d_idx]  + base_lr * ej * v_src[j], -2.0f, 2.0f);
                bw->w_v_rot[d_idx]   = std::clamp(bw->w_v_rot[d_idx]   + base_lr * ej * v_src[(j + rot_v) & (D_H - 1)], -2.0f, 2.0f);
                bw->w_q_diag[d_idx]  = std::clamp(bw->w_q_diag[d_idx]  + 0.5f * base_lr * ej * q_src[j], -2.0f, 2.0f);
                bw->w_q_rot[d_idx]   = std::clamp(bw->w_q_rot[d_idx]   + 0.5f * base_lr * ej * q_src[(j + rot_q) & (D_H - 1)], -2.0f, 2.0f);
                bw->w_k_diag[d_idx]  = std::clamp(bw->w_k_diag[d_idx]  + 0.5f * base_lr * ej * k_src[j], -2.0f, 2.0f);
                bw->w_k_rot[d_idx]   = std::clamp(bw->w_k_rot[d_idx]   + 0.5f * base_lr * ej * k_src[(j + rot_k) & (D_H - 1)], -2.0f, 2.0f);
                if (active_ltype == LAYER_GATED_DELTANET) {
                    bw->w_dn_in[d_idx]   = std::clamp(bw->w_dn_in[d_idx]   + 0.5f * base_lr * ej * q_src[j], 0.1f, 3.0f);
                    bw->w_dn_gate[d_idx] = std::clamp(bw->w_dn_gate[d_idx] + 0.25f * base_lr * ej * q_src[j], -3.0f, 3.0f);
                }
            }
        }
        mmod_quantize_base_ternary(bw);
    }

    // Coupled low-rank TT16 gradient update matching model_apply_tt16_residual:
    // z_r = <v_r, x>, y += sum_r z_r * u_r  =>  du_r = lr * z_r * e, dv_r = lr * <u_r, e> * x
    for (size_t r = 0; r < MODEL_DEFAULT_RANK_R; ++r) {
        float* u_row = &layer->tt_u[r * MODEL_DEFAULT_DIM_N];
        float* v_row = &layer->tt_v[r * MODEL_DEFAULT_DIM_M];

        __m512 acc_vx = _mm512_setzero_ps();
        __m512 acc_ue = _mm512_setzero_ps();
        __m512 acc_vv = _mm512_setzero_ps();
        for (size_t j = 0; j < MODEL_DEFAULT_DIM_N; j += 16) {
            __m512 vu = _mm512_load_ps(u_row + j);
            __m512 vv = _mm512_load_ps(v_row + j);
            __m512 vx = _mm512_loadu_ps(in_vec + j);
            __m512 ve = _mm512_load_ps(err_vec + j);
            acc_vx = _mm512_fmadd_ps(vv, vx, acc_vx);
            acc_ue = _mm512_fmadd_ps(vu, ve, acc_ue);
            acc_vv = _mm512_fmadd_ps(vv, vv, acc_vv);
        }
        const float vv_sq = _mm512_reduce_add_ps(acc_vv);
        if (vv_sq < 1e-10f) {
            const __m512 v_init_v = _mm512_set1_ps(inv_sqrt_r_x);
            const __m512 v_step_u = _mm512_set1_ps(lr * inv_sqrt_r_x);
            for (size_t j = 0; j < MODEL_DEFAULT_DIM_M; j += 16) {
                __m512 vu   = _mm512_load_ps(u_row + j);
                __m512 vx   = _mm512_loadu_ps(in_vec + j);
                __m512 verr = _mm512_load_ps(err_vec + j);
                _mm512_store_ps(v_row + j, _mm512_mul_ps(v_init_v, vx));
                _mm512_store_ps(u_row + j, _mm512_fmadd_ps(v_step_u, verr, vu));
            }
        } else {
            const float z_r = _mm512_reduce_add_ps(acc_vx);
            const float g_r = _mm512_reduce_add_ps(acc_ue);
            const __m512 v_step_u = _mm512_set1_ps((lr / x_norm_sq) * z_r);
            const __m512 v_step_v = _mm512_set1_ps((lr / x_norm_sq) * g_r);
            for (size_t j = 0; j < MODEL_DEFAULT_DIM_M; j += 16) {
                __m512 vu   = _mm512_load_ps(u_row + j);
                __m512 vv   = _mm512_load_ps(v_row + j);
                __m512 vx   = _mm512_loadu_ps(in_vec + j);
                __m512 verr = _mm512_load_ps(err_vec + j);
                _mm512_store_ps(u_row + j, _mm512_fmadd_ps(v_step_u, verr, vu));
                _mm512_store_ps(v_row + j, _mm512_fmadd_ps(v_step_v, vx, vv));
            }
        }
    }
    return 0;
}


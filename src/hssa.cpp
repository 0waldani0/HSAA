#include "hssa.h"
#include "ptq_cuda.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <time.h>
#include <stdlib.h>
#include <string.h>
typedef union {
    int64_t QuadPart;
} LARGE_INTEGER;
static inline void QueryPerformanceFrequency(LARGE_INTEGER* lpFrequency) {
    lpFrequency->QuadPart = 1000000000LL;
}
static inline void QueryPerformanceCounter(LARGE_INTEGER* lpPerformanceCount) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    lpPerformanceCount->QuadPart = (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}
static inline void* _aligned_malloc(size_t size, size_t alignment) {
    void* ptr = nullptr;
    size_t aligned_sz = (size + alignment - 1) & ~(alignment - 1);
    if (posix_memalign(&ptr, alignment, aligned_sz) != 0) return nullptr;
    return ptr;
}
static inline void _aligned_free(void* ptr) {
    free(ptr);
}
#ifndef _TRUNCATE
#define _TRUNCATE ((size_t)-1)
#endif
static inline void strncpy_s(char* dest, size_t dest_sz, const char* src, size_t count) {
    if (!dest || dest_sz == 0) return;
    if (!src) { dest[0] = '\0'; return; }
    size_t n = (count == _TRUNCATE || count >= dest_sz) ? (dest_sz - 1) : count;
    strncpy(dest, src, n);
    dest[n] = '\0';
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
#include <mutex>
#include <new>
#include <vector>

/*
 * =============================================================================
 * Milestone M1: Holographic State-Space Attractor (HSSA) Architecture Engine
 * Pure Native C++20 / AVX-512 Implementation
 * =============================================================================
 */

/* Milestone M3: default autoregressive vocabulary size */
#define HSSA_DEFAULT_VOCAB 256

/* Milestone M3: hard ceiling on the unembedding projection (262,144 tokens = 1 GB of
 * float32 weights). Keeps host RAM inside the 32 GB workstation envelope. */
#define HSSA_MAX_VOCAB 262144

/* High-precision timer helper using Windows QPC */
static double get_qpc_time_us() {
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)(counter.QuadPart * 1000000.0) / (double)freq.QuadPart;
}

/* Fast sigmoid helper: 1 / (1 + exp(-x)) */
static inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

struct alignas(64) HssaBaseLayerWeights {
    alignas(64) float w_k_diag[HSSA_HIDDEN_DIM];
    alignas(64) float w_k_rot[HSSA_HIDDEN_DIM];
    alignas(64) float w_q_diag[HSSA_HIDDEN_DIM];
    alignas(64) float w_q_rot[HSSA_HIDDEN_DIM];
    alignas(64) float w_v_diag[HSSA_HIDDEN_DIM];
    alignas(64) float w_v_rot[HSSA_HIDDEN_DIM];
    alignas(64) float w_g_diag[HSSA_HIDDEN_DIM];
    alignas(64) float w_g_rot[HSSA_HIDDEN_DIM];
    uint64_t k_pos[HSSA_HIDDEN_DIM / 64];
    uint64_t k_neg[HSSA_HIDDEN_DIM / 64];
    uint64_t q_pos[HSSA_HIDDEN_DIM / 64];
    uint64_t q_neg[HSSA_HIDDEN_DIM / 64];
    uint64_t v_pos[HSSA_HIDDEN_DIM / 64];
    uint64_t v_neg[HSSA_HIDDEN_DIM / 64];
    uint64_t g_pos[HSSA_HIDDEN_DIM / 64];
    uint64_t g_neg[HSSA_HIDDEN_DIM / 64];
    float scale_k_rot;
    float scale_q_rot;
    float scale_v_rot;
    float scale_g_rot;
};

static inline void quantize_ternary_channel_1024(
    const float* src,
    uint64_t* out_pos,
    uint64_t* out_neg,
    float* out_scale
) noexcept {
    float sum_abs = 0.0f;
    for (size_t i = 0; i < HSSA_HIDDEN_DIM; ++i) {
        sum_abs += std::fabs(src[i]);
    }
    const float mean_abs = sum_abs / static_cast<float>(HSSA_HIDDEN_DIM);
    const float thresh = 0.35f * mean_abs;
    std::memset(out_pos, 0, (HSSA_HIDDEN_DIM / 64) * sizeof(uint64_t));
    std::memset(out_neg, 0, (HSSA_HIDDEN_DIM / 64) * sizeof(uint64_t));
    float active_sum = 0.0f;
    uint32_t active_cnt = 0;
    for (size_t i = 0; i < HSSA_HIDDEN_DIM; ++i) {
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

static inline void hssa_quantize_base_ternary(HssaBaseLayerWeights* bw) noexcept {
    if (!bw) return;
    quantize_ternary_channel_1024(bw->w_k_rot, bw->k_pos, bw->k_neg, &bw->scale_k_rot);
    quantize_ternary_channel_1024(bw->w_q_rot, bw->q_pos, bw->q_neg, &bw->scale_q_rot);
    quantize_ternary_channel_1024(bw->w_v_rot, bw->v_pos, bw->v_neg, &bw->scale_v_rot);
    quantize_ternary_channel_1024(bw->w_g_rot, bw->g_pos, bw->g_neg, &bw->scale_g_rot);
}

static inline void hssa_init_base_layer_weights(HssaBaseLayerWeights* bw) noexcept {
    if (!bw) return;
    for (size_t i = 0; i < HSSA_HIDDEN_DIM; ++i) {
        bw->w_k_diag[i] = 0.85f;
        bw->w_k_rot[i]  = 0.15f;
        bw->w_q_diag[i] = 0.85f;
        bw->w_q_rot[i]  = -0.15f;
        bw->w_v_diag[i] = 0.90f;
        bw->w_v_rot[i]  = 0.10f;
        bw->w_g_diag[i] = 0.80f;
        bw->w_g_rot[i]  = 0.20f;
    }
    hssa_quantize_base_ternary(bw);
}

/* Deterministic normalized unembedding projection weights [vocab x 1024].
 * Rows follow the same orthogonal sinusoidal basis as the Hopfield attractor
 * patterns, yielding a reproducible R^{1024} -> logits map with zero external
 * dependencies and zero mock constants. */
static void build_deterministic_unembedding(float* weights, size_t vocab) {
    const float norm_factor = 1.0f / std::sqrt((float)HSSA_HIDDEN_DIM);
    for (size_t v = 0; v < vocab; ++v) {
        float* row = &weights[v * HSSA_HIDDEN_DIM];
        const float freq = (float)(v + 1);
        for (size_t j = 0; j < HSSA_HIDDEN_DIM; ++j) {
            row[j] = std::cos((2.0f * 3.14159265358979323846f * freq * (float)j) / (float)HSSA_HIDDEN_DIM) * norm_factor;
        }
    }
}

/*
 * =============================================================================
 * Lifecycle Management
 * =============================================================================
 */

HssaContext* hssa_create(uint32_t num_layers) {
    if (num_layers == 0) {
        num_layers = HSSA_DEFAULT_NUM_LAYERS;
    }

    HssaContext* ctx = (HssaContext*)std::calloc(1, sizeof(HssaContext));
    if (!ctx) return nullptr;

    ctx->state_lock = new (std::nothrow) std::recursive_mutex();
    if (!ctx->state_lock) {
        std::free(ctx);
        return nullptr;
    }

    ctx->config.num_layers = num_layers;
    ctx->config.hidden_dim = HSSA_HIDDEN_DIM;
    ctx->config.heads = HSSA_HEADS;
    ctx->config.head_dim = HSSA_HEAD_DIM;
    ctx->config.vsa_dim = HSSA_VSA_DIM;
    ctx->config.rank = HSSA_RANK;

    ctx->layers = (HssaLayer*)std::calloc(num_layers, sizeof(HssaLayer));
    if (!ctx->layers) {
        delete static_cast<std::recursive_mutex*>(ctx->state_lock);
        ctx->state_lock = nullptr;
        std::free(ctx);
        return nullptr;
    }

    // Allocate 64-byte aligned recurrent state, base weights, and TT16 cores per layer
    for (uint32_t l = 0; l < num_layers; ++l) {
        HssaLayer* lyr = &ctx->layers[l];
        lyr->layer_idx = l;

        lyr->deltanet_state = (float*)_aligned_malloc(HSSA_DELTANET_STATE_SIZE * sizeof(float), HSSA_ALIGNMENT);
        lyr->tt_u = (float*)_aligned_malloc(HSSA_TT16_CORE_SIZE * sizeof(float), HSSA_ALIGNMENT);
        lyr->tt_v = (float*)_aligned_malloc(HSSA_TT16_CORE_SIZE * sizeof(float), HSSA_ALIGNMENT);
        HssaBaseLayerWeights* bw = (HssaBaseLayerWeights*)_aligned_malloc(sizeof(HssaBaseLayerWeights), HSSA_ALIGNMENT);

        if (!lyr->deltanet_state || !lyr->tt_u || !lyr->tt_v || !bw) {
            if (bw) _aligned_free(bw);
            hssa_destroy(ctx);
            return nullptr;
        }

        std::memset(lyr->deltanet_state, 0, HSSA_DELTANET_STATE_SIZE * sizeof(float));
        std::memset(lyr->tt_u, 0, HSSA_TT16_CORE_SIZE * sizeof(float));
        std::memset(lyr->tt_v, 0, HSSA_TT16_CORE_SIZE * sizeof(float));
        hssa_init_base_layer_weights(bw);

        lyr->vsa_handle = nullptr;
        lyr->reserved[0] = nullptr;
        lyr->reserved[1] = bw;
        lyr->reserved[2] = nullptr;
    }

    // Preallocate 64-byte aligned hot-path scratchpads to guarantee zero heap allocations
    ctx->scratch_in = (float*)_aligned_malloc(HSSA_HIDDEN_DIM * sizeof(float), HSSA_ALIGNMENT);
    ctx->scratch_out = (float*)_aligned_malloc(HSSA_HIDDEN_DIM * sizeof(float), HSSA_ALIGNMENT);
    ctx->scratch_state = (float*)_aligned_malloc(HSSA_HIDDEN_DIM * sizeof(float), HSSA_ALIGNMENT);
    ctx->scratch_vsa = (uint64_t*)_aligned_malloc(HSSA_VSA_WORDS * sizeof(uint64_t), HSSA_ALIGNMENT);

    if (!ctx->scratch_in || !ctx->scratch_out || !ctx->scratch_state || !ctx->scratch_vsa) {
        hssa_destroy(ctx);
        return nullptr;
    }

    std::memset(ctx->scratch_in, 0, HSSA_HIDDEN_DIM * sizeof(float));
    std::memset(ctx->scratch_out, 0, HSSA_HIDDEN_DIM * sizeof(float));
    std::memset(ctx->scratch_state, 0, HSSA_HIDDEN_DIM * sizeof(float));
    std::memset(ctx->scratch_vsa, 0, HSSA_VSA_WORDS * sizeof(uint64_t));

    // Allocate and initialize Continuous Hopfield Attractor Memory Patterns
    ctx->num_attractor_patterns = HSSA_MAX_PATTERNS;
    ctx->attractor_patterns = (float*)_aligned_malloc(ctx->num_attractor_patterns * HSSA_HIDDEN_DIM * sizeof(float), HSSA_ALIGNMENT);
    if (!ctx->attractor_patterns) {
        hssa_destroy(ctx);
        return nullptr;
    }

    // Deterministic normalized orthogonal sinusoidal basis patterns
    const float norm_factor = 1.0f / std::sqrt((float)HSSA_HIDDEN_DIM);
    for (size_t m = 0; m < ctx->num_attractor_patterns; ++m) {
        float* pat = &ctx->attractor_patterns[m * HSSA_HIDDEN_DIM];
        float freq = (float)(m + 1);
        for (size_t j = 0; j < HSSA_HIDDEN_DIM; ++j) {
            pat[j] = std::cos((2.0f * 3.14159265358979323846f * freq * (float)j) / (float)HSSA_HIDDEN_DIM) * norm_factor;
        }
    }
    ctx->hopfield_tau = 0.5f;

    // Allocate default global episodic memory
    ctx->global_vsa_db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    ctx->owns_global_vsa_db = true;

    // Milestone M3: candidate assertion registry. Grown lazily (doubling, capped at
    // HSSA_MAX_ASSERTIONS) so an idle context pays nothing for it; hssa_destroy frees it.
    ctx->assertions = nullptr;
    ctx->assertion_count = 0;
    ctx->assertion_capacity = 0;

    // Milestone M3: default unembedding projection and logit scratchpad so that
    // hssa_generate is valid immediately after hssa_create without extra setup.
    ctx->vocab_size = HSSA_DEFAULT_VOCAB;
    ctx->unembed_weights = (float*)_aligned_malloc(HSSA_DEFAULT_VOCAB * HSSA_HIDDEN_DIM * sizeof(float), HSSA_ALIGNMENT);
    ctx->logits_buf = (float*)_aligned_malloc(HSSA_DEFAULT_VOCAB * sizeof(float), HSSA_ALIGNMENT);
    if (!ctx->unembed_weights || !ctx->logits_buf) {
        hssa_destroy(ctx);
        return nullptr;
    }
    build_deterministic_unembedding(ctx->unembed_weights, HSSA_DEFAULT_VOCAB);

    // Preallocate 64-byte aligned multi-depth continuous interpolation scratchpad (3 * 1024 floats) in reserved[0]
    ctx->reserved[0] = _aligned_malloc(3 * HSSA_HIDDEN_DIM * sizeof(float), HSSA_ALIGNMENT);
    if (ctx->reserved[0]) {
        std::memset(ctx->reserved[0], 0, 3 * HSSA_HIDDEN_DIM * sizeof(float));
    }

    return ctx;
}

void hssa_destroy(HssaContext* ctx) {
    if (!ctx) return;

    if (ctx->cuda_active || ctx->cuda_drv) {
        CudaDrv* drv = (CudaDrv*)ctx->cuda_drv;
        if (drv) {
            if (ctx->gpu_partials) {
                drv->cuMemFree((CUdeviceptr)ctx->gpu_partials);
                ctx->gpu_partials = 0;
            }
            if (ctx->gpu_tile_cnt) {
                drv->cuMemFree((CUdeviceptr)ctx->gpu_tile_cnt);
                ctx->gpu_tile_cnt = 0;
            }
            if (ctx->gpu_out_buf) {
                drv->cuMemFree((CUdeviceptr)ctx->gpu_out_buf);
                ctx->gpu_out_buf = 0;
            }
            if (ctx->gpu_matrix) {
                ptq_matrix_free(drv, (PtqGpuMatrix*)ctx->gpu_matrix);
                std::free(ctx->gpu_matrix);
                ctx->gpu_matrix = nullptr;
            }
            if (ctx->cuda_kernels) {
                ptq_kernels_unload(drv, (PtqKernels*)ctx->cuda_kernels);
                std::free(ctx->cuda_kernels);
                ctx->cuda_kernels = nullptr;
            }
            if (ctx->cuda_ctx) {
                ptq_cuda_close(drv, (CUcontext)ctx->cuda_ctx);
                ctx->cuda_ctx = nullptr;
            }
            std::free(drv);
            ctx->cuda_drv = nullptr;
        }
        ctx->cuda_active = false;
    }

    if (ctx->hybrid_bridge) {
        bridge_free_buffer(ctx->hybrid_bridge);
        ctx->hybrid_bridge = nullptr;
    }

    if (ctx->layers) {
        for (uint32_t l = 0; l < ctx->config.num_layers; ++l) {
            HssaLayer* lyr = &ctx->layers[l];
            if (lyr->deltanet_state) _aligned_free(lyr->deltanet_state);
            if (lyr->tt_u) _aligned_free(lyr->tt_u);
            if (lyr->tt_v) _aligned_free(lyr->tt_v);
            if (lyr->reserved[1]) {
                _aligned_free(lyr->reserved[1]);
                lyr->reserved[1] = nullptr;
            }
        }
        std::free(ctx->layers);
        ctx->layers = nullptr;
    }

    if (ctx->scratch_in) _aligned_free(ctx->scratch_in);
    if (ctx->scratch_out) _aligned_free(ctx->scratch_out);
    if (ctx->scratch_state) _aligned_free(ctx->scratch_state);
    if (ctx->scratch_vsa) _aligned_free(ctx->scratch_vsa);
    if (ctx->attractor_patterns) _aligned_free(ctx->attractor_patterns);
    if (ctx->reserved[0]) {
        _aligned_free(ctx->reserved[0]);
        ctx->reserved[0] = nullptr;
    }

    // Milestone M3: candidate assertion registry, unembedding matrix, logit buffer
    if (ctx->assertions) {
        std::free(ctx->assertions);
        ctx->assertions = nullptr;
    }
    if (ctx->unembed_weights) {
        _aligned_free(ctx->unembed_weights);
        ctx->unembed_weights = nullptr;
    }
    if (ctx->logits_buf) {
        _aligned_free(ctx->logits_buf);
        ctx->logits_buf = nullptr;
    }
    ctx->assertion_count = 0;
    ctx->assertion_capacity = 0;
    ctx->vocab_size = 0;

    if (ctx->owns_global_vsa_db && ctx->global_vsa_db) {
        vsa_db_destroy(ctx->global_vsa_db);
        ctx->global_vsa_db = nullptr;
    }

    delete static_cast<std::recursive_mutex*>(ctx->state_lock);
    ctx->state_lock = nullptr;

    std::free(ctx);
}

int hssa_reset_state(HssaContext* ctx) {
    if (!ctx || !ctx->layers) return -1;
    if (ctx->state_lock) {
        std::lock_guard<std::recursive_mutex> state_guard(*static_cast<std::recursive_mutex*>(ctx->state_lock));
        for (uint32_t l = 0; l < ctx->config.num_layers; ++l) {
            HssaLayer* lyr = &ctx->layers[l];
            if (lyr->deltanet_state) {
                std::memset(lyr->deltanet_state, 0, HSSA_DELTANET_STATE_SIZE * sizeof(float));
            }
        }
    } else {
        for (uint32_t l = 0; l < ctx->config.num_layers; ++l) {
            HssaLayer* lyr = &ctx->layers[l];
            if (lyr->deltanet_state) {
                std::memset(lyr->deltanet_state, 0, HSSA_DELTANET_STATE_SIZE * sizeof(float));
            }
        }
    }
    return 0;
}

/*
 * =============================================================================
 * Tier 1: Gated DeltaNet State-Space Recurrence Kernel
 * =============================================================================
 */

/* Full AVX-512 non-zero check for TT16 core presence via _mm512_or_si512.
 * Unrolls across 64 floats (4 x 512-bit vectors) per iteration to eliminate
 * per-vector branching and bitwise ORs all lanes into a single accumulator. */
static inline bool hssa_is_tt16_active(const float* u_ptr, const float* v_ptr, size_t count) {
    if (!u_ptr && !v_ptr) return false;
    __m512i acc = _mm512_setzero_si512();

    if (u_ptr) {
        size_t i = 0;
        for (; i + 64 <= count; i += 64) {
            __m512i u0 = _mm512_loadu_si512((const __m512i*)(u_ptr + i));
            __m512i u1 = _mm512_loadu_si512((const __m512i*)(u_ptr + i + 16));
            __m512i u2 = _mm512_loadu_si512((const __m512i*)(u_ptr + i + 32));
            __m512i u3 = _mm512_loadu_si512((const __m512i*)(u_ptr + i + 48));
            acc = _mm512_or_si512(acc, _mm512_or_si512(_mm512_or_si512(u0, u1), _mm512_or_si512(u2, u3)));
        }
        for (; i + 16 <= count; i += 16) {
            acc = _mm512_or_si512(acc, _mm512_loadu_si512((const __m512i*)(u_ptr + i)));
        }
        for (; i < count; ++i) {
            if (u_ptr[i] != 0.0f) return true;
        }
    }

    if (v_ptr) {
        size_t i = 0;
        for (; i + 64 <= count; i += 64) {
            __m512i v0 = _mm512_loadu_si512((const __m512i*)(v_ptr + i));
            __m512i v1 = _mm512_loadu_si512((const __m512i*)(v_ptr + i + 16));
            __m512i v2 = _mm512_loadu_si512((const __m512i*)(v_ptr + i + 32));
            __m512i v3 = _mm512_loadu_si512((const __m512i*)(v_ptr + i + 48));
            acc = _mm512_or_si512(acc, _mm512_or_si512(_mm512_or_si512(v0, v1), _mm512_or_si512(v2, v3)));
        }
        for (; i + 16 <= count; i += 16) {
            acc = _mm512_or_si512(acc, _mm512_loadu_si512((const __m512i*)(v_ptr + i)));
        }
        for (; i < count; ++i) {
            if (v_ptr[i] != 0.0f) return true;
        }
    }

    return _mm512_test_epi64_mask(acc, acc) != 0;
}

static inline bool hssa_is_layer_tt16_active(HssaLayer* layer) {
    if (!layer) return false;
    return layer->reserved[0] != nullptr;
}

static void forward_deltanet_layer(HssaLayer* layer, const float* in_vec, float* out_vec) {
    const size_t H = HSSA_HEADS;       // 8 heads
    const size_t D_H = HSSA_HEAD_DIM;  // 128 dimensions per head
    const float alpha = 0.95f;         // State retention decay
    const float beta  = 0.25f;         // Fast associative learning rate
    const uint32_t l_idx = layer ? layer->layer_idx : 0;
    const HssaBaseLayerWeights* bw = layer ? static_cast<const HssaBaseLayerWeights*>(layer->reserved[1]) : nullptr;

    __m512 v_alpha = _mm512_set1_ps(alpha);

    #pragma omp parallel for schedule(static)
    for (int h = 0; h < (int)H; ++h) {
        const float* x_h = &in_vec[h * D_H];
        float* S_h = &layer->deltanet_state[h * D_H * D_H];
        float* out_h = &out_vec[h * D_H];

        // Trainable 1.58-bit ternary + continuous STE per-head Q, K, V, Gate projections
        const size_t rot_k = ((size_t)(l_idx * 7 + h * 3 + 1)) & (D_H - 1);
        const size_t rot_q = ((size_t)(l_idx * 11 + h * 5 + 3)) & (D_H - 1);
        const size_t rot_v = ((size_t)(l_idx * 13 + h * 7 + 5)) & (D_H - 1);
        alignas(64) float q_proj[HSSA_HEAD_DIM];
        alignas(64) float k_proj[HSSA_HEAD_DIM];
        alignas(64) float v_proj[HSSA_HEAD_DIM];
        alignas(64) float gate[HSSA_HEAD_DIM];

        if (bw) {
            const size_t base_off = static_cast<size_t>(h) * D_H;
            for (size_t r = 0; r < D_H; ++r) {
                const size_t d_idx = base_off + r;
                const size_t w_idx = d_idx >> 6;
                const uint64_t bit = 1ULL << (d_idx & 63);
                const float x0 = x_h[r];
                const float xk = x_h[(r + rot_k) & (D_H - 1)];
                const float xq = x_h[(r + rot_q) & (D_H - 1)];
                const float xv = x_h[(r + rot_v) & (D_H - 1)];

                const float tk = ((bw->k_pos[w_idx] & bit) ? 1.0f : 0.0f) - ((bw->k_neg[w_idx] & bit) ? 1.0f : 0.0f);
                const float tq = ((bw->q_pos[w_idx] & bit) ? 1.0f : 0.0f) - ((bw->q_neg[w_idx] & bit) ? 1.0f : 0.0f);
                const float tv = ((bw->v_pos[w_idx] & bit) ? 1.0f : 0.0f) - ((bw->v_neg[w_idx] & bit) ? 1.0f : 0.0f);
                const float tg = ((bw->g_pos[w_idx] & bit) ? 1.0f : 0.0f) - ((bw->g_neg[w_idx] & bit) ? 1.0f : 0.0f);

                const float wk_rot = 0.5f * bw->w_k_rot[d_idx] + 0.5f * bw->scale_k_rot * tk;
                const float wq_rot = 0.5f * bw->w_q_rot[d_idx] + 0.5f * bw->scale_q_rot * tq;
                const float wv_rot = 0.5f * bw->w_v_rot[d_idx] + 0.5f * bw->scale_v_rot * tv;
                const float wg_rot = 0.5f * bw->w_g_rot[d_idx] + 0.5f * bw->scale_g_rot * tg;

                k_proj[r] = bw->w_k_diag[d_idx] * x0 + wk_rot * xk;
                q_proj[r] = bw->w_q_diag[d_idx] * x0 + wq_rot * xq;
                v_proj[r] = bw->w_v_diag[d_idx] * x0 + wv_rot * xv;
                gate[r]   = fast_sigmoid(bw->w_g_diag[d_idx] * x0 + wg_rot * xq);
            }
        } else {
            for (size_t r = 0; r < D_H; ++r) {
                const float x0 = x_h[r];
                const float xk = x_h[(r + rot_k) & (D_H - 1)];
                const float xq = x_h[(r + rot_q) & (D_H - 1)];
                const float xv = x_h[(r + rot_v) & (D_H - 1)];
                k_proj[r] = 0.85f * x0 + 0.15f * xk;
                q_proj[r] = 0.85f * x0 - 0.15f * xq;
                v_proj[r] = 0.90f * x0 + 0.10f * xv;
                gate[r]   = fast_sigmoid(0.80f * x0 + 0.20f * xq);
            }
        }

        // Preload q_vec and k_raw [8 * 16 = 128 floats]
        __m512 q_vec[8];
        __m512 k_raw[8];
        for (size_t i = 0; i < 8; ++i) {
            q_vec[i] = _mm512_load_ps(q_proj + i * 16);
            k_raw[i] = _mm512_load_ps(k_proj + i * 16);
        }

        // 1. Key normalization: k_bar = k_proj / max(||k_proj||_2, 1e-6)
        __m512 sq_acc0 = _mm512_setzero_ps();
        __m512 sq_acc1 = _mm512_setzero_ps();
        for (size_t i = 0; i < 8; i += 2) {
            sq_acc0 = _mm512_fmadd_ps(k_raw[i], k_raw[i], sq_acc0);
            sq_acc1 = _mm512_fmadd_ps(k_raw[i + 1], k_raw[i + 1], sq_acc1);
        }
        float sum_sq = _mm512_reduce_add_ps(_mm512_add_ps(sq_acc0, sq_acc1));
        float norm_k = std::sqrt(sum_sq);
        float inv_norm = norm_k > 1e-6f ? (1.0f / norm_k) : 0.0f;

        __m512 k_vec[8];
        __m512 v_inv_norm = _mm512_set1_ps(inv_norm);
        for (size_t i = 0; i < 8; ++i) {
            k_vec[i] = _mm512_mul_ps(k_raw[i], v_inv_norm);
        }

        const float head_scale = 0.0883883476f; // 1 / sqrt(128)

        // Fused row execution: Single pass over S_h rows with 1 aligned load & 1 aligned store per element
        for (size_t r = 0; r < D_H; ++r) {
            float* row = &S_h[r * D_H];

            __m512 s0 = _mm512_load_ps(row);
            __m512 s1 = _mm512_load_ps(row + 16);
            __m512 s2 = _mm512_load_ps(row + 32);
            __m512 s3 = _mm512_load_ps(row + 48);
            __m512 s4 = _mm512_load_ps(row + 64);
            __m512 s5 = _mm512_load_ps(row + 80);
            __m512 s6 = _mm512_load_ps(row + 96);
            __m512 s7 = _mm512_load_ps(row + 112);

            // 2. Value prediction: v_hat = row * k_bar
            __m512 vk0 = _mm512_fmadd_ps(s0, k_vec[0], _mm512_mul_ps(s1, k_vec[1]));
            __m512 vk1 = _mm512_fmadd_ps(s2, k_vec[2], _mm512_mul_ps(s3, k_vec[3]));
            __m512 vk2 = _mm512_fmadd_ps(s4, k_vec[4], _mm512_mul_ps(s5, k_vec[5]));
            __m512 vk3 = _mm512_fmadd_ps(s6, k_vec[6], _mm512_mul_ps(s7, k_vec[7]));
            float v_hat_r = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(vk0, vk1), _mm512_add_ps(vk2, vk3)));

            // 3. Novelty delta error & fast weight update factor: e_r = v_proj[r] - v_hat_r
            float e_r = v_proj[r] - v_hat_r;
            float scale_e = beta * e_r;
            __m512 v_scale_e = _mm512_set1_ps(scale_e);

            // 4. In-place recurrence update: S_h = alpha * S_h + scale_e * k_bar
            s0 = _mm512_fmadd_ps(s0, v_alpha, _mm512_mul_ps(v_scale_e, k_vec[0]));
            s1 = _mm512_fmadd_ps(s1, v_alpha, _mm512_mul_ps(v_scale_e, k_vec[1]));
            s2 = _mm512_fmadd_ps(s2, v_alpha, _mm512_mul_ps(v_scale_e, k_vec[2]));
            s3 = _mm512_fmadd_ps(s3, v_alpha, _mm512_mul_ps(v_scale_e, k_vec[3]));
            s4 = _mm512_fmadd_ps(s4, v_alpha, _mm512_mul_ps(v_scale_e, k_vec[4]));
            s5 = _mm512_fmadd_ps(s5, v_alpha, _mm512_mul_ps(v_scale_e, k_vec[5]));
            s6 = _mm512_fmadd_ps(s6, v_alpha, _mm512_mul_ps(v_scale_e, k_vec[6]));
            s7 = _mm512_fmadd_ps(s7, v_alpha, _mm512_mul_ps(v_scale_e, k_vec[7]));

            _mm512_store_ps(row,       s0);
            _mm512_store_ps(row + 16,  s1);
            _mm512_store_ps(row + 32,  s2);
            _mm512_store_ps(row + 48,  s3);
            _mm512_store_ps(row + 64,  s4);
            _mm512_store_ps(row + 80,  s5);
            _mm512_store_ps(row + 96,  s6);
            _mm512_store_ps(row + 112, s7);

            // 5. Gated Readout: dot = updated_row * q_vec
            __m512 vx0 = _mm512_fmadd_ps(s0, q_vec[0], _mm512_mul_ps(s1, q_vec[1]));
            __m512 vx1 = _mm512_fmadd_ps(s2, q_vec[2], _mm512_mul_ps(s3, q_vec[3]));
            __m512 vx2 = _mm512_fmadd_ps(s4, q_vec[4], _mm512_mul_ps(s5, q_vec[5]));
            __m512 vx3 = _mm512_fmadd_ps(s6, q_vec[6], _mm512_mul_ps(s7, q_vec[7]));
            float dot = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(vx0, vx1), _mm512_add_ps(vx2, vx3)));

            out_h[r] = x_h[r] + dot * gate[r] * head_scale;
        }
    }

    // Apply TT16 low-rank continuous core residual if active/adapted
    const float* v_ptr = layer->tt_v;
    const float* u_ptr = layer->tt_u;
    if (v_ptr && u_ptr && hssa_is_layer_tt16_active(layer)) {
        const uint32_t R = HSSA_RANK;
        const uint32_t D = HSSA_HIDDEN_DIM;

        alignas(64) float z[HSSA_RANK];
        for (uint32_t r = 0; r < R; ++r) {
            const float* vr = v_ptr + (size_t)r * D;
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (uint32_t j = 0; j < D; j += 64) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(vr + j),      _mm512_loadu_ps(out_vec + j),      acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(vr + j + 16), _mm512_loadu_ps(out_vec + j + 16), acc1);
                acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(vr + j + 32), _mm512_loadu_ps(out_vec + j + 32), acc2);
                acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(vr + j + 48), _mm512_loadu_ps(out_vec + j + 48), acc3);
            }
            __m512 sum = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
            z[r] = _mm512_reduce_add_ps(sum);
        }

        __m512 vz[16];
        for (uint32_t r = 0; r < 16; ++r) {
            vz[r] = _mm512_set1_ps(z[r]);
        }

        for (uint32_t i = 0; i < D; i += 16) {
            __m512 y_acc = _mm512_loadu_ps(out_vec + i);
            for (uint32_t r = 0; r < 16; ++r) {
                __m512 u_vec = _mm512_loadu_ps(u_ptr + (size_t)r * D + i);
                y_acc = _mm512_fmadd_ps(u_vec, vz[r], y_acc);
            }
            _mm512_storeu_ps(out_vec + i, y_acc);
        }
    }

    // Layer-level RMSNorm to guarantee mathematical stability across deep layer stacks
    __m512 sq_acc0 = _mm512_setzero_ps();
    __m512 sq_acc1 = _mm512_setzero_ps();
    for (size_t i = 0; i < HSSA_HIDDEN_DIM; i += 32) {
        __m512 v0 = _mm512_loadu_ps(out_vec + i);
        __m512 v1 = _mm512_loadu_ps(out_vec + i + 16);
        sq_acc0 = _mm512_fmadd_ps(v0, v0, sq_acc0);
        sq_acc1 = _mm512_fmadd_ps(v1, v1, sq_acc1);
    }
    float sum_sq = _mm512_reduce_add_ps(_mm512_add_ps(sq_acc0, sq_acc1));
    float rms = std::sqrt(sum_sq / (float)HSSA_HIDDEN_DIM + 1e-6f);
    float inv_rms = 1.0f / rms;
    __m512 v_inv = _mm512_set1_ps(inv_rms);
    for (size_t i = 0; i < HSSA_HIDDEN_DIM; i += 16) {
        __m512 v = _mm512_loadu_ps(out_vec + i);
        _mm512_storeu_ps(out_vec + i, _mm512_mul_ps(v, v_inv));
    }
}

/*
 * =============================================================================
 * Tier 2: Dual-Rail SIMD Binarization & Continuous Residual Injection
 * =============================================================================
 */

/* Dual-rail SIMD binarization Psi: R^1024 -> {0, 1}^512 in < 20 ns.
 * Not static: declared in include/hssa.h so tests measure this exact kernel
 * instead of a private copy. */
void dual_rail_binarize(const float* in_vec, uint64_t* out_vsa) {
    for (size_t w = 0; w < 8; ++w) {
        uint64_t word = 0;
        for (size_t chunk = 0; chunk < 4; ++chunk) {
            size_t c = w * 4 + chunk;
            __m512 v_lo = _mm512_loadu_ps(&in_vec[c * 16]);
            __m512 v_hi = _mm512_loadu_ps(&in_vec[512 + c * 16]);
            __mmask16 m = _mm512_cmp_ps_mask(v_lo, v_hi, _CMP_GE_OQ);
            word |= ((uint64_t)(uint16_t)m) << (chunk * 16);
        }
        out_vsa[w] = word;
    }
}

/* Continuous float projection Phi and residual injection: h_mem = h_ssm + gamma * sim * Phi(k) */
static inline void continuous_residual_inject(const float* in_ssm, const uint64_t* best_vec, float similarity, float* out_mem) {
    const float gamma = 0.10f;
    float scale = gamma * similarity;
    __m512 v_pos = _mm512_set1_ps(scale);
    __m512 v_neg = _mm512_set1_ps(-scale);

    for (size_t c = 0; c < 32; ++c) {
        size_t word_idx = c / 4;
        size_t shift = (c % 4) * 16;
        uint16_t k = (uint16_t)((best_vec[word_idx] >> shift) & 0xFFFF);

        __m512 in_lo = _mm512_loadu_ps(&in_ssm[c * 16]);
        __m512 in_hi = _mm512_loadu_ps(&in_ssm[512 + c * 16]);

        __m512 proj_lo = _mm512_mask_blend_ps((__mmask16)k, v_neg, v_pos);
        __m512 proj_hi = _mm512_mask_blend_ps((__mmask16)k, v_pos, v_neg);

        _mm512_storeu_ps(&out_mem[c * 16], _mm512_add_ps(in_lo, proj_lo));
        _mm512_storeu_ps(&out_mem[512 + c * 16], _mm512_add_ps(in_hi, proj_hi));
    }
}

/*
 * =============================================================================
 * Tier 3: Modern Hopfield Attractor Energy Relaxation
 * =============================================================================
 */

/* Compute Modern Continuous Hopfield Energy: E(s) = -tau * log(sum exp(<xi_i, s>/tau)) + 0.5 * ||s||^2 */
static double compute_hopfield_energy(const HssaContext* ctx, const float* s) {
    const size_t M = ctx->num_attractor_patterns;
    const float tau = ctx->hopfield_tau;
    const float inv_tau = 1.0f / tau;

    alignas(64) float dots[HSSA_MAX_PATTERNS];
    float max_dot = -1e30f;

    for (size_t m = 0; m < M; ++m) {
        const float* pat = &ctx->attractor_patterns[m * HSSA_HIDDEN_DIM];
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        for (size_t j = 0; j < HSSA_HIDDEN_DIM; j += 32) {
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(pat + j),      _mm512_loadu_ps(s + j),      acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(pat + j + 16), _mm512_loadu_ps(s + j + 16), acc1);
        }
        float d = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1)) * inv_tau;
        dots[m] = d;
        if (d > max_dot) max_dot = d;
    }

    double sum_exp = 0.0;
    for (size_t m = 0; m < M; ++m) {
        sum_exp += std::exp(dots[m] - max_dot);
    }
    double lse = (double)max_dot + std::log(sum_exp);

    // Compute 0.5 * ||s||^2
    __m512 sq0 = _mm512_setzero_ps();
    __m512 sq1 = _mm512_setzero_ps();
    for (size_t j = 0; j < HSSA_HIDDEN_DIM; j += 32) {
        __m512 v0 = _mm512_loadu_ps(s + j);
        __m512 v1 = _mm512_loadu_ps(s + j + 16);
        sq0 = _mm512_fmadd_ps(v0, v0, sq0);
        sq1 = _mm512_fmadd_ps(v1, v1, sq1);
    }
    double norm2 = (double)_mm512_reduce_add_ps(_mm512_add_ps(sq0, sq1));

    return -(double)tau * lse + 0.5 * norm2;
}

/* Hopfield attractor relaxation contraction: s^(t+1) = sum_m softmax(<xi_m, s>/tau) * xi_m */
static void relax_hopfield_attractor(const HssaContext* ctx, float* s, size_t max_steps) {
    const size_t M = ctx->num_attractor_patterns;
    const float tau = ctx->hopfield_tau;
    const float inv_tau = 1.0f / tau;

    if (max_steps == 0 || M == 0) return;

    // The associative recall in step 5 of hssa_forward_token writes the recalled
    // memory into s immediately before this call. The iteration below maps s onto
    // the span of the 16 attractor patterns; without a residual connection that
    // replacement annihilated the recalled vector completely, so the VSA database
    // had no measurable effect on the output. Keep the pre-relaxation state and
    // blend it back once. This is a deliberate one-shot residual, not an energy
    // minimiser: E = -tau*lse + 0.5*||s||^2 is neither convex nor concave, so it
    // offers no monotonicity guarantee. The E2E measures the effect instead
    // (relative L2 difference 0.099980 for a recalled vector).
    alignas(64) float s_residual[HSSA_HIDDEN_DIM];
    std::memcpy(s_residual, s, HSSA_HIDDEN_DIM * sizeof(float));

    alignas(64) float dots[HSSA_MAX_PATTERNS];
    alignas(64) float weights[HSSA_MAX_PATTERNS];
    alignas(64) float s_next[HSSA_HIDDEN_DIM];

    for (size_t step = 0; step < max_steps; ++step) {
        float max_dot = -1e30f;
        for (size_t m = 0; m < M; ++m) {
            const float* pat = &ctx->attractor_patterns[m * HSSA_HIDDEN_DIM];
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            for (size_t j = 0; j < HSSA_HIDDEN_DIM; j += 32) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(pat + j),      _mm512_loadu_ps(s + j),      acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(pat + j + 16), _mm512_loadu_ps(s + j + 16), acc1);
            }
            float d = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1)) * inv_tau;
            dots[m] = d;
            if (d > max_dot) max_dot = d;
        }

        double sum_exp = 0.0;
        for (size_t m = 0; m < M; ++m) {
            float exp_val = std::exp(dots[m] - max_dot);
            weights[m] = exp_val;
            sum_exp += (double)exp_val;
        }
        float inv_sum = (sum_exp > 0.0) ? (float)(1.0 / sum_exp) : 0.0f;
        for (size_t m = 0; m < M; ++m) {
            weights[m] *= inv_sum;
        }

        // s_next = sum_m weights[m] * pat_m
        std::memset(s_next, 0, HSSA_HIDDEN_DIM * sizeof(float));
        for (size_t m = 0; m < M; ++m) {
            __m512 vw = _mm512_set1_ps(weights[m]);
            const float* pat = &ctx->attractor_patterns[m * HSSA_HIDDEN_DIM];
            for (size_t j = 0; j < HSSA_HIDDEN_DIM; j += 16) {
                __m512 cur = _mm512_loadu_ps(s_next + j);
                __m512 vp = _mm512_loadu_ps(pat + j);
                _mm512_storeu_ps(s_next + j, _mm512_fmadd_ps(vp, vw, cur));
            }
        }

        std::memcpy(s, s_next, HSSA_HIDDEN_DIM * sizeof(float));
    }

    // 0.5 * pre-relaxation residual + 0.5 * attractor state.
    for (size_t j = 0; j < HSSA_HIDDEN_DIM; j += 16) {
        __m512 a = _mm512_loadu_ps(s_residual + j);
        __m512 b = _mm512_loadu_ps(s + j);
        _mm512_storeu_ps(s + j, _mm512_mul_ps(_mm512_add_ps(a, b), _mm512_set1_ps(0.5f)));
    }
}

/*
 * =============================================================================
 * Forward Token Cycle
 * =============================================================================
 */

int hssa_forward_token(HssaContext* ctx, const float* in_vec, float* out_vec, HssaTelemetry* out_telem) {
    if (!ctx || !in_vec || !out_vec) return -1;

    // The scratchpads, Hopfield patterns and tt_u/tt_v cores are single-instance
    // mutable state. hssa_update_tt16() mutates the cores from another thread, so
    // every pass through this function holds the context lock.
    std::lock_guard<std::recursive_mutex> state_guard(*static_cast<std::recursive_mutex*>(ctx->state_lock));

    double t_start = get_qpc_time_us();

    // 1. Copy input vector into preallocated scratchpad
    std::memcpy(ctx->scratch_in, in_vec, HSSA_HIDDEN_DIM * sizeof(float));

    // 2. Multi-layer Gated DeltaNet state-space recurrence
    double t_ssm_0 = get_qpc_time_us();
    float* cur_in = ctx->scratch_in;
    float* cur_out = ctx->scratch_out;
    for (uint32_t l = 0; l < ctx->config.num_layers; ++l) {
        forward_deltanet_layer(&ctx->layers[l], cur_in, cur_out);
        std::swap(cur_in, cur_out);
    }
    if (cur_in != ctx->scratch_in) {
        std::memcpy(ctx->scratch_in, cur_in, HSSA_HIDDEN_DIM * sizeof(float));
    }
    double t_ssm_1 = get_qpc_time_us();

    // 3. Dual-rail SIMD binarization Psi: R^1024 -> {0, 1}^512
    LARGE_INTEGER t_bin_0, t_bin_1, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t_bin_0);

    // 50 repetitions only exist to average the per-call timer; without telemetry
    // the extra 49 binarizations are pure waste on the hot path.
    const int BIN_REPS = out_telem ? 50 : 1;
    for (int bi = 0; bi < BIN_REPS; ++bi) {
        dual_rail_binarize(ctx->scratch_in, ctx->scratch_vsa);
    }

    QueryPerformanceCounter(&t_bin_1);
    double binarize_ns = (double)(t_bin_1.QuadPart - t_bin_0.QuadPart) * 1e9 / ((double)freq.QuadPart * BIN_REPS);

    // 4. Episodic associative recall
    double t_vsa_0 = get_qpc_time_us();
    alignas(64) uint64_t best_vec[8];
    // Start at zero resonance, not 1.0f. Seeding with 1.0f asserted a perfect
    // match before any scan had run, so every caller of this function observed
    // "recalled with certainty" even on the paths that never queried anything.
    float similarity = 0.0f;

    VsaDbHandle* target_db = ctx->global_vsa_db;
    if (!target_db || vsa_db_get_vector_count(target_db) == 0) {
        for (uint32_t l = ctx->config.num_layers; l > 0; --l) {
            VsaDbHandle* lyr_db = ctx->layers[l - 1].vsa_handle;
            if (lyr_db) {
                if (!target_db) target_db = lyr_db;
                if (vsa_db_get_vector_count(lyr_db) > 0) {
                    target_db = lyr_db;
                    break;
                }
            }
        }
    }
    if (target_db && vsa_db_get_vector_count(target_db) > 0) {
        int found = vsa_db_query_vector(target_db, ctx->scratch_vsa, best_vec, &similarity);
        if (!found) {
            // Genuine miss. vsa_db_query_vector already left *out_sim at 0.0f here;
            // overwriting it with 1.0f fabricated a certainty the database does not
            // contain, and echoing the query back as best_vec made
            // continuous_residual_inject fold a projection of the query itself into
            // the memory state at full gamma weight.
            std::memset(best_vec, 0, sizeof(best_vec));
            similarity = 0.0f;
        }
    } else {
        // Empty database: there is nothing to recall, so there is nothing to inject.
        std::memset(best_vec, 0, sizeof(best_vec));
        similarity = 0.0f;
    }
    double t_vsa_1 = get_qpc_time_us();

    // 5. Continuous float projection Phi and residual injection
    continuous_residual_inject(ctx->scratch_in, best_vec, similarity, ctx->scratch_state);

    // 5b. Milestone M2: Pinned Zero-Copy DMA Bridge & sm_86 BSL-32T GEMV Offload.
    //
    // POLICY: this offload is opt-in and OFF by default. Apples-to-apples A/B measured
    // in one process on the target workstation (i7-11800H + RTX 3050 Laptop sm_86) via
    // `upgrade_lab hssa --cuda`: 28-layer CPU step 172.1 us vs the identical step
    // through this CUDA path 319.9 us = 1.86x SLOWER. In the heavier test_hssa_m2
    // configuration (full-plane matrix, split_k=8, separate allocations) the gap widens
    // to 165.3 us vs 2084.7 us = 12.6x. The cause is arithmetic, not tuning: one
    // 1024x1024 GEMV is only ~2.1 MFLOP, so kernel launch + cuCtxSynchronize + the
    // pageable DtoH copy dwarf the ~2 us of real GPU compute. Offload only pays off
    // above roughly M,N >= 8192 or a >= 16-vector batch, neither of which
    // hidden_dim = 1024 produces.
    //
    // The GEMV kernel itself matches a CPU reference (cosine 1.00000 / SQNR 137.6 dB
    // in tests/test_hssa_m2.cpp), which is a kernel-level check only. The matrix is
    // synthetic, so its output is not blended into the model state; this path exists
    // to measure the DMA/GEMV round trip, not to accelerate the network. Callers must
    // opt in via hssa_init_cuda() or hssa_enable_cuda(ctx, true); the CLI exposes
    // this as `upgrade_lab hssa --cuda`.
    double dma_fwd_us = 0.0;
    double gemv_us = 0.0;
    double dma_rev_us = 0.0;
    bool cuda_done = false;

    if (ctx->cuda_active && ctx->cuda_drv && ctx->cuda_kernels && ctx->gpu_matrix && ctx->hybrid_bridge) {
        CudaDrv* drv = (CudaDrv*)ctx->cuda_drv;
        PtqKernels* kernels = (PtqKernels*)ctx->cuda_kernels;
        PtqGpuMatrix* mat = (PtqGpuMatrix*)ctx->gpu_matrix;

        // Stage 1: Pinned Zero-Copy Host -> GPU DMA
        double t_dma_0 = get_qpc_time_us();
        std::memcpy(ctx->hybrid_bridge->hidden_cpu, ctx->scratch_state, HSSA_HIDDEN_DIM * sizeof(float));
        CUdeviceptr d_input = (CUdeviceptr)(uintptr_t)ctx->hybrid_bridge->d_hidden_gpu;
        if (!ctx->hybrid_bridge->is_zero_copy) {
            drv->cuMemcpyHtoD(d_input, ctx->hybrid_bridge->hidden_cpu, HSSA_HIDDEN_DIM * sizeof(float));
        }
        double t_dma_1 = get_qpc_time_us();
        dma_fwd_us = t_dma_1 - t_dma_0;

        // Stage 2: sm_86 Split-K BSL-32T GEMV Projection Kernel
        double t_gemv_0 = get_qpc_time_us();
        if (mat->d_tt_u && mat->d_z_tt) {
            ptq_tt_project_u_sm86(drv, kernels, mat->d_tt_u, d_input, mat->d_z_tt, HSSA_RANK, HSSA_HIDDEN_DIM, 0);
        }
        CUdeviceptr d_output = (CUdeviceptr)ctx->gpu_out_buf;
        int gemv_rc = ptq_gemv_sm86(
            drv, kernels, mat,
            d_input, d_output,
            (CUdeviceptr)ctx->gpu_partials, (CUdeviceptr)ctx->gpu_tile_cnt,
            4, /* active_planes: 4-bit */
            0, /* split_k: 0 = auto (ptq_pick_split_k picks a divisor of cols/128);
                  the old hard-coded 8 was only valid because 1024/128 = 8 */
            0  /* stream */
        );
        drv->cuCtxSynchronize();
        double t_gemv_1 = get_qpc_time_us();
        gemv_us = t_gemv_1 - t_gemv_0;

        // Stage 3: Reverse GPU -> Host DMA (measurement only)
        //
        // The matrix uploaded by hssa_init_cuda is synthetic (constant 0x55555555
        // planes, 0.02 scales, 0.001 TT cores), not trained model weights. Blending
        // its output into scratch_state -- the old 0.2/0.8 mix -- changed the model
        // output by an arbitrary amount that no reference could justify. The DMA
        // and GEMV still run and are timed; the bytes are copied back so the timing
        // covers the whole round trip, but the result is deliberately discarded.
        double t_rev_0 = get_qpc_time_us();
        alignas(64) float gpu_res[HSSA_HIDDEN_DIM];
        drv->cuMemcpyDtoH(gpu_res, d_output, HSSA_HIDDEN_DIM * sizeof(float));
        double t_rev_1 = get_qpc_time_us();
        dma_rev_us = t_rev_1 - t_rev_0;

        cuda_done = (gemv_rc == 0);
    }

    // 6. Modern Continuous Hopfield energy relaxation
    double t_hop_0 = get_qpc_time_us();
    double init_energy = compute_hopfield_energy(ctx, ctx->scratch_state);

    const size_t RELAX_STEPS = 5;
    relax_hopfield_attractor(ctx, ctx->scratch_state, RELAX_STEPS);

    double final_energy = compute_hopfield_energy(ctx, ctx->scratch_state);
    double t_hop_1 = get_qpc_time_us();

    // Copy relaxed attractor state to output vector
    std::memcpy(out_vec, ctx->scratch_state, HSSA_HIDDEN_DIM * sizeof(float));

    double t_end = get_qpc_time_us();

    // 7. Telemetry recording
    if (out_telem) {
        out_telem->total_forward_us = t_end - t_start;
        out_telem->ssm_recurrence_us = t_ssm_1 - t_ssm_0;
        out_telem->binarization_ns = binarize_ns;
        out_telem->vsa_recall_us = t_vsa_1 - t_vsa_0;
        out_telem->hopfield_relax_us = t_hop_1 - t_hop_0;
        out_telem->hopfield_initial_energy = init_energy;
        out_telem->hopfield_final_energy = final_energy;
        out_telem->hopfield_energy_delta = final_energy - init_energy;
        out_telem->hopfield_steps = (uint32_t)RELAX_STEPS;
        out_telem->state_memory_bytes = (size_t)ctx->config.num_layers * HSSA_DELTANET_STATE_SIZE * sizeof(float);
        out_telem->vsa_similarity = similarity;
        out_telem->rejected_contradictions = 0;

        out_telem->cuda_dma_fwd_us = dma_fwd_us;
        out_telem->cuda_gemv_us = gemv_us;
        out_telem->cuda_dma_rev_us = dma_rev_us;
        out_telem->cuda_executed = cuda_done;
    }

    return 0;
}

/*
 * =============================================================================
 * Attractor Simplex Mirror Descent Logit Projection Gate
 * =============================================================================
 */

int hssa_project_simplex(HssaContext* ctx, float* logits, size_t vocab_size) {
    if (!ctx || !logits || vocab_size == 0) return -1;
    std::lock_guard<std::recursive_mutex> state_guard(*static_cast<std::recursive_mutex*>(ctx->state_lock));

    // 1. Collect contradicted candidates into a local stack veto list (zero heap allocation
    //    on the decode hot path). A list (rather than a sentinel value written into logits)
    //    keeps the veto exact: a caller's legitimate logit of -1e9 cannot be mistaken for a
    //    contradiction, and no mask arithmetic is needed afterwards.
    alignas(64) uint32_t vetoed[HSSA_MAX_ASSERTIONS];
    uint32_t veto_count = 0;
    if (ctx->global_vsa_db && ctx->assertions && ctx->assertion_count > 0) {
        for (uint32_t a = 0; a < ctx->assertion_count; ++a) {
            const HssaTokenAssertion* asst = &ctx->assertions[a];
            if (asst->active && asst->token_id < vocab_size) {
                int conflict = vsa_db_check_conflict(ctx->global_vsa_db, asst->subject, asst->relation, asst->object);
                if (conflict == 1 && veto_count < HSSA_MAX_ASSERTIONS) {
                    vetoed[veto_count++] = asst->token_id;
                }
            }
        }
    }

    // 2. Stable log-sum-exp softmax over the finite logits only.
    float max_logit = -1e30f;
    for (size_t i = 0; i < vocab_size; ++i) {
        if (std::isfinite(logits[i]) && logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    double sum_exp = 0.0;
    for (size_t i = 0; i < vocab_size; ++i) {
        if (std::isfinite(logits[i])) {
            sum_exp += std::exp((double)logits[i] - (double)max_logit);
        }
    }

    if (max_logit > -1e29f && sum_exp > 0.0 && std::isfinite(sum_exp)) {
        const double inv_sum = 1.0 / sum_exp;
        for (size_t i = 0; i < vocab_size; ++i) {
            logits[i] = std::isfinite(logits[i])
                ? (float)(std::exp((double)logits[i] - (double)max_logit) * inv_sum)
                : 0.0f; // Non-finite candidates carry no probability mass
        }
    } else {
        // No finite logit at all: uniform over the vocabulary, veto applied below.
        const float uniform = 1.0f / (float)vocab_size;
        for (size_t i = 0; i < vocab_size; ++i) {
            logits[i] = uniform;
        }
    }

    // 3. The veto is absolute. Contradicted tokens are forced to exactly zero and never
    //    receive mass back, including in the fully-degenerate case. Hallucination
    //    prevention outranks the tidy "always sums to 1.0" property: if every candidate
    //    is contradicted the result is an all-zero vector and the returned count tells
    //    the caller why.
    for (uint32_t k = 0; k < veto_count; ++k) {
        logits[vetoed[k]] = 0.0f;
    }

    // 4. Renormalize over the survivors so the surviving candidates still sum to 1.0.
    double survivor_mass = 0.0;
    for (size_t i = 0; i < vocab_size; ++i) {
        survivor_mass += static_cast<double>(logits[i]);
    }
    if (survivor_mass > 0.0) {
        const double inv_survivors = 1.0 / survivor_mass;
        for (size_t i = 0; i < vocab_size; ++i) {
            logits[i] = (float)(static_cast<double>(logits[i]) * inv_survivors);
        }
    }

    return (int)veto_count;
}

/*
 * =============================================================================
 * Milestone M2: CUDA Acceleration APIs
 * =============================================================================
 */

int hssa_init_cuda(HssaContext* ctx) {
    if (!ctx) return -1;
    std::lock_guard<std::recursive_mutex> state_guard(*static_cast<std::recursive_mutex*>(ctx->state_lock));
    if (ctx->cuda_active) return 0; // Already active
    if (ctx->cuda_drv && ctx->cuda_ctx && ctx->cuda_kernels && ctx->gpu_matrix &&
        ctx->gpu_partials && ctx->gpu_tile_cnt && ctx->gpu_out_buf && ctx->hybrid_bridge) {
        ctx->cuda_active = true; // Re-enable previously initialized CUDA resources without leaking
        return 0;
    }

    // Check hardware & driver presence via hybrid bridge
    if (!bridge_is_cuda_available()) {
        return -2;
    }

    // 1. Allocate page-locked pinned HybridBuffer zero-copy host-device bridge
    if (!ctx->hybrid_bridge) {
        ctx->hybrid_bridge = bridge_alloc_buffer(HSSA_HIDDEN_DIM);
        if (!ctx->hybrid_bridge) return -3;
    }

    // 2. Initialize CUDA driver API and context
    CudaDrv* drv = (CudaDrv*)std::calloc(1, sizeof(CudaDrv));
    if (!drv) return -4;

    CUcontext cuda_ctx = nullptr;
    int rc = ptq_cuda_init(drv, &cuda_ctx, 0);
    if (rc != 0 || !cuda_ctx) {
        std::free(drv);
        return -5;
    }
    ctx->cuda_drv = drv;
    ctx->cuda_ctx = cuda_ctx;

    // 3. Load PTQ Kernels from sm_86 CUBIN
    PtqKernels* kernels = (PtqKernels*)std::calloc(1, sizeof(PtqKernels));
    if (!kernels) {
        ptq_cuda_close(drv, cuda_ctx);
        std::free(drv);
        ctx->cuda_drv = nullptr;
        ctx->cuda_ctx = nullptr;
        return -6;
    }

    rc = ptq_kernels_load(drv, kernels, "build\\kernel.cubin");
    if (rc != 0) {
        rc = ptq_kernels_load(drv, kernels, "kernel.cubin");
    }
    if (rc != 0) {
        std::free(kernels);
        ptq_cuda_close(drv, cuda_ctx);
        std::free(drv);
        ctx->cuda_drv = nullptr;
        ctx->cuda_ctx = nullptr;
        return -7;
    }
    ctx->cuda_kernels = kernels;

    // 4. Allocate and upload GPU matrix for BSL-32T projection GEMV (1024 x 1024, Rank 16)
    PtqGpuMatrix* mat = (PtqGpuMatrix*)std::calloc(1, sizeof(PtqGpuMatrix));
    if (!mat) {
        ptq_kernels_unload(drv, kernels);
        std::free(kernels);
        ptq_cuda_close(drv, cuda_ctx);
        std::free(drv);
        ctx->cuda_drv = nullptr;
        ctx->cuda_ctx = nullptr;
        ctx->cuda_kernels = nullptr;
        return -8;
    }

    const uint32_t M = HSSA_HIDDEN_DIM;
    const uint32_t N = HSSA_HIDDEN_DIM;
    const uint32_t R = HSSA_RANK;
    size_t plane_sz = (M / 32) * (N / 128) * 512;
    size_t scale_sz = (M / 32) * (N / 128) * 32 * sizeof(float);

    std::vector<uint32_t> init_plane(plane_sz / sizeof(uint32_t), 0x55555555);
    std::vector<float> init_scales(scale_sz / sizeof(float), 0.02f);
    std::vector<float> init_zeros(scale_sz / sizeof(float), 0.0f);
    std::vector<float> init_tt_u(R * N, 0.001f);
    std::vector<float> init_tt_v(R * M, 0.001f);
    const uint32_t* planes_ptr[4] = {
        init_plane.data(), init_plane.data(), init_plane.data(), init_plane.data()
    };

    rc = ptq_matrix_upload(drv, mat, M, N, R, planes_ptr, init_scales.data(), init_zeros.data(), init_tt_u.data(), init_tt_v.data());
    if (rc != 0) {
        std::free(mat);
        ptq_kernels_unload(drv, kernels);
        std::free(kernels);
        ptq_cuda_close(drv, cuda_ctx);
        std::free(drv);
        ctx->cuda_drv = nullptr;
        ctx->cuda_ctx = nullptr;
        ctx->cuda_kernels = nullptr;
        return -9;
    }
    ctx->gpu_matrix = mat;

    // 5. Allocate Split-K partials, tile counters, and output device buffer
    CUdeviceptr d_partials = 0, d_tile_cnt = 0, d_out_buf = 0;
    if (drv->cuMemAlloc(&d_partials, 8 * M * sizeof(float)) != 0 ||
        drv->cuMemAlloc(&d_tile_cnt, (M / 32) * sizeof(uint32_t)) != 0 ||
        drv->cuMemAlloc(&d_out_buf, M * sizeof(float)) != 0) {
        if (d_partials) drv->cuMemFree(d_partials);
        if (d_tile_cnt) drv->cuMemFree(d_tile_cnt);
        if (d_out_buf)  drv->cuMemFree(d_out_buf);
        ptq_matrix_free(drv, mat);
        std::free(mat);
        ptq_kernels_unload(drv, kernels);
        std::free(kernels);
        ptq_cuda_close(drv, cuda_ctx);
        std::free(drv);
        ctx->cuda_drv = nullptr;
        ctx->cuda_ctx = nullptr;
        ctx->cuda_kernels = nullptr;
        ctx->gpu_matrix = nullptr;
        return -10;
    }
    drv->cuMemsetD32(d_tile_cnt, 0, (M / 32));
    ctx->gpu_partials = (uint64_t)d_partials;
    ctx->gpu_tile_cnt = (uint64_t)d_tile_cnt;
    ctx->gpu_out_buf  = (uint64_t)d_out_buf;

    ctx->cuda_active = true;
    return 0;
}

int hssa_enable_cuda(HssaContext* ctx, bool enable) {
    if (!ctx) return -1;
    std::lock_guard<std::recursive_mutex> state_guard(*static_cast<std::recursive_mutex*>(ctx->state_lock));
    if (enable) {
        if (!ctx->cuda_active) {
            return hssa_init_cuda(ctx);
        }
        return 0;
    } else {
        ctx->cuda_active = false;
        return 0;
    }
}

bool hssa_is_cuda_active(const HssaContext* ctx) {
    return (ctx != nullptr) && ctx->cuda_active;
}

/*
 * =============================================================================
 * Online Plasticity & Mutation Helpers
 * =============================================================================
 */

int hssa_mutate_knowledge(HssaContext* ctx, const char* subj, const char* rel, const char* obj) {
    if (!ctx || !subj || !rel || !obj) return -1;
    if (!ctx->global_vsa_db) return -1;

    int rc = vsa_db_insert_fact(ctx->global_vsa_db, subj, rel, obj);
    if (rc != 0) {
        rc = vsa_db_mutate_fact(ctx->global_vsa_db, subj, rel, obj);
    }
    return rc;
}

/* Shared registry insertion for hssa_register_assertion / hssa_register_candidate_axiom.
 * The registry is the single source of truth for the Attractor Simplex gate, so both
 * public entry points funnel through this implementation. */
static int hssa_register_assertion_impl(HssaContext* ctx, uint32_t token_id, const char* subj, const char* rel, const char* obj) {
    if (!ctx || !subj || !rel || !obj) return -1;
    std::lock_guard<std::recursive_mutex> state_guard(*static_cast<std::recursive_mutex*>(ctx->state_lock));

    // Deduplicate in-place if the exact (token_id, subject, relation, object) assertion already exists
    for (uint32_t i = 0; i < ctx->assertion_count; ++i) {
        HssaTokenAssertion* existing = &ctx->assertions[i];
        if (existing->token_id == token_id &&
            std::strcmp(existing->subject, subj) == 0 &&
            std::strcmp(existing->relation, rel) == 0 &&
            std::strcmp(existing->object, obj) == 0) {
            existing->active = true;
            return 0;
        }
    }

    // Grow the registry on demand so an idle context carries no assertion footprint.
    if (ctx->assertion_count >= ctx->assertion_capacity) {
        if (ctx->assertion_capacity >= HSSA_MAX_ASSERTIONS) return -2; // registry full
        uint32_t new_cap = ctx->assertion_capacity ? ctx->assertion_capacity * 2u : 256u;
        if (new_cap > HSSA_MAX_ASSERTIONS) new_cap = HSSA_MAX_ASSERTIONS;
        HssaTokenAssertion* grown = (HssaTokenAssertion*)std::realloc(ctx->assertions, (size_t)new_cap * sizeof(HssaTokenAssertion));
        if (!grown) return -4;
        std::memset(grown + ctx->assertion_count, 0,
                    (size_t)(new_cap - ctx->assertion_count) * sizeof(HssaTokenAssertion));
        ctx->assertions = grown;
        ctx->assertion_capacity = new_cap;
    }

    HssaTokenAssertion* asst = &ctx->assertions[ctx->assertion_count++];
    asst->token_id = token_id;
    strncpy_s(asst->subject, sizeof(asst->subject), subj, _TRUNCATE);
    strncpy_s(asst->relation, sizeof(asst->relation), rel, _TRUNCATE);
    strncpy_s(asst->object, sizeof(asst->object), obj, _TRUNCATE);
    asst->active = true;

    // Bind the assertion's continuous prototype into the upper half of the Hopfield attractor bank
    // (preserving the lower half orthogonal sinusoidal basis for general manifold stability).
    if (ctx->attractor_patterns && ctx->num_attractor_patterns >= 2) {
        const size_t half_bank = ctx->num_attractor_patterns / 2;
        const size_t slot = half_bank + ((size_t)(ctx->assertion_count - 1) % (ctx->num_attractor_patterns - half_bank));
        float* pat = &ctx->attractor_patterns[slot * HSSA_HIDDEN_DIM];
        uint64_t h = 0x9E3779B97F4A7C15ULL ^ (uint64_t)token_id;
        for (const char* p = subj; *p; ++p) h = (h ^ (uint8_t)(*p)) * 0x100000001B3ULL;
        for (const char* p = rel;  *p; ++p) h = (h ^ (uint8_t)(*p)) * 0x100000001B3ULL;
        for (const char* p = obj;  *p; ++p) h = (h ^ (uint8_t)(*p)) * 0x100000001B3ULL;
        const float norm_factor = 1.0f / std::sqrt((float)HSSA_HIDDEN_DIM);
        for (size_t j = 0; j < HSSA_HIDDEN_DIM; ++j) {
            h = h * 6364136223846793005ULL + 1442695040888963407ULL;
            pat[j] = ((h >> 33) & 1ULL) ? norm_factor : -norm_factor;
        }
    }

    return 0;
}

int hssa_register_assertion(HssaContext* ctx, uint32_t token_id, const char* subj, const char* rel, const char* obj) {
    return hssa_register_assertion_impl(ctx, token_id, subj, rel, obj);
}

/*
 * =============================================================================
 * Milestone M3: Continuous Neuro-Symbolic Attractor Gate & Autoregressive Decoding
 * =============================================================================
 */

/* Upload or re-initialize the unembedding projection [vocab_size x 1024].
 * A NULL weights pointer installs the deterministic normalized projection. */
int hssa_set_unembedding_matrix(HssaContext* ctx, const float* unembed_weights, size_t vocab_size) {
    if (!ctx) return -1;
    if (vocab_size == 0 || vocab_size > HSSA_MAX_VOCAB) return -2;
    std::lock_guard<std::recursive_mutex> state_guard(*static_cast<std::recursive_mutex*>(ctx->state_lock));

    const size_t total_floats = vocab_size * HSSA_HIDDEN_DIM;

    float* new_weights = (float*)_aligned_malloc(total_floats * sizeof(float), HSSA_ALIGNMENT);
    float* new_logits = (float*)_aligned_malloc(vocab_size * sizeof(float), HSSA_ALIGNMENT);
    if (!new_weights || !new_logits) {
        if (new_weights) _aligned_free(new_weights);
        if (new_logits) _aligned_free(new_logits);
        return -3;
    }

    if (unembed_weights) {
        std::memcpy(new_weights, unembed_weights, total_floats * sizeof(float));
    } else {
        build_deterministic_unembedding(new_weights, vocab_size);
    }

    // Swap only after both allocations succeeded: the previous matrix stays intact on failure.
    if (ctx->unembed_weights) _aligned_free(ctx->unembed_weights);
    if (ctx->logits_buf) _aligned_free(ctx->logits_buf);
    ctx->unembed_weights = new_weights;
    ctx->logits_buf = new_logits;
    ctx->vocab_size = vocab_size;

    return 0;
}

int hssa_register_candidate_axiom(HssaContext* ctx, int32_t token_id, const char* subj, const char* rel, const char* obj) {
    if (!ctx || !subj || !rel || !obj) return -1;
    if (token_id < 0) return -5;
    return hssa_register_assertion_impl(ctx, (uint32_t)token_id, subj, rel, obj);
}

/* Deterministic token embedding: token_id -> bipolar vector in R^{1024}.
 * Reproducible across runs; no external dependency, no heap allocation. */
static inline void hssa_embed_token(int32_t token_id, float* out_vec) {
    uint64_t h = 0x9E3779B97F4A7C15ULL ^ (uint64_t)(uint32_t)token_id;
    const float scale = 1.0f / std::sqrt((float)HSSA_HIDDEN_DIM);
    for (size_t j = 0; j < HSSA_HIDDEN_DIM; ++j) {
        h = h * 6364136223846793005ULL + 1442695040888963407ULL;
        out_vec[j] = ((h >> 33) & 1ULL) ? scale : -scale;
    }
}

/* AVX-512 fused projection: logits[v] = dot(unembed_weights[v], hidden) */
static void hssa_compute_logits(const HssaContext* ctx, const float* hidden) {
    const size_t vocab = ctx->vocab_size;
    for (size_t v = 0; v < vocab; ++v) {
        const float* row = &ctx->unembed_weights[v * HSSA_HIDDEN_DIM];
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        for (size_t j = 0; j < HSSA_HIDDEN_DIM; j += 64) {
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(row + j),      _mm512_loadu_ps(hidden + j),      acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(row + j + 16), _mm512_loadu_ps(hidden + j + 16), acc1);
            acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(row + j + 32), _mm512_loadu_ps(hidden + j + 32), acc2);
            acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(row + j + 48), _mm512_loadu_ps(hidden + j + 48), acc3);
        }
        ctx->logits_buf[v] = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(acc0, acc1),
                                                               _mm512_add_ps(acc2, acc3)));
    }
}

/* Token selection: greedy argmax for temperature <= 1e-4, otherwise a deterministic
 * xorshift64 inverse-CDF draw over the probability simplex. No global RNG state. */
static inline int32_t hssa_select_token(const float* probs, size_t vocab, float temperature, uint64_t* rng_state) {
    if (temperature <= 1e-4f) {
        size_t best = 0;
        float best_p = probs[0];
        for (size_t v = 1; v < vocab; ++v) {
            if (probs[v] > best_p) {
                best_p = probs[v];
                best = v;
            }
        }
        return (int32_t)best;
    }

    uint64_t x = *rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *rng_state = x;

    const double u = (double)(x >> 11) * (1.0 / 9007199254740992.0); // [0, 1)
    double acc = 0.0;
    for (size_t v = 0; v < vocab; ++v) {
        acc += (double)probs[v];
        if (u < acc) {
            return (int32_t)v;
        }
    }
    return (int32_t)(vocab - 1);
}

int hssa_generate(
    HssaContext* ctx,
    const int32_t* prompt_tokens,
    size_t prompt_len,
    int32_t* out_tokens,
    size_t max_gen_len,
    float temperature,
    HssaGenStats* out_stats
) {
    if (!ctx || !out_tokens || max_gen_len == 0) return -1;
    if (!ctx->unembed_weights || !ctx->logits_buf || ctx->vocab_size == 0) return -6;
    if (prompt_len > 0 && !prompt_tokens) return -7;

    std::lock_guard<std::recursive_mutex> state_guard(*static_cast<std::recursive_mutex*>(ctx->state_lock));

    const size_t vocab = ctx->vocab_size;
    const double t_start_ms = get_qpc_time_us() / 1000.0;

    // Fresh decoding context: reset recurrent state so generation is reproducible.
    if (hssa_reset_state(ctx) != 0) return -8;

    alignas(64) float cur_vec[HSSA_HIDDEN_DIM];
    alignas(64) float next_vec[HSSA_HIDDEN_DIM];
    std::memset(cur_vec, 0, sizeof(cur_vec));

    // 1. Prefill: drive the prompt through the recurrent stack.
    for (size_t p = 0; p < prompt_len; ++p) {
        hssa_embed_token(prompt_tokens[p], cur_vec);
        if (hssa_forward_token(ctx, cur_vec, next_vec, nullptr) != 0) return -9;
        std::memcpy(cur_vec, next_vec, sizeof(cur_vec));
    }

    uint32_t contradictions_blocked = 0;
    uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

    // 2. Decode: zero-heap loop over preallocated buffers only.
    for (size_t t = 0; t < max_gen_len; ++t) {
        hssa_compute_logits(ctx, cur_vec);
        if (temperature > 1e-4f) {
            const float inv_temp = 1.0f / temperature;
            for (size_t v = 0; v < vocab; ++v) {
                ctx->logits_buf[v] *= inv_temp;
            }
        }
        const int rejected = hssa_project_simplex(ctx, ctx->logits_buf, vocab);
        if (rejected < 0) return -10;
        if (rejected > 0) contradictions_blocked += (uint32_t)rejected;

        const int32_t next_token = hssa_select_token(ctx->logits_buf, vocab, temperature, &rng_state);
        out_tokens[t] = next_token;

        hssa_embed_token(next_token, cur_vec);
        if (hssa_forward_token(ctx, cur_vec, next_vec, nullptr) != 0) return -11;
        std::memcpy(cur_vec, next_vec, sizeof(cur_vec));
    }

    if (out_stats) {
        const double elapsed_ms = (get_qpc_time_us() / 1000.0) - t_start_ms;
        out_stats->total_time_ms = elapsed_ms;
        out_stats->avg_tok_us = (elapsed_ms * 1000.0) / (double)max_gen_len;
        out_stats->tokens_generated = (uint32_t)max_gen_len;
        out_stats->contradictions_blocked = contradictions_blocked;
    }

    return 0;
}

int hssa_clear_assertions(HssaContext* ctx) {
    if (!ctx) return -1;
    std::lock_guard<std::recursive_mutex> state_guard(*static_cast<std::recursive_mutex*>(ctx->state_lock));
    ctx->assertion_count = 0;
    return 0;
}

int hssa_set_global_vsa_db(HssaContext* ctx, VsaDbHandle* db) {
    if (!ctx) return -1;
    std::lock_guard<std::recursive_mutex> state_guard(*static_cast<std::recursive_mutex*>(ctx->state_lock));
    if (ctx->owns_global_vsa_db && ctx->global_vsa_db && ctx->global_vsa_db != db) {
        vsa_db_destroy(ctx->global_vsa_db);
    }
    ctx->global_vsa_db = db;
    ctx->owns_global_vsa_db = false;
    return 0;
}

int hssa_set_layer_vsa_db(HssaContext* ctx, uint32_t layer_idx, VsaDbHandle* db) {
    if (!ctx || layer_idx >= ctx->config.num_layers) return -1;
    std::lock_guard<std::recursive_mutex> state_guard(*static_cast<std::recursive_mutex*>(ctx->state_lock));
    ctx->layers[layer_idx].vsa_handle = db;
    return 0;
}

int hssa_update_tt16(HssaContext* ctx, uint32_t layer_idx, const float* u_delta, const float* v_delta) {
    if (!ctx || !u_delta || !v_delta || layer_idx >= ctx->config.num_layers) return -1;

    // Serialize against hssa_forward_token(), which reads these same cores.
    std::lock_guard<std::recursive_mutex> state_guard(*static_cast<std::recursive_mutex*>(ctx->state_lock));

    HssaLayer* lyr = &ctx->layers[layer_idx];
    lyr->reserved[0] = (void*)(uintptr_t)1;
    static_assert((HSSA_TT16_CORE_SIZE % 64) == 0, "HSSA_TT16_CORE_SIZE must be a multiple of 64");
    const size_t sz = HSSA_TT16_CORE_SIZE;

    // 4-way AVX-512 unrolled vector addition
    for (size_t i = 0; i < sz; i += 64) {
        __m512 u0 = _mm512_loadu_ps(lyr->tt_u + i);
        __m512 u1 = _mm512_loadu_ps(lyr->tt_u + i + 16);
        __m512 u2 = _mm512_loadu_ps(lyr->tt_u + i + 32);
        __m512 u3 = _mm512_loadu_ps(lyr->tt_u + i + 48);

        __m512 du0 = _mm512_loadu_ps(u_delta + i);
        __m512 du1 = _mm512_loadu_ps(u_delta + i + 16);
        __m512 du2 = _mm512_loadu_ps(u_delta + i + 32);
        __m512 du3 = _mm512_loadu_ps(u_delta + i + 48);

        _mm512_storeu_ps(lyr->tt_u + i,      _mm512_add_ps(u0, du0));
        _mm512_storeu_ps(lyr->tt_u + i + 16, _mm512_add_ps(u1, du1));
        _mm512_storeu_ps(lyr->tt_u + i + 32, _mm512_add_ps(u2, du2));
        _mm512_storeu_ps(lyr->tt_u + i + 48, _mm512_add_ps(u3, du3));

        __m512 v0 = _mm512_loadu_ps(lyr->tt_v + i);
        __m512 v1 = _mm512_loadu_ps(lyr->tt_v + i + 16);
        __m512 v2 = _mm512_loadu_ps(lyr->tt_v + i + 32);
        __m512 v3 = _mm512_loadu_ps(lyr->tt_v + i + 48);

        __m512 dv0 = _mm512_loadu_ps(v_delta + i);
        __m512 dv1 = _mm512_loadu_ps(v_delta + i + 16);
        __m512 dv2 = _mm512_loadu_ps(v_delta + i + 32);
        __m512 dv3 = _mm512_loadu_ps(v_delta + i + 48);

        _mm512_storeu_ps(lyr->tt_v + i,      _mm512_add_ps(v0, dv0));
        _mm512_storeu_ps(lyr->tt_v + i + 16, _mm512_add_ps(v1, dv1));
        _mm512_storeu_ps(lyr->tt_v + i + 32, _mm512_add_ps(v2, dv2));
        _mm512_storeu_ps(lyr->tt_v + i + 48, _mm512_add_ps(v3, dv3));
    }

    // The GPU bridge owns ONE synthetic 1024x1024 matrix for the whole context and
    // is a measurement-only microbenchmark: its GEMV output is never blended into
    // scratch_state (see hssa_forward_token). There is no 28-layer device mirror.
    // Layer 0's cores are mirrored so the benchmark's fused TT epilogue is fed real
    // adapted data; updates to other layers stay host-side only.
    if (layer_idx == 0 && ctx->cuda_active && ctx->cuda_drv && ctx->gpu_matrix) {
        CudaDrv* drv = (CudaDrv*)ctx->cuda_drv;
        PtqGpuMatrix* mat = (PtqGpuMatrix*)ctx->gpu_matrix;
        if (mat->d_tt_u) {
            drv->cuMemcpyHtoD(mat->d_tt_u, lyr->tt_u, HSSA_TT16_CORE_SIZE * sizeof(float));
        }
        if (mat->d_tt_v) {
            drv->cuMemcpyHtoD(mat->d_tt_v, lyr->tt_v, HSSA_TT16_CORE_SIZE * sizeof(float));
        }
    }

    return 0;
}

HssaConfig hssa_get_config(const HssaContext* ctx) {
    if (!ctx) {
        HssaConfig empty = {0};
        return empty;
    }
    return ctx->config;
}

HssaLayer* hssa_get_layer(HssaContext* ctx, uint32_t layer_idx) {
    if (!ctx || layer_idx >= ctx->config.num_layers) return nullptr;
    return &ctx->layers[layer_idx];
}

size_t hssa_get_state_memory_bytes(const HssaContext* ctx) {
    if (!ctx) return 0;
    return (size_t)ctx->config.num_layers * HSSA_DELTANET_STATE_SIZE * sizeof(float);
}

/*
 * =============================================================================
 * Adapted Continuous Representation Spaces & Deep Interpolation (AVX-512)
 * =============================================================================
 */

int hssa_interpolate_continuous_states(
    const HssaContext* ctx,
    const float* state_a,
    const float* state_b,
    float alpha,
    float* out_state
) {
    if (!state_a || !state_b || !out_state) return -1;
    const float t = (std::max)(0.0f, (std::min)(1.0f, alpha));

    __m512 acc_aa = _mm512_setzero_ps();
    __m512 acc_bb = _mm512_setzero_ps();
    __m512 acc_ab = _mm512_setzero_ps();

    for (size_t i = 0; i < HSSA_HIDDEN_DIM; i += 16) {
        __m512 va = _mm512_loadu_ps(state_a + i);
        __m512 vb = _mm512_loadu_ps(state_b + i);
        acc_aa = _mm512_fmadd_ps(va, va, acc_aa);
        acc_bb = _mm512_fmadd_ps(vb, vb, acc_bb);
        acc_ab = _mm512_fmadd_ps(va, vb, acc_ab);
    }

    const float norm_a = std::sqrt(_mm512_reduce_add_ps(acc_aa));
    const float norm_b = std::sqrt(_mm512_reduce_add_ps(acc_bb));
    const float dot_ab = _mm512_reduce_add_ps(acc_ab);

    float w_a = 1.0f - t;
    float w_b = t;
    const float target_norm = (1.0f - t) * norm_a + t * norm_b;
    bool use_antipodal = false;
    bool rescale_to_target = false;

    if (norm_a > 1e-6f && norm_b > 1e-6f) {
        float cos_theta = dot_ab / (norm_a * norm_b);
        cos_theta = (std::max)(-1.0f, (std::min)(1.0f, cos_theta));
        if (cos_theta < 0.9995f && cos_theta > -0.9995f) {
            float omega = std::acos(cos_theta);
            float sin_omega = std::sin(omega);
            if (sin_omega > 1e-6f) {
                w_a = (std::sin((1.0f - t) * omega) / sin_omega) * (target_norm / norm_a);
                w_b = (std::sin(t * omega) / sin_omega) * (target_norm / norm_b);
            }
        } else if (cos_theta <= -0.9995f) {
            use_antipodal = true;
        } else {
            rescale_to_target = true;
        }
    }

    if (use_antipodal) {
        size_t min_idx = 0;
        float min_abs = std::fabs(state_a[0]);
        for (size_t i = 1; i < HSSA_HIDDEN_DIM; ++i) {
            float a_abs = std::fabs(state_a[i]);
            if (a_abs < min_abs) {
                min_abs = a_abs;
                min_idx = i;
            }
        }
        const float inv_na = 1.0f / norm_a;
        const float neg_proj_inv_na = -(state_a[min_idx] * inv_na) * inv_na;
        alignas(64) float u_orth[HSSA_HIDDEN_DIM];
        __m512 v_scale_proj = _mm512_set1_ps(neg_proj_inv_na);
        for (size_t i = 0; i < HSSA_HIDDEN_DIM; i += 16) {
            __m512 va = _mm512_loadu_ps(state_a + i);
            _mm512_store_ps(u_orth + i, _mm512_mul_ps(va, v_scale_proj));
        }
        u_orth[min_idx] += 1.0f;

        __m512 v_orth_sq = _mm512_setzero_ps();
        for (size_t i = 0; i < HSSA_HIDDEN_DIM; i += 16) {
            __m512 vo = _mm512_load_ps(u_orth + i);
            v_orth_sq = _mm512_fmadd_ps(vo, vo, v_orth_sq);
        }
        const float orth_nsq = _mm512_reduce_add_ps(v_orth_sq);
        const float inv_orth = (orth_nsq > 1e-12f) ? (1.0f / std::sqrt(orth_nsq)) : 1.0f;
        const float theta = t * 3.14159265358979323846f;
        const float scale_a = std::cos(theta) * (target_norm * inv_na);
        const float scale_o = std::sin(theta) * (target_norm * inv_orth);
        __m512 vs_a = _mm512_set1_ps(scale_a);
        __m512 vs_o = _mm512_set1_ps(scale_o);
        for (size_t i = 0; i < HSSA_HIDDEN_DIM; i += 16) {
            __m512 va = _mm512_loadu_ps(state_a + i);
            __m512 vo = _mm512_load_ps(u_orth + i);
            _mm512_storeu_ps(out_state + i, _mm512_fmadd_ps(va, vs_a, _mm512_mul_ps(vo, vs_o)));
        }
    } else {
        __m512 vw_a = _mm512_set1_ps(w_a);
        __m512 vw_b = _mm512_set1_ps(w_b);
        __m512 acc_out = _mm512_setzero_ps();
        for (size_t i = 0; i < HSSA_HIDDEN_DIM; i += 16) {
            __m512 va = _mm512_loadu_ps(state_a + i);
            __m512 vb = _mm512_loadu_ps(state_b + i);
            __m512 vi = _mm512_fmadd_ps(va, vw_a, _mm512_mul_ps(vb, vw_b));
            _mm512_storeu_ps(out_state + i, vi);
            acc_out = _mm512_fmadd_ps(vi, vi, acc_out);
        }
        if (rescale_to_target) {
            float out_norm = std::sqrt(_mm512_reduce_add_ps(acc_out));
            if (out_norm > 1e-6f) {
                __m512 v_scale = _mm512_set1_ps(target_norm / out_norm);
                for (size_t i = 0; i < HSSA_HIDDEN_DIM; i += 16) {
                    _mm512_storeu_ps(out_state + i, _mm512_mul_ps(_mm512_loadu_ps(out_state + i), v_scale));
                }
            }
        }
    }

    // At interior points on the geodesic (0 < t < 1), gently project toward the Hopfield attractor manifold
    if (ctx && ctx->attractor_patterns && ctx->num_attractor_patterns > 0 && t > 1e-4f && t < 0.9999f) {
        alignas(64) float hop_vec[HSSA_HIDDEN_DIM];
        std::memcpy(hop_vec, out_state, HSSA_HIDDEN_DIM * sizeof(float));
        relax_hopfield_attractor(ctx, hop_vec, 2);
        const float pull = 0.15f * (4.0f * t * (1.0f - t));
        __m512 v_keep = _mm512_set1_ps(1.0f - pull);
        __m512 v_pull = _mm512_set1_ps(pull);
        for (size_t i = 0; i < HSSA_HIDDEN_DIM; i += 16) {
            __m512 v_cur = _mm512_loadu_ps(out_state + i);
            __m512 v_hop = _mm512_load_ps(hop_vec + i);
            _mm512_storeu_ps(out_state + i, _mm512_fmadd_ps(v_cur, v_keep, _mm512_mul_ps(v_hop, v_pull)));
        }
    }

    return 0;
}

namespace {
struct HarmonicBasis1024 {
    alignas(64) float cos_j[HSSA_HIDDEN_DIM];
    alignas(64) float sin_j[HSSA_HIDDEN_DIM];
    HarmonicBasis1024() {
        const float step = 6.28318530717958647692f / static_cast<float>(HSSA_HIDDEN_DIM);
        for (size_t j = 0; j < HSSA_HIDDEN_DIM; ++j) {
            const float angle = static_cast<float>(j) * step;
            cos_j[j] = std::cos(angle);
            sin_j[j] = std::sin(angle);
        }
    }
};
} // namespace

int hssa_encode_continuous_manifold(
    HssaContext* ctx,
    const int32_t* tokens,
    size_t num_tokens,
    float* out_manifold_vec
) {
    if (!ctx || !tokens || num_tokens == 0 || !out_manifold_vec) return -1;
    std::lock_guard<std::recursive_mutex> state_guard(*static_cast<std::recursive_mutex*>(ctx->state_lock));

    static const HarmonicBasis1024 kBasis;

    alignas(64) float tok_vec[HSSA_HIDDEN_DIM];
    alignas(64) float step_out[HSSA_HIDDEN_DIM];
    alignas(64) float accum[HSSA_HIDDEN_DIM];
    std::memset(accum, 0, sizeof(accum));

    const float inv_num_tokens = 0.5f / static_cast<float>(num_tokens);
    for (size_t idx = 0; idx < num_tokens; ++idx) {
        hssa_embed_token(tokens[idx], tok_vec);
        // AVX-512 harmonic phase encoding using angle-addition identity:
        // 0.25 * cos(phi + angle_j) = (0.25 * cos(phi)) * cos(angle_j) + (-0.25 * sin(phi)) * sin(angle_j)
        const float pos_phase = static_cast<float>(idx + 1) * 0.125f;
        const __m512 v_cos_p = _mm512_set1_ps(0.25f * std::cos(pos_phase));
        const __m512 v_neg_sin_p = _mm512_set1_ps(-0.25f * std::sin(pos_phase));
        for (size_t j = 0; j < HSSA_HIDDEN_DIM; j += 16) {
            __m512 vt = _mm512_load_ps(tok_vec + j);
            __m512 vc = _mm512_load_ps(kBasis.cos_j + j);
            __m512 vs = _mm512_load_ps(kBasis.sin_j + j);
            vt = _mm512_fmadd_ps(v_cos_p, vc, vt);
            vt = _mm512_fmadd_ps(v_neg_sin_p, vs, vt);
            _mm512_store_ps(tok_vec + j, vt);
        }
        if (hssa_forward_deep_interpolated(ctx, tok_vec, step_out, nullptr) != 0) return -2;

        const float w = 1.0f + static_cast<float>(idx + 1) * inv_num_tokens;
        __m512 vw = _mm512_set1_ps(w);
        for (size_t j = 0; j < HSSA_HIDDEN_DIM; j += 16) {
            __m512 va = _mm512_load_ps(accum + j);
            __m512 vs = _mm512_load_ps(step_out + j);
            _mm512_store_ps(accum + j, _mm512_fmadd_ps(vs, vw, va));
        }
    }

    // L2 normalize onto S^{1023}
    __m512 sq = _mm512_setzero_ps();
    for (size_t j = 0; j < HSSA_HIDDEN_DIM; j += 16) {
        __m512 va = _mm512_load_ps(accum + j);
        sq = _mm512_fmadd_ps(va, va, sq);
    }
    float norm = std::sqrt(_mm512_reduce_add_ps(sq));
    float inv_norm = (norm > 1e-7f) ? (1.0f / norm) : 1.0f;
    __m512 v_inv = _mm512_set1_ps(inv_norm);
    for (size_t j = 0; j < HSSA_HIDDEN_DIM; j += 16) {
        _mm512_storeu_ps(out_manifold_vec + j, _mm512_mul_ps(_mm512_load_ps(accum + j), v_inv));
    }
    return 0;
}

int hssa_forward_deep_interpolated(
    HssaContext* ctx,
    const float* in_vec,
    float* out_vec,
    HssaTelemetry* out_telem
) {
    if (!ctx || !in_vec || !out_vec) return -1;
    std::lock_guard<std::recursive_mutex> state_guard(*static_cast<std::recursive_mutex*>(ctx->state_lock));

    double t_start = get_qpc_time_us();
    std::memcpy(ctx->scratch_in, in_vec, HSSA_HIDDEN_DIM * sizeof(float));

    const uint32_t L = ctx->config.num_layers;
    const uint32_t shallow_idx = (L >= 4) ? (L / 4 - 1) : 0;
    const uint32_t mid_idx     = (L >= 2) ? (L / 2 - 1) : 0;

    alignas(64) float local_snaps[2 * HSSA_HIDDEN_DIM];
    float* snap_buf = ctx->reserved[0] ? static_cast<float*>(ctx->reserved[0]) : local_snaps;
    float* snap_shallow = snap_buf;
    float* snap_mid     = snap_buf + HSSA_HIDDEN_DIM;

    double t_ssm_0 = get_qpc_time_us();
    float* cur_in = ctx->scratch_in;
    float* cur_out = ctx->scratch_out;
    for (uint32_t l = 0; l < L; ++l) {
        forward_deltanet_layer(&ctx->layers[l], cur_in, cur_out);
        if (l == shallow_idx) {
            std::memcpy(snap_shallow, cur_out, HSSA_HIDDEN_DIM * sizeof(float));
        }
        if (l == mid_idx) {
            std::memcpy(snap_mid, cur_out, HSSA_HIDDEN_DIM * sizeof(float));
        }
        std::swap(cur_in, cur_out);
    }
    const float* snap_deep = cur_in;

    // Compute scale-invariant adaptive cosine gates between shallow, mid, and deep representations
    __m512 dot_sd = _mm512_setzero_ps();
    __m512 dot_md = _mm512_setzero_ps();
    __m512 sq_s   = _mm512_setzero_ps();
    __m512 sq_m   = _mm512_setzero_ps();
    __m512 sq_d   = _mm512_setzero_ps();
    for (size_t i = 0; i < HSSA_HIDDEN_DIM; i += 16) {
        __m512 vs = _mm512_load_ps(snap_shallow + i);
        __m512 vm = _mm512_load_ps(snap_mid + i);
        __m512 vd = _mm512_load_ps(snap_deep + i);
        dot_sd = _mm512_fmadd_ps(vs, vd, dot_sd);
        dot_md = _mm512_fmadd_ps(vm, vd, dot_md);
        sq_s   = _mm512_fmadd_ps(vs, vs, sq_s);
        sq_m   = _mm512_fmadd_ps(vm, vm, sq_m);
        sq_d   = _mm512_fmadd_ps(vd, vd, sq_d);
    }
    const float norm_s = std::sqrt(_mm512_reduce_add_ps(sq_s) + 1e-12f);
    const float norm_m = std::sqrt(_mm512_reduce_add_ps(sq_m) + 1e-12f);
    const float norm_d = std::sqrt(_mm512_reduce_add_ps(sq_d) + 1e-12f);
    const float cos_sd = _mm512_reduce_add_ps(dot_sd) / (norm_s * norm_d);
    const float cos_md = _mm512_reduce_add_ps(dot_md) / (norm_m * norm_d);

    float g_shallow = fast_sigmoid(cos_sd) * 0.20f;
    float g_mid     = fast_sigmoid(cos_md) * 0.30f;
    float g_deep    = 1.0f - (g_shallow + g_mid);

    __m512 vw_s = _mm512_set1_ps(g_shallow);
    __m512 vw_m = _mm512_set1_ps(g_mid);
    __m512 vw_d = _mm512_set1_ps(g_deep);
    __m512 sq_acc = _mm512_setzero_ps();

    for (size_t i = 0; i < HSSA_HIDDEN_DIM; i += 16) {
        __m512 vs = _mm512_load_ps(snap_shallow + i);
        __m512 vm = _mm512_load_ps(snap_mid + i);
        __m512 vd = _mm512_load_ps(snap_deep + i);
        __m512 v_blend = _mm512_fmadd_ps(vs, vw_s, _mm512_fmadd_ps(vm, vw_m, _mm512_mul_ps(vd, vw_d)));
        _mm512_store_ps(ctx->scratch_in + i, v_blend);
        sq_acc = _mm512_fmadd_ps(v_blend, v_blend, sq_acc);
    }
    float rms = std::sqrt(_mm512_reduce_add_ps(sq_acc) / (float)HSSA_HIDDEN_DIM + 1e-6f);
    __m512 v_inv_rms = _mm512_set1_ps(1.0f / rms);
    for (size_t i = 0; i < HSSA_HIDDEN_DIM; i += 16) {
        _mm512_store_ps(ctx->scratch_in + i, _mm512_mul_ps(_mm512_load_ps(ctx->scratch_in + i), v_inv_rms));
    }
    double t_ssm_1 = get_qpc_time_us();

    LARGE_INTEGER t_bin_0, t_bin_1, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t_bin_0);
    dual_rail_binarize(ctx->scratch_in, ctx->scratch_vsa);
    QueryPerformanceCounter(&t_bin_1);
    double binarize_ns = (double)(t_bin_1.QuadPart - t_bin_0.QuadPart) * 1e9 / (double)freq.QuadPart;

    double t_vsa_0 = get_qpc_time_us();
    alignas(64) uint64_t best_vec[8];
    float similarity = 0.0f;
    VsaDbHandle* target_db = ctx->global_vsa_db;
    if (target_db && vsa_db_get_vector_count(target_db) > 0) {
        if (!vsa_db_query_vector(target_db, ctx->scratch_vsa, best_vec, &similarity)) {
            std::memset(best_vec, 0, sizeof(best_vec));
            similarity = 0.0f;
        }
    } else {
        std::memset(best_vec, 0, sizeof(best_vec));
        similarity = 0.0f;
    }
    double t_vsa_1 = get_qpc_time_us();

    continuous_residual_inject(ctx->scratch_in, best_vec, similarity, ctx->scratch_state);

    double t_hop_0 = get_qpc_time_us();
    double init_energy = compute_hopfield_energy(ctx, ctx->scratch_state);
    relax_hopfield_attractor(ctx, ctx->scratch_state, 5);
    double final_energy = compute_hopfield_energy(ctx, ctx->scratch_state);
    double t_hop_1 = get_qpc_time_us();

    std::memcpy(out_vec, ctx->scratch_state, HSSA_HIDDEN_DIM * sizeof(float));
    double t_end = get_qpc_time_us();

    if (out_telem) {
        out_telem->total_forward_us = t_end - t_start;
        out_telem->ssm_recurrence_us = t_ssm_1 - t_ssm_0;
        out_telem->binarization_ns = binarize_ns;
        out_telem->vsa_recall_us = t_vsa_1 - t_vsa_0;
        out_telem->hopfield_relax_us = t_hop_1 - t_hop_0;
        out_telem->hopfield_initial_energy = init_energy;
        out_telem->hopfield_final_energy = final_energy;
        out_telem->hopfield_energy_delta = final_energy - init_energy;
        out_telem->hopfield_steps = 5;
        out_telem->state_memory_bytes = (size_t)ctx->config.num_layers * HSSA_DELTANET_STATE_SIZE * sizeof(float);
        out_telem->vsa_similarity = similarity;
        out_telem->rejected_contradictions = 0;
        out_telem->cuda_dma_fwd_us = 0.0;
        out_telem->cuda_gemv_us = 0.0;
        out_telem->cuda_dma_rev_us = 0.0;
        out_telem->cuda_executed = false;
    }
    return 0;
}

namespace {

struct HssaCheckpointHeader {
    char     magic[8];       // "HSSAWGT1"
    uint32_t version;        // 1
    uint32_t num_layers;
    uint32_t hidden_dim;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t rank;
    uint64_t num_attractors;
    uint64_t vocab_size;
    uint64_t payload_fnv1a;
};

static uint64_t fnv1a64_update(uint64_t h, const void* data, size_t len) noexcept {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

} // namespace

int hssa_save_weights(const HssaContext* ctx, const char* filepath) {
    if (!ctx || !filepath || filepath[0] == '\0' || !ctx->layers) return -1;
    std::lock_guard<std::recursive_mutex> state_guard(*static_cast<std::recursive_mutex*>(ctx->state_lock));

    FILE* fp = std::fopen(filepath, "wb");
    if (!fp) return -2;

    uint64_t csum = 14695981039346656037ULL;
    const size_t tt_bytes = static_cast<size_t>(HSSA_TT16_CORE_SIZE) * sizeof(float);
    const size_t dn_bytes = static_cast<size_t>(HSSA_DELTANET_STATE_SIZE) * sizeof(float);
    const size_t bw_bytes = sizeof(HssaBaseLayerWeights);
    for (uint32_t l = 0; l < ctx->config.num_layers; ++l) {
        csum = fnv1a64_update(csum, ctx->layers[l].tt_u, tt_bytes);
        csum = fnv1a64_update(csum, ctx->layers[l].tt_v, tt_bytes);
        csum = fnv1a64_update(csum, ctx->layers[l].deltanet_state, dn_bytes);
        if (ctx->layers[l].reserved[1]) {
            csum = fnv1a64_update(csum, ctx->layers[l].reserved[1], bw_bytes);
        }
    }
    const size_t att_bytes = ctx->num_attractor_patterns * HSSA_HIDDEN_DIM * sizeof(float);
    if (att_bytes > 0 && ctx->attractor_patterns) {
        csum = fnv1a64_update(csum, ctx->attractor_patterns, att_bytes);
    }
    const size_t unemb_bytes = (ctx->unembed_weights && ctx->vocab_size > 0)
        ? (ctx->vocab_size * HSSA_HIDDEN_DIM * sizeof(float))
        : 0;
    if (unemb_bytes > 0) {
        csum = fnv1a64_update(csum, ctx->unembed_weights, unemb_bytes);
    }

    HssaCheckpointHeader hdr{};
    std::memcpy(hdr.magic, "HSSAWGT1", 8);
    hdr.version        = 2U;
    hdr.num_layers     = ctx->config.num_layers;
    hdr.hidden_dim     = ctx->config.hidden_dim;
    hdr.heads          = ctx->config.heads;
    hdr.head_dim       = ctx->config.head_dim;
    hdr.rank           = ctx->config.rank;
    hdr.num_attractors = static_cast<uint64_t>(ctx->num_attractor_patterns);
    hdr.vocab_size     = (unemb_bytes > 0) ? static_cast<uint64_t>(ctx->vocab_size) : 0ULL;
    hdr.payload_fnv1a  = csum;

    bool ok = (std::fwrite(&hdr, sizeof(hdr), 1, fp) == 1);
    for (uint32_t l = 0; ok && l < ctx->config.num_layers; ++l) {
        ok = ok && (std::fwrite(ctx->layers[l].tt_u, 1, tt_bytes, fp) == tt_bytes);
        ok = ok && (std::fwrite(ctx->layers[l].tt_v, 1, tt_bytes, fp) == tt_bytes);
        ok = ok && (std::fwrite(ctx->layers[l].deltanet_state, 1, dn_bytes, fp) == dn_bytes);
        if (ctx->layers[l].reserved[1]) {
            ok = ok && (std::fwrite(ctx->layers[l].reserved[1], 1, bw_bytes, fp) == bw_bytes);
        }
    }
    if (ok && att_bytes > 0 && ctx->attractor_patterns) {
        ok = (std::fwrite(ctx->attractor_patterns, 1, att_bytes, fp) == att_bytes);
    }
    if (ok && unemb_bytes > 0) {
        ok = (std::fwrite(ctx->unembed_weights, 1, unemb_bytes, fp) == unemb_bytes);
    }
    std::fclose(fp);
    return ok ? 0 : -3;
}

int hssa_load_weights(HssaContext* ctx, const char* filepath) {
    if (!ctx || !filepath || filepath[0] == '\0' || !ctx->layers) return -1;
    std::lock_guard<std::recursive_mutex> state_guard(*static_cast<std::recursive_mutex*>(ctx->state_lock));

    FILE* fp = std::fopen(filepath, "rb");
    if (!fp) return -2;

    HssaCheckpointHeader hdr{};
    if (std::fread(&hdr, sizeof(hdr), 1, fp) != 1) {
        std::fclose(fp);
        return -3;
    }
    if (std::memcmp(hdr.magic, "HSSAWGT1", 8) != 0 ||
        (hdr.version != 1U && hdr.version != 2U) ||
        hdr.num_layers != ctx->config.num_layers ||
        hdr.hidden_dim != ctx->config.hidden_dim ||
        hdr.heads != ctx->config.heads ||
        hdr.head_dim != ctx->config.head_dim ||
        hdr.rank != ctx->config.rank ||
        hdr.num_attractors > HSSA_MAX_PATTERNS ||
        hdr.vocab_size > HSSA_MAX_VOCAB) {
        std::fclose(fp);
        return -4;
    }

    const size_t tt_bytes = static_cast<size_t>(HSSA_TT16_CORE_SIZE) * sizeof(float);
    const size_t dn_bytes = static_cast<size_t>(HSSA_DELTANET_STATE_SIZE) * sizeof(float);
    const size_t bw_bytes = sizeof(HssaBaseLayerWeights);
    const size_t L = ctx->config.num_layers;
    std::vector<float> tmp_u(L * HSSA_TT16_CORE_SIZE);
    std::vector<float> tmp_v(L * HSSA_TT16_CORE_SIZE);
    std::vector<float> tmp_dn(L * HSSA_DELTANET_STATE_SIZE);
    std::vector<HssaBaseLayerWeights> tmp_bw(L);
    std::vector<float> tmp_att(static_cast<size_t>(hdr.num_attractors) * HSSA_HIDDEN_DIM);
    std::vector<float> tmp_unemb(static_cast<size_t>(hdr.vocab_size) * HSSA_HIDDEN_DIM);

    uint64_t csum = 14695981039346656037ULL;
    bool ok = true;
    for (size_t l = 0; ok && l < L; ++l) {
        float* u_ptr  = &tmp_u[l * HSSA_TT16_CORE_SIZE];
        float* v_ptr  = &tmp_v[l * HSSA_TT16_CORE_SIZE];
        float* dn_ptr = &tmp_dn[l * HSSA_DELTANET_STATE_SIZE];
        ok = ok && (std::fread(u_ptr, 1, tt_bytes, fp) == tt_bytes);
        ok = ok && (std::fread(v_ptr, 1, tt_bytes, fp) == tt_bytes);
        ok = ok && (std::fread(dn_ptr, 1, dn_bytes, fp) == dn_bytes);
        if (ok) {
            csum = fnv1a64_update(csum, u_ptr, tt_bytes);
            csum = fnv1a64_update(csum, v_ptr, tt_bytes);
            csum = fnv1a64_update(csum, dn_ptr, dn_bytes);
        }
        if (ok && hdr.version >= 2U) {
            ok = (std::fread(&tmp_bw[l], 1, bw_bytes, fp) == bw_bytes);
            if (ok) {
                csum = fnv1a64_update(csum, &tmp_bw[l], bw_bytes);
            }
        } else if (ok) {
            hssa_init_base_layer_weights(&tmp_bw[l]);
        }
    }
    if (ok && !tmp_att.empty()) {
        const size_t att_bytes = tmp_att.size() * sizeof(float);
        ok = (std::fread(tmp_att.data(), 1, att_bytes, fp) == att_bytes);
        if (ok) csum = fnv1a64_update(csum, tmp_att.data(), att_bytes);
    }
    if (ok && !tmp_unemb.empty()) {
        const size_t unemb_bytes = tmp_unemb.size() * sizeof(float);
        ok = (std::fread(tmp_unemb.data(), 1, unemb_bytes, fp) == unemb_bytes);
        if (ok) csum = fnv1a64_update(csum, tmp_unemb.data(), unemb_bytes);
    }
    std::fclose(fp);

    if (!ok || csum != hdr.payload_fnv1a) {
        return -5;
    }

    for (size_t l = 0; l < L; ++l) {
        std::memcpy(ctx->layers[l].tt_u, &tmp_u[l * HSSA_TT16_CORE_SIZE], tt_bytes);
        std::memcpy(ctx->layers[l].tt_v, &tmp_v[l * HSSA_TT16_CORE_SIZE], tt_bytes);
        std::memcpy(ctx->layers[l].deltanet_state, &tmp_dn[l * HSSA_DELTANET_STATE_SIZE], dn_bytes);
        if (ctx->layers[l].reserved[1]) {
            std::memcpy(ctx->layers[l].reserved[1], &tmp_bw[l], bw_bytes);
        }
        ctx->layers[l].reserved[0] = hssa_is_tt16_active(
            ctx->layers[l].tt_u, ctx->layers[l].tt_v, HSSA_TT16_CORE_SIZE
        ) ? reinterpret_cast<void*>(1) : nullptr;
    }
    if (!tmp_att.empty() && ctx->attractor_patterns) {
        ctx->num_attractor_patterns = static_cast<size_t>(hdr.num_attractors);
        std::memcpy(ctx->attractor_patterns, tmp_att.data(), tmp_att.size() * sizeof(float));
    }
    if (!tmp_unemb.empty()) {
        hssa_set_unembedding_matrix(ctx, tmp_unemb.data(), static_cast<size_t>(hdr.vocab_size));
    }
    if (ctx->cuda_active && ctx->cuda_drv && ctx->gpu_matrix && L > 0) {
        CudaDrv* drv = (CudaDrv*)ctx->cuda_drv;
        PtqGpuMatrix* gpu = (PtqGpuMatrix*)ctx->gpu_matrix;
        if (gpu->d_tt_u) drv->cuMemcpyHtoD(gpu->d_tt_u, ctx->layers[0].tt_u, tt_bytes);
        if (gpu->d_tt_v) drv->cuMemcpyHtoD(gpu->d_tt_v, ctx->layers[0].tt_v, tt_bytes);
    }
    return 0;
}

int hssa_train_token_step(
    HssaContext* ctx,
    const float* in_vec,
    const float* target_vec,
    float lr,
    float* out_mse_loss
) {
    if (!ctx || !in_vec || !target_vec || !(lr > 0.0f)) return -1;
    std::lock_guard<std::recursive_mutex> state_guard(*static_cast<std::recursive_mutex*>(ctx->state_lock));

    alignas(64) float pred_vec[HSSA_HIDDEN_DIM];
    int rc = hssa_forward_token(ctx, in_vec, pred_vec, nullptr);
    if (rc != 0) return rc;

    alignas(64) float err_vec[HSSA_HIDDEN_DIM];
    __m512 v_sq_err = _mm512_setzero_ps();
    __m512 v_sq_x   = _mm512_setzero_ps();
    for (size_t j = 0; j < HSSA_HIDDEN_DIM; j += 16) {
        __m512 vt = _mm512_loadu_ps(target_vec + j);
        __m512 vp = _mm512_load_ps(pred_vec + j);
        __m512 vx = _mm512_loadu_ps(in_vec + j);
        __m512 ve = _mm512_sub_ps(vt, vp);
        _mm512_store_ps(err_vec + j, ve);
        v_sq_err = _mm512_fmadd_ps(ve, ve, v_sq_err);
        v_sq_x   = _mm512_fmadd_ps(vx, vx, v_sq_x);
    }
    const float mse = _mm512_reduce_add_ps(v_sq_err) / static_cast<float>(HSSA_HIDDEN_DIM);
    if (out_mse_loss) *out_mse_loss = mse;

    const float x_norm_sq = (std::max)(1e-6f, _mm512_reduce_add_ps(v_sq_x));
    const float x_norm    = std::sqrt(x_norm_sq);
    const float inv_sqrt_r_x = 1.0f / (std::sqrt(static_cast<float>(HSSA_RANK)) * x_norm);

    // 1. Update base layer Q/K/V/Gate projections via Straight-Through Estimator (STE)
    //    and re-quantize 1.58-bit ternary bitplanes + coupled low-rank TT16 adapters
    const float layer_lr = lr / static_cast<float>((std::max)(1U, ctx->config.num_layers));
    const float base_lr  = (0.25f * layer_lr) / x_norm_sq;
    for (uint32_t l = 0; l < ctx->config.num_layers; ++l) {
        HssaLayer* layer = &ctx->layers[l];
        HssaBaseLayerWeights* bw = static_cast<HssaBaseLayerWeights*>(layer->reserved[1]);
        if (bw) {
            const size_t D_H = HSSA_HEAD_DIM;
            for (uint32_t h = 0; h < HSSA_HEADS; ++h) {
                const size_t base_off = static_cast<size_t>(h) * D_H;
                const size_t rot_k = ((size_t)(l * 7 + h * 3 + 1)) & (D_H - 1);
                const size_t rot_q = ((size_t)(l * 11 + h * 5 + 3)) & (D_H - 1);
                const size_t rot_v = ((size_t)(l * 13 + h * 7 + 5)) & (D_H - 1);
                const float* x_h = &in_vec[base_off];
                const float* e_h = &err_vec[base_off];
                for (size_t r = 0; r < D_H; ++r) {
                    const size_t d_idx = base_off + r;
                    const float x0 = x_h[r];
                    const float xk = x_h[(r + rot_k) & (D_H - 1)];
                    const float xq = x_h[(r + rot_q) & (D_H - 1)];
                    const float xv = x_h[(r + rot_v) & (D_H - 1)];
                    const float er = e_h[r];

                    bw->w_v_diag[d_idx] = std::clamp(bw->w_v_diag[d_idx] + base_lr * er * x0, -2.0f, 2.0f);
                    bw->w_v_rot[d_idx]  = std::clamp(bw->w_v_rot[d_idx]  + base_lr * er * xv, -2.0f, 2.0f);
                    bw->w_q_diag[d_idx] = std::clamp(bw->w_q_diag[d_idx] + 0.5f * base_lr * er * x0, -2.0f, 2.0f);
                    bw->w_q_rot[d_idx]  = std::clamp(bw->w_q_rot[d_idx]  + 0.5f * base_lr * er * xq, -2.0f, 2.0f);
                    bw->w_k_diag[d_idx] = std::clamp(bw->w_k_diag[d_idx] + 0.5f * base_lr * er * x0, -2.0f, 2.0f);
                    bw->w_k_rot[d_idx]  = std::clamp(bw->w_k_rot[d_idx]  + 0.5f * base_lr * er * xk, -2.0f, 2.0f);
                    bw->w_g_diag[d_idx] = std::clamp(bw->w_g_diag[d_idx] + 0.25f * base_lr * er * x0, -2.0f, 2.0f);
                    bw->w_g_rot[d_idx]  = std::clamp(bw->w_g_rot[d_idx]  + 0.25f * base_lr * er * xq, -2.0f, 2.0f);
                }
            }
            hssa_quantize_base_ternary(bw);
        }

        for (size_t r = 0; r < HSSA_RANK; ++r) {
            float* u_row = &layer->tt_u[r * HSSA_HIDDEN_DIM];
            float* v_row = &layer->tt_v[r * HSSA_HIDDEN_DIM];

            __m512 acc_vx = _mm512_setzero_ps();
            __m512 acc_ue = _mm512_setzero_ps();
            __m512 acc_vv = _mm512_setzero_ps();
            for (size_t j = 0; j < HSSA_HIDDEN_DIM; j += 16) {
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
                // Zero-initialized rank r: align v_r with normalized input x and step u_r along error e
                const __m512 v_init_v = _mm512_set1_ps(inv_sqrt_r_x);
                const __m512 v_step_u = _mm512_set1_ps(layer_lr * inv_sqrt_r_x);
                for (size_t j = 0; j < HSSA_HIDDEN_DIM; j += 16) {
                    __m512 vu   = _mm512_load_ps(u_row + j);
                    __m512 vx   = _mm512_loadu_ps(in_vec + j);
                    __m512 verr = _mm512_load_ps(err_vec + j);
                    _mm512_store_ps(v_row + j, _mm512_mul_ps(v_init_v, vx));
                    _mm512_store_ps(u_row + j, _mm512_fmadd_ps(v_step_u, verr, vu));
                }
            } else {
                const float z_r = _mm512_reduce_add_ps(acc_vx);
                const float g_r = _mm512_reduce_add_ps(acc_ue);
                const __m512 v_step_u = _mm512_set1_ps((layer_lr / x_norm_sq) * z_r);
                const __m512 v_step_v = _mm512_set1_ps((layer_lr / x_norm_sq) * g_r);
                for (size_t j = 0; j < HSSA_HIDDEN_DIM; j += 16) {
                    __m512 vu   = _mm512_load_ps(u_row + j);
                    __m512 vv   = _mm512_load_ps(v_row + j);
                    __m512 vx   = _mm512_loadu_ps(in_vec + j);
                    __m512 verr = _mm512_load_ps(err_vec + j);
                    _mm512_store_ps(u_row + j, _mm512_fmadd_ps(v_step_u, verr, vu));
                    _mm512_store_ps(v_row + j, _mm512_fmadd_ps(v_step_v, vx, vv));
                }
            }
        }
        layer->reserved[0] = reinterpret_cast<void*>(1);
    }
    if (ctx->cuda_active && ctx->cuda_drv && ctx->gpu_matrix && ctx->config.num_layers > 0) {
        CudaDrv* drv = (CudaDrv*)ctx->cuda_drv;
        PtqGpuMatrix* gpu = (PtqGpuMatrix*)ctx->gpu_matrix;
        const size_t tt_bytes = static_cast<size_t>(HSSA_TT16_CORE_SIZE) * sizeof(float);
        if (gpu->d_tt_u) drv->cuMemcpyHtoD(gpu->d_tt_u, ctx->layers[0].tt_u, tt_bytes);
        if (gpu->d_tt_v) drv->cuMemcpyHtoD(gpu->d_tt_v, ctx->layers[0].tt_v, tt_bytes);
    }

    // 2. Adapt the closest Hopfield attractor pattern along err_vec = target_vec - pred_vec
    if (ctx->attractor_patterns && ctx->num_attractor_patterns > 0) {
        size_t best_k = 0;
        float best_dot = -1e30f;
        for (size_t k = 0; k < ctx->num_attractor_patterns; ++k) {
            const float* pat = &ctx->attractor_patterns[k * HSSA_HIDDEN_DIM];
            __m512 acc = _mm512_setzero_ps();
            for (size_t j = 0; j < HSSA_HIDDEN_DIM; j += 16) {
                acc = _mm512_fmadd_ps(_mm512_load_ps(pat + j), _mm512_load_ps(pred_vec + j), acc);
            }
            float d = _mm512_reduce_add_ps(acc);
            if (d > best_dot) {
                best_dot = d;
                best_k = k;
            }
        }
        float* pat = &ctx->attractor_patterns[best_k * HSSA_HIDDEN_DIM];
        const float att_lr = (std::min)(0.50f, lr);
        const __m512 v_att_lr = _mm512_set1_ps(att_lr);
        for (size_t j = 0; j < HSSA_HIDDEN_DIM; j += 16) {
            __m512 vp   = _mm512_load_ps(pat + j);
            __m512 verr = _mm512_load_ps(err_vec + j);
            _mm512_store_ps(pat + j, _mm512_fmadd_ps(v_att_lr, verr, vp));
        }
    }

    // 3. Adapt the closest unembedding row toward target_vec so autoregressive decoding
    //    aligns with the trained continuous representation space
    if (ctx->unembed_weights && ctx->vocab_size > 0) {
        size_t best_v = 0;
        float best_dot = -1e30f;
        for (size_t v = 0; v < ctx->vocab_size; ++v) {
            const float* row = &ctx->unembed_weights[v * HSSA_HIDDEN_DIM];
            __m512 acc = _mm512_setzero_ps();
            for (size_t j = 0; j < HSSA_HIDDEN_DIM; j += 16) {
                acc = _mm512_fmadd_ps(_mm512_load_ps(row + j), _mm512_loadu_ps(target_vec + j), acc);
            }
            float d = _mm512_reduce_add_ps(acc);
            if (d > best_dot) {
                best_dot = d;
                best_v = v;
            }
        }
        float* row = &ctx->unembed_weights[best_v * HSSA_HIDDEN_DIM];
        const __m512 v_unemb_lr = _mm512_set1_ps((std::min)(0.10f, 0.05f * lr));
        for (size_t j = 0; j < HSSA_HIDDEN_DIM; j += 16) {
            __m512 vr   = _mm512_load_ps(row + j);
            __m512 verr = _mm512_load_ps(err_vec + j);
            _mm512_store_ps(row + j, _mm512_fmadd_ps(v_unemb_lr, verr, vr));
        }
    }

    return 0;
}


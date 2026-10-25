#include "hssa.h"
#include "ptq_cuda.h"

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

    // Allocate 64-byte aligned recurrent state and TT16 cores per layer
    for (uint32_t l = 0; l < num_layers; ++l) {
        HssaLayer* lyr = &ctx->layers[l];
        lyr->layer_idx = l;

        lyr->deltanet_state = (float*)_aligned_malloc(HSSA_DELTANET_STATE_SIZE * sizeof(float), HSSA_ALIGNMENT);
        lyr->tt_u = (float*)_aligned_malloc(HSSA_TT16_CORE_SIZE * sizeof(float), HSSA_ALIGNMENT);
        lyr->tt_v = (float*)_aligned_malloc(HSSA_TT16_CORE_SIZE * sizeof(float), HSSA_ALIGNMENT);

        if (!lyr->deltanet_state || !lyr->tt_u || !lyr->tt_v) {
            hssa_destroy(ctx);
            return nullptr;
        }

        std::memset(lyr->deltanet_state, 0, HSSA_DELTANET_STATE_SIZE * sizeof(float));
        std::memset(lyr->tt_u, 0, HSSA_TT16_CORE_SIZE * sizeof(float));
        std::memset(lyr->tt_v, 0, HSSA_TT16_CORE_SIZE * sizeof(float));

        lyr->vsa_handle = nullptr;
        lyr->reserved[0] = nullptr;
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
        }
        std::free(ctx->layers);
        ctx->layers = nullptr;
    }

    if (ctx->scratch_in) _aligned_free(ctx->scratch_in);
    if (ctx->scratch_out) _aligned_free(ctx->scratch_out);
    if (ctx->scratch_state) _aligned_free(ctx->scratch_state);
    if (ctx->scratch_vsa) _aligned_free(ctx->scratch_vsa);
    if (ctx->attractor_patterns) _aligned_free(ctx->attractor_patterns);

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

    __m512 v_alpha = _mm512_set1_ps(alpha);

    #pragma omp parallel for schedule(static)
    for (int h = 0; h < (int)H; ++h) {
        const float* x_h = &in_vec[h * D_H];
        float* S_h = &layer->deltanet_state[h * D_H * D_H];
        float* out_h = &out_vec[h * D_H];

        // Preload x_vec [8 * 16 = 128 floats]
        __m512 x_vec[8];
        for (size_t i = 0; i < 8; ++i) {
            x_vec[i] = _mm512_loadu_ps(x_h + i * 16);
        }

        // 1. Key normalization: k_bar = x_h / max(||x_h||_2, 1e-6)
        __m512 sq_acc0 = _mm512_setzero_ps();
        __m512 sq_acc1 = _mm512_setzero_ps();
        for (size_t i = 0; i < 8; i += 2) {
            sq_acc0 = _mm512_fmadd_ps(x_vec[i], x_vec[i], sq_acc0);
            sq_acc1 = _mm512_fmadd_ps(x_vec[i + 1], x_vec[i + 1], sq_acc1);
        }
        float sum_sq = _mm512_reduce_add_ps(_mm512_add_ps(sq_acc0, sq_acc1));
        float norm_k = std::sqrt(sum_sq);
        float inv_norm = norm_k > 1e-6f ? (1.0f / norm_k) : 0.0f;

        __m512 k_vec[8];
        __m512 v_inv_norm = _mm512_set1_ps(inv_norm);
        for (size_t i = 0; i < 8; ++i) {
            k_vec[i] = _mm512_mul_ps(x_vec[i], v_inv_norm);
        }

        // Precompute sigmoid gates
        alignas(64) float gate[HSSA_HEAD_DIM];
        for (size_t r = 0; r < D_H; ++r) {
            gate[r] = fast_sigmoid(x_h[r]);
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

            // 3. Novelty delta error & fast weight update factor
            float e_r = x_h[r] - v_hat_r;
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

            // 5. Gated Readout: dot = updated_row * x_h
            __m512 vx0 = _mm512_fmadd_ps(s0, x_vec[0], _mm512_mul_ps(s1, x_vec[1]));
            __m512 vx1 = _mm512_fmadd_ps(s2, x_vec[2], _mm512_mul_ps(s3, x_vec[3]));
            __m512 vx2 = _mm512_fmadd_ps(s4, x_vec[4], _mm512_mul_ps(s5, x_vec[5]));
            __m512 vx3 = _mm512_fmadd_ps(s6, x_vec[6], _mm512_mul_ps(s7, x_vec[7]));
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
    size_t sz = HSSA_TT16_CORE_SIZE;

    // 4-way AVX-512 unrolled vector addition
    size_t i = 0;
    for (; i + 64 <= sz; i += 64) {
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
    for (; i < sz; ++i) {
        lyr->tt_u[i] += u_delta[i];
        lyr->tt_v[i] += v_delta[i];
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

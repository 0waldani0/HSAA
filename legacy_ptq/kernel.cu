/*
 * HARD INVARIANTS (ANTI-WASHOUT PIN):
 * C11/CUDA sm_86 Split-K Bit-Serial LUT (BSL-32T) Kernel for RTX 3050 (4GB VRAM).
 * M3 Upgrades:
 * 1. Coalesced 128B Warp Loads (`LDG.E.U32`): 1 uint32_t load per lane = 4 pairs = 32 weights!
 * 2. Single-Kernel Lock-Free Last-Split Fence Reduction: ZERO cuMemsetD32 & ZERO 2nd kernel launch!
 * 3. Fused Rank-R Tensor Train (`V * z_tt`) epilogue inside the last-split reduction!
 */
#include <stdint.h>

extern "C" __global__ void bsl_lut_gemv_sm86(
    const uint32_t* __restrict__ p0_u32,
    const uint32_t* __restrict__ p1_u32,
    const uint32_t* __restrict__ p2_u32,
    const uint32_t* __restrict__ p3_u32,
    const float*    __restrict__ scales,     /* [row_tiles * col_blocks * 32] */
    const float*    __restrict__ zeros,      /* [row_tiles * col_blocks * 32] */
    const float*    __restrict__ x,          /* [cols] */
    const float*    __restrict__ d_tt_v,     /* [R * rows] or NULL */
    const float*    __restrict__ d_z_tt,     /* [R] or NULL */
    float*          __restrict__ d_partials, /* [gridDim.y * rows] L2 staging */
    uint32_t*       __restrict__ d_tile_cnt, /* [row_tiles] monotonic split counter */
    float*          __restrict__ y,          /* [rows] final exact output */
    uint32_t rows,
    uint32_t cols,
    uint32_t tt_rank,
    uint32_t active_planes                   /* 2 = fast self-draft, 3 = 3-bit, 4 = full verify */
) {
    __shared__ float s_lut_lo[16][16];
    __shared__ float s_lut_hi[16][16];
    __shared__ float s_x_sum;
    __shared__ float s_z_tt[16];

    const uint32_t tid        = threadIdx.x;         /* 0..255 (8 warps per CTA) */
    const uint32_t warp_id    = tid >> 5;            /* 0..7 */
    const uint32_t lane       = tid & 31;            /* 0..31 (each lane = 1 row in tile!) */
    const uint32_t rt         = blockIdx.x * 8 + warp_id;
    const uint32_t row        = rt * 32 + lane;
    const uint32_t col_blocks = cols >> 7;           /* cols / 128 */
    const uint32_t row_tiles  = rows >> 5;           /* rows / 32 */
    const uint32_t req_rank   = tt_rank & 0x7Fu;
    const uint32_t fused_u    = (tt_rank >> 7) & 1u;
    /* The fused epilogue stages z in s_z_tt[16]. The host wrapper rejects a fused
     * request above 16 ranks; a direct launch that gets here is a caller bug, so
     * trap instead of clamping and silently returning a truncated projection. */
    if (fused_u && req_rank > 16u) __trap();
    const uint32_t real_rank  = req_rank;

    /* Ceil-division, not truncation: col_blocks / gridDim.y dropped the remainder,
     * so e.g. 7 column blocks over 2 splits silently skipped one block of every
     * row. Empty trailing splits still run and contribute a zero partial. */
    const uint32_t cb_per_split = (col_blocks + gridDim.y - 1u) / gridDim.y;
    const uint32_t cb_start     = (blockIdx.y * cb_per_split < col_blocks)
                                    ? blockIdx.y * cb_per_split : col_blocks;
    const uint32_t cb_end       = (cb_start + cb_per_split < col_blocks)
                                    ? cb_start + cb_per_split : col_blocks;

    /* Hybrid Split-Local float4 Fused TT-U Projection (only 8 float4 loads/thread, 0 extra kernel launch!) */
    if (fused_u && active_planes != 24 && active_planes != 34 && real_rank > 0 && d_z_tt != NULL) {
        uint32_t r_id     = tid >> 4; /* 0..15 */
        uint32_t sub      = tid & 15; /* 0..15 */
        uint32_t c4_start = cb_start << 5;
        uint32_t c4_end   = cb_end   << 5;
        float z_sum = 0.0f;
        if (r_id < real_rank) {
            const float4 *u4 = (const float4*)(d_z_tt + (size_t)r_id * cols);
            const float4 *x4 = (const float4*)x;
            for (uint32_t j4 = c4_start + sub; j4 < c4_end; j4 += 16) {
                float4 uv = u4[j4];
                float4 xv = x4[j4];
                z_sum += uv.x * xv.x + uv.y * xv.y + uv.z * xv.z + uv.w * xv.w;
            }
        }
        z_sum += __shfl_down_sync(0xffffffff, z_sum, 8);
        z_sum += __shfl_down_sync(0xffffffff, z_sum, 4);
        z_sum += __shfl_down_sync(0xffffffff, z_sum, 2);
        z_sum += __shfl_down_sync(0xffffffff, z_sum, 1);
        if (sub == 0 && r_id < 16) s_z_tt[r_id] = z_sum;
    }

    float acc = 0.0f;

    for (uint32_t cb = cb_start; cb < cb_end; ++cb) {
        /* 256 threads build the 2KB LUT for 128 activations in 1 parallel step */
        uint32_t pair = tid >> 4;  /* 0..15 */
        uint32_t mask = tid & 15;  /* 0..15 */
        const float *x_lo = x + (cb << 7) + (pair << 3);
        const float *x_hi = x_lo + 4;

        float sum_lo = 0.0f, sum_hi = 0.0f;
        if (mask & 1) { sum_lo += x_lo[0]; sum_hi += x_hi[0]; }
        if (mask & 2) { sum_lo += x_lo[1]; sum_hi += x_hi[1]; }
        if (mask & 4) { sum_lo += x_lo[2]; sum_hi += x_hi[2]; }
        if (mask & 8) { sum_lo += x_lo[3]; sum_hi += x_hi[3]; }
        s_lut_lo[pair][mask] = sum_lo;
        s_lut_hi[pair][mask] = sum_hi;
        __syncthreads();

        if (warp_id == 0) {
            float p_sum = (lane < 16) ? (s_lut_lo[lane][15] + s_lut_hi[lane][15]) : 0.0f;
            p_sum += __shfl_down_sync(0xffffffff, p_sum, 8);
            p_sum += __shfl_down_sync(0xffffffff, p_sum, 4);
            p_sum += __shfl_down_sync(0xffffffff, p_sum, 2);
            p_sum += __shfl_down_sync(0xffffffff, p_sum, 1);
            if (lane == 0) s_x_sum = p_sum;
        }
        __syncthreads();

        if (rt < row_tiles) {
            size_t tile_blk_idx = (size_t)rt * col_blocks + cb;
            size_t base_u32     = tile_blk_idx * 128 + lane; /* 128 uint32_t (512B) per tile */

            float p0_sum = 0.0f, p1_sum = 0.0f, p2_sum = 0.0f, p3_sum = 0.0f;

            if (active_planes == 2) {
                #pragma unroll 4
                for (uint32_t grp4 = 0; grp4 < 4; ++grp4) {
                    size_t off   = base_u32 + (grp4 << 5);
                    uint32_t w0  = p0_u32[off];
                    uint32_t w1  = p1_u32[off];
                    uint32_t pr0 = grp4 << 2;

                    #pragma unroll 4
                    for (uint32_t k = 0; k < 4; ++k) {
                        uint32_t pr = pr0 + k;
                        uint32_t b0 = (w0 >> (k << 3)) & 0xFFu;
                        uint32_t b1 = (w1 >> (k << 3)) & 0xFFu;
                        p0_sum += s_lut_lo[pr][b0 & 15u] + s_lut_hi[pr][b0 >> 4];
                        p1_sum += s_lut_lo[pr][b1 & 15u] + s_lut_hi[pr][b1 >> 4];
                    }
                }
            } else if (active_planes == 24) {
                /* Incremental Speculative Upgrade (2-bit Draft -> 4-bit Verify):
                 * Reads ONLY LSB planes P2 & P3 (2 bits/weight) and computes exact delta! */
                #pragma unroll 4
                for (uint32_t grp4 = 0; grp4 < 4; ++grp4) {
                    size_t off   = base_u32 + (grp4 << 5);
                    uint32_t w2  = p2_u32[off];
                    uint32_t w3  = p3_u32[off];
                    uint32_t pr0 = grp4 << 2;

                    #pragma unroll 4
                    for (uint32_t k = 0; k < 4; ++k) {
                        uint32_t pr = pr0 + k;
                        uint32_t b2 = (w2 >> (k << 3)) & 0xFFu;
                        uint32_t b3 = (w3 >> (k << 3)) & 0xFFu;
                        p2_sum += s_lut_lo[pr][b2 & 15u] + s_lut_hi[pr][b2 >> 4];
                        p3_sum += s_lut_lo[pr][b3 & 15u] + s_lut_hi[pr][b3 >> 4];
                    }
                }
            } else if (active_planes == 34) {
                /* Incremental Speculative Upgrade (3-bit Draft -> 4-bit Verify):
                 * Reads ONLY LSB plane P3 (1 bit/weight!) and computes exact delta! */
                #pragma unroll 4
                for (uint32_t grp4 = 0; grp4 < 4; ++grp4) {
                    size_t off   = base_u32 + (grp4 << 5);
                    uint32_t w3  = p3_u32[off];
                    uint32_t pr0 = grp4 << 2;

                    #pragma unroll 4
                    for (uint32_t k = 0; k < 4; ++k) {
                        uint32_t pr = pr0 + k;
                        uint32_t b3 = (w3 >> (k << 3)) & 0xFFu;
                        p3_sum += s_lut_lo[pr][b3 & 15u] + s_lut_hi[pr][b3 >> 4];
                    }
                }
            } else if (active_planes == 3) {
                #pragma unroll 4
                for (uint32_t grp4 = 0; grp4 < 4; ++grp4) {
                    size_t off   = base_u32 + (grp4 << 5);
                    uint32_t w0  = p0_u32[off];
                    uint32_t w1  = p1_u32[off];
                    uint32_t w2  = p2_u32[off];
                    uint32_t pr0 = grp4 << 2;

                    #pragma unroll 4
                    for (uint32_t k = 0; k < 4; ++k) {
                        uint32_t pr = pr0 + k;
                        uint32_t b0 = (w0 >> (k << 3)) & 0xFFu;
                        uint32_t b1 = (w1 >> (k << 3)) & 0xFFu;
                        uint32_t b2 = (w2 >> (k << 3)) & 0xFFu;
                        p0_sum += s_lut_lo[pr][b0 & 15u] + s_lut_hi[pr][b0 >> 4];
                        p1_sum += s_lut_lo[pr][b1 & 15u] + s_lut_hi[pr][b1 >> 4];
                        p2_sum += s_lut_lo[pr][b2 & 15u] + s_lut_hi[pr][b2 >> 4];
                    }
                }
            } else {
                #pragma unroll 4
                for (uint32_t grp4 = 0; grp4 < 4; ++grp4) {
                    size_t off   = base_u32 + (grp4 << 5);
                    uint32_t w0  = p0_u32[off];
                    uint32_t w1  = p1_u32[off];
                    uint32_t w2  = p2_u32[off];
                    uint32_t w3  = p3_u32[off];
                    uint32_t pr0 = grp4 << 2;

                    #pragma unroll 4
                    for (uint32_t k = 0; k < 4; ++k) {
                        uint32_t pr = pr0 + k;
                        uint32_t b0 = (w0 >> (k << 3)) & 0xFFu;
                        uint32_t b1 = (w1 >> (k << 3)) & 0xFFu;
                        uint32_t b2 = (w2 >> (k << 3)) & 0xFFu;
                        uint32_t b3 = (w3 >> (k << 3)) & 0xFFu;
                        p0_sum += s_lut_lo[pr][b0 & 15u] + s_lut_hi[pr][b0 >> 4];
                        p1_sum += s_lut_lo[pr][b1 & 15u] + s_lut_hi[pr][b1 >> 4];
                        p2_sum += s_lut_lo[pr][b2 & 15u] + s_lut_hi[pr][b2 >> 4];
                        p3_sum += s_lut_lo[pr][b3 & 15u] + s_lut_hi[pr][b3 >> 4];
                    }
                }
            }

            size_t sc_idx = tile_blk_idx * 32 + lane;
            float sc      = scales[sc_idx];
            if (active_planes == 24) {
                /* Delta from 2-bit unbiased draft to exact 4-bit: skip zeros & TT! */
                acc += sc * (2.0f * p2_sum + 1.0f * p3_sum - 1.5f * s_x_sum);
            } else if (active_planes == 34) {
                /* Delta from 3-bit unbiased draft to exact 4-bit: reads ONLY P3 (1 bit/weight), skip zeros & TT! */
                acc += sc * (1.0f * p3_sum - 0.5f * s_x_sum);
            } else {
                float plane_tot = 8.0f * p0_sum + 4.0f * p1_sum + 2.0f * p2_sum + 1.0f * p3_sum;
                float mid_bias  = (active_planes < 4) ? (0.5f * (float)((1u << (4u - active_planes)) - 1u)) : 0.0f;
                acc += sc * plane_tot + (zeros[sc_idx] + sc * mid_bias) * s_x_sum;
            }
        }
        __syncthreads();
    }

    if (rt < row_tiles && row < rows) {
        if (fused_u && active_planes != 24 && active_planes != 34 && real_rank > 0 && d_tt_v != NULL) {
            for (uint32_t r = 0; r < real_rank; ++r) {
                acc += d_tt_v[(size_t)r * rows + row] * s_z_tt[r];
            }
        }
        if (gridDim.y == 1) {
            if (active_planes == 24 || active_planes == 34) {
                y[row] += acc;
            } else {
                float tot = acc;
                if (!fused_u && real_rank > 0 && d_tt_v != NULL && d_z_tt != NULL) {
                    for (uint32_t r = 0; r < real_rank; ++r) {
                        tot += d_tt_v[(size_t)r * rows + row] * d_z_tt[r];
                    }
                }
                y[row] = tot;
            }
        } else {
            /* Single-Kernel Lock-Free Last-Split Fence Reduction (Release + Acquire, volatile L2 bypass) */
            volatile float *v_part = (volatile float*)d_partials;
            v_part[(size_t)blockIdx.y * rows + row] = acc;
            __threadfence(); /* Release: ensure d_partials write is visible in L2 */
            __syncwarp(0xffffffff);

            uint32_t is_last = 0;
            if (lane == 0) {
                uint32_t prev = atomicAdd(&d_tile_cnt[rt], 1u);
                is_last = ((prev % gridDim.y) == (gridDim.y - 1u)) ? 1u : 0u;
            }
            is_last = __shfl_sync(0xffffffff, is_last, 0);

            if (is_last) {
                __threadfence(); /* Acquire (O-AUDIT2 Pt 4 fix): prevent stale L1 read of d_partials */
                float tot = 0.0f;
                for (uint32_t s = 0; s < gridDim.y; ++s) {
                    tot += v_part[(size_t)s * rows + row];
                }
                if (active_planes == 24 || active_planes == 34) {
                    y[row] += tot;
                } else {
                    if (!fused_u && real_rank > 0 && d_tt_v != NULL && d_z_tt != NULL) {
                        for (uint32_t r = 0; r < real_rank; ++r) {
                            tot += d_tt_v[(size_t)r * rows + row] * d_z_tt[r];
                        }
                    }
                    y[row] = tot;
                }
            }
        }
    }
}

/*
 * Vectorized 128-bit (`float4` LDG.E.128) Device-side TT input projection:
 * Computes d_z_tt[r] = dot(d_tt_u[r], d_x) for r = 0..R-1 in a single 1-load-per-thread wave for N=1024!
 */
extern "C" __global__ void tt_project_u_sm86(
    const float* __restrict__ d_tt_u, /* [R * cols] */
    const float* __restrict__ d_x,    /* [cols] */
    float*       __restrict__ d_z_tt, /* [R] */
    uint32_t cols
) {
    __shared__ float s_warp[8];
    const uint32_t r       = blockIdx.x;
    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid >> 5;
    const uint32_t lane    = tid & 31;
    const float4 *u4       = (const float4*)(d_tt_u + (size_t)r * cols);
    const float4 *x4       = (const float4*)d_x;
    const uint32_t cols4   = cols >> 2;

    float sum = 0.0f;
    for (uint32_t j = tid; j < cols4; j += 256) {
        float4 uv = u4[j];
        float4 xv = x4[j];
        sum += uv.x * xv.x + uv.y * xv.y + uv.z * xv.z + uv.w * xv.w;
    }
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    if (lane == 0) s_warp[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        float wsum = (lane < 8) ? s_warp[lane] : 0.0f;
        wsum += __shfl_down_sync(0xffffffff, wsum, 4);
        wsum += __shfl_down_sync(0xffffffff, wsum, 2);
        wsum += __shfl_down_sync(0xffffffff, wsum, 1);
        if (lane == 0) d_z_tt[r] = wsum;
    }
}

/*
 * Fused Residual-Add + RMSNorm (1 CTA of 256 threads):
 * If res_add != NULL, updates x_inout[i] += res_add[i] in-place before normalizing into y[i].
 */
extern "C" __global__ void rmsnorm_res_sm86(
    float*       __restrict__ y,
    float*       __restrict__ x_inout,
    const float* __restrict__ res_add,
    const float* __restrict__ w,
    uint32_t n,
    float eps
) {
    __shared__ float s_warp[8];
    __shared__ float s_scale;
    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid >> 5;
    const uint32_t lane    = tid & 31;

    float ss = 0.0f;
    for (uint32_t i = tid; i < n; i += 256) {
        float v = x_inout[i];
        if (res_add != NULL) {
            v += res_add[i];
            x_inout[i] = v;
        }
        ss += v * v;
    }
    ss += __shfl_down_sync(0xffffffff, ss, 16);
    ss += __shfl_down_sync(0xffffffff, ss, 8);
    ss += __shfl_down_sync(0xffffffff, ss, 4);
    ss += __shfl_down_sync(0xffffffff, ss, 2);
    ss += __shfl_down_sync(0xffffffff, ss, 1);
    if (lane == 0) s_warp[warp_id] = ss;
    __syncthreads();

    if (warp_id == 0) {
        float tot = (lane < 8) ? s_warp[lane] : 0.0f;
        tot += __shfl_down_sync(0xffffffff, tot, 4);
        tot += __shfl_down_sync(0xffffffff, tot, 2);
        tot += __shfl_down_sync(0xffffffff, tot, 1);
        if (lane == 0) s_scale = rsqrtf(tot / (float)n + eps);
    }
    __syncthreads();

    const float sc = s_scale;
    for (uint32_t i = tid; i < n; i += 256) {
        y[i] = (x_inout[i] * sc) * w[i];
    }
}

/*
 * Fused Per-Head QK-RMSNorm + NeoX Half-Split RoPE + GPU KV-Cache Append:
 * Grid: (n_heads + n_kv_heads), Block: 128 threads.
 */
extern "C" __global__ void qknorm_rope_kv_sm86(
    float*       __restrict__ qkv_buf,
    const float* __restrict__ w_q_norm,
    const float* __restrict__ w_k_norm,
    float*       __restrict__ k_cache_l,
    float*       __restrict__ v_cache_l,
    uint32_t n_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim,
    uint32_t pos,
    float eps,
    float rope_theta
) {
    __shared__ float s_h[256];
    __shared__ float s_warp[4];
    __shared__ float s_scale;

    const uint32_t bid     = blockIdx.x;
    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid >> 5;
    const uint32_t lane    = tid & 31;
    const uint32_t q_dim   = n_heads * head_dim;
    const uint32_t kv_dim  = n_kv_heads * head_dim;
    const int      is_k    = (bid >= n_heads);
    const uint32_t h_idx   = is_k ? (bid - n_heads) : bid;
    float *h_ptr           = is_k ? (qkv_buf + q_dim + h_idx * head_dim) : (qkv_buf + h_idx * head_dim);
    const float *w_norm    = is_k ? w_k_norm : w_q_norm;

    if (w_norm != NULL) {
        float ss = 0.0f;
        for (uint32_t d = tid; d < head_dim; d += 128) {
            float v = h_ptr[d];
            ss += v * v;
        }
        ss += __shfl_down_sync(0xffffffff, ss, 16);
        ss += __shfl_down_sync(0xffffffff, ss, 8);
        ss += __shfl_down_sync(0xffffffff, ss, 4);
        ss += __shfl_down_sync(0xffffffff, ss, 2);
        ss += __shfl_down_sync(0xffffffff, ss, 1);
        if (lane == 0) s_warp[warp_id] = ss;
        __syncthreads();

        if (warp_id == 0) {
            float tot = (lane < 4) ? s_warp[lane] : 0.0f;
            tot += __shfl_down_sync(0xffffffff, tot, 2);
            tot += __shfl_down_sync(0xffffffff, tot, 1);
            if (lane == 0) s_scale = rsqrtf(tot / (float)head_dim + eps);
        }
        __syncthreads();

        const float sc = s_scale;
        for (uint32_t d = tid; d < head_dim; d += 128) {
            s_h[d] = (h_ptr[d] * sc) * w_norm[d];
        }
    } else {
        for (uint32_t d = tid; d < head_dim; d += 128) {
            s_h[d] = h_ptr[d];
        }
    }
    __syncthreads();

    const uint32_t half = head_dim >> 1;
    for (uint32_t i = tid; i < half; i += 128) {
        float freq = 1.0f / powf(rope_theta, (float)(2 * i) / (float)head_dim);
        float ang  = (float)pos * freq;
        float c    = cosf(ang);
        float s    = sinf(ang);
        float x0   = s_h[i];
        float x1   = s_h[i + half];
        float r0   = x0 * c - x1 * s;
        float r1   = x1 * c + x0 * s;
        h_ptr[i]        = r0;
        h_ptr[i + half] = r1;
        if (is_k) {
            k_cache_l[(size_t)pos * kv_dim + h_idx * head_dim + i]        = r0;
            k_cache_l[(size_t)pos * kv_dim + h_idx * head_dim + i + half] = r1;
        }
    }
    if (is_k) {
        for (uint32_t d = tid; d < head_dim; d += 128) {
            v_cache_l[(size_t)pos * kv_dim + h_idx * head_dim + d] = qkv_buf[q_dim + kv_dim + h_idx * head_dim + d];
        }
    }
}

/*
 * Causal Grouped-Query Attention (GQA) over GPU-Resident KV-Cache:
 * Grid: n_heads (16), Block: 128 threads (4 warps).
 */
extern "C" __global__ void gqa_attn_sm86(
    float*       __restrict__ attn_out,
    const float* __restrict__ q,
    const float* __restrict__ k_cache_l,
    const float* __restrict__ v_cache_l,
    uint32_t n_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim,
    uint32_t seq_len
) {
    __shared__ float s_q[256];
    __shared__ float s_scores[2048];
    __shared__ float s_red[4];
    __shared__ float s_max;
    __shared__ float s_inv_sum;

    /* s_scores is fixed at 2048 floats of shared memory; qwen3_gqa_attn() on the
     * host already rejects seq_len > 2048. Trap rather than write out of bounds
     * if this kernel is ever launched directly. */
    if (seq_len > 2048u) {
        __trap();
    }

    const uint32_t h        = blockIdx.x;
    const uint32_t tid      = threadIdx.x;
    const uint32_t warp_id  = tid >> 5;
    const uint32_t lane     = tid & 31;
    const uint32_t kv_dim   = n_kv_heads * head_dim;
    const uint32_t kv_group = n_heads / n_kv_heads;
    const uint32_t kh       = h / kv_group;
    const float    scale    = rsqrtf((float)head_dim);

    for (uint32_t d = tid; d < head_dim; d += 128) {
        s_q[d] = q[h * head_dim + d];
    }
    __syncthreads();

    for (uint32_t t = warp_id; t < seq_len; t += 4) {
        const float *kt = k_cache_l + (size_t)t * kv_dim + kh * head_dim;
        float dot = 0.0f;
        for (uint32_t d = lane; d < head_dim; d += 32) {
            dot += s_q[d] * kt[d];
        }
        dot += __shfl_down_sync(0xffffffff, dot, 16);
        dot += __shfl_down_sync(0xffffffff, dot, 8);
        dot += __shfl_down_sync(0xffffffff, dot, 4);
        dot += __shfl_down_sync(0xffffffff, dot, 2);
        dot += __shfl_down_sync(0xffffffff, dot, 1);
        if (lane == 0) s_scores[t] = dot * scale;
    }
    __syncthreads();

    float lmax = -1e30f;
    for (uint32_t t = tid; t < seq_len; t += 128) {
        float v = s_scores[t];
        if (v > lmax) lmax = v;
    }
    lmax = fmaxf(lmax, __shfl_down_sync(0xffffffff, lmax, 16));
    lmax = fmaxf(lmax, __shfl_down_sync(0xffffffff, lmax, 8));
    lmax = fmaxf(lmax, __shfl_down_sync(0xffffffff, lmax, 4));
    lmax = fmaxf(lmax, __shfl_down_sync(0xffffffff, lmax, 2));
    lmax = fmaxf(lmax, __shfl_down_sync(0xffffffff, lmax, 1));
    if (lane == 0) s_red[warp_id] = lmax;
    __syncthreads();
    if (tid == 0) {
        s_max = fmaxf(fmaxf(s_red[0], s_red[1]), fmaxf(s_red[2], s_red[3]));
    }
    __syncthreads();

    const float mx = s_max;
    float lsum = 0.0f;
    for (uint32_t t = tid; t < seq_len; t += 128) {
        float e = expf(s_scores[t] - mx);
        s_scores[t] = e;
        lsum += e;
    }
    lsum += __shfl_down_sync(0xffffffff, lsum, 16);
    lsum += __shfl_down_sync(0xffffffff, lsum, 8);
    lsum += __shfl_down_sync(0xffffffff, lsum, 4);
    lsum += __shfl_down_sync(0xffffffff, lsum, 2);
    lsum += __shfl_down_sync(0xffffffff, lsum, 1);
    if (lane == 0) s_red[warp_id] = lsum;
    __syncthreads();
    if (tid == 0) {
        s_inv_sum = 1.0f / (s_red[0] + s_red[1] + s_red[2] + s_red[3] + 1e-30f);
    }
    __syncthreads();

    const float inv_s = s_inv_sum;
    for (uint32_t d = tid; d < head_dim; d += 128) {
        float acc = 0.0f;
        for (uint32_t t = 0; t < seq_len; ++t) {
            acc += (s_scores[t] * inv_s) * v_cache_l[(size_t)t * kv_dim + kh * head_dim + d];
        }
        attn_out[h * head_dim + d] = acc;
    }
}

/*
 * Fused SwiGLU Activation: out[i] = silu(gu_buf[i]) * gu_buf[i + inter]
 */
extern "C" __global__ void swiglu_sm86(
    float*       __restrict__ out,
    const float* __restrict__ gu_buf,
    uint32_t inter
) {
    const uint32_t i = blockIdx.x * 256 + threadIdx.x;
    if (i < inter) {
        float g = gu_buf[i];
        out[i] = (g / (1.0f + expf(-g))) * gu_buf[i + inter];
    }
}

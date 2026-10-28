#ifndef SIMD_COMMON_H
#define SIMD_COMMON_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

/*
 * =============================================================================
 * Unified Single-Source-of-Truth AVX-512 / AVX2 SIMD Kernels
 * Shared across VSA DB, SecondBrain (PopCountBrain), HSSA, and UnifiedEngine
 * =============================================================================
 */

static inline uint32_t simd_hamming_512(const uint64_t* a, const uint64_t* b) noexcept {
#if (defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)) || (defined(_MSC_VER) && defined(__AVX512F__))
    __m512i va = _mm512_loadu_si512(reinterpret_cast<const void*>(a));
    __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void*>(b));
    __m512i vx = _mm512_xor_si512(va, vb);
    __m512i vp = _mm512_popcnt_epi64(vx);
    return static_cast<uint32_t>(_mm512_reduce_add_epi64(vp));
#else
    uint32_t d = 0;
    for (int i = 0; i < 8; ++i) {
#if defined(_MSC_VER)
        d += static_cast<uint32_t>(__popcnt64(a[i] ^ b[i]));
#else
        d += static_cast<uint32_t>(__builtin_popcountll(a[i] ^ b[i]));
#endif
    }
    return d;
#endif
}

static inline void simd_xnor_512(const uint64_t* a, const uint64_t* b, uint64_t* out) noexcept {
#if defined(__AVX512F__)
    __m512i va = _mm512_loadu_si512(reinterpret_cast<const void*>(a));
    __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void*>(b));
    __m512i res = _mm512_ternarylogic_epi64(va, vb, va, 0x99);
    _mm512_storeu_si512(reinterpret_cast<void*>(out), res);
#else
    for (int i = 0; i < 8; ++i) {
        out[i] = ~(a[i] ^ b[i]);
    }
#endif
}

static inline float simd_dot_4096_f32(const float* a, const float* b) noexcept {
#if defined(__AVX512F__)
    __m512 vdot0 = _mm512_setzero_ps();
    __m512 vdot1 = _mm512_setzero_ps();
    __m512 vdot2 = _mm512_setzero_ps();
    __m512 vdot3 = _mm512_setzero_ps();
    for (size_t i = 0; i < 4096; i += 64) {
        vdot0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i),      vdot0);
        vdot1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), vdot1);
        vdot2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), vdot2);
        vdot3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), vdot3);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(vdot0, vdot1), _mm512_add_ps(vdot2, vdot3)));
#else
    float dot = 0.0f;
    for (size_t i = 0; i < 4096; ++i) dot += a[i] * b[i];
    return dot;
#endif
}

static inline float simd_norm_sq_4096_f32(const float* a) noexcept {
#if defined(__AVX512F__)
    __m512 vna0 = _mm512_setzero_ps();
    __m512 vna1 = _mm512_setzero_ps();
    __m512 vna2 = _mm512_setzero_ps();
    __m512 vna3 = _mm512_setzero_ps();
    for (size_t i = 0; i < 4096; i += 64) {
        __m512 va0 = _mm512_loadu_ps(a + i);
        __m512 va1 = _mm512_loadu_ps(a + i + 16);
        __m512 va2 = _mm512_loadu_ps(a + i + 32);
        __m512 va3 = _mm512_loadu_ps(a + i + 48);
        vna0 = _mm512_fmadd_ps(va0, va0, vna0);
        vna1 = _mm512_fmadd_ps(va1, va1, vna1);
        vna2 = _mm512_fmadd_ps(va2, va2, vna2);
        vna3 = _mm512_fmadd_ps(va3, va3, vna3);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(vna0, vna1), _mm512_add_ps(vna2, vna3)));
#else
    float na = 0.0f;
    for (size_t i = 0; i < 4096; ++i) na += a[i] * a[i];
    return na;
#endif
}

static inline float simd_cosine_4096_f32(const float* a, const float* b) noexcept {
    if (!a || !b) return 0.0f;
#if defined(__AVX512F__)
    __m512 vdot0 = _mm512_setzero_ps();
    __m512 vdot1 = _mm512_setzero_ps();
    __m512 vdot2 = _mm512_setzero_ps();
    __m512 vdot3 = _mm512_setzero_ps();
    __m512 vna0  = _mm512_setzero_ps();
    __m512 vna1  = _mm512_setzero_ps();
    __m512 vna2  = _mm512_setzero_ps();
    __m512 vna3  = _mm512_setzero_ps();
    __m512 vnb0  = _mm512_setzero_ps();
    __m512 vnb1  = _mm512_setzero_ps();
    __m512 vnb2  = _mm512_setzero_ps();
    __m512 vnb3  = _mm512_setzero_ps();
    for (size_t i = 0; i < 4096; i += 64) {
        __m512 va0 = _mm512_loadu_ps(a + i);
        __m512 vb0 = _mm512_loadu_ps(b + i);
        __m512 va1 = _mm512_loadu_ps(a + i + 16);
        __m512 vb1 = _mm512_loadu_ps(b + i + 16);
        __m512 va2 = _mm512_loadu_ps(a + i + 32);
        __m512 vb2 = _mm512_loadu_ps(b + i + 32);
        __m512 va3 = _mm512_loadu_ps(a + i + 48);
        __m512 vb3 = _mm512_loadu_ps(b + i + 48);

        vdot0 = _mm512_fmadd_ps(va0, vb0, vdot0);
        vdot1 = _mm512_fmadd_ps(va1, vb1, vdot1);
        vdot2 = _mm512_fmadd_ps(va2, vb2, vdot2);
        vdot3 = _mm512_fmadd_ps(va3, vb3, vdot3);

        vna0  = _mm512_fmadd_ps(va0, va0, vna0);
        vna1  = _mm512_fmadd_ps(va1, va1, vna1);
        vna2  = _mm512_fmadd_ps(va2, va2, vna2);
        vna3  = _mm512_fmadd_ps(va3, va3, vna3);

        vnb0  = _mm512_fmadd_ps(vb0, vb0, vnb0);
        vnb1  = _mm512_fmadd_ps(vb1, vb1, vnb1);
        vnb2  = _mm512_fmadd_ps(vb2, vb2, vnb2);
        vnb3  = _mm512_fmadd_ps(vb3, vb3, vnb3);
    }
    const float dot = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(vdot0, vdot1), _mm512_add_ps(vdot2, vdot3)));
    const float na  = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(vna0, vna1),   _mm512_add_ps(vna2, vna3)));
    const float nb  = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(vnb0, vnb1),   _mm512_add_ps(vnb2, vnb3)));
    const float denom = std::sqrt(na * nb);
    return (denom > 1e-12f) ? (dot / denom) : 0.0f;
#else
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < 4096; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    const float denom = std::sqrt(na * nb);
    return (denom > 1e-12f) ? (dot / denom) : 0.0f;
#endif
}

#endif // SIMD_COMMON_H

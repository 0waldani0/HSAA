#ifndef AVX512_PORTABLE_SHIM_H
#define AVX512_PORTABLE_SHIM_H

#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>

/*
 * Hardware-Accelerated AVX2 + FMA Register Shim for AVX-512 Intrinsics
 * Active ONLY when the target CPU does not support hardware AVX-512F
 * (e.g. AMD EPYC Zen2/Zen3 with AVX2+FMA).
 * Maps each 512-bit vector to a pair of 256-bit YMM registers (__m256 lo, hi)
 * so all arithmetic executes in hardware SIMD registers with zero stack spills.
 */
#if !defined(__AVX512F__)

struct SimdShimPs512 {
#if defined(__AVX2__)
    __m256 lo;
    __m256 hi;
#else
    alignas(32) float f[16];
#endif
};

struct SimdShimSi512 {
#if defined(__AVX2__)
    __m256i lo;
    __m256i hi;
#else
    alignas(32) uint64_t u64[8];
#endif
};

using SimdShimMask16 = uint16_t;
using SimdShimMask8  = uint8_t;

static inline SimdShimPs512 simd_shim_setzero_ps() noexcept {
#if defined(__AVX2__)
    return {_mm256_setzero_ps(), _mm256_setzero_ps()};
#else
    SimdShimPs512 r{};
    return r;
#endif
}

static inline SimdShimSi512 simd_shim_setzero_si512() noexcept {
#if defined(__AVX2__)
    return {_mm256_setzero_si256(), _mm256_setzero_si256()};
#else
    SimdShimSi512 r{};
    return r;
#endif
}

static inline SimdShimPs512 simd_shim_set1_ps(float v) noexcept {
#if defined(__AVX2__)
    __m256 vv = _mm256_set1_ps(v);
    return {vv, vv};
#else
    SimdShimPs512 r;
    for (int i = 0; i < 16; ++i) r.f[i] = v;
    return r;
#endif
}

static inline SimdShimPs512 simd_shim_loadu_ps(const void* p) noexcept {
#if defined(__AVX2__)
    const float* fp = static_cast<const float*>(p);
    return {_mm256_loadu_ps(fp), _mm256_loadu_ps(fp + 8)};
#else
    SimdShimPs512 r;
    std::memcpy(r.f, p, 64);
    return r;
#endif
}

static inline SimdShimPs512 simd_shim_load_ps(const void* p) noexcept {
#if defined(__AVX2__)
    const float* fp = static_cast<const float*>(p);
    return {_mm256_loadu_ps(fp), _mm256_loadu_ps(fp + 8)};
#else
    SimdShimPs512 r;
    std::memcpy(r.f, p, 64);
    return r;
#endif
}

static inline void simd_shim_storeu_ps(void* p, SimdShimPs512 a) noexcept {
#if defined(__AVX2__)
    float* fp = static_cast<float*>(p);
    _mm256_storeu_ps(fp, a.lo);
    _mm256_storeu_ps(fp + 8, a.hi);
#else
    std::memcpy(p, a.f, 64);
#endif
}

static inline void simd_shim_store_ps(void* p, SimdShimPs512 a) noexcept {
#if defined(__AVX2__)
    float* fp = static_cast<float*>(p);
    _mm256_storeu_ps(fp, a.lo);
    _mm256_storeu_ps(fp + 8, a.hi);
#else
    std::memcpy(p, a.f, 64);
#endif
}

static inline SimdShimSi512 simd_shim_loadu_si512(const void* p) noexcept {
#if defined(__AVX2__)
    const __m256i* ip = static_cast<const __m256i*>(p);
    return {_mm256_loadu_si256(ip), _mm256_loadu_si256(ip + 1)};
#else
    SimdShimSi512 r;
    std::memcpy(r.u64, p, 64);
    return r;
#endif
}

static inline void simd_shim_storeu_si512(void* p, SimdShimSi512 a) noexcept {
#if defined(__AVX2__)
    __m256i* ip = static_cast<__m256i*>(p);
    _mm256_storeu_si256(ip, a.lo);
    _mm256_storeu_si256(ip + 1, a.hi);
#else
    std::memcpy(p, a.u64, 64);
#endif
}

static inline SimdShimPs512 simd_shim_add_ps(SimdShimPs512 a, SimdShimPs512 b) noexcept {
#if defined(__AVX2__)
    return {_mm256_add_ps(a.lo, b.lo), _mm256_add_ps(a.hi, b.hi)};
#else
    SimdShimPs512 r;
    for (int i = 0; i < 16; ++i) r.f[i] = a.f[i] + b.f[i];
    return r;
#endif
}

static inline SimdShimPs512 simd_shim_sub_ps(SimdShimPs512 a, SimdShimPs512 b) noexcept {
#if defined(__AVX2__)
    return {_mm256_sub_ps(a.lo, b.lo), _mm256_sub_ps(a.hi, b.hi)};
#else
    SimdShimPs512 r;
    for (int i = 0; i < 16; ++i) r.f[i] = a.f[i] - b.f[i];
    return r;
#endif
}

static inline SimdShimPs512 simd_shim_mul_ps(SimdShimPs512 a, SimdShimPs512 b) noexcept {
#if defined(__AVX2__)
    return {_mm256_mul_ps(a.lo, b.lo), _mm256_mul_ps(a.hi, b.hi)};
#else
    SimdShimPs512 r;
    for (int i = 0; i < 16; ++i) r.f[i] = a.f[i] * b.f[i];
    return r;
#endif
}

static inline SimdShimPs512 simd_shim_fmadd_ps(SimdShimPs512 a, SimdShimPs512 b, SimdShimPs512 c) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    return {_mm256_fmadd_ps(a.lo, b.lo, c.lo), _mm256_fmadd_ps(a.hi, b.hi, c.hi)};
#elif defined(__AVX2__)
    return {_mm256_add_ps(_mm256_mul_ps(a.lo, b.lo), c.lo), _mm256_add_ps(_mm256_mul_ps(a.hi, b.hi), c.hi)};
#else
    SimdShimPs512 r;
    for (int i = 0; i < 16; ++i) r.f[i] = a.f[i] * b.f[i] + c.f[i];
    return r;
#endif
}

static inline SimdShimPs512 simd_shim_fnmadd_ps(SimdShimPs512 a, SimdShimPs512 b, SimdShimPs512 c) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    return {_mm256_fnmadd_ps(a.lo, b.lo, c.lo), _mm256_fnmadd_ps(a.hi, b.hi, c.hi)};
#elif defined(__AVX2__)
    return {_mm256_sub_ps(c.lo, _mm256_mul_ps(a.lo, b.lo)), _mm256_sub_ps(c.hi, _mm256_mul_ps(a.hi, b.hi))};
#else
    SimdShimPs512 r;
    for (int i = 0; i < 16; ++i) r.f[i] = -(a.f[i] * b.f[i]) + c.f[i];
    return r;
#endif
}

static inline float simd_shim_reduce_add_ps(SimdShimPs512 a) noexcept {
#if defined(__AVX2__)
    __m256 sum256 = _mm256_add_ps(a.lo, a.hi);
    __m128 lo128  = _mm256_castps256_ps128(sum256);
    __m128 hi128  = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(lo128, hi128);
    __m128 shuf   = _mm_movehdup_ps(sum128);
    __m128 sums   = _mm_add_ps(sum128, shuf);
    shuf          = _mm_movehl_ps(shuf, sums);
    sums          = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
#else
    float s = 0.0f;
    for (int i = 0; i < 16; ++i) s += a.f[i];
    return s;
#endif
}

static inline SimdShimPs512 simd_shim_roundscale_ps(SimdShimPs512 a, int /*imm*/) noexcept {
#if defined(__AVX2__)
    return {
        _mm256_round_ps(a.lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
        _mm256_round_ps(a.hi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
    };
#else
    SimdShimPs512 r;
    for (int i = 0; i < 16; ++i) r.f[i] = std::nearbyintf(a.f[i]);
    return r;
#endif
}

static inline SimdShimMask16 simd_shim_cmp_ps_mask(SimdShimPs512 a, SimdShimPs512 b, int imm) noexcept {
#if defined(__AVX2__)
    alignas(32) float fa[16], fb[16];
    simd_shim_store_ps(fa, a);
    simd_shim_store_ps(fb, b);
    SimdShimMask16 m = 0;
    for (int i = 0; i < 16; ++i) {
        bool cond = false;
        switch (imm & 0x1f) {
            case _CMP_EQ_OQ: case _CMP_EQ_OS: cond = (fa[i] == fb[i]); break;
            case _CMP_LT_OS: case _CMP_LT_OQ: cond = (fa[i] <  fb[i]); break;
            case _CMP_LE_OS: case _CMP_LE_OQ: cond = (fa[i] <= fb[i]); break;
            case _CMP_GT_OS: case _CMP_GT_OQ: cond = (fa[i] >  fb[i]); break;
            case _CMP_GE_OS: case _CMP_GE_OQ: cond = (fa[i] >= fb[i]); break;
            case _CMP_NEQ_UQ: case _CMP_NEQ_US: cond = (fa[i] != fb[i]); break;
            default: cond = (fa[i] > fb[i]); break;
        }
        if (cond) m |= static_cast<SimdShimMask16>(1U << i);
    }
    return m;
#else
    SimdShimMask16 m = 0;
    for (int i = 0; i < 16; ++i) {
        if (a.f[i] > b.f[i]) m |= static_cast<SimdShimMask16>(1U << i);
    }
    return m;
#endif
}

static inline SimdShimPs512 simd_shim_maskz_mov_ps(SimdShimMask16 k, SimdShimPs512 a) noexcept {
    alignas(32) float fa[16], fr[16];
    simd_shim_store_ps(fa, a);
    for (int i = 0; i < 16; ++i) {
        fr[i] = ((k >> i) & 1U) ? fa[i] : 0.0f;
    }
    return simd_shim_load_ps(fr);
}

static inline SimdShimPs512 simd_shim_mask_blend_ps(SimdShimMask16 k, SimdShimPs512 a, SimdShimPs512 b) noexcept {
    alignas(32) float fa[16], fb[16], fr[16];
    simd_shim_store_ps(fa, a);
    simd_shim_store_ps(fb, b);
    for (int i = 0; i < 16; ++i) {
        fr[i] = ((k >> i) & 1U) ? fb[i] : fa[i];
    }
    return simd_shim_load_ps(fr);
}

static inline SimdShimSi512 simd_shim_or_si512(SimdShimSi512 a, SimdShimSi512 b) noexcept {
#if defined(__AVX2__)
    return {_mm256_or_si256(a.lo, b.lo), _mm256_or_si256(a.hi, b.hi)};
#else
    SimdShimSi512 r;
    for (int i = 0; i < 8; ++i) r.u64[i] = a.u64[i] | b.u64[i];
    return r;
#endif
}

static inline SimdShimSi512 simd_shim_xor_si512(SimdShimSi512 a, SimdShimSi512 b) noexcept {
#if defined(__AVX2__)
    return {_mm256_xor_si256(a.lo, b.lo), _mm256_xor_si256(a.hi, b.hi)};
#else
    SimdShimSi512 r;
    for (int i = 0; i < 8; ++i) r.u64[i] = a.u64[i] ^ b.u64[i];
    return r;
#endif
}

static inline SimdShimSi512 simd_shim_and_si512(SimdShimSi512 a, SimdShimSi512 b) noexcept {
#if defined(__AVX2__)
    return {_mm256_and_si256(a.lo, b.lo), _mm256_and_si256(a.hi, b.hi)};
#else
    SimdShimSi512 r;
    for (int i = 0; i < 8; ++i) r.u64[i] = a.u64[i] & b.u64[i];
    return r;
#endif
}

static inline SimdShimMask8 simd_shim_test_epi64_mask(SimdShimSi512 a, SimdShimSi512 b) noexcept {
    alignas(32) uint64_t ua[8], ub[8];
    simd_shim_storeu_si512(ua, a);
    simd_shim_storeu_si512(ub, b);
    SimdShimMask8 m = 0;
    for (int i = 0; i < 8; ++i) {
        if ((ua[i] & ub[i]) != 0ULL) m |= static_cast<SimdShimMask8>(1U << i);
    }
    return m;
}

#define __m512                    SimdShimPs512
#define __m512i                   SimdShimSi512
#define __mmask16                 SimdShimMask16
#define __mmask8                  SimdShimMask8
#undef  _mm512_setzero_ps
#define _mm512_setzero_ps         simd_shim_setzero_ps
#undef  _mm512_setzero_si512
#define _mm512_setzero_si512      simd_shim_setzero_si512
#undef  _mm512_set1_ps
#define _mm512_set1_ps            simd_shim_set1_ps
#undef  _mm512_loadu_ps
#define _mm512_loadu_ps           simd_shim_loadu_ps
#undef  _mm512_load_ps
#define _mm512_load_ps            simd_shim_load_ps
#undef  _mm512_storeu_ps
#define _mm512_storeu_ps          simd_shim_storeu_ps
#undef  _mm512_store_ps
#define _mm512_store_ps           simd_shim_store_ps
#undef  _mm512_loadu_si512
#define _mm512_loadu_si512        simd_shim_loadu_si512
#undef  _mm512_storeu_si512
#define _mm512_storeu_si512       simd_shim_storeu_si512
#undef  _mm512_add_ps
#define _mm512_add_ps             simd_shim_add_ps
#undef  _mm512_sub_ps
#define _mm512_sub_ps             simd_shim_sub_ps
#undef  _mm512_mul_ps
#define _mm512_mul_ps             simd_shim_mul_ps
#undef  _mm512_fmadd_ps
#define _mm512_fmadd_ps           simd_shim_fmadd_ps
#undef  _mm512_fnmadd_ps
#define _mm512_fnmadd_ps          simd_shim_fnmadd_ps
#undef  _mm512_reduce_add_ps
#define _mm512_reduce_add_ps      simd_shim_reduce_add_ps
#undef  _mm512_roundscale_ps
#define _mm512_roundscale_ps      simd_shim_roundscale_ps
#undef  _mm512_cmp_ps_mask
#define _mm512_cmp_ps_mask        simd_shim_cmp_ps_mask
#undef  _mm512_maskz_mov_ps
#define _mm512_maskz_mov_ps       simd_shim_maskz_mov_ps
#undef  _mm512_mask_blend_ps
#define _mm512_mask_blend_ps      simd_shim_mask_blend_ps
#undef  _mm512_or_si512
#define _mm512_or_si512           simd_shim_or_si512
#undef  _mm512_xor_si512
#define _mm512_xor_si512          simd_shim_xor_si512
#undef  _mm512_and_si512
#define _mm512_and_si512          simd_shim_and_si512
#undef  _mm512_test_epi64_mask
#define _mm512_test_epi64_mask    simd_shim_test_epi64_mask

#endif // !defined(__AVX512F__)

#endif // AVX512_PORTABLE_SHIM_H

// assoc_sketch.h -- deterministic 512-bit lexical sketch used by the C-ABI
// associative fallback (brain_query_relation / brain_check_conflict).
//
// This is NOT the neural / VSA bank. It is a character-level random-projection
// SimHash over the trigrams of one normalized string, so that two strings that
// differ by a typo, letter case or whitespace land close in Hamming space while
// unrelated strings stay near the 50% baseline. It is deliberately lexical:
// "car" and "automobile" are NOT close here. The only purpose is to let a
// query that misses the exact dictionary still find the stored fact it was
// clearly derived from.
//
// One source of truth: the engine's fallback and measur/assoc_calib.cpp both
// include this header, so the calibrated threshold describes shipped code.
//
// Contract of the entries:
//   * normalize(): ASCII lowercase, whitespace runs collapsed to one space,
//     trimmed, truncated to kMaxNormBytes bytes (deterministic).
//   * sketch(): trigrams of "\x02" + text + "\x03". For every trigram a 64-bit
//     FNV-1a hash seeds eight splitmix64 streams; the 512 sign bits are the
//     random projection. Bit b is 1 iff the accumulated sign is positive.
//   * dot512(): 2 * (number of equal bits) - 512, i.e. [-512, +512]; +512 is
//     identical sketches, 0 is the unrelated-string baseline.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <immintrin.h>

namespace hsaa_assoc {

constexpr int kDim = 512;
constexpr int kW = kDim / 64;
constexpr std::size_t kMaxNormBytes = 1024;

inline std::uint64_t mix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

inline std::uint64_t fnv1a3(const unsigned char* p) {
    std::uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < 3; ++i) {
        h ^= static_cast<std::uint64_t>(p[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

inline int popcount64(std::uint64_t x) {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(x));
#else
    return __builtin_popcountll(x);
#endif
}

// ASCII lowercase + whitespace collapse + trim. Non-ASCII bytes pass through
// unchanged (two strings equal byte-wise still normalize equal).
inline std::string normalize(const char* s) {
    std::string out;
    if (!s) return out;
    bool pending_space = false;
    for (const char* p = s; *p && out.size() < kMaxNormBytes; ++p) {
        unsigned char c = static_cast<unsigned char>(*p);
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            pending_space = !out.empty();
            continue;
        }
        if (c >= 'A' && c <= 'Z') c = static_cast<unsigned char>(c - 'A' + 'a');
        if (pending_space) {
            out.push_back(' ');
            pending_space = false;
        }
        out.push_back(static_cast<char>(c));
    }
    return out;
}

inline void sketch(const std::string& text, std::uint64_t out[kW]) {
    alignas(64) std::int32_t acc[kDim] = {};

    // Boundary markers make leading/trailing trigrams position-distinct.
    std::string padded;
    padded.reserve(text.size() + 2);
    padded.push_back('\x02');
    padded += text;
    padded.push_back('\x03');

    const unsigned char* p = reinterpret_cast<const unsigned char*>(padded.data());
    const std::size_t n = padded.size();

#if defined(__AVX512F__)
    const __m512i v_pos = _mm512_set1_epi32(1);
    const __m512i v_neg = _mm512_set1_epi32(-1);
    for (std::size_t i = 0; i + 3 <= n; ++i) {
        const std::uint64_t h = fnv1a3(p + i);
        for (int w = 0; w < kW; ++w) {
            const std::uint64_t r =
                mix64(h ^ (0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(w + 1)));
            std::int32_t* dst = &acc[w * 64];
            for (int g = 0; g < 4; ++g) {
                __mmask16 m = static_cast<__mmask16>((r >> (g * 16)) & 0xFFFFU);
                __m512i va = _mm512_load_si512(reinterpret_cast<const void*>(dst + g * 16));
                __m512i vd = _mm512_mask_mov_epi32(v_neg, m, v_pos);
                _mm512_store_si512(reinterpret_cast<void*>(dst + g * 16), _mm512_add_epi32(va, vd));
            }
        }
    }
    const __m512i v_zero = _mm512_setzero_si512();
    for (int w = 0; w < kW; ++w) {
        std::uint64_t bits = 0;
        const std::int32_t* src = &acc[w * 64];
        for (int g = 0; g < 4; ++g) {
            __m512i va = _mm512_load_si512(reinterpret_cast<const void*>(src + g * 16));
            __mmask16 m = _mm512_cmpgt_epi32_mask(va, v_zero);
            bits |= (static_cast<std::uint64_t>(m) << (g * 16));
        }
        out[w] = bits;
    }
#else
    for (std::size_t i = 0; i + 3 <= n; ++i) {
        const std::uint64_t h = fnv1a3(p + i);
        for (int w = 0; w < kW; ++w) {
            const std::uint64_t r =
                mix64(h ^ (0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(w + 1)));
            std::int32_t* dst = &acc[w * 64];
            for (int bit = 0; bit < 64; ++bit) {
                dst[bit] += ((r >> bit) & 1ULL) ? 1 : -1;
            }
        }
    }

    for (int w = 0; w < kW; ++w) out[w] = 0;
    for (int b = 0; b < kDim; ++b) {
        if (acc[b] > 0) out[b >> 6] |= (1ULL << (b & 63));
    }
#endif
}

inline std::int32_t dot512(const std::uint64_t a[kW], const std::uint64_t b[kW]) {
#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
    __m512i va = _mm512_loadu_si512(reinterpret_cast<const void*>(a));
    __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void*>(b));
    __m512i vxor = _mm512_xor_si512(va, vb);
    __m512i vpop = _mm512_popcnt_epi64(vxor);
    int hd = static_cast<int>(_mm512_reduce_add_epi64(vpop));
    return static_cast<std::int32_t>(kDim - 2 * hd);
#else
    int matches = 0;
    for (int w = 0; w < kW; ++w) {
        matches += popcount64(~(a[w] ^ b[w]));
    }
    return static_cast<std::int32_t>(matches) * 2 - kDim;
#endif
}

} // namespace hsaa_assoc

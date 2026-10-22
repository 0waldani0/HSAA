/*
 * HARD INVARIANTS (ANTI-WASHOUT PIN - NEVER VIOLATE):
 * 1. LANG: Code/files/bridge = concise EN. User chat = concise RU, zero fluff.
 * 2. HONESTY: Zero unmeasured claims. Brutal self/peer critique.
 * 3. STACK: C11 + AVX2/AVX512 intrinsics + CUDA Driver API (.cubin sm_86). Zero STL/bloat.
 * 4. MEMORY: Zero heap allocs on hot path. 64B-aligned VirtualAlloc Arena + MapViewOfFile.
 * 5. THINKING: 3-5 options + Hybrid synthesis for every arch/algo decision.
 */
#ifndef QUANT_CORE_H
#define QUANT_CORE_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#define ALIGN64(x) (((size_t)(x) + 63ULL) & ~63ULL)
#define BSL_BLOCK_WEIGHTS 128
#define BSL_MAX_PLANES    5    /* Supports 1..5 bit-serial planes on the fly */
#define BSL_MAX_COL_BLKS  256  /* Up to N = 32768 (256 * 128) in Arena */

/*
 * 3-Core / Low-Rank Tensor Train (TT) Outlier Residual:
 * Captures global cross-channel outliers in O((M + N) * R) before bit-plane quantization.
 */
typedef struct TtResidual {
    uint32_t M, N, R;         /* R = TT rank (e.g., 16) */
    float   *U;               /* [R * N] input projection core, 64B aligned */
    float   *V;               /* [R * M] col-major output core for SIMD 32-row tiles, 64B aligned */
} TtResidual;

/*
 * 32-Row-Tiled Bit-Serial LUT Container (.bsl32):
 * - planes_cpu[p]: [row_tiles][col_blocks][16 pairs][32 rows] of uint8_t (for AVX-512 vpermw / AVX2 vpshufb)
 * - planes_gpu[p]: [row_tiles][col_blocks][4 grp4 ][32 rows] of uint32_t (for GPU coalesced LDG.E.U32!)
 */
typedef struct BslTiledMatrix {
    uint32_t  M;              /* Rows (multiple of 32) */
    uint32_t  N;              /* Cols (multiple of 128) */
    uint32_t  num_planes;     /* 2..5 */
    uint32_t  row_tiles;      /* M / 32 */
    uint32_t  col_blocks;     /* N / 128 */
    float    *scales;         /* [row_tiles * col_blocks * 32], 64B aligned */
    float    *zeros;          /* [row_tiles * col_blocks * 32], 64B aligned */
    uint8_t  *planes[BSL_MAX_PLANES];     /* CPU SIMD layout: [pair=0..15][r_sub=0..31] */
    uint32_t *planes_u32[BSL_MAX_PLANES]; /* GPU U32 layout : [grp4=0..3 ][r_sub=0..31] (4 pairs/u32) */
} BslTiledMatrix;

/*
 * Pre-allocated Scratch Workspace in Arena (eliminates 64KB stack LUTs!).
 */
typedef struct BslWorkspace {
    __m512i *luts_avx512;     /* [BSL_MAX_COL_BLKS * 16] (128 KB in Arena) */
    __m256i *luts_avx2_lo;    /* [BSL_MAX_COL_BLKS * 16] */
    __m256i *luts_avx2_hi;    /* [BSL_MAX_COL_BLKS * 16] */
    float   *block_x_scale;   /* [BSL_MAX_COL_BLKS] */
    float   *block_x_sum;     /* [BSL_MAX_COL_BLKS] */
    float   *z_tt;            /* [64] */
} BslWorkspace;

/*
 * Hybrid B+C Memory Arena: VirtualAlloc reserve + 2MB chunked commit-on-demand (64B aligned).
 */
typedef struct MemArena {
    uint8_t *base;
    size_t   reserved;
    size_t   committed;
    size_t   offset;
} MemArena;

static inline int arena_init(MemArena *a, size_t reserve_bytes) {
    a->reserved  = ALIGN64(reserve_bytes);
    a->committed = 0;
    a->offset    = 0;
    a->base = (uint8_t*)VirtualAlloc(NULL, a->reserved, MEM_RESERVE, PAGE_NOACCESS);
    return a->base != NULL ? 0 : -1;
}

static inline void* arena_alloc(MemArena *a, size_t bytes) {
    size_t req = ALIGN64(bytes);
    size_t end = a->offset + req;
    if (end > a->reserved) {
        printf("[!] FATAL: Arena exhausted (req=%zu, reserved=%zu)\n", req, a->reserved);
        ExitProcess(1);
    }
    if (end > a->committed) {
        size_t commit_chunk = (end - a->committed + (2ULL << 20) - 1) & ~((2ULL << 20) - 1);
        if (a->committed + commit_chunk > a->reserved)
            commit_chunk = a->reserved - a->committed;
        if (!VirtualAlloc(a->base + a->committed, commit_chunk, MEM_COMMIT, PAGE_READWRITE)) {
            printf("[!] FATAL: VirtualAlloc MEM_COMMIT failed (%zu bytes)\n", commit_chunk);
            ExitProcess(1);
        }
        a->committed += commit_chunk;
    }
    void *ptr = a->base + a->offset;
    a->offset = end;
    return ptr;
}

static inline void arena_free(MemArena *a) {
    if (a->base) VirtualFree(a->base, 0, MEM_RELEASE);
    a->base = NULL;
}

/* ============================================================================
 * ZERO-COPY WIN32 MapViewOfFile .safetensors LOADER (M4 + O-AUDIT2 Hardened)
 * ============================================================================ */
typedef struct SafetensorView {
    char     name[96];
    char     dtype[8];        /* "BF16", "F16", "F32" */
    uint32_t ndim;            /* 1 or 2 */
    uint32_t shape[2];        /* [M, N] (for 1D: [M, 1]) */
    uint64_t data_start;
    uint64_t data_end;
    const void *data_ptr;     /* Direct zero-copy pointer into MapViewOfFile! */
} SafetensorView;

typedef struct SafetensorsFile {
    HANDLE   hFile;
    HANDLE   hMap;
    const uint8_t *map_base;
    size_t   file_size;
    uint64_t header_len;
    const uint8_t *data_base;
} SafetensorsFile;

static inline void safetensors_close(SafetensorsFile *sf) {
    if (sf->map_base) UnmapViewOfFile(sf->map_base);
    if (sf->hMap)     CloseHandle(sf->hMap);
    if (sf->hFile && sf->hFile != INVALID_HANDLE_VALUE) CloseHandle(sf->hFile);
    memset(sf, 0, sizeof(*sf));
}

static inline int safetensors_open(SafetensorsFile *sf, const char *path) {
    memset(sf, 0, sizeof(*sf));
    sf->hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (sf->hFile == INVALID_HANDLE_VALUE) return -1;
    LARGE_INTEGER fsz;
    if (!GetFileSizeEx(sf->hFile, &fsz) || fsz.QuadPart < 16) {
        safetensors_close(sf);
        return -2;
    }
    sf->file_size = (size_t)fsz.QuadPart;
    sf->hMap = CreateFileMappingA(sf->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!sf->hMap) { safetensors_close(sf); return -3; }
    sf->map_base = (const uint8_t*)MapViewOfFile(sf->hMap, FILE_MAP_READ, 0, 0, 0);
    if (!sf->map_base) { safetensors_close(sf); return -4; }
    sf->header_len = *(const uint64_t*)sf->map_base;
    if (sf->header_len == 0 || 8ULL + sf->header_len > sf->file_size) {
        safetensors_close(sf); /* Fixed O-AUDIT2 Pt 5: zero handle leak on -5! */
        return -5;
    }
    sf->data_base = sf->map_base + 8ULL + sf->header_len;
    return 0;
}

/* Zero-allocation lookup of 1D or 2D tensor by name inside mapped .safetensors JSON header */
static inline int safetensors_find(const SafetensorsFile *sf, const char *tensor_name, SafetensorView *out) {
    const char *json = (const char*)(sf->map_base + 8);
    size_t hlen = (size_t)sf->header_len;
    size_t nlen = strlen(tensor_name);
    size_t data_region_size = sf->file_size - (size_t)(8ULL + sf->header_len);
    for (size_t i = 0; i + nlen + 16 < hlen; ++i) {
        if (json[i] == '"' && memcmp(json + i + 1, tensor_name, nlen) == 0 && json[i + 1 + nlen] == '"') {
            memset(out, 0, sizeof(*out));
            memcpy(out->name, tensor_name, nlen < 95 ? nlen : 95);
            const char *p = json + i + 2 + nlen;
            const char *end = json + hlen;
            while (p + 10 < end && *p != '}') {
                if (memcmp(p, "\"dtype\"", 7) == 0) {
                    while (p < end && *p != ':') ++p;
                    while (p < end && *p != '"') ++p;
                    if (p < end) ++p;
                    int k = 0;
                    while (p < end && *p != '"' && k < 7) out->dtype[k++] = *p++;
                } else if (memcmp(p, "\"shape\"", 7) == 0) {
                    while (p < end && *p != '[') ++p;
                    if (p < end) ++p;
                    out->shape[0] = (uint32_t)strtoul(p, (char**)&p, 10);
                    out->shape[1] = 1;
                    out->ndim = 1;
                    while (p < end && *p == ' ') ++p;
                    if (p < end && *p == ',') {
                        ++p;
                        out->shape[1] = (uint32_t)strtoul(p, (char**)&p, 10);
                        out->ndim = 2;
                    }
                } else if (memcmp(p, "\"data_offsets\"", 14) == 0) {
                    while (p < end && *p != '[') ++p;
                    if (p < end) ++p;
                    out->data_start = strtoull(p, (char**)&p, 10);
                    while (p < end && (*p == ',' || *p == ' ')) ++p;
                    out->data_end   = strtoull(p, (char**)&p, 10);
                    if (out->data_start > out->data_end || out->data_end > data_region_size) {
                        return -2; /* Fixed O-AUDIT2 Pt 5: strict bounds check! */
                    }
                    out->data_ptr = sf->data_base + out->data_start;
                    return 0;
                }
                ++p;
            }
        }
    }
    return -1;
}

/* Fast Multi-Threaded AVX-512 BF16 -> FP32 converter (16 BF16 elements per ZMM shift) */
static inline void bf16_to_fp32_avx512(const uint16_t *src_bf16, float *dst_fp32, size_t count) {
    ptrdiff_t n_vec = (ptrdiff_t)(count & ~15ULL);
    ptrdiff_t idx;
    #pragma omp parallel for schedule(static) if(count >= 65536)
    for (idx = 0; idx < n_vec; idx += 16) {
        __m256i bf  = _mm256_loadu_si256((const __m256i*)(src_bf16 + idx));
        __m512i u32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf), 16);
        _mm512_storeu_ps(dst_fp32 + idx, _mm512_castsi512_ps(u32));
    }
    for (size_t i = (size_t)n_vec; i < count; ++i) {
        uint32_t bits = ((uint32_t)src_bf16[i]) << 16;
        memcpy(dst_fp32 + i, &bits, sizeof(float));
    }
}

/* Dtype-checked tensor loader from mapped .safetensors into Arena FP32 buffer */
static inline float* safetensors_load_fp32(
    const SafetensorsFile *sf, const char *name, MemArena *arena,
    uint32_t *out_rows, uint32_t *out_cols
) {
    SafetensorView v;
    if (safetensors_find(sf, name, &v) != 0) return NULL;
    size_t count = (size_t)v.shape[0] * (size_t)v.shape[1];
    size_t byte_len = (size_t)(v.data_end - v.data_start);
    float *dst = (float*)arena_alloc(arena, count * sizeof(float));
    if (strcmp(v.dtype, "BF16") == 0) {
        if (byte_len != count * sizeof(uint16_t)) return NULL;
        bf16_to_fp32_avx512((const uint16_t*)v.data_ptr, dst, count);
    } else if (strcmp(v.dtype, "F32") == 0) {
        if (byte_len != count * sizeof(float)) return NULL;
        memcpy(dst, v.data_ptr, byte_len);
    } else {
        return NULL;
    }
    if (out_rows) *out_rows = v.shape[0];
    if (out_cols) *out_cols = v.shape[1];
    return dst;
}

/* High-resolution Win32 QPC timer */
static inline double qpc_sec(void) {
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
}

#endif /* QUANT_CORE_H */

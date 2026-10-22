#include "bridge.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <immintrin.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#define CUDAAPI __stdcall
typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef unsigned long long CUdeviceptr;

#define CU_MEMHOSTREGISTER_PORTABLE  0x01
#define CU_MEMHOSTREGISTER_DEVICEMAP 0x02

typedef CUresult (CUDAAPI *pfn_cuInit)(unsigned int);
typedef CUresult (CUDAAPI *pfn_cuDeviceGet)(CUdevice*, int);
typedef CUresult (CUDAAPI *pfn_cuCtxCreate)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (CUDAAPI *pfn_cuCtxGetCurrent)(CUcontext*);
typedef CUresult (CUDAAPI *pfn_cuMemAlloc)(CUdeviceptr*, size_t);
typedef CUresult (CUDAAPI *pfn_cuMemFree)(CUdeviceptr);
typedef CUresult (CUDAAPI *pfn_cuMemAllocHost)(void**, size_t);
typedef CUresult (CUDAAPI *pfn_cuMemFreeHost)(void*);
typedef CUresult (CUDAAPI *pfn_cuMemHostRegister)(void*, size_t, unsigned int);
typedef CUresult (CUDAAPI *pfn_cuMemHostUnregister)(void*);
typedef CUresult (CUDAAPI *pfn_cuMemHostGetDevicePointer)(CUdeviceptr*, void*, unsigned int);
typedef CUresult (CUDAAPI *pfn_cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
typedef CUresult (CUDAAPI *pfn_cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
typedef CUresult (CUDAAPI *pfn_cuCtxSynchronize)(void);

static HMODULE                       s_cuda_lib                      = NULL;
static pfn_cuInit                    s_cuInit                        = NULL;
static pfn_cuDeviceGet               s_cuDeviceGet                   = NULL;
static pfn_cuCtxCreate               s_cuCtxCreate                   = NULL;
static pfn_cuCtxGetCurrent           s_cuCtxGetCurrent               = NULL;
static pfn_cuMemAlloc                s_cuMemAlloc                    = NULL;
static pfn_cuMemFree                 s_cuMemFree                     = NULL;
static pfn_cuMemAllocHost            s_cuMemAllocHost                = NULL;
static pfn_cuMemFreeHost             s_cuMemFreeHost                 = NULL;
static pfn_cuMemHostRegister         s_cuMemHostRegister             = NULL;
static pfn_cuMemHostUnregister       s_cuMemHostUnregister           = NULL;
static pfn_cuMemHostGetDevicePointer s_cuMemHostGetDevicePointer     = NULL;
static pfn_cuMemcpyHtoD              s_cuMemcpyHtoD                  = NULL;
static pfn_cuMemcpyDtoH              s_cuMemcpyDtoH                  = NULL;
static pfn_cuCtxSynchronize          s_cuCtxSynchronize              = NULL;

static int                           s_cuda_initialized              = 0;
static int                           s_cuda_available                = 0;
static CUcontext                     s_cuda_ctx                      = NULL;

static double                        g_last_transfer_latency_us      = 0.0;
static LARGE_INTEGER                 s_perf_freq                     = {0};

static void init_cuda_driver(void) {
    if (s_cuda_initialized) return;
    s_cuda_initialized = 1;

    QueryPerformanceFrequency(&s_perf_freq);

    s_cuda_lib = LoadLibraryA("nvcuda.dll");
    if (!s_cuda_lib) {
        s_cuda_available = 0;
        return;
    }

    #define BIND_SYM(name, sym) \
        s_##name = (pfn_##name)GetProcAddress(s_cuda_lib, sym); \
        if (!s_##name) { FreeLibrary(s_cuda_lib); s_cuda_lib = NULL; s_cuda_available = 0; return; }

    BIND_SYM(cuInit,            "cuInit")
    BIND_SYM(cuDeviceGet,       "cuDeviceGet")
    BIND_SYM(cuCtxCreate,       "cuCtxCreate_v2")
    BIND_SYM(cuCtxGetCurrent,   "cuCtxGetCurrent")
    BIND_SYM(cuMemAlloc,        "cuMemAlloc_v2")
    BIND_SYM(cuMemFree,         "cuMemFree_v2")
    BIND_SYM(cuMemAllocHost,    "cuMemAllocHost_v2")
    BIND_SYM(cuMemFreeHost,     "cuMemFreeHost")
    BIND_SYM(cuMemcpyHtoD,      "cuMemcpyHtoD_v2")
    BIND_SYM(cuMemcpyDtoH,      "cuMemcpyDtoH_v2")
    BIND_SYM(cuCtxSynchronize,  "cuCtxSynchronize")
    #undef BIND_SYM

    s_cuMemHostRegister = (pfn_cuMemHostRegister)GetProcAddress(s_cuda_lib, "cuMemHostRegister_v2");
    s_cuMemHostUnregister = (pfn_cuMemHostUnregister)GetProcAddress(s_cuda_lib, "cuMemHostUnregister");
    s_cuMemHostGetDevicePointer = (pfn_cuMemHostGetDevicePointer)GetProcAddress(s_cuda_lib, "cuMemHostGetDevicePointer_v2");

    if (s_cuInit(0) != 0) {
        s_cuda_available = 0;
        return;
    }

    CUcontext current_ctx = NULL;
    if (s_cuCtxGetCurrent(&current_ctx) == 0 && current_ctx != NULL) {
        s_cuda_ctx = current_ctx;
        s_cuda_available = 1;
        return;
    }

    CUdevice dev = 0;
    if (s_cuDeviceGet(&dev, 0) != 0) {
        s_cuda_available = 0;
        return;
    }

    if (s_cuCtxCreate(&s_cuda_ctx, 0, dev) != 0) {
        s_cuda_available = 0;
        return;
    }

    s_cuda_available = 1;
}

int bridge_is_cuda_available(void) {
    init_cuda_driver();
    return s_cuda_available;
}

double bridge_last_transfer_latency_us(void) {
    return g_last_transfer_latency_us;
}

HybridBuffer* bridge_alloc_buffer(size_t hidden_dim) {
    if (hidden_dim == 0) {
        hidden_dim = HIDDEN_DIM;
    }
    if (hidden_dim > HIDDEN_DIM) {
        return NULL;
    }

    init_cuda_driver();

    // Allocate 64KB page-aligned cacheable host memory
    HybridBuffer* buf = (HybridBuffer*)VirtualAlloc(NULL, sizeof(HybridBuffer), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!buf) return NULL;

    memset(buf, 0, sizeof(HybridBuffer));
    buf->bytes_allocated = sizeof(HybridBuffer);
    buf->is_pinned = false;
    buf->is_zero_copy = false;
    buf->d_hidden_gpu = NULL;

    if (s_cuda_available) {
        // Register page-locked pinned memory mapped for direct GPU Zero-Copy DMA access
        if (s_cuMemHostRegister) {
            CUresult rc = s_cuMemHostRegister(buf, sizeof(HybridBuffer), CU_MEMHOSTREGISTER_DEVICEMAP);
            if (rc == 0) {
                buf->is_pinned = true;
                CUdeviceptr d_ptr = 0;
                if (s_cuMemHostGetDevicePointer &&
                    s_cuMemHostGetDevicePointer(&d_ptr, buf->hidden_cpu, 0) == 0 && d_ptr != 0) {
                    buf->d_hidden_gpu = (void*)(uintptr_t)d_ptr;
                    buf->is_zero_copy = true;
                }
            }
        }

        // Fallback to discrete device memory if host registration didn't map device pointer
        if (!buf->d_hidden_gpu && s_cuMemAlloc) {
            CUdeviceptr d_ptr = 0;
            if (s_cuMemAlloc(&d_ptr, hidden_dim * sizeof(float)) == 0 && d_ptr != 0) {
                buf->d_hidden_gpu = (void*)(uintptr_t)d_ptr;
                buf->is_zero_copy = false;
            }
        }
    }

    return buf;
}

void bridge_free_buffer(HybridBuffer* buf) {
    if (!buf) return;

    if (buf->is_pinned && s_cuda_available && s_cuMemHostUnregister) {
        s_cuMemHostUnregister(buf);
        buf->is_pinned = false;
    }

    if (buf->d_hidden_gpu && !buf->is_zero_copy && s_cuda_available && s_cuMemFree) {
        s_cuMemFree((CUdeviceptr)(uintptr_t)buf->d_hidden_gpu);
        buf->d_hidden_gpu = NULL;
    }

    VirtualFree(buf, 0, MEM_RELEASE);
}

int bridge_sync_vsa_to_gpu(HybridBuffer* buf, const uint64_t* vsa_vec, float scale) {
    if (!buf) return -1;
    if (!vsa_vec) return -2;
    if (scale == 0.0f) scale = 1.0f;

    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);

    memcpy(buf->vsa_vec, vsa_vec, sizeof(buf->vsa_vec));

#if defined(__AVX512F__)
    __m512 v_pos = _mm512_set1_ps(scale);
    __m512 v_neg = _mm512_set1_ps(-scale);

    for (size_t c = 0; c < 32; ++c) {
        size_t word_idx = c / 4;
        size_t shift = (c % 4) * 16;
        uint16_t k = (uint16_t)((vsa_vec[word_idx] >> shift) & 0xFFFF);

        __m512 val0 = _mm512_mask_blend_ps((__mmask16)k, v_neg, v_pos);
        _mm512_store_ps(&buf->hidden_cpu[c * 16], val0);

        __m512 val1 = _mm512_mask_blend_ps((__mmask16)k, v_pos, v_neg);
        _mm512_store_ps(&buf->hidden_cpu[512 + c * 16], val1);
    }
#else
    for (size_t i = 0; i < 512; ++i) {
        size_t w = i / 64;
        size_t b = i % 64;
        bool bit = (vsa_vec[w] >> b) & 1ULL;
        buf->hidden_cpu[i] = bit ? scale : -scale;
        buf->hidden_cpu[512 + i] = bit ? -scale : scale;
    }
#endif

    // If zero-copy mapped, serialize store buffer so GPU DMA engine sees updated cache lines
    if (buf->is_zero_copy) {
        _mm_sfence();
    } else if (buf->d_hidden_gpu && s_cuda_available && s_cuMemcpyHtoD) {
        // Discrete device buffer: execute PCIe DMA transfer
        CUresult rc = s_cuMemcpyHtoD(
            (CUdeviceptr)(uintptr_t)buf->d_hidden_gpu,
            buf->hidden_cpu,
            HIDDEN_DIM * sizeof(float)
        );
        if (rc != 0) return -3;
    }

    QueryPerformanceCounter(&t1);
    if (s_perf_freq.QuadPart > 0) {
        g_last_transfer_latency_us = (double)(t1.QuadPart - t0.QuadPart) * 1000000.0 / (double)s_perf_freq.QuadPart;
    }

    return 0;
}

int bridge_sync_gpu_to_vsa(HybridBuffer* buf, uint64_t* out_vsa_vec) {
    if (!buf) return -1;
    if (!out_vsa_vec) return -2;

    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);

    // If zero-copy mapped, serialize memory load barrier
    if (buf->is_zero_copy) {
        _mm_lfence();
    } else if (buf->d_hidden_gpu && s_cuda_available && s_cuMemcpyDtoH) {
        // Discrete device buffer: retrieve PCIe DMA transfer
        CUresult rc = s_cuMemcpyDtoH(
            buf->hidden_cpu,
            (CUdeviceptr)(uintptr_t)buf->d_hidden_gpu,
            HIDDEN_DIM * sizeof(float)
        );
        if (rc != 0) return -3;
    }

#if defined(__AVX512F__)
    for (size_t w = 0; w < VSA_WORDS; ++w) {
        uint64_t word = 0;
        for (size_t chunk = 0; chunk < 4; ++chunk) {
            size_t c = w * 4 + chunk;
            __m512 v_lo = _mm512_load_ps(&buf->hidden_cpu[c * 16]);
            __m512 v_hi = _mm512_load_ps(&buf->hidden_cpu[512 + c * 16]);
            __mmask16 m = _mm512_cmp_ps_mask(v_lo, v_hi, _CMP_GE_OQ);
            word |= ((uint64_t)(uint16_t)m) << (chunk * 16);
        }
        out_vsa_vec[w] = word;
        buf->vsa_vec[w] = word;
    }
#else
    for (size_t w = 0; w < VSA_WORDS; ++w) {
        uint64_t word = 0;
        for (size_t b = 0; b < 64; ++b) {
            size_t idx = w * 64 + b;
            if (buf->hidden_cpu[idx] >= buf->hidden_cpu[512 + idx]) {
                word |= (1ULL << b);
            }
        }
        out_vsa_vec[w] = word;
        buf->vsa_vec[w] = word;
    }
#endif

    QueryPerformanceCounter(&t1);
    if (s_perf_freq.QuadPart > 0) {
        g_last_transfer_latency_us = (double)(t1.QuadPart - t0.QuadPart) * 1000000.0 / (double)s_perf_freq.QuadPart;
    }

    return 0;
}

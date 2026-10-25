#ifndef PTQ_CUDA_H
#define PTQ_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define CUDAAPI __stdcall
#else
typedef void* HMODULE;
#define CUDAAPI
#endif
#include <stdint.h>
#include <stddef.h>
typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef unsigned long long CUdeviceptr;

/* Function pointer prototypes matching CUDA Driver API */
typedef CUresult (CUDAAPI *pfn_cuInit)(unsigned int);
typedef CUresult (CUDAAPI *pfn_cuDeviceGet)(CUdevice*, int);
typedef CUresult (CUDAAPI *pfn_cuDeviceGetName)(char*, int, CUdevice);
typedef CUresult (CUDAAPI *pfn_cuCtxCreate)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (CUDAAPI *pfn_cuCtxDestroy)(CUcontext);
typedef CUresult (CUDAAPI *pfn_cuMemGetInfo)(size_t*, size_t*);
typedef CUresult (CUDAAPI *pfn_cuMemAlloc)(CUdeviceptr*, size_t);
typedef CUresult (CUDAAPI *pfn_cuMemFree)(CUdeviceptr);
typedef CUresult (CUDAAPI *pfn_cuMemAllocHost)(void**, size_t);
typedef CUresult (CUDAAPI *pfn_cuMemFreeHost)(void*);
typedef CUresult (CUDAAPI *pfn_cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
typedef CUresult (CUDAAPI *pfn_cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
typedef CUresult (CUDAAPI *pfn_cuMemsetD32)(CUdeviceptr, unsigned int, size_t);
typedef CUresult (CUDAAPI *pfn_cuModuleLoad)(CUmodule*, const char*);
typedef CUresult (CUDAAPI *pfn_cuModuleUnload)(CUmodule);
typedef CUresult (CUDAAPI *pfn_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
typedef CUresult (CUDAAPI *pfn_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                                               unsigned int, unsigned int, unsigned int,
                                               unsigned int, CUstream, void**, void**);
typedef CUresult (CUDAAPI *pfn_cuCtxSynchronize)(void);

typedef struct CudaDrv {
    HMODULE                 lib;
    pfn_cuInit              cuInit;
    pfn_cuDeviceGet         cuDeviceGet;
    pfn_cuDeviceGetName     cuDeviceGetName;
    pfn_cuCtxCreate         cuCtxCreate;
    pfn_cuCtxDestroy        cuCtxDestroy;
    pfn_cuMemGetInfo        cuMemGetInfo;
    pfn_cuMemAlloc          cuMemAlloc;
    pfn_cuMemFree           cuMemFree;
    pfn_cuMemAllocHost      cuMemAllocHost;
    pfn_cuMemFreeHost       cuMemFreeHost;
    pfn_cuMemcpyHtoD        cuMemcpyHtoD;
    pfn_cuMemcpyDtoH        cuMemcpyDtoH;
    pfn_cuMemsetD32         cuMemsetD32;
    pfn_cuModuleLoad        cuModuleLoad;
    pfn_cuModuleUnload      cuModuleUnload;
    pfn_cuModuleGetFunction cuModuleGetFunction;
    pfn_cuLaunchKernel      cuLaunchKernel;
    pfn_cuCtxSynchronize    cuCtxSynchronize;
} CudaDrv;

typedef struct PtqKernels {
    CUmodule   mod;
    CUfunction fn_gemv;      /* "bsl_lut_gemv_sm86" */
    CUfunction fn_tt_u;      /* "tt_project_u_sm86" */
    CUfunction fn_rmsnorm;   /* "rmsnorm_res_sm86" */
    CUfunction fn_qk_rope;   /* "qknorm_rope_kv_sm86" */
    CUfunction fn_gqa;       /* "gqa_attn_sm86" */
    CUfunction fn_swiglu;    /* "swiglu_sm86" */
} PtqKernels;

typedef struct PtqGpuMatrix {
    uint32_t    M;              /* Rows (multiple of 32) */
    uint32_t    N;              /* Cols (multiple of 128) */
    uint32_t    R;              /* TT Rank (e.g. 16, or 0) */
    CUdeviceptr d_p[4];         /* 4 bit-planes in GPU U32 layout [M/32 * N/128 * 128 * 4 bytes] */
    CUdeviceptr d_sc;           /* Scales [M/32 * N/128 * 32 * sizeof(float)] */
    CUdeviceptr d_zr;           /* Zeros [M/32 * N/128 * 32 * sizeof(float)] */
    CUdeviceptr d_tt_u;         /* TT input core U [R * N * sizeof(float)] */
    CUdeviceptr d_tt_v;         /* TT output core V [R * M * sizeof(float)] */
    CUdeviceptr d_z_tt;         /* TT projected intermediate [R * sizeof(float)] */
} PtqGpuMatrix;

typedef struct PtqGpuWorkspace {
    CUdeviceptr d_h_cur;        /* Current hidden vector [H * sizeof(float)] */
    CUdeviceptr d_x_norm;       /* Post-RMSNorm buffer [H * sizeof(float)] */
    CUdeviceptr d_qkv_buf;      /* QKV projection buffer [(qd + 2*kd) * sizeof(float)] */
    CUdeviceptr d_attn_out;     /* Attention output [qd * sizeof(float)] */
    CUdeviceptr d_o_buf;        /* Output projection [H * sizeof(float)] */
    CUdeviceptr d_h_norm;       /* Post-attention RMSNorm [H * sizeof(float)] */
    CUdeviceptr d_gu_buf;       /* Gate/Up MLP buffer [2*I * sizeof(float)] */
    CUdeviceptr d_mlp_act;      /* SwiGLU activation [I * sizeof(float)] */
    CUdeviceptr d_down_buf;     /* Down projection [H * sizeof(float)] */
    CUdeviceptr d_h_fin;        /* Final normalized state [H * sizeof(float)] */
    CUdeviceptr d_logits;       /* Final logits [V * sizeof(float)] */
    CUdeviceptr d_part[4];      /* Split-K partials [8 * max_rows * sizeof(float)] */
    CUdeviceptr d_cnt[4];       /* Atomic tile counters [max_tiles * sizeof(uint32_t)] */
    CUdeviceptr d_k_cache;      /* KV cache K [n_layers * max_t * kd * sizeof(float)] */
    CUdeviceptr d_v_cache;      /* KV cache V [n_layers * max_t * kd * sizeof(float)] */
    uint32_t    H;              /* Hidden dim */
    uint32_t    I;              /* Intermediate dim */
    uint32_t    max_rows;
    uint32_t    max_tiles;
} PtqGpuWorkspace;

/* Driver and Module Lifecycle */
int  ptq_cuda_init(CudaDrv *cu, CUcontext *ctx, int device_id);
void ptq_cuda_close(CudaDrv *cu, CUcontext ctx);

int  ptq_kernels_load(CudaDrv *cu, PtqKernels *k, const char *cubin_path);
void ptq_kernels_unload(CudaDrv *cu, PtqKernels *k);

/* GPU Buffer Management */
int  ptq_matrix_upload(
    CudaDrv *cu, PtqGpuMatrix *gpu_mat,
    uint32_t M, uint32_t N, uint32_t R,
    const uint32_t *const planes_u32[4],
    const float *scales, const float *zeros,
    const float *tt_u, const float *tt_v
);
void ptq_matrix_free(CudaDrv *cu, PtqGpuMatrix *gpu_mat);

int  ptq_workspace_alloc(
    CudaDrv *cu, PtqGpuWorkspace *ws,
    uint32_t H, uint32_t I, uint32_t qd, uint32_t kd, uint32_t V,
    uint32_t n_layers, uint32_t max_t
);
void ptq_workspace_free(CudaDrv *cu, PtqGpuWorkspace *ws);

/* Kernel Execution */
uint32_t ptq_pick_split_k(uint32_t cols, uint32_t target_gy);

int ptq_gemv_sm86(
    CudaDrv *cu, const PtqKernels *k,
    const PtqGpuMatrix *mat,
    CUdeviceptr d_x,
    CUdeviceptr d_y,
    CUdeviceptr d_partials,
    CUdeviceptr d_tile_cnt,
    uint32_t active_planes,
    uint32_t split_k,
    CUstream stream
);

int ptq_tt_project_u_sm86(
    CudaDrv *cu, const PtqKernels *k,
    CUdeviceptr d_tt_u,
    CUdeviceptr d_x,
    CUdeviceptr d_z_tt,
    uint32_t R,
    uint32_t cols,
    CUstream stream
);

#ifdef __cplusplus
}
#endif

#endif /* PTQ_CUDA_H */

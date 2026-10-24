#include "ptq_cuda.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int ptq_cuda_init(CudaDrv *c, CUcontext *ctx, int device_id) {
    if (!c || !ctx) return -1;
    memset(c, 0, sizeof(*c));
    c->lib = LoadLibraryA("nvcuda.dll");
    if (!c->lib) return -1;

    #define LOAD_SYM(name, sym) \
        c->name = (pfn_##name)GetProcAddress(c->lib, sym); \
        if (!c->name) { FreeLibrary(c->lib); memset(c, 0, sizeof(*c)); return -2; }

    LOAD_SYM(cuInit,              "cuInit")
    LOAD_SYM(cuDeviceGet,         "cuDeviceGet")
    LOAD_SYM(cuDeviceGetName,     "cuDeviceGetName")
    LOAD_SYM(cuCtxCreate,         "cuCtxCreate_v2")
    LOAD_SYM(cuCtxDestroy,        "cuCtxDestroy_v2")
    LOAD_SYM(cuMemGetInfo,        "cuMemGetInfo_v2")
    LOAD_SYM(cuMemAlloc,          "cuMemAlloc_v2")
    LOAD_SYM(cuMemFree,           "cuMemFree_v2")
    LOAD_SYM(cuMemAllocHost,      "cuMemAllocHost_v2")
    LOAD_SYM(cuMemFreeHost,       "cuMemFreeHost")
    LOAD_SYM(cuMemcpyHtoD,        "cuMemcpyHtoD_v2")
    LOAD_SYM(cuMemcpyDtoH,        "cuMemcpyDtoH_v2")
    LOAD_SYM(cuMemsetD32,         "cuMemsetD32_v2")
    LOAD_SYM(cuModuleLoad,        "cuModuleLoad")
    LOAD_SYM(cuModuleUnload,      "cuModuleUnload")
    LOAD_SYM(cuModuleGetFunction, "cuModuleGetFunction")
    LOAD_SYM(cuLaunchKernel,      "cuLaunchKernel")
    LOAD_SYM(cuCtxSynchronize,    "cuCtxSynchronize")
    #undef LOAD_SYM

    if (c->cuInit(0) != 0) {
        FreeLibrary(c->lib);
        memset(c, 0, sizeof(*c));
        return -3;
    }
    CUdevice dev = 0;
    if (c->cuDeviceGet(&dev, device_id) != 0) {
        FreeLibrary(c->lib);
        memset(c, 0, sizeof(*c));
        return -4;
    }
    if (c->cuCtxCreate(ctx, 0, dev) != 0) {
        FreeLibrary(c->lib);
        memset(c, 0, sizeof(*c));
        return -5;
    }
    return 0;
}

void ptq_cuda_close(CudaDrv *cu, CUcontext ctx) {
    if (!cu) return;
    if (ctx && cu->cuCtxDestroy) {
        cu->cuCtxDestroy(ctx);
    }
    if (cu->lib) {
        FreeLibrary(cu->lib);
    }
    memset(cu, 0, sizeof(*cu));
}

int ptq_kernels_load(CudaDrv *cu, PtqKernels *k, const char *cubin_path) {
    if (!cu || !k || !cubin_path) return -1;
    memset(k, 0, sizeof(*k));
    if (cu->cuModuleLoad(&k->mod, cubin_path) != 0) return -1;
    if (cu->cuModuleGetFunction(&k->fn_gemv,    k->mod, "bsl_lut_gemv_sm86") != 0) return -2;
    if (cu->cuModuleGetFunction(&k->fn_tt_u,    k->mod, "tt_project_u_sm86") != 0) return -3;
    if (cu->cuModuleGetFunction(&k->fn_rmsnorm, k->mod, "rmsnorm_res_sm86") != 0) return -4;
    if (cu->cuModuleGetFunction(&k->fn_qk_rope, k->mod, "qknorm_rope_kv_sm86") != 0) return -5;
    if (cu->cuModuleGetFunction(&k->fn_gqa,     k->mod, "gqa_attn_sm86") != 0) return -6;
    if (cu->cuModuleGetFunction(&k->fn_swiglu,  k->mod, "swiglu_sm86") != 0) return -7;
    return 0;
}

void ptq_kernels_unload(CudaDrv *cu, PtqKernels *k) {
    if (!k) return;
    if (k->mod && cu && cu->cuModuleUnload) {
        cu->cuModuleUnload(k->mod);
    }
    memset(k, 0, sizeof(*k));
}

int ptq_matrix_upload(
    CudaDrv *cu, PtqGpuMatrix *gpu_mat,
    uint32_t M, uint32_t N, uint32_t R,
    const uint32_t *const planes_u32[4],
    const float *scales, const float *zeros,
    const float *tt_u, const float *tt_v
) {
    if (!cu || !gpu_mat) return -1;
    memset(gpu_mat, 0, sizeof(*gpu_mat));
    gpu_mat->M = M;
    gpu_mat->N = N;
    gpu_mat->R = R;

    uint32_t row_tiles  = M / 32;
    uint32_t col_blocks = N / 128;
    size_t n_blks = (size_t)row_tiles * col_blocks;
    size_t plane_sz = n_blks * 512;
    size_t scale_sz = n_blks * 32 * sizeof(float);

    for (uint32_t p = 0; p < 4; ++p) {
        if (cu->cuMemAlloc(&gpu_mat->d_p[p], plane_sz) != 0) {
            ptq_matrix_free(cu, gpu_mat);
            return -2;
        }
        if (planes_u32 && planes_u32[p]) {
            cu->cuMemcpyHtoD(gpu_mat->d_p[p], planes_u32[p], plane_sz);
        }
    }

    if (cu->cuMemAlloc(&gpu_mat->d_sc, scale_sz) != 0) {
        ptq_matrix_free(cu, gpu_mat);
        return -3;
    }
    if (scales) {
        cu->cuMemcpyHtoD(gpu_mat->d_sc, scales, scale_sz);
    }

    if (cu->cuMemAlloc(&gpu_mat->d_zr, scale_sz) != 0) {
        ptq_matrix_free(cu, gpu_mat);
        return -4;
    }
    if (zeros) {
        cu->cuMemcpyHtoD(gpu_mat->d_zr, zeros, scale_sz);
    }

    if (R > 0) {
        size_t u_sz = (size_t)R * N * sizeof(float);
        size_t v_sz = (size_t)R * M * sizeof(float);
        size_t z_sz = (size_t)R * sizeof(float);

        if (cu->cuMemAlloc(&gpu_mat->d_tt_u, u_sz) != 0 ||
            cu->cuMemAlloc(&gpu_mat->d_tt_v, v_sz) != 0 ||
            cu->cuMemAlloc(&gpu_mat->d_z_tt, z_sz) != 0) {
            ptq_matrix_free(cu, gpu_mat);
            return -5;
        }
        if (tt_u) cu->cuMemcpyHtoD(gpu_mat->d_tt_u, tt_u, u_sz);
        if (tt_v) cu->cuMemcpyHtoD(gpu_mat->d_tt_v, tt_v, v_sz);
    }

    return 0;
}

void ptq_matrix_free(CudaDrv *cu, PtqGpuMatrix *gpu_mat) {
    if (!cu || !gpu_mat) return;
    for (uint32_t p = 0; p < 4; ++p) {
        if (gpu_mat->d_p[p]) {
            cu->cuMemFree(gpu_mat->d_p[p]);
            gpu_mat->d_p[p] = 0;
        }
    }
    if (gpu_mat->d_sc) { cu->cuMemFree(gpu_mat->d_sc); gpu_mat->d_sc = 0; }
    if (gpu_mat->d_zr) { cu->cuMemFree(gpu_mat->d_zr); gpu_mat->d_zr = 0; }
    if (gpu_mat->d_tt_u) { cu->cuMemFree(gpu_mat->d_tt_u); gpu_mat->d_tt_u = 0; }
    if (gpu_mat->d_tt_v) { cu->cuMemFree(gpu_mat->d_tt_v); gpu_mat->d_tt_v = 0; }
    if (gpu_mat->d_z_tt) { cu->cuMemFree(gpu_mat->d_z_tt); gpu_mat->d_z_tt = 0; }
    memset(gpu_mat, 0, sizeof(*gpu_mat));
}

int ptq_workspace_alloc(
    CudaDrv *cu, PtqGpuWorkspace *ws,
    uint32_t H, uint32_t I, uint32_t qd, uint32_t kd, uint32_t V,
    uint32_t n_layers, uint32_t max_t
) {
    if (!cu || !ws) return -1;
    memset(ws, 0, sizeof(*ws));
    ws->H = H;
    ws->I = I;

    if (cu->cuMemAlloc(&ws->d_h_cur,   H * sizeof(float)) != 0 ||
        cu->cuMemAlloc(&ws->d_x_norm,  H * sizeof(float)) != 0 ||
        cu->cuMemAlloc(&ws->d_qkv_buf, (size_t)(qd + 2 * kd) * sizeof(float)) != 0 ||
        cu->cuMemAlloc(&ws->d_attn_out, qd * sizeof(float)) != 0 ||
        cu->cuMemAlloc(&ws->d_o_buf,   H * sizeof(float)) != 0 ||
        cu->cuMemAlloc(&ws->d_h_norm,  H * sizeof(float)) != 0 ||
        cu->cuMemAlloc(&ws->d_gu_buf,  2ULL * I * sizeof(float)) != 0 ||
        cu->cuMemAlloc(&ws->d_mlp_act, I * sizeof(float)) != 0 ||
        cu->cuMemAlloc(&ws->d_down_buf, H * sizeof(float)) != 0 ||
        cu->cuMemAlloc(&ws->d_h_fin,   H * sizeof(float)) != 0 ||
        cu->cuMemAlloc(&ws->d_logits,  V * sizeof(float)) != 0 ||
        cu->cuMemAlloc(&ws->d_k_cache, (size_t)n_layers * max_t * kd * sizeof(float)) != 0 ||
        cu->cuMemAlloc(&ws->d_v_cache, (size_t)n_layers * max_t * kd * sizeof(float)) != 0) {
        ptq_workspace_free(cu, ws);
        return -2;
    }

    uint32_t max_rows = (2 * I > (qd + 2 * kd)) ? (2 * I) : (qd + 2 * kd);
    if (H > max_rows) max_rows = H;
    uint32_t max_tiles = (max_rows + 31) / 32;
    ws->max_rows = max_rows;
    ws->max_tiles = max_tiles;

    for (int m = 0; m < 4; ++m) {
        if (cu->cuMemAlloc(&ws->d_part[m], 8ULL * max_rows * sizeof(float)) != 0 ||
            cu->cuMemAlloc(&ws->d_cnt[m],  max_tiles * sizeof(uint32_t)) != 0) {
            ptq_workspace_free(cu, ws);
            return -3;
        }
        cu->cuMemsetD32(ws->d_cnt[m], 0, max_tiles);
    }

    return 0;
}

void ptq_workspace_free(CudaDrv *cu, PtqGpuWorkspace *ws) {
    if (!cu || !ws) return;
    if (ws->d_h_cur)   { cu->cuMemFree(ws->d_h_cur);   ws->d_h_cur = 0; }
    if (ws->d_x_norm)  { cu->cuMemFree(ws->d_x_norm);  ws->d_x_norm = 0; }
    if (ws->d_qkv_buf) { cu->cuMemFree(ws->d_qkv_buf); ws->d_qkv_buf = 0; }
    if (ws->d_attn_out){ cu->cuMemFree(ws->d_attn_out);ws->d_attn_out = 0; }
    if (ws->d_o_buf)   { cu->cuMemFree(ws->d_o_buf);   ws->d_o_buf = 0; }
    if (ws->d_h_norm)  { cu->cuMemFree(ws->d_h_norm);  ws->d_h_norm = 0; }
    if (ws->d_gu_buf)  { cu->cuMemFree(ws->d_gu_buf);  ws->d_gu_buf = 0; }
    if (ws->d_mlp_act) { cu->cuMemFree(ws->d_mlp_act); ws->d_mlp_act = 0; }
    if (ws->d_down_buf){ cu->cuMemFree(ws->d_down_buf);ws->d_down_buf = 0; }
    if (ws->d_h_fin)   { cu->cuMemFree(ws->d_h_fin);   ws->d_h_fin = 0; }
    if (ws->d_logits)  { cu->cuMemFree(ws->d_logits);  ws->d_logits = 0; }
    if (ws->d_k_cache) { cu->cuMemFree(ws->d_k_cache); ws->d_k_cache = 0; }
    if (ws->d_v_cache) { cu->cuMemFree(ws->d_v_cache); ws->d_v_cache = 0; }
    for (int m = 0; m < 4; ++m) {
        if (ws->d_part[m]) { cu->cuMemFree(ws->d_part[m]); ws->d_part[m] = 0; }
        if (ws->d_cnt[m])  { cu->cuMemFree(ws->d_cnt[m]);  ws->d_cnt[m] = 0; }
    }
    memset(ws, 0, sizeof(*ws));
}

uint32_t ptq_pick_split_k(uint32_t cols, uint32_t target_gy) {
    uint32_t cb = cols >> 7;
    for (uint32_t g = target_gy; g >= 1; --g) {
        if (cb % g == 0) return g;
    }
    return 1u;
}

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
) {
    if (!cu || !k || !mat) return -1;
    uint32_t M = mat->M;
    uint32_t N = mat->N;
    uint32_t R = mat->R;

    /* The kernel tiles rows by 32 and columns by 128. Anything else used to be
     * truncated silently (cols >> 7, rows >> 5), producing a partial product.
     * The fused TT epilogue also stages at most 16 ranks in shared memory. */
    if ((N & 127u) != 0u || (M & 31u) != 0u) return -2;
    if (((R >> 7) & 1u) != 0u && (R & 0x7Fu) > 16u) return -2;

    uint32_t gy = (split_k > 0) ? split_k : ptq_pick_split_k(N, 8);
    uint32_t gx = (M + 255) / 256;

    uint32_t rows_arg = M, cols_arg = N, rank_arg = R, planes_arg = active_planes;
    CUdeviceptr d_tt_v = mat->d_tt_v;
    CUdeviceptr d_z_tt = mat->d_z_tt;

    void *args[] = {
        (void*)&mat->d_p[0], (void*)&mat->d_p[1], (void*)&mat->d_p[2], (void*)&mat->d_p[3],
        (void*)&mat->d_sc, (void*)&mat->d_zr, (void*)&d_x,
        (void*)&d_tt_v, (void*)&d_z_tt, (void*)&d_partials, (void*)&d_tile_cnt, (void*)&d_y,
        (void*)&rows_arg, (void*)&cols_arg, (void*)&rank_arg, (void*)&planes_arg
    };

    CUresult res = cu->cuLaunchKernel(
        k->fn_gemv,
        gx, gy, 1,
        256, 1, 1,
        0,
        stream,
        args,
        NULL
    );
    return (res == 0) ? 0 : (int)res;
}

int ptq_tt_project_u_sm86(
    CudaDrv *cu, const PtqKernels *k,
    CUdeviceptr d_tt_u,
    CUdeviceptr d_x,
    CUdeviceptr d_z_tt,
    uint32_t R,
    uint32_t cols,
    CUstream stream
) {
    if (!cu || !k || R == 0) return -1;
    uint32_t cols_arg = cols;
    void *args[] = { (void*)&d_tt_u, (void*)&d_x, (void*)&d_z_tt, (void*)&cols_arg };
    CUresult res = cu->cuLaunchKernel(
        k->fn_tt_u,
        R, 1, 1,
        256, 1, 1,
        0,
        stream,
        args,
        NULL
    );
    return (res == 0) ? 0 : (int)res;
}

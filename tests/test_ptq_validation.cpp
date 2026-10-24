// test_ptq_validation.cpp -- negative-path checks for the PTQ host wrapper.
//
// The sm_86 kernel tiles rows by 32 and columns by 128, and the fused TT
// epilogue stages at most 16 ranks in shared memory. Before the remediation the
// host wrapper truncated these values silently, so a wrong shape produced a
// partial product instead of an error. These cases assert the rejection happens
// before cuLaunchKernel is reached: the wrapper is passed zeroed (but non-null)
// driver/kernel structs, and returns -2 without dereferencing them.

#include "ptq_cuda.h"

#include <cstdio>
#include <cstring>

static int g_run = 0;
static int g_failed = 0;

static void check(bool cond, const char* msg) {
    ++g_run;
    if (cond) {
        std::printf("[PASS] %s\n", msg);
    } else {
        std::printf("[FAIL] %s\n", msg);
        ++g_failed;
    }
}

int main() {
    std::printf("=== PTQ host validation (/arch:sm_86 tiling contract) ===\n");

    CudaDrv cu;
    PtqKernels kern;
    PtqGpuMatrix mat;
    std::memset(&cu, 0, sizeof(cu));
    std::memset(&kern, 0, sizeof(kern));
    std::memset(&mat, 0, sizeof(mat));

    check(ptq_gemv_sm86(nullptr, &kern, &mat, 0, 0, 0, 0, 1, 0, 0) == -1,
          "null driver rejected");

    mat.N = 128;
    mat.M = 32;
    mat.R = 0;

    mat.N = 127;
    check(ptq_gemv_sm86(&cu, &kern, &mat, 0, 0, 0, 0, 1, 0, 0) == -2,
          "N % 128 != 0 rejected before launch");

    mat.N = 128;
    mat.M = 31;
    check(ptq_gemv_sm86(&cu, &kern, &mat, 0, 0, 0, 0, 1, 0, 0) == -2,
          "M % 32 != 0 rejected before launch");

    mat.M = 32;
    mat.R = (1u << 7) | 17u; // fused epilogue, requested rank 17
    check(ptq_gemv_sm86(&cu, &kern, &mat, 0, 0, 0, 0, 1, 0, 0) == -2,
          "fused rank > 16 rejected before launch");

    mat.R = 16; // non-fused rank 16 is legal; don't launch with zeroed structs
    mat.N = 256;
    mat.M = 64;
    check(ptq_tt_project_u_sm86(&cu, &kern, 0, 0, 0, 0, 128, 0) == -1,
          "TT projection with R == 0 rejected");

    std::printf("=== %d run, %d failed ===\n", g_run, g_failed);
    return g_failed == 0 ? 0 : 1;
}

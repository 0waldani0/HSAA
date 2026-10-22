#ifndef BRIDGE_H
#define BRIDGE_H

#pragma once
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

#define VSA_DIM 512
#define VSA_WORDS (VSA_DIM / 64)       // 8 x uint64_t = 64 bytes (1 cache line / 1 __m512i)
#define HIDDEN_DIM 1024
#define TENSOR_ALIGNMENT 64

struct HybridBuffer {
    alignas(64) uint64_t vsa_vec[VSA_WORDS];       // 512-bit binary hypervector
    alignas(64) float    hidden_cpu[HIDDEN_DIM];    // 1024-float pinned host buffer
    void*                d_hidden_gpu;              // CUdeviceptr on RTX 3050
    size_t               bytes_allocated;
    bool                 is_pinned;
    bool                 is_zero_copy;
};

// Allocate and register page-locked pinned memory for zero-copy / fast DMA
HybridBuffer* bridge_alloc_buffer(size_t hidden_dim);
void bridge_free_buffer(HybridBuffer* buf);

// Forward cycle transfer: inject AVX-512 VSA hypervector into GPU hidden state (<2 us)
int bridge_sync_vsa_to_gpu(HybridBuffer* buf, const uint64_t* vsa_vec, float scale);

// Retrieve GPU hidden state back to CPU for associative retrieval (<2 us)
int bridge_sync_gpu_to_vsa(HybridBuffer* buf, uint64_t* out_vsa_vec);

// Bridge status and profiling
int bridge_is_cuda_available(void);
double bridge_last_transfer_latency_us(void);

#ifdef __cplusplus
}
#endif

#endif // BRIDGE_H

#include "bridge.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>
#include <cstring>
#include <cmath>
#include <windows.h>

static uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static int count_hamming_distance(const uint64_t* a, const uint64_t* b, size_t words) {
    int dist = 0;
    for (size_t i = 0; i < words; ++i) {
        uint64_t diff = a[i] ^ b[i];
        while (diff) {
            diff &= (diff - 1);
            dist++;
        }
    }
    return dist;
}

static void test_memory_alignment() {
    std::cout << "[*] Running test_memory_alignment..." << std::endl;

    HybridBuffer* buf = bridge_alloc_buffer(HIDDEN_DIM);
    assert(buf != nullptr);

    uintptr_t addr_buf = reinterpret_cast<uintptr_t>(buf);
    uintptr_t addr_vsa = reinterpret_cast<uintptr_t>(buf->vsa_vec);
    uintptr_t addr_cpu = reinterpret_cast<uintptr_t>(buf->hidden_cpu);

    std::cout << "    HybridBuffer base address: 0x" << std::hex << addr_buf << std::dec << std::endl;
    std::cout << "    vsa_vec address:           0x" << std::hex << addr_vsa << std::dec << std::endl;
    std::cout << "    hidden_cpu address:        0x" << std::hex << addr_cpu << std::dec << std::endl;
    std::cout << "    sizeof(HybridBuffer):      " << sizeof(HybridBuffer) << " bytes" << std::endl;

    assert((addr_buf % 64) == 0 && "HybridBuffer pointer must be 64-byte aligned");
    assert((addr_vsa % 64) == 0 && "vsa_vec must be 64-byte aligned");
    assert((addr_cpu % 64) == 0 && "hidden_cpu must be 64-byte aligned");
    assert(sizeof(HybridBuffer) % 64 == 0 && "sizeof(HybridBuffer) must be multiple of 64");
    assert(buf->bytes_allocated == sizeof(HybridBuffer));

    int cuda_avail = bridge_is_cuda_available();
    std::cout << "    CUDA Driver available:     " << (cuda_avail ? "YES (RTX 3050)" : "NO (CPU Fallback)") << std::endl;
    if (cuda_avail) {
        assert(buf->is_pinned && "Buffer should be page-locked when CUDA is available");
        assert(buf->d_hidden_gpu != nullptr && "d_hidden_gpu should be allocated on CUDA");
    }

    bridge_free_buffer(buf);
    std::cout << "    [PASS] test_memory_alignment" << std::endl;
}

static void test_roundtrip_bit_fidelity() {
    std::cout << "[*] Running test_roundtrip_bit_fidelity..." << std::endl;

    HybridBuffer* buf = bridge_alloc_buffer(HIDDEN_DIM);
    assert(buf != nullptr);

    uint64_t rng_state = 0x123456789ABCDEF0ULL;
    const float scales[] = { 0.1f, 0.5f, 1.0f, 2.5f, 10.0f, 100.0f };
    const int num_vectors = 500;

    for (float scale : scales) {
        for (int v = 0; v < num_vectors; ++v) {
            uint64_t orig_vec[VSA_WORDS];
            for (size_t w = 0; w < VSA_WORDS; ++w) {
                orig_vec[w] = xorshift64(&rng_state);
            }

            int rc_forward = bridge_sync_vsa_to_gpu(buf, orig_vec, scale);
            assert(rc_forward == 0 && "bridge_sync_vsa_to_gpu failed");

            uint64_t recovered_vec[VSA_WORDS] = {0};
            int rc_reverse = bridge_sync_gpu_to_vsa(buf, recovered_vec);
            assert(rc_reverse == 0 && "bridge_sync_gpu_to_vsa failed");

            int dist = count_hamming_distance(orig_vec, recovered_vec, VSA_WORDS);
            if (dist != 0) {
                std::cerr << "    [FAIL] Hamming distance mismatch at scale " << scale << ": " << dist << std::endl;
                assert(dist == 0 && "Bit-exact roundtrip failed");
            }

            assert(std::memcmp(orig_vec, recovered_vec, sizeof(orig_vec)) == 0);
            assert(std::memcmp(orig_vec, buf->vsa_vec, sizeof(orig_vec)) == 0);
        }
    }

    bridge_free_buffer(buf);
    std::cout << "    [PASS] test_roundtrip_bit_fidelity (3000 vectors across 6 scales, 0 bit errors)" << std::endl;
}

static void test_differential_properties() {
    std::cout << "[*] Running test_differential_properties..." << std::endl;

    HybridBuffer* buf = bridge_alloc_buffer(HIDDEN_DIM);
    assert(buf != nullptr);

    uint64_t rng_state = 0xFEEDFACECAFEBEEFULL;
    uint64_t orig_vec[VSA_WORDS];
    for (size_t w = 0; w < VSA_WORDS; ++w) {
        orig_vec[w] = xorshift64(&rng_state);
    }

    float scale = 1.0f;
    int rc = bridge_sync_vsa_to_gpu(buf, orig_vec, scale);
    assert(rc == 0);

    // 1. Verify zero-mean property: sum(h[0..1023]) == 0.0f
    double sum = 0.0;
    for (size_t i = 0; i < HIDDEN_DIM; ++i) {
        sum += buf->hidden_cpu[i];
    }
    std::cout << "    Zero-mean sum across 1024 floats: " << sum << std::endl;
    assert(std::abs(sum) < 1e-4 && "Differential encoding must have zero mean");

    // 2. Verify common-mode DC noise cancellation:
    // Adding arbitrary constant offset +42.0f to all floats must not change the recovered bits!
    for (size_t i = 0; i < HIDDEN_DIM; ++i) {
        buf->hidden_cpu[i] += 42.0f;
    }

    // Call reverse sync without GPU DtoH overwriting the corrupted CPU buffer
    // (If GPU is active, temporarily clear d_hidden_gpu to test CPU differential rejection directly)
    void* saved_d_hidden = buf->d_hidden_gpu;
    buf->d_hidden_gpu = nullptr;

    uint64_t out_vec[VSA_WORDS] = {0};
    rc = bridge_sync_gpu_to_vsa(buf, out_vec);
    assert(rc == 0);

    buf->d_hidden_gpu = saved_d_hidden;

    int dist = count_hamming_distance(orig_vec, out_vec, VSA_WORDS);
    std::cout << "    Hamming distance after +42.0f DC common-mode noise: " << dist << std::endl;
    assert(dist == 0 && "Common-mode noise must be completely rejected by differential encoding");

    bridge_free_buffer(buf);
    std::cout << "    [PASS] test_differential_properties" << std::endl;
}

static void test_adversarial_boundaries() {
    std::cout << "[*] Running test_adversarial_boundaries (0-1-N-NULL-BOUND)..." << std::endl;

    // NULL buffer checks
    uint64_t dummy_vec[VSA_WORDS] = {0};
    assert(bridge_sync_vsa_to_gpu(nullptr, dummy_vec, 1.0f) == -1);
    assert(bridge_sync_gpu_to_vsa(nullptr, dummy_vec) == -1);
    bridge_free_buffer(nullptr); // Must safely handle NULL

    // NULL vector checks
    HybridBuffer* buf = bridge_alloc_buffer(HIDDEN_DIM);
    assert(buf != nullptr);
    assert(bridge_sync_vsa_to_gpu(buf, nullptr, 1.0f) == -2);
    assert(bridge_sync_gpu_to_vsa(buf, nullptr) == -2);

    // Dimension bounds
    HybridBuffer* buf_zero = bridge_alloc_buffer(0);
    assert(buf_zero != nullptr && "bridge_alloc_buffer(0) should default to HIDDEN_DIM");
    bridge_free_buffer(buf_zero);

    HybridBuffer* buf_overflow = bridge_alloc_buffer(1025);
    assert(buf_overflow == nullptr && "bridge_alloc_buffer(>HIDDEN_DIM) must return NULL");

    HybridBuffer* buf_huge = bridge_alloc_buffer(999999);
    assert(buf_huge == nullptr && "bridge_alloc_buffer(huge) must return NULL");

    // Scale 0.0f auto-fallback
    int rc = bridge_sync_vsa_to_gpu(buf, dummy_vec, 0.0f);
    assert(rc == 0 && "scale 0.0f should be accepted and defaulted to 1.0f");

    // All-zero vector boundary
    uint64_t all_zeros[VSA_WORDS] = {0};
    rc = bridge_sync_vsa_to_gpu(buf, all_zeros, 1.0f);
    assert(rc == 0);
    uint64_t out_zeros[VSA_WORDS] = {1};
    rc = bridge_sync_gpu_to_vsa(buf, out_zeros);
    assert(rc == 0);
    assert(std::memcmp(all_zeros, out_zeros, sizeof(all_zeros)) == 0);

    // All-ones vector boundary
    uint64_t all_ones[VSA_WORDS];
    std::memset(all_ones, 0xFF, sizeof(all_ones));
    rc = bridge_sync_vsa_to_gpu(buf, all_ones, 1.0f);
    assert(rc == 0);
    uint64_t out_ones[VSA_WORDS] = {0};
    rc = bridge_sync_gpu_to_vsa(buf, out_ones);
    assert(rc == 0);
    assert(std::memcmp(all_ones, out_ones, sizeof(all_ones)) == 0);

    // Alternating patterns
    uint64_t alt_a[VSA_WORDS], alt_b[VSA_WORDS];
    for (size_t w = 0; w < VSA_WORDS; ++w) {
        alt_a[w] = 0xAAAAAAAAAAAAAAAAULL;
        alt_b[w] = 0x5555555555555555ULL;
    }
    rc = bridge_sync_vsa_to_gpu(buf, alt_a, 1.0f);
    assert(rc == 0);
    uint64_t out_alt[VSA_WORDS] = {0};
    rc = bridge_sync_gpu_to_vsa(buf, out_alt);
    assert(rc == 0);
    assert(std::memcmp(alt_a, out_alt, sizeof(alt_a)) == 0);

    rc = bridge_sync_vsa_to_gpu(buf, alt_b, 1.0f);
    assert(rc == 0);
    rc = bridge_sync_gpu_to_vsa(buf, out_alt);
    assert(rc == 0);
    assert(std::memcmp(alt_b, out_alt, sizeof(alt_b)) == 0);

    // 1-hot vector test across all 512 bits
    for (int bit = 0; bit < 512; ++bit) {
        uint64_t one_hot[VSA_WORDS] = {0};
        one_hot[bit / 64] = (1ULL << (bit % 64));

        rc = bridge_sync_vsa_to_gpu(buf, one_hot, 1.0f);
        assert(rc == 0);
        uint64_t out_hot[VSA_WORDS] = {0};
        rc = bridge_sync_gpu_to_vsa(buf, out_hot);
        assert(rc == 0);
        assert(count_hamming_distance(one_hot, out_hot, VSA_WORDS) == 0);
    }

    bridge_free_buffer(buf);
    std::cout << "    [PASS] test_adversarial_boundaries" << std::endl;
}

static void test_latency_benchmark() {
    std::cout << "[*] Running test_latency_benchmark (2000 iterations)..." << std::endl;

    HybridBuffer* buf = bridge_alloc_buffer(HIDDEN_DIM);
    assert(buf != nullptr);

    uint64_t rng_state = 0xCAFEBABE12345678ULL;
    uint64_t test_vec[VSA_WORDS];
    for (size_t w = 0; w < VSA_WORDS; ++w) {
        test_vec[w] = xorshift64(&rng_state);
    }

    // Warm-up
    for (int i = 0; i < 50; ++i) {
        bridge_sync_vsa_to_gpu(buf, test_vec, 1.0f);
        bridge_sync_gpu_to_vsa(buf, test_vec);
    }

    const int iterations = 2000;
    LARGE_INTEGER freq, t_start, t_end;
    QueryPerformanceFrequency(&freq);

    // Benchmark forward transfer (AVX-512 unpack + PCIe HtoD)
    QueryPerformanceCounter(&t_start);
    for (int i = 0; i < iterations; ++i) {
        bridge_sync_vsa_to_gpu(buf, test_vec, 1.0f);
    }
    QueryPerformanceCounter(&t_end);
    double forward_total_us = (double)(t_end.QuadPart - t_start.QuadPart) * 1000000.0 / (double)freq.QuadPart;
    double forward_avg_us = forward_total_us / iterations;

    // Benchmark reverse transfer (PCIe DtoH + AVX-512 pack)
    QueryPerformanceCounter(&t_start);
    for (int i = 0; i < iterations; ++i) {
        bridge_sync_gpu_to_vsa(buf, test_vec);
    }
    QueryPerformanceCounter(&t_end);
    double reverse_total_us = (double)(t_end.QuadPart - t_start.QuadPart) * 1000000.0 / (double)freq.QuadPart;
    double reverse_avg_us = reverse_total_us / iterations;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "    Forward transfer average latency: " << forward_avg_us << " us / step" << std::endl;
    std::cout << "    Reverse transfer average latency: " << reverse_avg_us << " us / step" << std::endl;
    std::cout << "    Last reported step latency:       " << bridge_last_transfer_latency_us() << " us" << std::endl;

    assert(forward_avg_us < 2.0 && "Forward transfer latency must be < 2.0 microseconds");
    assert(reverse_avg_us < 2.0 && "Reverse transfer latency must be < 2.0 microseconds");

    bridge_free_buffer(buf);
    std::cout << "    [PASS] test_latency_benchmark: Sub-2us requirement verified!" << std::endl;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "   HYBRID CPU-GPU INTEROP BRIDGE TEST HARNESS (M1)        " << std::endl;
    std::cout << "==========================================================" << std::endl;

    test_memory_alignment();
    test_roundtrip_bit_fidelity();
    test_differential_properties();
    test_adversarial_boundaries();
    test_latency_benchmark();

    std::cout << "==========================================================" << std::endl;
    std::cout << "   ALL BRIDGE TESTS PASSED CLEANLY (exit code 0)          " << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}

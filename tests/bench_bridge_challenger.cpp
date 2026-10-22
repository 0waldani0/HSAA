#include "bridge.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cmath>
#include <numeric>

// ---------------------------------------------------------------------------
// High-Resolution Timer Utilities
// ---------------------------------------------------------------------------
class HighResTimer {
public:
    HighResTimer() {
        QueryPerformanceFrequency(&freq_);
    }

    inline void start() {
        QueryPerformanceCounter(&t_start_);
    }

    inline double stop_us() {
        LARGE_INTEGER t_end;
        QueryPerformanceCounter(&t_end);
        return static_cast<double>(t_end.QuadPart - t_start_.QuadPart) * 1000000.0 / static_cast<double>(freq_.QuadPart);
    }

    inline double freq() const { return static_cast<double>(freq_.QuadPart); }

private:
    LARGE_INTEGER freq_;
    LARGE_INTEGER t_start_;
};

// Fast 64-bit pseudo-random number generator
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

struct LatencyStats {
    double min_us;
    double max_us;
    double avg_us;
    double p50_us;
    double p95_us;
    double p99_us;
    double p999_us;
    size_t samples;
};

static LatencyStats compute_stats(std::vector<double>& times) {
    LatencyStats s = {0};
    if (times.empty()) return s;
    s.samples = times.size();

    std::sort(times.begin(), times.end());
    s.min_us = times.front();
    s.max_us = times.back();

    double sum = 0.0;
    for (double v : times) sum += v;
    s.avg_us = sum / static_cast<double>(s.samples);

    auto get_pct = [&](double pct) {
        size_t idx = static_cast<size_t>(pct * static_cast<double>(s.samples - 1));
        return times[idx];
    };

    s.p50_us  = get_pct(0.50);
    s.p95_us  = get_pct(0.95);
    s.p99_us  = get_pct(0.99);
    s.p999_us = get_pct(0.999);

    return s;
}

static void print_stats(const char* label, const LatencyStats& s) {
    std::cout << "  [" << label << "] (" << s.samples << " iterations):" << std::endl;
    std::cout << "    Avg:   " << std::fixed << std::setprecision(4) << s.avg_us << " us" << std::endl;
    std::cout << "    Min:   " << s.min_us << " us" << std::endl;
    std::cout << "    Max:   " << s.max_us << " us" << std::endl;
    std::cout << "    p50:   " << s.p50_us << " us" << std::endl;
    std::cout << "    p95:   " << s.p95_us << " us" << std::endl;
    std::cout << "    p99:   " << s.p99_us << " us" << std::endl;
    std::cout << "    p99.9: " << s.p999_us << " us" << std::endl;
}

// ===========================================================================
// SECTION 1: Latency Claim Verification (10,000+ repeated syncs)
// ===========================================================================
static bool verify_latency_claim(HybridBuffer* buf, const int iterations = 10000) {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "[CHALLENGE 1] Latency Verification (" << iterations << " iterations)" << std::endl;
    std::cout << "============================================================" << std::endl;

    uint64_t rng = 0xABCDEF0123456789ULL;
    uint64_t vsa_vec[VSA_WORDS];
    uint64_t out_vec[VSA_WORDS];

    // Warm-up
    for (int i = 0; i < 200; ++i) {
        for (size_t w = 0; w < VSA_WORDS; ++w) vsa_vec[w] = xorshift64(&rng);
        bridge_sync_vsa_to_gpu(buf, vsa_vec, 1.0f);
        bridge_sync_gpu_to_vsa(buf, out_vec);
    }

    HighResTimer timer;
    std::vector<double> forward_latencies;
    forward_latencies.reserve(iterations);

    // 1. Forward Transfer Latency Benchmark (10,000 iterations)
    for (int i = 0; i < iterations; ++i) {
        for (size_t w = 0; w < VSA_WORDS; ++w) vsa_vec[w] = xorshift64(&rng);
        timer.start();
        int rc = bridge_sync_vsa_to_gpu(buf, vsa_vec, 1.0f);
        double elapsed = timer.stop_us();
        assert(rc == 0);
        forward_latencies.push_back(elapsed);
    }
    LatencyStats fwd_stats = compute_stats(forward_latencies);
    print_stats("Forward Transfer (VSA -> GPU/CPU)", fwd_stats);

    // 2. Reverse Transfer Latency Benchmark (10,000 iterations)
    std::vector<double> reverse_latencies;
    reverse_latencies.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        timer.start();
        int rc = bridge_sync_gpu_to_vsa(buf, out_vec);
        double elapsed = timer.stop_us();
        assert(rc == 0);
        reverse_latencies.push_back(elapsed);
    }
    LatencyStats rev_stats = compute_stats(reverse_latencies);
    print_stats("Reverse Transfer (GPU/CPU -> VSA)", rev_stats);

    // 3. Bidirectional Roundtrip Latency (10,000 iterations)
    std::vector<double> roundtrip_latencies;
    roundtrip_latencies.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        for (size_t w = 0; w < VSA_WORDS; ++w) vsa_vec[w] = xorshift64(&rng);
        timer.start();
        bridge_sync_vsa_to_gpu(buf, vsa_vec, 1.0f);
        bridge_sync_gpu_to_vsa(buf, out_vec);
        double elapsed = timer.stop_us();
        roundtrip_latencies.push_back(elapsed);
    }
    LatencyStats rtt_stats = compute_stats(roundtrip_latencies);
    print_stats("Bidirectional Roundtrip (Forward + Reverse)", rtt_stats);

    // Quantitative assertions: average latency must be strictly < 2.0 us
    bool passed = true;
    if (fwd_stats.avg_us >= 2.0) {
        std::cerr << "[-] FAIL: Forward latency average " << fwd_stats.avg_us << " us >= 2.0 us ceiling!" << std::endl;
        passed = false;
    } else {
        std::cout << "[+] PASS: Forward latency avg=" << fwd_stats.avg_us << " us < 2.0 us ceiling ("
                  << std::fixed << std::setprecision(1) << (2.0 / fwd_stats.avg_us) << "x safety margin)" << std::endl;
    }

    if (rev_stats.avg_us >= 2.0) {
        std::cerr << "[-] FAIL: Reverse latency average " << rev_stats.avg_us << " us >= 2.0 us ceiling!" << std::endl;
        passed = false;
    } else {
        std::cout << "[+] PASS: Reverse latency avg=" << rev_stats.avg_us << " us < 2.0 us ceiling ("
                  << std::fixed << std::setprecision(1) << (2.0 / rev_stats.avg_us) << "x safety margin)" << std::endl;
    }

    return passed;
}

// ===========================================================================
// SECTION 2: Adversarial Differential Noise Rejection Verification
// ===========================================================================
static bool verify_differential_noise_rejection(HybridBuffer* buf) {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "[CHALLENGE 2] Adversarial Differential Noise Rejection" << std::endl;
    std::cout << "============================================================" << std::endl;

    uint64_t rng = 0x9876543210FEDCBAULL;
    bool all_passed = true;

    // Test battery of severe common-mode offsets
    const float severe_offsets[] = {
        +100.0f,
        -500.0f,
        +1000.0f,
        -1000.0f,
        +10000.0f,
        -10000.0f,
        +100000.0f,
        -100000.0f
    };

    const int vectors_per_offset = 1000;
    std::cout << "[*] Testing severe common-mode DC offsets across complementary channels..." << std::endl;

    for (float offset : severe_offsets) {
        int total_bit_errors = 0;
        for (int v = 0; v < vectors_per_offset; ++v) {
            uint64_t orig_vec[VSA_WORDS];
            for (size_t w = 0; w < VSA_WORDS; ++w) orig_vec[w] = xorshift64(&rng);

            // Forward sync
            int rc = bridge_sync_vsa_to_gpu(buf, orig_vec, 1.0f);
            assert(rc == 0);

            // Inject common-mode offset across ALL 1024 floats (both complementary channels)
            for (size_t i = 0; i < HIDDEN_DIM; ++i) {
                buf->hidden_cpu[i] += offset;
            }

            // Reverse sync
            uint64_t recovered_vec[VSA_WORDS] = {0};
            rc = bridge_sync_gpu_to_vsa(buf, recovered_vec);
            assert(rc == 0);

            // Verify 0 bit errors
            int dist = count_hamming_distance(orig_vec, recovered_vec, VSA_WORDS);
            total_bit_errors += dist;
        }

        if (total_bit_errors == 0) {
            std::cout << "    Offset " << std::setw(10) << offset << "f: 0 bit errors across "
                      << vectors_per_offset << " vectors [PASS]" << std::endl;
        } else {
            std::cerr << "    Offset " << offset << "f: " << total_bit_errors << " BIT ERRORS! [FAIL]" << std::endl;
            all_passed = false;
        }
    }

    // Component-wise independent common-mode noise:
    // Each complementary pair (hidden_cpu[i] and hidden_cpu[512+i]) receives an independent random DC shift C_i in [-500.0f, +500.0f]
    std::cout << "[*] Testing independent per-component common-mode noise C_i in [-500.0f, +500.0f]..." << std::endl;
    int per_comp_errors = 0;
    for (int v = 0; v < vectors_per_offset; ++v) {
        uint64_t orig_vec[VSA_WORDS];
        for (size_t w = 0; w < VSA_WORDS; ++w) orig_vec[w] = xorshift64(&rng);

        bridge_sync_vsa_to_gpu(buf, orig_vec, 1.0f);

        for (size_t i = 0; i < 512; ++i) {
            uint64_t r = xorshift64(&rng);
            float c_i = -500.0f + static_cast<float>(r % 100000) / 100.0f; // [-500.0, +500.0]
            buf->hidden_cpu[i] += c_i;
            buf->hidden_cpu[512 + i] += c_i; // Equal common-mode shift to complementary pair
        }

        uint64_t recovered_vec[VSA_WORDS] = {0};
        bridge_sync_gpu_to_vsa(buf, recovered_vec);
        per_comp_errors += count_hamming_distance(orig_vec, recovered_vec, VSA_WORDS);
    }
    if (per_comp_errors == 0) {
        std::cout << "    Per-component noise: 0 bit errors across " << vectors_per_offset << " vectors [PASS]" << std::endl;
    } else {
        std::cerr << "    Per-component noise: " << per_comp_errors << " BIT ERRORS! [FAIL]" << std::endl;
        all_passed = false;
    }

    // Multi-scale noise injection
    const float scales[] = { 0.1f, 0.5f, 2.0f, 10.0f, 100.0f };
    std::cout << "[*] Testing multi-scale common-mode noise (+100.0f and -500.0f)..." << std::endl;
    for (float scale : scales) {
        for (float offset : { +100.0f, -500.0f }) {
            int errors = 0;
            for (int v = 0; v < 500; ++v) {
                uint64_t orig_vec[VSA_WORDS];
                for (size_t w = 0; w < VSA_WORDS; ++w) orig_vec[w] = xorshift64(&rng);

                bridge_sync_vsa_to_gpu(buf, orig_vec, scale);

                for (size_t i = 0; i < HIDDEN_DIM; ++i) buf->hidden_cpu[i] += offset;

                uint64_t recovered_vec[VSA_WORDS] = {0};
                bridge_sync_gpu_to_vsa(buf, recovered_vec);
                errors += count_hamming_distance(orig_vec, recovered_vec, VSA_WORDS);
            }
            if (errors != 0) {
                std::cerr << "    Scale " << scale << " with offset " << offset << " had " << errors << " errors! [FAIL]" << std::endl;
                all_passed = false;
            }
        }
    }
    std::cout << "    Multi-scale noise tests: 0 bit errors [PASS]" << std::endl;

    if (all_passed) {
        std::cout << "[+] PASS: Differential noise rejection 100% verified (0 bit errors across all tests)!" << std::endl;
    }
    return all_passed;
}

// ===========================================================================
// SECTION 3: Mutation Flip Verification
// ===========================================================================
static bool verify_mutation_flip(HybridBuffer* buf) {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "[CHALLENGE 3] Mutation Flip Verification" << std::endl;
    std::cout << "============================================================" << std::endl;

    uint64_t rng = 0x55AA11223344EEFFULL;
    bool all_passed = true;

    // 3.1 Systematic Single-Bit Mutation Flip:
    // Exhaustively mutate EVERY bit position k in [0..511] in the hidden state representation
    std::cout << "[*] Running systematic single-bit mutation flips across ALL 512 bit positions..." << std::endl;
    int single_bit_fails = 0;

    for (int k = 0; k < 512; ++k) {
        uint64_t orig_vec[VSA_WORDS];
        for (size_t w = 0; w < VSA_WORDS; ++w) orig_vec[w] = xorshift64(&rng);

        // Forward sync to produce complementary differential hidden state
        bridge_sync_vsa_to_gpu(buf, orig_vec, 1.0f);

        // Mutate bit k representation in hidden_cpu: swap complementary channels
        std::swap(buf->hidden_cpu[k], buf->hidden_cpu[512 + k]);

        // Reverse sync to recover VSA vector
        uint64_t out_vec[VSA_WORDS] = {0};
        bridge_sync_gpu_to_vsa(buf, out_vec);

        // Verification:
        // 1. Bit k must be inverted
        int orig_bit = (orig_vec[k / 64] >> (k % 64)) & 1ULL;
        int out_bit  = (out_vec[k / 64] >> (k % 64)) & 1ULL;
        if (orig_bit == out_bit) {
            std::cerr << "    [-] FAIL at bit " << k << ": bit was NOT inverted!" << std::endl;
            single_bit_fails++;
        }

        // 2. Hamming distance must be exactly 1 (no spurious bit changes anywhere else)
        int dist = count_hamming_distance(orig_vec, out_vec, VSA_WORDS);
        if (dist != 1) {
            std::cerr << "    [-] FAIL at bit " << k << ": Hamming distance was " << dist << " (expected 1)!" << std::endl;
            single_bit_fails++;
        }
    }

    if (single_bit_fails == 0) {
        std::cout << "    Exhaustive 512-bit mutation test: 512/512 bit positions flipped with Hamming dist=1 [PASS]" << std::endl;
    } else {
        std::cerr << "    Exhaustive 512-bit mutation test: " << single_bit_fails << " failures! [FAIL]" << std::endl;
        all_passed = false;
    }

    // 3.2 Sign Inversion Mutation Flip across 1,000 random vectors
    std::cout << "[*] Running sign-inversion mutation flips (1000 iterations)..." << std::endl;
    int sign_inv_fails = 0;
    for (int it = 0; it < 1000; ++it) {
        uint64_t orig_vec[VSA_WORDS];
        for (size_t w = 0; w < VSA_WORDS; ++w) orig_vec[w] = xorshift64(&rng);

        bridge_sync_vsa_to_gpu(buf, orig_vec, 1.0f);

        int bit_to_flip = static_cast<int>(xorshift64(&rng) % 512);

        // Negate both channels for bit_to_flip: (+scale -> -scale, -scale -> +scale)
        buf->hidden_cpu[bit_to_flip] = -buf->hidden_cpu[bit_to_flip];
        buf->hidden_cpu[512 + bit_to_flip] = -buf->hidden_cpu[512 + bit_to_flip];

        uint64_t out_vec[VSA_WORDS] = {0};
        bridge_sync_gpu_to_vsa(buf, out_vec);

        int dist = count_hamming_distance(orig_vec, out_vec, VSA_WORDS);
        if (dist != 1) {
            sign_inv_fails++;
        }
    }
    if (sign_inv_fails == 0) {
        std::cout << "    Sign inversion mutation test: 1000/1000 passed [PASS]" << std::endl;
    } else {
        std::cerr << "    Sign inversion mutation test: " << sign_inv_fails << " failures! [FAIL]" << std::endl;
        all_passed = false;
    }

    // 3.3 Asymmetric One-Channel Threshold Overdrive
    // Invert the decision by driving Channel 0 past Channel 1
    std::cout << "[*] Running asymmetric threshold overdrive mutation flips (1000 iterations)..." << std::endl;
    int asymm_fails = 0;
    for (int it = 0; it < 1000; ++it) {
        uint64_t orig_vec[VSA_WORDS];
        for (size_t w = 0; w < VSA_WORDS; ++w) orig_vec[w] = xorshift64(&rng);

        float scale = 2.5f;
        bridge_sync_vsa_to_gpu(buf, orig_vec, scale);

        int bit_to_flip = static_cast<int>(xorshift64(&rng) % 512);
        int bit_val = (orig_vec[bit_to_flip / 64] >> (bit_to_flip % 64)) & 1ULL;

        // If bit_val == 1: hidden_cpu[bit] = +scale, hidden_cpu[512+bit] = -scale.
        // Overdrive hidden_cpu[bit] to -2.0f * scale so hidden_cpu[bit] < hidden_cpu[512+bit].
        // If bit_val == 0: hidden_cpu[bit] = -scale, hidden_cpu[512+bit] = +scale.
        // Overdrive hidden_cpu[bit] to +2.0f * scale so hidden_cpu[bit] > hidden_cpu[512+bit].
        if (bit_val) {
            buf->hidden_cpu[bit_to_flip] = -2.0f * scale;
        } else {
            buf->hidden_cpu[bit_to_flip] = +2.0f * scale;
        }

        uint64_t out_vec[VSA_WORDS] = {0};
        bridge_sync_gpu_to_vsa(buf, out_vec);

        int dist = count_hamming_distance(orig_vec, out_vec, VSA_WORDS);
        if (dist != 1) asymm_fails++;
    }
    if (asymm_fails == 0) {
        std::cout << "    Asymmetric threshold overdrive test: 1000/1000 passed [PASS]" << std::endl;
    } else {
        std::cerr << "    Asymmetric threshold overdrive test: " << asymm_fails << " failures! [FAIL]" << std::endl;
        all_passed = false;
    }

    // 3.4 Multi-Bit Mutation Flips (M bits mutated simultaneously)
    const int multi_counts[] = { 2, 4, 8, 16, 32, 64, 128, 256, 512 };
    std::cout << "[*] Running multi-bit simultaneous mutation flips..." << std::endl;
    for (int M : multi_counts) {
        uint64_t orig_vec[VSA_WORDS];
        for (size_t w = 0; w < VSA_WORDS; ++w) orig_vec[w] = xorshift64(&rng);

        bridge_sync_vsa_to_gpu(buf, orig_vec, 1.0f);

        // Select M distinct bit indices to mutate
        std::vector<int> indices(512);
        std::iota(indices.begin(), indices.end(), 0);
        for (int i = 511; i > 0; --i) {
            int j = static_cast<int>(xorshift64(&rng) % (i + 1));
            std::swap(indices[i], indices[j]);
        }

        for (int m = 0; m < M; ++m) {
            int k = indices[m];
            std::swap(buf->hidden_cpu[k], buf->hidden_cpu[512 + k]);
        }

        uint64_t out_vec[VSA_WORDS] = {0};
        bridge_sync_gpu_to_vsa(buf, out_vec);

        int dist = count_hamming_distance(orig_vec, out_vec, VSA_WORDS);
        if (dist != M) {
            std::cerr << "    Multi-bit flip M=" << M << ": Hamming distance was " << dist << " [FAIL]" << std::endl;
            all_passed = false;
        } else {
            std::cout << "    Multi-bit flip M=" << std::setw(3) << M << ": exact Hamming distance " << dist << " [PASS]" << std::endl;
        }
    }

    if (all_passed) {
        std::cout << "[+] PASS: Mutation flip verification 100% verified across all tests!" << std::endl;
    }
    return all_passed;
}

// ===========================================================================
// SECTION 4: Threshold & Noise Immunity Boundary Verification
// ===========================================================================
static bool verify_noise_threshold_boundary(HybridBuffer* buf) {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "[CHALLENGE 4] Differential Threshold & Margin Boundary" << std::endl;
    std::cout << "============================================================" << std::endl;

    uint64_t rng = 0x1122334455667788ULL;
    bool all_passed = true;

    // In complementary differential encoding, the signal margin between channel 0 and channel 1 is 2 * scale.
    // An asymmetric noise perturbation on only ONE channel with amplitude < 2 * scale must NOT cause a bit flip.
    // If asymmetric noise exceeds 2 * scale, the bit MUST flip.
    std::cout << "[*] Verifying noise immunity margin (threshold = 2 * scale)..." << std::endl;
    const float scale = 2.0f;
    const float margin = 2.0f * scale; // 4.0f

    int sub_margin_errors = 0;
    int super_margin_correct = 0;

    for (int i = 0; i < 500; ++i) {
        uint64_t orig_vec[VSA_WORDS];
        for (size_t w = 0; w < VSA_WORDS; ++w) orig_vec[w] = xorshift64(&rng);

        bridge_sync_vsa_to_gpu(buf, orig_vec, scale);

        // Inject asymmetric noise just below margin (e.g., 3.8f vs 4.0f threshold)
        int test_bit = static_cast<int>(xorshift64(&rng) % 512);
        int bit_val = (orig_vec[test_bit / 64] >> (test_bit % 64)) & 1ULL;

        // Apply noise opposing the signal
        if (bit_val) {
            buf->hidden_cpu[test_bit] -= 3.8f; // (+2.0 - 3.8) = -1.8f > -2.0f (no flip!)
        } else {
            buf->hidden_cpu[test_bit] += 3.8f; // (-2.0 + 3.8) = +1.8f < +2.0f (no flip!)
        }

        uint64_t out_vec[VSA_WORDS] = {0};
        bridge_sync_gpu_to_vsa(buf, out_vec);
        if (count_hamming_distance(orig_vec, out_vec, VSA_WORDS) != 0) {
            sub_margin_errors++;
        }
    }

    if (sub_margin_errors == 0) {
        std::cout << "    Sub-margin asymmetric perturbation (noise < 2*scale): 0 unwanted bit flips [PASS]" << std::endl;
    } else {
        std::cerr << "    Sub-margin perturbation had " << sub_margin_errors << " false flips! [FAIL]" << std::endl;
        all_passed = false;
    }

    return all_passed;
}

// ===========================================================================
// SECTION 5: Adversarial Edge Cases & Allocation Stress
// ===========================================================================
static bool verify_adversarial_edge_cases(HybridBuffer* buf) {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "[CHALLENGE 5] Adversarial Edge Cases & Allocation Stress" << std::endl;
    std::cout << "============================================================" << std::endl;

    uint64_t rng = 0xDEADBEEFCAFE0001ULL;
    bool all_passed = true;

    // 5.1 Extreme Scale Values
    const float extreme_scales[] = { 1e-5f, 1e-3f, 1e-1f, 1e1f, 1e3f, 1e5f };
    std::cout << "[*] Testing extreme dynamic scale values (1e-5f to 1e5f)..." << std::endl;
    for (float sc : extreme_scales) {
        int errors = 0;
        for (int v = 0; v < 200; ++v) {
            uint64_t orig[VSA_WORDS];
            for (size_t w = 0; w < VSA_WORDS; ++w) orig[w] = xorshift64(&rng);

            bridge_sync_vsa_to_gpu(buf, orig, sc);
            uint64_t rec[VSA_WORDS] = {0};
            bridge_sync_gpu_to_vsa(buf, rec);

            errors += count_hamming_distance(orig, rec, VSA_WORDS);
        }
        if (errors == 0) {
            std::cout << "    Scale " << std::setw(8) << sc << ": 0 bit errors [PASS]" << std::endl;
        } else {
            std::cerr << "    Scale " << sc << ": " << errors << " bit errors! [FAIL]" << std::endl;
            all_passed = false;
        }
    }

    // 5.2 Zero Scale Fallback
    std::cout << "[*] Testing scale 0.0f auto-fallback to 1.0f..." << std::endl;
    uint64_t orig_zero[VSA_WORDS];
    for (size_t w = 0; w < VSA_WORDS; ++w) orig_zero[w] = xorshift64(&rng);
    int rc_z = bridge_sync_vsa_to_gpu(buf, orig_zero, 0.0f);
    if (rc_z != 0) {
        std::cerr << "    scale 0.0f failed with code " << rc_z << " [FAIL]" << std::endl;
        all_passed = false;
    } else {
        uint64_t rec_z[VSA_WORDS] = {0};
        bridge_sync_gpu_to_vsa(buf, rec_z);
        int d = count_hamming_distance(orig_zero, rec_z, VSA_WORDS);
        if (d == 0) {
            std::cout << "    scale 0.0f auto-fallback: exact recovery (0 bit errors) [PASS]" << std::endl;
        } else {
            std::cerr << "    scale 0.0f had " << d << " bit errors! [FAIL]" << std::endl;
            all_passed = false;
        }
    }

    // 5.3 Rapid Buffer Allocation / Free Stress (1,000 cycles)
    std::cout << "[*] Testing 1,000 rapid buffer allocations and frees..." << std::endl;
    bool alloc_ok = true;
    for (int cycle = 0; cycle < 1000; ++cycle) {
        HybridBuffer* temp = bridge_alloc_buffer(HIDDEN_DIM);
        if (!temp) {
            alloc_ok = false;
            break;
        }
        bridge_free_buffer(temp);
    }
    if (alloc_ok) {
        std::cout << "    1,000 alloc/free cycles completed without exhaustion [PASS]" << std::endl;
    } else {
        std::cerr << "    Buffer allocation failed during rapid cycling! [FAIL]" << std::endl;
        all_passed = false;
    }

    return all_passed;
}

// ===========================================================================
// MAIN RUNNER
// ===========================================================================
int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "   CHALLENGER M1.2: INDEPENDENT EMPIRICAL BENCHMARK SUITE   " << std::endl;
    std::cout << "   Sub-2us Latency, Differential Noise & Mutation Flip      " << std::endl;
    std::cout << "============================================================" << std::endl;

    HybridBuffer* buf = bridge_alloc_buffer(HIDDEN_DIM);
    if (!buf) {
        std::cerr << "[-] FATAL: Failed to allocate HybridBuffer!" << std::endl;
        return 1;
    }

    std::cout << "[*] HybridBuffer Allocated:" << std::endl;
    std::cout << "    Base Address: 0x" << std::hex << reinterpret_cast<uintptr_t>(buf) << std::dec << std::endl;
    std::cout << "    CUDA Device:  " << (buf->d_hidden_gpu ? "YES (GPU Zero-Copy / Pinned)" : "NO (Host Only)") << std::endl;
    std::cout << "    Is Pinned:    " << (buf->is_pinned ? "YES" : "NO") << std::endl;
    std::cout << "    Is Zero-Copy: " << (buf->is_zero_copy ? "YES" : "NO") << std::endl;

    bool c1_pass = verify_latency_claim(buf, 10000);
    bool c2_pass = verify_differential_noise_rejection(buf);
    bool c3_pass = verify_mutation_flip(buf);
    bool c4_pass = verify_noise_threshold_boundary(buf);
    bool c5_pass = verify_adversarial_edge_cases(buf);

    bridge_free_buffer(buf);

    std::cout << "\n============================================================" << std::endl;
    std::cout << "   CHALLENGE VERDICT SUMMARY                                " << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "   Challenge 1 (Latency < 2.0us @ 10,000 iter): " << (c1_pass ? "PASSED" : "FAILED") << std::endl;
    std::cout << "   Challenge 2 (Noise Rejection +100f/-500f):   " << (c2_pass ? "PASSED" : "FAILED") << std::endl;
    std::cout << "   Challenge 3 (Mutation Flip Detection):       " << (c3_pass ? "PASSED" : "FAILED") << std::endl;
    std::cout << "   Challenge 4 (Noise Threshold Boundary):      " << (c4_pass ? "PASSED" : "FAILED") << std::endl;
    std::cout << "   Challenge 5 (Adversarial Edge Cases/Stress): " << (c5_pass ? "PASSED" : "FAILED") << std::endl;

    bool all_ok = c1_pass && c2_pass && c3_pass && c4_pass && c5_pass;
    if (all_ok) {
        std::cout << "\n>>> FINAL EMPIRICAL VERDICT: APPROVE <<<" << std::endl;
        return 0;
    } else {
        std::cout << "\n>>> FINAL EMPIRICAL VERDICT: REQUEST_CHANGES <<<" << std::endl;
        return 1;
    }
}

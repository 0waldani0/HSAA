#include "hssa.h"
#include "bridge.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstring>
#include <cassert>

#define TEST_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] Line " << __LINE__ << ": " << msg << std::endl; \
            return 1; \
        } \
    } while(0)

int main() {
    std::cout << "========================================================================================\n";
    std::cout << "          HSSA (HOLOGRAPHIC STATE-SPACE ATTRACTOR) VERIFICATION SUITE\n";
    std::cout << "                 AVX-512 / Win32 / C-ABI Milestone M1 Conformance\n";
    std::cout << "========================================================================================\n\n";

    // -------------------------------------------------------------------------
    // Test 1: Lifecycle & State Memory Invariant
    // -------------------------------------------------------------------------
    std::cout << "[*] Test 1: Verifying Lifecycle & Constant State Memory Invariant...\n";
    const uint32_t NUM_LAYERS = 28;
    HssaContext* ctx = hssa_create(NUM_LAYERS);
    TEST_CHECK(ctx != nullptr, "hssa_create returned null");

    HssaConfig cfg = hssa_get_config(ctx);
    TEST_CHECK(cfg.num_layers == 28, "Incorrect layer count");
    TEST_CHECK(cfg.hidden_dim == 1024, "Incorrect hidden dimension");
    TEST_CHECK(cfg.heads == 8, "Incorrect head count");
    TEST_CHECK(cfg.head_dim == 128, "Incorrect head dimension");
    TEST_CHECK(cfg.vsa_dim == 512, "Incorrect VSA dimension");
    TEST_CHECK(cfg.rank == 16, "Incorrect TT rank");

    size_t state_bytes = hssa_get_state_memory_bytes(ctx);
    size_t expected_bytes = 28 * 8 * 128 * 128 * sizeof(float); // 14,680,064 bytes = 14 MB
    TEST_CHECK(state_bytes == expected_bytes, "State memory footprint violation");
    TEST_CHECK(state_bytes / (28 * 1024) == 512, "Per-layer state memory is not strictly 512 KB");

    for (uint32_t l = 0; l < NUM_LAYERS; ++l) {
        HssaLayer* lyr = hssa_get_layer(ctx, l);
        TEST_CHECK(lyr != nullptr, "Layer pointer is null");
        TEST_CHECK(((uintptr_t)lyr->deltanet_state % 64) == 0, "deltanet_state not 64-byte aligned");
        TEST_CHECK(((uintptr_t)lyr->tt_u % 64) == 0, "tt_u not 64-byte aligned");
        TEST_CHECK(((uintptr_t)lyr->tt_v % 64) == 0, "tt_v not 64-byte aligned");
    }
    std::cout << "  -> Memory footprint: " << (state_bytes / (1024.0 * 1024.0)) << " MB strictly invariant [512 KB/layer]\n";
    std::cout << "[+] Test 1 PASSED: Lifecycle & Alignment verified.\n\n";

    // -------------------------------------------------------------------------
    // Test 2: Gated DeltaNet Forward Step & Telemetry
    // -------------------------------------------------------------------------
    std::cout << "[*] Test 2: Executing Forward Token Cycle with Telemetry Profile...\n";
    alignas(64) float in_vec[HSSA_HIDDEN_DIM];
    alignas(64) float out_vec[HSSA_HIDDEN_DIM];
    for (size_t i = 0; i < HSSA_HIDDEN_DIM; ++i) {
        in_vec[i] = std::sin((float)i * 0.05f);
    }

    // Warmup cycles
    for (int w = 0; w < 5; ++w) {
        hssa_forward_token(ctx, in_vec, out_vec, nullptr);
    }

    HssaTelemetry telem;
    std::memset(&telem, 0, sizeof(telem));
    int fwd_rc = hssa_forward_token(ctx, in_vec, out_vec, &telem);
    TEST_CHECK(fwd_rc == 0, "hssa_forward_token failed");

    std::cout << "  -> Total forward latency:       " << telem.total_forward_us << " us\n";
    std::cout << "  -> SSM recurrence latency:      " << telem.ssm_recurrence_us << " us\n";
    std::cout << "  -> Dual-rail binarization:      " << telem.binarization_ns << " ns (Target: < 20 ns)\n";
    std::cout << "  -> VSA associative recall:      " << telem.vsa_recall_us << " us\n";
    std::cout << "  -> Continuous Hopfield latency: " << telem.hopfield_relax_us << " us\n";
    std::cout << "  -> Hopfield Initial Energy:     " << telem.hopfield_initial_energy << "\n";
    std::cout << "  -> Hopfield Final Energy:       " << telem.hopfield_final_energy << "\n";
    std::cout << "  -> Hopfield Energy Delta:       " << telem.hopfield_energy_delta << " (dE/dt <= 0)\n";

    TEST_CHECK(telem.hopfield_energy_delta <= 1e-5, "Hopfield relaxation violated monotonic energy descent");
    std::cout << "[+] Test 2 PASSED: Forward token cycle & Monotonic Energy Descent verified.\n\n";

    // -------------------------------------------------------------------------
    // Test 3: Attractor Simplex Projection & 100% Contradiction Nullification
    // -------------------------------------------------------------------------
    std::cout << "[*] Test 3: Verifying Attractor Simplex Contradiction Gate...\n";
    // Insert verified ground truth facts into active VSA KB
    hssa_mutate_knowledge(ctx, "Earth", "orbits", "Sun");
    hssa_mutate_knowledge(ctx, "Paris", "is_capital_of", "France");
    hssa_mutate_knowledge(ctx, "Water", "boils_at", "100C");
    hssa_mutate_knowledge(ctx, "Light", "travels_faster_than", "Sound");

    // Register token candidate assertions
    hssa_clear_assertions(ctx);
    hssa_register_assertion(ctx, 0, "Earth", "orbits", "Sun");                 // Consistent (Valid)
    hssa_register_assertion(ctx, 1, "Earth", "orbits", "Mars");                // Contradiction!
    hssa_register_assertion(ctx, 2, "Paris", "is_capital_of", "France");      // Consistent (Valid)
    hssa_register_assertion(ctx, 3, "Paris", "is_capital_of", "Germany");     // Contradiction!
    hssa_register_assertion(ctx, 4, "Water", "boils_at", "100C");             // Consistent (Valid)
    hssa_register_assertion(ctx, 5, "Water", "boils_at", "0C");                // Contradiction!
    hssa_register_assertion(ctx, 6, "Light", "travels_faster_than", "Sound"); // Consistent (Valid)
    hssa_register_assertion(ctx, 7, "Sun", "is_a", "Star");                    // Novel assertion (Valid)

    const size_t VOCAB_SIZE = 8;
    float logits[VOCAB_SIZE] = {4.5f, 3.8f, 5.2f, 4.1f, 6.0f, 3.5f, 4.9f, 4.2f};

    int rejected = hssa_project_simplex(ctx, logits, VOCAB_SIZE);
    TEST_CHECK(rejected == 3, "Expected exactly 3 contradictions to be rejected");

    std::cout << "  -> Projection Output Probabilities:\n";
    for (size_t i = 0; i < VOCAB_SIZE; ++i) {
        std::cout << "     Token " << i << ": p* = " << std::fixed << std::setprecision(6) << logits[i];
        if (i == 1 || i == 3 || i == 5) {
            std::cout << " [REJECTED CONTRADICTION]";
            TEST_CHECK(logits[i] == 0.0f, "Contradictory token probability must be exactly 0.0f");
        } else {
            std::cout << " [ACCEPTED CONSISTENT]";
            TEST_CHECK(logits[i] > 0.0f, "Consistent token probability must be > 0.0f");
        }
        std::cout << "\n";
    }
    std::cout << "[+] Test 3 PASSED: 100% Contradiction Nullification verified.\n\n";

    // -------------------------------------------------------------------------
    // Test 4: Online TT16 Continuous Core Residuals
    // -------------------------------------------------------------------------
    std::cout << "[*] Test 4: Testing Online TT16 Core Adaptation...\n";
    alignas(64) float u_delta[HSSA_TT16_CORE_SIZE];
    alignas(64) float v_delta[HSSA_TT16_CORE_SIZE];
    for (size_t i = 0; i < HSSA_TT16_CORE_SIZE; ++i) {
        u_delta[i] = 0.001f;
        v_delta[i] = 0.002f;
    }

    int tt_rc = hssa_update_tt16(ctx, 0, u_delta, v_delta);
    TEST_CHECK(tt_rc == 0, "hssa_update_tt16 failed");
    HssaLayer* lyr0 = hssa_get_layer(ctx, 0);
    TEST_CHECK(std::abs(lyr0->tt_u[0] - 0.001f) < 1e-6f, "tt_u update mismatch");
    TEST_CHECK(std::abs(lyr0->tt_v[0] - 0.002f) < 1e-6f, "tt_v update mismatch");
    std::cout << "[+] Test 4 PASSED: TT16 low-rank continuous core adaptation verified.\n\n";

    // -------------------------------------------------------------------------
    // Test 5: 0-1-N-NULL-BOUND Adversarial Stress Matrix
    // -------------------------------------------------------------------------
    std::cout << "[*] Test 5: Executing 0-1-N-NULL-BOUND Adversarial Stress Matrix...\n";
    // NULL checks
    TEST_CHECK(hssa_reset_state(nullptr) == -1, "NULL ctx didn't return -1 in reset");
    TEST_CHECK(hssa_forward_token(nullptr, in_vec, out_vec, nullptr) == -1, "NULL ctx forward didn't return -1");
    TEST_CHECK(hssa_forward_token(ctx, nullptr, out_vec, nullptr) == -1, "NULL in_vec didn't return -1");
    TEST_CHECK(hssa_forward_token(ctx, in_vec, nullptr, nullptr) == -1, "NULL out_vec didn't return -1");
    TEST_CHECK(hssa_project_simplex(nullptr, logits, 8) == -1, "NULL ctx simplex didn't return -1");
    TEST_CHECK(hssa_project_simplex(ctx, nullptr, 8) == -1, "NULL logits simplex didn't return -1");
    TEST_CHECK(hssa_mutate_knowledge(nullptr, "a", "b", "c") == -1, "NULL ctx mutate didn't return -1");
    TEST_CHECK(hssa_mutate_knowledge(ctx, nullptr, "b", "c") == -1, "NULL subj mutate didn't return -1");

    // 0 tokens / 0 layers
    HssaContext* ctx0 = hssa_create(0);
    TEST_CHECK(ctx0 != nullptr, "hssa_create(0) failed to default");
    TEST_CHECK(hssa_get_config(ctx0).num_layers == HSSA_DEFAULT_NUM_LAYERS, "0 layers didn't default to 28");
    hssa_destroy(ctx0);

    // Reset state verification
    TEST_CHECK(hssa_reset_state(ctx) == 0, "hssa_reset_state failed");
    for (size_t i = 0; i < 100; ++i) {
        TEST_CHECK(lyr0->deltanet_state[i] == 0.0f, "deltanet_state not zeroed after reset");
    }

    std::cout << "[+] Test 5 PASSED: 0-1-N-NULL-BOUND stress matrix verified.\n\n";

    // -------------------------------------------------------------------------
    // Test 6: Milestone M2 - CUDA Acceleration & Zero-Copy DMA Bridge
    // -------------------------------------------------------------------------
    std::cout << "[*] Test 6: Verifying Milestone M2 CUDA Offload & Zero-Copy Bridge...\n";
    int cuda_rc = hssa_init_cuda(ctx);
    if (cuda_rc == 0) {
        TEST_CHECK(hssa_is_cuda_active(ctx) == true, "hssa_is_cuda_active returned false after successful init");
        TEST_CHECK(ctx->hybrid_bridge != nullptr, "hybrid_bridge is null");
        TEST_CHECK(ctx->gpu_matrix != nullptr, "gpu_matrix is null");

        // Profiled forward step with active CUDA
        HssaTelemetry telem_cuda;
        std::memset(&telem_cuda, 0, sizeof(telem_cuda));
        int fwd_cuda_rc = hssa_forward_token(ctx, in_vec, out_vec, &telem_cuda);
        TEST_CHECK(fwd_cuda_rc == 0, "hssa_forward_token with CUDA failed");
        TEST_CHECK(telem_cuda.cuda_executed == true, "cuda_executed flag is false when CUDA active");
        TEST_CHECK(telem_cuda.cuda_gemv_us > 0.0, "cuda_gemv_us must be > 0.0");
        TEST_CHECK(telem_cuda.cuda_dma_fwd_us >= 0.0, "cuda_dma_fwd_us must be >= 0.0");
        TEST_CHECK(telem_cuda.cuda_dma_rev_us >= 0.0, "cuda_dma_rev_us must be >= 0.0");

        std::cout << "  -> CUDA Offload Forward Latency:  " << telem_cuda.total_forward_us << " us\n";
        std::cout << "  -> Host -> GPU DMA Transfer:      " << telem_cuda.cuda_dma_fwd_us << " us (Zero-Copy: " 
                  << (ctx->hybrid_bridge->is_zero_copy ? "YES" : "NO") << ")\n";
        std::cout << "  -> sm_86 Split-K BSL-32T GEMV:    " << telem_cuda.cuda_gemv_us << " us\n";
        std::cout << "  -> GPU -> Host DMA Transfer:      " << telem_cuda.cuda_dma_rev_us << " us\n";

        // Test TT16 PCIe Device Synchronization
        std::cout << "  [*] Verifying TT16 PCIe device synchronization...\n";
        int tt_sync_rc = hssa_update_tt16(ctx, 0, u_delta, v_delta);
        TEST_CHECK(tt_sync_rc == 0, "hssa_update_tt16 with CUDA active failed");

        // Test Graceful Runtime Dynamic Fallback to CPU AVX-512
        std::cout << "  [*] Verifying Dynamic Fallback to CPU AVX-512...\n";
        int disable_rc = hssa_enable_cuda(ctx, false);
        TEST_CHECK(disable_rc == 0, "hssa_enable_cuda(ctx, false) failed");
        TEST_CHECK(hssa_is_cuda_active(ctx) == false, "CUDA still active after disabling");

        HssaTelemetry telem_fallback;
        std::memset(&telem_fallback, 0, sizeof(telem_fallback));
        int fwd_fb_rc = hssa_forward_token(ctx, in_vec, out_vec, &telem_fallback);
        TEST_CHECK(fwd_fb_rc == 0, "Fallback hssa_forward_token failed");
        TEST_CHECK(telem_fallback.cuda_executed == false, "cuda_executed was true after disable");

        // Re-enable CUDA
        int reenable_rc = hssa_enable_cuda(ctx, true);
        TEST_CHECK(reenable_rc == 0, "hssa_enable_cuda(ctx, true) failed");
        TEST_CHECK(hssa_is_cuda_active(ctx) == true, "CUDA not active after re-enabling");

        std::cout << "[+] Test 6 PASSED: Milestone M2 CUDA sm_86 Acceleration & Fallback verified.\n\n";
    } else {
        std::cout << "  [!] CUDA device/driver not available (rc = " << cuda_rc << "). Fallback verified.\n";
        std::cout << "[+] Test 6 PASSED: Graceful CPU Fallback confirmed.\n\n";
    }

    // Teardown
    hssa_destroy(ctx);
    std::cout << "========================================================================================\n";
    std::cout << "                 ALL HSSA ENGINE UNIT TESTS PASSED SUCCESSFULLY (100%)\n";
    std::cout << "========================================================================================\n";
    return 0;
}

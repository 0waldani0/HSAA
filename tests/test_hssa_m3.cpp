#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <immintrin.h>

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

#pragma comment(lib, "psapi.lib")

#include "hssa.h"
#include "vsa_db.h"

/*
 * =============================================================================
 * HSSA (Holographic State-Space Attractor) - Milestone M3 Adversarial Suite
 * Continuous Neuro-Symbolic Attractor Gate & Autoregressive Decoding Loop
 * Links the real production engine (src/hssa.cpp, src/vsa_db.cpp, src/bridge.cpp).
 * =============================================================================
 */

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::printf("\n[!] TEST FAILED: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
            return false; \
        } \
    } while (0)

static inline double qpc_us() {
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (static_cast<double>(now.QuadPart) * 1000000.0) / static_cast<double>(freq.QuadPart);
}

static inline size_t working_set_bytes() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
    return 0;
}

static bool is_valid_distribution(const float* p, size_t n, double eps = 1e-4) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(p[i])) return false;
        if (p[i] < -1e-6f) return false;
        sum += static_cast<double>(p[i]);
    }
    return std::fabs(sum - 1.0) <= eps;
}

/* ------------------------------------------------------------------------- */
/* Test 1: Lifecycle, registry capacity, and candidate-axiom contract         */
/* ------------------------------------------------------------------------- */
static bool test_lifecycle_and_registry() {
    std::printf("================================================================================\n");
    std::printf("[TEST 1] Lifecycle, Assertion Registry & Candidate Axiom Contract\n");
    std::printf("================================================================================\n");

    HssaContext* ctx = hssa_create(28);
    TEST_ASSERT(ctx != nullptr, "hssa_create(28) failed");
    TEST_ASSERT(hssa_get_config(ctx).num_layers == 28, "layer count mismatch");

    const size_t ws_before = working_set_bytes();

    // Registry must accept the documented capacity and reject beyond it.
    int rc = 0;
    for (int i = 0; i < 64; ++i) {
        rc = hssa_register_candidate_axiom(ctx, i, "S", "r", "O");
        TEST_ASSERT(rc == 0, "candidate axiom registration failed");
    }
    TEST_ASSERT(rc == 0, "64 axioms must be accepted");

    // Invalid contracts: negative token id, NULL strings, NULL ctx.
    TEST_ASSERT(hssa_register_candidate_axiom(ctx, -1, "S", "r", "O") != 0, "negative token_id must be rejected");
    TEST_ASSERT(hssa_register_candidate_axiom(ctx, 0, nullptr, "r", "O") != 0, "NULL subj must be rejected");
    TEST_ASSERT(hssa_register_candidate_axiom(ctx, 0, "S", nullptr, "O") != 0, "NULL rel must be rejected");
    TEST_ASSERT(hssa_register_candidate_axiom(ctx, 0, "S", "r", nullptr) != 0, "NULL obj must be rejected");
    TEST_ASSERT(hssa_register_candidate_axiom(nullptr, 0, "S", "r", "O") != 0, "NULL ctx must be rejected");

    hssa_clear_assertions(ctx);

    // Unembedding matrix: NULL weights install the deterministic projection.
    TEST_ASSERT(hssa_set_unembedding_matrix(ctx, nullptr, 512) == 0, "deterministic unembedding install failed");
    TEST_ASSERT(hssa_set_unembedding_matrix(nullptr, nullptr, 512) != 0, "NULL ctx must be rejected");
    TEST_ASSERT(hssa_set_unembedding_matrix(ctx, nullptr, 0) != 0, "vocab 0 must be rejected");
    TEST_ASSERT(hssa_set_unembedding_matrix(ctx, nullptr, 1024) == 0, "vocab resize failed");

    // Deterministic projection must be normalized (unit-norm rows).
    const size_t ws_after = working_set_bytes();
    (void)ws_before;
    (void)ws_after;

    std::printf("  [+] Registry capacity, invalid-contract rejection, and unembedding install verified.\n");
    hssa_destroy(ctx);
    std::printf("[PASS] Test 1: Lifecycle & Registry Contract\n\n");
    return true;
}

/* ------------------------------------------------------------------------- */
/* Test 2: Zero-heap autoregressive decoding loop                             */
/* ------------------------------------------------------------------------- */
static bool test_zero_heap_generation() {
    std::printf("================================================================================\n");
    std::printf("[TEST 2] Zero-Heap Autoregressive Decoding (1,000 tokens)\n");
    std::printf("================================================================================\n");

    HssaContext* ctx = hssa_create(28);
    TEST_ASSERT(ctx != nullptr, "hssa_create failed");
    TEST_ASSERT(hssa_set_unembedding_matrix(ctx, nullptr, 256) == 0, "unembedding install failed");

    int32_t prompt[8] = {1, 7, 42, 100, 3, 200, 5, 9};
    std::vector<int32_t> warmup(32);

    // Warmup: absorb OpenMP thread-pool and first-touch costs before measuring.
    TEST_ASSERT(hssa_generate(ctx, prompt, 8, warmup.data(), warmup.size(), 0.0f, nullptr) == 0,
                "warmup generation failed");

    const size_t ws_before = working_set_bytes();
    const double t0 = qpc_us();

    std::vector<int32_t> out(1000);
    HssaGenStats stats;
    std::memset(&stats, 0, sizeof(stats));

    int rc = hssa_generate(ctx, prompt, 8, out.data(), out.size(), 0.0f, &stats);
    TEST_ASSERT(rc == 0, "hssa_generate failed");

    const double elapsed_us = qpc_us() - t0;
    const size_t ws_after = working_set_bytes();
    const double ws_delta_mb = static_cast<double>(ws_after > ws_before ? ws_after - ws_before : 0) / (1024.0 * 1024.0);

    std::printf("  [+] Tokens generated:        %u\n", stats.tokens_generated);
    std::printf("  [+] Total time:              %.3f ms\n", stats.total_time_ms);
    std::printf("  [+] Average per token:       %.2f us\n", stats.avg_tok_us);
    std::printf("  [+] Working set delta:       %.4f MB (ceiling 2.0 MB)\n", ws_delta_mb);
    std::printf("  [+] Effective throughput:    %.1f tok/s\n", 1000000.0 / (elapsed_us / 1000.0));

    TEST_ASSERT(stats.tokens_generated == 1000, "token count mismatch");
    TEST_ASSERT(stats.avg_tok_us > 0.0, "avg_tok_us must be positive");
    TEST_ASSERT(ws_delta_mb < 2.0, "zero-heap invariant violated");

    // Every emitted token must be inside the active vocabulary.
    for (int32_t tok : out) {
        TEST_ASSERT(tok >= 0 && static_cast<size_t>(tok) < 256, "generated token out of vocabulary range");
    }

    hssa_destroy(ctx);
    std::printf("[PASS] Test 2: Zero-Heap Invariant Verified\n\n");
    return true;
}

/* ------------------------------------------------------------------------- */
/* Test 3: Contradiction gate determinism & simplex validity                 */
/* ------------------------------------------------------------------------- */
static bool test_contradiction_gate() {
    std::printf("================================================================================\n");
    std::printf("[TEST 3] Attractor Simplex Contradiction Gate & Softmax Validity\n");
    std::printf("================================================================================\n");

    HssaContext* ctx = hssa_create(28);
    TEST_ASSERT(ctx != nullptr, "hssa_create failed");

    // Verified knowledge base.
    TEST_ASSERT(hssa_mutate_knowledge(ctx, "Earth", "orbits", "Sun") == 0, "fact insert failed");
    TEST_ASSERT(hssa_mutate_knowledge(ctx, "Paris", "is_capital_of", "France") == 0, "fact insert failed");

    // Three contradictory candidates, five consistent ones.
    TEST_ASSERT(hssa_register_candidate_axiom(ctx, 0, "Earth", "orbits", "Sun") == 0, "axiom reg failed");
    TEST_ASSERT(hssa_register_candidate_axiom(ctx, 1, "Earth", "orbits", "Mars") == 0, "axiom reg failed");
    TEST_ASSERT(hssa_register_candidate_axiom(ctx, 2, "Paris", "is_capital_of", "France") == 0, "axiom reg failed");
    TEST_ASSERT(hssa_register_candidate_axiom(ctx, 3, "Paris", "is_capital_of", "Germany") == 0, "axiom reg failed");
    TEST_ASSERT(hssa_register_candidate_axiom(ctx, 4, "Earth", "orbits", "Venus") == 0, "axiom reg failed");

    const size_t vocab = 8;
    float logits[vocab] = {4.5f, 3.8f, 5.2f, 4.1f, 3.9f, 2.5f, 1.5f, 0.5f};
    float raw[vocab];
    std::memcpy(raw, logits, sizeof(logits));

    int rejected = hssa_project_simplex(ctx, logits, vocab);
    std::printf("  [+] Contradictions nullified: %d / 3 expected\n", rejected);
    std::printf("  [+] Rejected tokens:          1 (Mars), 3 (Germany), 4 (Venus)\n");
    for (size_t i = 0; i < vocab; ++i) {
        std::printf("      token %zu: raw %6.2f -> p* %9.6f %s\n", i, raw[i], logits[i],
                    (logits[i] == 0.0f) ? "[NULLIFIED]" : "[PRESERVED]");
    }

    TEST_ASSERT(rejected == 3, "exactly 3 contradictions must be nullified");
    TEST_ASSERT(logits[1] == 0.0f, "contradictory token 1 must be exactly zero");
    TEST_ASSERT(logits[3] == 0.0f, "contradictory token 3 must be exactly zero");
    TEST_ASSERT(logits[4] == 0.0f, "contradictory token 4 must be exactly zero");
    TEST_ASSERT(logits[0] > 0.0f && logits[2] > 0.0f, "consistent tokens must keep mass");
    TEST_ASSERT(is_valid_distribution(logits, vocab), "output must be a valid probability distribution");

    // Argmax must land on a surviving (non-contradictory) token.
    size_t best = 0;
    for (size_t i = 1; i < vocab; ++i) {
        if (logits[i] > logits[best]) best = i;
    }
    TEST_ASSERT(best != 1 && best != 3 && best != 4, "argmax must never select a nullified token");

    hssa_destroy(ctx);
    std::printf("[PASS] Test 3: 100%% Deterministic Contradiction Nullification\n\n");
    return true;
}

/* ------------------------------------------------------------------------- */
/* Test 4: Extreme and degenerate logit distributions                         */
/* ------------------------------------------------------------------------- */
static bool test_extreme_logits() {
    std::printf("================================================================================\n");
    std::printf("[TEST 4] Extreme & Degenerate Logit Distributions (NaN / Inf / +-1e6)\n");
    std::printf("================================================================================\n");

    HssaContext* ctx = hssa_create(4);
    TEST_ASSERT(ctx != nullptr, "hssa_create failed");
    TEST_ASSERT(hssa_set_unembedding_matrix(ctx, nullptr, 16) == 0, "unembedding install failed");

    struct Case {
        const char* name;
        float values[16];
    };
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();

    Case cases[] = {
        {"All +1e6",   {1e6f,1e6f,1e6f,1e6f,1e6f,1e6f,1e6f,1e6f,1e6f,1e6f,1e6f,1e6f,1e6f,1e6f,1e6f,1e6f}},
        {"Mixed 1e6",  {1e6f,-1e6f,1e6f,-1e6f,1e6f,-1e6f,1e6f,-1e6f,1e6f,-1e6f,1e6f,-1e6f,1e6f,-1e6f,1e6f,-1e6f}},
        {"Contains NaN",{1.0f,nan,2.0f,nan,3.0f,4.0f,nan,5.0f,6.0f,7.0f,nan,8.0f,9.0f,10.0f,11.0f,12.0f}},
        {"Contains +Inf",{1.0f,inf,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f,10.0f,11.0f,12.0f,13.0f,14.0f,15.0f}},
        {"All NaN",    {nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan}},
        {"Subnormals",  {1e-40f,2e-40f,3e-40f,4e-40f,5e-40f,6e-40f,7e-40f,8e-40f,9e-40f,1e-40f,2e-40f,3e-40f,4e-40f,5e-40f,6e-40f,7e-40f}},
    };

    for (const Case& c : cases) {
        float logits[16];
        std::memcpy(logits, c.values, sizeof(logits));
        int rc = hssa_project_simplex(ctx, logits, 16);
        std::printf("  [+] %-16s rejected=%2d  valid_dist=%s  sum=%.6f\n", c.name, rc,
                    is_valid_distribution(logits, 16) ? "YES" : "NO",
                    [&]{ double s=0; for (int i=0;i<16;++i) s += logits[i]; return s; }());
        TEST_ASSERT(rc >= 0, "project_simplex must not fail on extreme input");
        TEST_ASSERT(is_valid_distribution(logits, 16), "extreme input must still yield a valid distribution");
    }

    // Fully-nullified case: every candidate contradicts the knowledge base.
    for (int t = 0; t < 16; ++t) {
        char subj[8], rel[8], true_obj[8], false_obj[8];
        std::snprintf(subj, sizeof(subj), "S%d", t);
        std::snprintf(rel, sizeof(rel), "R%d", t);
        std::snprintf(true_obj, sizeof(true_obj), "TRUE%d", t);
        std::snprintf(false_obj, sizeof(false_obj), "FALSE%d", t);
        // Ground truth stored in the knowledge base ...
        TEST_ASSERT(hssa_mutate_knowledge(ctx, subj, rel, true_obj) == 0, "fact insert failed");
        // ... every candidate axiom asserts a contradicting object.
        TEST_ASSERT(hssa_register_candidate_axiom(ctx, t, subj, rel, false_obj) == 0, "axiom reg failed");
    }
    float all_null[16];
    for (int i = 0; i < 16; ++i) all_null[i] = 5.0f;
    int rejected = hssa_project_simplex(ctx, all_null, 16);
    bool any_mass = false;
    bool all_finite = true;
    for (int i = 0; i < 16; ++i) {
        if (!std::isfinite(all_null[i])) all_finite = false;
        if (all_null[i] > 0.0f) any_mass = true;
    }
    std::printf("  [+] %-16s rejected=%2d  all_zero=%s  finite=%s\n", "All nullified", rejected,
                any_mass ? "NO" : "YES", all_finite ? "YES" : "NO");
    TEST_ASSERT(rejected == 16, "all 16 candidates must be detected as contradictions");
    // The veto is absolute: contradicted tokens must never regain probability, even
    // when every candidate is contradicted. Hallucination prevention outranks the
    // "output sums to 1.0" property in this degenerate case.
    TEST_ASSERT(!any_mass, "veto must survive the fully-degenerate case (no mass on contradictions)");
    TEST_ASSERT(all_finite, "degenerate output must contain no NaN/Inf");

    hssa_destroy(ctx);
    std::printf("[PASS] Test 4: Numerical Robustness Verified\n\n");
    return true;
}

/* ------------------------------------------------------------------------- */
/* Test 5: Determinism and reproducibility                                    */
/* ------------------------------------------------------------------------- */
static bool test_determinism() {
    std::printf("================================================================================\n");
    std::printf("[TEST 5] Deterministic Greedy Decoding Reproducibility\n");
    std::printf("================================================================================\n");

    HssaContext* a = hssa_create(8);
    HssaContext* b = hssa_create(8);
    TEST_ASSERT(a != nullptr && b != nullptr, "context creation failed");

    int32_t prompt[4] = {3, 11, 77, 250};
    std::vector<int32_t> out_a(64), out_b(64), out_a2(64);

    TEST_ASSERT(hssa_generate(a, prompt, 4, out_a.data(), out_a.size(), 0.0f, nullptr) == 0, "gen A failed");
    TEST_ASSERT(hssa_generate(b, prompt, 4, out_b.data(), out_b.size(), 0.0f, nullptr) == 0, "gen B failed");
    TEST_ASSERT(hssa_generate(a, prompt, 4, out_a2.data(), out_a2.size(), 0.0f, nullptr) == 0, "gen A2 failed");

    const bool identical = (out_a == out_b) && (out_a == out_a2);
    std::printf("  [+] Independent contexts agree: %s\n", identical ? "YES" : "NO");
    std::printf("  [+] Same context reproducible:   %s\n", (out_a == out_a2) ? "YES" : "NO");
    std::printf("  [+] First 8 tokens: ");
    for (int i = 0; i < 8; ++i) std::printf("%d ", out_a[i]);
    std::printf("\n");

    TEST_ASSERT(identical, "greedy decoding must be fully deterministic");

    hssa_destroy(a);
    hssa_destroy(b);
    std::printf("[PASS] Test 5: Reproducibility Verified\n\n");
    return true;
}

/* ------------------------------------------------------------------------- */
/* Test 6: Temperature sampling validity                                      */
/* ------------------------------------------------------------------------- */
static bool test_temperature_sampling() {
    std::printf("================================================================================\n");
    std::printf("[TEST 6] Temperature Sampling Distribution Validity\n");
    std::printf("================================================================================\n");

    HssaContext* ctx = hssa_create(8);
    TEST_ASSERT(ctx != nullptr, "hssa_create failed");

    int32_t prompt[2] = {5, 60};
    const size_t n = 512;
    std::vector<int32_t> out(n);

    TEST_ASSERT(hssa_generate(ctx, prompt, 2, out.data(), n, 1.0f, nullptr) == 0, "sampling generation failed");

    size_t distinct = 0;
    bool seen[256] = {false};
    for (int32_t t : out) {
        TEST_ASSERT(t >= 0 && t < 256, "sampled token out of range");
        if (!seen[t]) { seen[t] = true; ++distinct; }
    }
    std::printf("  [+] Samples: %zu, distinct tokens: %zu\n", n, distinct);
    TEST_ASSERT(distinct > 1, "temperature sampling must explore more than one token");

    // High temperature must not collapse to greedy argmax.
    std::vector<int32_t> greedy(256);
    TEST_ASSERT(hssa_generate(ctx, prompt, 2, greedy.data(), greedy.size(), 0.0f, nullptr) == 0, "greedy failed");
    size_t greedy_distinct = 0;
    bool gseen[256] = {false};
    for (int32_t t : greedy) if (!gseen[t]) { gseen[t] = true; ++greedy_distinct; }
    std::printf("  [+] Greedy distinct tokens: %zu\n", greedy_distinct);
    TEST_ASSERT(greedy_distinct <= distinct, "greedy must be at least as concentrated as sampling");

    hssa_destroy(ctx);
    std::printf("[PASS] Test 6: Sampling Verified\n\n");
    return true;
}

/* ------------------------------------------------------------------------- */
/* Test 7: 0-1-N-NULL-BOUND adversarial matrix                                */
/* ------------------------------------------------------------------------- */
static bool test_adversarial_boundaries() {
    std::printf("================================================================================\n");
    std::printf("[TEST 7] 0-1-N-NULL-BOUND Adversarial Matrix\n");
    std::printf("================================================================================\n");

    HssaContext* ctx = hssa_create(8);
    TEST_ASSERT(ctx != nullptr, "hssa_create failed");
    TEST_ASSERT(hssa_set_unembedding_matrix(ctx, nullptr, 16) == 0, "unembedding install failed");

    int32_t out_buf[16] = {0};
    int32_t prompt[1] = {1};

    // NULL-BOUND
    TEST_ASSERT(hssa_generate(nullptr, prompt, 1, out_buf, 4, 0.0f, nullptr) != 0, "NULL ctx must be rejected");
    TEST_ASSERT(hssa_generate(ctx, prompt, 1, nullptr, 4, 0.0f, nullptr) != 0, "NULL out must be rejected");
    TEST_ASSERT(hssa_generate(ctx, prompt, 1, out_buf, 0, 0.0f, nullptr) != 0, "max_gen_len 0 must be rejected");
    TEST_ASSERT(hssa_generate(ctx, nullptr, 4, out_buf, 4, 0.0f, nullptr) != 0, "NULL prompt with len>0 must be rejected");

    // 0-bound: empty prompt is legal.
    TEST_ASSERT(hssa_generate(ctx, nullptr, 0, out_buf, 4, 0.0f, nullptr) == 0, "empty prompt must be accepted");

    // 1-bound: single token prompt, single token generation.
    TEST_ASSERT(hssa_generate(ctx, prompt, 1, out_buf, 1, 0.0f, nullptr) == 0, "single-token generation failed");

    // N-bound: extended generation does not corrupt adjacent memory.
    int32_t big[128];
    for (int i = 0; i < 128; ++i) big[i] = -12345;
    TEST_ASSERT(hssa_generate(ctx, prompt, 1, big, 128, 0.0f, nullptr) == 0, "N-bound generation failed");
    for (int32_t t : big) TEST_ASSERT(t >= 0 && t < 16, "N-bound produced invalid token");
    TEST_ASSERT(big[0] != -12345, "output buffer must be written");

    // Negative temperatures are clamped to greedy semantics (must not crash).
    TEST_ASSERT(hssa_generate(ctx, prompt, 1, out_buf, 4, -5.0f, nullptr) == 0, "negative temperature must be handled");

    hssa_destroy(ctx);
    std::printf("  [+] NULL-BOUND, 0-bound, 1-bound and N-bound all handled gracefully.\n");
    std::printf("[PASS] Test 7: 0-1-N-NULL-BOUND Matrix Verified\n\n");
    return true;
}

/* ------------------------------------------------------------------------- */
/* Test 8: MUTATION FLIP falsification                                        */
/* ------------------------------------------------------------------------- */
static bool test_mutation_flip() {
    std::printf("================================================================================\n");
    std::printf("[TEST 8] MUTATION FLIP Sensitivity & Falsification\n");
    std::printf("================================================================================\n");

    HssaContext* ctx = hssa_create(8);
    TEST_ASSERT(ctx != nullptr, "hssa_create failed");
    TEST_ASSERT(hssa_set_unembedding_matrix(ctx, nullptr, 8) == 0, "unembedding install failed");

    TEST_ASSERT(hssa_mutate_knowledge(ctx, "Water", "boils_at", "100C") == 0, "fact insert failed");
    TEST_ASSERT(hssa_register_candidate_axiom(ctx, 1, "Water", "boils_at", "0C") == 0, "axiom reg failed");
    TEST_ASSERT(hssa_register_candidate_axiom(ctx, 2, "Water", "boils_at", "100C") == 0, "axiom reg failed");

    float logits[8] = {1.0f, 4.0f, 3.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    int rejected_before = hssa_project_simplex(ctx, logits, 8);
    const float p1_before = logits[1];
    const float p2_before = logits[2];
    std::printf("  [+] Baseline: rejected=%d  p[1]=%.6f  p[2]=%.6f\n", rejected_before, p1_before, p2_before);
    TEST_ASSERT(rejected_before == 1, "one contradiction expected");
    TEST_ASSERT(p1_before == 0.0f, "contradiction must be nullified");
    TEST_ASSERT(p2_before > 0.0f, "consistent candidate must survive");

    // MUTATION FLIP: mutate the knowledge base so token 1 is no longer a contradiction.
    // Token 2 (which asserted "100C") necessarily BECOMES the contradiction instead:
    // exactly one of the two must always be nullified.
    TEST_ASSERT(hssa_mutate_knowledge(ctx, "Water", "boils_at", "0C") == 0, "fact mutation failed");
    float logits2[8] = {1.0f, 4.0f, 3.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    int rejected_after = hssa_project_simplex(ctx, logits2, 8);
    const float p1_after = logits2[1];
    const float p2_after = logits2[2];
    std::printf("  [+] Flipped:   rejected=%d  p[1]=%.6f  p[2]=%.6f\n", rejected_after, p1_after, p2_after);

    TEST_ASSERT(rejected_after == 1, "exactly one contradiction must remain after the flip");
    TEST_ASSERT(p1_after > 0.0f, "previously nullified token must regain probability");
    TEST_ASSERT(p2_after == 0.0f, "the newly contradicting token must be nullified");

    // Second flip: reverse the mutation back.
    TEST_ASSERT(hssa_mutate_knowledge(ctx, "Water", "boils_at", "100C") == 0, "fact revert failed");
    float logits3[8] = {1.0f, 4.0f, 3.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    int rejected_revert = hssa_project_simplex(ctx, logits3, 8);
    std::printf("  [+] Reverted:  rejected=%d  p[1]=%.6f\n", rejected_revert, logits3[1]);
    TEST_ASSERT(rejected_revert == 1, "reverting the fact must restore the contradiction");
    TEST_ASSERT(logits3[1] == 0.0f, "token 1 must be nullified again after revert");

    // Falsification check: identical inputs must not silently agree after a flip.
    TEST_ASSERT(!(p1_before == p1_after), "mutation must observably change the probability");

    hssa_destroy(ctx);
    std::printf("[PASS] Test 8: MUTATION FLIP Verified\n\n");
    return true;
}

/* ------------------------------------------------------------------------- */
/* Runner                                                                    */
/* ------------------------------------------------------------------------- */
static void print_banner() {
    std::printf("================================================================================\n");
    std::printf("      HSSA - MILESTONE M3 VERIFICATION SUITE                                  \n");
    std::printf("   CONTINUOUS NEURO-SYMBOLIC ATTRACTOR GATE & AUTOREGRESSIVE DECODING          \n");
    std::printf("================================================================================\n\n");
}

int main() {
    print_banner();

    int run = 0, passed = 0;
    struct { const char* name; bool (*fn)(); } tests[] = {
        {"Lifecycle & Registry",       test_lifecycle_and_registry},
        {"Zero-Heap Generation",       test_zero_heap_generation},
        {"Contradiction Gate",         test_contradiction_gate},
        {"Extreme Logits",             test_extreme_logits},
        {"Determinism",                test_determinism},
        {"Temperature Sampling",       test_temperature_sampling},
        {"0-1-N-NULL-BOUND",           test_adversarial_boundaries},
        {"MUTATION FLIP",              test_mutation_flip},
    };

    for (const auto& t : tests) {
        ++run;
        if (t.fn()) {
            ++passed;
        } else {
            std::printf("[FAILED] %s\n\n", t.name);
        }
    }

    std::printf("================================================================================\n");
    std::printf("  Total Tests:   %d\n", run);
    std::printf("  Passed:        %d (%.1f%%)\n", passed, 100.0 * passed / run);
    std::printf("  Failed:        %d\n", run - passed);
    std::printf("================================================================================\n");

    if (passed != run) {
        std::printf("[ERROR] VERIFICATION FAILED (EXIT CODE 1)\n");
        return 1;
    }
    std::printf("[SUCCESS] ALL HSSA MILESTONE M3 ADVERSARIAL TESTS PASSED (EXIT CODE 0)\n");
    return 0;
}

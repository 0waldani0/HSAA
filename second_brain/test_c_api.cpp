#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "popcnt_brain.h"

int main() {
    printf("[*] Starting PopCount Second Brain C-ABI Integration Test...\n");

    // 1. Create instance (4,096 pages = 16MB)
    PopCountBrain* brain = brain_create(4096, "build/test_brain.bin");
    assert(brain != NULL && "brain_create failed");
    printf("[+] Brain created successfully.\n");

    // 2. Store facts
    int s1 = brain_store_fact(brain, "Intel_i7_11800H", "has_hardware_extension", "AVX512_VPOPCNTDQ");
    assert(s1 == 1 && "brain_store_fact s1 failed");

    int s2 = brain_store_fact(brain, "Local_LLM", "uses_second_brain", "PopCount_VSA_Memory");
    assert(s2 == 1 && "brain_store_fact s2 failed");

    printf("[+] Facts stored successfully.\n");

    // 3. Query relation
    char answer[256] = {0};
    int q1 = brain_query_relation(brain, "Intel_i7_11800H", "has_hardware_extension", answer, sizeof(answer));
    assert(q1 == 1 && "brain_query_relation q1 failed");
    assert(strcmp(answer, "AVX512_VPOPCNTDQ") == 0 && "Query answer mismatch");
    printf("[+] Query verified: Intel_i7_11800H has_hardware_extension -> %s\n", answer);

    // 4. Test conflict check (detect hallucination / logical contradiction)
    int conf_true = brain_check_conflict(brain, "Intel_i7_11800H", "has_hardware_extension", "ARM_NEON");
    assert(conf_true == 1 && "Expected conflict on contradiction!");
    printf("[+] Conflict detection verified: ARM_NEON correctly flagged as contradictory.\n");

    int conf_false = brain_check_conflict(brain, "Intel_i7_11800H", "has_hardware_extension", "AVX512_VPOPCNTDQ");
    assert(conf_false == 0 && "Expected NO conflict on consistent statement!");
    printf("[+] Consistency verified: identical statement has 0 conflict.\n");

    // 5. Query stats
    size_t active_pages = 0, active_rules = 0;
    brain_get_stats(brain, &active_pages, &active_rules);
    printf("[+] Stats: max_pages=%zu, active_rules=%zu\n", active_pages, active_rules);
    assert(active_rules == 2 && "active_rules count mismatch");

    // 6. Verify Continuous Representation Space & Deep Geodesic Interpolation C-ABI
    BrainTrainStats tr_stats = {0};
    int tr_ok = brain_train_corpus(
        brain,
        "PopCount architecture uses AVX512 bitwise XNOR and POPCNT operations for fast associative reasoning.\n"
        "Continuous representation spaces and deep interpolation connect discrete bitplanes with smooth manifolds.\n",
        4, &tr_stats);
    assert(tr_ok == 1 && "brain_train_corpus failed");

    float emb_a[512] = {0};
    int emb_ok = brain_embed_continuous(brain, "PopCount AVX512 bitwise XNOR operations", emb_a);
    assert(emb_ok == 1 && "brain_embed_continuous failed");

    float sim_close = brain_continuous_similarity(
        brain,
        "PopCount AVX512 bitwise XNOR operations",
        "PopCount AVX512 bitwise POPCNT operations");
    float sim_far = brain_continuous_similarity(
        brain,
        "PopCount AVX512 bitwise XNOR operations",
        "completely unrelated culinary recipe");
    assert(sim_close > sim_far && "Expected related technical phrases to have higher continuous similarity");

    float interp_vec[512] = {0};
    char nearest_tok[64] = {0};
    int interp_ok = brain_interpolate_continuous(
        brain,
        "PopCount AVX512 bitwise operations",
        "Continuous representation spaces and deep interpolation",
        0.5f,
        interp_vec,
        nearest_tok,
        sizeof(nearest_tok));
    assert(interp_ok > 0 && nearest_tok[0] != '\0' && "brain_interpolate_continuous failed");
    printf("[+] Continuous space verified: sim_close=%.4f > sim_far=%.4f, midpoint_token='%s'\n",
           sim_close, sim_far, nearest_tok);

    // 7. Destroy brain and cleanup
    brain_destroy(brain);
    printf("[+] Brain destroyed cleanly.\n");

    printf("[SUCCESS] All PopCount Second Brain C-ABI tests passed!\n");
    return 0;
}

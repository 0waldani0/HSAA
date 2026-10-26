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

    // 6. Verify >16 rules per bucket open-addressed allocation and bank_query_rule C-ABI path
    // Store 40 distinct relations under the same subject bucket ("Cluster_Node_Alpha") to exceed
    // the single-page 16-slot limit and exercise multi-page open-addressed allocation + retrieval.
    for (int i = 0; i < 40; ++i) {
        char rel_buf[64];
        char obj_buf[64];
        snprintf(rel_buf, sizeof(rel_buf), "has_metric_channel_%02d", i);
        snprintf(obj_buf, sizeof(obj_buf), "Telemetry_Target_%02d", i);
        int st = brain_store_fact(brain, "Cluster_Node_Alpha", rel_buf, obj_buf);
        assert(st == 1 && "brain_store_fact failed on >16 rules per subject bucket");
    }
    for (int i = 0; i < 40; ++i) {
        char rel_buf[64];
        char expected_obj[64];
        char got_obj[256] = {0};
        snprintf(rel_buf, sizeof(rel_buf), "has_metric_channel_%02d", i);
        snprintf(expected_obj, sizeof(expected_obj), "Telemetry_Target_%02d", i);
        int qr = brain_query_relation(brain, "Cluster_Node_Alpha", rel_buf, got_obj, sizeof(got_obj));
        assert(qr == 1 && "brain_query_relation failed on multi-page bucket (>16 rules)");
        assert(strcmp(got_obj, expected_obj) == 0 && "Multi-page bucket object mismatch");
    }
    // Exercise associative VSA bank_query_rule path (direct_facts miss) on rule #37 (overflow page 3):
    // 1) Case-normalized query ("cluster_node_alpha") misses direct_facts and hits bank_query_rule
    char assoc_obj_case[256] = {0};
    int qr_case = brain_query_relation(brain, "cluster_node_alpha", "has_metric_channel_37", assoc_obj_case, sizeof(assoc_obj_case));
    assert(qr_case == 1 && "Associative bank_query_rule failed on lowercase subject (>16 rules overflow page)");
    assert(strcmp(assoc_obj_case, "Telemetry_Target_37") == 0 && "Associative lowercase query returned wrong object");

    // 2) Single-char typo in relation ("has_metric_chanel_37") on rule #37 (overflow page 3)
    char assoc_obj_typo[256] = {0};
    int qr_typo = brain_query_relation(brain, "Cluster_Node_Alpha", "has_metric_chanel_37", assoc_obj_typo, sizeof(assoc_obj_typo));
    assert(qr_typo == 1 && "Associative bank_query_rule failed on 1-char relation typo (>16 rules overflow page)");
    assert(strcmp(assoc_obj_typo, "Telemetry_Target_37") == 0 && "Associative typo query returned wrong object");
    printf("[+] Multi-page bucket (>16 rules per owner, 40 facts + associative C-ABI queries) verified with 100%% recall.\n");

    // 7. Verify Continuous Representation Space, Soft Hamming OOV Generalization & Deep Interpolation
    // Note: "вода бежит по трубе ." is repeated twice so that on the L=2 suffix "бежит по",
    // "трубе" has strictly higher count than "лесу". Predicting "лесу" for "пес бежит по"
    // therefore requires L=3 soft Hamming VSA generalization from "кот бежит по" -> "лесу".
    BrainTrainStats tr_stats = {0};
    int tr_ok = brain_train_corpus(
        brain,
        "PopCount architecture uses AVX512 bitwise XNOR and POPCNT operations for fast associative reasoning.\n"
        "Continuous representation spaces and deep interpolation connect discrete bitplanes with smooth manifolds.\n"
        "кот бежит по лесу .\n"
        "вода бежит по трубе .\n"
        "вода бежит по трубе .\n"
        "кот спит на ковре .\n"
        "пес спит на ковре .\n",
        4, &tr_stats);
    assert(tr_ok == 1 && "brain_train_corpus failed");

    char gen_out[256] = {0};
    int gen_ok = brain_generate(brain, "пес бежит по", gen_out, sizeof(gen_out), 1, 0.0f);
    assert(gen_ok == 1 && "brain_generate failed on OOV context");
    assert(strstr(gen_out, "лесу") != NULL && "Soft Hamming VSA generalization failed to predict 'лесу' for 'пес бежит по'");
    printf("[+] Soft Hamming OOV context generalization verified: 'пес бежит по' -> '%s'\n", gen_out);

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

    // 8. Destroy brain and cleanup
    brain_destroy(brain);
    printf("[+] Brain destroyed cleanly.\n");

    printf("[SUCCESS] All PopCount Second Brain C-ABI tests passed!\n");
    return 0;
}

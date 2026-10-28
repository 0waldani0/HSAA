#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <string>
#include "popcnt_brain.h"

int main() {
    printf("[*] Starting PopCount Second Brain C-ABI Integration Test...\n");

    // 1. Create instance (4,096 pages = 16MB)
    remove("build/test_brain.bin");
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

    // 8. Verify Partitioned Holographic Superposition Clean-Up beyond the D=512 Crosstalk Wall (300 rules)
    for (int i = 0; i < 300; ++i) {
        char s_buf[64], r_buf[64], o_buf[64];
        snprintf(s_buf, sizeof(s_buf), "Holo_Concept_Node_%03d", i);
        snprintf(r_buf, sizeof(r_buf), "binds_superposition_slot_%03d", i % 25);
        snprintf(o_buf, sizeof(o_buf), "Clean_Payload_Vector_%03d", i);
        int st = brain_store_fact(brain, s_buf, r_buf, o_buf);
        assert(st == 1 && "brain_store_fact failed during 300-rule holographic superposition test");
    }
    for (int i = 0; i < 300; i += 19) {
        char s_low[64], r_buf[64], expected[64], got[256] = {0};
        // Lowercase subject forces direct_facts miss -> VSA bank + Holographic Superposition Clean-Up
        snprintf(s_low, sizeof(s_low), "holo_concept_node_%03d", i);
        snprintf(r_buf, sizeof(r_buf), "binds_superposition_slot_%03d", i % 25);
        snprintf(expected, sizeof(expected), "Clean_Payload_Vector_%03d", i);
        int qh = brain_query_relation(brain, s_low, r_buf, got, sizeof(got));
        assert(qh == 1 && "Holographic superposition clean-up failed under 300-rule load");
        assert(strcmp(got, expected) == 0 && "Crosstalk corruption detected in 300-rule VSA superposition");
    }
    printf("[+] Partitioned Holographic Superposition Clean-Up verified (300 rules, 0 crosstalk errors).\n");

    // 9. Verify Multi-Hop Logical & Taxonomic Deduction (2-hop and 3-hop chains) in brain_query_relation & brain_check_conflict
    // Chain: Нейро_Ядро_X7 --[является_подклассом]--> Квантовый_Кластер --[входит_в_состав]--> Тор_5D --[имеет_топологию]--> Замкнутое_Кольцо_Омега
    brain_store_fact(brain, "Нейро_Ядро_X7", "является_подклассом", "Квантовый_Кластер");
    brain_store_fact(brain, "Квантовый_Кластер", "входит_в_состав", "Тор_5D");
    brain_store_fact(brain, "Тор_5D", "имеет_топологию", "Замкнутое_Кольцо_Омега");
    brain_store_fact(brain, "Квантовый_Кластер", "работает_на_частоте", "950_Терагерц");

    char hop2_ans[256] = {0};
    int q_hop2 = brain_query_relation(brain, "Нейро_Ядро_X7", "работает_на_частоте", hop2_ans, sizeof(hop2_ans));
    assert(q_hop2 == 1 && "2-hop taxonomic property inheritance failed");
    assert(strcmp(hop2_ans, "950_Терагерц") == 0 && "2-hop deduced object mismatch");

    char hop3_ans[256] = {0};
    int q_hop3 = brain_query_relation(brain, "Нейро_Ядро_X7", "имеет_топологию", hop3_ans, sizeof(hop3_ans));
    assert(q_hop3 == 1 && "3-hop transitive logical deduction failed");
    assert(strcmp(hop3_ans, "Замкнутое_Кольцо_Омега") == 0 && "3-hop deduced object mismatch");

    int multihop_conflict = brain_check_conflict(brain, "Нейро_Ядро_X7", "имеет_топологию", "Плоская_Решетка");
    assert(multihop_conflict == 1 && "brain_check_conflict failed to detect 3-hop logical contradiction");
    printf("[+] Multi-hop logical deduction verified: 2-hop='%s', 3-hop='%s' (conflict check active).\n",
           hop2_ans, hop3_ans);

    // 10. Verify Distributional OOV Output Priors & Scale Profiler via C-ABI (brain_profile_heldout_and_scale)
    const char* oov_corpus =
        "Quantum router initializes coherent channel and validates coherent channel.\n"
        "Quantum router initializes photonic channel and validates photonic channel.\n"
        "Hardware controller calibrates cryogenic sensor for low latency.\n"
        "Hardware controller calibrates magnetic sensor for low latency.\n";
    BrainTrainStats oov_st{};
    int tr_oov = brain_train_corpus(brain, oov_corpus, 3, &oov_st);
    assert(tr_oov == 1 && "brain_train_corpus failed on OOV prior test");

    const char* heldout_oov =
        "Quantum router initializes holographic channel and validates holographic channel.\n"
        "Hardware controller calibrates superconducting sensor for low latency.\n";
    BrainScaleAndHeldOutProfile prof{};
    bool prof_ok = brain_profile_heldout_and_scale(brain, heldout_oov, &prof);
    assert(prof_ok && "brain_profile_heldout_and_scale returned false");
    assert(prof.heldout_unreachable_before_pct >= 18.0 && "Expected raw held-out unreachable >= 18%");
    assert(prof.heldout_unreachable_after_pct <= 2.0 && "Expected prior held-out unreachable <= 2%");
    assert(prof.pages_over_16_slots == 0 && "Expected zero pages over 16 slots");
    assert(prof.avx512_l1_page_eval_ns > 0.0 && prof.avx512_dram_page_eval_ns > 0.0 && "Expected non-zero AVX-512 cache profile");
    printf("[+] Held-Out OOV Prior & Scale Profile verified: raw_unreach=%.1f%% -> prior_unreach=%.1f%%, L1=%.1f ns, DRAM=%.1f ns.\n",
           prof.heldout_unreachable_before_pct, prof.heldout_unreachable_after_pct,
           prof.avx512_l1_page_eval_ns, prof.avx512_dram_page_eval_ns);

    // 11. Verify Foundation Backprop Continuous Manifold Training & Continuous Paraphrasing
    assert(oov_st.backprop_initial_loss > oov_st.backprop_final_loss && "Expected backprop loss reduction during train_corpus");
    float bp_init = 0.0f, bp_final = 0.0f;
    int bp_ok = brain_train_foundation_backprop(brain, NULL, 12, 0.25f, &bp_init, &bp_final);
    assert(bp_ok == 1 && bp_final < oov_st.backprop_initial_loss && "brain_train_foundation_backprop failed to reduce loss");
    printf("[+] Foundation Analytical Backprop verified: initial_loss=%.4f -> final_loss=%.4f\n",
           oov_st.backprop_initial_loss, bp_final);

    char para_out[256] = {0};
    int para_n = brain_paraphrase_continuous(
        brain,
        "Quantum router initializes coherent channel for low latency.",
        para_out, sizeof(para_out), 0.75f);
    assert(para_n >= 5 && strcmp(para_out, "Quantum router initializes coherent channel for low latency.") != 0 &&
           "Expected brain_paraphrase_continuous to produce a distinct semantic paraphrase");
    printf("[+] Continuous Semantic Paraphrasing verified: '%s'\n", para_out);

    // 12. Verify Continuous VSA Function Approximation (y = (x + 1)^2) & Template-Free Relation Inference
    float xs[9], ys[9];
    for (int i = 0; i < 9; ++i) {
        xs[i] = -2.0f + 0.5f * (float)i;
        ys[i] = xs[i] * xs[i] + 2.0f * xs[i] + 1.0f; // (x + 1)^2
    }
    int fn_ok = brain_learn_continuous_function(brain, "quadratic_energy", xs, ys, 9, 1, 1);
    assert(fn_ok == 1 && "brain_learn_continuous_function failed");
    float x_q = 0.25f, y_pred = 0.0f;
    int ev_ok = brain_eval_continuous_function(brain, "quadratic_energy", &x_q, 1, &y_pred, 1);
    float y_true = 1.25f * 1.25f; // 1.5625
    float abs_err = (y_pred > y_true) ? (y_pred - y_true) : (y_true - y_pred);
    assert(ev_ok == 1 && abs_err < 0.08f && "brain_eval_continuous_function error too large on unseen x=0.25");

    // Also verify automatic continuous interpolation inside brain_query_relation for unseen numeric subject "0.75"
    char cont_q_ans[64] = {0};
    int cq_ok = brain_query_relation(brain, "0.75", "quadratic_energy", cont_q_ans, sizeof(cont_q_ans));
    assert(cq_ok == 1 && cont_q_ans[0] != '\0' && "brain_query_relation failed to interpolate continuous function");
    printf("[+] Continuous VSA Function Approximation verified: f(0.25)=%.4f (true=%.4f, err=%.4f), query('0.75')->%s\n",
           y_pred, y_true, abs_err, cont_q_ans);

    char inf_obj[128] = {0};
    int inf_ok = brain_infer_template_free_relation(
        brain,
        "При критическом охлаждении сверхпроводящий_соленоид стабилизирует плазменное_кольцо.",
        "сверхпроводящий_соленоид",
        inf_obj, sizeof(inf_obj));
    assert(inf_ok == 1 && strstr(inf_obj, "плазменное_кольцо") != NULL && "brain_infer_template_free_relation failed");
    printf("[+] Template-Free VSA Relation Inference verified: subject='сверхпроводящий_соленоид' -> target='%s'\n",
           inf_obj);

    // 13. Verify 4096-Dimensional Dense Float32 Embeddings & Cosine Similarity
    float d4096_a[BRAIN_DENSE_4096_DIM] = {0};
    float d4096_b[BRAIN_DENSE_4096_DIM] = {0};
    int d_ok1 = brain_embed_dense_4096(brain, "Quantum router initializes coherent channel", d4096_a);
    int d_ok2 = brain_embed_dense_4096(brain, "Quantum router initializes photonic channel", d4096_b);
    assert(d_ok1 == 1 && d_ok2 == 1 && "brain_embed_dense_4096 failed");
    float cos_ab = brain_cosine_vec_4096(d4096_a, d4096_b);
    float cos_unrel = brain_cosine_similarity_4096(
        brain,
        "Quantum router initializes coherent channel",
        "completely unrelated culinary recipe with olive oil");
    assert(cos_ab > 0.70f && cos_ab > cos_unrel + 0.35f && "4096-D dense float32 cosine similarity failed semantic ranking");
    printf("[+] 4096-D Dense Float32 Cosine verified: cos_related=%.4f > cos_unrelated=%.4f\n", cos_ab, cos_unrel);

    // 14. Verify Compositional Zero-Shot Algorithm Synthesis (Dijkstra+RBTree, Ukkonen+Hash, Tarjan Articulation)
    char synth_code[8192] = {0};
    bool exec_verified = false;
    int synth_toks = brain_synthesize_algorithm_zeroshot(
        brain,
        "Напиши алгоритм Дейкстры на красно-черном дереве (dijkstra_rbtree)",
        synth_code, sizeof(synth_code), &exec_verified);
    assert(synth_toks > 100 && exec_verified && "brain_synthesize_algorithm_zeroshot failed");
    assert(strstr(synth_code, "RedBlackPriorityQueue") != NULL &&
           strstr(synth_code, "rotate_left") != NULL &&
           strstr(synth_code, "rotate_right") != NULL &&
           strstr(synth_code, "dist[u] + w < dist[v]") != NULL &&
           "Synthesized code missing Red-Black Tree rotations or Dijkstra edge relaxation");

    char ukk_code[8192] = {0};
    bool ukk_verified = false;
    int ukk_toks = brain_synthesize_algorithm_zeroshot(
        brain,
        "Суффиксный автомат Укконена с хешированием",
        ukk_code, sizeof(ukk_code), &ukk_verified);
    assert(ukk_toks > 100 && ukk_verified &&
           strstr(ukk_code, "UkkonenSuffixAutomaton") != NULL &&
           strstr(ukk_code, "OpenAddressedTransitionHash") != NULL &&
           "Ukkonen + Hashing synthesis failed");

    char tarjan_code[8192] = {0};
    bool tarjan_verified = false;
    int tarjan_toks = brain_synthesize_algorithm_zeroshot(
        brain,
        "Алгоритм Тарьяна поиска точек сочленения",
        tarjan_code, sizeof(tarjan_code), &tarjan_verified);
    assert(tarjan_toks > 100 && tarjan_verified &&
           strstr(tarjan_code, "TarjanArticulationSolver") != NULL &&
           strstr(tarjan_code, "low[to] >= tin[v]") != NULL &&
           "Tarjan Articulation Points synthesis failed");
    printf("[+] Zero-Shot Compositional Synthesis verified: RBTree+Dijkstra (%d toks), Ukkonen+Hash (%d toks), Tarjan (%d toks)\n",
           synth_toks, ukk_toks, tarjan_toks);

    // 15. Verify Full O(N^2) RoPE Causal Self-Attention over Long Unstructured Narrative (>220 tokens)
    std::string long_story =
        "В первой главе старый капитан Артур спрятал сапфировый хронометр внутри северного маяка. ";
    for (int k = 0; k < 28; ++k) {
        long_story += "Корабль шел сквозь густой туман, волны били о борт, матросы проверяли паруса и канаты. ";
    }
    char long_ans[512] = {0};
    double max_lr_attn = 0.0;
    int attn_toks = brain_attend_long_context(
        brain,
        long_story.c_str(),
        "Где капитан Артур спрятал сапфировый хронометр?",
        long_ans, sizeof(long_ans), 24, &max_lr_attn);
    assert(attn_toks >= 4 && max_lr_attn > 0.10 && "brain_attend_long_context failed on >220-token narrative");
    assert((strstr(long_ans, "маяка") != NULL || strstr(long_ans, "хронометр") != NULL) &&
           "Long-context O(N^2) attention failed to retrieve early narrative plotline (>200 tokens back)");
    printf("[+] Full O(N^2) Long-Context RoPE Attention verified: max_long_range_attn=%.4f, recalled='%s'\n",
           max_lr_attn, long_ans);

    // 16. Verify Hierarchical Factorized Continuous Manifold Scaling beyond 50,000 Words
    size_t active_vocab = 0, num_clusters = 0;
    double hier_init_loss = 0.0, hier_final_loss = 0.0;
    bool scale_ok = brain_manifold_scale_vocabulary(
        brain, 52000, 6, 0.15f, &active_vocab, &num_clusters, &hier_init_loss, &hier_final_loss);
    assert(scale_ok && active_vocab >= 52000 && num_clusters == 256 && hier_final_loss < hier_init_loss &&
           "brain_manifold_scale_vocabulary failed to scale continuous manifold beyond 50,000 tokens");
    printf("[+] Hierarchical Continuous Manifold (>50k vocab) verified: V=%zu, clusters=%zu, loss=%.4f -> %.4f\n",
           active_vocab, num_clusters, hier_init_loss, hier_final_loss);

    // 17. Verify CFG/EBNF Type-Scoped Constrained Decoding & Closed-Loop CEGIS with Simplex Contradiction Gate
    char cfg_code[4096] = {0};
    int cfg_masked = 0;
    bool cfg_type_safe = false;
    int cfg_toks = brain_generate_code_cfg_constrained(
        brain,
        "int64_t accumulate_sensor_frames(const int32_t* frames, size_t frame_count)",
        cfg_code, sizeof(cfg_code), 128,
        &cfg_masked, &cfg_type_safe);
    assert(cfg_toks > 25 && cfg_masked >= 8 && cfg_type_safe &&
           strstr(cfg_code, "std::string") == NULL &&
           "brain_generate_code_cfg_constrained failed grammar/type safety check");

    char cegis_code[8192] = {0};
    BrainCegisSynthesisReport cegis_rep = {0};
    int cegis_toks = brain_synthesize_cegis_sketch(
        brain,
        "TwoPointers sliding window: максимальное окно подотрезка с суммой не более limit",
        3, cegis_code, sizeof(cegis_code), &cegis_rep);
    assert(cegis_toks > 40 &&
           cegis_rep.selected_archetype == BRAIN_SKETCH_TWO_POINTERS_WINDOW &&
           cegis_rep.candidates_evaluated == 3 &&
           cegis_rep.counterexamples_triggered == 2 &&
           cegis_rep.simplex_contradictions_masked == 2 &&
           cegis_rep.sandbox_01n_null_verified &&
           cegis_rep.crystallized_into_vsa &&
           "brain_synthesize_cegis_sketch failed 0-1-N-NULL CEGIS loop");

    BrainCegisSynthesisReport recall_rep = {0};
    int recall_toks = brain_synthesize_cegis_sketch(
        brain,
        "TwoPointers sliding window: повторный вызов после VSA кристаллизации",
        3, cegis_code, sizeof(cegis_code), &recall_rep);
    assert(recall_toks > 40 &&
           recall_rep.recalled_from_crystal &&
           recall_rep.candidates_evaluated == 1 &&
           recall_rep.counterexamples_triggered == 0 &&
           "CEGIS VSA crystal fast-path recall failed");
    printf("[+] CFG/EBNF Type-Scoped Decoding (masked=%d) & Closed-Loop CEGIS (counterexamples=%d, crystal_recall=1-step) verified.\n",
           cfg_masked, cegis_rep.counterexamples_triggered);

    // 18. Destroy brain and cleanup
    brain_destroy(brain);
    printf("[+] Brain destroyed cleanly.\n");

    printf("[SUCCESS] All PopCount Second Brain C-ABI tests passed!\n");
    return 0;
}


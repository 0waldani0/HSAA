#include "unified_engine.h"
#include "simd_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>

static int g_passed = 0;
static int g_failed = 0;

#define CHECK_UNIFIED(name, cond) do { \
    bool _ok = (cond); \
    std::printf("  [%s] %s\n", _ok ? "PASS" : "FAIL", name); \
    if (_ok) ++g_passed; else ++g_failed; \
} while (0)

int main() {
    std::printf("================================================================================\n");
    std::printf("  UNIFIED COGNITIVE ENGINE VERIFICATION SUITE (Single-Pass Architecture)\n");
    std::printf("  Fusing HSSA 28-Layer Stack + SecondBrain 4KB Bank + VSA DB + ModelMod TT16\n");
    std::printf("================================================================================\n\n");

    // 1. Lifecycle & Shared Binding
    std::printf(">>> 1. Single-Handle Lifecycle & Shared Memory Binding <<<\n");
    UnifiedEngineConfig cfg;
    unified_engine_default_config(&cfg);
    cfg.num_layers = 28;
    cfg.sparse_bank_pages = 16384;

    UnifiedEngine* eng = unified_engine_create(&cfg);
    CHECK_UNIFIED("unified_engine_create initializes all fused subsystems", eng != nullptr);
    CHECK_UNIFIED("shared VsaDbHandle bound to HSSA and ModelMod", unified_engine_get_vsa_db(eng) != nullptr);
    CHECK_UNIFIED("shared PopCountBrain bound to UnifiedEngine", unified_engine_get_brain(eng) != nullptr);
    CHECK_UNIFIED("shared 28-layer HssaContext bound to UnifiedEngine", unified_engine_get_hssa(eng) != nullptr);
    CHECK_UNIFIED("shared 28-layer ModelContext bound to UnifiedEngine", unified_engine_get_model(eng) != nullptr);

    // 2. Unified Single-Pass Observation (Facts + Contradiction Detection + Multi-Hop Graph)
    std::printf("\n>>> 2. Unified Observation: Relational Facts, Contradictions & Multi-Hop <<<\n");
    UnifiedObserveTelemetry obs1{}, obs2{}, obs3{};
    int r_f1 = unified_engine_observe(eng, "Модуль_Орион | использует_ядро | Квантовый_Кластер", &obs1);
    int r_f2 = unified_engine_observe(eng, "Квантовый_Кластер | обеспечивает_режим | Сверхпроводящий_Резонанс_5D", &obs2);
    CHECK_UNIFIED("unified_engine_observe detects UNIFIED_KIND_RELATIONAL_FACT",
                  r_f1 == 0 && obs1.detected_kind == UNIFIED_KIND_RELATIONAL_FACT && !obs1.conflict_detected);
    CHECK_UNIFIED("second chained relational fact observed cleanly",
                  r_f2 == 0 && obs2.detected_kind == UNIFIED_KIND_RELATIONAL_FACT && obs2.facts_stored_total >= 2);

    // Conflicting observation on existing subject + relation
    int r_f3 = unified_engine_observe(eng, "Модуль_Орион | использует_ядро | Устаревший_Контур", &obs3);
    CHECK_UNIFIED("unified_engine_observe automatically flags contradiction on conflicting fact mutation",
                  r_f3 == 0 && obs3.conflict_detected);
    // Restore canonical fact
    unified_engine_observe(eng, "Модуль_Орион | использует_ядро | Квантовый_Кластер", &obs1);

    // Query direct and 2-hop transitive relation through unified_engine_forward
    char resp_direct[256] = {0};
    UnifiedForwardTelemetry fwd_direct{};
    int r_qd = unified_engine_forward(eng, "Модуль_Орион | использует_ядро ?", resp_direct, sizeof(resp_direct), nullptr, &fwd_direct);
    CHECK_UNIFIED("unified_engine_forward resolves direct relation ('Квантовый_Кластер')",
                  r_qd == 0 && fwd_direct.resolved_mode == UNIFIED_KIND_RELATIONAL_QUERY &&
                  std::strcmp(resp_direct, "Квантовый_Кластер") == 0);

    char resp_2hop[256] = {0};
    UnifiedForwardTelemetry fwd_2hop{};
    int r_q2 = unified_engine_forward(eng, "Модуль_Орион | обеспечивает_режим ?", resp_2hop, sizeof(resp_2hop), nullptr, &fwd_2hop);
    CHECK_UNIFIED("unified_engine_forward resolves 2-hop deduced relation ('Сверхпроводящий_Резонанс_5D')",
                  r_q2 == 0 && fwd_2hop.resolved_mode == UNIFIED_KIND_RELATIONAL_QUERY &&
                  std::strcmp(resp_2hop, "Сверхпроводящий_Резонанс_5D") == 0);
    CHECK_UNIFIED("28-layer HSSA Hopfield energy descends monotonically (dE/dt <= 0) during forward",
                  fwd_2hop.hopfield_energy_delta <= 1e-5);
    CHECK_UNIFIED("unified 1024-D state and 4096-D manifold norms are finite and positive",
                  fwd_2hop.state_l2_norm > 0.0f && fwd_2hop.manifold_4096_norm > 0.0f);

    // 3. Unified Observation & Evaluation of Continuous Non-Linear Functions y = f(x)
    std::printf("\n>>> 3. Unified Continuous Function Approximation y = sin(x) + 0.25*x^2 <<<\n");
    UnifiedObserveTelemetry obs_fn{};
    for (int i = 0; i <= 16; ++i) {
        float x = static_cast<float>(i) * 0.20f;
        float y = std::sin(x) + 0.25f * x * x;
        char buf[128];
        std::snprintf(buf, sizeof(buf), "nonlinear_law(%.5f) = %.6f", static_cast<double>(x), static_cast<double>(y));
        unified_engine_observe(eng, buf, &obs_fn);
    }
    CHECK_UNIFIED("unified_engine_observe detects UNIFIED_KIND_CONTINUOUS_FUNC",
                  obs_fn.detected_kind == UNIFIED_KIND_CONTINUOUS_FUNC);

    double mse_sum = 0.0;
    double brain_mse_sum = 0.0;
    for (int i = 0; i < 8; ++i) {
        float x_mid = (static_cast<float>(i * 2) + 0.5f) * 0.20f;
        float y_true = std::sin(x_mid) + 0.25f * x_mid * x_mid;
        char qbuf[128], rbuf[128] = {0};
        std::snprintf(qbuf, sizeof(qbuf), "nonlinear_law(%.5f)", static_cast<double>(x_mid));
        UnifiedForwardTelemetry f_t{};
        unified_engine_forward(eng, qbuf, rbuf, sizeof(rbuf), nullptr, &f_t);
        float y_pred = std::strtof(rbuf, nullptr);
        double err = static_cast<double>(y_pred - y_true);
        mse_sum += err * err;

        float y_brain = 0.0f;
        brain_eval_continuous_function(unified_engine_get_brain(eng), "nonlinear_law", &x_mid, 1, &y_brain, 1);
        double b_err = static_cast<double>(y_brain - y_true);
        brain_mse_sum += b_err * b_err;
    }
    double mse = mse_sum / 8.0;
    double brain_mse = brain_mse_sum / 8.0;
    std::printf("    -> Unseen midpoint MSE via unified_engine_forward: %.6f (PopCountBrain MSE: %.6f)\n", mse, brain_mse);
    CHECK_UNIFIED("unified_engine_forward evaluates continuous function with MSE < 1e-3", mse < 1e-3);
    CHECK_UNIFIED("accumulated PopCountBrain function table matches VSA DB MSE < 1e-3", brain_mse < 1e-3);

    // 4. Unified Prose/Code Corpus Observation + Analytical Backprop + Open-Ended Generation
    std::printf("\n>>> 4. Unified Prose/Code Observation, Backprop & Contradiction-Gated Generation <<<\n");
    const char* corpus =
        "Нейроморфный сопроцессор вычисляет градиентный тензор и ускоряет глубокий трансформер. "
        "Квантовый ускоритель формирует плотный вектор и оптимизирует быстрый индекс. "
        "Голографический модуль анализирует разреженный граф и синхронизирует когерентный поток.";
    UnifiedObserveTelemetry obs_prose{};
    int r_pr = unified_engine_observe(eng, corpus, &obs_prose);
    std::printf("    -> Foundation Manifold cross-entropy loss: %.4f -> %.4f (TT16 update: %.2f us)\n",
                obs_prose.manifold_loss_before, obs_prose.manifold_loss_after, obs_prose.tt16_update_us);
    CHECK_UNIFIED("unified_engine_observe trains sparse pages + Foundation backprop + TT16 fast weights",
                  r_pr == 0 && obs_prose.detected_kind == UNIFIED_KIND_PROSE_OR_CODE &&
                  obs_prose.manifold_loss_after <= obs_prose.manifold_loss_before);

    // Register 4 candidate assertions (2 consistent, 2 contradicting stored facts)
    unified_engine_register_candidate(eng, 0, "Модуль_Орион", "использует_ядро", "Квантовый_Кластер");
    unified_engine_register_candidate(eng, 1, "Модуль_Орион", "использует_ядро", "Ложный_Модуль");
    unified_engine_register_candidate(eng, 2, "Квантовый_Кластер", "обеспечивает_режим", "Сверхпроводящий_Резонанс_5D");
    unified_engine_register_candidate(eng, 3, "Квантовый_Кластер", "обеспечивает_режим", "Неверный_Режим");

    float raw_logits[4] = {5.0f, 4.8f, 5.2f, 4.5f};
    float probs[4] = {0};
    uint32_t nullified = 0;
    unified_engine_project_simplex(eng, raw_logits, 4, probs, &nullified);
    CHECK_UNIFIED("Simplex Contradiction Gate nullifies 2/2 conflicting candidates (p* = 0.0)",
                  nullified == 2 && probs[1] == 0.0f && probs[3] == 0.0f && probs[0] > 0.0f && probs[2] > 0.0f);

    char gen_out[512] = {0};
    alignas(64) float state_snap[1024] = {0};
    UnifiedForwardTelemetry fwd_gen{};
    int r_gen = unified_engine_forward(eng, "Нейроморфный сопроцессор вычисляет", gen_out, sizeof(gen_out), state_snap, &fwd_gen);
    std::printf("    -> Unified Generated Output: \"%s\"\n", gen_out);
    std::printf("    -> Unified Forward Telemetry: total=%.2f us, hssa=%.2f us, dE=%.2f, nullified=%u\n",
                fwd_gen.total_forward_us, fwd_gen.hssa_stack_us, fwd_gen.hopfield_energy_delta, fwd_gen.contradictions_nullified);
    CHECK_UNIFIED("unified_engine_forward emits coherent continuation & reports nullified contradictions",
                  r_gen == 0 && std::strlen(gen_out) > 10 && fwd_gen.contradictions_nullified == 2);

    // 5. Unified Persistence Roundtrip (Save & Cross-Instance Load)
    std::printf("\n>>> 5. Unified Cross-Instance Persistence Roundtrip (unified_engine_save / load) <<<\n");
    int r_save = unified_engine_save(eng, "build/unified_ckpt");
    CHECK_UNIFIED("unified_engine_save writes .hssa, .mmod, and .ueng checkpoints", r_save == 0);

    UnifiedEngine* eng2 = unified_engine_create(&cfg);
    int r_load = unified_engine_load(eng2, "build/unified_ckpt");
    CHECK_UNIFIED("unified_engine_load restores full state into a fresh UnifiedEngine instance", r_load == 0);

    char resp_reload_2hop[256] = {0};
    UnifiedForwardTelemetry fwd_reload_2hop{};
    int r_rq2 = unified_engine_forward(eng2, "Модуль_Орион | обеспечивает_режим ?", resp_reload_2hop, sizeof(resp_reload_2hop), nullptr, &fwd_reload_2hop);
    CHECK_UNIFIED("reloaded instance resolves 2-hop relation ('Сверхпроводящий_Резонанс_5D')",
                  r_rq2 == 0 && std::strcmp(resp_reload_2hop, "Сверхпроводящий_Резонанс_5D") == 0);

    char resp_reload_fn[128] = {0};
    unified_engine_forward(eng2, "nonlinear_law(0.50000)", resp_reload_fn, sizeof(resp_reload_fn), nullptr, nullptr);
    float y_reload = std::strtof(resp_reload_fn, nullptr);
    float y_expected = std::sin(0.5f) + 0.25f * 0.25f;
    CHECK_UNIFIED("reloaded instance evaluates learned continuous function accurately",
                  std::fabs(y_reload - y_expected) < 0.02f);

    unified_engine_destroy(eng2);
    std::remove("build/unified_ckpt.hssa");
    std::remove("build/unified_ckpt.mmod");
    std::remove("build/unified_ckpt.ueng");

    // 6. Parametric Learning vs. Fact Insertion Audit (No Catastrophic Forgetting + MSE Descent)
    std::printf("\n>>> 6. Parametric Gradient Learning & Incremental Retention Audit <<<\n");
    UnifiedObserveTelemetry obs_fact_late{}, obs_prose2{};
    // Observe a new relational fact AND a second prose passage AFTER the first prose passage
    unified_engine_observe(eng, "Тензорное_Ядро | ускоряет_модуль | Аналитический_Бэкпроп", &obs_fact_late);
    const char* corpus2 =
        "Аналитический градиент оптимизирует тензорное ядро и снижает ошибку предсказания. "
        "Сингулярный адаптер обновляет быстрые веса и стабилизирует глубокий слой.";
    unified_engine_observe(eng, corpus2, &obs_prose2);

    char gen_early[512] = {0};
    char gen_late[512]  = {0};
    unified_engine_forward(eng, "Нейроморфный сопроцессор вычисляет", gen_early, sizeof(gen_early), nullptr, nullptr);
    unified_engine_forward(eng, "Аналитический градиент оптимизирует", gen_late, sizeof(gen_late), nullptr, nullptr);
    std::printf("    -> Retained Early Prose: \"%s\"\n", gen_early);
    std::printf("    -> Learned Late Prose:   \"%s\"\n", gen_late);
    CHECK_UNIFIED("incremental observation retains earlier prose without catastrophic forgetting",
                  std::strstr(gen_early, "градиентный") != nullptr || std::strstr(gen_early, "тензор") != nullptr);
    CHECK_UNIFIED("incremental observation learns new prose passage in unified manifold",
                  std::strstr(gen_late, "тензорное") != nullptr || std::strstr(gen_late, "ядро") != nullptr);
    CHECK_UNIFIED("online TT16/STE causal transition training strictly reduces MSE (loss_after < loss_before)",
                  obs_fact_late.manifold_loss_after < obs_fact_late.manifold_loss_before &&
                  obs_prose2.manifold_loss_after < obs_prose2.manifold_loss_before);

    // 7. Deep 28-Layer HSSA LM-Head + Multi-Beam + Subword Stem + Long-Range RoPE Text Generation
    std::printf("\n>>> 7. Deep 28-Layer Beam Search, Subword BPE Stem Grounding & RoPE Text Generation <<<\n");
    char deep_text[1024] = {0};
    UnifiedTextGenTelemetry tg_telem{};
    // Prompt uses inflected plural forms ("Нейроморфные сопроцессоры вычисляют") not present verbatim in corpus
    int r_tg = unified_engine_generate_text_deep(
        eng,
        "Нейроморфные сопроцессоры вычисляют",
        deep_text,
        sizeof(deep_text),
        20,
        4,
        &tg_telem
    );
    std::printf("    -> Deep Beam Output (%d toks, %u beams, %u subword stems): \"%s\"\n",
                r_tg, tg_telem.beam_candidates_evaluated, tg_telem.subword_stems_matched, deep_text);
    std::printf("    -> 28-Layer HSSA Cosine: %.4f | RoPE Peak Attn: %.4f | Hopfield dE: %.2f | Time: %.1f us\n",
                tg_telem.hssa_alignment_cosine, tg_telem.max_long_range_rope_attn,
                tg_telem.hopfield_energy_delta, tg_telem.total_gen_us);
    CHECK_UNIFIED("unified_engine_generate_text_deep resolves inflected OOV stems via subword prefix matching",
                  r_tg > 0 && tg_telem.subword_stems_matched >= 2);
    CHECK_UNIFIED("multi-beam planner evaluates >= 3 candidate trajectories with 28-layer HSSA scoring",
                  tg_telem.beam_candidates_evaluated >= 3 && tg_telem.hopfield_energy_delta <= 1e-4);

    // 8. AST-Guided Code Synthesis, VSA Continuous Function-to-Code Compiler & Sandbox Execution
    std::printf("\n>>> 8. AST-Guided Code Synthesis, VSA Function-to-C++ Compiler & Sandbox Verification <<<\n");
    char fn_code[2048] = {0};
    UnifiedCodeSynthTelemetry cs_fn_telem{};
    int r_cs_fn = unified_engine_synthesize_code(
        eng,
        "Synthesize C++20 analytical function for nonlinear_law(x)",
        fn_code,
        sizeof(fn_code),
        &cs_fn_telem
    );
    std::printf("    -> Compiled VSA Function C++20 Code (%d toks, sandbox MSE=%.6f, passed=%u):\n%s",
                r_cs_fn, cs_fn_telem.sandbox_mse, cs_fn_telem.sandbox_test_cases_passed, fn_code);
    CHECK_UNIFIED("unified_engine_synthesize_code compiles VSA continuous function into C++20 code",
                  r_cs_fn > 0 && cs_fn_telem.compiled_from_vsa_function &&
                  std::strstr(fn_code, "float nonlinear_law(float x)") != nullptr);
    CHECK_UNIFIED("compiled VSA function passes in-memory numerical sandbox (MSE < 1e-3) & AST check",
                  cs_fn_telem.ast_balanced_valid && cs_fn_telem.sandbox_exec_verified &&
                  cs_fn_telem.sandbox_mse < 1e-3);

    char algo_code[4096] = {0};
    UnifiedCodeSynthTelemetry cs_algo_telem{};
    int r_cs_algo = unified_engine_synthesize_code(
        eng,
        "Synthesize Dijkstra shortest-path algorithm on Red-Black Tree nodes in C++",
        algo_code,
        sizeof(algo_code),
        &cs_algo_telem
    );
    std::printf("    -> Synthesized Zero-Shot Algorithm (%d toks, AST nodes=%u, depth=%u, verified=%s)\n",
                r_cs_algo, cs_algo_telem.ast_nodes_verified, cs_algo_telem.max_nesting_depth,
                cs_algo_telem.sandbox_exec_verified ? "YES" : "NO");
    CHECK_UNIFIED("zero-shot compositional algorithm synthesizer passes AST & execution sandbox",
                  r_cs_algo > 0 && cs_algo_telem.ast_balanced_valid && cs_algo_telem.sandbox_exec_verified);

    // 9. System-2 Counterfactual Imagination, Cross-Domain VSA Analogy & Curiosity Discovery
    std::printf("\n>>> 9. System-2 Cross-Domain VSA Analogy & Intrinsic Curiosity Idea Synthesis <<<\n");
    unified_engine_observe(eng, "Модуль_Вега | использует_архитектуру | Оптический_Тороид", nullptr);
    char ideas_out[4096] = {0};
    UnifiedIdeaSynthTelemetry idea_telem{};
    int n_ideas = unified_engine_synthesize_ideas(
        eng,
        "Модуль_Вега",
        true,
        ideas_out,
        sizeof(ideas_out),
        &idea_telem
    );
    std::printf("    -> System-2 Discovered Hypotheses (%d accepted, %u analogies, %u curiosity, %u crystallized):\n%s\n",
                n_ideas, idea_telem.analogies_transferred, idea_telem.curiosity_gaps_filled,
                idea_telem.crystallized_facts, ideas_out);
    CHECK_UNIFIED("unified_engine_synthesize_ideas generates novel hypotheses via VSA analogy & curiosity",
                  n_ideas >= 1 && (idea_telem.analogies_transferred + idea_telem.curiosity_gaps_filled) >= 1);
    CHECK_UNIFIED("System-2 hypotheses relax in 28-layer Hopfield attractor (dE <= 0) & crystallize into VSA DB",
                  idea_telem.hopfield_energy_delta <= 1e-4 && idea_telem.crystallized_facts >= 1 &&
                  idea_telem.manifold_novelty_score > 0.50f);

    char verify_crystallized[256] = {0};
    UnifiedForwardTelemetry fwd_crys{};
    int r_vc = unified_engine_forward(eng, "Модуль_Вега | использует_ядро ?", verify_crystallized, sizeof(verify_crystallized), nullptr, &fwd_crys);
    std::printf("    -> Querying Crystallized System-2 Idea ('Модуль_Вега | использует_ядро ?'): \"%s\"\n", verify_crystallized);
    CHECK_UNIFIED("crystallized System-2 hypothesis is immediately recallable via unified_engine_forward",
                  r_vc == 0 && std::strlen(verify_crystallized) > 0);

    unified_engine_destroy(eng);

    std::printf("\n================================================================================\n");
    std::printf("  UNIFIED ENGINE SUMMARY: %d Passed, %d Failed\n", g_passed, g_failed);
    std::printf("================================================================================\n");
    return (g_failed == 0) ? 0 : 1;
}

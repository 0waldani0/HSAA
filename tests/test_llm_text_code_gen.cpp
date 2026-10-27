#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <fstream>
#include "../second_brain/popcnt_brain.h"

static int g_passed = 0;
static int g_failed = 0;

static void check(bool cond, const char* name, const char* detail = "") {
    if (cond) {
        std::printf("  [PASS] %s\n", name);
        ++g_passed;
    } else {
        std::printf("  [FAIL] %s %s\n", name, detail);
        ++g_failed;
    }
}

static bool has_balanced_delimiters(const std::string& s) {
    std::vector<char> st;
    for (char c : s) {
        if (c == '(' || c == '[' || c == '{') st.push_back(c);
        else if (c == ')') {
            if (st.empty() || st.back() != '(') return false;
            st.pop_back();
        } else if (c == ']') {
            if (st.empty() || st.back() != '[') return false;
            st.pop_back();
        } else if (c == '}') {
            if (st.empty() || st.back() != '{') return false;
            st.pop_back();
        }
    }
    return st.empty();
}

int main() {
    std::printf("================================================================\n");
    std::printf("  PopCount Generative LLM: Text & Code Verification Suite\n");
    std::printf("================================================================\n");

    std::filesystem::create_directories("build");
    PopCountBrain* brain = brain_create(4096, "build/test_llm_gen.bin");
    check(brain != nullptr, "T0: brain_create initializes instance");
    if (!brain) return 1;

    // -------------------------------------------------------------------------
    // TEST 1: Boundary & Adversarial Null/Empty Gate (0-1-N-NULL-BOUND)
    // -------------------------------------------------------------------------
    char buf[4096] = {0};
    check(brain_train_corpus(nullptr, "hello world", 1, nullptr) == 0, "T1.1: null brain rejected in brain_train_corpus");
    check(brain_train_corpus(brain, nullptr, 1, nullptr) == 0, "T1.2: null corpus rejected in brain_train_corpus");
    check(brain_train_corpus(brain, "", 1, nullptr) == 0, "T1.3: empty corpus rejected in brain_train_corpus");
    check(brain_train_file(brain, "build/no_such_corpus_file_999.txt", 1, nullptr) == 0, "T1.4: missing file rejected in brain_train_file");
    check(brain_generate(nullptr, "prompt", buf, sizeof(buf), 16, 0.0f) == 0, "T1.5: null brain rejected in brain_generate");
    check(brain_generate(brain, nullptr, buf, sizeof(buf), 16, 0.0f) == 0, "T1.6: null prompt rejected in brain_generate");
    check(brain_generate(brain, "prompt", nullptr, sizeof(buf), 16, 0.0f) == 0, "T1.7: null buffer rejected in brain_generate");
    check(brain_generate(brain, "prompt", buf, 0, 16, 0.0f) == 0, "T1.8: zero max_len rejected in brain_generate");
    check(brain_generate(brain, "prompt", buf, sizeof(buf), 0, 0.0f) == 0, "T1.9: zero max_tokens rejected in brain_generate");

    // -------------------------------------------------------------------------
    // TEST 2: End-to-End Multi-Domain Corpus Training & Convergence Gate
    // -------------------------------------------------------------------------
    BrainTrainStats stats{};
    int tr_ok = brain_train_file(brain, "second_brain/measur/data/llm_text_code_corpus.txt", 6, &stats);
    std::printf("  [INFO] Trained vocab=%zu, seqs=%zu, tokens=%zu, transitions=%zu, replay_acc=%.2f%%, cov=%.2f%%, time=%.2f ms (%.0f tok/s)\n",
                stats.vocab_size, stats.num_sequences, stats.total_tokens, stats.total_transitions,
                stats.replay_accuracy_percent, stats.shortlist_coverage_percent,
                stats.train_time_ms, stats.tokens_per_sec);

    check(tr_ok == 1, "T2.1: brain_train_file succeeded on mixed prose + code corpus");
    check(stats.vocab_size >= 150 && stats.num_sequences >= 12 && stats.total_tokens >= 300,
          "T2.2: corpus vocabulary and sequence counts populated");
    check(stats.replay_accuracy_percent >= 95.0 && stats.shortlist_coverage_percent >= 99.9,
          "T2.3: multi-horizon + STE replay accuracy >= 95% and coverage == 100%");

    // Small buffer truncation boundary check after training
    char tiny_buf[8] = {'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'};
    int tiny_n = brain_generate(brain, "Архитектура PopCount заменяет", tiny_buf, sizeof(tiny_buf), 16, 0.0f);
    check(tiny_n > 0 && tiny_buf[7] == '\0' && std::strlen(tiny_buf) < 8,
          "T2.4: small output buffer truncates safely with null terminator");

    // -------------------------------------------------------------------------
    // TEST 3: Coherent Multi-Sentence Prose Generation (RU & EN)
    // -------------------------------------------------------------------------
    std::memset(buf, 0, sizeof(buf));
    int n_ru_prefix = brain_generate(brain, "Архитектура PopCount заменяет", buf, sizeof(buf), 64, 0.0f);
    std::string s_ru_prefix(buf);
    std::printf("  [GEN RU Prefix]: %s\n", s_ru_prefix.c_str());
    check(n_ru_prefix >= 20 &&
          s_ru_prefix.find("POPCNT") != std::string::npos &&
          s_ru_prefix.find("512") != std::string::npos &&
          s_ru_prefix.find("процессоре") != std::string::npos,
          "T3.1: coherent multi-sentence Russian prose generated from prefix");

    std::memset(buf, 0, sizeof(buf));
    int n_ru_qa = brain_generate(brain, "Как гипермерная память VSA устраняет галлюцинации?", buf, sizeof(buf), 64, 0.0f);
    std::string s_ru_qa(buf);
    std::printf("  [GEN RU Topic QA]: %s\n", s_ru_qa.c_str());
    check(n_ru_qa >= 20 &&
          s_ru_qa.find("VSA") != std::string::npos &&
          s_ru_qa.find("резонирует") != std::string::npos &&
          s_ru_qa.find("переобучения") != std::string::npos,
          "T3.2: coherent Russian answer launched via 512-bit VSA discourse topic resonance");

    std::memset(buf, 0, sizeof(buf));
    int n_en = brain_generate(brain, "The PopCount generative engine combines", buf, sizeof(buf), 64, 0.0f);
    std::string s_en(buf);
    std::printf("  [GEN EN Prefix]: %s\n", s_en.c_str());
    check(n_en >= 20 &&
          s_en.find("512-bit") != std::string::npos &&
          s_en.find("bitplanes") != std::string::npos &&
          s_en.find("CPU") != std::string::npos,
          "T3.3: coherent multi-sentence English prose generated from prefix");

    // -------------------------------------------------------------------------
    // TEST 4: Multi-Line C++ & Python Code Generation + Syntax Stack Automaton
    // -------------------------------------------------------------------------
    std::memset(buf, 0, sizeof(buf));
    int n_cpp = brain_generate(brain, "int sum_array(const int* arr, int n) {", buf, sizeof(buf), 64, 0.0f);
    std::string s_cpp(buf);
    std::printf("  [GEN C++ Code]:\n%s\n", s_cpp.c_str());
    check(n_cpp >= 15 &&
          has_balanced_delimiters(s_cpp) &&
          s_cpp.find("int total = 0;") != std::string::npos &&
          s_cpp.find("for (int i = 0; i < n; ++i)") != std::string::npos &&
          s_cpp.find("total += arr[i];") != std::string::npos &&
          s_cpp.find("return total;") != std::string::npos,
          "T4.1: multi-line C++ function generated with balanced braces and loop structure");

    std::memset(buf, 0, sizeof(buf));
    int n_py = brain_generate(brain, "def binary_search(arr, target):", buf, sizeof(buf), 80, 0.0f);
    std::string s_py(buf);
    std::printf("  [GEN Python Code]:\n%s\n", s_py.c_str());
    check(n_py >= 25 &&
          has_balanced_delimiters(s_py) &&
          s_py.find("while left <= right:") != std::string::npos &&
          s_py.find("mid = (left + right) // 2") != std::string::npos &&
          s_py.find("return mid") != std::string::npos &&
          s_py.find("return -1") != std::string::npos,
          "T4.2: multi-line Python binary_search generated with valid indentation");

    // -------------------------------------------------------------------------
    // TEST 5: Zero-Shot Variable Renaming via Symbolic Scope-Register Induction
    // -------------------------------------------------------------------------
    std::memset(buf, 0, sizeof(buf));
    int n_zs_py = brain_generate(brain, "def clamp(score, min_score, max_score):", buf, sizeof(buf), 64, 0.0f);
    std::string s_zs_py(buf);
    std::printf("  [GEN Zero-Shot Python Variable Renaming]:\n%s\n", s_zs_py.c_str());
    check(n_zs_py >= 15 &&
          s_zs_py.find("if score < min_score:") != std::string::npos &&
          s_zs_py.find("return min_score") != std::string::npos &&
          s_zs_py.find("if score > max_score:") != std::string::npos &&
          s_zs_py.find("return max_score") != std::string::npos &&
          s_zs_py.find("return score") != std::string::npos &&
          s_zs_py.find("val") == std::string::npos,
          "T5.1: zero-shot Python parameter renaming (score, min_score, max_score) succeeded");

    std::memset(buf, 0, sizeof(buf));
    int n_zs_cpp = brain_generate(brain, "int count_matches(const int* buffer, int length, int needle) {", buf, sizeof(buf), 64, 0.0f);
    std::string s_zs_cpp(buf);
    std::printf("  [GEN Zero-Shot C++ Variable Renaming]:\n%s\n", s_zs_cpp.c_str());
    check(n_zs_cpp >= 15 &&
          has_balanced_delimiters(s_zs_cpp) &&
          s_zs_cpp.find("i < length") != std::string::npos &&
          s_zs_cpp.find("buffer[i] == needle") != std::string::npos &&
          s_zs_cpp.find("return matches;") != std::string::npos &&
          s_zs_cpp.find("arr") == std::string::npos,
          "T5.2: zero-shot C++ parameter renaming (buffer, length, needle) succeeded");

    // -------------------------------------------------------------------------
    // TEST 6: Real-Time Neuro-Symbolic Fact Steering (MUTATION FLIP Proof)
    // -------------------------------------------------------------------------
    std::memset(buf, 0, sizeof(buf));
    brain_generate(brain, "Сервер сервер_альфа использует контур охлаждения", buf, sizeof(buf), 20, 0.0f);
    std::string before_fact(buf);
    check(before_fact.find("криогенный_контур") != std::string::npos,
          "T6.1: baseline before brain_store_fact outputs corpus default (криогенный_контур)");

    // Inject live overriding fact without retraining
    brain_store_fact(brain, "сервер_альфа", "охлаждения", "жидкий_гелий_сверхпроводник");
    std::memset(buf, 0, sizeof(buf));
    brain_generate(brain, "Сервер сервер_альфа использует контур охлаждения", buf, sizeof(buf), 20, 0.0f);
    std::string after_fact(buf);
    std::printf("  [GEN After Live Fact Override]: %s\n", after_fact.c_str());
    check(after_fact.find("жидкий_гелий_сверхпроводник") != std::string::npos,
          "T6.2: live brain_store_fact steers generation immediately without retraining (MUTATION FLIP)");

    // Brand-new entity and relation never seen in corpus
    brain_store_fact(brain, "узел_квазар", "активирует_протокол", "квантовый_мост_7");
    std::memset(buf, 0, sizeof(buf));
    brain_generate(brain, "узел_квазар активирует_протокол", buf, sizeof(buf), 8, 0.0f);
    std::string unseen_fact_gen(buf);
    std::printf("  [GEN Unseen Fact]: %s\n", unseen_fact_gen.c_str());
    check(unseen_fact_gen.find("квантовый_мост_7") != std::string::npos,
          "T6.3: unseen entity + relation stored via brain_store_fact is generated accurately");

    // -------------------------------------------------------------------------
    // TEST 7: Persistence Round-Trip (brain_save_state / brain_load_state)
    // -------------------------------------------------------------------------
    const char* state_file = "build/тест_сохранения_ллм.tsv";
    int save_ok = brain_save_state(brain, state_file);
    check(save_ok == 1, "T7.1: brain_save_state succeeded on UTF-8 Cyrillic path");
    brain_destroy(brain);

    PopCountBrain* brain2 = brain_create(4096, "build/test_llm_gen_restored.bin");
    int load_ok = brain_load_state(brain2, state_file);
    check(load_ok == 1, "T7.2: brain_load_state restored facts and LLM state from UTF-8 Cyrillic path");

    char q_ans[256] = {0};
    int q_ok = brain_query_relation(brain2, "узел_квазар", "активирует_протокол", q_ans, sizeof(q_ans));
    check(q_ok == 1 && std::strcmp(q_ans, "квантовый_мост_7") == 0,
          "T7.3: restored brain_query_relation returns exact stored fact");

    std::memset(buf, 0, sizeof(buf));
    int n_restored = brain_generate(brain2, "def clamp(val_x, min_v, max_v):", buf, sizeof(buf), 64, 0.0f);
    std::string s_restored(buf);
    check(n_restored >= 15 &&
          s_restored.find("if val_x < min_v:") != std::string::npos &&
          s_restored.find("return max_v") != std::string::npos &&
          s_restored.find("return val_x") != std::string::npos,
          "T7.4: restored brain_generate synthesizes zero-shot renamed code identically");

    std::error_code ec;
    std::filesystem::remove(std::filesystem::u8path(state_file), ec);

    // -------------------------------------------------------------------------
    // TEST 8: Compositional Prose Generalization via Distributional Role Clusters
    // -------------------------------------------------------------------------
    PopCountBrain* brain3 = brain_create(1024, "build/test_llm_role_gen.bin");
    const char* role_corpus =
        "Модуль альфа кодирует поток данных в ядре.\n\n"
        "Модуль бета кодирует поток данных в кластере.\n\n"
        "Модуль гамма фильтрует поток данных в буфере.\n\n"
        "Инженер тестирует альфа перед запуском протокола синхронизации.\n";
    BrainTrainStats role_stats{};
    int role_tr_ok = brain_train_corpus(brain3, role_corpus, 4, &role_stats);
    std::memset(buf, 0, sizeof(buf));
    // Strictly unseen combination "Инженер тестирует бета":
    // "бета" was never seen after "Инженер тестирует", but shares the exact syntactic frame
    // (Модуль, кодирует) with "альфа", so role_horizon_tables generalizes to "перед запуском протокола синхронизации".
    int n_role = brain_generate(brain3, "Инженер тестирует бета", buf, sizeof(buf), 16, 0.0f);
    std::string s_role(buf);
    check(role_tr_ok == 1 && n_role >= 3 && s_role.find("перед запуском протокола") != std::string::npos,
          "T8.1: unseen compositional prose phrase ('Инженер тестирует бета') generalized via role_horizon_tables");

    // Morphological frame clustering: "гамма" is followed by a different verb ("фильтрует" vs "кодирует")
    // that shares the morphological suffix ("ует"), clustering into the same role as "альфа".
    std::memset(buf, 0, sizeof(buf));
    int n_morph_role = brain_generate(brain3, "Инженер тестирует гамма", buf, sizeof(buf), 16, 0.0f);
    std::string s_morph_role(buf);
    check(n_morph_role >= 3 && s_morph_role.find("перед запуском протокола") != std::string::npos,
          "T8.2: morphological suffix frame clustering ('кодирует'/'фильтрует') generalizes 'Инженер тестирует гамма'");

    // Joint VSA morphological inflection tolerance in brain_query_relation
    brain_store_fact(brain3, "сервер_нейрон", "использует_шину", "оптический_канал");
    char q_morph[256] = {0};
    int q_morph_ok = brain_query_relation(brain3, "сервер_нейрона", "использует_шины", q_morph, sizeof(q_morph));
    check(q_morph_ok == 1 && std::strcmp(q_morph, "оптический_канал") == 0,
          "T8.3: joint subject+relation morphological inflection resolved via VSA similarity in brain_query_relation");
    brain_destroy(brain3);

    // -------------------------------------------------------------------------
    // TEST 8.4-8.7: Morphological Question Routing, Mid-Sentence Alignment & Local Scope Registers
    // -------------------------------------------------------------------------
    std::memset(buf, 0, sizeof(buf));
    int n_q_short = brain_generate(
        brain2,
        "Почему не нужна видеокарта для обучения и генерации?",
        buf, sizeof(buf), 45, 0.0f);
    std::string s_q_short(buf);
    std::memset(buf, 0, sizeof(buf));
    int n_q_morph = brain_generate(
        brain2,
        "Почему не нужна видеокарта для обучения и генерации?",
        buf, sizeof(buf), 64, 0.0f);
    std::string s_q_morph(buf);
    std::printf("[T8.4 Morphological Question Routing] (%d tokens):\n  %s\n\n", n_q_morph, buf);
    check(n_q_short >= 20 && s_q_short.back() == '.' &&
          n_q_morph >= 40 && s_q_morph.find("видеокарты") != std::string::npos &&
          s_q_morph.back() == '.',
          "T8.4: inflected question ('видеокарта/обучения/генерации') routes to GPU/CPU paragraph & ends at sentence boundary");

    std::memset(buf, 0, sizeof(buf));
    int n_mid_morph = brain_generate(
        brain2,
        "Архитектуры PopCount заменяют матричные умножения",
        buf, sizeof(buf), 30, 0.0f);
    std::string s_mid_morph(buf);
    std::memset(buf, 0, sizeof(buf));
    int n_prep_ctx = brain_generate(
        brain2,
        "Гипермерная память VSA связывает понятия через",
        buf, sizeof(buf), 25, 0.0f);
    std::string s_prep_ctx(buf);
    std::printf("[T8.5 Mid-Sentence & Preposition Continuation]:\n  1) %s\n  2) %s\n\n",
                s_mid_morph.c_str(), s_prep_ctx.c_str());
    check(n_mid_morph >= 5 && s_mid_morph.find("с плавающей точкой") != std::string::npos &&
          !s_mid_morph.empty() && s_mid_morph.back() == '.' &&
          n_prep_ctx >= 4 && s_prep_ctx.find("побитовое исключающее ИЛИ") != std::string::npos &&
          !s_prep_ctx.empty() && s_prep_ctx.back() == '.',
          "T8.5: inflected mid-sentence prefix & skipped-word preposition prefix continue seamlessly and terminate cleanly at '.'");

    std::memset(buf, 0, sizeof(buf));
    int n_local_cpp = brain_generate(
        brain2,
        "int sum_array(const int* data, int len) {\n    int acc = 0;",
        buf, sizeof(buf), 48, 0.0f);
    std::string s_local_cpp(buf);
    std::printf("[T8.6 Local Variable Register C++] (%d tokens):\n  %s\n\n", n_local_cpp, buf);
    check(n_local_cpp >= 10 &&
          s_local_cpp.find("acc += data[i]") != std::string::npos &&
          s_local_cpp.find("return acc") != std::string::npos,
          "T8.6: local variable 'acc' + parameters 'data, len' generalized across 26+ token loop horizon via <reg_0..7>");

    std::memset(buf, 0, sizeof(buf));
    int n_local_py = brain_generate(
        brain2,
        "def factorial_iter(n):\n    prod = 1",
        buf, sizeof(buf), 28, 0.0f);
    std::string s_local_py(buf);
    std::printf("[T8.7 Local Variable Register Python] (%d tokens):\n  %s\n\n", n_local_py, buf);
    check(n_local_py >= 8 &&
          s_local_py.find("prod = prod * i") != std::string::npos &&
          s_local_py.find("return prod") != std::string::npos,
          "T8.7: Python local variable 'prod' generalized inside loop body and return statement");

    // -------------------------------------------------------------------------
    // TEST 9: Rigorous Critique Adversarial Regression Suite (T9.1 - T9.5)
    // -------------------------------------------------------------------------
    // T9.1: Compiled binary model serialization (#__POPCNT_LLM_BIN2__) without
    // storing raw training corpus text or re-running train_corpus on load
    const char* bin2_file = "build/test_compiled_bin2.tsv";
    int save_bin2 = brain_save_state(brain2, bin2_file);
    bool has_bin2_header = false;
    {
        std::ifstream ifs(bin2_file, std::ios::binary);
        std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        has_bin2_header = (content.find("#__POPCNT_LLM_BIN2__ ") != std::string::npos) &&
                          (content.find("#__POPCNT_LLM_LZ1__ ") == std::string::npos);
    }
    PopCountBrain* brain_fast = brain_create(4096, "build/test_llm_fast_load.bin");
    auto t_load_start = std::chrono::steady_clock::now();
    int load_bin2 = brain_load_state(brain_fast, bin2_file);
    auto t_load_end = std::chrono::steady_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
    std::memset(buf, 0, sizeof(buf));
    int n_fast_gen = brain_generate(brain_fast, "def clamp(x_val, lo_v, hi_v):", buf, sizeof(buf), 64, 0.0f);
    std::string s_fast_gen(buf);
    std::printf("[T9.1 Compiled Model Deserialization]: %.3f ms vs train %.3f ms (has_bin2_header=%d)\n",
                load_ms, stats.train_time_ms, has_bin2_header ? 1 : 0);
    check(save_bin2 == 1 && load_bin2 == 1 && has_bin2_header &&
          load_ms < 50.0 &&
          n_fast_gen >= 15 && s_fast_gen.find("return hi_v") != std::string::npos,
          "T9.1: brain_save_state writes compiled binary tables (#__POPCNT_LLM_BIN2__) & loads in <50ms without retraining");
    brain_destroy(brain_fast);
    std::filesystem::remove(bin2_file, ec);

    // T9.2: Same-arity C++ function disambiguation (sum_array vs max_array)
    PopCountBrain* brain_arity = brain_create(2048, "build/test_llm_arity.bin");
    const char* arity_corpus =
        "int sum_array(const int* arr, int n) {\n"
        "    int total = 0;\n"
        "    for (int i = 0; i < n; ++i) {\n"
        "        total += arr[i];\n"
        "    }\n"
        "    return total;\n"
        "}\n\n"
        "int max_array(const int* arr, int n) {\n"
        "    int best = arr[0];\n"
        "    for (int i = 1; i < n; ++i) {\n"
        "        if (arr[i] > best) best = arr[i];\n"
        "    }\n"
        "    return best;\n"
        "}\n";
    BrainTrainStats arity_stats{};
    int arity_tr_ok = brain_train_corpus(brain_arity, arity_corpus, 4, &arity_stats);
    std::memset(buf, 0, sizeof(buf));
    int n_sum_arity = brain_generate(
        brain_arity,
        "int compute_sum(const int* data, int len) {",
        buf, sizeof(buf), 48, 0.0f);
    std::string s_sum_arity(buf);
    std::memset(buf, 0, sizeof(buf));
    int n_max_arity = brain_generate(
        brain_arity,
        "int find_max(const int* data, int len) {",
        buf, sizeof(buf), 48, 0.0f);
    std::string s_max_arity(buf);
    std::printf("[T9.2 Same-Arity Disambiguation]:\n  sum -> %s\n  max -> %s\n\n",
                s_sum_arity.c_str(), s_max_arity.c_str());
    check(arity_tr_ok == 1 &&
          n_sum_arity >= 12 && s_sum_arity.find("+= data[i]") != std::string::npos &&
          s_sum_arity.find("return ") != std::string::npos &&
          n_max_arity >= 12 && s_max_arity.find("if (data[i] >") != std::string::npos &&
          s_max_arity.find("return ") != std::string::npos,
          "T9.2: same-arity functions (sum_array vs max_array) disambiguated with complete return statement");

    // -------------------------------------------------------------------------
    // TEST 10: Self-Audit Adversarial Regression Suite (T10.1 - T10.5)
    // -------------------------------------------------------------------------
    // T10.1: Short prose sentence (<12 tokens) stops at period without dangling tail,
    // and C++ function body with max_tokens=48 completes its `return` before `}`
    check(!s_prep_ctx.empty() && s_prep_ctx.back() == '.' &&
          s_prep_ctx.find("по смысловому маршруту") == std::string::npos &&
          s_max_arity.find("return best;") != std::string::npos,
          "T10.1: short sentence (<12 steps) terminates at period & C++ function completes return statement before closing brace");

    // T10.2: Neutral function name with body prefix disambiguation (active_target_segment
    // tie-break must NOT overpower a longer L=10 reg_horizon_tables match)
    std::memset(buf, 0, sizeof(buf));
    int n_neutral_max = brain_generate(
        brain_arity,
        "int process_items(const int* data, int len) {\n"
        "    int best = data[0];\n"
        "    for (int i = 1; i < len; ++i) {",
        buf, sizeof(buf), 32, 0.0f);
    std::string s_neutral_max(buf);
    std::printf("[T10.2 Neutral Function Name + Body Prefix]:\n  %s\n\n", s_neutral_max.c_str());
    check(n_neutral_max >= 8 &&
          s_neutral_max.find("if (data[i] > best)") != std::string::npos &&
          s_neutral_max.find("+=") == std::string::npos &&
          s_neutral_max.find("return best;") != std::string::npos,
          "T10.2: neutral function name with max_array body prefix preserves L=10 horizon match without sum_array override");

    // T10.3: Multi-token C++ uniform initialization `int best{data[0]};`
    std::memset(buf, 0, sizeof(buf));
    int n_multi_brace = brain_generate(
        brain_arity,
        "int find_max(const int* data, int len) {\n"
        "    int best{data[0]};",
        buf, sizeof(buf), 48, 0.0f);
    std::string s_multi_brace(buf);
    std::printf("[T10.3 Multi-Token Brace Init `int best{data[0]};`]:\n  %s\n\n", s_multi_brace.c_str());
    check(n_multi_brace >= 12 &&
          s_multi_brace.find("if (data[i] > best) best = data[i];") != std::string::npos &&
          s_multi_brace.find("return best;") != std::string::npos,
          "T10.3: multi-token C++ brace initialization `int best{data[0]};` canonicalizes to `<reg_4> = <reg_0> [ 0 ] ;`");
    brain_destroy(brain_arity);

    // T9.3: C++ uniform brace initialization (int acc{0};) and uninitialized declaration (int acc;)
    std::memset(buf, 0, sizeof(buf));
    int n_brace_cpp = brain_generate(
        brain2,
        "int sum_array(const int* data, int len) {\n    int acc{0};",
        buf, sizeof(buf), 48, 0.0f);
    std::string s_brace_cpp(buf);
    std::memset(buf, 0, sizeof(buf));
    int n_uninit_cpp = brain_generate(
        brain2,
        "int sum_array(const int* data, int len) {\n    int acc;",
        buf, sizeof(buf), 48, 0.0f);
    std::string s_uninit_cpp(buf);
    std::printf("[T9.3 Uniform & Uninitialized Local C++ Decls]:\n  brace  -> %s\n  uninit -> %s\n\n",
                s_brace_cpp.c_str(), s_uninit_cpp.c_str());
    check(n_brace_cpp >= 10 && s_brace_cpp.find("acc += data[i]") != std::string::npos &&
          s_brace_cpp.find("return acc") != std::string::npos &&
          n_uninit_cpp >= 10 && s_uninit_cpp.find("acc += data[i]") != std::string::npos &&
          s_uninit_cpp.find("return acc") != std::string::npos,
          "T9.3: C++ uniform initialization 'int acc{0};' and uninitialized 'int acc;' bind to <reg_4> and generalize");

    // T9.4: Dynamic neural vocabulary growth beyond +64 unseen OOV prompt tokens
    std::string oov_prompt;
    for (int i = 0; i < 85; ++i) {
        oov_prompt += "oov_dynamic_tok_" + std::to_string(i) + " ";
    }
    oov_prompt += "Как работает архитектура PopCount?";
    std::memset(buf, 0, sizeof(buf));
    int n_oov = brain_generate(brain2, oov_prompt.c_str(), buf, sizeof(buf), 32, 0.0f);
    std::string s_oov(buf);
    check(n_oov >= 8 && s_oov.find("PopCount") != std::string::npos,
          "T9.4: >80 unseen OOV tokens dynamically expand sublinear_net vocabulary without freezing or crashing");

    // T9.5: TSV injection protection (\t, \n, \r, \\, #__POPCNT_LLM_) in brain_save_state / brain_load_state
    const char* adv_subj = "узел\tс_табом\nи_переводом_строки";
    const char* adv_rel  = "#__POPCNT_LLM_BIN2__\\инъекция";
    const char* adv_obj  = "значение\r\nс_возвратом\tкаретки";
    brain_store_fact(brain2, adv_subj, adv_rel, adv_obj);
    const char* tsv_inj_file = "build/test_tsv_injection.tsv";
    int save_inj = brain_save_state(brain2, tsv_inj_file);
    PopCountBrain* brain_inj = brain_create(4096, "build/test_llm_inj.bin");
    int load_inj = brain_load_state(brain_inj, tsv_inj_file);
    char inj_ans[256] = {0};
    int q_inj_ok = brain_query_relation(brain_inj, adv_subj, adv_rel, inj_ans, sizeof(inj_ans));
    check(save_inj == 1 && load_inj == 1 && q_inj_ok == 1 && std::strcmp(inj_ans, adv_obj) == 0,
          "T9.5: TSV escaping protects brain_save_state/brain_load_state against embedded \\t, \\n, \\r, \\\\, and # headers");

    // T10.4: Fact overwrite & case-variant deduplication + idempotent brain_load_state
    PopCountBrain* brain_dedup = brain_create(1024, "build/test_llm_dedup.bin");
    brain_store_fact(brain_dedup, "Alice", "role", "developer");
    brain_store_fact(brain_dedup, "alice", "ROLE", "architect");
    brain_store_fact(brain_dedup, "ALICE", "Role", "principal_engineer");
    size_t dedup_pages = 0, dedup_rules = 0;
    brain_get_stats(brain_dedup, &dedup_pages, &dedup_rules);
    char dedup_ans[128] = {0};
    int q_dedup_ok = brain_query_relation(brain_dedup, "alice", "role", dedup_ans, sizeof(dedup_ans));
    const char* dedup_file = "build/test_dedup_state.tsv";
    brain_save_state(brain_dedup, dedup_file);
    // Load twice into the same instance to verify idempotence
    int ld1 = brain_load_state(brain_dedup, dedup_file);
    int ld2 = brain_load_state(brain_dedup, dedup_file);
    size_t dedup_pages_after = 0, dedup_rules_after = 0;
    brain_get_stats(brain_dedup, &dedup_pages_after, &dedup_rules_after);
    check(dedup_rules == 1 && q_dedup_ok == 1 &&
          std::strcmp(dedup_ans, "principal_engineer") == 0 &&
          ld1 == 1 && ld2 == 1 && dedup_rules_after == 1,
          "T10.4: case-variant fact overwrites update in-place without inflating active_rules, and brain_load_state is idempotent");
    brain_destroy(brain_dedup);
    std::filesystem::remove(dedup_file, ec);

    // T10.5: Corrupted #__POPCNT_LLM_BIN2__ header rejection in brain_load_state
    const char* corrupt_file = "build/test_corrupt_bin2.tsv";
    {
        std::ofstream ofs(corrupt_file, std::ios::binary);
        ofs << "#__POPCNT_LLM_BIN2__ 999999 ZZZ_INVALID_HEX\n";
    }
    int load_corrupt1 = brain_load_state(brain_inj, corrupt_file);
    {
        std::ofstream ofs(corrupt_file, std::ios::binary);
        // Valid hex, but bogus magic / truncated binary payload
        ofs << "#__POPCNT_LLM_BIN2__ 100 00010203040506070809\n";
    }
    int load_corrupt2 = brain_load_state(brain_inj, corrupt_file);
    check(load_corrupt1 == 0 && load_corrupt2 == 0,
          "T10.5: corrupted #__POPCNT_LLM_BIN2__ headers (invalid hex or truncated binary payload) are deterministically rejected");
    brain_destroy(brain_inj);
    std::filesystem::remove(tsv_inj_file, ec);
    std::filesystem::remove(corrupt_file, ec);

    // -------------------------------------------------------------------------
    // TEST 11: Adapted Continuous Representation Spaces & Deep Interpolation (T11.1 - T11.6)
    // -------------------------------------------------------------------------
    // T11.1: Unit hypersphere normalization S^511 & determinism of brain_embed_continuous
    float vec_arch1[512] = {0.0f};
    float vec_arch2[512] = {0.0f};
    int emb_ok1 = brain_embed_continuous(brain2, "Архитектура PopCount использует битовые операции XNOR и POPCNT", vec_arch1);
    int emb_ok2 = brain_embed_continuous(brain2, "Архитектура PopCount использует битовые операции XNOR и POPCNT", vec_arch2);
    double norm_arch = 0.0;
    double max_diff_arch = 0.0;
    for (int d = 0; d < 512; ++d) {
        norm_arch += static_cast<double>(vec_arch1[d]) * static_cast<double>(vec_arch1[d]);
        max_diff_arch = std::max(max_diff_arch, std::abs(static_cast<double>(vec_arch1[d] - vec_arch2[d])));
    }
    norm_arch = std::sqrt(norm_arch);
    check(emb_ok1 == 1 && emb_ok2 == 1 && std::abs(norm_arch - 1.0) < 1e-4 && max_diff_arch == 0.0,
          "T11.1: brain_embed_continuous projects text onto unit hypersphere S^511 (||v||_2 = 1.0) deterministically");

    // T11.2: Continuous representation manifold preserves morphological & semantic clustering
    float sim_identical = brain_continuous_similarity(
        brain2,
        "быстрые битовые операции процессора",
        "быстрые битовые операции процессора");
    float sim_morph = brain_continuous_similarity(
        brain2,
        "быстрые битовые операции процессора",
        "быстрыми битовыми операциями процессоров");
    float sim_unrelated = brain_continuous_similarity(
        brain2,
        "быстрые битовые операции процессора",
        "def accumulate_values(items):");
    std::printf("[T11.2 Continuous Manifold Similarity]:\n  identical=%.4f  morph_variant=%.4f  unrelated_code=%.4f\n\n",
                sim_identical, sim_morph, sim_unrelated);
    check(sim_identical > 0.999f && sim_morph > 0.60f && sim_morph > sim_unrelated + 0.25f,
          "T11.2: continuous representation space ranks identical > morphological inflection > unrelated domain");

    // T11.3: Geodesic SLERP + Hopfield attractor interpolation endpoint fidelity (alpha=0.0 and alpha=1.0)
    float vec_end_a[512] = {0.0f};
    float vec_end_b[512] = {0.0f};
    float vec_interp_0[512] = {0.0f};
    float vec_interp_1[512] = {0.0f};
    char near_tok_0[128] = {0};
    char near_tok_1[128] = {0};
    const char* concept_a = "Архитектура PopCount использует битовые операции XNOR и POPCNT";
    const char* concept_b = "int sum_array(const int* arr, int n) { return total; }";
    brain_embed_continuous(brain2, concept_a, vec_end_a);
    brain_embed_continuous(brain2, concept_b, vec_end_b);
    int i0_ok = brain_interpolate_continuous(brain2, concept_a, concept_b, 0.0f, vec_interp_0, near_tok_0, sizeof(near_tok_0));
    int i1_ok = brain_interpolate_continuous(brain2, concept_a, concept_b, 1.0f, vec_interp_1, near_tok_1, sizeof(near_tok_1));
    double dot_0a = 0.0, dot_1b = 0.0;
    for (int d = 0; d < 512; ++d) {
        dot_0a += static_cast<double>(vec_interp_0[d]) * static_cast<double>(vec_end_a[d]);
        dot_1b += static_cast<double>(vec_interp_1[d]) * static_cast<double>(vec_end_b[d]);
    }
    check(i0_ok > 0 && i1_ok > 0 && dot_0a > 0.999 && dot_1b > 0.999 && near_tok_0[0] != '\0' && near_tok_1[0] != '\0',
          "T11.3: brain_interpolate_continuous preserves exact endpoints at alpha=0.0 and alpha=1.0 and resolves nearest tokens");

    // T11.4: Monotonic geodesic trajectory & midpoint alignment across alpha in {0.0, 0.25, 0.50, 0.75, 1.0}
    float alphas[5] = {0.0f, 0.25f, 0.50f, 0.75f, 1.0f};
    double cos_to_a[5] = {0.0};
    double cos_to_b[5] = {0.0};
    char mid_tok[128] = {0};
    bool norms_unit = true;
    for (int k = 0; k < 5; ++k) {
        float vk[512] = {0.0f};
        char tk[128] = {0};
        brain_interpolate_continuous(brain2, concept_a, concept_b, alphas[k], vk, tk, sizeof(tk));
        if (k == 2) std::strncpy(mid_tok, tk, sizeof(mid_tok) - 1);
        double nsq = 0.0, da = 0.0, db = 0.0;
        for (int d = 0; d < 512; ++d) {
            nsq += static_cast<double>(vk[d]) * static_cast<double>(vk[d]);
            da  += static_cast<double>(vk[d]) * static_cast<double>(vec_end_a[d]);
            db  += static_cast<double>(vk[d]) * static_cast<double>(vec_end_b[d]);
        }
        if (std::abs( std::sqrt(nsq) - 1.0 ) > 1e-3) norms_unit = false;
        cos_to_a[k] = da;
        cos_to_b[k] = db;
    }
    std::printf("[T11.4 Geodesic Trajectory Cosines to A / B]:\n"
                "  alpha=0.00: (%.3f, %.3f)  alpha=0.25: (%.3f, %.3f)  alpha=0.50: (%.3f, %.3f) [tok='%s']  alpha=0.75: (%.3f, %.3f)  alpha=1.00: (%.3f, %.3f)\n\n",
                cos_to_a[0], cos_to_b[0], cos_to_a[1], cos_to_b[1], cos_to_a[2], cos_to_b[2], mid_tok, cos_to_a[3], cos_to_b[3], cos_to_a[4], cos_to_b[4]);
    check(norms_unit &&
          cos_to_a[0] > cos_to_a[1] && cos_to_a[1] > cos_to_a[2] && cos_to_a[2] > cos_to_a[3] && cos_to_a[3] > cos_to_a[4] &&
          cos_to_b[0] < cos_to_b[1] && cos_to_b[1] < cos_to_b[2] && cos_to_b[2] < cos_to_b[3] && cos_to_b[3] < cos_to_b[4] &&
          cos_to_a[2] > 0.45 && cos_to_b[2] > 0.45,
          "T11.4: deep geodesic interpolation maintains unit norm and strictly monotonic cosine progression from A to B");

    // T11.5: Continuous space invariance across brain_save_state / brain_load_state
    const char* t11_5_file = "build/test_t11_5_state.tsv";
    brain_save_state(brain2, t11_5_file);
    PopCountBrain* brain_t11_5 = brain_create(4096, "build/test_llm_t11_5.bin");
    brain_load_state(brain_t11_5, t11_5_file);
    float vec_b1[512] = {0.0f};
    float vec_b2[512] = {0.0f};
    brain_embed_continuous(brain_t11_5, "Семантический резонанс и голографическая память", vec_b1);
    brain_embed_continuous(brain2,       "Семантический резонанс и голографическая память", vec_b2);
    double dot_b1_b2 = 0.0;
    for (int d = 0; d < 512; ++d) {
        dot_b1_b2 += static_cast<double>(vec_b1[d]) * static_cast<double>(vec_b2[d]);
    }
    std::printf("[T11.5 Save/Load Manifold Cosine]: %.6f\n\n", dot_b1_b2);
    check(dot_b1_b2 > 0.9999,
          "T11.5: continuous representation manifold is deterministically reconstructed on brain_load_state (cosine > 0.9999)");
    brain_destroy(brain_t11_5);
    std::filesystem::remove(t11_5_file, ec);

    // T11.6: High-speed AVX-512 continuous embedding & interpolation throughput benchmark
    auto t_cont0 = std::chrono::steady_clock::now();
    float bench_vec[512];
    char bench_tok[64];
    const int kContIters = 200;
    for (int it = 0; it < kContIters; ++it) {
        brain_interpolate_continuous(brain2, concept_a, concept_b, 0.5f, bench_vec, bench_tok, sizeof(bench_tok));
    }
    double cont_us_per_call = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - t_cont0
    ).count() / static_cast<double>(kContIters);
    std::printf("[T11.6 Continuous Interpolation Latency]: %.2f us/call (%d iterations)\n\n",
                cont_us_per_call, kContIters);
    check(cont_us_per_call < 2500.0 && bench_tok[0] != '\0',
          "T11.6: AVX-512 continuous manifold interpolation executes in < 2.5 ms/call");

    // -------------------------------------------------------------------------
    // TEST 12: Deep Context Horizon Scaling Gate (L > 32 up to 64 tokens)
    // -------------------------------------------------------------------------
    // T12.1: Deep 45+ token C++ prefix with full multi-line loop context completes with 'acc;' and '}'
    std::memset(buf, 0, sizeof(buf));
    const char* deep_cpp_prompt =
        "int sum_array(const int* data, int len) {\n"
        "    int acc = 0;\n"
        "    for (int i = 0; i < len; ++i) {\n"
        "        acc += data[i];\n"
        "    }\n"
        "    return";
    int n_deep_cpp = brain_generate(brain2, deep_cpp_prompt, buf, sizeof(buf), 16, 0.0f);
    std::string s_deep_cpp(buf);
    std::printf("[T12.1 Deep Context Horizon C++ (48-token prompt)]:\n  %s\n\n", s_deep_cpp.c_str());
    check(n_deep_cpp >= 2 && s_deep_cpp.find("return acc;") != std::string::npos &&
          has_balanced_delimiters(s_deep_cpp),
          "T12.1: deep context horizon (48-token C++ prefix) matches beyond L=32 and completes return acc; with balanced braces");

    // T12.2: Deep 45+ token Python binary_search prefix completes with ' -1'
    std::memset(buf, 0, sizeof(buf));
    const char* deep_py_prompt =
        "def binary_search(arr, target):\n"
        "    left = 0\n"
        "    right = len(arr) - 1\n"
        "    while left <= right:\n"
        "        mid = (left + right) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        if arr[mid] < target:\n"
        "            left = mid + 1\n"
        "        else:\n"
        "            right = mid - 1\n"
        "    return";
    int n_deep_py = brain_generate(brain2, deep_py_prompt, buf, sizeof(buf), 16, 0.0f);
    std::string s_deep_py(buf);
    std::printf("[T12.2 Deep Context Horizon Python (52-token prompt)]:\n  %s\n\n", s_deep_py.c_str());
    check(n_deep_py >= 1 && s_deep_py.find("return -1") != std::string::npos,
          "T12.2: deep context horizon (52-token Python prefix) completes return -1 from beyond L=32 context window");

    brain_destroy(brain2);

    // -------------------------------------------------------------------------
    // TEST 12.3 & 12.4: >1000-Token Hard Context Horizon (L up to 2048) Without Degradation
    // -------------------------------------------------------------------------
    PopCountBrain* brain_long = brain_create(4096, "build/test_llm_long_horizon_1000.bin");
    std::string shared_1100_chain;
    for (int step = 1; step <= 135; ++step) {
        shared_1100_chain += " такт_" + std::to_string(step) +
                             " синхронизирует векторный регистр ядра через общий конвейер памяти";
    }
    std::string doc_alpha =
        "Система гиперконтекста запускает контур_альфа_сверхдальний и" +
        shared_1100_chain +
        " завершает вычисление результатом терминал_альфа_успех_1080.";
    std::string doc_beta =
        "Система гиперконтекста запускает контур_бета_сверхдальний и" +
        shared_1100_chain +
        " завершает вычисление результатом терминал_бета_успех_1080.";
    std::string long_corpus = doc_alpha + "\n\n" + doc_beta + "\n";

    BrainTrainStats long_stats{};
    int long_tr_ok = brain_train_corpus(brain_long, long_corpus.c_str(), 3, &long_stats);
    std::printf("[T12.3 >1000-Token Corpus Stats]: total_tokens=%zu, transitions=%zu, replay_acc=%.2f%%, cov=%.2f%%, time=%.2f ms (%.0f tok/s)\n",
                long_stats.total_tokens, long_stats.total_transitions,
                long_stats.replay_accuracy_percent, long_stats.shortlist_coverage_percent,
                long_stats.train_time_ms, long_stats.tokens_per_sec);
    check(long_tr_ok == 1 &&
          long_stats.total_tokens >= 2100 &&
          long_stats.replay_accuracy_percent >= 99.9 &&
          long_stats.shortlist_coverage_percent >= 99.9,
          "T12.3: >1000-token sequences (2160+ corpus tokens) achieve 100% replay accuracy & 100% coverage with O(1) rolling horizon hashes");

    std::string prompt_alpha_1080 =
        "Система гиперконтекста запускает контур_альфа_сверхдальний и" +
        shared_1100_chain +
        " завершает вычисление результатом";
    std::string prompt_beta_1080 =
        "Система гиперконтекста запускает контур_бета_сверхдальний и" +
        shared_1100_chain +
        " завершает вычисление результатом";

    std::vector<char> long_buf(32768, 0);
    int n_long_a = brain_generate(brain_long, prompt_alpha_1080.c_str(), long_buf.data(), long_buf.size(), 16, 0.0f);
    std::string gen_long_a(long_buf.data());

    std::fill(long_buf.begin(), long_buf.end(), 0);
    int n_long_b = brain_generate(brain_long, prompt_beta_1080.c_str(), long_buf.data(), long_buf.size(), 16, 0.0f);
    std::string gen_long_b(long_buf.data());

    std::printf("[T12.4 >1000-Token Horizon Disambiguation]:\n  Alpha (L>1080) -> %s\n  Beta  (L>1080) -> %s\n\n",
                gen_long_a.c_str(), gen_long_b.c_str());
    check(n_long_a >= 1 && n_long_b >= 1 &&
          gen_long_a.find("терминал_альфа_успех_1080") != std::string::npos &&
          gen_long_a.find("терминал_бета_успех_1080") == std::string::npos &&
          gen_long_b.find("терминал_бета_успех_1080") != std::string::npos &&
          gen_long_b.find("терминал_альфа_успех_1080") == std::string::npos,
          "T12.4: >1000-token hard context horizon (L > 1080 tokens) disambiguates identical 1080-token middle chains with zero degradation");

    brain_destroy(brain_long);

    // -------------------------------------------------------------------------
    // TEST 13: 32-Layer Non-Linear Hierarchy, Contrastive Riemannian SGD Manifold,
    //          Partitioned Holographic Superposition & Multi-Hop Logical Deduction
    // -------------------------------------------------------------------------
    PopCountBrain* brain_deep = brain_create(4096, "build/test_llm_deep_32layer.bin");
    const char* deep_reasoning_corpus =
        "Квантовый фотон переносит спиновую энергию сквозь вакуумный волновод.\n"
        "Квантовый гравитон переносит спиновую энергию сквозь вакуумный волновод.\n"
        "Фотон ускоряет плазменный солитон в магнитной ловушке.\n"
        "Гравитон ускоряет плазменный солитон в магнитной ловушке.\n"
        "Кулинарный рецепт требует свежих овощей и оливкового масла.\n"
        "Модуль_Орион подключается к шине Квантовый_Кластер.\n"
        "Шина Квантовый_Кластер маршрутизирует пакеты через Тор_5D.\n"
        "Топология Тор_5D гарантирует задержку Субнаносекундный_Отклик.\n"
        "Топология Плоский_Меш гарантирует задержку Миллисекундный_Отклик.\n";

    BrainTrainStats deep_stats{};
    int deep_tr_ok = brain_train_corpus(brain_deep, deep_reasoning_corpus, 6, &deep_stats);
    check(deep_tr_ok == 1, "T13.0: 32-layer non-linear & Contrastive SGD corpus training succeeds");

    // T13.1: Contrastive Riemannian SGD aligns lexically disjoint functional equivalents ("фотон" vs "гравитон")
    // that have completely different character stems but identical syntactic/functional frames in the corpus.
    float sim_func_equiv = brain_continuous_similarity(brain_deep, "фотон", "гравитон");
    float sim_func_unrel = brain_continuous_similarity(brain_deep, "фотон", "масла");
    std::printf("[T13.1 32-Layer + Contrastive SGD Disjoint Lexical Alignment]:\n"
                "  sim('фотон', 'гравитон') = %.4f vs sim('фотон', 'масла') = %.4f\n\n",
                sim_func_equiv, sim_func_unrel);
    check(sim_func_equiv > 0.45f && sim_func_equiv > sim_func_unrel + 0.30f,
          "T13.1: Contrastive Riemannian SGD + 32-layer stack aligns lexically disjoint functional equivalents ('фотон' vs 'гравитон' > 0.45)");

    // T13.2: Multi-hop logical deduction across 3 relational hops in brain_query_relation & brain_generate
    brain_store_fact(brain_deep, "Модуль_Орион", "является_подклассом", "Квантовый_Кластер");
    brain_store_fact(brain_deep, "Квантовый_Кластер", "входит_в_состав", "Тор_5D");
    brain_store_fact(brain_deep, "Тор_5D", "обеспечивает_режим", "Сверхпроводящий_Резонанс_5D");

    char deduced_obj[256] = {0};
    int ded_ok = brain_query_relation(brain_deep, "Модуль_Орион", "обеспечивает_режим", deduced_obj, sizeof(deduced_obj));
    std::printf("[T13.2 3-Hop Relational Deduction]: Модуль_Орион -> Квантовый_Кластер -> Тор_5D -> %s\n\n", deduced_obj);
    check(ded_ok == 1 && std::string(deduced_obj) == "Сверхпроводящий_Резонанс_5D",
          "T13.2: 3-hop transitive & taxonomic deduction resolves 'Модуль_Орион' -> 'Сверхпроводящий_Резонанс_5D'");

    // T13.3: Generative multi-hop deduction (Deductive Live-Fact Closure + 3-Hop Discourse Graph Propagation)
    std::memset(buf, 0, sizeof(buf));
    int n_gen_ded = brain_generate(brain_deep, "Модуль_Орион обеспечивает_режим", buf, sizeof(buf), 6, 0.0f);
    std::string s_gen_ded(buf);
    std::printf("[T13.3 Generative Multi-Hop Fact Deduction]:\n  %s\n\n", s_gen_ded.c_str());
    check(n_gen_ded >= 1 && s_gen_ded.find("Сверхпроводящий_Резонанс_5D") != std::string::npos,
          "T13.3: brain_generate synthesizes 2/3-hop deduced object 'Сверхпроводящий_Резонанс_5D' for 'Модуль_Орион обеспечивает_режим'");

    // T13.4: 3-Hop Discourse Graph Propagation across prose segments disambiguates unseen cross-segment prompt
    std::memset(buf, 0, sizeof(buf));
    int n_gen_disc = brain_generate(brain_deep, "Модуль_Орион гарантирует задержку", buf, sizeof(buf), 4, 0.0f);
    std::string s_gen_disc(buf);
    std::printf("[T13.4 3-Hop Prose Discourse Propagation]:\n  %s\n\n", s_gen_disc.c_str());
    check(n_gen_disc >= 1 && s_gen_disc.find("Субнаносекундный_Отклик") != std::string::npos,
          "T13.4: 3-hop discourse propagation (Модуль_Орион -> Квантовый_Кластер -> Тор_5D -> Субнаносекундный_Отклик) disambiguates target property");

    brain_destroy(brain_deep);

    // -------------------------------------------------------------------------
    // TEST 14: Distributional OOV Output Priors (Breaking the 20.6–28.1% Held-Out
    //          Target Barrier) & Large-Scale (10+ MB Corpus / 1.0 GB Bank) Profiling
    // -------------------------------------------------------------------------
    PopCountBrain* brain_oov_scale = brain_create(4096, "build/test_llm_oov_and_scale.bin");
    const char* oov_train_corpus =
        "Модуль квантового ядра инициализирует когерентный процессор и возвращает когерентный процессор.\n"
        "Модуль квантового ядра инициализирует тензорный сопроцессор и возвращает тензорный сопроцессор.\n"
        "Модуль квантового ядра инициализирует векторный ускоритель и возвращает векторный ускоритель.\n"
        "Аппаратный диспетчер активирует фотонный маршрутизатор и проверяет фотонный маршрутизатор.\n"
        "Аппаратный диспетчер активирует спиновый коммутатор и проверяет спиновый коммутатор.\n"
        "Система телеметрии калибрует криогенный резонатор для стабильной синхронизации.\n"
        "Система телеметрии калибрует плазменный осциллятор для стабильной синхронизации.\n";

    BrainTrainStats oov_tr_stats{};
    int oov_tr_ok = brain_train_corpus(brain_oov_scale, oov_train_corpus, 4, &oov_tr_stats);
    check(oov_tr_ok == 1, "T14.0: base training for held-out OOV output prior evaluation succeeds");

    // Held-out evaluation corpus where >25% of target tokens are completely unseen OOV words
    // ("голографический", "нейроморфный", "сверхпроводящий", "магнитный", "тороидальный", "интерферометр")
    // or unseen context-target combinations ("фотонный осциллятор", "криогенный маршрутизатор").
    const char* heldout_eval_text =
        "Модуль квантового ядра инициализирует голографический процессор и возвращает голографический процессор.\n"
        "Модуль квантового ядра инициализирует нейроморфный сопроцессор и возвращает нейроморфный сопроцессор.\n"
        "Аппаратный диспетчер активирует сверхпроводящий маршрутизатор и проверяет сверхпроводящий маршрутизатор.\n"
        "Аппаратный диспетчер активирует магнитный коммутатор и проверяет магнитный коммутатор.\n"
        "Система телеметрии калибрует тороидальный резонатор для стабильной синхронизации.\n"
        "Система телеметрии калибрует фотонный осциллятор для стабильной синхронизации.\n";

    BrainScaleAndHeldOutProfile heldout_prof{};
    bool prof_ok = brain_profile_heldout_and_scale(brain_oov_scale, heldout_eval_text, &heldout_prof);
    std::printf("[T14.1 Held-Out OOV Target Reachability Profile]:\n"
                "  heldout_transitions=%zu, heldout_oov_targets=%zu\n"
                "  unreachable_without_prior=%zu (%.2f%%) -> unreachable_with_prior=%zu (%.2f%%)\n"
                "  heldout_reachability_after=%.2f%%, heldout_accuracy=%.2f%%\n\n",
                heldout_prof.heldout_total_transitions,
                heldout_prof.heldout_oov_targets,
                heldout_prof.heldout_unreachable_without_prior,
                heldout_prof.heldout_unreachable_before_pct,
                heldout_prof.heldout_unreachable_with_prior,
                heldout_prof.heldout_unreachable_after_pct,
                heldout_prof.heldout_reachability_after_pct,
                heldout_prof.heldout_accuracy_pct);
    check(prof_ok &&
          heldout_prof.heldout_oov_targets >= 8 &&
          heldout_prof.heldout_unreachable_before_pct >= 20.0 &&
          heldout_prof.heldout_unreachable_after_pct <= 2.0 &&
          heldout_prof.heldout_reachability_after_pct >= 98.0,
          "T14.1: Distributional & Morphological OOV Output Priors break the 20.6-28.1% held-out unreachable target barrier (>= 98% reachability)");

    // T14.2: Zero-shot unseen OOV target generation in brain_generate
    // "голографический" was never in the training corpus, but appears in the prompt in the adjective role
    // before "процессор", so after "и возвращает" the model emits the unseen OOV token "голографический"!
    std::memset(buf, 0, sizeof(buf));
    int n_oov_gen = brain_generate(
        brain_oov_scale,
        "Модуль квантового ядра инициализирует голографический процессор и возвращает",
        buf, sizeof(buf), 6, 0.0f
    );
    std::string s_oov_gen(buf);
    std::printf("[T14.2 Zero-Shot Unseen OOV Target Generation]:\n  %s\n\n", s_oov_gen.c_str());
    check(n_oov_gen >= 2 &&
          s_oov_gen.find("голографический процессор") != std::string::npos,
          "T14.2: brain_generate emits unseen OOV target 'голографический' via Distributional OOV Output Prior");

    brain_destroy(brain_oov_scale);

    // T14.3: Real-Scale 10+ MB (10.5+ MB) Multi-Language Code + Prose Streaming Benchmark &
    //        1.0 GB (262,144 pages) Sparse Bank Fragmentation / Open-Addressing Audit
    PopCountBrain* brain_10mb = brain_create(4096, "build/test_llm_10mb_scale.bin");
    const char* scale_10mb_path = "build/test_corpus_10mb_scale.txt";
    {
        std::vector<std::string> unique_blocks;
        unique_blocks.reserve(256);
        for (int m = 0; m < 128; ++m) {
            std::string cpp_blk =
                "int compute_kernel_" + std::to_string(m) + "(const int* src_buf, int count) {\n"
                "    int acc_" + std::to_string(m) + " = " + std::to_string(m + 1) + ";\n"
                "    for (int i = 0; i < count; ++i) {\n"
                "        acc_" + std::to_string(m) + " += src_buf[i];\n"
                "    }\n"
                "    return acc_" + std::to_string(m) + ";\n"
                "}\n\n";
            std::string prose_blk =
                "Архитектурный контур_" + std::to_string(m) +
                " маршрутизирует битовый поток через разреженную страницу_" + std::to_string(m) +
                " и стабилизирует векторный отклик_" + std::to_string(m) + ".\n\n";
            unique_blocks.push_back(std::move(cpp_blk));
            unique_blocks.push_back(std::move(prose_blk));
        }

        std::string chunk_64k;
        chunk_64k.reserve(65536);
        for (const auto& blk : unique_blocks) {
            chunk_64k += blk;
        }
        std::ofstream ofs(scale_10mb_path, std::ios::binary);
        size_t written_bytes = 0;
        const size_t target_10mb_bytes = 10600000ULL; // > 10.1 MB (10,600,000 bytes)
        while (written_bytes < target_10mb_bytes) {
            ofs.write(chunk_64k.data(), static_cast<std::streamsize>(chunk_64k.size()));
            written_bytes += chunk_64k.size();
        }
    }
    const size_t file_bytes_10mb = std::filesystem::file_size(scale_10mb_path, ec);
    BrainTrainStats stats_10mb{};
    int tr_10mb_ok = brain_train_file(brain_10mb, scale_10mb_path, 1, &stats_10mb);
    BrainScaleAndHeldOutProfile scale_prof{};
    bool scale_prof_ok = brain_profile_heldout_and_scale(brain_10mb, nullptr, &scale_prof);

    std::printf("[T14.3 10+ MB Corpus & 1.0 GB Sparse Bank Fragmentation Profile]:\n"
                "  corpus_file_bytes=%zu (%.2f MB), total_tokens=%zu, num_sequences=%zu\n"
                "  train_time=%.2f ms, throughput=%.0f tok/s, replay_acc=%.2f%%\n"
                "  bank_capacity=%zu pages (%.2f GB virtual), active_pages=%zu (%zu KB resident)\n"
                "  stored_vsa_slots=%zu, mean_slots/page=%.2f, max_slots/page=%zu, pages_over_16_slots=%zu\n"
                "  mean_open_address_probes=%.4f, max_open_address_probes=%zu, fragmentation_index=%.4f\n\n",
                file_bytes_10mb,
                static_cast<double>(file_bytes_10mb) / (1024.0 * 1024.0),
                stats_10mb.total_tokens,
                stats_10mb.num_sequences,
                stats_10mb.train_time_ms,
                stats_10mb.tokens_per_sec,
                stats_10mb.replay_accuracy_percent,
                scale_prof.bank_total_capacity_pages,
                static_cast<double>(scale_prof.bank_virtual_bytes) / (1024.0 * 1024.0 * 1024.0),
                scale_prof.bank_active_pages,
                scale_prof.bank_resident_bytes / 1024,
                scale_prof.bank_total_stored_slots,
                scale_prof.mean_slots_per_active_page,
                scale_prof.max_slots_per_page,
                scale_prof.pages_over_16_slots,
                scale_prof.mean_open_address_probes,
                scale_prof.max_open_address_probes,
                scale_prof.fragmentation_index);

    check(tr_10mb_ok == 1 && scale_prof_ok &&
          file_bytes_10mb >= 10485760ULL &&
          stats_10mb.total_tokens >= 1000000ULL &&
          stats_10mb.replay_accuracy_percent >= 99.0 &&
          scale_prof.bank_total_capacity_pages == 262144ULL &&
          scale_prof.bank_active_pages >= 500ULL &&
          scale_prof.pages_over_16_slots == 0ULL &&
          scale_prof.max_slots_per_page <= 16ULL,
          "T14.3: 10+ MB corpus (>1M tokens) streams & trains with >=99% accuracy and 0 page overflows across the 1.0 GB (262,144-page) bank");

    // T14.4: Hardware AVX-512 Cache-Hierarchy Latency Sweep (L1D 32KB vs L2 512KB vs L3 16MB vs DRAM 64MB)
    std::printf("[T14.4 AVX-512 4KB Expert Page Cache-Hierarchy Latency Profile]:\n"
                "  L1D  (32 KB  / 8 pages):     %.2f ns/page\n"
                "  L2   (512 KB / 128 pages):   %.2f ns/page\n"
                "  L3   (16 MB  / 4096 pages):  %.2f ns/page\n"
                "  DRAM (64 MB  / 16384 pages): %.2f ns/page\n\n",
                scale_prof.avx512_l1_page_eval_ns,
                scale_prof.avx512_l2_page_eval_ns,
                scale_prof.avx512_l3_page_eval_ns,
                scale_prof.avx512_dram_page_eval_ns);
    check(scale_prof.avx512_l1_page_eval_ns > 0.0 &&
          scale_prof.avx512_l2_page_eval_ns > 0.0 &&
          scale_prof.avx512_l3_page_eval_ns > 0.0 &&
          scale_prof.avx512_dram_page_eval_ns > 0.0 &&
          scale_prof.avx512_dram_page_eval_ns < 50000.0,
          "T14.4: AVX-512 4KB expert page cache-hierarchy sweep profiles L1D, L2, L3, and DRAM (64 MB across 1 GB bank) latencies");

    brain_destroy(brain_10mb);
    std::filesystem::remove(scale_10mb_path, ec);
    std::filesystem::remove("build/test_llm_10mb_scale.bin", ec);
    std::filesystem::remove("build/test_llm_10mb_scale.bin.llm_bank", ec);
    std::filesystem::remove("build/test_llm_oov_and_scale.bin", ec);
    std::filesystem::remove("build/test_llm_oov_and_scale.bin.llm_bank", ec);

    std::printf("================================================================\n");
    std::printf("  SUMMARY: %d Passed, %d Failed\n", g_passed, g_failed);
    std::printf("================================================================\n");
    return (g_failed == 0) ? 0 : 1;
}

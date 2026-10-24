#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <filesystem>
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

    brain_destroy(brain2);
    std::error_code ec;
    std::filesystem::remove(std::filesystem::u8path(state_file), ec);

    std::printf("================================================================\n");
    std::printf("  SUMMARY: %d Passed, %d Failed\n", g_passed, g_failed);
    std::printf("================================================================\n");
    return (g_failed == 0) ? 0 : 1;
}

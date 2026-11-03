// Proof battery for the remaining claimed capabilities. The shipped suites
// assert only that some output is produced, never that a *trained* brain is
// responsible for it. Each probe runs the exact prompt from the shipped test
// on (i) an untrained brain, (ii) a brain trained on the same corpus as the
// shipped test, (iii) - for long-context attention - a brain trained on
// unrelated prose, so corpus causality is separated from mere presence of
// model state.
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <filesystem>

#include "popcnt_brain.h"

static bool has_adjacent_duplicate_words(const std::string& text) {
    std::string cur, prev;
    bool dup = false;
    for (size_t i = 0; i <= text.size() && !dup; ++i) {
        const char ch = (i < text.size()) ? text[i] : ' ';
        if (ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t' || ch == ',' || ch == '.') {
            if (!cur.empty()) {
                if (cur == prev) dup = true;
                prev = cur;
                cur.clear();
            }
        } else {
            cur += ch;
        }
    }
    return dup;
}

static PopCountBrain* fresh(const char* path, const std::string& corpus, int epochs, int* trained) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(std::string(path) + ".llm_bank", ec);
    PopCountBrain* b = brain_create(4096, path);
    if (!b) { *trained = 0; return nullptr; }
    BrainTrainStats st;
    std::memset(&st, 0, sizeof(st));
    *trained = corpus.empty() ? 0 : brain_train_corpus(b, corpus.c_str(), epochs, &st);
    return b;
}

int main() {
    // Corpus from test_llm_text_code_gen.cpp T16 block (used for F8.1 and T16.5).
    const char* f_corpus =
        "Нейроморфный сопроцессор вычисляет градиентный тензор и ускоряет глубокий трансформер.\n"
        "Квантовый ускоритель вычисляет спиновый тензор и ускоряет плотный трансформер.\n"
        "Голографический маршрутизатор синтезирует непрерывный вектор и стабилизирует фазовый резонанс.\n"
        "Оптический коммутатор синтезирует когерентный вектор и стабилизирует частотный резонанс.\n"
        "Аналитический алгоритм обучает скрытое многообразие через обратное распространение ошибки.\n"
        "Вариационный оптимизатор обучает гладкое многообразие через градиентное распространение ошибки.\n";
    const char* search_corpus =
        "int linear_search(const int* arr, int n, int target) {\n"
        "    for (int i = 0; i < n; ++i) { if (arr[i] == target) return i; } return -1;\n"
        "}\n"
        "int binary_search(const int* arr, int n, int target) {\n"
        "    int lo = 0, hi = n - 1; while (lo <= hi) { int mid = lo + (hi - lo) / 2;\n"
        "        if (arr[mid] == target) return mid; if (arr[mid] < target) lo = mid + 1; else hi = mid - 1;\n"
        "    } return -1;\n"
        "}\n";
    const char* prose_corpus =
        "the cat sits on the window sill. a dog sleeps near the door. rain falls on the roof.\n";

    int trained = 0;

    // ---- Probe 1: F8.1 / T16.2 "Open-Ended OOD Extrapolation" gate ---------
    PopCountBrain* untrained = fresh("build/claim_untrained.bin", "", 0, &trained);
    PopCountBrain* trained_f = fresh("build/claim_trained_f.bin", f_corpus, 5, &trained);
    std::printf("probe1_ood: trained(f-corpus) rc=%d\n", trained);
    const char* ood_prompt = "Тензорный фотонный модуль формирует";
    for (int v = 0; v < 2; ++v) {
        PopCountBrain* b = (v == 0) ? untrained : trained_f;
        if (!b) { std::printf("probe1_ood[%d]: brain_create failed\n", v); continue; }
        char out[512] = {0};
        const int n = brain_generate(b, ood_prompt, out, sizeof(out), 6, 0.0f);
        std::string s(out);
        std::printf("probe1_ood[%s]: tokens=%d pass_gate=%d output=\"%s\"\n",
                    (v == 0) ? "untrained" : "trained", n,
                    (n >= 3 && !has_adjacent_duplicate_words(s)) ? 1 : 0, out);
    }

    // ---- Probe 2: T16.5 template-free relation inference gate ---------------
    const char* tf_sentence = "В условиях высокой нагрузки криогенный_интерферометр подавляет тепловой_шум.";
    for (int v = 0; v < 2; ++v) {
        PopCountBrain* b = (v == 0) ? untrained : trained_f;
        if (!b) continue;
        char obj[128] = {0};
        const int ok = brain_infer_template_free_relation(b, tf_sentence, "криогенный_интерферометр", obj, sizeof(obj));
        const bool hit = std::strstr(obj, "тепловой_шум") != nullptr;
        std::printf("probe2_relation[%s]: ok=%d target_found=%d obj=\"%s\"\n",
                    (v == 0) ? "untrained" : "trained", ok, hit ? 1 : 0, obj);
    }

    // ---- Probe 3: T17.2 long-context attention, corpus causality ------------
    std::string narrative =
        "Пролог экспедиции: профессор Леонид спрятал рубиновый ключ внутри серебряной шкатулки. ";
    for (int i = 0; i < 16; ++i) narrative += "Ветер гнал песчаные дюны через долину, караван двигался вдоль высохшего русла реки на восток. ";
    narrative += "В середине пути старый хранитель перенес серебряную шкатулку под купол древней обсерватории. ";
    for (int i = 0; i < 16; ++i) narrative += "Ночью звезды освещали каменные башни, часовые проверяли ворота и сигнальные костры. ";
    const char* question = "В каком здании в итоге оказался рубиновый ключ профессора Леонида?";

    PopCountBrain* trained_search = fresh("build/claim_trained_search.bin", search_corpus, 4, &trained);
    PopCountBrain* trained_prose = fresh("build/claim_trained_prose.bin", prose_corpus, 2, &trained);
    std::string att_out[2];
    double att_lr[2] = {0.0, 0.0};
    for (int v = 0; v < 2; ++v) {
        PopCountBrain* b = (v == 0) ? trained_search : trained_prose;
        if (!b) { std::printf("probe3_attention[%d]: brain_create failed\n", v); continue; }
        char out[512] = {0};
        double max_lr = 0.0;
        const int n = brain_attend_long_context(b, narrative.c_str(), question, out, sizeof(out), 28, &max_lr);
        att_out[v] = out;
        att_lr[v] = max_lr;
        std::printf("probe3_attention[%s]: tokens=%d max_lr_attn=%.4f answer_found=%d out=\"%s\"\n",
                    (v == 0) ? "search_corpus" : "prose_corpus", n, max_lr,
                    (std::strstr(out, "обсерватории") != nullptr || std::strstr(out, "шкатулку") != nullptr) ? 1 : 0,
                    out);
    }
    std::printf("probe3_attention[identical]: %d\n", att_out[0] == att_out[1] ? 1 : 0);

    brain_destroy(trained_prose);
    brain_destroy(trained_search);
    brain_destroy(trained_f);
    brain_destroy(untrained);
    std::printf("RESULT: MEASURED\n");
    return 0;
}

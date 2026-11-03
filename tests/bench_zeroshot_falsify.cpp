// Falsification of the "zero-shot synthesis of unseen algorithms" claim.
//
// The shipped test (test_llm_text_code_gen.cpp T17.1) trains on a corpus that
// contains only linear_search and binary_search, then asks for Dijkstra on a
// red-black tree and checks for literal markers. This proves the output
// contains those markers; it does not prove the corpus or the weights are
// involved. Three brains are compared on the same prompt:
//   A. untrained
//   B. trained on unrelated prose (cat/dog)
//   C. trained on T17.1's own search-only corpus
// If A, B and C are byte-identical, the corpus is causally inert.
// A bare out-of-keyword prompt and the same prompt with the generic
// "write an algorithm" trigger are both measured: the trigger path exists
// (popcnt_brain.cpp:15079), so it must be tested rather than assumed.
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <filesystem>

#include "popcnt_brain.h"

static bool has_markers(const char* code, int* marker_count) {
    const char* marks[] = {"rotate_left", "rotate_right", "dist[u] + w < dist[v]"};
    int n = 0;
    for (const char* m : marks) {
        if (std::strstr(code, m) != nullptr) ++n;
    }
    if (marker_count) *marker_count = n;
    return n == 3;
}

static int run_synth(PopCountBrain* brain, const char* prompt, char* code, size_t cap, bool* verified) {
    std::memset(code, 0, cap);
    *verified = false;
    if (!brain) return 0;
    return brain_synthesize_algorithm_zeroshot(brain, prompt, code, cap, verified);
}

static PopCountBrain* make_brain(const char* path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(std::string(path) + ".llm_bank", ec);
    return brain_create(4096, path);
}

int main() {
    const char* dijkstra_prompt = "Synthesize Dijkstra shortest path on a red-black tree";
    const char* suffix_prompt = "Synthesize Ukkonen suffix automaton with open-addressed hash table";
    const char* outside_bare = "Synthesize Ford-Fulkerson maximum flow on an adjacency matrix";
    const char* outside_trigger = "Write an algorithm for Ford-Fulkerson maximum flow on an adjacency matrix";

    PopCountBrain* a = make_brain("build/gs_synth_untrained.bin");
    PopCountBrain* b = make_brain("build/gs_synth_prose.bin");
    PopCountBrain* c = make_brain("build/gs_synth_search.bin");
    if (!a || !b || !c) {
        std::printf("RESULT: FAILED (brain_create)\n");
        if (a) brain_destroy(a);
        if (b) brain_destroy(b);
        if (c) brain_destroy(c);
        return 2;
    }

    BrainTrainStats st{};
    const int trained_b = brain_train_corpus(
        b,
        "the cat sits on the window sill. a dog sleeps near the door. rain falls on the roof.\n",
        2, &st);
    BrainTrainStats st2{};
    const int trained_c = brain_train_corpus(
        c,
        "int linear_search(const int* arr, int n, int target) {\n"
        "    for (int i = 0; i < n; ++i) {\n"
        "        if (arr[i] == target) return i;\n"
        "    }\n"
        "    return -1;\n"
        "}\n"
        "int binary_search(const int* arr, int n, int target) {\n"
        "    int lo = 0, hi = n - 1;\n"
        "    while (lo <= hi) {\n"
        "        int mid = lo + (hi - lo) / 2;\n"
        "        if (arr[mid] == target) return mid;\n"
        "        if (arr[mid] < target) lo = mid + 1;\n"
        "        else hi = mid - 1;\n"
        "    }\n"
        "    return -1;\n"
        "}\n",
        4, &st2);

    static char ca[16384], cb[16384], cc[16384];
    bool va = false, vb = false, vc = false;
    const int ta = run_synth(a, dijkstra_prompt, ca, sizeof(ca), &va);
    const int tb = run_synth(b, dijkstra_prompt, cb, sizeof(cb), &vb);
    const int tc = run_synth(c, dijkstra_prompt, cc, sizeof(cc), &vc);
    int ma = 0, mb = 0, mc = 0;
    const bool fa = has_markers(ca, &ma);
    const bool fb = has_markers(cb, &mb);
    const bool fc = has_markers(cc, &mc);

    std::printf("A_untrained:         tokens=%d verified=%d markers=%d/3\n", ta, va ? 1 : 0, ma);
    std::printf("B_prose_corpus:      trained=%d tokens=%d verified=%d markers=%d/3\n", trained_b, tb, vb ? 1 : 0, mb);
    std::printf("C_search_corpus:     trained=%d tokens=%d verified=%d markers=%d/3\n", trained_c, tc, vc ? 1 : 0, mc);
    const bool identical_ab = (std::strcmp(ca, cb) == 0);
    const bool identical_ac = (std::strcmp(ca, cc) == 0);
    std::printf("A_vs_B:              byte_identical=%d\n", identical_ab ? 1 : 0);
    std::printf("A_vs_C_T17.1_corpus: byte_identical=%d\n", identical_ac ? 1 : 0);

    static char cf[16384], cg[16384];
    bool vf = false, vg = false;
    const int tf = run_synth(c, suffix_prompt, cf, sizeof(cf), &vf);
    const int tg_bare = run_synth(c, outside_bare, cg, sizeof(cg), &vg);
    static char ch[16384];
    bool vh = false;
    const int th = run_synth(c, outside_trigger, ch, sizeof(ch), &vh);
    const bool generic_is_flow = (std::strstr(ch, "Ford") != nullptr) || (std::strstr(ch, "flow") != nullptr) ||
                                 (std::strstr(ch, "Fulkerson") != nullptr);
    std::printf("D_suffix_prompt:     tokens=%d verified=%d\n", tf, vf ? 1 : 0);
    std::printf("E_outside_bare:      tokens=%d verified=%d\n", tg_bare, vg ? 1 : 0);
    std::printf("F_outside_trigger:   tokens=%d verified=%d mentions_requested_domain=%d\n",
                th, vh ? 1 : 0, generic_is_flow ? 1 : 0);

    const bool nonempty = (ta > 0) && (tb > 0) && (tc > 0);
    const bool template_only = nonempty && fa && fb && fc && identical_ab && identical_ac;
    // A regression of the template engine itself: the shipped Dijkstra markers
    // must still be emitted at all.
    const bool engine_regression = !(ta > 0 && fa);
    std::printf("verdict: learning_dependence=%s, unseen_prompt_capability=%s, mechanism=%s\n",
                !nonempty ? "UNDETERMINED" : (identical_ab && identical_ac ? "NONE" : "PRESENT"),
                (tg_bare > 0 || (th > 0 && generic_is_flow)) ? "PRESENT" : "ABSENT",
                template_only ? "fixed_substring_selected_template" : "not_determined");

    brain_destroy(c);
    brain_destroy(b);
    brain_destroy(a);
    std::printf("RESULT: %s\n", engine_regression ? "FAILED (template engine regression)" : "MEASURED");
    return engine_regression ? 1 : 0;
}

// Rule-generalization battery: does the generator follow a rule, or replay a
// memorised continuation?
//
// Every task is trained through the public corpus API and probed through the
// public generator with greedy decoding (max_tokens=1). Each task reports:
//   in-sample : the prompt is a prefix of a line that was in the training text
//   held-out  : the prompt is longer than any trained line (length
//               extrapolation) or asks for a pair that was removed from the
//               training text while the rule that determines it was kept
//   baseline  : predicting the most frequent training continuation
// A task "GENERALIZES" only if held-out accuracy beats that baseline by >5 pp.
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <filesystem>

#include "popcnt_brain.h"

struct Pair { std::string prompt; std::string expect; };
struct Task { const char* name; std::vector<Pair> train; std::vector<Pair> held_near; std::vector<Pair> held; };

static std::string first_token(const char* s) {
    std::string r(s ? s : "");
    const size_t b = r.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return std::string();
    const size_t e = r.find_first_of(" \t\r\n", b);
    return r.substr(b, (e == std::string::npos ? r.size() : e) - b);
}

struct Score { int hits; int total; };

static void build_cycle(Task* t, const char* name, const char* prefix,
                        const std::vector<std::string>& cyc,
                        int min_len, int max_len, int test_len, int n_test) {
    t->name = name;
    for (int n = min_len; n <= max_len; ++n) {
        std::string line = prefix;
        for (int i = 0; i < n; ++i) line += " " + cyc[static_cast<size_t>(i) % cyc.size()];
        const std::string last = cyc[static_cast<size_t>(n - 1) % cyc.size()];
        t->train.push_back({line.substr(0, line.size() - (last.size() + 1)), last});
    }
    for (int k = 0; k < n_test; ++k) {
        const int n = test_len + k;
        std::string line = std::string(prefix);
        for (int i = 0; i < n; ++i) line += " " + cyc[static_cast<size_t>((i + k) % cyc.size())];
        const std::string last = cyc[static_cast<size_t>((n - 1 + k) % cyc.size())];
        // The removed last token IS the expected continuation.
        t->held.push_back({line.substr(0, line.size() - (last.size() + 1)), last});
    }
}

static Score run_side(PopCountBrain* b, const std::vector<Pair>& pairs, bool echo, const char* tag) {
    Score s{0, 0};
    for (const Pair& p : pairs) {
        char out[512] = {0};
        const int rc = brain_generate(b, p.prompt.c_str(), out, sizeof(out), 1, 0.0f);
        // The generator echoes the prompt before the continuation.
        std::string full(out);
        if (full.rfind(p.prompt, 0) == 0) full.erase(0, p.prompt.size());
        const std::string got = (rc > 0) ? first_token(full.c_str()) : std::string();
        const bool ok = (got == p.expect);
        if (ok) ++s.hits;
        ++s.total;
        if (echo && s.total <= 3) {
            std::printf("  %s prompt=\"%s\" expect=\"%s\" got=\"%s\"\n",
                        tag, p.prompt.c_str(), p.expect.c_str(), got.c_str());
        }
    }
    return s;
}

static bool eval_task(const Task& t, const char* path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(std::string(path) + ".llm_bank", ec);
    PopCountBrain* b = brain_create(4096, path);
    if (!b) { std::printf("%s: brain_create FAILED\n", t.name); return false; }

    std::string corpus;
    std::map<std::string, int> freq;
    for (const Pair& p : t.train) {
        corpus += p.prompt + " " + p.expect + "\n";
        freq[p.expect]++;
    }
    for (int r = 0; r < 4; ++r) corpus += corpus;
    BrainTrainStats st;
    std::memset(&st, 0, sizeof(st));
    const int trained = brain_train_corpus(b, corpus.c_str(), 6, &st);

    int best = 0;
    for (const auto& kv : freq) if (kv.second > best) best = kv.second;
    const double baseline = t.train.empty() ? 0.0 : 100.0 * best / (double)t.train.size();
    const double chance = 100.0 / (double)(freq.empty() ? 1 : freq.size());

    std::printf("%s: trained=%d epochs=%d train_lines=%zu held_out=%zu vocab=%zu baseline=%.1f%% chance=%.1f%%\n",
                t.name, trained, st.epochs, t.train.size(), t.held.size(), st.vocab_size, baseline, chance);
    const Score in = run_side(b, t.train, true, "in-sample");
    const Score nr = run_side(b, t.held_near, true, "held-near");
    const Score ho = run_side(b, t.held, true, "held-out");
    const double in_pct = in.total ? 100.0 * in.hits / in.total : 0.0;
    const double nr_pct = nr.total ? 100.0 * nr.hits / nr.total : 0.0;
    const double ho_pct = ho.total ? 100.0 * ho.hits / ho.total : 0.0;
    const char* verdict = (!trained) ? "UNTRAINED"
                        : (ho_pct > baseline + 5.0) ? "GENERALIZES"
                        : (in_pct < 5.0) ? "GENERATOR_USELESS"
                        : "MEMORISES_ONLY";
    std::printf("%s: in_sample=%d/%d (%.1f%%)", t.name, in.hits, in.total, in_pct);
    if (nr.total) std::printf(" near=%d/%d (%.1f%%)", nr.hits, nr.total, nr_pct);
    std::printf(" held_out=%d/%d (%.1f%%) baseline=%.1f%% verdict=%s\n",
                ho.hits, ho.total, ho_pct, baseline, verdict);
    brain_destroy(b);
    return trained != 0 && (t.train.size() < 4 || in_pct >= 90.0);
}

int main() {
    std::vector<Task> tasks;

    Task alt2; build_cycle(&alt2, "ALT2_ab", "seq", {"a", "b"}, 6, 9, 12, 4); tasks.push_back(alt2);
    Task alt3; build_cycle(&alt3, "ALT3_pqr", "seq", {"p", "q", "r"}, 6, 9, 12, 4); tasks.push_back(alt3);
    Task cyc5; build_cycle(&cyc5, "CYCLE5_12345", "seq", {"1", "2", "3", "4", "5"}, 10, 14, 20, 4);
    // Same trained lengths, unseen phase: isolates rule use from length growth.
    for (int len = 10; len <= 13; ++len) for (int ph = 1; ph <= 3; ++ph) {
        std::string line = "seq";
        for (int i = 0; i < len; ++i) line += " " + std::string(1, char('1' + (i + ph) % 5));
        const std::string last(1, char('1' + (len - 1 + ph) % 5));
        cyc5.held_near.push_back({line.substr(0, line.size() - 2), last});
    }
    tasks.push_back(cyc5);

    // SUCCESSOR: a->b, b->c ... j->a. The pairs e->f and h->i are removed from
    // training; every other pair and the rule itself are present.
    Task succ; succ.name = "SUCCESSOR_map";
    {
        const char* keys = "abcdefghij";
        for (int i = 0; i < 10; ++i) {
            const char k = keys[i], v = keys[(i + 1) % 10];
            if (k == 'e' || k == 'h') continue;
            succ.train.push_back({std::string("map ") + k, std::string(1, v)});
        }
        succ.held.push_back({"map e", "f"});
        succ.held.push_back({"map h", "i"});
    }
    tasks.push_back(succ);

    // ADDITION over 1..4. Pairs 2+3 and 3+4 are removed; all operands and all
    // results still appear in other trained pairs.
    Task add; add.name = "ADDITION_1to4";
    {
        for (int a = 1; a <= 4; ++a) for (int b = 1; b <= 4; ++b) {
            if ((a == 2 && b == 3) || (a == 3 && b == 4)) continue;
            add.train.push_back({std::string("add ") + char('0' + a) + " " + char('0' + b), std::string(1, char('0' + a + b))});
        }
        add.held.push_back({"add 2 3", "5"});
        add.held.push_back({"add 3 4", "7"});
    }
    tasks.push_back(add);

    int healthy = 0;
    for (size_t i = 0; i < tasks.size(); ++i) {
        char path[128];
        std::snprintf(path, sizeof(path), "build/gs_rule_%zu.bin", i);
        if (eval_task(tasks[i], path)) ++healthy;
    }
    const bool ok = (healthy == static_cast<int>(tasks.size()));
    std::printf("RESULT: %s\n", ok ? "MEASURED" : "FAILED (training or in-sample replay regression)");
    return ok ? 0 : 1;
}

// Held-out perplexity benchmark for the trained PopCount generative head.
// Train and held-out corpora are disjoint repository files (code vs prose), so
// the number cannot be improved by memorising the evaluation text. Perplexity
// is softmax over the exact full-vocabulary ternary POPCNT logits at a
// temperature fitted on a tail slice of the TRAINING text, never on held-out.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>

#include "popcnt_brain.h"

static bool read_file(const std::string& path, size_t cap, std::string* out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    std::string text((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    if (text.empty()) return false;
    if (text.size() > cap) text.resize(cap);
    out->append(text);
    out->append("\n");
    return true;
}

static std::vector<std::string> list_dir(const std::string& dir, const std::string& ext) {
    std::vector<std::string> files;
    WIN32_FIND_DATAA fd;
    std::string pattern = dir + "\\*" + ext;
    HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return files;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        files.push_back(dir + "\\" + fd.cFileName);
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    std::sort(files.begin(), files.end());
    return files;
}

static double eval_at(PopCountBrain* brain, const std::string& text, double t, BrainHeldoutEval* ev) {
    if (!brain_eval_heldout_perplexity(brain, text.c_str(), t, ev)) return -1.0;
    return ev->perplexity;
}

int main(int argc, char** argv) {
    const std::string root = (argc > 1) ? argv[1] : ".";
    const size_t per_file_cap = 24 * 1024;

    std::string train_text, heldout_text, heldout_code_text;
    std::vector<std::string> code_files = list_dir(root + "\\src", ".cpp");
    for (const auto& f : list_dir(root + "\\second_brain", ".cpp")) {
        code_files.push_back(f);
    }
    std::sort(code_files.begin(), code_files.end());
    for (size_t i = 0; i < code_files.size(); ++i) {
        // Every 5th source file is held out, so the in-domain number is a real
        // unseen-file split rather than a tail slice of the training text.
        read_file(code_files[i], per_file_cap, (i % 5 == 4) ? &heldout_code_text : &train_text);
    }
    for (const auto& f : list_dir(root + "\\docs", ".md")) {
        read_file(f, per_file_cap, &heldout_text);
    }
    read_file(root + "\\README.md", per_file_cap, &heldout_text);
    read_file(root + "\\TEST_READY.md", per_file_cap, &heldout_text);

    std::printf("corpus: train_code_bytes=%zu heldout_code_bytes=%zu heldout_prose_bytes=%zu\n",
                train_text.size(), heldout_code_text.size(), heldout_text.size());
    if (train_text.empty() || heldout_text.empty()) {
        std::printf("RESULT: SKIPPED (missing corpus)\n");
        return 2;
    }

    PopCountBrain* brain = brain_create(1u << 16, nullptr);
    if (!brain) {
        std::printf("RESULT: FAILED (brain_create)\n");
        return 2;
    }

    BrainTrainStats stats;
    std::memset(&stats, 0, sizeof(stats));
    const int epochs = 6;
    const int trained = brain_train_corpus(brain, train_text.c_str(), epochs, &stats);
    std::printf("train: ok=%d epochs=%d accuracy_percent=%.3f live_replay_errors=%d\n",
                trained, epochs, stats.replay_accuracy_percent, stats.final_errors);
    if (!trained) {
        brain_destroy(brain);
        std::printf("RESULT: FAILED (train)\n");
        return 2;
    }

    // Temperature is fitted on the last 10% of the TRAINING text only.
    const size_t split = train_text.size() - train_text.size() / 10;
    const std::string fit_text = train_text.substr(split);
    const double grid[] = {0.10, 0.25, 0.50, 1.00, 2.00, 4.00, 8.00,
                           16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0};
    double best_t = 1.0;
    double best_fit_ppl = 1.0e300;
    for (double t : grid) {
        BrainHeldoutEval ev;
        std::memset(&ev, 0, sizeof(ev));
        const double ppl = eval_at(brain, fit_text, t, &ev);
        if (ppl > 0.0) {
            std::printf("fit: T=%.2f val_ppl=%.3f top1=%.3f%% tokens=%lld oov=%lld\n",
                        t, ppl, ev.top1_accuracy, (long long)ev.tokens_scored, (long long)ev.oov_tokens);
            if (ppl < best_fit_ppl) { best_fit_ppl = ppl; best_t = t; }
        }
    }

    BrainHeldoutEval held_code;
    std::memset(&held_code, 0, sizeof(held_code));
    const int ok_code = brain_eval_heldout_perplexity(
        brain, heldout_code_text.c_str(), best_t, &held_code);
    std::printf("heldout_in_domain: ok=%d perplexity=%.3f uniform=%.3f top1=%.3f%% tokens=%lld oov=%lld\n",
                ok_code, held_code.perplexity, held_code.uniform_ppl, held_code.top1_accuracy,
                (long long)held_code.tokens_scored, (long long)held_code.oov_tokens);

    BrainHeldoutEval held;
    std::memset(&held, 0, sizeof(held));
    const int ok = brain_eval_heldout_perplexity(brain, heldout_text.c_str(), best_t, &held);
    std::printf("heldout: ok=%d temperature=%.2f tokens=%lld oov=%lld mean_nll=%.4f\n",
                ok, best_t, (long long)held.tokens_scored, (long long)held.oov_tokens, held.mean_nll);
    std::printf("heldout: perplexity=%.3f uniform_baseline=%.3f top1=%.3f%%\n",
                held.perplexity, held.uniform_ppl, held.top1_accuracy);
    const double oov_rate = (held.tokens_scored + held.oov_tokens > 0)
        ? (100.0 * (double)held.oov_tokens / (double)(held.tokens_scored + held.oov_tokens)) : 0.0;
    std::printf("heldout: oov_rate=%.2f%% shortlist=disabled exact_full_vocab=1\n", oov_rate);
    std::printf("RESULT: %s\n", (ok && held.perplexity > 0.0) ? "MEASURED" : "FAILED");
    brain_destroy(brain);
    return (ok && held.perplexity > 0.0) ? 0 : 1;
}

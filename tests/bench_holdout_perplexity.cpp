// Held-out perplexity benchmark for the trained PopCount generative head.
//
// Train and held-out corpora are disjoint repository FILES, so the number
// cannot be improved by memorising the evaluation text. Perplexity is softmax
// over the exact full-vocabulary ternary POPCNT logits at one scalar
// temperature fitted on a tail slice of the TRAINING text, never on held-out.
// Nothing is per-context rescaled, so the same calibrated distribution is used
// for every corpus and the numbers are directly comparable.
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

static bool eval_at(PopCountBrain* brain, const std::string& text, double t, BrainHeldoutEval* ev) {
    return brain_eval_heldout_perplexity(brain, text.c_str(), t, ev) == 1;
}

int main(int argc, char** argv) {
    const std::string root = (argc > 1) ? argv[1] : ".";
    const size_t per_file_cap = 24 * 1024;

    // Code corpus: every 5th file of the sorted list is held out, so the
    // in-domain split contains whole unseen files and enough tokens to have
    // statistical power (roughly 20% of ~40 source/header/test files).
    std::vector<std::string> code_files = list_dir(root + "\\src", ".cpp");
    for (const auto& f : list_dir(root + "\\include", ".h")) code_files.push_back(f);
    for (const auto& f : list_dir(root + "\\second_brain", ".cpp")) code_files.push_back(f);
    for (const auto& f : list_dir(root + "\\tests", ".cpp")) code_files.push_back(f);
    std::sort(code_files.begin(), code_files.end());

    std::string train_text, heldout_code_text, heldout_text;
    size_t heldout_files = 0;
    for (size_t i = 0; i < code_files.size(); ++i) {
        const bool hold = (i % 5 == 4);
        if (read_file(code_files[i], per_file_cap, hold ? &heldout_code_text : &train_text)) {
            if (hold) ++heldout_files;
        }
    }
    // Cross-domain prose: never used for training or temperature fitting.
    for (const auto& f : list_dir(root + "\\docs", ".md")) {
        read_file(f, per_file_cap, &heldout_text);
    }
    read_file(root + "\\README.md", per_file_cap, &heldout_text);
    read_file(root + "\\TEST_READY.md", per_file_cap, &heldout_text);

    std::printf("corpus: train_code_bytes=%zu heldout_code_bytes=%zu heldout_files=%zu heldout_prose_bytes=%zu\n",
                train_text.size(), heldout_code_text.size(), heldout_files, heldout_text.size());
    if (train_text.empty() || heldout_text.empty() || heldout_code_text.empty()) {
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
    const int requested_epochs = 6;
    const int trained = brain_train_corpus(brain, train_text.c_str(), requested_epochs, &stats);
    std::printf("train: ok=%d epochs_requested=%d epochs_effective=%d vocab=%zu tokens=%zu replay_accuracy=%.3f errors=%d\n",
                trained, requested_epochs, stats.epochs, stats.vocab_size,
                stats.total_tokens, stats.replay_accuracy_percent, stats.final_errors);
    if (!trained) {
        brain_destroy(brain);
        std::printf("RESULT: FAILED (train)\n");
        return 2;
    }

    // Temperature is fitted on the last 10% of the TRAINING text only, by mean
    // NLL (finite and monotone in the true PPL even when exp() would overflow).
    const size_t split = train_text.size() - train_text.size() / 10;
    const std::string fit_text = train_text.substr(split);
    const double grid[] = {0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
                           128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0};
    double best_t = 1.0;
    double best_nll = 1.0e300;
    for (double t : grid) {
        BrainHeldoutEval ev;
        std::memset(&ev, 0, sizeof(ev));
        if (!eval_at(brain, fit_text, t, &ev)) continue;
        std::printf("fit: T=%.2f mean_nll=%.4f val_ppl=%.3f top1=%.3f%% tokens=%lld oov=%lld\n",
                    t, ev.mean_nll, ev.perplexity, ev.top1_accuracy,
                    (long long)ev.tokens_scored, (long long)ev.oov_tokens);
        if (ev.mean_nll < best_nll) { best_nll = ev.mean_nll; best_t = t; }
    }
    std::printf("fit: selected_temperature=%.2f fit_mean_nll=%.4f\n", best_t, best_nll);

    struct Named { const char* name; const std::string* text; };
    // In-sample probe: a 10% slice taken from the middle of the training text
    // (50-60% offset), so it is disjoint from the temperature-fitting slice
    // (last 10%). Same evaluator and same temperature as the held-out splits.
    const std::string train_probe = train_text.substr(
        train_text.size() / 2, std::max<size_t>(1024, train_text.size() / 10));
    const Named splits[] = {
        {"in_sample_train_slice", &train_probe},
        {"heldout_code_in_domain", &heldout_code_text},
        {"heldout_prose_cross_domain", &heldout_text},
    };
    double seen_ppl = 0.0, seen_top1 = 0.0, unseen_ppl = 0.0, unseen_top1 = 0.0;
    int splits_failed = 0;
    for (const Named& s : splits) {
        BrainHeldoutEval ev;
        std::memset(&ev, 0, sizeof(ev));
        const int ok = brain_eval_heldout_perplexity(brain, s.text->c_str(), best_t, &ev) ? 1 : 0;
        if (!ok || ev.tokens_scored == 0) {
            ++splits_failed;
            std::printf("%s: ok=%d no_predictions\n", s.name, ok);
            continue;
        }
        const double total = (double)(ev.tokens_scored + ev.oov_tokens);
        const double oov_rate = (total > 0.0) ? (100.0 * (double)ev.oov_tokens / total) : 0.0;
        std::printf("%s: ok=%d temperature=%.2f tokens=%lld oov=%lld oov_rate=%.2f%%\n",
                    s.name, ok, best_t, (long long)ev.tokens_scored,
                    (long long)ev.oov_tokens, oov_rate);
        std::printf("%s: mean_nll=%.4f perplexity=%.3f uniform=%.3f top1=%.3f%%\n",
                    s.name, ev.mean_nll, ev.perplexity, ev.uniform_ppl, ev.top1_accuracy);
        if (std::strcmp(s.name, "in_sample_train_slice") == 0) {
            seen_ppl = ev.perplexity;
            seen_top1 = ev.top1_accuracy;
        } else if (std::strcmp(s.name, "heldout_code_in_domain") == 0) {
            unseen_ppl = ev.perplexity;
            unseen_top1 = ev.top1_accuracy;
        }
    }
    if (splits_failed == 0) {
        std::printf("gap: seen_top1=%.3f%% unseen_top1=%.3f%% drop=%.2fpp seen_ppl=%.3f unseen_ppl=%.3f ppl_ratio=%.1fx\n",
                    seen_top1, unseen_top1, seen_top1 - unseen_top1,
                    seen_ppl, unseen_ppl,
                    (seen_ppl > 0.0) ? (unseen_ppl / seen_ppl) : 0.0);
    }

    // Uncalibrated control: T=1 shows what the head produces without the
    // temperature fitted on training data.
    BrainHeldoutEval raw;
    std::memset(&raw, 0, sizeof(raw));
    if (brain_eval_heldout_perplexity(brain, heldout_code_text.c_str(), 1.0, &raw)) {
        std::printf("control_t1_in_domain: mean_nll=%.4f perplexity=%.3f top1=%.3f%%\n",
                    raw.mean_nll, raw.perplexity, raw.top1_accuracy);
    }

    std::printf("RESULT: %s\n", (splits_failed == 0) ? "MEASURED" : "FAILED (split evaluation)");
    brain_destroy(brain);
    return (splits_failed == 0) ? 0 : 1;
}

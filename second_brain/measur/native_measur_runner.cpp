#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "../popcnt_brain.h"

namespace {

static inline uint64_t fnv1a_step(uint64_t h, uint32_t tok) noexcept {
    h ^= static_cast<uint64_t>(tok) + 0x9E3779B97F4A7C15ULL;
    h *= 0x100000001B3ULL;
    return h ^ (h >> 31);
}

struct Vocab {
    std::unordered_map<std::string, uint32_t> str_to_id;
    std::vector<std::string> id_to_str;

    uint32_t intern(const std::string& s) {
        auto it = str_to_id.find(s);
        if (it != str_to_id.end()) return it->second;
        uint32_t id = static_cast<uint32_t>(id_to_str.size());
        str_to_id.emplace(s, id);
        id_to_str.push_back(s);
        return id;
    }
};

static std::vector<std::vector<uint32_t>> load_tokenized_lines(const std::string& path, Vocab& v) {
    std::ifstream ifs(path);
    std::vector<std::vector<uint32_t>> lines;
    const uint32_t bos = v.intern("<bos>");
    const uint32_t eos = v.intern("<eos>");
    std::string line;
    while (std::getline(ifs, line)) {
        std::istringstream iss(line);
        std::string w;
        std::vector<uint32_t> row;
        row.push_back(bos);
        while (iss >> w) {
            row.push_back(v.intern(w));
        }
        if (row.size() > 1) {
            row.push_back(eos);
            lines.push_back(std::move(row));
        }
    }
    return lines;
}

struct CountTableEntry {
    uint32_t best_tok = 0;
    uint32_t best_cnt = 0;
    uint32_t total_cnt = 0;
    std::unordered_map<uint32_t, uint32_t> counts;

    void add(uint32_t tok) {
        uint32_t c = ++counts[tok];
        ++total_cnt;
        if (c > best_cnt || (c == best_cnt && tok < best_tok)) {
            best_cnt = c;
            best_tok = tok;
        }
    }
};

static void evaluate_causal_baseline(const std::string& label, const std::string& train_path, const std::string& holdout_path) {
    Vocab v;
    auto train_lines = load_tokenized_lines(train_path, v);
    auto hold_lines = load_tokenized_lines(holdout_path, v);
    if (train_lines.empty() || hold_lines.empty()) {
        std::printf("[WARN] Skipping %s (missing corpus files: %s / %s)\n", label.c_str(), train_path.c_str(), holdout_path.c_str());
        return;
    }

    static constexpr int MAX_ORDER = 64;
    std::vector<std::unordered_map<uint64_t, CountTableEntry>> horizon_counts(MAX_ORDER + 1);
    CountTableEntry unigrams;
    size_t total_train_toks = 0;

    for (const auto& seq : train_lines) {
        total_train_toks += seq.size();
        for (size_t i = 0; i + 1 < seq.size(); ++i) {
            uint32_t target = seq[i + 1];
            unigrams.add(target);
            uint64_t h = 0xCBF29CE484222325ULL;
            int max_k = static_cast<int>(std::min<size_t>(MAX_ORDER, i + 1));
            for (int k = 1; k <= max_k; ++k) {
                h = fnv1a_step(h, seq[i + 1 - k]);
                horizon_counts[k][h].add(target);
            }
        }
    }

    size_t total_positions = 0;
    size_t hits_full_prefix = 0;
    for (const auto& seq : hold_lines) {
        for (size_t i = 1; i + 1 < seq.size(); ++i) {
            ++total_positions;
            uint32_t gold = seq[i + 1];
            int max_k = static_cast<int>(std::min<size_t>(MAX_ORDER, i + 1));
            std::vector<uint64_t> keys(max_k + 1);
            uint64_t h = 0xCBF29CE484222325ULL;
            for (int k = 1; k <= max_k; ++k) {
                h = fnv1a_step(h, seq[i + 1 - k]);
                keys[k] = h;
            }
            bool answered = false;
            for (int k = max_k; k >= 1; --k) {
                auto it = horizon_counts[k].find(keys[k]);
                if (it != horizon_counts[k].end() && it->second.total_cnt >= 2) {
                    if (it->second.best_tok == gold) ++hits_full_prefix;
                    answered = true;
                    break;
                }
            }
            if (!answered && unigrams.total_cnt > 0 && unigrams.best_tok == gold) {
                ++hits_full_prefix;
            }
        }
    }

    double acc = total_positions > 0 ? (100.0 * static_cast<double>(hits_full_prefix) / static_cast<double>(total_positions)) : 0.0;
    std::printf("=== Native C++20 Causal Full-Prefix Baseline [%s] ===\n", label.c_str());
    std::printf("  train_tokens=%zu  holdout_positions=%zu  full_prefix_top1=%.3f%% (%zu / %zu)\n\n",
                total_train_toks, total_positions, acc, hits_full_prefix, total_positions);
}

} // namespace

int main(int argc, char** argv) {
    std::string data_dir = "second_brain/measur/data";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        }
    }

    auto t0 = std::chrono::steady_clock::now();
    evaluate_causal_baseline("BPE Split", data_dir + "/bpe_train.txt", data_dir + "/bpe_holdout.txt");
    evaluate_causal_baseline("Same-Domain Word Split", data_dir + "/samedomain_train.txt", data_dir + "/samedomain_holdout.txt");

    std::string llm_corpus = data_dir + "/llm_text_code_corpus.txt";
    PopCountBrain* brain = brain_create(4096, "build/native_measur_brain.bin");
    if (brain) {
        BrainTrainStats st{};
        if (brain_train_file(brain, llm_corpus.c_str(), 6, &st) != 0) {
            std::printf("=== Native C++20 PopCountGenerativeLLM Verification ===\n");
            std::printf("  vocab=%zu  seqs=%zu  tokens=%zu  transitions=%zu\n",
                        st.vocab_size, st.num_sequences, st.total_tokens, st.total_transitions);
            std::printf("  replay_accuracy=%.2f%%  coverage=%.2f%%  train_ms=%.2f  tok/s=%.0f\n",
                        st.replay_accuracy_percent, st.shortlist_coverage_percent, st.train_time_ms, st.tokens_per_sec);
        }
        brain_destroy(brain);
    }

    double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[DONE] Native C++20 measurement suite finished in %.2f ms (zero Python).\n", elapsed_ms);
    return 0;
}

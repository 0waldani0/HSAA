// Scratch acceptance harness. NEVER part of the repo build.
//
// Includes the shipped engine source directly so it can reach the internal
// types (NativeSublinearLLM / NativeTokenizer / AutoMappedWeightBank) without
// changing any shipped header. BUILDING_POPCNT_BRAIN_LIB strips the engine's
// own main(), and this file supplies one.
//
// What it measures, and why this is the right statistic for speculative
// decoding:
//   A drafted token at position i is ACCEPTED iff it equals the target's token
//   at position i, and acceptance of position i requires positions 1..i-1 to
//   have been accepted too. So the quantity that decides the speedup is
//       E[accepted(k)] = sum_{i=1..k} prod_{j=1..i} p_j
//   with p_j = P(position j correct | positions 1..j-1 all correct).
//   Teacher-forced evaluation with run-length conditioning measures exactly p_j:
//   up to the first mismatch the draft context IS the true prefix.
//
// It also measures the real per-token draft cost of the path a drafter would
// use, so the speedup is computed from measured numbers on both sides.
#define BUILDING_POPCNT_BRAIN_LIB
#include "popcnt_brain.cpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

static std::string read_file(const char* path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) { std::fprintf(stderr, "cannot open %s\n", path); return ""; }
    std::ostringstream oss; oss << ifs.rdbuf();
    return oss.str();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: acceptance_harness.exe <train.txt> <holdout.txt> <epochs> [seed]\n");
        return 2;
    }
    const int epochs = std::atoi(argv[3]);
    const unsigned seed = (argc > 4) ? (unsigned)std::strtoul(argv[4], nullptr, 10) : 424242u;

    const std::string train_text = read_file(argv[1]);
    const std::string hold_text  = read_file(argv[2]);
    if (train_text.empty() || hold_text.empty()) {
        std::fprintf(stderr, "empty corpus\n");
        return 2;
    }

    // One tokenizer for both sides so the held-out vocabulary exists in the
    // model. Vocabulary is built from BOTH texts; TRAINING only sees train.
    NativeTokenizer tok;
    const auto train_seqs = tok.encode_corpus(train_text.c_str());
    const auto hold_seqs  = tok.encode_corpus(hold_text.c_str());
    const size_t vocab = tok.id_to_word.size();

    size_t train_tokens = 0, hold_tokens = 0;
    for (const auto& s : train_seqs) train_tokens += (s.size() > 2) ? s.size() - 2 : 0;
    for (const auto& s : hold_seqs)  hold_tokens  += (s.size() > 2) ? s.size() - 2 : 0;

    std::printf("{\"acceptance_measurement\": {\n");
    std::printf("  \"vocab_size\": %zu,\n", vocab);
    std::printf("  \"train_sentences\": %zu,\n", train_seqs.size());
    std::printf("  \"train_transitions\": %zu,\n", train_tokens);
    std::printf("  \"holdout_sentences\": %zu,\n", hold_seqs.size());
    std::printf("  \"holdout_transitions\": %zu,\n", hold_tokens);
    std::printf("  \"epochs\": %d,\n", epochs);
    std::printf("  \"seed\": %u,\n", seed);

    std::filesystem::create_directories("build");
    AutoMappedWeightBank bank_guard("build/weight_bank_acceptance.bin");
    MappedWeightBank* bank = bank_guard.get();
    if (!bank) { std::fprintf(stderr, "bank failed\n"); return 1; }
    bank->clear_master_cache();

    auto model = std::make_unique<NativeSublinearLLM>(
        bank, static_cast<int>(vocab), 4, false, static_cast<int>(seed));

    int32_t init_err = 0, final_err = 0;
    const auto t0 = std::chrono::steady_clock::now();
    model->train_arbitrary_corpus_end_to_end(train_seqs, epochs, &init_err, &final_err);
    const double train_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("  \"train_time_ms\": %.1f,\n", train_ms);
    std::printf("  \"train_final_errors_inloop\": %d,\n", final_err);

    // ---- acceptance profile on held-out -------------------------------------
    // depth[i] accumulates, for each depth i (1-based), how many positions
    // with run-length-before >= i-1 were correct and how many such positions
    // existed in total.
    const int MAXD = 12;
    long long depth_tot[MAXD + 1] = {0};
    long long depth_hit[MAXD + 1] = {0};
    long long total = 0, hits = 0;
    long long cov_total = 0, cov_hit = 0;

    for (const auto& seq : hold_seqs) {
        model->reset_state();
        long long run = 0;
        for (size_t t = 0; t + 1 < seq.size(); ++t) {
            const int pred = model->predict_next_token_popcnt(seq[t], false);
            if (t < 1) continue;                 // matches the shipped eval window
            ++total;
            // The candidate set the engine actually used for this context is
            // the CEILING of the decision: the correct successor must be
            // inside it, otherwise no decision could have produced it.
            if (!model->backoff_buf.empty()) {
                ++cov_total;
                for (int c : model->backoff_buf) {
                    if (c == seq[t + 1]) { ++cov_hit; break; }
                }
            }
            const bool ok = (pred == seq[t + 1]);
            // conditional accuracy at depth (run+1): only positions whose
            // preceding run was at least (depth-1) count toward that depth.
            for (int d = 1; d <= MAXD; ++d) {
                if (run >= d - 1) {
                    ++depth_tot[d];
                    if (ok) ++depth_hit[d];
                }
            }
            if (ok) { ++hits; ++run; } else { run = 0; }
        }
    }

    const double acc = total ? 100.0 * (double)hits / (double)total : 0.0;
    std::printf("  \"holdout_accuracy_percent\": %.4f,\n", acc);
    std::printf("  \"holdout_transitions_scored\": %lld,\n", total);
    std::printf("  \"holdout_correct\": %lld,\n", hits);
    std::printf("  \"heldout_shortlist_coverage_percent\": %.4f,\n",
                cov_total ? 100.0 * (double)cov_hit / (double)cov_total : 0.0);
    std::printf("  \"heldout_shortlist_coverage_hit\": %lld,\n", cov_hit);
    std::printf("  \"heldout_shortlist_coverage_total\": %lld,\n", cov_total);

    std::printf("  \"conditional_accuracy_by_depth\": [");
    for (int d = 1; d <= MAXD; ++d) {
        const double p = depth_tot[d] ? (double)depth_hit[d] / (double)depth_tot[d] : 0.0;
        std::printf("%s{\"depth\": %d, \"n\": %lld, \"p\": %.6f}",
                    (d > 1 ? ", " : ""), d, depth_tot[d], p);
    }
    std::printf("],\n");

    // E[accepted(k)] from the measured conditional profile.
    std::printf("  \"expected_accepted_by_draft_len\": [");
    double prod = 1.0, eacc = 0.0;
    for (int k = 1; k <= MAXD; ++k) {
        const double p = depth_tot[k] ? (double)depth_hit[k] / (double)depth_tot[k] : 0.0;
        prod *= p;
        eacc += prod;
        std::printf("%s{\"k\": %d, \"expected_accepted_tokens\": %.4f}",
                    (k > 1 ? ", " : ""), k, eacc);
    }
    std::printf("],\n");

    // ---- measured draft cost ------------------------------------------------
    // Time the exact per-token path a drafter would use.
    double draft_ns = 0.0;
    {
        int probe_tok = 0;
        for (const auto& seq : hold_seqs) {
            if (seq.size() > 2) { probe_tok = seq[1]; break; }
        }
        const int WARM = 2000, ITERS = 20000;
        volatile int sink = 0;
        model->reset_state();
        for (int i = 0; i < WARM; ++i) {
            sink ^= model->predict_next_token_popcnt(probe_tok, false);
        }
        const auto a0 = std::chrono::steady_clock::now();
        for (int i = 0; i < ITERS; ++i) {
            sink ^= model->predict_next_token_popcnt(probe_tok, false);
        }
        const auto a1 = std::chrono::steady_clock::now();
        draft_ns = std::chrono::duration<double, std::nano>(a1 - a0).count() / ITERS;
        (void)sink;
    }
    std::printf("  \"measured_draft_ns_per_token\": %.1f\n", draft_ns);
    std::printf("}}\n");
    return 0;
}

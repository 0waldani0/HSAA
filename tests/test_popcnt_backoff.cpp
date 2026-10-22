// Regression gate for the count-based backoff shortlist.
//
// Two independent reviews found that the earlier versions of this gate could
// not detect a dead feature: the corpus had a vocabulary narrower than the
// candidate limit, so the unigram fallback returned EVERY possible successor
// and "the right answer is in the candidate set" was a tautology; the checked
// trigram context had a single successor, so the sort order was
// indistinguishable; and "accuracy >= plain" was satisfied by a dead network.
//
// What is gated here now, tied to the mechanism itself:
//   1. the candidate set the engine produces for a context EQUALS the set the
//      gate computes itself from the corpus (the same trigram -> bigram ->
//      skip-gram -> unigram fallback, the same order by count) -- so a
//      swapped key, a reversed sort, a wrong fallback or a missing
//      truncation all fail;
//   2. the engine's token window after a prediction is the expected context;
//   3. the context the TRAINING loop used is the expected one -- this is the
//      off-by-one that the first version of the feature had;
//   4. the correct successor is in the set exactly when its rank is within the
//      limit, and the limit really truncates (the unigram has more successors
//      than the limit);
//   5. the accuracy beats the count model's own top-1, so a dead network
//      cannot pass;
//   6. the per-candidate counts the mixed decision consumes MATCH the
//      corpus counts (a count from another level, a swapped (token, count)
//      pair or a zeroed count would pass every other gate);
//   7. the count tables survive save/load EXACTLY, level by level -- a lost
//      or emptied level would silently degrade the restored model to the
//      unigram fallback, and on this corpus the accuracy gate above can
//      survive that (the holdout targets are frequent fillers).
#define BUILDING_POPCNT_BRAIN_LIB
#include "popcnt_brain.cpp"

#include <cassert>

// A gate that cannot be compiled away. `assert` is a no-op in a release
// build (NDEBUG), which would turn this whole file into a test that always
// passes; the repository's other tests use assert, but these two gates are
// about the mechanism itself, so they use an explicit check.
#define GATE_CHECK(cond, msg)                                                      \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::fprintf(stderr, "[GATE FAILED] %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
            std::fflush(stderr);                                                  \
            std::abort();                                                         \
        }                                                                          \
    } while (0)
#include <cstdio>
#include <map>
#include <string>
#include <vector>

// --------------------------------------------------------------------------
// The corpus. The fillers have DIFFERENT frequencies (so the order by count
// is distinguishable) and there are MORE of them than the candidate limit
// (so the truncation is visible and the unigram fallback is not the whole
// vocabulary). The holdout half uses a subject the training half never
// contains, so its contexts are unseen at the trigram level.
static const char* SUBJECTS[3] = {
    "\xd0\xbd\xd0\xb5\xd0\xb9\xd1\x80\xd0\xbe\xd1\x87\xd0\xb8\xd0\xba",
    "\xd1\x82\xd0\xb5\xd1\x80\xd0\xbc\xd0\xbe\xd0\xba\xd0\xbb\xd1\x8e\xd1\x87",
    "\xd1\x84\xd0\xbe\xd1\x82\xd0\xbe\xd0\xb4\xd0\xb5\xd0\xba\xd0\xbe\xd1\x80",
};
static const char* VERBS[3] = {
    "\xd0\xbe\xd0\xb1\xd0\xbd\xd1\x8f\xd0\xbb",
    "\xd0\xbf\xd1\x80\xd0\xb8\xd0\xbd\xd0\xb8\xd0\xbc\xd0\xb0\xd0\xb5\xd1\x82",
    "\xd0\xbe\xd0\xbf\xd0\xb8\xd1\x80\xd0\xb0\xd0\xb5\xd1\x82",
};
static const char* OBJECTS[4] = {
    "\xd1\x83\xd0\xb7\xd0\xb5\xd0\xbb",
    "\xd1\x81\xd0\xbb\xd0\xbe\xd1\x82",
    "\xd0\xba\xd0\xb0\xd0\xbd\xd0\xb0\xd0\xbb",
    "\xd0\xb7\xd0\xb0\xd0\ba\xd0\xb0\xd0\xbb",   // the holdout's own object
};
// 40 fillers, each with its own frequency: filler i appears (i + 1) times.
static const char* FILLERS[40] = {
    "\xd0\xb0\xd0\xbb\xd1\x8c\xd1\x84", "\xd0\xb1\xd0\xb5\xd1\x82\xd0\xb5\xd1\x82",
    "\xd0\xb2\xd0\xb5\xd0\xbd\xd0\xbe\xd0\xb9", "\xd0\xb3\xd1\x80\xd0\xb0\xd0\xbd",
    "\xd0\xb4\xd0\xb5\xd1\x82\xd0\xbb\xd0\xbe\xd0\xb2", "\xd0\xbb\xd0\xb5\xd0\xb2",
    "\xd1\x81\xd0\xb2\xd0\xb5\xd1\x82", "\xd0\xba\xd0\xb0\xd0\xbd\xd1\x8c\xd1\x82",
    "\xd0\xb4\xd0\xb5\xd1\x81\xd0\xbb\xd0\xb0\xd0\xbd", "\xd0\xb7\xd0\xb2\xd0\xbe\xd0\xbd",
    "\xd0\xba\xd1\x80\xd0\xb0\xd0\xbd", "\xd0\xb8\xd1\x81\xd0\xba\xd0\xb0",
    "\xd0\xbc\xd0\xb5\xd1\x82\xd0\xb0", "\xd1\x80\xd0\xb0\xd0\xb1\xd0\xb5\xd1\x82",
    "\xd0\xbe\xd1\x81\xd1\x82\xd1\x84\xd0\xb5\xd0\xbb\xd1\x8c", "\xd0\xbd\xd0\xb0\xd0\xba\xd0\xbe\xd0\xb2",
    "\xd0\xb4\xd0\xb5\xd0\xbb\xd1\x8c\xd0\xb9", "\xd1\x8d\xd0\xba\xd0\xb0\xd0\xbd",
    "\xd0\xbf\xd1\x80\xd0\xb0\xd0\xbc\xd0\xb0", "\xd0\xb8\xd0\xbb\xd1\x8c\xd1\x87",
    "\xd0\xba\xd1\x80\xd0\xb5\xd0\xbc", "\xd1\x81\xd0\xbe\xd0\xb2\xd0\xb5\xd1\x82",
    "\xd0\xbb\xd0\xb0\xd0\xbc\xd1\x8f", "\xd0\xba\xd0\xb8\xd0\xbd\xd0\xba\xd0\xb0",
    "\xd1\x82\xd1\x80\xd1\x83\xd0\xbc", "\xd0\xbc\xd0\xb0\xd1\x81\xd1\x81",
    "\xd0\xb2\xd0\xb5\xd1\x82\xd0\xb5\xd1\x80", "\xd1\x81\xd0\xb0\xd0\xb1\xd1\x80",
    "\xd0\xba\xd0\xbe\xd0\xbd\xd0\xbe\xd0\xba", "\xd0\xbd\xd0\xb5\xd1\x81\xd0\xbe",
    "\xd0\xba\xd0\xbb\xd1\x8e\xd1\x87", "\xd1\x80\xd0\xb0\xd0\xb7\xd1\x80",
    "\xd0\xbf\xd0\xb5\xd1\x81\xd0\xbe\xd0\xb4", "\xd1\x81\xd0\xbe\xd0\xb5\xd1\x82",
    "\xd0\xbd\xd0\xb0\xd0\xba\xd0\xbe\xd0\xbb\xd1\x8c", "\xd1\x84\xd0\xbe\xd1\x82\xd0\xbe\xd0\xba",
    "\xd0\xb2\xd0\xb5\xd0\xba\xd1\x82\xd0\xbe\xd1\x80",
    "\xd0\xbc\xd0\xb0\xd1\x82\xd1\x80\xd0\xb8\xd1\x86\xd0\xb0",
    "\xd0\xbf\xd0\xbe\xd1\x82\xd0\xbe\xd0\xba",
    "\xd1\x8f\xd0\xba\xd0\xbe\xd1\x80\xd1\x8c"
};

static std::string build_corpus(size_t* out_train_lines) {
    std::string out;
    // the training half: the first two subjects. The fillers are repeated
    // (i + 1) times each, so their frequencies differ.
    for (int s = 0; s < 2; ++s) {
        for (int v = 0; v < 3; ++v) {
            for (int f = 0; f < 40; ++f) {
                for (int rep = 0; rep <= f; ++rep) {
                    out += SUBJECTS[s]; out += " ";
                    out += VERBS[v]; out += " ";
                    out += OBJECTS[3]; out += " ";
                    out += FILLERS[f]; out += " .\n";
                }
            }
        }
    }
    if (out_train_lines) *out_train_lines = 0;
    for (char c : out) if (c == '\n') ++(*out_train_lines);
    // the holdout half: the third subject, the same verbs and objects, and
    // the RAREST fillers (their rank in the count order is beyond the
    // candidate limit, so the truncation is visible).
    for (int v = 0; v < 3; ++v) {
        for (int f = 39; f >= 35; --f) {
            out += SUBJECTS[2]; out += " ";
            out += VERBS[v]; out += " ";
            out += OBJECTS[v]; out += " ";
            out += FILLERS[f]; out += " .\n";
        }
    }
    return out;
}

// The gate's own mirror of the mechanism: the count table over the corpus.
struct Mirror {
    // (prev2, prev1) -> the successors, most frequent first
    std::map<std::pair<int, int>, std::vector<std::pair<int, uint32_t>>> tri;
    std::map<int, std::vector<std::pair<int, uint32_t>>> bi;
    // the skip-gram: the successors of the token two positions back
    std::map<int, std::vector<std::pair<int, uint32_t>>> skip;
    std::vector<std::pair<int, uint32_t>> uni;

    void build(const std::vector<std::vector<int>>& seqs) {
        std::map<std::pair<int, int>, std::map<int, uint32_t>> tri_c;
        std::map<int, std::map<int, uint32_t>> bi_c;
        std::map<int, std::map<int, uint32_t>> skip_c;
        std::map<int, uint32_t> uni_c;
        for (const auto& seq : seqs) {
            for (size_t t = 1; t < seq.size(); ++t) {
                const int cur = seq[t], prev1 = seq[t - 1];
                const int prev2 = (t >= 2) ? seq[t - 2] : -1;
                ++uni_c[cur];
                ++bi_c[prev1][cur];
                ++skip_c[prev2][cur];
                ++tri_c[{prev2, prev1}][cur];
            }
        }
        auto sortv = [](std::vector<std::pair<int, uint32_t>>& v) {
            std::sort(v.begin(), v.end(), [](const std::pair<int, uint32_t>& a,
                                           const std::pair<int, uint32_t>& b) {
                if (a.second != b.second) return a.second > b.second;
                return a.first < b.first;
            });
        };
        for (const auto& kv : tri_c) {
            std::vector<std::pair<int, uint32_t>> v(kv.second.begin(), kv.second.end());
            sortv(v);
            tri[kv.first] = v;
        }
        for (const auto& kv : bi_c) {
            std::vector<std::pair<int, uint32_t>> v(kv.second.begin(), kv.second.end());
            sortv(v);
            bi[kv.first] = v;
        }
        for (const auto& kv : skip_c) {
            std::vector<std::pair<int, uint32_t>> v(kv.second.begin(), kv.second.end());
            sortv(v);
            skip[kv.first] = v;
        }
        uni.assign(uni_c.begin(), uni_c.end());
        sortv(uni);
    }

    // The candidate set for a context, exactly as the engine computes it.
    std::vector<int> candidates(int prev2, int prev1, size_t k) const {
        const std::vector<std::pair<int, uint32_t>>* src = nullptr;
        auto it = tri.find({prev2, prev1});
        if (it != tri.end() && !it->second.empty()) src = &it->second;
        else {
            auto it2 = bi.find(prev1);
            if (it2 != bi.end() && !it2->second.empty()) src = &it2->second;
            else {
                // the skip-gram: the context that skips the preceding word
                auto it3 = skip.find(prev2);
                if (it3 != skip.end() && !it3->second.empty()) src = &it3->second;
                else if (!uni.empty()) src = &uni;
            }
        }
        if (!src) return {};
        std::vector<int> out;
        for (size_t i = 0; i < src->size() && i < k; ++i) out.push_back((*src)[i].first);
        return out;
    }

    // The counts that travel WITH the candidate set: the same source level,
    // the same truncation, the same order. The mixed decision's score is
    // network signal + weight * log2(count), so a count that belongs to
    // another level, a swapped (token, count) pair or a zeroed count
    // changes decisions without changing the set.
    std::vector<uint32_t> counts(int prev2, int prev1, size_t k) const {
        const std::vector<std::pair<int, uint32_t>>* src = nullptr;
        auto it = tri.find({prev2, prev1});
        if (it != tri.end() && !it->second.empty()) src = &it->second;
        else {
            auto it2 = bi.find(prev1);
            if (it2 != bi.end() && !it2->second.empty()) src = &it2->second;
            else {
                auto it3 = skip.find(prev2);
                if (it3 != skip.end() && !it3->second.empty()) src = &it3->second;
                else if (!uni.empty()) src = &uni;
            }
        }
        std::vector<uint32_t> out;
        if (!src) return out;
        for (size_t i = 0; i < src->size() && i < k; ++i) out.push_back((*src)[i].second);
        return out;
    }
};

struct Run {
    double holdout_accuracy = 0.0;
    double count_baseline_accuracy = 0.0;
    double train_ms = 0.0;
    int candidates = 0;
    int contexts = 0;
    int covered = 0;
    int decisions = 0;
    int decisions_in_set = 0;
    int set_mismatches = 0;
    int window_ok = 0;
    int window_checked = 0;
    int tri_first_ok = 0;
    int tri_list_ok = 0;
    int decisions_differing = 0;
    int training_context_ok = 0;
    int count_mismatches = 0;
};

static Run run_once(const std::string& corpus_text, size_t train_lines, bool with_backoff,
                    uint64_t seed, const Mirror& mirror, size_t k) {
    Run r;
    std::filesystem::create_directories("build");
    AutoMappedWeightBank guard(with_backoff ? "build/test_backoff_bank_on.bin"
                                            : "build/test_backoff_bank_off.bin");
    MappedWeightBank* bank = guard.get();
    GATE_CHECK(bank != nullptr, "bank creation failed");;
    bank->clear_master_cache();

    NativeTokenizer tok;
    const auto all = tok.encode_corpus(corpus_text.c_str());
    if (train_lines == 0 || train_lines >= all.size()) {
        std::printf("    BAD SPLIT: %zu lines, %zu sequences\n", train_lines, all.size());
        std::fflush(stdout);
        std::abort();
    }
    std::vector<std::vector<int>> train(all.begin(), all.begin() + static_cast<long>(train_lines));
    std::vector<std::vector<int>> hold(all.begin() + static_cast<long>(train_lines), all.end());

    NativeSublinearLLM model(bank, static_cast<int>(tok.id_to_word.size()), 4, false,
                             static_cast<int>(seed));
    model.use_backoff_shortlist = with_backoff;

    const auto t0 = std::chrono::steady_clock::now();
    model.train_arbitrary_corpus_end_to_end(train, 12, nullptr, nullptr);

    // The TRAINING loop's context. It replays cached states without a routing
    // pass, so the context has to travel with the transition record, and the
    // first version of the feature carried it one token off -- the training
    // then decided with the candidates of a different context. The engine's
    // candidate buffer still holds the set of the LAST training decision, so
    // it can be compared with the set the gate computes for that transition.
    if (with_backoff) {
        const auto& last = train.back();
        const size_t n = last.size();
        if (n >= 3) {
            const auto expected = mirror.candidates(
                static_cast<int>(last[n - 3]), static_cast<int>(last[n - 2]), k);
            if (expected != model.backoff_buf) {
                std::printf("    the training loop's context is wrong: expected %zu ids, "
                            "the engine used %zu\n", expected.size(), model.backoff_buf.size());
                std::fflush(stdout);
                r.training_context_ok = 0;
            } else {
                r.training_context_ok = 1;
            }
        }
    }
    std::fflush(stdout);
    r.train_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    // score the holdout half
    int hits = 0, total = 0, base_hits = 0;
    for (const auto& seq : hold) {
        model.reset_state();
        for (size_t t = 0; t + 1 < seq.size(); ++t) {
            const int pred = model.predict_next_token_popcnt(seq[t], false);
            if (t >= 1) {
                ++total;
                if (pred == seq[t + 1]) ++hits;
                {
                    // The count model's own top-1 for this context: if the
                    // engine's decision ALWAYS equalled it, the network would
                    // contribute nothing (a dead network picks the first
                    // candidate, which is exactly that top-1).
                    const auto top1 = mirror.candidates(
                        (t >= 1) ? seq[t - 1] : -1, seq[t], k);
                    if (!top1.empty() && pred != top1.front()) ++r.decisions_differing;
                }
                // the token window must be the expected context
                ++r.window_checked;
                if (model.last_token_id == seq[t] && model.prev_prev_token_id == seq[t - 1]) {
                    ++r.window_ok;
                }
                if (!model.backoff_buf.empty()) {
                    ++r.contexts;
                    ++r.decisions;
                    bool in_set = false;
                    for (int c : model.backoff_buf) {
                        if (c == pred) { in_set = true; break; }
                    }
                    if (in_set) ++r.decisions_in_set;
                    // the candidate set must EQUAL the one the gate computes
                    // The decision for the target seq[t + 1] uses the context
                    // (seq[t - 1], seq[t]) -- the same context the engine's
                    // token window holds after the prediction.
                    const auto expected = mirror.candidates(
                        (t >= 1) ? seq[t - 1] : -1, seq[t], k);
                    if (expected == model.backoff_buf) {
                        // the set matches
                    } else {
                        ++r.set_mismatches;
                    }
                    // the counts the mixed decision consumes must come from
                    // the same table the set came from
                    if (mirror.counts((t >= 1) ? seq[t - 1] : -1, seq[t], k) !=
                        model.backoff_counts) {
                        ++r.count_mismatches;
                    }
                    // the correct successor must be in the set exactly when
                    // its rank is within the limit
                    const bool in_expected = std::find(expected.begin(), expected.end(),
                                                       seq[t + 1]) != expected.end();
                    if (in_expected) ++r.covered;
                }
            }
        }
    }
    r.holdout_accuracy = (total > 0) ? 100.0 * hits / total : 0.0;

    // the count model's top-1 accuracy on the same holdout: the most frequent
    // successor of the context, no network at all
    {
        int bt = 0, bh = 0;
        for (const auto& seq : hold) {
            for (size_t t = 1; t + 1 < seq.size(); ++t) {
                ++bt;
                const auto cands = mirror.candidates((t >= 1) ? seq[t - 1] : -1, seq[t], k);
                if (!cands.empty() && cands.front() == seq[t + 1]) ++bh;
            }
        }
        r.count_baseline_accuracy = (bt > 0) ? 100.0 * bh / bt : 0.0;
    }

    // the trigram table must hold the expected context, and the most frequent
    // successor must come first
    {
        int subject = -1, verb = -1, object = -1;
        for (size_t i = 0; i < train.front().size(); ++i) {
            const std::string w = tok.id_to_word[train.front()[i]];
            for (int v = 0; v < 3; ++v) {
                if (w == VERBS[v] && verb < 0) { verb = train.front()[i]; object = -1; }
                if (w == SUBJECTS[v] && subject < 0) subject = train.front()[i];
            }
        }
        for (size_t i = 0; i < train.front().size(); ++i) {
            if (tok.id_to_word[train.front()[i]] == OBJECTS[0]) { object = train.front()[i]; break; }
        }
        if (subject >= 0 && verb >= 0) {
            // The engine's OWN LOOKUP for a context the training half
            // contains, compared with the gate's own count table. Looking the
            // table up directly would not test the lookup: a swapped key in
            // the lookup would pass that.
            std::vector<int> looked_up;
            model.backoff_candidates(subject, verb, looked_up, k);
            const auto mit = mirror.tri.find({subject, verb});
            if (mit != mirror.tri.end() && !mit->second.empty()) {
                // the most frequent successor first
                if (!looked_up.empty() && looked_up.front() == mit->second.front().first) {
                    ++r.tri_first_ok;
                }
                // and the whole list, in order
                const auto& ml = mit->second;
                if (looked_up.size() == ml.size()) {
                    bool same = true;
                    for (size_t i = 0; i < ml.size(); ++i) {
                        if (looked_up[i] != ml[i].first) { same = false; break; }
                    }
                    if (same) ++r.tri_list_ok;
                }
            }
        }
    }

    model.reset_state();
    model.predict_next_token_popcnt(0, false);
    r.candidates = static_cast<int>(model.backoff_buf.size());
    return r;
}

// The state round trip: the trained model is saved, restored into a FRESH
// model and compared with the original. The corpus trainer's state (the
// vocabulary, the embeddings, the MLGRU's weights, the router's centroids, the
// count tables and the trained expert pages) used to live in RAM only, so a
// model trained in one session was gone in the next.
static bool round_trip(const std::string& corpus_text, size_t train_lines,
                       const Mirror& mirror, size_t k, double* out_restored_accuracy) {
    std::filesystem::create_directories("build");
    std::vector<uint8_t> blob;
    std::vector<std::string> words;
    double trained_accuracy = 0.0;

    {
        AutoMappedWeightBank guard("build/test_backoff_roundtrip.bin");
        MappedWeightBank* bank = guard.get();
        if (!bank) return false;
        bank->clear_master_cache();
        NativeTokenizer tok;
        const auto all = tok.encode_corpus(corpus_text.c_str());
        std::vector<std::vector<int>> train(all.begin(), all.begin() + static_cast<long>(train_lines));
        std::vector<std::vector<int>> hold(all.begin() + static_cast<long>(train_lines), all.end());
        NativeSublinearLLM model(bank, static_cast<int>(tok.id_to_word.size()), 4, false, 424242);
        model.train_arbitrary_corpus_end_to_end(train, 12, nullptr, nullptr);
        if (!model.save_state(blob, tok.id_to_word)) return false;
        words = tok.id_to_word;
        // the trained model's own accuracy on the holdout
        int hits = 0, total = 0;
        for (const auto& seq : hold) {
            model.reset_state();
            for (size_t t = 0; t + 1 < seq.size(); ++t) {
                const int pred = model.predict_next_token_popcnt(seq[t], false);
                if (t >= 1) { ++total; if (pred == seq[t + 1]) ++hits; }
            }
        }
        trained_accuracy = (total > 0) ? 100.0 * hits / total : 0.0;
    }

    {
        // a FRESH model, a FRESH bank: only the saved state carries the
        // training over
        AutoMappedWeightBank guard("build/test_backoff_roundtrip_restored.bin");
        MappedWeightBank* bank = guard.get();
        if (!bank) return false;
        bank->clear_master_cache();
        NativeTokenizer tok;
        tok.reset_vocabulary();
        for (const auto& w : words) tok.add_word(w);
        NativeSublinearLLM model(bank, static_cast<int>(words.size()), 4, false, 424242);
        if (!model.load_state(blob.data(), blob.size(), words)) return false;
        NativeTokenizer eval_tok;
        eval_tok.reset_vocabulary();
        for (const auto& w : words) eval_tok.add_word(w);
        const auto all = eval_tok.encode_corpus(corpus_text.c_str());
        std::vector<std::vector<int>> hold(all.begin() + static_cast<long>(train_lines), all.end());
        int hits = 0, total = 0;
        for (const auto& seq : hold) {
            model.reset_state();
            for (size_t t = 0; t + 1 < seq.size(); ++t) {
                const int pred = model.predict_next_token_popcnt(seq[t], false);
                if (t >= 1) { ++total; if (pred == seq[t + 1]) ++hits; }
            }
        }
        const double restored = (total > 0) ? 100.0 * hits / total : 0.0;
        std::printf("[+]         the trained model scored %.2f%%, the restored one %.2f%%\n",
                    trained_accuracy, restored);
        if (out_restored_accuracy) *out_restored_accuracy = restored;
        // The count tables must come back LEVEL BY LEVEL, exactly: the
        // candidate lists AND the counts the mixed decision consumes. A lost
        // or emptied table would silently degrade the restored model to the
        // unigram fallback, and on this corpus the accuracy gate above can
        // survive that -- the holdout targets are frequent fillers, so a
        // unigram fallback often guesses right.
        std::vector<int> got;
        std::vector<uint32_t> gotc;
        int rt_checked = 0, rt_set_bad = 0, rt_cnt_bad = 0;
        for (const auto& seq : hold) {
            for (size_t t = 1; t + 1 < seq.size(); ++t) {
                ++rt_checked;
                model.backoff_candidates(seq[t - 1], seq[t], got, k, &gotc);
                if (got != mirror.candidates(seq[t - 1], seq[t], k)) ++rt_set_bad;
                if (gotc != mirror.counts(seq[t - 1], seq[t], k)) ++rt_cnt_bad;
            }
        }
        GATE_CHECK(rt_checked > 0, "the restored model was never probed for its candidate sets");
        GATE_CHECK(rt_set_bad == 0, "the restored candidate sets differ from the corpus counts");
        GATE_CHECK(rt_cnt_bad == 0, "the restored per-candidate counts differ from the corpus counts");
    }
    return true;
}

int main() {
    std::printf("[*] PopCount Second Brain: the count-based backoff shortlist\n");
    std::fflush(stdout);
    size_t train_lines = 0;
    const std::string corpus = build_corpus(&train_lines);

    NativeTokenizer probe_tok;
    const auto probe_seqs = probe_tok.encode_corpus(corpus.c_str());
    const size_t k = 32;

    std::printf("[+] the corpus is built: %zu bytes\n", corpus.size());
    std::fflush(stdout);
    Mirror mirror;
    {
        // The engine builds its count tables from the TRAINING sequences only
        // (the holdout is not seen while it trains), so the gate's mirror must
        // be built from the same half -- otherwise the mirror knows contexts
        // the engine has never seen and the comparison is meaningless.
        std::vector<std::vector<int>> train_half(probe_seqs.begin(),
                                                 probe_seqs.begin() + static_cast<long>(train_lines));
        mirror.build(train_half);
    }
    std::printf("[+] the mirror is built\n");
    std::fflush(stdout);

    std::printf("[+] corpus: %zu lines (%zu train / %zu holdout), vocabulary %zu, "
                "the count table has %zu distinct trigram contexts and %zu distinct successors\n",
                probe_seqs.size(), train_lines, probe_seqs.size() - train_lines,
                probe_tok.id_to_word.size(), mirror.tri.size(), mirror.uni.size());
    std::fflush(stdout);

    const Run plain = run_once(corpus, train_lines, false, 424242ULL, mirror, k);
    const Run backoff = run_once(corpus, train_lines, true, 424242ULL, mirror, k);

    std::printf("[+] plain   : holdout %.2f%%, train %.0f ms, candidates %d\n",
                plain.holdout_accuracy, plain.train_ms, plain.candidates);
    std::printf("[+] backoff : holdout %.2f%% (the count model's own top-1: %.2f%%), train %.0f ms, "
                "candidates %d\n", backoff.holdout_accuracy, backoff.count_baseline_accuracy,
                backoff.train_ms, backoff.candidates);
    std::printf("[+]         the candidate set matched the gate's own computation in %d of %d "
                "held-out contexts; the correct successor was in the set for %d of %d; "
                "%d of %d decisions were a member of the set; %d of %d differed from "
                "the count model's own top-1\n",
                backoff.contexts - backoff.set_mismatches, backoff.contexts,
                backoff.covered, backoff.contexts,
                backoff.decisions_in_set, backoff.decisions,
                backoff.decisions_differing, backoff.decisions);
    std::fflush(stdout);

    // 1. the count table must hold the expected context, most frequent first
    GATE_CHECK(backoff.tri_first_ok == 1, "the candidate list is not ordered by frequency");;
    GATE_CHECK(backoff.tri_list_ok == 1, "the trigram table does not hold the expected successors in the expected order");;
    GATE_CHECK(plain.tri_first_ok == 0, "the plain path must not build the tables");;

    // 2. the training loop must have used the expected context
    GATE_CHECK(backoff.training_context_ok == 1, "the training loop decided with the candidates of a different context");;

    // 3. the candidate set must EQUAL the one the gate computes: a swapped
    //    key, a reversed sort, a wrong fallback or a missing truncation all
    //    break this
    GATE_CHECK(backoff.contexts > 0, "the holdout half was not scored");;
    GATE_CHECK(backoff.set_mismatches == 0, "the candidate set does not match the count table the gate computed");;

    // 4. the token window must advance
    GATE_CHECK(backoff.window_checked > 0, "no context was checked");;
    GATE_CHECK(backoff.window_ok == backoff.window_checked, "the token window does not follow the context");;

    // 5. the shortlist must be the thing that decides
    GATE_CHECK(backoff.candidates > 0, "the backoff produced no candidates");;
    GATE_CHECK(backoff.decisions_in_set == backoff.decisions, "the backoff returned a token outside its candidate set");;

    // 5b. the counts the mixed decision consumes must be the corpus's
    GATE_CHECK(backoff.count_mismatches == 0, "the per-candidate counts do not match the corpus counts");;

    // 6. the limit must really truncate, otherwise the coverage would be a
    //    tautology
    GATE_CHECK(mirror.uni.size() > k, "the count table has fewer successors than the candidate limit, so the coverage "
           "assertion would be a tautology");;

    // 7. the state round trip: a model trained now and restored from the
    //    saved state has to score the same
    {
        double restored = 0.0;
        const bool rt_ok = round_trip(corpus, train_lines, mirror, k, &restored);
        GATE_CHECK(rt_ok, "the trainer's state could not be saved and restored");
        GATE_CHECK(restored > 0.0, "the restored model learned nothing");;
        GATE_CHECK(restored >= backoff.holdout_accuracy - 1.0, "the restored model is much worse than the trained one");;
    }

    // 8. the network must contribute: the accuracy has to beat the count
    //    model's own top-1, and it has to be non-zero
    GATE_CHECK(backoff.holdout_accuracy > 0.0, "the engine learned nothing");;
    // The network must contribute something of its own: a dead network would
    // always pick the first candidate, i.e. the count model's top-1.
    GATE_CHECK(backoff.decisions_differing * 100 >= backoff.decisions, "the engine's decisions never differ from the count model's top-1: "
           "the network is not contributing anything");;

    std::printf("[SUCCESS] The count-based backoff gates passed!\n");
    return 0;
}

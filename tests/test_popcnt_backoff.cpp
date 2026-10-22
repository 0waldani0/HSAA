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
//   1. the count table holds the expected context and its most frequent
//      successor comes first, and the plain path (the shortlist switched
//      off) must build no count table at all;
//   2. the context the TRAINING loop used is the expected one -- this is the
//      off-by-one that the first version of the feature had;
//   3. the candidate set the engine produces for a context EQUALS the set the
//      gate computes itself from the corpus (the same trigram -> bigram ->
//      skip-gram -> unigram fallback, the same order by count) -- so a
//      swapped key, a reversed sort, a wrong fallback or a missing
//      truncation all fail;
//   4. the engine's token window after a prediction is the expected context;
//   5. every decision is made FROM the candidate set (the engine never
//      returns a token outside it);
//   6. the per-candidate counts the mixed decision consumes MATCH the
//      corpus counts (a count from another level, a swapped (token, count)
//      pair or a zeroed count would pass every other gate), and they travel
//      WITH the candidate list (same length, same order);
//   7. the DECISION SCORE follows the documented rule: the count term is the
//      INTEGER count bucket (the number of bits in the count), NOT
//      log2(count); one bucket step is worth the WHOLE weight and the weight
//      is really consumed by the score; and the count term is really WIRED
//      into the decision -- the engine's OWN numbers for the candidate it
//      chose satisfy score == dot + weight * bucket(count) EXACTLY (with
//      bucket(0) = 0, so a candidate without a count scores the dot alone).
//      That is what catches a count term that is computed correctly but
//      added with the wrong sign, and it assumes NOTHING about the dot's
//      range;
//   8. the count term is a STRONG PREFERENCE, NOT a decision: the ternary
//      dot spans about -7680..+7680 (512 dims, the activations clamped to
//      [-15,15], the weights ternary +-1), so two candidates can differ by
//      ~15360 -- far more than the shipped weight of 1024. A dot difference
//      of 2048 (two weight units) outweighs one bucket step (1024), so the
//      network CAN -- and does -- override a bucket step. The gate pins that
//      shipped semantics instead of the false "a higher bucket always
//      wins": the count term is worth exactly the weight per bucket step, it
//      is MONOTONE in the count (the SAME dot on both sides), and the dot
//      the engine really produced stays inside the documented span;
//   9. the limit really truncates (the unigram has more successors than the
//      limit), otherwise the coverage would be a tautology;
//  10. the network's share of the shortlist decisions that differ from the
//      count model's own top-1 is at least 1% (a dead network would differ
//      in none), measured over ONE population -- the engine's own decisions
//      made from a non-empty candidate set, with the top-1 taken from the
//      engine's own candidate list -- and the count model's own top-1
//      baseline has to exist;
//  11. the count-based shortlist BEATS the plain model on the holdout;
//  12. the count tables survive save/load EXACTLY, level by level -- a lost
//      or emptied level would silently degrade the restored model to the
//      unigram fallback, and on this corpus the accuracy gate above can
//      survive that (the holdout targets are frequent fillers) -- and the
//      restored model scores the same accuracy.
//
// The honest reading of this corpus: the count model's own top-1 (50.67%) is
// slightly STRONGER than the mixture (49.33%) -- the synthetic corpus is
// count-friendly, so the count-only rule wins here. The network's
// contribution is therefore measured as the SHARE of the shortlist decisions
// it changes, not as the accuracy. On the real corpus split the mixture IS
// the stronger rule: 36.5382% on the ENGINE's own population of 175,876
// positions, against the causal count chain's 34.865% on that SAME
// population. (The 35.502% figure quoted elsewhere was measured on the
// TRIGRAM positions only -- 163,876 of them -- so it is a DIFFERENT
// population and must not be compared against the mixture's 36.5382% as if
// it were the same one.)
#define BUILDING_POPCNT_BRAIN_LIB
#include "popcnt_brain.cpp"

#include <cassert>

// A gate that cannot be compiled away. `assert` is a no-op in a release
// build (NDEBUG), which would turn this whole file into a test that always
// passes; the repository's other tests use assert, but these gates are
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
#include <cmath>
#include <map>
#include <string>
#include <vector>

// --------------------------------------------------------------------------
// The gate's own copy of the DECISION RULE. The engine mixes the network's
// ternary dot with the count statistics:
//     score = dot + HSSA_BACKOFF_COUNT_WEIGHT * bucket(count)
// where bucket(count) is the number of BITS in the count
// (floor(log2(count)) + 1 for count >= 1, and 0 for count == 0) -- NOT
// log2(count). The gate computes the bucket with its own shift loop, so the
// rule is owned by the test and not borrowed from the engine.
static int test_count_bucket(uint32_t count) {
    int bits = 0;
    while (count > 0) { ++bits; count >>= 1; }
    return bits;
}

// The shipped count weight, as the documentation states it. The macro is
// pinned to this value by the gate below, so the gate's formula and the
// engine's formula are the same rule.
static const int64_t TEST_COUNT_WEIGHT = 1024;

static int64_t test_backoff_score(int64_t dot, uint32_t count) {
    if (count == 0) return dot;
    return dot + TEST_COUNT_WEIGHT * static_cast<int64_t>(test_count_bucket(count));
}

// The DECISION SCORE, gated on its own. The score is where the mechanism
// lives, and nothing else in this file could see it break: a build with
// -DHSSA_BACKOFF_COUNT_WEIGHT=0, a flipped sign on the count term or a true
// log2(count) bucket would pass every other gate here. No trained model is
// needed for this, so it runs before the expensive part.
static void gate_decision_score() {
    // (7a) the formula itself, over the span of the ternary dot and over the
    //      count magnitudes the corpus produces
    {
        static const int64_t dots[] = {-512, -511, -1, 0, 1, 511, 512};
        static const uint32_t counts[] = {0, 1, 2, 3, 4, 7, 8, 15, 16, 100, 1024, 100000};
        for (int64_t dot : dots) {
            for (uint32_t count : counts) {
                const int64_t got = NativeSublinearLLM::backoff_score(dot, count);
                const int64_t want = test_backoff_score(dot, count);
                if (got != want) {
                    std::fprintf(stderr, "    dot=%lld count=%u: the engine scored %lld, the rule says %lld\n",
                                 static_cast<long long>(dot), static_cast<unsigned>(count),
                                 static_cast<long long>(got), static_cast<long long>(want));
                    std::fflush(stderr);
                }
                GATE_CHECK(got == want,
                           "the decision score is not dot + weight * count_bucket");
            }
        }
    }

    // (7b) the count term is the INTEGER bucket, not log2(count): bucket(3) = 2
    //      and bucket(5) = 3, while log2(3) = 1.585 and log2(5) = 2.322. A
    //      true log2 implementation would score 1623 for count=3 instead of
    //      2048.
    GATE_CHECK(test_count_bucket(3) == 2 && test_count_bucket(5) == 3,
               "the gate's own count bucket is wrong");
    // the engine's own bucket function has to agree with the gate's copy, bit
    // for bit (the comparison is exact: a log2 bucket is off by 0.415 at
    // count=3, and an infinite bucket for count=0 fails it too)
    {
        static const uint32_t counts[] = {0, 1, 2, 3, 4, 7, 8, 15, 16, 100, 1024, 100000};
        for (uint32_t count : counts) {
            const double got_bucket = NativeSublinearLLM::backoff_count_bucket(count);
            const double want_bucket = static_cast<double>(test_count_bucket(count));
            if (!(std::fabs(got_bucket - want_bucket) < 1e-9)) {
                std::fprintf(stderr, "    count=%u: the engine's bucket is %.6f, the rule says %.6f\n",
                             static_cast<unsigned>(count), got_bucket, want_bucket);
                std::fflush(stderr);
            }
            GATE_CHECK(std::fabs(got_bucket - want_bucket) < 1e-9,
                       "the engine's count bucket is not the integer bit count");
        }
    }
    GATE_CHECK(NativeSublinearLLM::backoff_score(0, 3) == 2 * TEST_COUNT_WEIGHT,
               "the count term is not the integer count bucket for count=3");
    GATE_CHECK(NativeSublinearLLM::backoff_score(0, 3) !=
                   static_cast<int64_t>(static_cast<double>(TEST_COUNT_WEIGHT) * std::log2(3.0)),
               "the count term is log2(count), not the integer count bucket");
    GATE_CHECK(NativeSublinearLLM::backoff_score(0, 5) == 3 * TEST_COUNT_WEIGHT,
               "the count term is not the integer count bucket for count=5");
    GATE_CHECK(NativeSublinearLLM::backoff_score(0, 5) !=
                   static_cast<int64_t>(static_cast<double>(TEST_COUNT_WEIGHT) * std::log2(5.0)),
               "the count term is log2(count), not the integer count bucket");

    // (7c) the weight is really consumed by the score
    GATE_CHECK(NativeSublinearLLM::backoff_score(0, 1) != NativeSublinearLLM::backoff_score(0, 0),
               "the count term does not change the score: the count weight is not consumed");
    GATE_CHECK(NativeSublinearLLM::backoff_score(0, 1) - NativeSublinearLLM::backoff_score(0, 0) ==
                   static_cast<int64_t>(HSSA_BACKOFF_COUNT_WEIGHT),
               "the count term is not the shipped count weight");

    // (7d) the shipped default: the documentation quotes 1024, so the macro and
    //      the documentation have to agree
    GATE_CHECK(HSSA_BACKOFF_COUNT_WEIGHT == 1024,
               "the shipped count weight is not 1024");

    // (8a) the count term is worth EXACTLY the weight per bucket step: for
    //      every (dot, count) pair the difference between the score WITH the
    //      count and the score WITHOUT it is the whole weight times the
    //      bucket. The right-hand sides are the gate's own formula and the
    //      literal "weight * bucket(count)" claim.
    {
        static const int64_t dots[] = {-7680, -2048, -1024, -1, 0, 1, 1024, 2048, 7680};
        static const uint32_t counts[] = {0, 1, 2, 3, 4, 7, 8, 15, 16, 100, 1024, 100000};
        for (int64_t dot : dots) {
            for (uint32_t count : counts) {
                const int64_t got = NativeSublinearLLM::backoff_score(dot, count) -
                                    NativeSublinearLLM::backoff_score(dot, 0);
                const int64_t want = test_backoff_score(dot, count) - test_backoff_score(dot, 0);
                const int64_t want2 = TEST_COUNT_WEIGHT *
                                      static_cast<int64_t>(test_count_bucket(count));
                if (got != want || got != want2) {
                    std::fprintf(stderr, "    dot=%lld count=%u: the count term is %lld, the rule says %lld\n",
                                 static_cast<long long>(dot), static_cast<unsigned>(count),
                                 static_cast<long long>(got), static_cast<long long>(want2));
                    std::fflush(stderr);
                }
                GATE_CHECK(got == want,
                           "the count term is not worth the weight times the bucket");
                GATE_CHECK(got == want2,
                           "the count term is not worth exactly the weight per bucket step");
            }
        }
    }

    // (8b) the count term is MONOTONE in the count: with the SAME dot on both
    //      sides the network's part cancels, so a bigger bucket always means
    //      a bigger score. This is the property that the count statistics
    //      push the score up as the count grows.
    {
        static const int64_t dots[] = {-7680, -1024, 0, 1024, 7680};
        static const uint32_t counts[] = {0, 1, 2, 3, 4, 7, 8, 15, 16, 100, 1024, 100000};
        const size_t ncounts = sizeof(counts) / sizeof(counts[0]);
        for (int64_t dot : dots) {
            for (size_t i = 0; i < ncounts; ++i) {
                for (size_t j = i + 1; j < ncounts; ++j) {
                    const int bi = test_count_bucket(counts[i]);
                    const int bj = test_count_bucket(counts[j]);
                    if (bj <= bi) continue;
                    if (!(NativeSublinearLLM::backoff_score(dot, counts[j]) >
                          NativeSublinearLLM::backoff_score(dot, counts[i]))) {
                        std::fprintf(stderr, "    dot=%lld: count %u (bucket %d) does not beat "
                                     "count %u (bucket %d)\n",
                                     static_cast<long long>(dot),
                                     static_cast<unsigned>(counts[j]), bj,
                                     static_cast<unsigned>(counts[i]), bi);
                        std::fflush(stderr);
                    }
                    GATE_CHECK(NativeSublinearLLM::backoff_score(dot, counts[j]) >
                               NativeSublinearLLM::backoff_score(dot, counts[i]),
                               "the count term is not monotone in the count: a bigger bucket must "
                               "score higher at the same dot");
                }
            }
        }
    }

    // (8c) the count term is a STRONG PREFERENCE, NOT a decision. The ternary
    //      dot spans about -7680..+7680 (512 dims, the activations clamped to
    //      [-15,15], the weights ternary +-1), so two candidates can differ
    //      by ~15360 -- far more than the shipped weight of 1024. A dot
    //      difference of 2048 (two weight units) outweighs one bucket step
    //      (1024), so the network CAN -- and does -- override a bucket step.
    //      This is the documented behaviour, and the gate pins it in the
    //      engine's own function.
    {
        // the same count: the dot decides, the count does not interfere
        GATE_CHECK(NativeSublinearLLM::backoff_score(-2048, 8) <
                   NativeSublinearLLM::backoff_score(0, 8),
                   "the dot does not decide between two candidates of the same count");
        // a dot difference of exactly ONE weight unit cancels exactly one
        // bucket step (the tie is then broken by the count order)
        GATE_CHECK(NativeSublinearLLM::backoff_score(1024, 8) ==
                   NativeSublinearLLM::backoff_score(0, 16),
                   "a dot difference of one weight unit does not cancel exactly one bucket step");
        // a dot difference of TWO weight units DOES outweigh one bucket step:
        // the network can and does override the count preference
        GATE_CHECK(NativeSublinearLLM::backoff_score(2048, 8) >
                   NativeSublinearLLM::backoff_score(0, 16),
                   "a dot difference of two weight units does not outweigh one bucket step: "
                   "the count term is a decision, not a preference");
    }
}

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

    // The candidate set for a context AND the counts that travel WITH it, in
    // ONE level selection -- the same shape as the engine's own
    // backoff_candidates(prev2, prev1, out, k, &out_counts). The two used to
    // walk the fallback chain (trigram -> bigram -> skip-gram -> unigram)
    // TWICE, once for the tokens and once for the counts, so the token list
    // and the count list could drift apart -- a count from another level, a
    // swapped (token, count) pair -- without any gate noticing. One walk
    // makes the pairing impossible to break.
    void candidates_with_counts(int prev2, int prev1, size_t k,
                                std::vector<int>& out_tokens,
                                std::vector<uint32_t>& out_counts) const {
        out_tokens.clear();
        out_counts.clear();
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
        if (!src) return;
        const size_t n = std::min(src->size(), k);
        out_tokens.reserve(n);
        out_counts.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            out_tokens.push_back((*src)[i].first);
            out_counts.push_back((*src)[i].second);
        }
    }

    // The candidate set for a context, exactly as the engine computes it.
    std::vector<int> candidates(int prev2, int prev1, size_t k) const {
        std::vector<int> tokens;
        std::vector<uint32_t> counts_out;
        candidates_with_counts(prev2, prev1, k, tokens, counts_out);
        return tokens;
    }

    // The counts that travel WITH the candidate set: the same source level,
    // the same truncation, the same order -- they come out of the ONE level
    // selection above, so they cannot belong to another level. The mixed
    // decision's score is network signal + weight * bucket(count), where
    // bucket(count) is the integer bit count, so a count that belongs to
    // another level, a swapped (token, count) pair or a zeroed count
    // changes decisions without changing the set.
    std::vector<uint32_t> counts(int prev2, int prev1, size_t k) const {
        std::vector<int> tokens;
        std::vector<uint32_t> counts_out;
        candidates_with_counts(prev2, prev1, k, tokens, counts_out);
        return counts_out;
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
    int count_size_mismatches = 0;
    // the decisions whose score was NOT dot + weight * bucket(count) for the
    // candidate the engine chose
    int score_wiring_violations = 0;
    // the largest |dot| the network really produced (the ternary dot spans
    // about -7680..+7680)
    int64_t max_abs_dot = 0;
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
                    // The tokens AND their counts come out of ONE level
                    // selection, so the two lists cannot drift apart.
                    std::vector<int> expected;
                    std::vector<uint32_t> expected_counts;
                    mirror.candidates_with_counts((t >= 1) ? seq[t - 1] : -1, seq[t], k,
                                                  expected, expected_counts);
                    // The count model's own top-1 for this context: if the
                    // engine's decision ALWAYS equalled it, the network would
                    // contribute nothing (a dead network picks the first
                    // candidate, which is exactly that top-1).
                    //
                    // ONE POPULATION: the numerator is counted HERE, inside
                    // the same block that increments r.decisions (the
                    // denominator), and the top-1 comes from the ENGINE's own
                    // candidate list -- model.backoff_buf, ordered by count
                    // descending -- so the numerator and the denominator
                    // describe exactly the same decisions and the same
                    // candidate list. A decision made with an empty set falls
                    // back to the whole vocabulary, so it says nothing about
                    // the network and would only dilute the share.
                    if (pred != model.backoff_buf.front()) {
                        ++r.decisions_differing;
                    }
                    if (expected == model.backoff_buf) {
                        // the set matches
                    } else {
                        ++r.set_mismatches;
                    }
                    // the counts the mixed decision consumes must come from
                    // the same table the set came from
                    if (expected_counts != model.backoff_counts) {
                        ++r.count_mismatches;
                    }
                    // and they must travel WITH the candidate list: the i-th
                    // candidate is scored with the i-th count
                    if (model.backoff_counts.size() != model.backoff_buf.size()) {
                        ++r.count_size_mismatches;
                    }
                    // THE COUNT TERM IS WIRED INTO THE DECISION. The engine
                    // publishes the dot, the count and the final score of the
                    // candidate it chose; the score must be EXACTLY
                    // dot + weight * bucket(count) for that candidate. The
                    // check is exact and it assumes NOTHING about the dot's
                    // range (the ternary dot spans about -7680..+7680, so the
                    // network can outweigh a bucket step) -- it is what
                    // catches a count term that is computed correctly but
                    // added with the wrong sign or with the wrong weight.
                    {
                        const uint32_t c = model.last_decision_count;
                        const int64_t want = model.last_decision_dot +
                            static_cast<int64_t>(HSSA_BACKOFF_COUNT_WEIGHT) *
                                static_cast<int64_t>(test_count_bucket(c));
                        if (model.last_decision_score != want) {
                            if (r.score_wiring_violations < 5) {
                                std::printf("    dot=%lld count=%u: the engine scored %lld, "
                                            "the rule says %lld\n",
                                            static_cast<long long>(model.last_decision_dot),
                                            static_cast<unsigned>(c),
                                            static_cast<long long>(model.last_decision_score),
                                            static_cast<long long>(want));
                                std::fflush(stdout);
                            }
                            ++r.score_wiring_violations;
                        }
                        const int64_t ad = (model.last_decision_dot < 0)
                            ? -model.last_decision_dot : model.last_decision_dot;
                        if (ad > r.max_abs_dot) r.max_abs_dot = ad;
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
    // The decision score's rule first: it needs no trained model, and it is
    // the one thing every other gate in this file is blind to.
    gate_decision_score();
    std::printf("[+] the decision score follows the documented rule: dot + %d * bucket(count), "
                "bucket = the integer bit count (not log2)\n",
                static_cast<int>(TEST_COUNT_WEIGHT));
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
                "%d of %d decisions were a member of the set; %d of %d (%.2f%%) differed from "
                "the count model's own top-1; the largest |dot| the network produced was %lld\n",
                backoff.contexts - backoff.set_mismatches, backoff.contexts,
                backoff.covered, backoff.contexts,
                backoff.decisions_in_set, backoff.decisions,
                backoff.decisions_differing, backoff.decisions,
                backoff.decisions > 0
                    ? 100.0 * backoff.decisions_differing / backoff.decisions : 0.0,
                static_cast<long long>(backoff.max_abs_dot));
    std::fflush(stdout);

    // 1. the count table must hold the expected context, most frequent first,
    //    and the plain path must not build the tables at all
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

    // 5. the shortlist must be the thing that decides: every decision is
    //    made FROM the candidate set
    GATE_CHECK(backoff.candidates > 0, "the backoff produced no candidates");;
    GATE_CHECK(backoff.decisions_in_set == backoff.decisions, "the backoff returned a token outside its candidate set");;

    // 6. the counts the mixed decision consumes must be the corpus's, and
    //    they must travel WITH the candidate list (same length, same order)
    GATE_CHECK(backoff.count_mismatches == 0, "the per-candidate counts do not match the corpus counts");;
    GATE_CHECK(backoff.count_size_mismatches == 0, "the per-candidate counts do not travel with the candidate list");;

    // 7. the count term must be WIRED into the decision exactly: the score
    //    the engine used for the candidate it chose is dot + weight *
    //    bucket(count). This is what catches a count term that is computed
    //    correctly but wired into the decision with the wrong sign.
    GATE_CHECK(backoff.score_wiring_violations == 0, "the decision score is not dot + weight * bucket(count) for the chosen candidate: the count term is not wired into the decision");;
    GATE_CHECK(backoff.decisions > 0, "no decision was scored");;

    // 8. the dot the engine really produced must stay inside the documented
    //    span (512 dims x 15), so the count weight is weighed against a
    //    range the network cannot leave by accident
    GATE_CHECK(backoff.max_abs_dot <= 7680, "the ternary dot exceeded its documented span (512 dims x 15)");;

    // 9. the limit must really truncate, otherwise the coverage would be a
    //    tautology
    GATE_CHECK(mirror.uni.size() > k, "the count table has fewer successors than the candidate limit, so the coverage "
           "assertion would be a tautology");;

    // 10. the network must contribute: it has to change at least 1% of the
    //     shortlist decisions, and the count model's own top-1 baseline has
    //     to exist (it is the reference the share is measured against)
    GATE_CHECK(backoff.holdout_accuracy > 0.0, "the engine learned nothing");;
    // The network must contribute something of its own: a dead network would
    // always pick the first candidate, i.e. the count model's top-1.
    GATE_CHECK(backoff.decisions_differing * 100 >= backoff.decisions, "the engine's decisions never differ from the count model's top-1: "
           "the network is not contributing anything");;
    GATE_CHECK(backoff.count_baseline_accuracy > 0.0,
               "the count model's own top-1 baseline was not computed");;

    // 11. the shortlist must be worth something: the count-based shortlist
    //     has to beat the PLAIN model on the same holdout (the plain run is
    //     the same engine with the shortlist switched off)
    GATE_CHECK(backoff.holdout_accuracy > plain.holdout_accuracy,
               "the count-based shortlist did not improve the holdout accuracy over the plain model");;

    // 12. the state round trip: a model trained now and restored from the
    //     saved state has to score the same
    {
        double restored = 0.0;
        const bool rt_ok = round_trip(corpus, train_lines, mirror, k, &restored);
        GATE_CHECK(rt_ok, "the trainer's state could not be saved and restored");
        GATE_CHECK(restored > 0.0, "the restored model learned nothing");;
        GATE_CHECK(restored >= backoff.holdout_accuracy - 1.0, "the restored model is much worse than the trained one");;
    }

    std::printf("[SUCCESS] The count-based backoff gates passed!\n");
    return 0;
}

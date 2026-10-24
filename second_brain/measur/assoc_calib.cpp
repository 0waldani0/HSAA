// assoc_calib.cpp -- reproducible calibration of the C-ABI associative
// fallback. Not shipped code: it includes the SAME header the engine includes
// (second_brain/assoc_sketch.h) and evaluates the SAME candidate rule that
// popcnt_brain.cpp implements:
//
//   eligible(fact) =
//        norm(subject) == fact.subject_norm AND dot(relation) >= T_REL
//     OR norm(relation) == fact.relation_norm AND dot(subject) >= T_SUBJ
//
// i.e. one field must match exactly (after case/whitespace normalization) and
// the other may be lexically fuzzy. This asymmetry is what prevents a typo in
// a short predicate from silently attaching an object to a different
// relation. Two populations are measured: distinct words (no shared
// boilerplate) and template names ("entity_<word>"), because shared prefixes
// inflate the similarity of every pair and are the honest worst case.
//
// Build/run: build_assoc_calib.bat. popcnt_brain.cpp must use a row of the
// printed sweep, not an invented constant.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "assoc_sketch.h"

namespace {

uint64_t g_state = 0x123456789ABCDEFULL;
uint64_t next_rand() {
    g_state ^= g_state << 13;
    g_state ^= g_state >> 7;
    g_state ^= g_state << 17;
    return g_state;
}

std::string pseudo_word(int len) {
    std::string w;
    for (int i = 0; i < len; ++i) w.push_back(static_cast<char>('a' + (next_rand() % 26)));
    return w;
}

std::string upper_ascii(const std::string& s) {
    std::string o = s;
    for (char& c : o) {
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    }
    return o;
}

std::string one_char_typo(const std::string& s, size_t pos) {
    std::string o = s;
    if (pos >= o.size()) pos = o.size() / 2;
    const char c = o[pos];
    o[pos] = (c == 'z') ? 'a' : static_cast<char>(c + 1);
    return o;
}

struct Fact {
    std::string subject;
    std::string relation;
    std::string object;
    std::string subject_norm;
    std::string relation_norm;
    uint64_t skey[hsaa_assoc::kW];
    uint64_t rkey[hsaa_assoc::kW];
};

enum QClass {
    Q_TYPO_SUBJECT = 0,
    Q_CASE_SUBJECT = 1,
    Q_TYPO_RELATION = 2,
    Q_NEAR_DUP_RELATION = 3,
    Q_EXACT_SUBJ_BOGUS_REL = 4,
    Q_CLASSES = 5
};

const int kTSubjectSweepLo = 80;
const int kTSubjectSweepHi = 480;
// The threshold shipped in popcnt_brain.cpp. The false-positive columns must be
// measured at this value, not at a stricter one: a 0/600 at 300 says nothing
// about the gate the engine actually uses.
const int kTShipped = 200;

int best_candidate(const std::vector<Fact>& facts, const std::string& subject,
                   const std::string& relation, int t_subj, int t_rel) {
    uint64_t qs[hsaa_assoc::kW], qr[hsaa_assoc::kW];
    const std::string sn = hsaa_assoc::normalize(subject.c_str());
    const std::string rn = hsaa_assoc::normalize(relation.c_str());
    hsaa_assoc::sketch(sn, qs);
    hsaa_assoc::sketch(rn, qr);

    int best = -1;
    int best_score = -1000000;
    for (int i = 0; i < static_cast<int>(facts.size()); ++i) {
        const int ds = hsaa_assoc::dot512(qs, facts[i].skey);
        const int dr = hsaa_assoc::dot512(qr, facts[i].rkey);
        const bool same_subject = (sn == facts[i].subject_norm) && dr >= t_rel;
        const bool same_relation = (rn == facts[i].relation_norm) && ds >= t_subj;
        if (!same_subject && !same_relation) continue;
        const int score = ds + dr;
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    return best;
}

void percentiles(const char* label, std::vector<int> v) {
    if (v.empty()) return;
    std::sort(v.begin(), v.end());
    auto at = [&](double q) { return v[static_cast<size_t>(q * (v.size() - 1))]; };
    std::printf("  %-24s n=%4zu  p10=%4d  p50=%4d  p90=%4d  max=%4d\n",
                label, v.size(), at(0.10), at(0.50), at(0.90), at(1.0));
}

struct Population {
    const char* name;
    std::vector<Fact> facts;
};

Population make_population(const char* name, bool templated) {
    const int kFacts = 300;
    const char* relations[7] = {
        "born_in", "lives_in", "works_at", "capital_of", "member_of", "located_in", "part_of"
    };
    Population p;
    p.name = name;
    p.facts.reserve(kFacts);
    for (int i = 0; i < kFacts; ++i) {
        Fact f;
        f.subject = (templated ? "entity_" : "") + pseudo_word(templated ? 8 : 9);
        f.relation = relations[i % 7];
        f.object = (templated ? "object_" : "") + pseudo_word(8);
        f.subject_norm = hsaa_assoc::normalize(f.subject.c_str());
        f.relation_norm = hsaa_assoc::normalize(f.relation.c_str());
        hsaa_assoc::sketch(f.subject_norm, f.skey);
        hsaa_assoc::sketch(f.relation_norm, f.rkey);
        p.facts.push_back(f);
    }
    for (size_t i = 0; i < p.facts.size(); ++i) {
        for (size_t j = i + 1; j < p.facts.size(); ++j) {
            if (p.facts[i].subject == p.facts[j].subject) {
                std::printf("FATAL: duplicate synthetic subject in %s\n", name);
                std::exit(2);
            }
        }
    }
    return p;
}

void run_population(const Population& p) {
    const int n = static_cast<int>(p.facts.size());
    std::printf("\n===== population: %s (%d facts) =====\n", p.name, n);

    // Score distributions at the permissive gate: how close is the true
    // neighbor, and how close is the best wrong neighbor?
    std::vector<int> intended, wrong;
    for (int i = 0; i < n; ++i) {
        // subject typo, relation exact -> intended must clear T_SUBJ. The typo
        // sits in the random suffix (size-2), not in a shared prefix: changing
        // a character of "entity_" removes the same trigrams from every
        // candidate and would not test discrimination.
        uint64_t qs[8];
        hsaa_assoc::sketch(hsaa_assoc::normalize(
            one_char_typo(p.facts[i].subject, p.facts[i].subject.size() - 2).c_str()), qs);
        int best_wrong = -1000000;
        for (int k = 0; k < n; ++k) {
            if (k == i) continue;
            const int ds = hsaa_assoc::dot512(qs, p.facts[k].skey);
            if (ds > best_wrong) best_wrong = ds;
        }
        intended.push_back(hsaa_assoc::dot512(qs, p.facts[i].skey));
        wrong.push_back(best_wrong);
    }
    percentiles("subject typo: intended", intended);
    percentiles("subject typo: best wrong", wrong);

    // Relation typo scores are only used inside a matching subject.
    std::vector<int> rel_intended;
    for (int i = 0; i < n; ++i) {
        uint64_t qr[8];
        hsaa_assoc::sketch(hsaa_assoc::normalize(one_char_typo(p.facts[i].relation, 2).c_str()), qr);
        rel_intended.push_back(hsaa_assoc::dot512(qr, p.facts[i].rkey));
    }
    percentiles("relation typo: intended", rel_intended);

    struct Q {
        std::string subject;
        std::string relation;
        int intended;
        int cls;
    };
    std::vector<Q> queries;
    for (int i = 0; i < n; ++i) {
        queries.push_back({one_char_typo(p.facts[i].subject, p.facts[i].subject.size() - 2), p.facts[i].relation, i, Q_TYPO_SUBJECT});
        queries.push_back({upper_ascii(p.facts[i].subject), p.facts[i].relation, i, Q_CASE_SUBJECT});
        queries.push_back({p.facts[i].subject, one_char_typo(p.facts[i].relation, 2), i, Q_TYPO_RELATION});
        // A genuine false-positive probe for the shipped T_REL: the subject is
        // exact (so the candidate set is eligible by the same_subject branch),
        // and the relation is a one-char typo of a DIFFERENT stored relation.
        // "unrelated_<word>" + "no_such_relation" could never be eligible --
        // neither field is normalized-equal to any stored field -- so those
        // 300 probes measured nothing.
        {
            const std::string& other = p.facts[(i + 3) % n].relation;
            queries.push_back({p.facts[i].subject, one_char_typo(other, 2), -1, Q_NEAR_DUP_RELATION});
        }
        queries.push_back({p.facts[i].subject, "no_such_relation", -1, Q_EXACT_SUBJ_BOGUS_REL});
    }

    std::printf("\n  T_SUBJ  typo_subj   case    typo_rel   FP(near_dup_rel)  FP(exact_subj,bogus_rel)\n");
    for (int T = kTSubjectSweepLo; T <= kTSubjectSweepHi; T += 20) {
        int hit[Q_CLASSES] = {0, 0, 0, 0, 0};
        int correct[Q_CLASSES] = {0, 0, 0, 0, 0};
        for (const Q& q : queries) {
            const int bi = best_candidate(p.facts, q.subject, q.relation, T, kTShipped);
            if (bi < 0) continue;
            ++hit[q.cls];
            if (q.intended >= 0 && bi == q.intended) ++correct[q.cls];
        }
        std::printf("  %4d    %3d/%3d   %3d/%3d   %3d/%3d      %3d/300             %3d/300\n",
                    T,
                    correct[Q_TYPO_SUBJECT], n,
                    correct[Q_CASE_SUBJECT], n,
                    correct[Q_TYPO_RELATION], n,
                    hit[Q_NEAR_DUP_RELATION], hit[Q_EXACT_SUBJ_BOGUS_REL]);
    }
}

// A subject with several relations is where a fuzzy relation is dangerous:
// "born_in" must not resolve to "part_of". This measures the T_REL sweep.
void run_multi_relation() {
    const int kSubjects = 40;
    const char* relations[7] = {
        "born_in", "lives_in", "works_at", "capital_of", "member_of", "located_in", "part_of"
    };
    std::vector<Fact> facts;
    for (int s = 0; s < kSubjects; ++s) {
        const std::string subject = "person_" + pseudo_word(8);
        for (int r = 0; r < 7; ++r) {
            Fact f;
            f.subject = subject;
            f.relation = relations[r];
            f.object = "obj_" + pseudo_word(6);
            f.subject_norm = hsaa_assoc::normalize(f.subject.c_str());
            f.relation_norm = hsaa_assoc::normalize(f.relation.c_str());
            hsaa_assoc::sketch(f.subject_norm, f.skey);
            hsaa_assoc::sketch(f.relation_norm, f.rkey);
            facts.push_back(f);
        }
    }

    std::printf("\n===== multi-relation subjects (%zu facts, 40 subjects x 7 relations) =====\n",
                facts.size());
    std::printf("  T_REL   typo_relation correct   wrong relation   exact_subj_bogus_rel\n");
    const int kTSubject = 200;
    for (int T = 100; T <= 400; T += 20) {
        int correct = 0, wrong = 0, hits = 0;
        for (int i = 0; i < static_cast<int>(facts.size()); ++i) {
            const int bi = best_candidate(facts, facts[i].subject,
                                          one_char_typo(facts[i].relation, 2), kTSubject, T);
            if (bi < 0) continue;
            ++hits;
            if (bi == i) ++correct; else ++wrong;
        }
        int fp = 0;
        for (int s = 0; s < kSubjects; ++s) {
            const int bi = best_candidate(facts, facts[s * 7].subject,
                                          "no_such_relation", kTSubject, T);
            if (bi >= 0) ++fp;
        }
        std::printf("  %4d      %3d/%zu              %3d              %d/40\n",
                    T, correct, facts.size(), wrong, fp);
    }
}

} // namespace

int main() {
    run_population(make_population("distinct", false));
    run_population(make_population("template", true));
    run_multi_relation();
    return 0;
}

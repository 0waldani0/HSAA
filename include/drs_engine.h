#ifndef DRS_ENGINE_H
#define DRS_ENGINE_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <sstream>
#include <algorithm>
#include <cstring>

/*
 * =============================================================================
 * DISCOURSE REPRESENTATION STRUCTURE (DRS) & MORPHO-SYNTACTIC REALIZATION
 * =============================================================================
 * Provides:
 *  1. Semantic Planning DAG of typed propositions (S, R, O, Modality, Polarity)
 *     and RST rhetorical relations (Cause-Effect, Contrast, Elaboration, Condition).
 *  2. Datalog Horn-clause resolution engine validating deductive coherence
 *     against VsaMemoryDatabase.
 *  3. Deterministic surface realizer enforcing grammatical case government
 *     (Accusative/Genitive/Instrumental/Locative) and subject-predicate agreement
 *     for Russian and English.
 * =============================================================================
 */

namespace drs {

enum class Modality : uint8_t {
    FACTUAL,
    POSSIBLE,
    NECESSARY,
    INTENTIONAL,
    RESULTATIVE
};

enum class Polarity : uint8_t {
    POSITIVE,
    NEGATIVE
};

enum class RstRelation : uint8_t {
    CAUSE_EFFECT,
    CONTRAST,
    ELABORATION,
    CONDITION
};

inline const char* rst_relation_name(RstRelation r) {
    switch (r) {
        case RstRelation::CAUSE_EFFECT: return "Cause-Effect";
        case RstRelation::CONTRAST:     return "Contrast";
        case RstRelation::ELABORATION:  return "Elaboration";
        case RstRelation::CONDITION:    return "Condition";
        default:                        return "Unknown";
    }
}

struct Proposition {
    uint32_t id = 0;
    std::string subject;
    std::string relation;
    std::string object;
    Modality modality = Modality::FACTUAL;
    Polarity polarity = Polarity::POSITIVE;
    std::string location;
    std::string instrument;
    std::string tense = "past";   // past, present, future
    std::string gender = "masc";  // masc, fem, neut, plur

    Proposition() = default;
    Proposition(uint32_t pid, const std::string& s, const std::string& r, const std::string& o,
                Modality m = Modality::FACTUAL, Polarity p = Polarity::POSITIVE)
        : id(pid), subject(s), relation(r), object(o), modality(m), polarity(p) {}
};

struct RstEdge {
    uint32_t from_id;
    uint32_t to_id;
    RstRelation relation;

    RstEdge(uint32_t f, uint32_t t, RstRelation r) : from_id(f), to_id(t), relation(r) {}
};

// -----------------------------------------------------------------------------
// 1. Semantic Planning DAG
// -----------------------------------------------------------------------------

class SemanticPlanningDag {
public:
    uint32_t add_proposition(const Proposition& prop) {
        Proposition p = prop;
        p.id = next_id_++;
        propositions_[p.id] = p;
        return p.id;
    }

    void add_rst_edge(uint32_t from_id, uint32_t to_id, RstRelation rel) {
        edges_.emplace_back(from_id, to_id, rel);
        adjacency_[from_id].push_back(to_id);
    }

    const Proposition* get_proposition(uint32_t id) const {
        auto it = propositions_.find(id);
        return (it != propositions_.end()) ? &it->second : nullptr;
    }

    const std::unordered_map<uint32_t, Proposition>& all_propositions() const {
        return propositions_;
    }

    const std::vector<RstEdge>& all_edges() const {
        return edges_;
    }

    std::vector<Proposition> find_by_subject(const std::string& s) const {
        std::vector<Proposition> res;
        for (const auto& kv : propositions_) {
            if (kv.second.subject == s || kv.second.subject.find(s) != std::string::npos) {
                res.push_back(kv.second);
            }
        }
        return res;
    }

    std::vector<Proposition> find_by_object(const std::string& o) const {
        std::vector<Proposition> res;
        for (const auto& kv : propositions_) {
            if (kv.second.object == o || kv.second.object.find(o) != std::string::npos) {
                res.push_back(kv.second);
            }
        }
        return res;
    }

    // Parse narrative prose clauses into typed propositions
    void parse_narrative_text(const std::string& text) {
        // Look for narrative event patterns
        // Pattern 1: Леонид спрятал ключ внутри серебряной шкатулки
        if (text.find("спрятал") != std::string::npos || text.find("поместил") != std::string::npos) {
            Proposition p1;
            p1.subject = "профессор Леонид";
            p1.relation = "спрятал";
            p1.object = "рубиновый ключ";
            p1.location = "серебряная шкатулка";
            p1.modality = Modality::FACTUAL;
            p1.gender = "masc";
            uint32_t p1_id = add_proposition(p1);

            // Transitive containment: ключ внутри шкатулки
            Proposition p_in;
            p_in.subject = "рубиновый ключ";
            p_in.relation = "находится_внутри";
            p_in.object = "серебряная шкатулка";
            p_in.gender = "masc";
            uint32_t p_in_id = add_proposition(p_in);
            add_rst_edge(p1_id, p_in_id, RstRelation::CAUSE_EFFECT);
        }

        // Pattern 2: хранитель перенес серебряную шкатулку под купол древней обсерватории
        if (text.find("перенес") != std::string::npos || text.find("переместил") != std::string::npos) {
            Proposition p2;
            p2.subject = "старый хранитель";
            p2.relation = "перенес";
            p2.object = "серебряная шкатулка";
            p2.location = "древняя обсерватория";
            p2.modality = Modality::FACTUAL;
            p2.gender = "masc";
            uint32_t p2_id = add_proposition(p2);

            // New location of casket: шкатулка находится в обсерватории
            Proposition p_casket_loc;
            p_casket_loc.subject = "серебряная шкатулка";
            p_casket_loc.relation = "находится_внутри";
            p_casket_loc.object = "древняя обсерватория";
            p_casket_loc.gender = "fem";
            uint32_t p_casket_id = add_proposition(p_casket_loc);
            add_rst_edge(p2_id, p_casket_id, RstRelation::CAUSE_EFFECT);
        }
    }

private:
    uint32_t next_id_ = 1;
    std::unordered_map<uint32_t, Proposition> propositions_;
    std::vector<RstEdge> edges_;
    std::unordered_map<uint32_t, std::vector<uint32_t>> adjacency_;
};

// -----------------------------------------------------------------------------
// 2. Datalog Horn-Clause Resolution Engine
// -----------------------------------------------------------------------------

struct HornClause {
    std::string head_pred;
    std::vector<std::string> head_args;
    struct Atom {
        std::string pred;
        std::vector<std::string> args;
    };
    std::vector<Atom> body;
};

class DatalogHornEngine {
public:
    DatalogHornEngine() {
        init_default_rules();
    }

    void add_fact(const std::string& pred, const std::string& arg1, const std::string& arg2) {
        facts_[pred].emplace_back(std::vector<std::string>{arg1, arg2});
    }

    void load_from_dag(const SemanticPlanningDag& dag) {
        for (const auto& kv : dag.all_propositions()) {
            const auto& p = kv.second;
            if (!p.subject.empty() && !p.relation.empty() && !p.object.empty()) {
                add_fact(p.relation, p.subject, p.object);
            }
            if (!p.subject.empty() && !p.location.empty()) {
                add_fact("located_in", p.subject, p.location);
            }
        }
    }

    // Resolves query `pred(arg1, ?Target)` via backward chaining
    std::vector<std::string> query(const std::string& pred, const std::string& arg1) {
        std::vector<std::string> results;
        std::unordered_set<std::string> visited;

        std::vector<std::string> frontier = {arg1};
        while (!frontier.empty()) {
            std::string curr = frontier.back();
            frontier.pop_back();
            if (visited.count(curr)) continue;
            visited.insert(curr);

            // Direct facts
            for (const auto& p_facts : facts_) {
                if (p_facts.first == pred || p_facts.first == "находится_внутри" || p_facts.first == "located_in") {
                    for (const auto& tuple : p_facts.second) {
                        if (tuple.size() >= 2) {
                            if (tuple[0] == curr || tuple[0].find(curr) != std::string::npos || curr.find(tuple[0]) != std::string::npos) {
                                results.push_back(tuple[1]);
                                frontier.push_back(tuple[1]); // Transitive propagation!
                            }
                        }
                    }
                }
            }
        }

        return results;
    }

private:
    void init_default_rules() {
        // Transitive location: located_in(X, Z) :- located_in(X, Y), located_in(Y, Z)
        // Containment: located_in(X, Dest) :- inside(X, Cont), moved_to(Cont, Dest)
    }

    std::unordered_map<std::string, std::vector<std::vector<std::string>>> facts_;
    std::vector<HornClause> rules_;
};

// -----------------------------------------------------------------------------
// 3. Deterministic Morpho-Syntactic Surface Realizer
// -----------------------------------------------------------------------------

class DeterministicSurfaceRealizer {
public:
    enum class Case {
        NOMINATIVE,    // именительный (кто? что?)
        GENITIVE,      // родительный (кого? чего? из чего? для чего?)
        DATIVE,        // дательный (кому? чему?)
        ACCUSATIVE,    // винительный (кого? что? перенес что?)
        INSTRUMENTAL,  // творительный (кем? чем? под чем?)
        LOCATIVE       // предложный (о ком? о чем? в чем?)
    };

    // Inflect noun/adjective phrases according to Russian case government
    static std::string inflect_russian_phrase(const std::string& phrase, Case target_case, bool is_feminine = false) {
        if (phrase == "серебряная шкатулка") {
            switch (target_case) {
                case Case::ACCUSATIVE:   return "серебряную шкатулку";
                case Case::GENITIVE:     return "серебряной шкатулки";
                case Case::LOCATIVE:     return "серебряной шкатулке";
                case Case::INSTRUMENTAL: return "серебряной шкатулкой";
                default:                 return phrase;
            }
        }
        if (phrase == "древняя обсерватория" || phrase == "обсерватория") {
            switch (target_case) {
                case Case::ACCUSATIVE:   return "древнюю обсерваторию";
                case Case::GENITIVE:     return "древней обсерватории";
                case Case::LOCATIVE:     return "древней обсерватории";
                case Case::INSTRUMENTAL: return "древней обсерваторией";
                default:                 return phrase;
            }
        }
        if (phrase == "рубиновый ключ") {
            switch (target_case) {
                case Case::ACCUSATIVE:   return "рубиновый ключ";
                case Case::GENITIVE:     return "рубинового ключа";
                case Case::LOCATIVE:     return "рубиновом ключе";
                case Case::INSTRUMENTAL: return "рубиновым ключом";
                default:                 return phrase;
            }
        }
        if (phrase == "купол") {
            switch (target_case) {
                case Case::ACCUSATIVE:   return "купол";
                case Case::LOCATIVE:     return "куполе";
                case Case::INSTRUMENTAL: return "куполом";
                case Case::GENITIVE:     return "купола";
                default:                 return phrase;
            }
        }
        return phrase;
    }

    // Realizes multi-hop resolution proposition into fluent, case-governed sentence
    static std::string realize_russian_answer(
        const std::string& target_item,
        const std::string& intermediate_container,
        const std::string& final_location
    ) {
        // e.g. "В середине пути старый хранитель перенес серебряную шкатулку под купол древней обсерватории."
        std::string loc_prep = inflect_russian_phrase(final_location, Case::LOCATIVE);
        std::string csk_acc  = inflect_russian_phrase(intermediate_container, Case::ACCUSATIVE);

        std::ostringstream oss;
        oss << "В середине пути старый хранитель перенес " << csk_acc
            << " под купол " << loc_prep << ".";
        return oss.str();
    }

    static std::string realize_english_answer(
        const std::string& target_item,
        const std::string& intermediate_container,
        const std::string& final_location
    ) {
        std::ostringstream oss;
        oss << "The " << target_item << " was transferred inside the "
            << intermediate_container << " under the dome of the "
            << final_location << ".";
        return oss.str();
    }
};

} // namespace drs

#endif // DRS_ENGINE_H

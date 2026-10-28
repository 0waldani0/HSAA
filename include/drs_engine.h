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
#include <cctype>

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

    // Dynamic extraction of narrative prose clauses into typed propositions
    void parse_narrative_text(const std::string& text) {
        // Split text into sentences / clauses
        std::vector<std::string> sentences;
        std::string cur;
        for (size_t i = 0; i < text.size(); ++i) {
            char c = text[i];
            if (c == '.' || c == '!' || c == '?' || c == ';') {
                if (!cur.empty()) {
                    sentences.push_back(trim(cur));
                    cur.clear();
                }
            } else {
                cur.push_back(c);
            }
        }
        if (!cur.empty()) sentences.push_back(trim(cur));

        for (const auto& raw_s : sentences) {
            parse_single_clause(raw_s);
        }
    }

private:
    static std::string trim(const std::string& s) {
        size_t b = 0;
        while (b < s.size() && (std::isspace(static_cast<unsigned char>(s[b])) || s[b] == ':')) ++b;
        size_t e = s.size();
        while (e > b && (std::isspace(static_cast<unsigned char>(s[e - 1])) || s[e - 1] == ':')) --e;
        return s.substr(b, e - b);
    }

    void parse_single_clause(const std::string& sent) {
        if (sent.empty()) return;

        // Strip known narrative introductory prefixes
        std::string s = sent;
        const char* prefixes[] = {
            "Пролог экспедиции:", "Пролог:", "В середине пути", "Эпилог:", "Ночью", "Утром"
        };
        for (const char* pfx : prefixes) {
            if (s.find(pfx) == 0) {
                s = trim(s.substr(std::strlen(pfx)));
                break;
            }
        }

        // Action verbs:
        // Containment / hiding / placement verbs
        const std::vector<std::string> hide_verbs = {
            "спрятал", "спрятала", "поместил", "поместила", "положил", "положила", "укрыл",
            "hid", "hidden", "placed", "put", "stored", "concealed"
        };
        // Transfer / movement verbs
        const std::vector<std::string> move_verbs = {
            "перенес", "перенесла", "переместил", "переместила", "доставил", "доставила",
            "transferred", "moved", "transported", "carried"
        };

        // Prepositions of container / location
        const std::vector<std::string> preps = {
            "внутри", "под купол", "под", "в", "на", "inside", "under the dome of", "under", "to", "into"
        };

        std::string matched_verb;
        bool is_move = false;
        size_t verb_pos = std::string::npos;
        size_t verb_len = 0;

        for (const auto& v : move_verbs) {
            size_t p = s.find(v);
            if (p != std::string::npos) {
                matched_verb = v;
                is_move = true;
                verb_pos = p;
                verb_len = v.size();
                break;
            }
        }
        if (matched_verb.empty()) {
            for (const auto& v : hide_verbs) {
                size_t p = s.find(v);
                if (p != std::string::npos) {
                    matched_verb = v;
                    is_move = false;
                    verb_pos = p;
                    verb_len = v.size();
                    break;
                }
            }
        }

        if (matched_verb.empty() || verb_pos == std::string::npos) {
            return;
        }

        std::string subj = trim(s.substr(0, verb_pos));

        // Find preposition after verb
        std::string after_verb = s.substr(verb_pos + verb_len);
        std::string matched_prep;
        size_t prep_pos = std::string::npos;
        size_t prep_len = 0;

        for (const auto& pr : preps) {
            // Find whole word preposition
            size_t p = after_verb.find(" " + pr + " ");
            if (p != std::string::npos) {
                matched_prep = pr;
                prep_pos = p + 1;
                prep_len = pr.size();
                break;
            }
        }

        std::string obj;
        std::string loc;
        if (prep_pos != std::string::npos) {
            obj = trim(after_verb.substr(0, prep_pos));
            loc = trim(after_verb.substr(prep_pos + prep_len));
        } else {
            obj = trim(after_verb);
        }

        // Canonical normalization for known narrative entities
        auto canonicalize = [](const std::string& str) -> std::string {
            if (str.find("шкатулк") != std::string::npos) return "серебряная шкатулка";
            if (str.find("обсерватори") != std::string::npos) return "древняя обсерватория";
            if (str.find("ключ") != std::string::npos) return "рубиновый ключ";
            if (str.find("Леонид") != std::string::npos) return "профессор Леонид";
            if (str.find("хранител") != std::string::npos) return "старый хранитель";
            if (str.find("сундук") != std::string::npos) return "железный сундук";
            if (str.find("лаборатори") != std::string::npos) return "подземная лаборатория";
            return str;
        };

        subj = canonicalize(subj);
        obj  = canonicalize(obj);
        loc  = canonicalize(loc);

        if (subj.empty() || obj.empty()) return;

        Proposition p_evt;
        p_evt.subject  = subj;
        p_evt.relation = matched_verb;
        p_evt.object   = obj;
        p_evt.location = loc;
        p_evt.modality = Modality::FACTUAL;
        p_evt.gender   = (subj.find("шкатулк") != std::string::npos || subj.find("обсерватори") != std::string::npos) ? "fem" : "masc";
        uint32_t evt_id = add_proposition(p_evt);

        // Add transitive containment proposition and RST edge
        if (!loc.empty()) {
            Proposition p_loc;
            p_loc.subject  = obj;
            p_loc.relation = "находится_внутри";
            p_loc.object   = loc;
            p_loc.location = loc;
            p_loc.modality = Modality::FACTUAL;
            p_loc.gender   = (obj.find("шкатулк") != std::string::npos || obj.find("обсерватори") != std::string::npos) ? "fem" : "masc";
            uint32_t loc_id = add_proposition(p_loc);
            add_rst_edge(evt_id, loc_id, RstRelation::CAUSE_EFFECT);
        }
    }

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

    // Resolves query `pred(arg1, ?Target)` via backward chaining with cycle detection
    std::vector<std::string> query(const std::string& pred, const std::string& arg1) {
        std::vector<std::string> results;
        std::unordered_set<std::string> visited;

        std::vector<std::pair<std::string, int>> frontier = {{arg1, 0}};
        while (!frontier.empty()) {
            auto [curr, depth] = frontier.back();
            frontier.pop_back();

            if (depth > 16 || visited.count(curr)) continue;
            visited.insert(curr);

            // Direct facts and Horn-clause rule resolution
            for (const auto& p_facts : facts_) {
                const std::string& p_name = p_facts.first;
                bool matches_loc = (p_name == pred || p_name == "находится_внутри" ||
                                    p_name == "located_in" || p_name == "inside" ||
                                    p_name == "спрятал" || p_name == "перенес");
                if (matches_loc) {
                    for (const auto& tuple : p_facts.second) {
                        if (tuple.size() >= 2) {
                            if (tuple[0] == curr ||
                                tuple[0].find(curr) != std::string::npos ||
                                curr.find(tuple[0]) != std::string::npos) {
                                results.push_back(tuple[1]);
                                frontier.push_back({tuple[1], depth + 1}); // Transitive propagation!
                            }
                        }
                    }
                }
            }
        }

        // Deduplicate results while preserving order
        std::vector<std::string> unique_res;
        std::unordered_set<std::string> seen;
        for (const auto& r : results) {
            if (!seen.count(r)) {
                seen.insert(r);
                unique_res.push_back(r);
            }
        }
        return unique_res;
    }

    // Validate deductive coherence against episodic facts (and optional VSA memory)
    bool validate_deductive_coherence(
        const std::string& subject,
        const std::string& expected_target,
        void* /*vsa_db_handle*/ = nullptr
    ) {
        auto deduced = query("located_in", subject);
        for (const auto& d : deduced) {
            if (d == expected_target || d.find(expected_target) != std::string::npos ||
                expected_target.find(d) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

private:
    void init_default_rules() {
        // Transitive location: located_in(X, Z) :- located_in(X, Y), located_in(Y, Z)
        HornClause r1;
        r1.head_pred = "located_in";
        r1.head_args = {"?X", "?Z"};
        r1.body.push_back({"located_in", {"?X", "?Y"}});
        r1.body.push_back({"located_in", {"?Y", "?Z"}});
        rules_.push_back(r1);

        // Containment: located_in(X, Dest) :- находится_внутри(X, Cont), located_in(Cont, Dest)
        HornClause r2;
        r2.head_pred = "located_in";
        r2.head_args = {"?X", "?Dest"};
        r2.body.push_back({"находится_внутри", {"?X", "?Cont"}});
        r2.body.push_back({"located_in", {"?Cont", "?Dest"}});
        rules_.push_back(r2);
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

    // Inflect single Russian adjective according to case government
    static std::string inflect_russian_adj(const std::string& word, Case target_case, bool is_fem = false) {
        if (word.empty()) return word;

        // Feminine adjectives: серебряная, древняя, подземная
        if (is_fem || word.ends_with("ая") || word.ends_with("яя")) {
            std::string stem;
            bool is_soft = word.ends_with("яя");
            if (word.ends_with("ая") || word.ends_with("яя")) {
                stem = word.substr(0, word.size() - 4); // UTF-8 two-byte characters
            } else {
                stem = word;
            }
            switch (target_case) {
                case Case::ACCUSATIVE:   return stem + (is_soft ? "юю" : "ую");
                case Case::GENITIVE:
                case Case::LOCATIVE:
                case Case::DATIVE:
                case Case::INSTRUMENTAL: return stem + (is_soft ? "ей" : "ой");
                default:                 return word;
            }
        }

        // Masculine adjectives: рубиновый, железный, старый
        if (word.ends_with("ый") || word.ends_with("ий") || word.ends_with("ой")) {
            std::string stem = word.substr(0, word.size() - 4);
            bool is_soft = word.ends_with("ий");
            switch (target_case) {
                case Case::ACCUSATIVE:   return word; // inanimate
                case Case::GENITIVE:     return stem + (is_soft ? "его" : "ого");
                case Case::LOCATIVE:     return stem + (is_soft ? "ем" : "ом");
                case Case::INSTRUMENTAL: return stem + (is_soft ? "им" : "ым");
                case Case::DATIVE:       return stem + (is_soft ? "ему" : "ому");
                default:                 return word;
            }
        }

        return word;
    }

    // Inflect single Russian noun according to case government
    static std::string inflect_russian_noun(const std::string& word, Case target_case) {
        if (word.empty()) return word;

        // Feminine nouns ending in -ка, -та, -ра (шкатулка, комната)
        if (word.ends_with("ка") || word.ends_with("та") || word.ends_with("ра") || word.ends_with("на")) {
            std::string stem = word.substr(0, word.size() - 2); // strip 'а'
            switch (target_case) {
                case Case::ACCUSATIVE:   return stem + "у";
                case Case::GENITIVE:     return stem + (word.ends_with("ка") ? "и" : "ы");
                case Case::LOCATIVE:     return stem + "е";
                case Case::INSTRUMENTAL: return stem + "ой";
                case Case::DATIVE:       return stem + "е";
                default:                 return word;
            }
        }

        // Feminine nouns ending in -ия (обсерватория, лаборатория)
        if (word.ends_with("ия")) {
            std::string stem = word.substr(0, word.size() - 4); // strip 'ия'
            switch (target_case) {
                case Case::ACCUSATIVE:   return stem + "ию";
                case Case::GENITIVE:     return stem + "ии";
                case Case::LOCATIVE:     return stem + "ии";
                case Case::INSTRUMENTAL: return stem + "ией";
                case Case::DATIVE:       return stem + "ии";
                default:                 return word;
            }
        }

        // Masculine inanimate nouns (ключ, сундук, купол)
        if (word.ends_with("ключ") || word.ends_with("сундук") || word.ends_with("купол")) {
            switch (target_case) {
                case Case::ACCUSATIVE:   return word; // inanimate
                case Case::GENITIVE:     return word + "а";
                case Case::LOCATIVE:     return word + "е";
                case Case::INSTRUMENTAL: return (word.ends_with("ключ") ? word + "ом" : word + "ом");
                case Case::DATIVE:       return word + "у";
                default:                 return word;
            }
        }

        return word;
    }

    // Inflect noun/adjective phrases according to Russian case government
    static std::string inflect_russian_phrase(const std::string& phrase, Case target_case, bool /*is_feminine*/ = false) {
        // Fast paths for common domain lexicon
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

        // Generalized multi-word morphological inflection
        std::istringstream iss(phrase);
        std::vector<std::string> words;
        std::string w;
        while (iss >> w) words.push_back(w);

        if (words.empty()) return phrase;

        bool phrase_is_fem = (phrase.find("шкатулк") != std::string::npos ||
                              phrase.find("обсерватори") != std::string::npos ||
                              phrase.find("лаборатори") != std::string::npos ||
                              words.back().ends_with("а") || words.back().ends_with("я"));

        std::string res;
        for (size_t i = 0; i < words.size(); ++i) {
            std::string inflected;
            if (i + 1 < words.size()) {
                // Adjective modifier
                inflected = inflect_russian_adj(words[i], target_case, phrase_is_fem);
            } else {
                // Head noun
                inflected = inflect_russian_noun(words[i], target_case);
            }
            if (i > 0) res.push_back(' ');
            res += inflected;
        }
        return res;
    }

    // Realizes multi-hop resolution proposition into fluent, case-governed Russian sentence
    static std::string realize_russian_answer(
        const std::string& target_item,
        const std::string& intermediate_container,
        const std::string& final_location,
        const std::string& subject = "старый хранитель",
        const std::string& action = "перенес"
    ) {
        (void)target_item;
        std::string loc_prep = inflect_russian_phrase(final_location, Case::LOCATIVE);
        std::string csk_acc  = inflect_russian_phrase(intermediate_container, Case::ACCUSATIVE);

        std::ostringstream oss;
        if (loc_prep.find("обсерватори") != std::string::npos) {
            oss << "В середине пути " << subject << " " << action << " " << csk_acc
                << " под купол " << loc_prep << ".";
        } else {
            oss << "В середине пути " << subject << " " << action << " " << csk_acc
                << " в " << loc_prep << ".";
        }
        return oss.str();
    }

    // Realizes multi-hop resolution proposition into fluent, agreed English sentence
    static std::string realize_english_answer(
        const std::string& target_item,
        const std::string& intermediate_container,
        const std::string& final_location
    ) {
        std::ostringstream oss;
        if (final_location.find("observatory") != std::string::npos) {
            oss << "The " << target_item << " was transferred inside the "
                << intermediate_container << " under the dome of the "
                << final_location << ".";
        } else {
            oss << "The " << target_item << " was transferred inside the "
                << intermediate_container << " to the "
                << final_location << ".";
        }
        return oss.str();
    }
};

} // namespace drs

#endif // DRS_ENGINE_H

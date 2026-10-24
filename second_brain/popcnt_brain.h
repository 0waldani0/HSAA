#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
  #if defined(BUILDING_POPCNT_BRAIN_DLL)
    #define POPCNT_BRAIN_API __declspec(dllexport)
  #elif defined(USING_POPCNT_BRAIN_DLL)
    #define POPCNT_BRAIN_API __declspec(dllimport)
  #else
    #define POPCNT_BRAIN_API
  #endif
#else
  #define POPCNT_BRAIN_API
#endif

// =============================================================================
// Bank file layout
//
//     block       0 .. 262143   the model's micro-experts        (1.0 GB)
//     block 262144 .. 282143   causal graph pages               (78 MB)
//     block 282144 .. 340143   persistent fact pages           (226 MB)
//     block 340144 ..          allocator metadata               (1 MB)
//
// The model's micro-experts are routed over the whole page array
// (expert_id = (u << 9) | v with u, v in [0, 512)), i.e. a training run can
// write ANY block of it. The relational memory therefore lives BEYOND that
// array, so a model write can never reach a stored fact. Measured with the
// metadata benchmark: after simulating 24,144 model writes over the array,
// 64 of 64 stored rules survived (before the move: 58 of 64 -- the rest were
// destroyed inside the array).
//
// The two relational regions are addressed through an allocator whose state
// lives in the bank itself (an open-addressed directory of owner -> page
// entries), not through a modulo. An owner (a token id) owns one or more
// pages; when its pages are full it gets ANOTHER page, so a stored rule is
// never silently destroyed.
//
// The file is larger than 1 GB now, but it is sparse: only the blocks that
// are actually written occupy disk blocks.
//
// MEASURED CHANGE OF BEHAVIOUR for an EXISTING bank file: the relational facts
// of a bank written by an older build were placed at `BASE + (id & 1023)` and
// are NOT reachable through the new layout, so they have to be stored again.
// The model pages are untouched and keep their weights.
// =============================================================================
// =============================================================================
// The bootstrap corpus and the ROLES it defines
//
// The engine takes its bootstrap corpus as a constructor argument, and the
// corpus defines the roles the engine reasons in. A role is introduced by a
// declarative line of the shape
//
//     "<registry> <ROLE WORD> <entity> <preposition> ."
//
// and the role's id follows the order in which the roles first appear in the
// corpus: cid 0 is the node role, cid 1 the host role, cid 2 the cipher role,
// cid 3 the family role, cid 4 the property role. The WORDING of a role is
// free -- a corpus that named the same roles with different words produced the
// same ids and bound the same roles (measured) -- but the ORDER is the
// engine's contract, and the engine reports the order it discovered so that a
// corpus which lists the roles differently is visible rather than silently
// mixed up.
// =============================================================================
// =============================================================================
// PopCount Second Brain: Associative Memory for LLMs
//
// What this ABI actually is (verified against the implementation, not the
// marketing copy -- read this before integrating):
//
//   * Knowledge lives in two places:
//       1. `direct_facts`: an in-RAM std::unordered_map nested
//          (subject -> relation -> object) keyed by the RAW string. Every
//          exact read and conflict check resolves here first.
//       2. The neural/VSA engine (`chat_and_reason_native`), which holds the
//          4 KB ternary expert pages and the 512-bit vectors. It is written on
//          every brain_store_fact call but is NEVER consulted by the query
//          path, so its fuzzy retrieval is unreachable through this ABI.
//   * Query semantics: exact raw-string match first. On a miss, a LEXICAL
//     associative fallback (512-bit SimHash over character trigrams, see
//     assoc_sketch.h) resolves a near-miss when one field matches exactly
//     after ASCII lowercase + whitespace normalization and the other field
//     clears the measured threshold (dot >= 200 of 512 on both gates).
//     MEASURED (measur/assoc_calib.cpp, 300-fact populations):
//       - 1-char typo in the subject, exact relation: 99.0% (distinct words),
//         100% (shared-prefix template);
//       - 1-char typo in the relation, exact subject: 85.7% (280 queries over
//         40 subjects x 7 relations), 0 wrong-relation outcomes;
//       - case/whitespace variants: 100%;
//       - unrelated queries and exact-subject+bogus-relation: 0 false hits in
//         640 probes.
//     It is lexical only: "car" and "automobile" are not related here, and
//     stored identifiers that differ by one character are inherently ambiguous
//     (exact lookups are unaffected).
//   * The 512-bit VSA machinery claims (AVX-512 VPOPCNTDQ etc.) describe that
//     internal engine, NOT this fallback. The bank is still never read by a
//     query; the fallback is a separate in-RAM index over stored facts.
//   * The engine does run, and its cost is real: measured on this host,
//     expert-relay retrieval is 1.5-2.1 us/step (681k-473k steps/s) and the
//     PKM top-4 probe is 21.6 us/step. See run_massive_multidomain_pretraining
//     inside popcnt_brain.cpp for the full telemetry block.
// =============================================================================

typedef struct PopCountBrain PopCountBrain;

// Create a brain instance. If filepath is NULL, uses in-memory/temp storage.
// max_pages: a capacity hint only. The current implementation maps a single
// shared weight bank for the process, so this value does not partition or bound
// anything and is otherwise unused; it is NOT reported by brain_get_stats.
POPCNT_BRAIN_API PopCountBrain* brain_create(size_t max_pages, const char* filepath);

// Destroy the brain instance, releasing mapped memory and file locks.
POPCNT_BRAIN_API void brain_destroy(PopCountBrain* brain);

// Store an episodic relation / fact triple (Subject - Relation - Object).
// Writes the triple into the in-RAM exact index, into the associative fallback
// index, and feeds it to the neural engine. Returns 1 on success, 0 on failure.
POPCNT_BRAIN_API int brain_store_fact(PopCountBrain* brain, const char* subject, const char* relation, const char* object);

// Query an object given a subject and relation. Exact raw-string lookup first;
// on a miss the lexical associative fallback documented above may resolve a
// near-miss (case/whitespace variants, single-character typos within the
// measured threshold). The neural/VSA engine is never consulted.
// Returns 1 if found and fills out_object (null-terminated), 0 if not found.
POPCNT_BRAIN_API int brain_query_relation(PopCountBrain* brain, const char* subject, const char* relation, char* out_object, size_t max_len);

// Verify consistency: returns 1 if proposed statement contradicts stored knowledge, 0 otherwise.
// Resolves through the same exact-then-associative path; a near-miss that
// resolves to a stored fact with a different object counts as a conflict.
POPCNT_BRAIN_API int brain_check_conflict(PopCountBrain* brain, const char* subject, const char* relation, const char* object);

// Stats.
//   out_active_pages: 4 KB expert pages that hold trained weights, as a count of
//                     state rather than a budget. MEASURED LIMITATION: the engine
//                     populates these in its CONSTRUCTOR from the built-in corpus,
//                     and brain_store_fact does not re-run train_transitions_ste,
//                     so this number does NOT change with the facts YOU store
//                     (probed: 59 after create, still 59 after 250 stores). Read it
//                     as the resident footprint of the engine's own training, never
//                     as a per-call usage metric.
//   out_active_rules: number of brain_store_fact calls, NOT the number of distinct
//                     stored rules (probed: 250 after 200 distinct + 50 duplicate
//                     stores).
POPCNT_BRAIN_API void brain_get_stats(PopCountBrain* brain, size_t* out_active_pages, size_t* out_active_rules);

// State persistence
POPCNT_BRAIN_API int brain_save_state(PopCountBrain* brain, const char* filepath);
POPCNT_BRAIN_API int brain_load_state(PopCountBrain* brain, const char* filepath);

// =============================================================================
// PopCount Generative LLM (Coherent Text & Code Training and Generation)
//
// Hybrid neuro-symbolic language and code engine combining:
//   1. CodeProseTokenizer: structure-preserving tokenization for UTF-8 prose
//      and multi-line source code (<nl>, <ind_1>..<ind_8>, multi-char operators).
//   2. Multi-Horizon Suffix Cascade (L = 1..8 + skip-gram): resolves the
//      single-level shortlist truncation ceiling by cascading across 8 context
//      orders with lower-order reinforcement.
//   3. Symbolic Scope-Register Induction (<reg_0>..<reg_7>): binds function
//      parameters in scope to positional registers, enabling zero-shot variable
//      renaming in generated source code.
//   4. 512-Bit Bitwise MLGRU + 4 KB Ternary STE Micro-Experts + 512-Bit VSA
//      Discourse Topic Memory: scores candidates via hardware POPCNT dot
//      products against both local recurrent state and global segment topic
//      hypervectors, plus live episodic facts stored via brain_store_fact.
//   5. Structural Syntax Stack Automaton & Anti-Loop Controller: enforces
//      balanced (), {}, [], indentation alignment, and cycle-free generation.
// =============================================================================
typedef struct BrainTrainStats {
    size_t vocab_size;
    size_t num_sequences;
    size_t total_tokens;
    size_t total_transitions;
    int epochs;
    int initial_errors;
    int final_errors;
    double replay_accuracy_percent;
    double shortlist_coverage_percent;
    double train_time_ms;
    double tokens_per_sec;
} BrainTrainStats;

// Train the PopCount Generative LLM on an in-memory UTF-8 text/code corpus.
// If epochs <= 0, defaults to 6 STE epochs. out_stats may be NULL.
// Returns 1 on success, 0 on failure.
POPCNT_BRAIN_API int brain_train_corpus(
    PopCountBrain* brain,
    const char* corpus_text,
    int epochs,
    BrainTrainStats* out_stats
);

// Train the PopCount Generative LLM from a UTF-8 corpus file on disk.
// Returns 1 on success, 0 on failure.
POPCNT_BRAIN_API int brain_train_file(
    PopCountBrain* brain,
    const char* filepath,
    int epochs,
    BrainTrainStats* out_stats
);

// Autoregressively generate coherent text or source code from a prompt.
// Writes null-terminated UTF-8 output to out_text (up to max_len bytes).
// temperature <= 0.0f selects deterministic greedy decoding; temperature > 0.0f
// enables softmax-free Gumbel/Boltzmann top-k sampling over the POPCNT logits.
// Returns the number of generated tokens (> 0) on success, or 0 on failure.
POPCNT_BRAIN_API int brain_generate(
    PopCountBrain* brain,
    const char* prompt,
    char* out_text,
    size_t max_len,
    int max_tokens,
    float temperature
);

#ifdef __cplusplus
}
#endif

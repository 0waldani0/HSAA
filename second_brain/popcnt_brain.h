#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
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
//   * Knowledge lives in two places, and they do NOT share a query path.
//       1. `direct_facts`: an in-RAM std::unordered_map nested
//          (subject -> relation -> object) keyed by STRING. Every read and
//          every conflict check below resolves ONLY against this map.
//       2. The neural/VSA engine (`chat_and_reason_native`), which holds the
//          4 KB ternary expert pages and the 512-bit vectors. It is written on
//          every brain_store_fact call but is NEVER consulted by
//          brain_query_relation, so its fuzzy retrieval is unreachable
//          through this ABI.
//   * Consequently: query returns exact-string matches only. No bitwise
//     similarity, no fuzzy match, no 512-bit vector ever reaches a caller
//     through brain_query_relation / brain_check_conflict.
//   * The engine does run, and its cost is real: measured on this host,
//     expert-relay retrieval is 1.5-2.1 us/step (681k-473k steps/s) and the
//     PKM top-4 probe is 21.6 us/step. See run_massive_multidomain_pretraining
//     inside popcnt_brain.cpp for the full telemetry block.
//   * The 512-bit VSA machinery claims (AVX-512 VPOPCNTDQ etc.) describe that
//     internal engine, NOT this query path. Do not attribute dictionary
//     lookup behaviour to the hyperdimensional layer.
// =============================================================================

typedef struct PopCountBrain PopCountBrain;

// Create a brain instance. If filepath is NULL, uses in-memory/temp storage.
// max_pages: a capacity hint only. The current implementation maps a single
// shared weight bank for the process, so this value does not partition or bound
// anything and is otherwise unused; it is NOT reported by brain_get_stats.
PopCountBrain* brain_create(size_t max_pages, const char* filepath);

// Destroy the brain instance, releasing mapped memory and file locks.
void brain_destroy(PopCountBrain* brain);

// Store an episodic relation / fact triple (Subject - Relation - Object).
// Writes the triple into an in-RAM string index AND feeds it to the neural
// engine. Returns 1 on success, 0 on failure.
int brain_store_fact(PopCountBrain* brain, const char* subject, const char* relation, const char* object);

// Query an object given a subject and relation.
// EXACT STRING LOOKUP against the in-RAM index only -- the neural/VSA engine is
// not consulted, so this is a dictionary lookup and never a fuzzy match.
// Returns 1 if found and fills out_object (null-terminated), 0 if not found.
int brain_query_relation(PopCountBrain* brain, const char* subject, const char* relation, char* out_object, size_t max_len);

// Verify consistency: returns 1 if proposed statement contradicts stored knowledge, 0 otherwise.
// Also an exact-string check against the in-RAM index; the engine is not consulted.
int brain_check_conflict(PopCountBrain* brain, const char* subject, const char* relation, const char* object);

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
void brain_get_stats(PopCountBrain* brain, size_t* out_active_pages, size_t* out_active_rules);

#ifdef __cplusplus
}
#endif

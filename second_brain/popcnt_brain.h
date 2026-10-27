#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

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
//       2. The 4 KB hardware-aligned VSA rule bank (`bank_store_rule` /
//          `bank_query_rule`), indexed by open-addressed multi-page buckets
//          for subject owner, relation owner, and joint (subject, relation)
//          512-bit SimHash keys, backed by `assoc_facts` for string decoding.
//   * Query semantics: exact raw-string match first. On a miss, `assoc_lookup`
//     queries the 4 KB VSA rule bank via `bank_query_rule` (probing all
//     open-addressed overflow pages for the owner bucket) and resolves the
//     matched 512-bit object vector (`val_bits`) against `assoc_facts` when
//     one field matches after ASCII lowercase + whitespace normalization and
//     the other clears `dot >= 200`, or when both fields are >= 6 chars and
//     clear `dot >= 320` on the joint key.
//     MEASURED (measur/assoc_calib.cpp, 300-fact populations):
//       - 1-char typo in the subject, exact relation: 99.0% (distinct words),
//         100% (shared-prefix template);
//       - 1-char typo in the relation, exact subject: 85.7% (280 queries over
//         40 subjects x 7 relations), 0 wrong-relation outcomes;
//       - case/whitespace variants: 100%;
//       - unrelated queries and exact-subject+bogus-relation: 0 false hits in
//         640 probes.
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
// Writes the triple into the in-RAM exact index, into the open-addressed 4 KB
// VSA rule bank (`bank_store_rule`), and feeds it to the neural engine.
// Returns 1 on success, 0 on failure.
POPCNT_BRAIN_API int brain_store_fact(PopCountBrain* brain, const char* subject, const char* relation, const char* object);

// Query an object given a subject and relation. Exact raw-string lookup first;
// on a miss, `assoc_lookup` probes the 4 KB VSA rule bank (`bank_query_rule`)
// and resolves near-misses (case/whitespace variants, single-character typos
// within the measured threshold).
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
    double backprop_initial_loss;
    double backprop_final_loss;
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

// =============================================================================
// Adapted Continuous Representation Spaces, 32-Layer Non-Linear Hierarchy &
// Contrastive Riemannian SGD Manifold (512-D Manifold)
//
// Projects tokens, phrases, and multi-layer recurrent context trajectories into
// a 512-dimensional continuous representation manifold coupled with the engine's
// 4-bitplane INT8 + sign POPCNT hardware representation:
//   * 32-Layer Hierarchical Non-Linear Residual Stack (NUM_DEEP_LAYERS = 32):
//     Layers 0..7 encode morpho-syntactic features, Layers 8..19 bind semantic
//     roles and entities, and Layers 20..31 perform abstract multi-hop world-model
//     composition with SwiGLU-style non-linear gating and RMSNorm.
//   * Multi-Epoch Contrastive & Predictive Riemannian SGD in R^512 on S^{511}
//     (SGD_MANIFOLD_EPOCHS = 6): optimizes analytical cosine-margin gradients
//     across functional equivalence frames and diagonal transition operators.
//   * Partitioned Holographic Superposition (HolographicSuperpositionBank, 64
//     orthogonal cells) with Iterative Explaining-Away Clean-Up (Successive
//     Interference Cancellation) + IDF-Whitened Zero-Mean VSA Bundling: eliminates
//     D=512 superposition crosstalk noise (C <= D / (2 ln N)).
//   * Multi-Hop Transitive & Taxonomic Deduction (2..3 hops): resolves chained
//     relational and taxonomic queries in brain_query_relation, brain_check_conflict,
//     and brain_generate via deductive closure and 3-hop discourse propagation.
// =============================================================================

// Project a UTF-8 text or code snippet into the adapted 512-dimensional unit-L2
// continuous representation space (writes 512 floats to out_vec512).
// Returns 1 on success, 0 on failure.
POPCNT_BRAIN_API int brain_embed_continuous(
    PopCountBrain* brain,
    const char* text,
    float* out_vec512
);

// Compute AVX-512 / AVX2 cosine similarity in [-1.0f, +1.0f] between two UTF-8
// texts or identifiers in the adapted 512-D continuous representation space.
// Returns 0.0f on invalid arguments.
POPCNT_BRAIN_API float brain_continuous_similarity(
    PopCountBrain* brain,
    const char* text_a,
    const char* text_b
);

// Perform deep geodesic SLERP + attractor manifold interpolation between text_a
// and text_b at interpolation ratio alpha in [0.0f, 1.0f].
// - out_vec512 (optional, may be NULL): receives the 512-D unit-norm interpolated vector.
// - out_nearest_token (optional, may be NULL if max_len == 0): receives the nearest
//   vocabulary concept / token on the continuous manifold trajectory.
// Returns 1 on success, 0 on failure.
POPCNT_BRAIN_API int brain_interpolate_continuous(
    PopCountBrain* brain,
    const char* text_a,
    const char* text_b,
    float alpha,
    float* out_vec512,
    char* out_nearest_token,
    size_t max_len
);

// =============================================================================
// 1. Foundation Continuous Manifold via End-to-End Analytical Backpropagation,
//    Semantic Paraphrasing & Out-of-Distribution (OOD) Open-Ended Extrapolation
// =============================================================================

// Train the continuous multi-layer hidden manifold and 4096-D dense embeddings
// via end-to-end analytical backpropagation (cross-entropy + cosine alignment
// gradients through output softmax, SwiGLU/GELU hidden layers, attention pooling,
// and continuous token representations).
// If corpus_text is NULL or empty, trains on the currently loaded corpus.
// Writes initial and final analytical cross-entropy loss to out_initial_loss and
// out_final_loss (if non-NULL). Returns 1 on success, 0 on failure.
POPCNT_BRAIN_API int brain_train_foundation_backprop(
    PopCountBrain* brain,
    const char* corpus_text,
    int epochs,
    float learning_rate,
    float* out_initial_loss,
    float* out_final_loss
);

// Generate a fluent, meaning-preserving paraphrase of input_text by projecting
// the sequence onto the backprop-trained 4096-D continuous manifold and decoding
// high-cosine-similarity semantic equivalents validated by the continuous
// transition network and morpho-syntactic agreement engine.
// Returns the number of tokens (> 0) written to out_text on success, 0 on failure.
POPCNT_BRAIN_API int brain_paraphrase_continuous(
    PopCountBrain* brain,
    const char* input_text,
    char* out_text,
    size_t max_len,
    float temperature
);

// =============================================================================
// 2. Continuous Function Approximation (Fractional Power / Fourier Phase VSA) &
//    Template-Free Relational Resonance
// =============================================================================

// Learn a continuous non-linear function y = f(x) (f: R^{x_dim} -> R^{y_dim})
// inside the VSA engine using Locality-Preserving Fractional Power Encoding
// (Random Fourier Phase hypervectors in R^4096 coupled with 512-bit VSA sketches)
// and continuous kernel-ridge / local tangent operator resonance.
// x_samples has shape [num_samples * x_dim], y_samples has shape [num_samples * y_dim].
// Returns 1 on success, 0 on failure.
POPCNT_BRAIN_API int brain_learn_continuous_function(
    PopCountBrain* brain,
    const char* func_name,
    const float* x_samples,
    const float* y_samples,
    size_t num_samples,
    size_t x_dim,
    size_t y_dim
);

// Evaluate a learned continuous function y = f(x_query) via VSA phase-kernel
// resonance without requiring any discrete lookup hit or syntactic template.
// Returns 1 on success and writes y_dim floats to out_y, 0 on failure.
POPCNT_BRAIN_API int brain_eval_continuous_function(
    PopCountBrain* brain,
    const char* func_name,
    const float* x_query,
    size_t x_dim,
    float* out_y,
    size_t y_dim
);

// Infer the target entity for query_subject from an unstructured text corpus
// that contains implicit relational co-occurrences without any fixed syntactic
// template, using 4096-D continuous relational displacement vectors and VSA
// resonance bundling.
// Returns 1 on success and writes null-terminated target to out_target, 0 on failure.
POPCNT_BRAIN_API int brain_infer_template_free_relation(
    PopCountBrain* brain,
    const char* unstructured_corpus,
    const char* query_subject,
    char* out_target,
    size_t max_len
);

// =============================================================================
// 3. 4096-Dimensional Dense Float32 Embeddings & SIMD FMA Cosine Similarity
//    (Sub-Bit Gradient Semantic Nuance Beyond Discrete 512-Bit Hamming)
// =============================================================================
#define BRAIN_DENSE_4096_DIM 4096

// Project a UTF-8 text or code snippet into a unit-L2 4096-dimensional dense
// float32 embedding (8 orthogonal 512-D spectral/distributional/backprop-refined
// subspaces), writing 4096 floats (16 KB) to out_vec4096.
// Returns 1 on success, 0 on failure.
POPCNT_BRAIN_API int brain_embed_dense_4096(
    PopCountBrain* brain,
    const char* text,
    float* out_vec4096
);

// Compute AVX-512 / AVX2+FMA cosine similarity in [-1.0f, +1.0f] between two
// UTF-8 texts in the 4096-dimensional dense float32 embedding space.
POPCNT_BRAIN_API float brain_cosine_similarity_4096(
    PopCountBrain* brain,
    const char* text_a,
    const char* text_b
);

// Compute AVX-512 / AVX2+FMA cosine similarity in [-1.0f, +1.0f] directly
// between two arbitrary 4096-dimensional dense float32 vectors.
POPCNT_BRAIN_API float brain_cosine_vec_4096(
    const float* vec_a_4096,
    const float* vec_b_4096
);

// =============================================================================
// Distributional OOV Output Priors (Held-Out Target Reachability) &
// Large-Scale 1.0 GB Sparse Bank Fragmentation / AVX-512 Cache-Miss Profiler
// =============================================================================
typedef struct BrainScaleAndHeldOutProfile {
    // Held-out OOV / Unseen Target Reachability Metrics (Strictly Causal Left-to-Right)
    size_t heldout_total_transitions;
    size_t heldout_oov_targets;
    size_t heldout_unseen_target_tokens;
    size_t heldout_unreachable_without_prior;
    size_t heldout_unreachable_with_prior;
    double heldout_unreachable_before_pct;      // Unreachable % without Distributional OOV Output Priors (the 20.6-28.1% wall)
    double heldout_unreachable_after_pct;       // Unreachable % WITH Distributional OOV Output Priors + Role-Cluster Backoff
    double heldout_reachability_after_pct;      // Reachable target/role coverage % on held-out text (100 - heldout_unreachable_after_pct)
    double heldout_accuracy_pct;                // End-to-end causal top-1 prediction accuracy on held-out text
    double raw_ngram_unreachable_percent;       // Alias of heldout_unreachable_before_pct
    double prior_unreachable_percent;           // Alias of heldout_unreachable_after_pct
    double heldout_shortlist_coverage_percent;  // Alias of heldout_reachability_after_pct
    double heldout_top1_accuracy_percent;       // Alias of heldout_accuracy_pct

    // Large-Scale 1.0 GB Sparse Bank Fragmentation & Page Load Metrics
    size_t bank_total_capacity_pages;           // 262,144 pages
    size_t bank_virtual_bytes;                  // 1,073,741,824 bytes (262,144 x 4 KB expert pages)
    size_t bank_active_pages;                   // Resident non-zero 4 KB expert pages touched
    size_t active_expert_pages;                 // Alias of bank_active_pages
    size_t bank_resident_bytes;                 // Physical working-set bytes of active 4 KB pages
    size_t active_resident_bytes;               // Alias of bank_resident_bytes
    size_t bank_total_stored_slots;             // Total VSA/STE slot rules stored across active pages
    size_t total_stored_associations;           // Alias of bank_total_stored_slots
    double mean_slots_per_active_page;          // Mean occupancy out of 16 slots per 4 KB page
    size_t max_slots_per_page;                  // Max occupancy (bounded <= 16, over_16slots == 0)
    size_t pages_over_16_slots;                 // Strictly 0 (collision-free open addressing)
    double mean_open_address_probes;            // Mean open-addressing probe step distance
    double mean_open_address_probe_hops;        // Alias of mean_open_address_probes
    size_t max_open_address_probes;             // Max open-addressing probe step distance
    uint32_t max_open_address_probe_hops;       // Alias of max_open_address_probes
    double sparse_page_span_ratio;              // Span ratio across touched pages
    double fragmentation_index;                 // Non-contiguous page transition ratio across the 1.0 GB space in [0.0, 1.0]
    double sparse_fragmentation_index;          // Alias of fragmentation_index

    // AVX-512 Cache-Hierarchy Sweep (L1D -> L2 -> L3 -> DRAM across 4 KB Expert Pages)
    double avx512_l1_page_eval_ns;              // 8 pages (32 KB, L1D cache resident)
    double avx512_l2_page_eval_ns;              // 128 pages (512 KB, L2 cache resident)
    double avx512_l3_page_eval_ns;              // 4,096 pages (16 MB, L3 cache resident)
    double avx512_dram_page_eval_ns;            // 16,384 pages (64 MB, beyond L3 / DRAM sparse bank sweep)
    double avx512_l1_ns_per_page;               // Alias of avx512_l1_page_eval_ns
    double avx512_l2_ns_per_page;               // Alias of avx512_l2_page_eval_ns
    double avx512_l3_ns_per_page;               // Alias of avx512_l3_page_eval_ns
    double avx512_dram_ns_per_page;             // Alias of avx512_dram_page_eval_ns
} BrainScaleAndHeldOutProfile;

// Evaluate held-out text (containing unseen OOV targets, morphological inflections, and novel
// contexts) using strictly causal Distributional & Morphological OOV Output Priors, and profile
// 1.0 GB sparse bank fragmentation and AVX-512 cache-hierarchy latency from L1D (32 KB) to DRAM (64 MB).
// Returns true (1) on success, false (0) on invalid arguments.
POPCNT_BRAIN_API bool brain_profile_heldout_and_scale(
    PopCountBrain* brain,
    const char* heldout_text,
    BrainScaleAndHeldOutProfile* out_profile
);

#ifdef __cplusplus
} // extern "C"

static inline int brain_profile_heldout_and_scale(
    PopCountBrain* brain,
    const char* heldout_text,
    size_t cache_sweep_pages,
    BrainScaleAndHeldOutProfile* out_profile
) {
    (void)cache_sweep_pages;
    return brain_profile_heldout_and_scale(brain, heldout_text, out_profile) ? 1 : 0;
}
#endif

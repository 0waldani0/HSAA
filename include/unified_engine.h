#ifndef UNIFIED_ENGINE_H
#define UNIFIED_ENGINE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "hssa.h"
#include "vsa_db.h"
#include "model_mod.h"
#include "bridge.h"
#include "popcnt_brain.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * =============================================================================
 * UNIFIED COGNITIVE ENGINE (HSSA v2.0 + SecondBrain + VSA DB + ModelMod)
 * =============================================================================
 * Fuses the formerly separate subsystems into ONE single-pass cognitive pipeline
 * operating over a shared latent state:
 *   - Continuous state h_t in R^{1024} & manifold embedding in R^{4096}
 *   - Dual-rail 512-bit hypervector projection psi(h_t) in {0,1}^{512}
 *   - 28-layer hybrid stack (Gated DeltaNet O(1) recurrence + GQA/RoPE + TT16)
 *   - 4KB Sparse Micro-Expert Cortex + Foundation Continuous Manifold
 *   - Shared VsaDbHandle associative memory + Continuous Function Approximator
 *   - Continuous Hopfield Attractor Relaxation (dE/dt <= 0)
 *   - Attractor Simplex Contradiction Gate (p* = 0.0 for conflicting tokens)
 * =============================================================================
 */

typedef struct UnifiedEngine UnifiedEngine;

typedef enum UnifiedInputKind {
    UNIFIED_KIND_PROSE_OR_CODE = 0,
    UNIFIED_KIND_RELATIONAL_FACT = 1,
    UNIFIED_KIND_CONTINUOUS_FUNC = 2,
    UNIFIED_KIND_RELATIONAL_QUERY = 3,
    UNIFIED_KIND_FUNC_EVAL_QUERY = 4,
    UNIFIED_KIND_CODE_SYNTHESIS = 5,
    UNIFIED_KIND_IDEA_SYNTHESIS = 6
} UnifiedInputKind;

typedef struct UnifiedEngineConfig {
    uint32_t num_layers;           /* Default: 28 */
    uint32_t vsa_capacity_mb;      /* Default: VSA_CAP_MICRO (4 MB L3-resident) */
    uint32_t sparse_bank_pages;    /* Default: 16384 pages (64 MB) or 262144 (1 GB) */
    float    tt16_learning_rate;   /* Default: 0.015f */
    float    backprop_lr;          /* Default: 0.08f */
    uint32_t backprop_epochs;      /* Default: 8 */
} UnifiedEngineConfig;

typedef struct UnifiedObserveTelemetry {
    UnifiedInputKind detected_kind;
    double           total_observe_us;
    double           tt16_update_us;
    bool             conflict_detected;
    uint32_t         facts_stored_total;
    uint32_t         dense_vectors_total;
    float            manifold_loss_before;
    float            manifold_loss_after;
} UnifiedObserveTelemetry;

typedef struct UnifiedForwardTelemetry {
    UnifiedInputKind resolved_mode;
    double           total_forward_us;
    double           hssa_stack_us;
    double           vsa_recall_us;
    double           hopfield_relax_us;
    float            vsa_recall_similarity;
    double           hopfield_energy_initial;
    double           hopfield_energy_final;
    double           hopfield_energy_delta;
    uint32_t         contradictions_nullified;
    float            state_l2_norm;
    float            manifold_4096_norm;
} UnifiedForwardTelemetry;

/* Telemetry for Deep 28-Layer HSSA+ModelMod Multi-Beam + RoPE Text Generation */
typedef struct UnifiedTextGenTelemetry {
    uint32_t generated_tokens;
    uint32_t beam_candidates_evaluated;
    uint32_t subword_stems_matched;
    uint32_t contradictions_repaired;
    float    hssa_alignment_cosine;
    double   max_long_range_rope_attn;
    double   hopfield_energy_delta;
    double   total_gen_us;
} UnifiedTextGenTelemetry;

/* Telemetry for AST-Guided Code & Continuous Function-to-Code Synthesis + Sandbox */
typedef struct UnifiedCodeSynthTelemetry {
    uint32_t tokens_emitted;
    uint32_t ast_nodes_verified;
    uint32_t max_nesting_depth;
    uint32_t sandbox_test_cases_passed;
    bool     ast_balanced_valid;
    bool     sandbox_exec_verified;
    bool     compiled_from_vsa_function;
    double   sandbox_mse;
    double   total_synth_us;
} UnifiedCodeSynthTelemetry;

/* Telemetry for System-2 Cross-Domain VSA Analogy & Intrinsic Curiosity Idea Discovery */
typedef struct UnifiedIdeaSynthTelemetry {
    uint32_t analogies_transferred;
    uint32_t curiosity_gaps_filled;
    uint32_t candidates_explored;
    uint32_t contradictions_rejected;
    uint32_t crystallized_facts;
    float    manifold_novelty_score;
    float    analogy_structural_cosine;
    double   hopfield_energy_delta;
    double   total_idea_us;
} UnifiedIdeaSynthTelemetry;

/* 1. Lifecycle Management */
void unified_engine_default_config(UnifiedEngineConfig* out_cfg);
UnifiedEngine* unified_engine_create(const UnifiedEngineConfig* cfg);
void unified_engine_destroy(UnifiedEngine* eng);
int unified_engine_reset_state(UnifiedEngine* eng);

/* 2. Unified Single-Pass Observation (Ingests Facts, Functions, or Text/Code Corpus) */
int unified_engine_observe(
    UnifiedEngine* eng,
    const char* input,
    UnifiedObserveTelemetry* out_telem
);

/* 3. Unified Single-Pass Forward / Reasoning / Generation */
int unified_engine_forward(
    UnifiedEngine* eng,
    const char* query_or_prompt,
    char* out_response,
    size_t max_len,
    float* out_state_1024,
    UnifiedForwardTelemetry* out_telem
);

/* 4. Unified Contradiction-Free Candidate Simplex Gate */
int unified_engine_register_candidate(
    UnifiedEngine* eng,
    uint32_t token_id,
    const char* subj,
    const char* rel,
    const char* obj
);

int unified_engine_project_simplex(
    UnifiedEngine* eng,
    const float* raw_logits,
    size_t vocab_size,
    float* out_probs,
    uint32_t* out_nullified_count
);

/* 5. Deep 28-Layer HSSA LM-Head + Multi-Beam + Long-Range RoPE Text Generation */
int unified_engine_generate_text_deep(
    UnifiedEngine* eng,
    const char* prompt,
    char* out_text,
    size_t max_len,
    uint32_t max_tokens,
    uint32_t beam_width,
    UnifiedTextGenTelemetry* out_telem
);

/* 6. AST-Guided Code & Mathematical Function-to-Code Synthesis + Sandbox Verification */
int unified_engine_synthesize_code(
    UnifiedEngine* eng,
    const char* spec_or_func_prompt,
    char* out_code,
    size_t max_len,
    UnifiedCodeSynthTelemetry* out_telem
);

/* 7. System-2 Counterfactual Imagination, Cross-Domain VSA Analogy & Curiosity Discovery */
int unified_engine_synthesize_ideas(
    UnifiedEngine* eng,
    const char* domain_or_goal_prompt,
    bool crystallize_to_memory,
    char* out_ideas,
    size_t max_len,
    UnifiedIdeaSynthTelemetry* out_telem
);

/* 8. Unified Persistence (Single Call Saves/Loads Entire Engine State) */
int unified_engine_save(UnifiedEngine* eng, const char* path_prefix);
int unified_engine_load(UnifiedEngine* eng, const char* path_prefix);

/* 9. Direct Component Accessors for Interop with Existing Subsystems */
VsaDbHandle*   unified_engine_get_vsa_db(UnifiedEngine* eng);
PopCountBrain* unified_engine_get_brain(UnifiedEngine* eng);
HssaContext*   unified_engine_get_hssa(UnifiedEngine* eng);
void*          unified_engine_get_model(UnifiedEngine* eng);
const float*   unified_engine_get_state_1024(const UnifiedEngine* eng);
const float*   unified_engine_get_manifold_4096(const UnifiedEngine* eng);

#ifdef __cplusplus
}
#endif

#endif // UNIFIED_ENGINE_H

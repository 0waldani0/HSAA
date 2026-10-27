#ifndef VSA_DB_H
#define VSA_DB_H

#pragma once
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Dynamic Non-Fixed Memory & Knowledge Base Engine (VsaMemoryDatabase)
// Pure Native C++ / AVX-512 VSA Associative Hyperdimensional Computing
// =============================================================================

#define VSA_DIM 512
#define VSA_WORDS (VSA_DIM / 64)       // 8 x uint64_t = 64 bytes (1 cache line / __m512i)
#define VSA_DENSE_4096_DIM 4096        // 4096-D dense float32 embedding (16 KB)

typedef struct VsaDbHandle VsaDbHandle;

enum VsaDbCapacity {
    VSA_CAP_MICRO  = 1024,    // 1K pages = 4MB (L3-cache resident)
    VSA_CAP_MEDIUM = 16384,   // 16K pages = 64MB
    VSA_CAP_LARGE  = 262144   // 262K pages = 1GB
};

// Lifecycle Management
VsaDbHandle* vsa_db_create(VsaDbCapacity cap, const char* persist_path);
void vsa_db_destroy(VsaDbHandle* db);
int vsa_db_flush(VsaDbHandle* db);
int vsa_db_clone(const VsaDbHandle* src, VsaDbHandle** dst, const char* new_persist_path);

// Online Mutation & Retrieval
// insert_fact also registers the fact's 512-bit binding (XNOR(rel_key, obj))
// in the unified vector table under key hash(subject, relation), so a stored
// fact is reachable by vector distance as well as by exact strings. Deleting a
// fact removes that binding; mutating reuses the string-heap block when it
// fits and compacts the heap when it does not.
int vsa_db_insert_fact(VsaDbHandle* db, const char* subj, const char* rel, const char* obj);
int vsa_db_delete_fact(VsaDbHandle* db, const char* subj, const char* rel);
int vsa_db_mutate_fact(VsaDbHandle* db, const char* subj, const char* rel, const char* new_obj);
int vsa_db_query_relation(VsaDbHandle* db, const char* subj, const char* rel, char* out_obj, size_t max_len);
int vsa_db_check_conflict(VsaDbHandle* db, const char* subj, const char* rel, const char* obj);

// Raw 512-bit Vector Operations (Argmax Cosine Scan)
// insert_vector upserts by key_hash. query_vector scans the active vectors
// only and applies a resonance gate (normalized Hamming similarity >= 0.40);
// below it the function returns 0 and leaves *out_sim at 0.0 instead of
// returning an arbitrary "closest" slot from a non-empty table. The gate is
// calibrated: independent balanced 512-bit vectors score 0 +/- 0.044, so 0.40
// is ~9 sigma; an all-zero probe against a 180-ones vector scores 0.297 and is
// still a miss (pinned by tests/test_vsa_db.cpp).
int vsa_db_insert_vector(VsaDbHandle* db, uint64_t key_hash, const uint64_t* vec_512);
int vsa_db_query_vector(VsaDbHandle* db, const uint64_t* query_512, uint64_t* out_best_vec, float* out_sim);

// 4096-Dimensional Dense Float32 Embeddings & Cosine Similarity (Bridging Discrete Hamming Discreteness)
// Computes exact float32 cosine similarity in [-1.0, +1.0] across 4096 dimensions using SIMD FMA.
float vsa_cosine_sim_4096_f32(const float* a_4096, const float* b_4096);
int vsa_db_insert_dense_4096(VsaDbHandle* db, uint64_t key_hash, const float* vec_4096);
int vsa_db_query_dense_4096(
    VsaDbHandle* db,
    const float* query_4096,
    uint64_t* out_best_key,
    float* out_best_vec_4096,
    float* out_cosine_sim
);

// Continuous VSA Function Approximation (Fractional Power / Random Fourier Phase Encoding)
// Approximates continuous functions y = f(x) without rigid syntactic templates.
int vsa_db_bind_continuous_point(VsaDbHandle* db, const char* relation_name, float x_val, float y_val);
int vsa_db_eval_continuous_function(VsaDbHandle* db, const char* relation_name, float x_query, float* out_y);

// Telemetry & Introspection Helpers
// The count getters take the database mutex; persist_path/capacity are
// immutable after create and are safe to read without it.
VsaDbCapacity vsa_db_get_capacity(const VsaDbHandle* db);
size_t vsa_db_get_fact_count(const VsaDbHandle* db);
size_t vsa_db_get_vector_count(const VsaDbHandle* db);
size_t vsa_db_get_active_pages(const VsaDbHandle* db);
const char* vsa_db_get_persist_path(const VsaDbHandle* db);

#ifdef __cplusplus
}
#endif

#endif // VSA_DB_H

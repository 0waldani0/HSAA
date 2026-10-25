// Native end-to-end test: links the real modules and exercises the public
// C-ABI across the layers an integrator uses.
//
// The previous "E2E suite" was a set of source-text checks (does build.bat
// contain the word "clean"? does CMakeLists.txt contain "20"?) plus a mock
// harness; it never executed the engine. This runner replaces it with the
// actual pipeline:
//
//   1. VsaDbHandle: fact CRUD, conflict detection, raw-vector resonance;
//   2. PopCountBrain C-ABI: exact, case-normalized and conflict queries;
//   3. HSSA: deterministic forward, and a recalled vector measured in the
//      output (the residual blend in relax_hopfield_attractor);
//   4. model_mod: layer forward, runtime layer swap, atomic getters;
//   5. bench_runner: the step profiler must report accuracy as N/A.
//
// Any failed check makes the process exit non-zero.

#include "vsa_db.h"
#include "popcnt_brain.h"
#include "hssa.h"
#include "model_mod.h"
#include "bench_runner.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

static int g_run = 0;
static int g_failed = 0;

#define E2E_CHECK(cond, msg)                                                    \
    do {                                                                        \
        ++g_run;                                                                \
        if (cond) {                                                             \
            std::printf("[PASS] %s\n", (msg));                                  \
        } else {                                                                \
            ++g_failed;                                                         \
            std::printf("[FAIL] %s  (%s:%d)\n", (msg), __FILE__, __LINE__);     \
        }                                                                       \
    } while (0)

static double rms(const float* v, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += (double)v[i] * (double)v[i];
    return std::sqrt(s / (double)n);
}

static bool all_finite(const float* v, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(v[i])) return false;
    }
    return true;
}

static double relative_l2_diff(const float* a, const float* b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)a[i] * (double)a[i];
    }
    if (den <= 1e-30) return 0.0;
    return std::sqrt(num / den);
}

static void test_vsa_db(void) {
    std::printf("\n=== 1. VsaDbHandle: facts, conflicts, vector resonance ===\n");
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    E2E_CHECK(db != nullptr, "vsa_db_create");
    if (!db) return;

    char out[64];
    E2E_CHECK(vsa_db_insert_fact(db, "alice", "lives_in", "paris") == 0,
              "insert_fact returns 0");
    E2E_CHECK(vsa_db_query_relation(db, "alice", "lives_in", out, sizeof(out)) == 1 &&
                  std::strcmp(out, "paris") == 0,
              "query_relation returns the stored object");
    E2E_CHECK(vsa_db_check_conflict(db, "alice", "lives_in", "paris") == 0,
              "identical fact is not a conflict");
    E2E_CHECK(vsa_db_check_conflict(db, "alice", "lives_in", "rome") == 1,
              "different object is a conflict");
    E2E_CHECK(vsa_db_mutate_fact(db, "alice", "lives_in", "rome") == 0,
              "mutate_fact returns 0");
    E2E_CHECK(vsa_db_query_relation(db, "alice", "lives_in", out, sizeof(out)) == 1 &&
                  std::strcmp(out, "rome") == 0,
              "mutation is visible");
    E2E_CHECK(vsa_db_delete_fact(db, "alice", "lives_in") == 1,
              "delete_fact returns 1");
    E2E_CHECK(vsa_db_query_relation(db, "alice", "lives_in", out, sizeof(out)) == 0,
              "deleted fact is gone");
    E2E_CHECK(vsa_db_get_fact_count(db) == 0, "fact count returns to zero");
    vsa_db_destroy(db);

    // A fresh database for the vector gate: no fact bindings can interfere.
    VsaDbHandle* vdb = vsa_db_create(VSA_CAP_MICRO, nullptr);
    E2E_CHECK(vdb != nullptr, "vsa_db_create (vector db)");
    if (!vdb) return;

    uint64_t a[VSA_WORDS] = {
        0x1234567890ABCDEFULL, 0xFEDCBA0987654321ULL,
        0x0F0F0F0F0F0F0F0FULL, 0xF0F0F0F0F0F0F0F0ULL,
        0xAAAAAAAA55555555ULL, 0x55555555AAAAAAAAULL,
        0xCCCCCCCC33333333ULL, 0x33333333CCCCCCCCULL
    };
    uint64_t near_a[VSA_WORDS];
    std::memcpy(near_a, a, sizeof(a));
    near_a[0] ^= 1ULL;  // one flipped bit: similarity 510/512 = 0.996
    uint64_t zero[VSA_WORDS] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t best[VSA_WORDS];
    float sim = -1.0f;

    E2E_CHECK(vsa_db_insert_vector(vdb, 1001, a) == 0, "insert_vector returns 0");
    E2E_CHECK(vsa_db_query_vector(vdb, near_a, best, &sim) == 1 && sim > 0.99f,
              "one-bit-flipped probe resonates (sim > 0.99)");
    sim = -1.0f;
    E2E_CHECK(vsa_db_query_vector(vdb, zero, best, &sim) == 0 && sim == 0.0f,
              "unrelated probe misses and reports 0.0");
    vsa_db_destroy(vdb);
}

static void test_brain_abi(void) {
    std::printf("\n=== 2. PopCountBrain C-ABI ===\n");
    std::remove("build/e2e_brain.bin");
    PopCountBrain* b = brain_create(4096, "build/e2e_brain.bin");
    E2E_CHECK(b != nullptr, "brain_create");
    if (!b) return;

    char out[128];
    E2E_CHECK(brain_store_fact(b, "entity_alice", "lives_in", "paris") == 1,
              "brain_store_fact returns 1");
    E2E_CHECK(brain_query_relation(b, "entity_alice", "lives_in", out, sizeof(out)) == 1 &&
                  std::strcmp(out, "paris") == 0,
              "exact query resolves");
    E2E_CHECK(brain_query_relation(b, "ENTITY_ALICE", "lives_in", out, sizeof(out)) == 1 &&
                  std::strcmp(out, "paris") == 0,
              "case variant resolves through the normalized lookup");
    E2E_CHECK(brain_check_conflict(b, "entity_alice", "lives_in", "paris") == 0,
              "identical statement is not a conflict");
    E2E_CHECK(brain_check_conflict(b, "entity_alice", "lives_in", "rome") == 1,
              "contradictory statement is a conflict");
    E2E_CHECK(brain_query_relation(b, "no_such_entity_zzzz", "no_such_relation",
                                   out, sizeof(out)) == 0,
              "unrelated query misses");

    brain_destroy(b);
    std::remove("build/e2e_brain.bin");
}

static void test_hssa(void) {
    std::printf("\n=== 3. HSSA: determinism and recall in the output ===\n");
    HssaContext* h = hssa_create(4);
    E2E_CHECK(h != nullptr, "hssa_create(4)");
    if (!h) return;

    alignas(64) float in[HSSA_HIDDEN_DIM];
    alignas(64) float a[HSSA_HIDDEN_DIM];
    alignas(64) float b[HSSA_HIDDEN_DIM];
    alignas(64) float recalled[HSSA_HIDDEN_DIM];
    for (size_t i = 0; i < HSSA_HIDDEN_DIM; ++i) {
        in[i] = (float)std::sin((double)i * 0.013);
    }

    E2E_CHECK(hssa_forward_token(h, in, a, nullptr) == 0, "forward_token returns 0");
    E2E_CHECK(all_finite(a, HSSA_HIDDEN_DIM) && rms(a, HSSA_HIDDEN_DIM) > 1e-6,
              "forward output is finite and non-zero");

    // The post-SSM state is binarized into scratch_vsa; capture it before the
    // next forward overwrites it, so the recalled vector is a perfect match.
    alignas(64) uint64_t query[HSSA_VSA_WORDS];
    dual_rail_binarize(h->scratch_in, query);

    hssa_reset_state(h);
    E2E_CHECK(hssa_forward_token(h, in, a, nullptr) == 0, "forward after reset");
    hssa_reset_state(h);
    E2E_CHECK(hssa_forward_token(h, in, b, nullptr) == 0, "forward after second reset");
    E2E_CHECK(std::memcmp(a, b, sizeof(a)) == 0,
              "forward is bitwise deterministic after reset");

    // Empty database baseline, then the same pass with a resonant vector.
    hssa_reset_state(h);
    E2E_CHECK(hssa_forward_token(h, in, a, nullptr) == 0, "empty-DB baseline");

    VsaDbHandle* rdb = vsa_db_create(VSA_CAP_MICRO, nullptr);
    E2E_CHECK(rdb != nullptr, "recall db create");
    if (rdb) {
        E2E_CHECK(vsa_db_insert_vector(rdb, 777, query) == 0, "store recalled vector");
        hssa_set_global_vsa_db(h, rdb);
        hssa_reset_state(h);
        E2E_CHECK(hssa_forward_token(h, in, recalled, nullptr) == 0, "forward with recall");
        double d = relative_l2_diff(a, recalled, HSSA_HIDDEN_DIM);
        std::printf("       recall-induced relative L2 difference: %.6f\n", d);
        E2E_CHECK(d > 1e-3,
                  "recalled vector measurably reaches the output (residual blend)");
    }

    hssa_destroy(h);
    if (rdb) vsa_db_destroy(rdb);
}

static void test_model_mod(void) {
    std::printf("\n=== 4. model_mod: forward, swap, atomic getters ===\n");
    ModelContext* m = model_create_context(2);
    E2E_CHECK(m != nullptr, "model_create_context(2)");
    if (!m) return;

    alignas(64) float in[MODEL_DEFAULT_DIM_N];
    alignas(64) float out[MODEL_DEFAULT_DIM_M];
    for (size_t i = 0; i < MODEL_DEFAULT_DIM_N; ++i) {
        in[i] = (float)std::cos((double)i * 0.021);
    }

    E2E_CHECK(model_forward_layer(m, 0, in, out) == 0, "layer 0 forward (GQA)");
    E2E_CHECK(all_finite(out, MODEL_DEFAULT_DIM_M), "GQA output is finite");

    // Verify in-place aliasing (in_vec == out_vec) matches out-of-place execution
    // even when TT16 residual cores are active.
    alignas(64) float u_delta[MODEL_TT16_CORE_SIZE];
    alignas(64) float v_delta[MODEL_TT16_CORE_SIZE];
    for (size_t i = 0; i < MODEL_TT16_CORE_SIZE; ++i) {
        u_delta[i] = 0.01f * (float)std::sin((double)i * 0.07);
        v_delta[i] = 0.01f * (float)std::cos((double)i * 0.05);
    }
    E2E_CHECK(model_update_tt16_residual(m, 0, u_delta, v_delta) == 0, "update TT16 on layer 0");
    alignas(64) float out_ref[MODEL_DEFAULT_DIM_M];
    alignas(64) float inplace_buf[MODEL_DEFAULT_DIM_M];
    std::memcpy(inplace_buf, in, sizeof(in));
    E2E_CHECK(model_forward_layer(m, 0, in, out_ref) == 0, "GQA + TT16 out-of-place forward");
    E2E_CHECK(model_forward_layer(m, 0, inplace_buf, inplace_buf) == 0, "GQA + TT16 in-place forward");
    E2E_CHECK(std::memcmp(out_ref, inplace_buf, sizeof(out_ref)) == 0,
              "in-place model_forward_layer matches out-of-place bitwise");

    E2E_CHECK(model_set_layer_precision(m, 0, QUANT_3BIT) == 0 &&
                  model_get_layer_precision(m, 0) == QUANT_3BIT,
              "precision store/load is visible");
    E2E_CHECK(model_swap_layer_block(m, 0, LAYER_GATED_DELTANET) == 0 &&
                  model_get_layer_type(m, 0) == LAYER_GATED_DELTANET,
              "layer swap store/load is visible");
    E2E_CHECK(model_forward_layer(m, 0, in, out) == 0, "layer 0 forward (DeltaNet)");
    E2E_CHECK(all_finite(out, MODEL_DEFAULT_DIM_M), "DeltaNet output is finite");
    model_destroy_context(m);
}

static void test_bench_contract(void) {
    std::printf("\n=== 5. bench_runner: no fabricated accuracy ===\n");
    BenchmarkMetrics m;
    std::memset(&m, 0, sizeof(m));
    E2E_CHECK(bench_run_step_profile(nullptr, &m) == 0, "step profile returns 0");
    E2E_CHECK(m.latency_us > 0.0 && m.tokens_per_sec > 0.0,
              "latency and throughput are measured");
    E2E_CHECK(!m.accuracy_available,
              "accuracy is reported unavailable for the FP32 path");
    E2E_CHECK(std::isnan(m.cosine_similarity) && std::isnan(m.sqnr_db),
              "cosine/SQNR are NaN, not synthetic near-perfect values");
}

int main(void) {
    std::printf("NATIVE E2E: public C-ABI across VSA DB, Second Brain, HSSA, model_mod\n");

    test_vsa_db();
    test_brain_abi();
    test_hssa();
    test_model_mod();
    test_bench_contract();

    std::printf("\n=================================================================\n");
    std::printf("E2E Summary: %d Run, %d Passed, %d Failed\n",
                g_run, g_run - g_failed, g_failed);
    std::printf("=================================================================\n");
    return (g_failed == 0) ? 0 : 1;
}

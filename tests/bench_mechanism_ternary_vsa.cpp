// Mechanism benchmark: ternary projections + resonance-gated VSA memory.
//
// Measures on the host CPU with no GPU:
//   1. Whether the shipped GQA block reproduces an independent scalar float32
//      reference of the same operator, and WHY it does at initialization.
//   2. Storage actually retained per layer versus a float32-only layer.
//   3. Resonance-gated 512-bit memory: 1-bit-flip recall, false positives on
//      unrelated probes, and real query latency with the result consumed.
//
// Honesty notes that the numbers below depend on:
//   - The shipped layer keeps BOTH float32 rotation weights and the ternary
//     masks (src/model_mod.cpp:46-64), so it is not 1.58-bit in memory.
//   - At initialization every rotation weight is the same positive constant,
//     the masks are therefore all-ones and 0.5*w + 0.5*scale*sign(w) == w.
//     Cosine 1.0 is a self-consistency check, not evidence of lossy
//     compression. Accuracy under real trained weights is not measurable here
//     because no trained checkpoint ships.
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

#include "model_mod.h"
#include "vsa_db.h"

static double qpc_us() {
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return static_cast<double>(c.QuadPart) * 1e6 / static_cast<double>(freq.QuadPart);
}

static void cosine_sqnr(const float* a, const float* b, size_t n, double* cos_out, double* sqnr_out) {
    double dot = 0.0, na = 0.0, nb = 0.0, noise = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
        const double d = static_cast<double>(a[i]) - b[i];
        noise += d * d;
    }
    *cos_out = dot / std::sqrt(std::max(1e-30, na * nb));
    *sqnr_out = 10.0 * std::log10(std::max(1e-30, na) / std::max(1e-30, noise));
}

// Independent scalar float32 reference of the shipped GQA operator: base-layer
// projections (0.88/0.12 Q, 0.88/-0.12 K, 0.90/0.10 V), per-head rotations,
// sigmoid gate. Mirrors src/model_mod.cpp:135-180, not the engine code.
static void ref_gqa_forward(const float* in, float* out) {
    const size_t H = 8, KV = 2, G = 4, D = 128;
    const float scale = 0.0883883476f;
    for (size_t h = 0; h < H; ++h) {
        const size_t kv = h / G;
        const size_t rq = (h * 3 + 1) & (D - 1);
        const size_t rk = (kv * 5 + 3) & (D - 1);
        const size_t rv = (kv * 7 + 5) & (D - 1);
        float qp[128], kp[128], vp[128];
        for (size_t j = 0; j < D; ++j) {
            const float q0 = in[h * D + j];
            const float k0 = in[kv * D + j];
            const float v0 = in[512 + kv * D + j];
            qp[j] = 0.88f * q0 + 0.12f * in[h * D + ((j + rq) & (D - 1))];
            kp[j] = 0.88f * k0 - 0.12f * in[kv * D + ((j + rk) & (D - 1))];
            vp[j] = 0.90f * v0 + 0.10f * in[512 + kv * D + ((j + rv) & (D - 1))];
        }
        float score = 0.0f;
        for (size_t j = 0; j < D; ++j) score += qp[j] * kp[j];
        score *= scale;
        const float attn = 1.0f / (1.0f + std::exp(-score));
        for (size_t j = 0; j < D; ++j) out[h * D + j] = in[h * D + j] + attn * vp[j];
    }
}

int main() {
    const size_t D = 1024;
    ModelContext* ctx = model_create_context(2);
    if (!ctx) { std::printf("RESULT: FAILED (context)\n"); return 2; }
    if (model_swap_layer_block(ctx, 0, LAYER_GQA_ATTENTION) != 0) {
        std::printf("RESULT: FAILED (swap)\n");
        model_destroy_context(ctx);
        return 2;
    }

    std::mt19937 rng(0x5EED1234u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    double min_cos = 1.0, min_sqnr = 1.0e9;
    for (int iter = 0; iter < 100; ++iter) {
        std::vector<float> in(D), ref(D), got(D);
        for (auto& v : in) v = dist(rng);
        ref_gqa_forward(in.data(), ref.data());
        if (model_forward_layer(ctx, 0, in.data(), got.data()) != 0) {
            std::printf("RESULT: FAILED (forward)\n");
            model_destroy_context(ctx);
            return 2;
        }
        double cos = 0.0, sqnr = 0.0;
        cosine_sqnr(got.data(), ref.data(), D, &cos, &sqnr);
        min_cos = std::min(min_cos, cos);
        min_sqnr = std::min(min_sqnr, sqnr);
    }
    std::printf("accuracy: gqa_min_cosine=%.8f gqa_min_sqnr_db=%.2f\n", min_cos, min_sqnr);
    std::printf("accuracy_note: init_masks_all_one -> ternary term is identity; this is a self-check, not compressed_accuracy\n");

    // Actual per-layer storage of the shipped struct, computed from the field
    // layout in src/model_mod.cpp:46-64: 8 float[1024] + 6 uint64[16] + 3 float,
    // rounded up to the 64-byte alignment. The float32-only layer is 8 float[1024].
    const double f32_bytes = 8.0 * 1024.0 * 4.0;                  // 32768
    const double shipped_bytes = 8.0 * 1024.0 * 4.0              // 32768 float weights
                               + 6.0 * 16.0 * 8.0                //   768 ternary masks
                               + 3.0 * 4.0;                      //    12 scales
    const double shipped_padded = std::ceil(shipped_bytes / 64.0) * 64.0;
    std::printf("memory: shipped_layer_bytes=%.0f float32_only_bytes=%.0f ratio=%.3fx (shipped/float32)\n",
                shipped_padded, f32_bytes, shipped_padded / f32_bytes);
    std::printf("memory_note: masks are stored IN ADDITION to float32 weights; masks-only payload would be 780 B vs 12288 B (15.75x) only if the float arrays were dropped\n");

    std::vector<float> in(D), out(D);
    for (auto& v : in) v = dist(rng);
    for (int i = 0; i < 200; ++i) model_forward_layer(ctx, 0, in.data(), out.data());

    const int iters = 20000;
    volatile float sink_f = 0.0f;
    double t0 = qpc_us();
    for (int i = 0; i < iters; ++i) {
        model_forward_layer(ctx, 0, in.data(), out.data());
        sink_f = sink_f + out[i & 7];
    }
    double t1 = qpc_us();
    const double step_us = (t1 - t0) / iters;
    std::printf("speed: ternary_gqa_step_us=%.3f steps_per_sec=%.0f\n", step_us, 1.0e6 / step_us);

    for (int i = 0; i < 200; ++i) ref_gqa_forward(in.data(), out.data());
    t0 = qpc_us();
    for (int i = 0; i < iters; ++i) {
        ref_gqa_forward(in.data(), out.data());
        sink_f = sink_f + out[i & 7];
    }
    t1 = qpc_us();
    const double ref_us = (t1 - t0) / iters;
    std::printf("speed: scalar_float32_ref_step_us=%.3f relative=%.2fx (ref/shipped)\n", ref_us, ref_us / step_us);
    std::printf("speed_note: shipped_avx512_stream_vs_auto_vectorized_scalar; the ternary path is slower here because masks sit on the critical path\n");

    // ---- Resonance-gated memory -------------------------------------------
    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, nullptr);
    if (!db) {
        std::printf("RESULT: FAILED (vsa db)\n");
        model_destroy_context(ctx);
        return 2;
    }
    const int N = 4096;
    std::vector<std::vector<uint64_t>> vecs(N, std::vector<uint64_t>(8));
    for (int i = 0; i < N; ++i) {
        for (auto& w : vecs[i]) w = (static_cast<uint64_t>(rng()) << 32) ^ rng();
        if (vsa_db_insert_vector(db, 0x1000ULL + i, vecs[i].data()) != 0) {
            std::printf("RESULT: FAILED (vsa insert)\n");
            vsa_db_destroy(db);
            model_destroy_context(ctx);
            return 2;
        }
    }

    std::vector<uint64_t> probe(8), best(8);
    int fp = 0;
    const int fp_probes = 2000;
    for (int p = 0; p < fp_probes; ++p) {
        for (auto& w : probe) w = (static_cast<uint64_t>(rng()) << 32) ^ rng();
        float sim = 0.0f;
        vsa_db_query_vector(db, probe.data(), best.data(), &sim);
        if (sim >= 0.40f) ++fp;
    }

    int hits = 0;
    double sim_sum = 0.0;
    const int flips = 500;
    for (int p = 0; p < flips; ++p) {
        const int src = p % N;
        probe = vecs[src];
        probe[0] ^= (1ULL << (p % 64));
        float sim = 0.0f;
        if (vsa_db_query_vector(db, probe.data(), best.data(), &sim) == 1) ++hits;
        sim_sum += sim;
    }

    // Latency: each probe depends on the previous hit, so the compiler cannot
    // hoist the loop and the reported time is a real dependent chain.
    std::vector<uint64_t> chain = vecs[0];
    uint64_t sink_u = 0;
    const int lat_iters = 20000;
    for (int i = 0; i < 200; ++i) {
        float sim = 0.0f;
        vsa_db_query_vector(db, chain.data(), best.data(), &sim);
        chain[0] = best[0];
    }
    t0 = qpc_us();
    for (int i = 0; i < lat_iters; ++i) {
        float sim = 0.0f;
        const int rc = vsa_db_query_vector(db, chain.data(), best.data(), &sim);
        sink_u += static_cast<uint64_t>(rc) + static_cast<uint64_t>(sim * 1.0e6f) + best[0];
        chain[0] = best[0] ^ static_cast<uint64_t>(i);
    }
    t1 = qpc_us();
    const double lat_us = (t1 - t0) / lat_iters;

    std::printf("vsa: vectors=%d fp_rate=%.3f%% flip_recall=%.2f%% mean_flip_sim=%.4f\n",
                N, 100.0 * fp / fp_probes, 100.0 * hits / flips, sim_sum / flips);
    std::printf("vsa: dependent_query_latency_us=%.4f queries_per_sec=%.0f gate=0.40 sink=%llu\n",
                lat_us, 1.0e6 / lat_us, (unsigned long long)sink_u);
    std::printf("vsa_note: recall and false-positive rate follow from gate calibration (flip sim %.3f vs gate 0.40, null ~0.0); they are a calibration check, not an accuracy claim\n",
                sim_sum / flips);

    vsa_db_destroy(db);
    model_destroy_context(ctx);
    std::printf("RESULT: MEASURED\n");
    return 0;
}

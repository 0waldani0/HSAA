// Mechanism benchmark: 1.58-bit ternary projections + resonance-gated VSA memory.
// Measures on the host CPU with no GPU:
//   1. Accuracy of the shipped ternary GQA block against a float64 reference.
//   2. Per-step throughput and per-layer weight bytes, ternary vs float32.
//   3. Resonance-gated 512-bit memory: 1-bit-flip recall, false positives on
//      unrelated probes and probe throughput at the shipped 0.40 gate.
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

// Float32 reference of the shipped GQA operator: base-layer projections
// (0.88/0.12 Q, 0.88/-0.12 K, 0.90/0.10 V), per-head rotations, sigma gate.
static void ref_gqa_forward(const float* in, float* out) {
    const size_t H = 8, KV = 2, G = 4, D = 128;
    const float scale = 0.0883883476f;
    for (size_t h = 0; h < H; ++h) {
        const size_t kv = h / G;
        const size_t rq = (h * 3 + 1) & (D - 1);
        const size_t rk = (kv * 5 + 3) & (D - 1);
        const size_t rv = (kv * 7 + 5) & (D - 1);
        float qp[D], kp[D], vp[D];
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
    model_swap_layer_block(ctx, 0, LAYER_GQA_ATTENTION);

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

    // Weight footprint of one GQA layer. The rotation channels are stored as
    // two 1024-bit masks plus a float scale (1.58 bit/weight); the diagonal
    // channels stay float32, so this is the per-layer total, not a 16x claim.
    const double f32_channel = 1024.0 * 4.0;
    const double tern_channel = 2.0 * (1024.0 / 8.0) + 4.0;
    const double tern_total = 3.0 * (f32_channel + tern_channel);
    const double f32_total = 6.0 * f32_channel;
    std::printf("memory: gqa_layer_ternary_bytes=%.0f float32_bytes=%.0f ratio=%.3fx rot_channel_ratio=%.2fx\n",
                tern_total, f32_total, f32_total / tern_total, f32_channel / tern_channel);

    std::vector<float> in(D), out(D);
    for (auto& v : in) v = dist(rng);
    for (int i = 0; i < 100; ++i) model_forward_layer(ctx, 0, in.data(), out.data());

    const int iters = 20000;
    double t0 = qpc_us();
    for (int i = 0; i < iters; ++i) model_forward_layer(ctx, 0, in.data(), out.data());
    double t1 = qpc_us();
    const double step_us = (t1 - t0) / iters;
    std::printf("speed: ternary_gqa_step_us=%.3f steps_per_sec=%.0f\n", step_us, 1.0e6 / step_us);

    t0 = qpc_us();
    for (int i = 0; i < iters; ++i) ref_gqa_forward(in.data(), out.data());
    t1 = qpc_us();
    const double ref_us = (t1 - t0) / iters;
    std::printf("speed: float32_ref_step_us=%.3f ternary_speedup=%.2fx\n", ref_us, ref_us / step_us);

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
        vsa_db_insert_vector(db, 0x1000ULL + i, vecs[i].data());
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
        const int rc = vsa_db_query_vector(db, probe.data(), best.data(), &sim);
        if (rc == 1) ++hits;
        sim_sum += sim;
    }

    const int bench_queries = 20000;
    t0 = qpc_us();
    for (int p = 0; p < bench_queries; ++p) {
        const int src = p % N;
        vsa_db_query_vector(db, vecs[src].data(), best.data(), nullptr);
    }
    t1 = qpc_us();
    const double qps = bench_queries * 1.0e6 / (t1 - t0);
    std::printf("vsa: vectors=%d fp_rate=%.3f%% flip_recall=%.2f%% mean_flip_sim=%.4f\n",
                N, 100.0 * fp / fp_probes, 100.0 * hits / flips, sim_sum / flips);
    std::printf("vsa: queries_per_sec=%.0f gate=0.40\n", qps);

    vsa_db_destroy(db);
    model_destroy_context(ctx);
    std::printf("RESULT: MEASURED\n");
    return 0;
}

// True CEGIS over the finite induction DSL: seed demonstrations + oracle
// verification on a declared lattice + counterexample injection + re-induction.
//
// Gates:
//   addition on 0..9 x 0..9, seeded with ONE example (0,0)->0,
//     must converge to a program that matches add() everywhere;
//   successor mod 10 on 0..9, seeded with ONE example 0->1, must converge
//     and the first candidate (x+1) must be refuted by the boundary point;
//   an oracle outside the DSL (sin scaled to integers) must be REFUSED.
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <filesystem>

#include "popcnt_brain.h"

static int oracle_add(const float* x, size_t dim, float* y, void*) {
    if (dim != 2) return 0;
    *y = x[0] + x[1];
    return 1;
}
static int oracle_succ_mod10(const float* x, size_t dim, float* y, void*) {
    if (dim != 1) return 0;
    *y = static_cast<float>(std::fmod(static_cast<double>(x[0]) + 1.0, 10.0));
    return 1;
}
static int oracle_sin(const float* x, size_t dim, float* y, void*) {
    if (dim != 1) return 0;
    *y = static_cast<float>(std::lround(1000.0 * std::sin(static_cast<double>(x[0]))));
    return 1;
}

int main() {
    std::error_code ec;
    std::filesystem::remove("build/cegis.bin", ec);
    std::filesystem::remove("build/cegis.bin.llm_bank", ec);
    PopCountBrain* brain = brain_create(1u << 16, "build/cegis.bin");
    if (!brain) { std::printf("RESULT: FAILED (brain_create)\n"); return 2; }

    // ---- 1. addition, single seed ------------------------------------------
    {
        const float seed_x[2] = {0.0f, 0.0f};
        const float seed_y[1] = {0.0f};
        const float lo[2] = {0.0f, 0.0f}, hi[2] = {9.0f, 9.0f};
        const int steps[2] = {10, 10};
        char formula[512] = {0};
        BrainCegisRelationReport rep;
        const int ok = brain_synthesize_relation_cegis(
            brain, "cegis_add", seed_x, seed_y, 1, 2, lo, hi, steps, 12,
            &oracle_add, nullptr, formula, sizeof(formula), &rep);
        std::printf("cegis_add: ok=%d formula=\"%s\" iters=%d cex=%d checked=%d mismatches=%d nodes=%d ms=%.0f\n",
                    ok, formula, rep.iterations, rep.counterexamples, rep.last_checked,
                    rep.last_mismatches, rep.nodes, rep.elapsed_ms);
        bool pass = (ok == 1 && rep.converged == 1 && rep.counterexamples >= 1);
        if (pass) {
            // Independent check through the public evaluator, not the report.
            for (int a = 0; a <= 9 && pass; ++a) for (int b = 0; b <= 9 && pass; ++b) {
                float q[2] = {(float)a, (float)b}, out = -999.0f;
                if (brain_eval_induced_relation(brain, "cegis_add", q, 2, &out) != 1 ||
                    out != (float)(a + b)) pass = false;
            }
        }
        std::printf("cegis_add: exhaustive_eval=%s\n", pass ? "PASS" : "FAIL");
        if (!pass) { brain_destroy(brain); std::printf("RESULT: FAILED\n"); return 1; }
    }

    // ---- 2. successor mod 10: the seed fits (x+1); x=9 refutes it ----------
    {
        const float seed_x[1] = {0.0f};
        const float seed_y[1] = {1.0f};
        const float lo[1] = {0.0f}, hi[1] = {9.0f};
        const int steps[1] = {10};
        char formula[512] = {0};
        BrainCegisRelationReport rep;
        const int ok = brain_synthesize_relation_cegis(
            brain, "cegis_succ", seed_x, seed_y, 1, 1, lo, hi, steps, 12,
            &oracle_succ_mod10, nullptr, formula, sizeof(formula), &rep);
        std::printf("cegis_succ: ok=%d formula=\"%s\" iters=%d cex=%d mismatches=%d ms=%.0f\n",
                    ok, formula, rep.iterations, rep.counterexamples, rep.last_mismatches, rep.elapsed_ms);
        bool pass = (ok == 1 && rep.converged == 1 && rep.counterexamples >= 1 &&
                     std::strstr(formula, "mod") != nullptr);
        if (pass) {
            for (int a = 0; a <= 9 && pass; ++a) {
                float q[1] = {(float)a}, out = -1.0f;
                const float want = (float)((a + 1) % 10);
                if (brain_eval_induced_relation(brain, "cegis_succ", q, 1, &out) != 1 || out != want) pass = false;
            }
        }
        std::printf("cegis_succ: structural_check=%s (boundary counterexample forced mod)\n", pass ? "PASS" : "FAIL");
        if (!pass) { brain_destroy(brain); std::printf("RESULT: FAILED\n"); return 1; }
    }

    // ---- 3. inexpressible oracle must be refused ---------------------------
    {
        std::vector<float> sx, sy;
        for (int i = 0; i < 3; ++i) { sx.push_back((float)i); sy.push_back((float)std::lround(1000.0 * std::sin((double)i))); }
        const float lo[1] = {0.0f}, hi[1] = {15.0f};
        const int steps[1] = {16};
        char formula[512] = {0};
        BrainCegisRelationReport rep;
        const int ok = brain_synthesize_relation_cegis(
            brain, "cegis_sin", sx.data(), sy.data(), sx.size(), 1, lo, hi, steps, 8,
            &oracle_sin, nullptr, formula, sizeof(formula), &rep);
        // Two honest failure modes are acceptable: no DSL candidate fits the seeds at
        // all (fast refusal), or candidates keep being refuted until max_iters.
        const bool pass = (ok == 0 && rep.converged == 0);
        std::printf("cegis_sin: ok=%d converged=%d mismatches=%d iters=%d formula=\"%s\" gate=%s\n",
                    ok, rep.converged, rep.last_mismatches, rep.iterations, formula, pass ? "PASS" : "FAIL");
        if (!pass) { brain_destroy(brain); std::printf("RESULT: FAILED (accepted an inexpressible oracle)\n"); return 1; }
    }

    brain_destroy(brain);
    std::printf("RESULT: MEASURED\n");
    return 0;
}

// Finite inductive relation learner: the mechanism that can answer a pair that
// was never demonstrated, when the rule behind it was.
//
// brain_generate / brain_eval_continuous_function cannot do this (measured:
// 0/2 on the withheld successor and addition pairs). This suite measures
// brain_induce_relation / brain_eval_induced_relation instead. Every case here
// is decided by exact fit on the demonstrations; anything outside the DSL is
// expected to FAIL honestly.
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <chrono>

#include "popcnt_brain.h"

struct CaseOut {
    int ok;
    double ms;
    std::string formula;
    std::vector<float> preds;
    bool expected_found;
};

static CaseOut run_case(
    PopCountBrain* brain,
    const char* name,
    const std::vector<float>& xs,
    const std::vector<float>& ys,
    size_t n, size_t x_dim,
    const std::vector<float>& queries,
    bool expected_found) {
    const auto t0 = std::chrono::steady_clock::now();
    char formula[512] = {0};
    const int ok = brain_induce_relation(brain, name, xs.data(), ys.data(), n, x_dim, 9, formula, sizeof(formula));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::vector<float> preds;
    for (size_t i = 0; i < queries.size() / x_dim; ++i) {
        float y = 0.0f;
        const float* q = queries.data() + i * x_dim;
        if (ok == 1) brain_eval_induced_relation(brain, name, q, x_dim, &y);
        preds.push_back(y);
    }
    const bool correct = expected_found
        ? (ok == 1)
        : (ok == 0);
    std::printf("%s: ok=%d expected_found=%d fit_verdict=%s formula=\"%s\" ms=%.1f\n",
                name, ok, expected_found ? 1 : 0, correct ? "PASS" : "FAIL",
                formula, ms);
    return {ok, ms, formula, preds, expected_found};
}

int main() {
    PopCountBrain* brain = brain_create(1u << 16, "build/induction.bin");
    if (!brain) { std::printf("RESULT: FAILED (brain_create)\n"); return 2; }

    // ---- Task 1: ADDITION_1to4, pairs 2+3 and 3+4 withheld ------------------
    {
        std::vector<float> xs, ys;
        for (int a = 1; a <= 4; ++a) for (int b = 1; b <= 4; ++b) {
            if ((a == 2 && b == 3) || (a == 3 && b == 4)) continue;
            xs.push_back((float)a); xs.push_back((float)b); ys.push_back((float)(a + b));
        }
        const std::vector<float> q = {2.0f, 3.0f, 3.0f, 4.0f};
        CaseOut r = run_case(brain, "addition_1to4", xs, ys, 14, 2, q, true);
        std::printf("   query (2,3)=%g (3,4)=%g\n", r.preds[0], r.preds[1]);
        const bool good = (r.ok == 1 && r.preds[0] == 5.0f && r.preds[1] == 7.0f);
        std::printf("addition_1to4: gate=%s\n", good ? "PASS" : "FAIL");
        if (!good) { brain_destroy(brain); std::printf("RESULT: FAILED\n"); return 1; }
    }

    // ---- Task 2: SUCCESSOR_mod10, pairs 4->5 and 7->8 withheld --------------
    {
        std::vector<float> xs, ys;
        for (int i = 0; i < 10; ++i) {
            if (i == 4 || i == 7) continue;
            xs.push_back((float)i); ys.push_back((float)((i + 1) % 10));
        }
        const std::vector<float> q = {4.0f, 7.0f};
        CaseOut r = run_case(brain, "successor_mod10", xs, ys, 8, 1, q, true);
        std::printf("   query 4->%g 7->%g\n", r.preds[0], r.preds[1]);
        const bool good = (r.ok == 1 && r.preds[0] == 5.0f && r.preds[1] == 8.0f);
        std::printf("successor_mod10: gate=%s\n", good ? "PASS" : "FAIL");
        if (!good) { brain_destroy(brain); std::printf("RESULT: FAILED\n"); return 1; }
    }

    // ---- Task 3: MUL_mod10, pair (2,3)->6 withheld ---------------------------
    {
        std::vector<float> xs, ys;
        for (int a = 1; a <= 3; ++a) for (int b = 1; b <= 3; ++b) {
            if (a == 2 && b == 3) continue;
            xs.push_back((float)a); xs.push_back((float)b); ys.push_back((float)((a * b) % 10));
        }
        const std::vector<float> q = {2.0f, 3.0f};
        CaseOut r = run_case(brain, "mul_mod10", xs, ys, 8, 2, q, true);
        std::printf("   query (2,3)=%g\n", r.preds[0]);
        const bool good = (r.ok == 1 && r.preds[0] == 6.0f);
        std::printf("mul_mod10: gate=%s\n", good ? "PASS" : "FAIL");
        if (!good) { brain_destroy(brain); std::printf("RESULT: FAILED\n"); return 1; }
    }

    // ---- Task 4 (negative control): a random permutation has no DSL rule -----
    {
        std::vector<float> perm = {13, 2, 9, 0, 15, 6, 11, 4, 7, 1, 14, 8, 3, 12, 5, 10};
        std::vector<float> xs, ys;
        for (int i = 0; i < 16; ++i) { xs.push_back((float)i); ys.push_back(perm[i]); }
        CaseOut r = run_case(brain, "random_permutation", xs, ys, 16, 1, {}, false);
        const bool good = (r.ok == 0);
        std::printf("random_permutation: gate=%s (must NOT invent a rule)\n", good ? "PASS" : "FAIL");
        if (!good) { brain_destroy(brain); std::printf("RESULT: FAILED (spurious exact fit on random data)\n"); return 1; }
    }

    // ---- Task 5 (honest miss): the DSL cannot express sin(x)+0.25x^2 ---------
    {
        std::vector<float> xs, ys;
        for (int i = 0; i < 17; ++i) {
            const float x = -2.0f + 0.25f * (float)i;
            xs.push_back(x);
            ys.push_back(std::sin(x) + 0.25f * x * x);
        }
        CaseOut r = run_case(brain, "sin_plus_quarter_x2", xs, ys, 17, 1, {}, false);
        const bool good = (r.ok == 0);
        std::printf("sin_plus_quarter_x2: gate=%s (DSL must NOT claim a fit)\n", good ? "PASS" : "FAIL");
        if (!good) { brain_destroy(brain); std::printf("RESULT: FAILED (spurious exact fit on sin()+x^2)\n"); return 1; }
    }

    // ---- Task 6: SUCCESSOR (no mod), pairs 4 and 7 withheld ------------------
    {
        std::vector<float> xs, ys;
        for (int i = 0; i < 10; ++i) {
            if (i == 4 || i == 7) continue;
            xs.push_back((float)i); ys.push_back((float)(i + 1));
        }
        const std::vector<float> q = {4.0f, 7.0f};
        CaseOut r = run_case(brain, "successor", xs, ys, 8, 1, q, true);
        std::printf("   formula=\"%s\" query 4->%g 7->%g\n", r.formula.c_str(), r.preds[0], r.preds[1]);
        const bool good = (r.ok == 1 && r.preds[0] == 5.0f && r.preds[1] == 8.0f);
        std::printf("successor: gate=%s\n", good ? "PASS" : "FAIL");
        if (!good) { brain_destroy(brain); std::printf("RESULT: FAILED\n"); return 1; }
    }

    brain_destroy(brain);
    std::printf("RESULT: MEASURED\n");
    return 0;
}

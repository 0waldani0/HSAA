// Unseen-function benchmark: interpolation, extrapolation, and curvature.
//
// Important framing: brain_learn_continuous_function fits a fresh local model
// on every call. There is no "training set of families", so "unseen family" is
// not a property of this API - every call is unseen. What actually varies is
// how well a local quadratic smoother can represent the function between the
// supplied samples. This benchmark therefore measures:
//   1. inside-hull interpolation of a smooth low-curvature family (baseline)
//   2. inside-hull interpolation of the same family under higher curvature
//   3. a different low-curvature family (cubic) - control for "never fitted"
//   4. a high-curvature family (quintic) - where local smoothing degrades
//   5. extrapolation outside the sampled range, against a linear-fit baseline
//   6. a known model queried against a different law (no transfer expected)
// All errors are reported both absolutely and as NRMSE = RMSE / RMS(y), so
// families with different y-scales are comparable. The only hard gate is the
// shipped contract (known-family midpoints MSE < 1e-3), which makes this a
// regression test as well as a measurement.
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#include "popcnt_brain.h"

static double f_known(double x)  { return std::sin(x) + 0.25 * x * x; }
static double f_cubic(double x)  { return x * x * x - 2.0 * x; }
static double f_quint(double x)  { return x * x * x * x * x - 3.0 * x + 2.0; }

struct ErrStats {
    double mse;
    double max_abs;
    double y_rms;
    int    api_failures;
    double nrmse_pct() const { return 100.0 * std::sqrt(mse) / std::max(1e-12, y_rms); }
};

static ErrStats eval_range(
    PopCountBrain* brain,
    const char* name,
    double (*truth)(double),
    double x0, double step, int n
) {
    double se = 0.0, max_abs = 0.0, y_sq = 0.0;
    int failures = 0;
    for (int i = 0; i < n; ++i) {
        const float xq = static_cast<float>(x0 + step * i);
        const double gold = truth(static_cast<double>(xq));
        float y_hat = 0.0f;
        if (brain_eval_continuous_function(brain, name, &xq, 1, &y_hat, 1) != 1) {
            ++failures;
        }
        const double d = static_cast<double>(y_hat) - gold;
        se += d * d;
        if (std::fabs(d) > max_abs) max_abs = std::fabs(d);
        y_sq += gold * gold;
    }
    return { se / n, max_abs, std::sqrt(y_sq / n), failures };
}

// Least-squares line through the training samples, evaluated on the same grid.
// If the engine's extrapolation is no better than this, it is clamped
// boundary behaviour rather than a learned trend.
static ErrStats linear_baseline(
    const std::vector<float>& xs,
    const std::vector<float>& ys,
    double (*truth)(double),
    double x0, double step, int n
) {
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (size_t i = 0; i < xs.size(); ++i) {
        sx += xs[i]; sy += ys[i];
        sxx += static_cast<double>(xs[i]) * xs[i];
        sxy += static_cast<double>(xs[i]) * ys[i];
    }
    const double denom = xs.size() * sxx - sx * sx;
    const double slope = (std::fabs(denom) > 1e-12) ? (xs.size() * sxy - sx * sy) / denom : 0.0;
    const double intercept = (sy - slope * sx) / static_cast<double>(xs.size());
    double se = 0.0, max_abs = 0.0, y_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        const double xq = x0 + step * i;
        const double pred = intercept + slope * xq;
        const double gold = truth(xq);
        const double d = pred - gold;
        se += d * d;
        if (std::fabs(d) > max_abs) max_abs = std::fabs(d);
        y_sq += gold * gold;
    }
    return { se / n, max_abs, std::sqrt(y_sq / n), 0 };
}

int main() {
    PopCountBrain* brain = brain_create(1u << 16, "build/gs_function.bin");
    if (!brain) { std::printf("RESULT: FAILED (brain_create)\n"); return 2; }

    const int N = 17;
    std::vector<float> x(N), yk(N), yc(N), yq(N);
    for (int i = 0; i < N; ++i) {
        const float xv = -2.0f + 0.25f * static_cast<float>(i);
        x[i] = xv;
        yk[i] = static_cast<float>(f_known(xv));
        yc[i] = static_cast<float>(f_cubic(xv));
        yq[i] = static_cast<float>(f_quint(xv));
    }

    const int ok_known = brain_learn_continuous_function(brain, "sine_plus_quadratic", x.data(), yk.data(), N, 1, 1);
    const int ok_cubic = brain_learn_continuous_function(brain, "cubic_control", x.data(), yc.data(), N, 1, 1);
    const int ok_quint = brain_learn_continuous_function(brain, "quintic_control", x.data(), yq.data(), N, 1, 1);
    std::printf("learn: known=%d cubic=%d quintic=%d samples=%d range=[-2,2] step=0.25\n",
                ok_known, ok_cubic, ok_quint, N);
    if (ok_known != 1 || ok_cubic != 1 || ok_quint != 1) {
        brain_destroy(brain);
        std::printf("RESULT: FAILED (learn)\n");
        return 2;
    }

    const ErrStats train_pts = eval_range(brain, "sine_plus_quadratic", f_known, -2.0, 0.25, 17);
    const ErrStats interp = eval_range(brain, "sine_plus_quadratic", f_known, -1.875, 0.25, 16);
    const ErrStats cubic = eval_range(brain, "cubic_control", f_cubic, -1.875, 0.25, 16);
    const ErrStats quint = eval_range(brain, "quintic_control", f_quint, -1.875, 0.25, 16);
    const ErrStats extrap = eval_range(brain, "sine_plus_quadratic", f_known, 2.25, 0.25, 9);
    const ErrStats extrap_lin = linear_baseline(x, yk, f_known, 2.25, 0.25, 9);
    const ErrStats quint_extrap = eval_range(brain, "quintic_control", f_quint, 2.25, 0.25, 9);
    const ErrStats cross_law = eval_range(brain, "sine_plus_quadratic", f_quint, -1.875, 0.25, 16);

    std::printf("known_family:  train_mse=%.3e interp_mse=%.3e nrmse=%.4f%% maxerr=%.4f (low curvature)\n",
                train_pts.mse, interp.mse, interp.nrmse_pct(), interp.max_abs);
    std::printf("cubic_control: interp_mse=%.3e nrmse=%.4f%% maxerr=%.4f (never fitted, low curvature)\n",
                cubic.mse, cubic.nrmse_pct(), cubic.max_abs);
    std::printf("quintic_ctrl:  interp_mse=%.3e nrmse=%.4f%% maxerr=%.4f (never fitted, high curvature)\n",
                quint.mse, quint.nrmse_pct(), quint.max_abs);
    std::printf("extrapolation: engine_mse=%.3e nrmse=%.4f%% maxerr=%.4f | linear_fit_mse=%.3e nrmse=%.4f%%\n",
                extrap.mse, extrap.nrmse_pct(), extrap.max_abs,
                extrap_lin.mse, extrap_lin.nrmse_pct());
    std::printf("quintic_extrap: mse=%.3e nrmse=%.4f%% maxerr=%.4f\n",
                quint_extrap.mse, quint_extrap.nrmse_pct(), quint_extrap.max_abs);
    std::printf("cross_law:     mse=%.3e nrmse=%.4f%% maxerr=%.4f (sine model vs quintic truth)\n",
                cross_law.mse, cross_law.nrmse_pct(), cross_law.max_abs);

    const int api_ok = (train_pts.api_failures == 0 && interp.api_failures == 0 &&
                        cubic.api_failures == 0 && quint.api_failures == 0 &&
                        extrap.api_failures == 0 && quint_extrap.api_failures == 0 &&
                        cross_law.api_failures == 0);
    // Contract the shipped suite already enforces: known-family midpoints < 1e-3.
    const bool contract = (interp.mse < 1e-3) && api_ok;
    // Curvature, not novelty, is what degrades the fit.
    const bool curvature_controls = (cubic.mse < 1e-3) && (quint.mse > cubic.mse * 10.0);
    std::printf("verdict: interpolation_contract=%s, curvature_controls=%s, "
                "extrapolation=%s, symbolic_formula_output=ABSENT (numeric API only)\n",
                contract ? "PASS" : "FAIL",
                curvature_controls ? "CONFIRMED" : "NOT_CONFIRMED",
                (extrap.mse < extrap_lin.mse * 0.99) ? "BEATS_LEAST_SQUARES_LINE"
                : (extrap.mse <= extrap_lin.mse * 1.05) ? "MATCHES_LEAST_SQUARES_LINE"
                : "WORSE_THAN_LEAST_SQUARES_LINE");

    brain_destroy(brain);
    std::printf("RESULT: %s\n", contract ? "MEASURED" : "FAILED (interpolation regression)");
    return contract ? 0 : 1;
}

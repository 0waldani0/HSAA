# Test Infrastructure

**Status:** describes the suites that exist in the tree today. Measured results
and the current readiness statement are in `TEST_READY.md`.

## 1. Layers

- **Per-module suites** (built and run by `build.bat`):
  `test_bridge`, `test_vsa_db`, `test_model_mod`, `test_hssa`, `test_bench`,
  plus the second-brain gates (`tests\build_popcnt_backoff_tests.bat`,
  `tests\build_popcnt_selector_tests.bat`) and the C-ABI probe
  (`second_brain\measur\build_capi_probe.bat`).
- **Native E2E** (`tests\e2e_test_runner.cpp`): links the real modules and
  exercises the public C-ABI end to end (VSA DB, second brain, HSSA,
  model_mod, bench_runner). 43 checks; exit code 0 only if all pass.
  Built and run by `tests\build_tests.bat run`; `build.bat test` delegates to it.
- **Adversarial / stress suites** (`tests\build_stress*.bat`,
  `tests\build_hssa_m1_tests.bat`, `tests\build_hssa_m2_tests.bat`, `tests\build_hssa_m3_tests.bat`,
  `build_adversarial_stress.bat`, `tests\build_challenger2.bat`): boundary matrices,
  concurrency, lifecycle and leak checks, CUDA M2, mutation-flip sensitivity.
- **Benchmarks** (`tests\build_bench_model_mod.bat`,
  `tests\build_bench_hot_swap.bat`, `tests\build_stress_bench.bat`, `build_micro_bench.bat`): numerical fidelity
  against independent software references plus latency/throughput.
- **Regression driver** (`second_brain\measur\regress_par.py`): serial phase
  (gates, engine library, associative calibration, CUDA M2 on an idle host),
  parallel phase (module suites, C-ABI probe, E2E, corpus), serial phase
  (adversarial/benchmark suites, PTQ host validation). A step fails on a
  non-zero exit code or a `[FAIL]` line in its log.

## 2. Rules

- Tests link production objects; there is no mock engine and no source-text
  inspection. The deleted `tests/tier1..4_*.cpp` + `test_harness.h` suite was
  exactly that and had reported "184/184" without running the engine.
- A test script propagates the runner's exit code. Scripts that ran the
  executable inside an `if (...)` block used to report 0 unconditionally because
  `%ERRORLEVEL%` was expanded at parse time; those were rewritten.
- Assertions are only relaxed when the pinned contract itself was wrong (e.g.
  the fabricated profiler accuracy, the pre-gate "return nearest anyway"
  behaviour, the stale "vector count == 1" after fact bindings were added to
  the unified vector table). Each such change is accompanied by the measured
  contract that replaces it.
- Performance thresholds are host-sensitive. HSSA M2 Test 3 (400 us step
  ceiling) runs serially, never inside the parallel regression phase; the
  measured range in that mode is 360-392 us.

## 3. Layout

```
tests/
  e2e_test_runner.cpp          real E2E over the public C-ABI
  build_tests.bat              builds + runs the E2E (real modules)
  test_bridge.cpp              bridge round trip suite
  test_vsa_db.cpp              VSA DB contract + resonance gate
  test_model_mod.cpp           model modification suite
  test_hssa.cpp                HSSA conformance suite
  test_bench_runner.cpp        profiler metrics suite
  stress_vsa_db.cpp            concurrent VSA stress
  stress_bench_runner.cpp      profiler adversarial stress
  stress_tier5_system.cpp      integrated concurrency/lifecycle stress
  stress_tier5_adversarial.cpp boundary/leak adversarial stress
  stress_model_mod.cpp         model_mod stress
  bench_model_mod.cpp          model_mod numerical fidelity benchmark
  bench_vsa_hot_swap.cpp       vector scan / hot-swap benchmark
  test_ptq_validation.cpp      PTQ host tiling/rank rejection guards
  test_hssa_m1.cpp             HSSA M1 adversarial
  test_hssa_m2.cpp             HSSA M2 CUDA adversarial
  test_hssa_m3.cpp             HSSA M3 adversarial
  build_*.bat                  per-suite build/run scripts
```

## 4. Coverage boundaries

- No pretrained checkpoint is shipped, so no perplexity or generation-quality
  benchmark is claimed. `model_mod` is FP32/procedural; `bench_model_mod`
  checks SIMD against scalar references.
- The CUDA path in HSSA is a measurement-only GEMV/DMA microbenchmark with a
  synthetic matrix; its output is not blended into the model state. The kernel
  math is checked against a CPU reference in `test_hssa_m2.exe`.
- The profiler reports accuracy as N/A unless a genuine reference exists; NaN
  is the sentinel, and consumers print/check N/A rather than a fabricated value.

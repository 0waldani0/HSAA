# Test Readiness Declaration

**Status:** READY for the scope stated below (no trained-model claim).
**Build:** MSVC 2022 C++20, `/arch:AVX512`; CUDA sm_86 via nvcc for the PTQ kernels.
**Entry points:**
- `build.bat test` - full native E2E (`tests\build_tests.bat run`).
- `build.bat test_hssa`, `test_vsa_db`, `test_model_mod`, `test_bench`, `test_bridge` - per-module suites.
- `python second_brain\measur\regress_par.py` - the complete regression (module suites, E2E, C-ABI probe, corpus, CUDA M2, adversarial/benchmark suites). Logs: `second_brain\measur\regress_par_logs\`.

---

## 1. What the E2E actually executes

`tests\e2e_test_runner.cpp` links the real objects (`src/vsa_db.cpp`,
`src/hssa.cpp`, `src/model_mod.cpp`, `src/bridge.cpp`, `src/bench_runner.cpp`,
`build\popcnt_brain.lib`, `build\ptq_cuda.lib`) and drives them through the
public C-ABI:

1. `VsaDbHandle`: fact insert/query/mutate/delete, conflict detection, vector
   count, one-bit-flipped resonance hit, unrelated probe miss.
2. `PopCountBrain` C-ABI: exact query, case-normalized query, conflict/no-conflict,
   unrelated miss.
3. HSSA: forward is finite and non-zero, bitwise deterministic after
   `hssa_reset_state`, and a recalled vector measurably reaches the output
   (relative L2 difference 0.099980 with the 0.5 residual blend).
4. `model_mod`: GQA forward, precision store/load, layer hot-swap, DeltaNet
   forward after the swap.
5. `bench_runner`: latency/throughput are measured; accuracy is reported as N/A.

Measured: **43 checks run, 43 passed, 0 failed, exit code 0.**

The previous "184/184" suite was removed: `tests/tier1..4_*.cpp` and
`tests/test_harness.h` only inspected source text (for example, whether
`build.bat` contained the word "clean") and never executed engine code.

## 2. Measured suite results

| Suite | Command | Result |
|---|---|---|
| Native E2E | `build.bat test` | 43 / 43 |
| HSSA conformance | `build.bat test_hssa` | 6 / 6 |
| VSA database | `build.bat test_vsa_db` | 9 / 9 |
| model_mod | `build.bat test_model_mod` | 10 / 10 |
| bench_runner | `build.bat test_bench` | 40 / 40 |
| VSA stress | `tests\build_stress.bat` | 7 / 7 (192-202M concurrent ops in 2.0 s across runs, 0 errors) |
| bench stress | `tests\build_stress_bench.bat run` | 51 / 51 |
| Tier 5 system | `tests\build_stress_tier5.bat run` | 5 / 5 |
| Tier 5 adversarial | `tests\build_stress_tier5_adversarial.bat run` | 17 / 17 |
| HSSA M1 | `tests\build_hssa_m1_tests.bat run` | 5 / 5 |
| HSSA M2 (CUDA) | `tests\build_hssa_m2_tests.bat run` | 5 / 5; 28-layer step 317-372 us on an idle host (400 us SLA; loaded-host runs of 457-847 us are contention, not code), GEMV cosine 1.00000 / SQNR 137.64 dB vs CPU |
| HSSA M3 | `tests\build_hssa_m3_tests.bat run` | 8 / 8 |
| model_mod benchmark | `tests\build_bench_model_mod.bat run` | 5 / 5 |
| VSA hot-swap benchmark | `tests\build_bench_hot_swap.bat run` | 3 / 3 |
| Challenger 2 | `tests\build_challenger2.bat` | 4 / 4 |
| Adversarial stress | `build_adversarial_stress.bat` | pass |
| Micro-benchmark | `build_micro_bench.bat` | pass |
| PTQ host validation | `tests\build_ptq_validation.bat run` | 5 / 5 (N/M tiling and fused rank rejections) |
| PopCount Generative LLM | `tests\build_llm_text_code_tests.bat run` | 28 / 28 (RU/EN prose, C++/Python codegen, zero-shot param renaming, live fact override) |
| C-ABI probe (F22) | `second_brain\measur\build_capi_probe.bat` | exit 0 |
| Assoc calibration | `second_brain\measur\build_assoc_calib.bat` | recall/FP table at the shipped gate |

## 3. Contract changes that earlier tests had pinned incorrectly

- `vsa_db_query_vector` applies a **0.40** resonance gate. A non-resonant probe
  returns 0 and leaves the similarity at 0.0 instead of returning an arbitrary
  "nearest" slot. The gate is calibrated: independent balanced vectors score
  0 +/- 0.044 (~9 sigma); a one-bit flip scores 0.996.
- Every stored fact also owns its 512-bit binding in the unified vector table:
  after 25 facts and 1 raw vector, `vsa_db_get_vector_count` is 26.
- `bench_run_step_profile` reports `accuracy_available = false` and NaN for
  cosine/SQNR. The `model_mod` blocks are procedural FP32 (`cpu_matrix` is never
  populated), so no quantized reference exists in that path; the old numbers
  (~0.9999 cosine, ~120 dB) came from adding fixed noise to the profiler's own
  output.
- HSSA CUDA is a **measurement-only microbenchmark**: a synthetic 1024x1024
  matrix, GEMV and DMA are timed, and the result is not blended into the model
  state. The kernel itself is verified against a CPU reference by HSSA M2.

## 4. Scope and limitations (no inflated claims)

- There is no trained Qwen checkpoint in this repository and no perplexity
  benchmark against a production model. `model_mod` is a procedural FP32
  research block set; `bench_model_mod` checks it against independent software
  references, not against a pretrained network.
- The second brain's C-ABI resolves exact and normalized keys first, then a
  lexical fallback (SimHash over character trigrams, 512-bit dot gate 200/512).
  Calibrated: subject typo 297/300 distinct / 300/300 template, relation typo
  240/280 = 85.7%, case variants 300/300, false positives 0/1,240 (600
  near-duplicate/bogus-relation probes in each 300-fact population plus 40
  exact-subject/bogus-relation probes; the near-duplicate probes are the
  discriminating class).
- The VSA resonance threshold and the 16 slots/page layout are measured
  trade-offs, not universal constants. The slot-page experiment for 32/64/128
  slots was measured and rejected (see BRIDGE.md).
- The 28-layer step latency (317-372 us on an idle host; the 400 us ceiling is
  host-sensitive) is a CPU AVX-512 figure plus CUDA microbenchmark overhead on
  this host (i7-11800H, 32 GB RAM); it is not a token-generation throughput
  claim for a real model.
- `tests/test_hssa_m2.cpp` Test 3 has a 400 us ceiling and is sensitive to host
  load; `regress_par.py` runs it serially, before the parallel phase.

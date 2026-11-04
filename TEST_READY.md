# Test Readiness Declaration

**Status:** READY for the scope stated below (no trained-model claim).
**Build:** MSVC 2022 C++20, `/arch:AVX512`; CUDA sm_86 via nvcc for the PTQ kernels.
**Entry points:**
- `build.bat test` - full native E2E (`tests\build_tests.bat run`).
- `build.bat test_hssa`, `test_vsa_db`, `test_model_mod`, `test_bench`, `test_bridge`, `test_unified` - per-module suites.
- `tests\build_unified_tests.bat run` - unified engine + 256-bit ARX preimage solver (36 checks).
- `second_brain\measur\native_measur_runner.cpp` & `tests\stress_concurrent_builds.cpp` - native C++20 causal baseline and concurrent build verification (zero Python).

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

Measured: **47 checks run, 47 passed, 0 failed, exit code 0.**

The previous "184/184" suite was removed: `tests/tier1..4_*.cpp` and
`tests/test_harness.h` only inspected source text (for example, whether
`build.bat` contained the word "clean") and never executed engine code.

## 2. Measured suite results

| Suite | Command | Result |
|---|---|---|
| Native E2E | `build.bat test` | 47 / 47 |
| Unified engine + ARX solver | `tests\build_unified_tests.bat run` | 36 / 36 |
| HSSA conformance | `build.bat test_hssa` | 6 / 6 |
| VSA database | `build.bat test_vsa_db` | 9 / 9 |
| model_mod | `build.bat test_model_mod` | 10 / 10 |
| bench_runner | `build.bat test_bench` | 40 / 40 |
| VSA stress | `tests\build_stress.bat` | 7 / 7 (192-202M concurrent ops in 2.0 s across runs, 0 errors) |
| bench stress | `tests\build_stress_bench.bat run` | 51 / 51 |
| Tier 5 system | `tests\build_stress_tier5.bat run` | 5 / 5 |
| Tier 5 adversarial | `tests\build_stress_tier5_adversarial.bat run` | 17 / 17 |
| HSSA M1 | `tests\build_hssa_m1_tests.bat run` | 5 / 5 |
| HSSA M2 (CUDA) | `tests\build_hssa_m2_tests.bat run` | 5 / 5; GEMV cosine 1.00000 / SQNR 137.64 dB vs CPU; average 28-layer step 253-286 us, fastest step 180-228 us, 400 us SLA. Two engine/test fixes were required to reach it: the per-layer head loop is capped at one OpenMP thread per head (a 16-thread team over 8 iterations inflated the step by several times), and the timing thread is no longer pinned to a single core (workers inherit that affinity and ran the parallel loop on 1/8 of the machine) |
| HSSA M3 | `tests\build_hssa_m3_tests.bat run` | 8 / 8 |
| Held-out perplexity | `tests\build_holdout_perplexity.bat` | head top-1 60.84% / **mixture top-1 64.77%** (suffix lambda 0.75 fitted on the train tail) on 11 unseen files; PPL head 281.96 -> **mix 35.28** (8x) vs uniform 9270, OOV 6.72%; in-sample 80.90% head / 97.38% mix; prose mix PPL 166800 (15.19% OOV, still 18x worse than uniform); measured union oracle 66.68% |
| Mechanism benchmark | `tests\build_mechanism_bench.bat` | ternary GQA cosine 1.00000000 / SQNR 131.54 dB, but that is an identity at init (masks all-ones), not compressed accuracy; shipped layer is 1.025x LARGER than float32 because masks are stored in addition to float32 weights; speed 0.42-0.46x of the scalar float32 reference; resonance memory 0.000% FP, 100% 1-bit-flip recall, 5.18 us dependent query latency at 4096 vectors |
| Seen vs unseen text | `tests\build_holdout_perplexity.bat` | same evaluator: head top-1 80.90% seen vs 60.84% unseen (-20.06 pp), mixture 64.77%; PPL 13.22 vs 281.96 (21.3x), mix 35.28; prose mix PPL 166800; measured ceiling 66.68% |
| Unseen-function audit | `tests\build_generalization_bench.bat` | known-family interpolation NRMSE 0.029%; never-fitted cubic 0.382% (passes 1e-3); high-curvature quintic 1.699% (617x worse than cubic); extrapolation NRMSE 11.43% (out-of-hull queries now fall back to a least-squares straight line - matches the linear baseline exactly; quintic 97.4%); no cross-law transfer (94.4%); numeric API only |
| Zero-shot synthesis falsification | `tests\build_generalization_bench.bat` | untrained brain, unrelated-prose brain and T17.1 search-corpus brain emit byte-identical 1512-token Dijkstra code with sandbox verified=1 and 3/3 markers; bare out-of-keyword prompt emits 0 tokens, the generic trigger emits 174 tokens that never mention the requested domain. Learning dependence: none; mechanism: fixed template |
| Rule generalization | `tests\build_generalization_bench.bat` | 100% in-sample replay; periodic sequences continued beyond trained length and phase (ALT2/ALT3 4/4, CYCLE5 12/12 unseen-phase and 4/4 longer); abstract rules denied: successor map 0/2 and addition 0/2 on the one pair never trained, despite every constituent seen. Exits non-zero on training or replay regression |
| Claim proof battery | `tests\build_generalization_bench.bat` | probes the shipped T16.2/T16.5/T17.2 prompts on an untrained brain and on brains trained on unrelated corpora: OOD extrapolation gate fails untrained (passes trained); relation inference passes on an untrained brain (vacuous test); RoPE attention emits byte-identical answers on two unrelated corpora (corpus-inert). Native measur harness rerun: accuracy 38.3202% (README 36.5382%), coverage 60.9998% (58.9347%), draft 1770 ns (10395 ns). Detail: docs\CLAIM_PROOF_MATRIX.md |
| Induction mechanism | `tests\build_generalization_bench.bat` | new API `brain_induce_relation`: recovers addition (x0+x1) and mod-10 successor (mod((x0+1),10)) from 8-14 demos and answers withheld pairs; refuses random permutation and sin(x)+x^2; batch exits non-zero if it fabricates a fit or misses a real one |
| CEGIS synthesis | `tests\build_generalization_bench.bat` | `brain_synthesize_relation_cegis`: addition from ONE seed (0,0)->0 converges in 3 rounds / 2 counterexamples to (x0+x1); mod-10 successor from ONE seed 0->1 converges in 6 / 5 to mod((x0+1),10) (boundary x=9 forces the mod); sin() oracle refused; exhaustive public-API check passes |
| Claims hygiene gate | `python tests\check_claims.py` | fails if a doc repeats a falsified or unproven capability without an explicit verdict marker; docs\METRICS.md is the single source of numbers |
| model_mod benchmark | `tests\build_bench_model_mod.bat run` | 5 / 5 (GQA min cosine 1.00000000 / SQNR 131.63 dB against a reference that includes the shipped base-layer projections) |
| VSA hot-swap benchmark | `tests\build_bench_hot_swap.bat run` | 3 / 3 |
| Challenger 2 | `tests\build_challenger2.bat` | 4 / 4 |
| Adversarial stress | `tests\build_adversarial_stress.bat` | pass (24 checks; the latency mutation flip now perturbs the threshold around the measured value instead of assuming a 0.05 us floor that the faster table violates) |
| Micro-benchmark | `tests\build_micro_bench.bat` | pass |
| PTQ host validation | `tests\build_ptq_validation.bat run` | 5 / 5 (N/M tiling and fused rank rejections) |
| PopCount Generative LLM | `tests\build_llm_text_code_tests.bat run` | 108 / 108 (RU/EN prose, C++/Python codegen, zero-shot param renaming, live fact override, L>1000 hard context horizon, 32-layer non-linear + Riemannian SGD + 3-hop deduction, Distributional OOV Output Priors on held-out targets, and 10+ MB / 1.0 GB bank AVX-512 cache & fragmentation profiling) |
| C-ABI probe (F22) | `second_brain\measur\build_capi_probe.bat` | exit 0 (0 failures; duplicate stores are deduplicated, see section 3) |
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
- `brain_get_stats` reports the **deduplicated** count of distinct
  (Subject, Relation) rules. Re-storing an existing fact 50 times adds 0 rules
  (previously the probe counted calls); the C-ABI probe now pins the documented
  dedup behaviour instead of the old call-counting behaviour.
- `bench_layer_types_numerical_fidelity`'s GQA reference now includes the
  shipped base-layer projections (0.88/0.12 Q, 0.88/-0.12 K, 0.90/0.10 V) with
  the deterministic per-head rotation offsets. Without them the reference
  modeled a plain attention block that the engine has not shipped since
  online training landed, so every GQA iteration failed the 0.99999 gate.
- Tier 5 suite 3 leak detection now counts **live heap blocks** across the two
  rounds instead of comparing private bytes. The same worker threads serve both
  rounds, so thread heaps/stacks are constant; private bytes still drifted by
  3-8 MB per round from arena fragmentation alone (min step times unchanged,
  block delta measured 0 across 6 consecutive runs).

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

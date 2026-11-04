# Generalization audit: seen vs unseen, and "unseen formulas"

Status: measured on the host, reproducible. The corpus is read from the current
working tree, so absolute numbers move when repository files change; the
method and the ratios do not.

Reproduce:

```
tests\build_holdout_perplexity.bat      # seen vs unseen text
tests\build_generalization_bench.bat    # unseen functions + zero-shot falsification
```

Hardware: Intel i7-11800H, 32 GB RAM, no GPU. MSVC 2022, `/O2 /arch:AVX512`.

## 1. Text: memorisation vs generalisation

Training corpus: all repository `src/`, `include/`, `second_brain/` and `tests/`
sources (719 KB), with every 5th file held out (10 files, 165 KB) plus all prose
(51 KB) kept out of training and out of temperature fitting. The engine clamped
6 requested epochs to 1 for a 166230-token corpus, so this is a **single pass**.

| Split | Tokens | OOV | Mean NLL | PPL | Uniform | Top-1 |
|---|---|---|---|---|---|---|
| in-sample (middle 10% of training text) | 17667 | 0.01% | 2.7087 | **15.010** | 9196 | **80.14%** |
| held-out source files | 30757 | 8.82% | 6.0327 | **416.82** | 9196 | **58.11%** |
| held-out prose | 9584 | 16.33% | 12.4502 | **255312** | 9196 | **15.89%** |
| in-sample at T=1 (control) | 17667 | 0.01% | 3055.0 | inf | 9196 | 80.14% |

**Gap: -22.03 percentage points of top-1, 27.8x worse perplexity on unseen
files.** The in-sample probe sits at 50-60% of the training text and the
temperature fit at 90-100%, so the two do not overlap.

Both sides are scored over in-vocabulary transitions only: the unseen split
additionally drops 8.82% of its tokens to OOV (splicing is disabled), so the
gap is if anything understated - the hardest unseen tokens are never scored.

Three further readings:

- Against chance (1/9196 = 0.0109%), 58.11% top-1 on unseen files is real
  in-domain generalisation, and PPL is 22.1x better than uniform.
- Cross-domain prose ranks above chance (15.89%) but has PPL 27.8x **worse**
  than uniform: the distribution is over-confident outside its training
  domain. It does not transfer.
- The trainer's own `replay_accuracy` is 98.22% on seen text, while the pure
  full-vocabulary head scores 80.14%. The difference is the shipped
  shortlist/backoff decision rule, not the learned distribution: the rule fixes
  seen continuations, the softmax does not put 98% mass on them.
- At T=1 the distribution is so peaked that mean NLL is 3055 and PPL overflows
  double. A finite PPL only exists after temperature scaling; that is a property
  of the model, not of the evaluation.

## 2. Functions: what the approximator can and cannot do

Framing correction: `brain_learn_continuous_function` fits a fresh local model
on every call. There is no training set of families, so "unseen family" is not
a property of this API - every call is unseen. What actually varies is how well
a local quadratic smoother represents the function between the supplied samples.
The benchmark therefore uses curvature controls, not "novelty".

17 samples on `[-2, 2]` (step 0.25); errors are absolute and NRMSE = RMSE/RMS(y)
so families with different scales are comparable.

| Case | MSE | NRMSE | Max error | Verdict |
|---|---|---|---|---|
| training points, known family | 5.2e-16 | 0.00% | 0.0000 | exact-match path |
| midpoints, known family `sin(x)+0.25x^2` | 6.5e-08 | 0.029% | 0.0004 | interpolation contract passes |
| cubic `x^3-2x`, never fitted, low curvature | 2.3e-05 | 0.382% | 0.0113 | passes the 1e-3 gate |
| quintic `x^5-3x+2`, never fitted, high curvature | 1.42e-02 | 1.699% | 0.3080 | **617x worse than the cubic** |
| extrapolation `x=2.25..4.25`, known family | 0.0955 | 11.43% | 0.6199 | **= a least-squares straight line** (measured engine == linear fit, MSE 9.553e-02 both) |
| extrapolation, quintic | 4.15e+05 | 97.37% | 1350.2 | catastrophic |
| known model queried against a different law | 43.8 | 94.36% | 17.72 | no law transfer |

Reading:

- The failure axis is **curvature and sampling**, not novelty. A never-fitted
  cubic is recovered to 0.38% NRMSE; the quintic at the same 17 points is 617x
  worse. Both are "unseen"; only one is representable at this sample density.
- Outside the sampled range the original kernel estimator clamped to the boundary
  (22.5% NRMSE, worse than a straight line). **Fixed in `popcnt_brain.cpp`**: queries
  strictly outside the hull now fall back to a least-squares straight line fitted on the
  raw samples, which is the honest floor for extrapolation. Measured: NRMSE 11.43% ==
  the linear baseline exactly. On the quintic this is 97.4% (still no extrapolation, just
  a less wrong answer). The 1D limitation is explicit: x_dim == 1 only.
- Given only `f1`, querying `f2` returns `f1` (94.4% NRMSE against `f2`). The
  engine does not invent a different law; it has no mechanism to.
- There is **no symbolic output anywhere**: the public API returns numeric `y`
  only. The engine cannot emit a formula, so "obtaining an unseen formula" is
  not a capability that exists in this code, independent of accuracy.

## 3. Zero-shot algorithm synthesis: falsified

`brain_synthesize_algorithm_zeroshot` claims to compose unseen algorithmic logic
"when the training corpus only contained linear search and binary search".

| Run | Tokens | Sandbox verified | Markers (rotate_left/right, relaxation) |
|---|---|---|---|
| A. untrained brain | 1512 | 1 | 3/3 |
| B. brain trained on 3 unrelated sentences | 1512 | 1 | 3/3 |
| C. brain trained on T17.1's own search-only corpus | 1512 | 1 | 3/3 |
| A vs B | **byte-identical** | | |
| A vs C | **byte-identical** | | |
| D. suffix-automaton prompt | 1026 | 1 | n/a |
| E. out-of-keyword prompt (Ford-Fulkerson), bare | 0 | 0 | n/a |
| F. same prompt with "Write an algorithm ..." | 174 | 1 | **0 mentions of the requested domain** |

**The corpus and the trained weights are causally inert.** The same 1512 bytes
are produced with no training at all and with the shipped test's own corpus; the
sandbox reports success in every case. A bare out-of-keyword prompt produces
nothing, and the generic "write an algorithm" trigger produces a 174-token
template that never mentions Ford, flow or Fulkerson - it answers a different
question. This is a deterministic rule engine (substring -> code template),
which is a legitimate feature, but it is not learning, not zero-shot, and not
synthesis of an unseen algorithm.

A crash was found and fixed while building this test: on an untrained brain the
C API dereferenced a null `llm_engine` after a successful composition
(`popcnt_brain.cpp`). The wrapper now composes before taking `brain_mutex`,
creates the engine lazily, and holds the lock only for tokenization.

## 4. Rule induction: the generator recombines observed transitions

Five one-token prediction tasks, compiled binary, `greedy` decoding, one
brain per task. Two of them are positive controls (the continuation rule is a
local periodicity fully covered by training transitions); two remove one
determinate pair whose constituents all appear in training.

| Task | Rule tested | In-sample | Unseen phase/length | Verdict |
|---|---|---|---|---|
| ALT2 `a b a b...` | 2-cycle continuation | 4/4 | 4/4 at length 12-15 | GENERALIZES |
| ALT3 `p q r p q r...` | 3-cycle continuation | 4/4 | 4/4 at length 12-15 | GENERALIZES |
| CYCLE5 `1 2 3 4 5 ...` | 5-cycle continuation | 5/5 | 12/12 unseen phase at trained lengths; 4/4 at length 20-23 | GENERALIZES |
| SUCCESSOR `a->b ... j->a` | successor relation, pairs e->f and h->i withheld | 8/8 | 0/2 | MEMORISES_ONLY |
| ADDITION `1..4` | arity-2 addition, pairs 2+3 and 3+4 withheld | 14/14 | 0/2 | MEMORISES_ONLY |

Interpretation, backed by the mechanism and the corpus, not by a guess:

- The engine wins the cycle tasks because every transition the test queries
  (a->b, b->a, ..., 5->1) appeared many times in training. It recombines a
  **transition distribution**; it extrapolates over length and phase, which a
  finite-order n-gram/backoff model does not strictly forbid.
- The engine fails the withheld-pair tasks because the required transition
  never occurred, even though the relation is fully determined by the
  demonstrated pairs and the required result tokens appear in other pairs.
- This is the precise boundary: **composition over observed transitions, no
  abstraction over unseen relations**. It is not a scaling or sample-density
  effect; the training set already contained both operands and both outputs.

## 5. New mechanism: finite inductive relation learner

Constraint of scope: `brain_generate` cannot answer a pair whose required
transition never appeared (measured 0/2). The new public C API closes exactly
that gap: `brain_induce_relation` / `brain_eval_induced_relation` (engine
`induced_impl` in `popcnt_brain.cpp`).

Method: bottom-up enumeration of compositional programs (variables, the
integer constants 0..10, `+`, `-`, `*`, `/`, `mod`), de-duplicated by BEHAVIOUR
on the demonstrations (quantized output-vector hash; first - i.e. smallest -
representative kept per behaviour), returning the smallest program that fits
every one of the demonstrations within 1e-6. Explicit scope:
**finite demonstrations of a scalar source-compositional law**. Not general
reasoning, not open-ended synthesis; `sin(x) + x^2` and anything outside the
DSL correctly report failure.

Measured (`tests/bench_induction.cpp`, this host):

| Task (demonstrations) | Result | Smallest rule found | Time |
|---|---|---|---|
| addition 1..4, pairs 2+3, 3+4 withheld (14 demos) | **answers 5 and 7** | `(x0 + x1)` | 116 ms |
| successor mod 10, 4->5 and 7->8 withheld (8 demos) | **answers 5 and 8** | `mod((x0 + 1), 10)` | 81 ms |
| mul mod 10, pair (2,3) withheld (8 demos) | answers 6; finds `(x0 * x1)` (mod never wraps for operands <= 3; minimal consistent program is the product) | `(x0 * x1)` | 95 ms |
| random permutation (negative control) | **refused** (`ok=0`); no spurious exact fit | - | 98 ms |
| sin(x)+0.25x^2 (honest miss) | **refused** (`ok=0`); not expressible in DSL | - | 104 ms |
| plain successor, 4 and 7 withheld (8 demos) | **answers 5 and 8** | `(x0 + 1)` | 77 ms |

The negative controls are the important part: nothing was "learned" on random
or inexpressible data - the API returns 0 rather than a fabricated law.

### 5.1 CEGIS: counterexample-guided refinement on top of the induction

`brain_synthesize_relation_cegis` (same commit family) closes the loop that the
inductor alone cannot: seed demonstrations -> smallest consistent program ->
verify against the caller's oracle on a declared finite lattice -> inject the
first counterexample -> re-induce. Measured (`tests/bench_cegis.cpp`):

| spec | result | rounds | counterexamples | time |
|---|---|---|---|---|
| addition on 0..9 x 0..9, **one** seed (0,0)->0 | converges to `(x0 + x1)`; public-API exhaustive check 100/100 | 3 | 2 | 422 ms |
| successor mod 10 on 0..9, one seed 0->1 | converges to `mod((x0 + 1), 10)`; the boundary point x=9 refuted the initial `(x0 + 1)` | 6 | 5 | 624 ms |
| sin(x)*1000 rounded on 0..15 | **refused** (no exact candidate for the seeds) | 1 | 0 | - |

The successor case is the structural proof this mechanism adds something: the
first candidate is `(x0 + 1)`, the oracle at x=9 returns 10 vs the true 0, the
counterexample is injected, and the next candidate is the modular law. This is
finite-domain CEGIS over a small DSL, not general program synthesis; an oracle
outside the DSL is refused rather than approximated.

## 6. What the rest of the code actually proves

The inventory of all 23 harnesses, plus this audit, covers:

- **Extrapolation** outside the training range - covered here.
- **Different functional families** - covered here.
- **Rule generalisation in the generator** - covered here; abstract
  relation induction is absent, transition recombination works.
- **No test** of unseen-key recall in the VSA/associative path: every recall
  test uses stored keys, 1-bit flips, or fuzzy spellings. (Random-probe
  rejection is measured: 0.000% false positives at the 0.40 gate.)

Claims with no executable proof behind them:

| Claim | Where | What the code actually does |
|---|---|---|
| "Zero-shot synthesis of unseen composite algorithms" | `second_brain/README.md:440`, `tests/test_llm_text_code_gen.cpp:1258` | **FALSE**: fixed substring-selected template; verified above |
| "OOD open-ended extrapolation (F8.1)" | `PROJECT.md:39` | **UNPROVEN**: only proof is `>=3` tokens and no adjacent duplicate words (`test_llm_text_code_gen.cpp:1123`) |
| "Non-linear function learning" | `PROJECT.md:40` | stores samples and runs LOESS; per-sample 4096-D phases only scale the kernel, while the accumulated `holo_operator_trace` (`popcnt_brain.cpp:14007`) is never read |
| "Off-grid spectral solver recovers parameters" | `tests/test_unified_engine.cpp:309` | sandbox compares against the same LOESS evaluator that produced the fit (circular) |
| "Template-free relation inference" | `second_brain/README.md:424` | **VACUOUS**: subject substring match + sketch dot; single-subject sentences only |

## 7. Verdict

- **Memorisation vs generalisation:** 80.14% -> 58.11% top-1, PPL 15.0 -> 416.8
  on unseen files. Genuine in-domain generalisation, 22 pp below in-sample.
- **Unseen formulas:** no. No symbolic output, no law induction, no
  extrapolation beyond the sampled range (matches a straight line, not a curve); a
  never-fitted cubic is fine, an under-sampled quintic is 617x worse - the limit
  is representation, not novelty.
- **Unseen algorithms:** no. The synthesizer is a template engine that produces
  identical output with zero training, with the shipped corpus, and outside its
  keyword set it either emits nothing or an unrelated template.
- **Rule induction:** the *generator* continues periodic sequences it has never
  seen at longer lengths and unseen phases (12/12 and 4/4), and reproduces
  trained transitions perfectly (100%), but infers no new relation: withheld
  successor and addition pairs score 0/2. The new `brain_induce_relation`
  mechanism closes that specific gap for scalar compositional laws (it recovered
  addition and mod-10 successor in < 120 ms) - still not general reasoning.
- **What is genuinely useful:** in-domain code completion/ranking at 58% top-1
  on unseen files, periodic-sequence continuation, sub-1e-4 MSE interpolation
  of a smooth function inside its sampled range, 5 us associative lookup with a
  calibrated rejection gate, and a deterministic code-template generator.

Any paper or README claim beyond those five needs a new mechanism, not a new
benchmark.

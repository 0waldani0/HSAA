# METRICS — the single source of truth

Every numer this project quotes in README/docs/tests must appear here with its
provenance, or it is not owned. Where a number is stale, unreproducible or
falsified, that is stated here and the original claim must be marked, not
repeated.

Host: Intel i7-11800H, 32 GB RAM, RTX 3050 Laptop 4 GB. MSVC 2022 BuildTools,
`/std:c++20 /O2 /O2 /arch:AVX512 /openmp`. No GPU is used by the engine.

Reproduce everything except the text numbers (corpus changes with the working
tree):

```
tests\build_holdout_perplexity.bat
tests\build_generalization_bench.bat
tests\build_mechanism_bench.bat
tests\build_llm_text_code_tests.bat run
build.bat test
```

## 1. Text: seen vs unseen (bench_holdout_perplexity)

Corpus: repository sources; every 5th file held out; prose held out of training
and temperature fitting. Single pass (requested 6 epochs clamped to 1):
166,230 tokens, vocab 9,132.

| split | OOV | mean NLL | PPL | uniform | head top-1 | suffix top-1 |
|---|---|---|---|---|---|---|
| in-sample (50-60% of train) | 0.01% | 2.5830 | **13.237** | 9,259 | **80.875%** | **97.359%** |
| held-out code files (11) | 6.72% | 5.6434 | **282.420** | 9,259 | **60.832%** | **64.469%** |
| held-out prose | 15.18% | 12.7338 | **339,024.573** | 9,259 | **14.214%** | 13.992% |
| in-sample at T=1 (control) | 0.01% | 2,859.1 | inf | 9,259 | 80.875% | - |

gap: **-20.04 pp head top-1, 21.3x PPL**. Replay accuracy (trainer metric,
shortlist rule): **98.681%** over 160,652 tokens; pure full-vocabulary head on
the same text: 80.875%. Suffix is the exact-horizon predictor (L=1..64 over the
training tables, no held-out data in it). Absolute values move when repository
files change; the method and ratios do not.

Predictor comparison and the measured ceiling on held-out code:

| predictor | top-1 |
|---|---|
| ternary head, argmax | 60.832% |
| exact suffix tables (L=1..64) | **64.469%** |
| same-file copy (best head-scored seen token) | 29.271% |
| role-horizon abstraction, resolved by the head | 58.092% |
| head top-5 contains the target | 62.478% |
| union oracle: head or suffix or copy correct | **66.667%** |

**70-90% is not reachable by re-weighting or re-ranking these signals.** The
union oracle of every predictor in the engine caps at 66.67%, and the head top-5
is 62.48% - barely above its top-1 (60.83%), so a reranker has no headroom.
Reaching 70%+ requires a structurally stronger next-token model or external
pre-training data; see docs/HSAA_vs_LLM.md for the options and costs.

## 2. Rule continuation vs rule induction (bench_rule_generalization)

Five tasks, greedy one-token decoding, fresh brain per task.

| task | in-sample | unseen phase/length | verdict |
|---|---|---|---|
| ALT2 (2-cycle) | 4/4 | 4/4 at length 12-15 | generalises |
| ALT3 (3-cycle) | 4/4 | 4/4 at length 12-15 | generalises |
| CYCLE5 (5-cycle) | 5/5 | 12/12 unseen phase; 4/4 at length 20-23 | generalises |
| SUCCESSOR map (2 pairs withheld) | 8/8 | **0/2** | memorises only |
| ADDITION 1..4 (2 pairs withheld) | 14/14 | **0/2** | memorises only |

## 3. Continuous functions (bench_unseen_function)

17 samples on [-2, 2], NRMSE = RMSE / RMS(y).

| case | MSE | NRMSE | note |
|---|---|---|---|
| known family, training points | 5.2e-16 | 0.000% | exact-match path |
| known family, midpoints | 6.5e-08 | 0.029% | interpolation contract passes |
| cubic `x^3-2x`, never fitted | 2.30e-05 | 0.382% | low curvature |
| quintic `x^5-3x+2`, never fitted | 1.42e-02 | 1.699% | 617x worse than cubic |
| extrapolation 2.25..4.25, known family | 9.553e-02 | 11.432% | **= least-squares line exactly** (out-of-hull fallback, fixed) |
| extrapolation, quintic | 4.151e+05 | 97.373% | catastrophic |
| known model vs a different law | 43.83 | 94.363% | no transfer |

## 4. Claim probes (bench_claim_falsify)

| probe | untrained | trained | verdict |
|---|---|---|---|
| F8.1/T16.2 OOD gate (>=3 tokens, no adjacent duplicate) | 1 token, gate FAIL | 6 tokens, gate PASS, output is a collage of the 6 corpus sentences | test discriminates trained state; "open-ended" unsupported |
| T16.5 template-free relation inference | ok=1, finds target | ok=1, same output | **VACUOUS w.r.t. learning** |
| T17.2 long-context attention | returns 0 (guard) | search-corpus vs prose-corpus brains emit **byte-identical** answers containing the queried location; max_lr 0.4673 vs 0.3878 | **CORPUS-INERT prompt echo** |
| T17.3 "52,000 natural morphological words" | - | vocabulary synthesised from in-code affix tables (24x64x24 + 20x48x24 = 59,904 strings) + 24 synthetic 10-token sequences | **FALSE as stated**; capacity is real, "natural" is not |

## 5. Finite induction (bench_induction, new)

`brain_induce_relation`: bottom-up DSL enumeration (variables, constants 0..10,
`+ - * / mod`), behaviour-hash de-duplication, smallest exact fit within 1e-6.

| task | result | formula | time |
|---|---|---|---|
| addition 1..4, pairs 2+3 / 3+4 withheld | 5 and 7 correct | `(x0 + x1)` | 116 ms |
| successor mod 10, 4->5 / 7->8 withheld | 5 and 8 correct | `mod((x0 + 1), 10)` | 81 ms |
| mul mod 10, pair (2,3) withheld | 6 correct | `(x0 * x1)` (mod never wraps for operands <= 3) | 95 ms |
| plain successor, 4 / 7 withheld | 5 and 8 correct | `(x0 + 1)` | 77 ms |
| random permutation (negative control) | REFUSED | - | 98 ms |
| sin(x)+0.25x^2 (honest miss) | REFUSED | - | 104 ms |

## 6. CEGIS synthesis (bench_cegis, new)

`brain_synthesize_relation_cegis`: seed demonstrations -> smallest consistent
program -> verify against a caller oracle on a declared lattice -> inject first
counterexample -> re-induce.

| spec | result | iterations | counterexamples | time |
|---|---|---|---|---|
| addition, 0..9 x 0..9 lattice, **one** seed (0,0)->0 | converges, formula `(x0 + x1)`, exhaustive check 100/100 | 3 | 2 | 422 ms |
| successor mod 10 on 0..9, one seed 0->1 | converges, formula `mod((x0 + 1), 10)`; boundary point x=9 forced the mod | 6 | 5 | 624 ms |
| sin(x)*1000 rounded on 0..15 (outside DSL) | REFUSED (no exact candidate for the seeds) | 1 | 0 | - |

## 7. Mechanism numbers (bench_mechanism_ternary_vsa)

| metric | value | note |
|---|---|---|
| shipped layer storage | 33,600 B vs 32,768 B float32-only | **1.025x larger**, masks stored in addition to float32 weights |
| ternary GQA cosine / SQNR | 1.00000000 / 131.54 dB | identity at init (masks all-ones), not compressed accuracy |
| shipped path latency vs scalar float32 | 6.1-6.9 us vs 2.8-3.0 us | **0.42-0.46x** |
| VSA dependent query latency | 5.18 us/query (193k q/s at 4,096 vectors) | dependent chain with consumed sink |
| VSA rejection gate | 0.000% false positives; 100% 1-bit-flip recall | calibration check |

## 8. measur acceptance harness (second_brain/measur)

Native harness, `bpe_train.txt` / `bpe_holdout.txt`, seed 424242, 6 epochs,
vocab 4166, 576,806 train / 175,876 holdout transitions:

| metric | published in measur/README.md | re-measured |
|---|---|---|
| holdout accuracy | 36.5382% | **38.3202%** |
| shortlist coverage | 58.9347% | **60.9998%** |
| E[accepted] k=1/2/3 | 0.3654 / 0.5919 / 0.7591 | **0.3832 / 0.6278 / 0.8097** |
| draft cost | 10,395 ns/token | **1,770 ns/token** |
| pages over 16 slots | 21,946 | **0** |
| train time | n/a | 6,714 ms |

The Python scripts that produced the published table (`regress_par.py`,
`sweep_slots.py`, `baselines_word_causal.py`, `corpus_regression.py`) are
**NOT REPRODUCIBLE in this tree** (absent; `__pycache__` only). The reference
corpus guard (91.6 / 91.455 / 91.648%) is therefore unverified.

## 9. Regression state

| suite | result |
|---|---|
| LLM text/code (`tests\build_llm_text_code_tests.bat run`) | 108 / 108 |
| E2E (`build.bat test`) | 47 / 47 |
| generalization batch (`tests\build_generalization_bench.bat`) | exit 0 |

## 10. Not measured / analytical

- `docs/HSAA_vs_LLM.md` is analytical: no LLM was run on this host. All LLM
  cells are marked `~` and are not measurements.
- Zero-shot algorithm "synthesis", OOD open-ended extrapolation and
  52k "natural" vocabulary are falsified or unsupported; see sections 4 and
  `docs/CLAIM_PROOF_MATRIX.md`.
- CEGIS/induction are exact-fit mechanisms over a small DSL. Nothing here
  proves scaling to larger domains, real programs or open-ended novelty.

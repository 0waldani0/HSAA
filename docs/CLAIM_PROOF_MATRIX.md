# Proof matrix for the remaining claims

Status: push of the audit branch done; this document records executable evidence
for every claim that was unproven after the main generalization audit. Every
number below was produced on this host (i7-11800H, 32 GB, no GPU) on the pushed
commit. "Ship left" = this claim is still not proven; see the verdict column.

Commit under test: `2f427be` (now on `origin/main`).

## 1. Re-measurement of the `second_brain/measur` tables

`second_brain/measur/acceptance_harness.cpp` (includes the shipped engine
source, trains on `data/bpe_train.txt`, scores `data/bpe_holdout.txt`, seed
424242, epochs 6) now reports:

| metric (this machine, engine at `2f427be`) | README table (shipped config) | measured |
|---|---|---|
| holdout accuracy | 36.5382% | **38.3202%** |
| shortlist coverage | 58.9347% | **60.9998%** |
| E[accepted] k=1/2/3 | 0.3654 / 0.5919 / 0.7591 | **0.3832 / 0.6278 / 0.8097** |
| draft cost | 10,395 ns/token | **1,770 ns/token** |
| holdout transitions | 175,876 (same) | 175,876 |
| train time | n/a | 6,714 ms |
| conditional accuracy depth 1..12 | n/a | 0.383 → 0.874 |
| occupancy line | 21,946 pages over 16 slots | `over_16slots=0` |

The Python instrumenting scripts that produced the README table
(`regress_par.py`, `sweep_slots.py`, `baselines_word_causal.py`,
`corpus_regression.py`) **are not in this tree** — only their `__pycache__`
binaries remain. The native harness reproduces the split, seed and population
(175,876 transitions) but **not the numbers**: accuracy +1.78 pp, coverage
+2.07 pp, draft 5.9x faster, and zero over-capacity pages (README claims
21,946). Either the shipped engine changed after those numbers were taken, or
the harness measures a different path. Both the "36.5382% / 58.9347%" table and
the slot-sweep table are therefore **unverifiable in this tree**, and the
reference-corpus guard (91.6% / 91.455% / 91.648%) is not reproducible at all.

## 2. Backprop / manifold / OOD claims (README 9.1, T16.1, T16.2)

| Claim | Evidence | Verdict |
|---|---|---|
| Analytical backprop reduces CE loss | Suite T16.1 passes; test compared `init > final` and `extra_final < 0.65 * init` | Supported (loss decreases), but the test does not isolate *learned* state as the cause |
| `brain_paraphrase_continuous` produces lexically diverse agreed output | Suite T16.3 passes (`para_toks >= 7`, `cosine > 0.55`) | Not audited for vacuity; untested control |
| **F8.1 "OOD open-ended extrapolation"** | New probe `tests/bench_claim_falsify.cpp` on the exact T16.2 prompt `"Тензорный фотонный модуль формирует"`: **untrained brain: 1 token, gate fails**; trained on the F-corpus: 6 tokens, gate passes. Output: `"фотонный тензор и ускоряет плотный трансформер"` — a collage of corpus sentences | The test discriminates *trained vs untrained* state; it does not prove "open-ended" or out-of-distribution semantics. Weakly supported, overstated |
| Template-free relation inference | New probe: `brain_infer_template_free_relation` on the exact T16.5 sentence returns `тепловой_шум` (ok=1) from an **untrained brain and an equally from the trained one** | **VACUOUS — the shipped test does not detect learning at all** |
| 4096-D float32 cosine resolves micro-gradients where Hamming does not | Suite T16.6 passes (monotonic epsilon resolution) | Supported as an *embedding property*; it is corpus-independent by construction and evidences no learning |
| "Non-linear function learning" (LOESS + holo trace) | Bench (`bench_unseen_function.cpp`) | Interpolation proven (NRMSE 0.029%); out-of-hull queries now fall back to a least-squares straight line and exactly match the linear baseline (NRMSE 11.43%); `holo_operator_trace` is never read |

## 3. Zero-shot and attention claims (README 10)

| Claim | Evidence | Verdict |
|---|---|---|
| Zero-shot synthesis of unseen composite algorithms | `bench_zeroshot_falsify.cpp` (previous commit) | **Falsified**: byte-identical output with zero training, with the shipped T17.1 corpus, and with unrelated prose; `Ford-Fulkerson` bare prompt → 0 tokens, generic trigger → 174 tokens that never mention the requested domain |
| **T17.2: 2-layer RoPE attention resolves 2-hop implicit narrative** | New probe: the exact T17.2 narrative on two brains trained on **unrelated corpora** (linear/binary search vs cat/dog prose) emit **byte-identical answers containing `обсерватории`** (identical=1); `max_lr_attn` 0.467 / 0.388, both > 0.10. The sentence "…перенес … под купол древней обсерватории" is in the prompt | **Corpus-inert prompt echo.** Training is needed only trivially (untrained brain returns 0 via the `!llm_engine` guard). Not evidence of reading or reasoning |
| **T17.3: scales to >= 52,000 natural morphological words** | Source read: `scale_manifold_vocabulary_internal` synthesises the "vocabulary" from in-code affix tables: Russian `24 prefixes × 64 stems × 24 inflections = 36,864` and English `20 × 48 × 24 = 23,040` combinatorial strings, then creates **24 synthetic Zipf-Mandelbrot sequences of 10 tokens each** and backprops them for 6 epochs. `active_vocab=52,000+` is guaranteed by construction; "natural" is not involved | **Falsified as stated.** The capacity (large synthetic string table, 131k hard cap) is real; the claim about natural vocabulary acquisition is not evidenced by anything measurable here |

## 4. What this matrix proves overall

Claims that still do not have executable support, ranked by severity:

1. **The measur README tables cannot be reproduced** in this tree (+1.78 pp
   accuracy, +2.07 pp coverage, 5.9x draft cost, 0 pages over capacity vs the
   documented 21,946). The numbers the project quotes for its count-mixed
   backoff are stale or measured on a path that is unavailable here.
2. **"Template-free relation inference" is VACUOUS: it passes without any
   training** — it measures a subject substring match, not learning.
3. **The RoPE/attention 2-hop claim is prompt retrieval**, identical across
   independently trained corpora.
4. **"Open-ended OOD extrapolation" discriminates trained vs untrained**, but
   the produced text is a recombination of six training sentences; the
   open-endedness claim is unsupported.
5. **"52,000 natural words" is 59,904 combinatorial affix strings** built in a
   static table, plus 24 synthetic sequences — not natural text acquisition.
6. The reference-corpus guard (91.6% / 91.455% / 91.648%) and the 16-slot
   paging sweep are **not reproducible** (their scripts do not exist in the
   repository).

What *is* supported with executable evidence on this host (and on this push):

- Finite induction (new API `brain_induce_relation`): exact compositional fits recovered
  (`(x0 + x1)`, `mod((x0 + 1), 10)`, `(x0 + 1)`) and extrapolated to withheld pairs in
  < 120 ms; non-fitting demonstrations are refused (random permutation, sin(x)+x^2).
- 100% in-sample replay; periodic sequences extrapolated beyond trained length
  and across unseen phases (4/4, 4/4, 12/12, 4/4) — transition recombination.
- Held-out code files: top-1 58.11% vs uniform 0.011%; PPL 416.8 vs 9196.
- VSA lookup ~5.18 µs/query, 0.000% false positives at the 0.40 gate.
- Known-family function interpolation NRMSE 0.029% (fails outside the hull and
  on different-curvature families — measured).
- LLM 108/108 and E2E 47/47 pass on the pushed commit.

Every claim outside this list still has no executable proof behind it, and the
ones above that read as marketing should stop being repeated until they do.

# Mechanism note: 1.58-bit projections + resonance-gated VSA memory

Status: measured on the host, reproducible, **not** a SOTA claim.

Hardware: Intel Core i7-11800H (8C/16T), 32 GB RAM, no GPU used here.
Build: MSVC 2022, `/std:c++20 /O2 /arch:AVX512 /openmp`.

Reproduce:

```
build.bat ptq
tests\build_mechanism_bench.bat
tests\build_holdout_perplexity.bat
```

## 1. What is actually 1.58-bit

Only the **rotation** channels of a projection are ternary. Each rotation weight
is stored as two 1024-bit masks (`*_pos`, `*_neg`) plus one float scale and is
reconstructed as `0.5 * w + 0.5 * scale * sign`. The diagonal channel stays
float32.

Per GQA layer (3 projections x 2 channels):

| Channel | Bytes | Bits/weight |
|---|---|---|
| float32 diagonal | 4096 | 32.0 |
| ternary rotation | 260 | 2.03 (1.58 payload + scale) |
| whole layer, shipped | 13068 | 17.0 |
| whole layer, all-float32 | 24576 | 32.0 |

Layer footprint ratio: **1.88x smaller**; the rotation channel alone is
**15.75x smaller**. Any "16x compression" claim applies to the rotation channel
only, not to the layer.

## 2. Accuracy against an independent float32 reference

The reference re-implements the shipped operator (0.88/0.12 Q, 0.88/-0.12 K,
0.90/0.10 V projections, per-head rotations, sigmoid gate) in scalar float32.
100 random inputs, worst case:

- cosine similarity: **1.00000000**
- SQNR: **131.54 dB**

The ternary masks reproduce the float32 operator to float rounding. They do not
add accuracy; they preserve it.

## 3. Speed: the honest result

Same operands, 20,000 iterations after warm-up, best of three runs:

| Path | per step | throughput |
|---|---|---|
| shipped ternary AVX-512 block | 6.3-7.0 us | 143k-159k steps/s |
| scalar float32 reference | 2.8-3.1 us | 322k-357k steps/s |

**The ternary path is 0.42-0.45x the speed of the float32 reference on this
host.** At 1024 dimensions the shipped operator is a diagonal + rotation mix
(two non-zeros per output element), so bit-plane packing, mask unpacking and
per-head scale broadcasts dominate and the auto-vectorized scalar loop wins.
The ternary encoding is a memory play, not a throughput play, at this size.

## 4. Resonance-gated 512-bit memory (the strong half)

`vsa_db_query_vector` with the shipped 0.40 gate, 4096 random 512-bit vectors:

| Metric | Measured |
|---|---|
| false positives, 2000 unrelated probes | **0.000%** |
| one-bit-flip recall, 500 probes | **100.00%** |
| mean similarity on a one-bit flip | **0.9961** |
| probe throughput | **2.66e8 queries/s** |

The gate sits far outside the null distribution: independent balanced vectors
score 0 +/- 0.044 (~9 sigma), while a one-bit flip scores 0.996. This is the
usable part of the mechanism today: a sub-microsecond, typo-robust associative
lookup with a calibrated rejection boundary.

## 5. Held-out language metrics (no trained checkpoint claimed)

`tests\build_holdout_perplexity.bat` trains on repo source files and holds out
every 5th source file plus all prose. Perplexity is softmax over the exact
full-vocabulary ternary POPCNT logits, z-standardised per context; temperature
is fitted on the last 10% of the training text only.

| Split | PPL | Uniform baseline | Top-1 |
|---|---|---|---|
| train tail (fit) | 9.46 | 3197 | 82.24% |
| held-out source files (in-domain) | 146.81 | 3197 | 51.78% |
| held-out prose (cross-domain, 23% OOV) | 7468 | 3197 | 10.78% |

Reading: the head generalises to unseen code files (21.8x better than uniform,
half the tokens correct) but **does not transfer across domains**: on prose it
ranks above chance (10.8% vs 0.03%) yet is over-confident, so its perplexity is
worse than a uniform distribution. Any "language model" claim must be limited
to the in-domain code result; the cross-domain number is reported to keep the
claim falsifiable.

## 6. Scope

- No GPU numbers in this note; CUDA paths are measured by the HSSA M2 suite.
- The ternary speed result is specific to 1024-dim diagonal+rotation operators.
  It says nothing about dense GEMM-shaped ternary kernels, which are memory
  bound and were not implemented here.
- Held-out perplexity depends on the tokenizer's OOV handling (unknown words
  are dropped and counted); the 23% OOV rate on prose is part of the result,
  not a caveat.

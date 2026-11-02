# Mechanism note: ternary projections + resonance-gated VSA memory

Status: measured on the host, reproducible, **not** a SOTA claim.

Hardware: Intel Core i7-11800H (8C/16T), 32 GB RAM, no GPU used here.
Build: MSVC 2022, `/std:c++20 /O2 /arch:AVX512 /openmp`.

Reproduce:

```
build.bat ptq
tests\build_mechanism_bench.bat
tests\build_holdout_perplexity.bat
```

## 1. Storage: what is actually saved (nothing, as shipped)

`ModelBaseLayerWeights` (`src/model_mod.cpp:46-64`) stores **both** the float32
rotation weights and the ternary masks. Measured from the field layout:

| Component | Bytes |
|---|---|
| 8 x float32[1024] (diag, rot, delta-net) | 32768 |
| 6 x uint64[16] masks (q/k/v pos+neg) | 768 |
| 3 float scales | 12 |
| padded to 64-byte alignment | **33600** |
| float32-only layer | **32768** |

The shipped layer is **1.025x larger** than the float32-only layer. The
"1.58-bit" representation is an extra side channel, not a compressed store.

A masks-only variant (drop the float arrays, keep 780 B of masks + scales per
3 rotation channels instead of 12288 B) would be 15.75x smaller **for those
channels**, i.e. ~1.5x for the whole layer. That variant is not what ships and
was not benchmarked.

## 2. Accuracy against an independent float32 reference

The reference re-implements the shipped operator (0.88/0.12 Q, 0.88/-0.12 K,
0.90/0.10 V projections, per-head rotations, sigmoid gate) in scalar float32.
100 random inputs, worst case:

- cosine similarity: **1.00000000**
- SQNR: **131.54 dB**

**This is a tautology, not a compression result.** At initialization every
rotation weight is the same positive constant (`0.12`, `-0.12`, `0.10`), so the
masks quantize to all-ones and `0.5*w + 0.5*scale*sign(w) == w` exactly. The
check proves the shipped forward matches the documented operator; it says
nothing about fidelity under trained, non-uniform weights. No trained
checkpoint ships, so that number cannot be produced here.

## 3. Speed: the honest result

Same operands, 20,000 iterations after warm-up; the output is consumed through a
`volatile` sink so the loop cannot be eliminated:

| Path | per step | throughput |
|---|---|---|
| shipped AVX-512 block | 6.1-6.9 us | 145k-165k steps/s |
| scalar float32 reference | 2.8-3.0 us | 337k-360k steps/s |

**The shipped path is 0.42-0.46x the speed of the scalar float32 reference on
this host.** At 1024 dimensions the operator is a diagonal + rotation mix (two
non-zeros per output element), so mask extraction, the 50/50 float/ternary
blend and per-head scale broadcasts dominate, and the auto-vectorized scalar
loop wins. The ternary encoding is a memory idea that, as shipped, does not
save memory either.

## 4. Resonance-gated 512-bit memory (the strong half)

`vsa_db_query_vector` with the shipped 0.40 gate, 4096 random 512-bit vectors:

| Metric | Measured |
|---|---|
| false positives, 2000 unrelated probes | **0.000%** |
| one-bit-flip recall, 500 probes | **100.00%** |
| mean similarity on a one-bit flip | **0.9961** |
| dependent query latency | **5.18 us** (193k queries/s) |

Latency is measured as a dependent chain (each probe derives from the previous
result) because an earlier version of this benchmark passed `out_sim = nullptr`,
hit the early `return -1` guard and reported 4 ns/query that never happened.

The gate sits far outside the null distribution: independent balanced vectors
score 0 +/- 0.044 (~9 sigma), a one-bit flip scores 0.996. Both stated metrics
follow from that calibration, so they are a **calibration check**; they cannot
fail unless the gate or the similarity metric changes. The usable property is a
5 us, typo-robust associative lookup with a calibrated rejection boundary.

## 5. Held-out language metrics (no trained checkpoint claimed)

`tests\build_holdout_perplexity.bat` trains on repository source/header/test
files and holds out every 5th file (9 files, 170 KB) plus all prose. Perplexity
is softmax over the exact full-vocabulary ternary POPCNT logits at **one scalar
temperature** fitted on the last 10% of the training text. OOV tokens split the
sequence, so no transition is scored across an unseen token.

| Split | Mean NLL | PPL | Uniform | Top-1 | OOV |
|---|---|---|---|---|---|
| train tail (fit) | 2.518 | 12.4 | 9037 | — | — |
| held-out source files | 6.201 | **493.2** | 9037 | 57.0% | 7.3% |
| held-out prose | 12.329 | **226219** | 9037 | 16.8% | 16.5% |
| in-domain at T=1 (control) | 3140.9 | **inf** | 9037 | 57.0% | 7.3% |

Reading:

- In-domain, the head is **18.3x better than uniform** on unseen files and gets
  57% of next tokens exactly right.
- Cross-domain, it ranks far above chance (16.8% vs 0.011%) but is
  over-confident: PPL is **25x worse than uniform**. It does not transfer.
- At T=1 the distribution is so peaked that mean NLL is 3140 and PPL overflows
  double. Temperature scaling is what makes any finite number exist at all;
  that is a property of the result, not a preprocessing trick.

Any "language model" claim must be limited to the in-domain code result.

## 6. Scope

- No GPU numbers in this note; CUDA paths are measured by the HSSA M2 suite.
- The speed result is specific to 1024-dim diagonal+rotation operators. It says
  nothing about dense GEMM-shaped ternary kernels, which are memory bound and
  were not implemented here.
- OOV handling (unknown words split the sequence) is part of the measurement;
  the 16.5% prose OOV rate caps how much of that corpus is scored at all.

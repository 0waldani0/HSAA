# The measurement package

Every number this project quotes about the count-based backoff is produced by
one of the scripts in this directory, on the corpora in `data/`. Nothing here
is part of the shipped build; it is the instrument the claims rest on.

## The corpora (`data/`)

| file | what it is | how it was made |
|---|---|---|
| `bpe_train.txt` / `bpe_holdout.txt` | the subword split (train / held-out) that every backoff number is measured on | prose lines harvested from a local Python installation, split BY SOURCE FILE so no near-duplicate leaks across the boundary |
| `samedomain_train.txt` / `samedomain_holdout.txt` | the same-domain (word-level) split | the same harvest, a different split |
| `real_corpus.txt` | the 801-line reference corpus used by the regression guard | the project's reference corpus |

The harvest depends on the machine's Python installation, so the DERIVED
corpora are shipped here: with them the numbers below reproduce exactly on any
machine.

## The native C++20 / C11 harnesses

| source | what it measures | build / command |
|---|---|---|
| `native_measur_runner.cpp` | causal full-prefix count baseline on `bpe_train.txt` / `bpe_holdout.txt` and `samedomain_train.txt` / `samedomain_holdout.txt`, plus `PopCountGenerativeLLM` verification on `llm_text_code_corpus.txt` (zero Python) | link with `popcnt_brain` |
| `acceptance_harness.cpp` | includes the engine source directly, trains on the train split, then scores the held-out half (accuracy, E[accepted(k)] by depth, and the candidate-set coverage) | native C++20 build |
| `assoc_calib.cpp` | measures recall and false-positive rate of the 512-bit trigram SimHash lexical fallback across typo classes | `build_assoc_calib.bat` |
| `capi_probe.c` | verifies C11 ABI compatibility, lifecycle, fact storage, conflict checks, and lexical fallback | `build_capi_probe.bat` |

## The measured results (the shipped engine)

Re-measured on this host (i7-11800H, 32 GB RAM; commit under test recorded in
docs/CLAIM_PROOF_MATRIX.md) with the native `acceptance_harness` (includes the
shipped engine source; trains on `data/bpe_train.txt`, scores
`data/bpe_holdout.txt`; seed 424242, 6 epochs, vocab 4166, 40,000 train
sentences / 576,806 transitions, 12,000 hold-out sentences / 175,876 transitions):

| metric | old published | re-measured |
|---|---|---|
| holdout accuracy | 36.5382% | **38.3202%** |
| shortlist coverage | 58.9347% | **60.9998%** |
| E[accepted] k=1 / 2 / 3 | 0.3654 / 0.5919 / 0.7591 | **0.3832 / 0.6278 / 0.8097** |
| k=4 .. 12 draft acceptance | 0.9541 / 1.0728 / 1.1727 / 1.2581 / 1.3322 / 1.3970 / 1.4533 / 1.5022 / 1.5440 | same scale, +1.7 straight across |
| draft cost | 10,395 ns/token | **1,770 ns/token** |
| train time (6 epochs) | n/a | 6,714 ms |
| conditional accuracy depth 1 / 5 / 12 | n/a | 0.383 / 0.822 / 0.853 |
| pages over 16 slots | 21,946 | **0 (zero)** |

The old published table cannot be reproduced from this tree: the scripts that
produced it (`regress_par.py`, `sweep_slots.py`, `baselines_word_causal.py`,
`corpus_regression.py`) are absent; only `__pycache__` remnants remain and the
native harness measures a different path than the published count-mixed lane.
Either the shipped engine changed after those numbers were taken, or the
published table measures code that is not shipped. Both need the author's
answer; until then, treat the old numbers as historical, not current.

## The word-level (same-domain) split, in ONE population — NOT REPRODUCIBLE

(Measured with `baselines_word_causal.py` / `sweep_slots.py`, both absent from
this tree. Kept as historical numbers; do not quote them as current.)


The word-level comparison was invalid until it was re-measured: 38.462% was
measured on the TRIGRAM positions, while the engine scores the positions from
the second token on, with the whole prefix as its context. On the engine's own
positions (63,432; epochs 6, seed 424242):

| what | accuracy |
|---|---|
| the plain engine (no count backoff) | 25.5092% |
| the count-mixed engine, K=256 | 36.3350% |
| **the count-mixed engine, K=32 (shipped)** | **36.3365%** |
| the causal chain, trigram window | 34.432% |
| the causal chain, full prefix (the engine's context) | 32.211% |

The count mixture is worth **+10.83 points** on this corpus, and the engine ends
up **above** the count chain: +1.90 points over the trigram-window chain and
+4.13 over the chain that gets the engine's own context. Reproduce with
`python baselines_word_causal.py` and `python sweep_slots.py --word --word2`.

## The co-location limit (16 bottleneck slots per page), swept — NOT REPRODUCIBLE

(Measured with `sweep_slots.py`, absent from this tree; the native harness reports
zero over-capacity pages. Kept as historical numbers; do not quote as current.)


A micro-expert page holds D_BOTTLENECK bottleneck neurons and the STE writes
into them without eviction. On the subword held-out telemetry (576,806 write
events over 248,656 distinct pages): 21,946 pages hold more than 16
associations, 6,583 more than 32, 2,298 more than 64, 943 more than 128; the
hottest page takes 9,953 writes. The sweep widens the page in a scratch copy
of the engine (identical corpus, 6 epochs, seed 424242):

| slots | accuracy | pages over capacity | bank map | master accumulators |
|---|---|---|---|---|
| 16 (shipped) | 36.5382% | 21,946 | 1.07 GB | 4.1 GB |
| 32 | 36.7191% | 6,583 | 2.15 GB | 8.1 GB |
| 64 | 36.8248% | 2,298 | 4.3 GB | 16.3 GB |
| 128 | 37.0824% | 943 | 8.6 GB | 32.6 GB |

Coverage is identical in every build (58.9347%): the slot count changes the
granularity of what the pages remember, not how many of them are reachable.

Reading: widening the page recovers only **+0.54 points at 8 slots' worth of
extra memory** (the bank map grows 1.07 GB -> 8.6 GB and the master
accumulators -- `down_acc`/`up_acc`, an int8 per slot per dimension per page
-- grow 4.1 GB -> 32.6 GB, i.e. past the 32 GB of this host, where the 128-slot
lane already takes a large paging hit; its train time is an upper bound, not a
clean reference). Draft cost rises by **+10.6% at twice the slots** (10,395.1
-> 11,495.9 ns/token, the cleanest same-run pair; every measured speed carries
several percent of multi-lane noise), more beyond. The
telemetry across the builds cross-checks exactly (every "over N" bucket agrees
with the next build in the sweep), so the accuracy trend is real -- it is
simply a second-order lever, and 16 slots stay shipped.

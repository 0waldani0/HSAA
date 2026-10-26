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

Corpus `bpe_train.txt` / `bpe_holdout.txt`, 6 epochs, seed 424242; the
coverage is measured over the engine's own 175,876 held-out positions.

| configuration | accuracy | coverage | E[accepted(1..3)] |
|---|---|---|---|
| plain (no backoff) | 28.3876% | -- | 0.2839 / 0.4597 / 0.5896 |
| weight 64, k=32 | 34.5084% | 58.9347% | 0.3451 / 0.5616 / 0.7226 |
| weight 256, k=32 | 35.6478% | 58.9347% | 0.3565 / 0.5790 / 0.7434 |
| **weight 1024, k=32 (shipped)** | **36.5382%** | **58.9347%** | **0.3654 / 0.5919 / 0.7591** |
| weight 4096, k=32 | 35.7252% | 58.9347% | 0.3573 / 0.5655 / 0.7107 |
| weight 1024, k=4 | 36.0322% | 48.4768% | 0.3603 / 0.5767 / 0.7305 |
| weight 1024, k=64 | 36.5246% | 60.5006% | 0.3652 / 0.5920 / 0.7598 |
| weight 1024, k=256 | 36.4097% | 62.0949% | 0.3641 / 0.5900 / 0.7567 |
| weight 1024, k=1024 | 36.4097% | 62.4724% | 0.3641 / 0.5900 / 0.7567 |

The count-weight sweep peaks at 1024 and the shortlist size peaks at k=32:
a bigger k raises the coverage (58.93% -> 62.47%) but LOWERS the accuracy
(36.5382% -> 36.4097%), because the extra candidates are wrong more often
than right.

The Python baselines on the same files:

| chain | on the trigram positions (163,876) | on the engine's positions (175,876) |
|---|---|---|
| trigram | 27.717% | 27.468% |
| bigram fallback | 6.449% | 6.153% |
| unigram fallback | 1.335% | 1.244% |
| **causal chain total** | **35.502%** | **34.865%** |
| the non-causal skip level (uses the future gold token) | 12.733% | -- |
| the reference chain that included it | 46.899% | -- |

The mixture (36.5382%) beats the causal chain on the engine's own population
by 1.673 points. The word-level chain: trigram 15.413%, bigram 8.425%,
unigram 14.625%, causal total 38.462% (the non-causal skip level 6.185%, the
old reference total 30.022%).

The reference corpus (`corpus_regression.py`): 91.600 / 91.455 / 91.648%
(12 epochs seed 424242, 12 epochs seed 1337, 6 epochs seed 424242), the
guard's floor is 91.455% with 705 live-replay errors.

## The word-level (same-domain) split, in ONE population

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

## The co-location limit (16 bottleneck slots per page), swept

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

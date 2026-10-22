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

## The scripts

| script | what it measures | command |
|---|---|---|
| `baselines_bpe_causal.py` | the causal count chain on the subword split, decomposed by level, AND on the ENGINE's own population (175,876 positions) so it can be compared with the engine | `python baselines_bpe_causal.py` |
| `baselines_word_causal.py` | the same decomposition for the word-level (same-domain) split | `python baselines_word_causal.py` |
| `measure_all.py` | the engine itself: the count-weight sweep, the candidate-set coverage curve over k, and the plain-vs-backoff A/B. Each configuration is built from a fresh copy of the engine into its own scratch directory and runs on one core, six at a time | `python measure_all.py` |
| `corpus_regression.py` | the 801-line reference corpus at three (epochs, seed) settings; the guard fails if the accuracy drops below the recorded floor | `python corpus_regression.py` |
| `acceptance_harness.cpp` | the harness `measure_all.py` compiles: it includes the engine source directly, trains on the train split, then scores the held-out half (accuracy, E[accepted(k)] by depth, and the candidate-set coverage) | compiled by `measure_all.py` |
| `regress_par.py` | the FULL regression, with the independent steps in parallel: the two gates, the engine library, the self-test, the four suites, the C-API probe, the native e2e suite and the corpus group | `python regress_par.py` |
| `regress_fast.py` | the FAST lane: the same mechanical checks with incremental builds, parallel groups and one short corpus training. It does NOT include the full e2e suite or the full-epoch corpus accuracy -- it is a smoke lane, not the release gate | `python regress_fast.py` |
| `corpus_group.py` | the reference corpus, the state round trip and the state-format diagnostic, all from ONE training (the state file it writes is reused by the other two facts) | `python corpus_group.py` |
`measure_all.py` needs MSVC (`vswhere` -> `vcvars64`) and writes its report to
`%TEMP%\opencode\measure_all_report.txt`.

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

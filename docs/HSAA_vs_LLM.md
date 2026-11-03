# HSAA vs a general-purpose LLM (analytical, not measured here)

Premise: this host has no LLM. The user chose the analytical comparison over
installing a model (would download ~2 GB) or using an external API (needs a
key). Numbers under **HSAA** are measured on this machine; numbers under
**LLM** are mechanism/capability estimates from published results and are NOT
re-measured here - they are marked `~` and must not be quoted as measurements.

## Task-by-task

| Task | HSAA (measured) | LLM (analytical) | Winner |
|---|---|---|---|
| In-sample replay (trained continuations) | 100% | 100 | tie |
| Held-out code files, next-token top-1 | 58.11% (PPL 416.8 vs uniform 9196) | ~strong; published code models rank far above chance, but carry the same 22 pp-below-replay gap phenomenon | LLM |
| Held-out prose, cross-domain | 15.89% (PPL 255312, 27.8x worse than uniform) | strong domain transfer | LLM |
| Periodic extrapolation (2/3/5-cycle, unseen length/phase) | 100% (4/4, 4/4, 12/12 near, 4/4 far) | ~100% | tie |
| Withheld-pair rule application (successor e->f/h->i, addition 2+3=5/3+4=7) | **0/2** | ~2/2 | LLM |
| Zero-shot unseen composite algorithm (Dijkstra on RBTree from linear/binary-search corpus) | no - byte-identical canned output with 0-token Ford-Fulkerson bare prompt; fixed template | ~produces working code, quality varies, still needs review | LLM |
| Symbolic formula induction | none (numeric API only) | ~partial; LLMs can guess family then verify with a tool | LLM |
| In-context learning of an unseen relation from examples | none | ~yes, with distortion at scale/rate | LLM |
| Finding a *new* solution (not in distribution) | architecturally impossible: no novelty mechanism; every answer is a recombination of observed transitions or a static template | yes - search + sampling; still unreliable without verification | LLM |
| Deterministic greedy decoding of a fixed prompt | exact byte-identical output, reproducible | ~rarely bit-reproducible across stacks even at temp 0 | HSAA |
| Retrieval / associative lookup latency | ~5.18 us/query, rejection gate 0.000% FP | ms-class inference, plus network for API | HSAA |
| Memory/footprint | on-device CPU only | GBs of weights + accelerator | HSAA |
| Coverage guard 58.93/60.99 (measur) | measured; count chain 34.87% (unreproducible) | n/a | HSAA over the measured backoff, not a win over LLMs |

## Who "works better"

- **On what HSAA was measured to do**, a 2024-class 3B LLM would almost
  certainly score higher on the plain completion metrics, and not because it
  knows more code - because its attention can compose across arbitrary context,
  not only recombine observed transitions over `L=1..16`.
- **On induction and new solutions**, the gap is qualitative, not incremental.
  HSAA has no mechanism whose output was not present somewhere in training or
  in a static template. An LLM is not magic, but it is the only one of the two
  with a sampled candidate generator that can propose candidates it was never
  shown; pairing it with the HSAA-style 5 us rejection/verification gate is a
  sensible hybrid, not an either/or.
- **On determinism, offline latency, footprint and auditability of a narrow
  in-domain shortlist**, HSAA wins. Those are deployment properties, not
  intelligence properties.

## Honest limits of this comparison

- No live LLM run happened. The `~` cells are estimates; a fair live comparison
  on this machine is one `ollama pull` away and would use exactly the two
  existing batteries (`bench_rule_generalization.exe`, `bench_claim_falsify.exe`)
  extended with a prompt adapter, plus held-out PPL scoring per model.
- HSAA's own numbers are real; LLM numbers here must not be presented as a
  measurement result of this project.

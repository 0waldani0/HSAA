# E2E Test Infra: Native C++/CUDA Hybrid Neural Research Laboratory

## Test Philosophy
- Opaque-box, requirement-driven. Derived strictly from `ORIGINAL_REQUEST.md` and user-facing specifications without dependence on implementation design details.
- Methodology: Category-Partition + Boundary Value Analysis (BVA) + Pairwise Combinatorial Testing + Real-World Workload Testing.
- Pure Native Execution: Test runner and tests must compile and run on Windows MSVC 2022 + CUDA with zero Python runtime dependencies.

## Feature Inventory & Test Coverage Requirements (N=16 features)
| # | Feature | Requirement Source | Tier 1 (>=5) | Tier 2 (>=5) | Tier 3 (Pairwise) | Tier 4 (Workload) |
|---|---------|-------------------|:------------:|:------------:|:-----------------:|:-----------------:|
| 1 | F1.1 Unified Build Orchestrator | R1 Acceptance | 5 | 5 | Pair 1 | App 1 |
| 2 | F1.2 CUDA sm_86 Tensor Engine | R1 Acceptance | 5 | 5 | Pair 2 | App 2 |
| 3 | F1.3 AVX-512 VSA Computational Core | R1 Acceptance | 5 | 5 | Pair 3 | App 3 |
| 4 | F1.4 Hybrid CPU-GPU Interop Bridge | R1 Acceptance | 5 | 5 | Pair 4 | App 4 |
| 5 | F2.1 Dynamic VsaMemoryDatabase | R2 Acceptance | 5 | 5 | Pair 5 | App 5 |
| 6 | F2.2 True VSA Fact Engine | R2 Acceptance | 5 | 5 | Pair 6 | App 6 |
| 7 | F2.3 Online Memory Mutation API | R2 Acceptance | 5 | 5 | Pair 7 | App 7 |
| 8 | F2.4 Multi-Instance Hot-Swapping | R2 Acceptance | 5 | 5 | Pair 8 | App 8 |
| 9 | F3.1 Dynamic Precision Switching | R3 Acceptance | 5 | 5 | Pair 9 | App 1 |
| 10 | F3.2 Online Weight Plasticity | R3 Acceptance | 5 | 5 | Pair 10 | App 2 |
| 11 | F3.3 Layer Block Polymorphism | R3 Acceptance | 5 | 5 | Pair 11 | App 3 |
| 12 | F3.4 Architecture Modification API | R3 Acceptance | 5 | 5 | Pair 12 | App 4 |
| 13 | F4.1 Microsecond Profiler | R4 Acceptance | 5 | 5 | Pair 13 | App 5 |
| 14 | F4.2 Workstation Hardware Tracker | R4 Acceptance | 5 | 5 | Pair 14 | App 6 |
| 15 | F4.3 Numerical Accuracy Metrics | R4 Acceptance | 5 | 5 | Pair 15 | App 7 |
| 16 | F4.4 Boundary & Adversarial Suite | R4 Acceptance | 5 | 5 | Pair 16 | App 8 |

## Test Architecture
- Test Runner: Native binary `build/tests/e2e_runner.exe` invoked by `build.bat test`.
- Exit semantics: Exit code 0 on all tests passing; non-zero exit code with detailed failure report on any failure.
- Directory layout:
  - `tests/e2e_test_runner.cpp`: Main test harness and reporting framework.
  - `tests/tier1_feature_tests.cpp`: 80 Tier 1 unit & functional feature tests.
  - `tests/tier2_boundary_tests.cpp`: 80 Tier 2 boundary, corner case, and stress tests.
  - `tests/tier3_pairwise_tests.cpp`: 16 Tier 3 cross-feature interaction tests.
  - `tests/tier4_workload_tests.cpp`: 8 Tier 4 real-world application scenario benchmarks.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Target Metric |
|---|----------|--------------------|---------------|
| 1 | Hybrid Associative Retrieval + Tensor Projection | F1.3, F1.4, F1.2, F4.1 | Forward step < 15 ms, Cosine > 0.99 |
| 2 | Online Continuous Knowledge Mutation & Hot-Swapping | F2.1, F2.3, F2.4, F4.2 | Zero memory leak, swap time < 100 us |
| 3 | Dynamic Quantization Cascade (2-bit Draft -> 4-bit Target) | F3.1, F1.2, F4.1, F4.3 | Zero VRAM reallocation, delta eval < 2 ms |
| 4 | Online Plastic Adapter Training (TT16 SVD update) | F3.2, F1.4, F4.1, F4.3 | SVD rank update < 25 us, loss reduction > 50% |
| 5 | Execution Block Hot-Swapping (GQA vs DeltaNet vs VSA) | F3.3, F1.2, F1.3, F4.1 | Seamless layer toggle without process restart |
| 6 | Heavy Knowledge Graph Conflict Resolution at Scale | F2.2, F2.3, F4.1, F4.2 | 10,000 facts indexed, recall latency < 5 us |
| 7 | Multi-Model Checkpoint Coexistence in VRAM | F1.2, F2.4, F4.2 | 2 active models concurrently in < 3,500 MB VRAM |
| 8 | Stress & Adversarial Robustness (0-1-N-NULL-BOUND) | F4.4, F1.1-F4.3 | Zero crash, graceful error return on invalid inputs |

## Coverage Thresholds
- Tier 1: >= 5 tests per feature (Total: 80 tests)
- Tier 2: >= 5 tests per feature (Total: 80 tests)
- Tier 3: >= 16 pairwise interaction tests (Total: 16 tests)
- Tier 4: >= 8 realistic workload application scenarios (Total: 8 tests)
- **Minimum Test Suite Total: 184 tests**

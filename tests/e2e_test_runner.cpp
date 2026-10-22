#include "test_harness.h"

// Forward declaration of test suites to ensure translation units link
void register_tier1_tests();
void register_tier2_tests();
void register_tier3_tests();
void register_tier4_tests();

int main(int argc, char* argv[]) {
    // Command-line options
    int filter_tier = 0; // 0 = all
    std::string filter_feature = "";
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--tier" && i + 1 < argc) {
            filter_tier = std::atoi(argv[++i]);
        } else if (arg == "--feature" && i + 1 < argc) {
            filter_feature = argv[++i];
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: e2e_runner [options]\n"
                      << "Options:\n"
                      << "  --tier <1-4>       Filter tests by tier\n"
                      << "  --feature <F...>   Filter tests by feature (e.g. F1.3)\n"
                      << "  --verbose, -v      Enable verbose output\n"
                      << "  --help, -h         Show this help message\n";
            return 0;
        }
    }

    // The suites register themselves. Without these calls the registry stays
    // EMPTY and the runner reports a green "0 tests executed" -- a silent
    // no-op, which is exactly what this runner did for a long time.
    register_tier1_tests();
    register_tier2_tests();
    register_tier3_tests();
    register_tier4_tests();

    auto& all_tests = TestRegistry::instance().get_tests();

    std::cout << "===============================================================================\n";
    std::cout << "  E2E TEST RUNNER - Native C++/CUDA Hybrid Neural Research Laboratory\n";
    std::cout << "===============================================================================\n";
    std::cout << "Total registered test cases: " << all_tests.size() << "\n";
    if (filter_tier > 0) std::cout << "Filtering by Tier: " << filter_tier << "\n";
    if (!filter_feature.empty()) std::cout << "Filtering by Feature: " << filter_feature << "\n";
    std::cout << "-------------------------------------------------------------------------------\n";

    size_t ran = 0;
    size_t passed = 0;
    size_t failed = 0;

    double total_start_us = qpc_time_us();
    size_t initial_ram_mb = get_process_ram_mb();
    size_t initial_vram_mb = get_cuda_vram_mb();

    int current_tier = -1;

    for (auto& tc : all_tests) {
        if (filter_tier > 0 && tc.tier != filter_tier) continue;
        if (!filter_feature.empty() && tc.feature != filter_feature) continue;

        if (tc.tier != current_tier) {
            current_tier = tc.tier;
            std::cout << "\n>>> Tier " << current_tier << " Tests <<<\n";
        }

        ran++;
        double t0 = qpc_time_us();
        try {
            tc.func();
            tc.passed = true;
            passed++;
        } catch (const TestFailureException& ex) {
            tc.passed = false;
            tc.failure_message = ex.what();
            tc.failure_file = ex.file;
            tc.failure_line = ex.line;
            failed++;
        } catch (const std::exception& ex) {
            tc.passed = false;
            tc.failure_message = std::string("Unexpected exception: ") + ex.what();
            failed++;
        } catch (...) {
            tc.passed = false;
            tc.failure_message = "Unknown fatal exception caught";
            failed++;
        }
        double t1 = qpc_time_us();
        tc.elapsed_us = t1 - t0;

        if (tc.passed) {
            if (verbose) {
                std::cout << "  [PASS] [" << tc.feature << "] " << tc.name
                          << " (" << std::fixed << std::setprecision(1) << tc.elapsed_us << " us)\n";
            } else {
                std::cout << ".";
                if (ran % 50 == 0) std::cout << " [" << ran << "]\n";
            }
        } else {
            if (!verbose) std::cout << "\n";
            std::cout << "  [FAIL] [" << tc.feature << "] " << tc.name << "\n"
                      << "         " << tc.failure_message << "\n";
            if (tc.failure_line > 0) {
                std::cout << "         at " << tc.failure_file << ":" << tc.failure_line << "\n";
            }
        }
    }

    if (!verbose && (ran % 50 != 0)) {
        std::cout << "\n";
    }

    double total_elapsed_ms = (qpc_time_us() - total_start_us) / 1000.0;
    size_t final_ram_mb = get_process_ram_mb();
    size_t final_vram_mb = get_cuda_vram_mb();

    std::cout << "\n===============================================================================\n";
    std::cout << "  EXECUTION SUMMARY\n";
    std::cout << "===============================================================================\n";
    std::cout << "Tests Ran:    " << ran << " / " << all_tests.size() << "\n";
    std::cout << "Passed:       " << passed << " (" << (ran > 0 ? (passed * 100 / ran) : 0) << "%)\n";
    std::cout << "Failed:       " << failed << "\n";
    std::cout << "Duration:     " << std::fixed << std::setprecision(2) << total_elapsed_ms << " ms\n";
    std::cout << "Host RAM:     " << final_ram_mb << " MB (delta: "
              << (int)(final_ram_mb - initial_ram_mb) << " MB, ceiling: " << MAX_WORKSTATION_RAM_MB << " MB)\n";
    std::cout << "GPU VRAM:     " << final_vram_mb << " MB (ceiling: " << MAX_WORKSTATION_VRAM_MB << " MB)\n";
    std::cout << "-------------------------------------------------------------------------------\n";

    if (failed == 0 && ran >= 184) {
        std::cout << "[RESULT] ALL TESTS PASSED! E2E Suite satisfies 100% of TEST_INFRA.md.\n";
        return 0;
    } else if (failed == 0 && ran > 0) {
        std::cout << "[RESULT] PASS (Partial run: " << ran << " tests executed)\n";
        return 0;
    } else if (ran == 0) {
        // A run that executed NOTHING is not a pass: it means the suites
        // registered nothing (or the filter matched nothing). Reporting it
        // as green is how a broken suite hides.
        std::cout << "[RESULT] FAILED: no test was executed -- the suites registered "
                     "nothing (or the filter matched nothing).\n";
        return 2;
    } else {
        std::cout << "[RESULT] FAILED: " << failed << " test(s) failed!\n";
        return 1;
    }
}

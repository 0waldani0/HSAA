// Can the second brain be TRAINED for a task, and how fast is the decision it
// then makes?  The engine carries two different training mechanisms, and this
// test gates both of them with measured numbers.
//
//   GATE 1 - a policy that is NOT the shipped demo: the signed-popcnt STE
//            relay trainer, given its own state encoder and its own teacher
//            (a 16-way choice), must learn the mapping exactly and decide in
//            microseconds.  This is the "train it for instant selection" case.
//   GATE 2 - the shipped Snake demo of the same trainer must keep its
//            accuracy over the whole sensor space.
//   GATE 3 - the relational memory used as a LOOKUP: the chain terminal must
//            be resolved without any generation, and the lookup must be far
//            cheaper than the same question through the generation path.
//
// The engine source is included directly (as the shipped self-test does)
// because the trainer and the relay are internal classes, not C-ABI.
#define BUILDING_POPCNT_BRAIN_LIB
#include "popcnt_brain.cpp"

#include <cassert>

// A gate that cannot be compiled away. `assert` is a no-op in a release
// build (NDEBUG), which would turn this whole file into a test that always
// passes; the repository's other tests use assert, but these two gates are
// about the mechanism itself, so they use an explicit check.
#define GATE_CHECK(cond, msg)                                                      \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::fprintf(stderr, "[GATE FAILED] %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
            std::fflush(stderr);                                                  \
            std::abort();                                                         \
        }                                                                          \
    } while (0)
#include <cstdio>
#include <string>
#include <vector>

// A decision: relay forward + the popcnt argmax over the candidate list.
static int decide(
    const MultiLayerExpertRelayEngine& relay,
    const NativeSublinearLLM* model,
    const int8_t* x,
    const int* cand_ids,
    int n_cand,
    int8_t* y
) {
    relay.forward_expert_relay(x, y, NUM_RELAY_LAYERS, nullptr, true);
    uint64_t sign_bits[WORDS_PER_VEC];
    uint64_t mag_planes[NUM_BIT_PLANES][WORDS_PER_VEC];
    pack_activation_bitplanes(y, sign_bits, mag_planes);
    int best = 0;
    int32_t best_logit = -1000000;
    for (int c = 0; c < n_cand; ++c) {
        const uint64_t* vp = &model->vocab_pos[cand_ids[c] * WORDS_PER_VEC];
        const uint64_t* vn = &model->vocab_neg[cand_ids[c] * WORDS_PER_VEC];
        int32_t s = ternary_popcnt_dot_512(vp, vn, sign_bits, mag_planes);
        if (s > best_logit) { best_logit = s; best = c; }
    }
    return best;
}

// Microseconds per call, the best of `reps` runs of `n_calls` calls.
template <typename F>
static double best_per_call_us(F&& fn, int n_calls, int reps) {
    double best = 1e18;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n_calls; ++i) fn();
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / n_calls;
        if (us < best) best = us;
    }
    return best;
}

int main() {
    std::filesystem::create_directories("build");
    std::printf("[*] PopCount Second Brain: task-training gates\n");

    // ------------------------------------------------------------- GATE 1 --
    // A 16-way choice with its own encoder and its own teacher. Nothing of
    // the Snake demo is reused except the trainer.
    {
        AutoMappedWeightBank guard("build/test_selector_train.bin");
        MappedWeightBank* bank = guard.get();
        GATE_CHECK(bank != nullptr, "bank creation failed");;
        bank->clear_master_cache();

        NativeTokenizer tk;
        int cand[16];
        for (int i = 0; i < 16; ++i) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "CHOICE_%02d", i);
            cand[i] = tk.add_word(nm);
        }
        populate_expanded_4096_vocabulary(tk);

        NativeSublinearLLM model(bank, static_cast<int>(tk.id_to_word.size()), 4, false, 9001);
        MultiLayerExpertRelayEngine relay(bank, 0x5A4E2026ULL);

        std::vector<std::pair<std::vector<int8_t>, int>> experiences;
        std::vector<std::vector<int8_t>> states;
        uint64_t s = 0x1234567890ABCDEFULL;
        for (int i = 0; i < 16; ++i) {
            std::vector<int8_t> st(D_MODEL);
            for (size_t k = 0; k < D_MODEL; ++k) {
                s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
                st[k] = ((s * 0x2545F4914F6CDD1DULL) & 0x100ULL) ? 12 : -12;
            }
            states.push_back(st);
            experiences.push_back({st, cand[i]});
        }

        auto t0 = std::chrono::steady_clock::now();
        relay.train_chained_expert_relay_ste(
            experiences, model.embeddings.data(), model.vocab_pos.data(),
            model.vocab_neg.data(), model.vocab_size, 4, 3);
        const double train_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

        int ok = 0;
        int8_t x[D_MODEL], y[D_MODEL];
        for (int i = 0; i < 16; ++i) {
            if (decide(relay, &model, states[i].data(), cand, 16, y) == i) ++ok;
        }
        GATE_CHECK(ok == 16, "the STE relay did not learn the 16-way task");;

        const double us = best_per_call_us([&]() {
            decide(relay, &model, states[0].data(), cand, 16, y);
        }, 2000, 20);
        // Measured on this host: 1.4 us per decision. The ceiling is 140x that,
        // so it catches a pathological regression without failing under load.
        GATE_CHECK(us < 200.0, "the decision cost is no longer in the microsecond range");;

        std::printf("[+] GATE 1: 16-way task learned exactly (16/16), trained in %.1f ms, "
                    "decision %.3f us\n", train_ms, us);
    }

    // ------------------------------------------------------------- GATE 2 --
    // The shipped Snake demo of the same trainer.
    {
        AutoMappedWeightBank guard("build/test_selector_snake.bin");
        MappedWeightBank* bank = guard.get();
        GATE_CHECK(bank != nullptr, "bank creation failed");;
        bank->clear_master_cache();

        SnakePopcntAgent agent(bank);
        size_t pages = 0;
        const double train_ms = agent.train_in_snake_gym_ste(&pages);

        int ok = 0, total = 0;
        int8_t x[D_MODEL], y[D_MODEL];
        for (int g = 0; g < 9; ++g) {
            for (int dm = 0; dm < 16; ++dm) {
                agent.encode_state_from_sensors(g, dm, x);
                const int pred = decide(agent.relay_engine, agent.model, x,
                                        agent.action_token_ids, 4, y);
                if (pred == agent.compute_reward_optimal_action_for_training(g, dm)) ++ok;
                ++total;
            }
        }
        // Measured: 96.5% over the 144 sensor states (139/144). The gate is 90%.
        GATE_CHECK(ok * 100 >= total * 90, "the Snake selector lost accuracy");;

        const double us = best_per_call_us([&]() {
            agent.encode_state_from_sensors(3, 5, x);
            decide(agent.relay_engine, agent.model, x, agent.action_token_ids, 4, y);
        }, 2000, 20);
        GATE_CHECK(us < 200.0, "the Snake decision cost is no longer in the microsecond range");;

        std::printf("[+] GATE 2: Snake selector %d/%d (%.1f%%), trained in %.1f ms, "
                    "decision %.3f us, %zu pages touched\n",
                    ok, total, 100.0 * ok / total, train_ms, us, pages);
    }

    // ------------------------------------------------------------- GATE 3 --
    // The relational memory as a lookup: no generation at all.
    {
        AutoMappedWeightBank guard("build/test_selector_lookup.bin");
        MappedWeightBank* bank = guard.get();
        GATE_CHECK(bank != nullptr, "bank creation failed");;
        bank->clear_master_cache();

        PureNeuralReasoningEngine engine(bank, UNIFIED_REASONING_CORPUS);

        // "Where does the signal end up if the starting node is модуль_парсер ?"
        const std::string q =
            "\xd0\x9a\xd1\x83\xd0\xb4\xd0\xb0 \xd0\xb2 \xd0\xb8\xd1\x82\xd0\xbe\xd0\xb3\xd0\xb5 "
            "\xd0\xbf\xd0\xbe\xd0\xbf\xd0\xb0\xd0\xb4\xd0\xb0\xd0\xb5\xd1\x82 \xd1\x81\xd0\xb8\xd0\xb3\xd0\xbd\xd0\xb0\xd0\xbb "
            "\xd0\xb5\xd1\x81\xd0\xbb\xd0\xb8 \xd0\xbd\xd0\xb0\xd1\x87\xd0\xb0\xd0\xbb\xd1\x8c\xd0\xbd\xd1\x8b\xd0\xb9 "
            "\xd1\x83\xd0\xb7\xd0\xb5\xd0\xbb \xd0\xbc\xd0\xbe\xd0\xb4\xd1\x83\xd0\xbb\xd1\x8c_\xd0\xbf\xd0\xb0\xd1\x80\xd1\x81\xd0\xb5\xd1\x80 ? ";

        std::string terminal;
        int chain_len = 0;
        auto resolve = [&]() {
            const auto uids = engine.tokenizer.encode_line(q, false);
            const auto f = engine.parse_frame_unsupervised(uids);
            int chain_ids[MAX_MULTIHOP_HOPS + 1] = {-1};
            int32_t agree = 0;
            const int clen = deduce_multihop_in_1gb_bank_cpp(
                bank->base_ptr, f.src_entity, MAX_MULTIHOP_HOPS,
                engine.model->vocab_pos.data(), engine.model->vocab_size,
                chain_ids, &agree);
            chain_len = clen;
            terminal = (clen >= 2 && chain_ids[clen - 1] >= 0)
                     ? engine.tokenizer.id_to_word[chain_ids[clen - 1]]
                     : std::string();
        };
        resolve();

        GATE_CHECK(chain_len == 4, "the chain was not resolved to 4 nodes");;
        GATE_CHECK(terminal == "\xd1\x81\xd1\x82\xd1\x80\xd0\xb0\xd0\xbd\xd0\xb8\xd1\x86\xd1\x83_\xd1\x81\xd0\xb5\xd0\xb2\xd0\xb5\xd1\x80", "the chain terminal is not the true terminal");;

        const double us_lookup = best_per_call_us(resolve, 2000, 20);

        double us_generation = 1e18;
        for (int r = 0; r < 10; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            const auto res = engine.chat_and_reason_native(q);
            auto t1 = std::chrono::steady_clock::now();
            const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            if (us < us_generation) us_generation = us;
            (void)res;
        }
        // Measured: 28.7 us for the lookup against 2356.7 us with generation.
        // The gate is the ratio, not the absolute numbers, so that a loaded
        // host cannot fail it.
        GATE_CHECK(us_generation > us_lookup * 10.0, "the lookup is no longer far cheaper than the generation path");;

        std::printf("[+] GATE 3: chain of %d nodes resolved to '%s', lookup %.2f us "
                    "vs %.1f us with generation (%.0fx)\n",
                    chain_len, terminal.c_str(), us_lookup, us_generation,
                    us_generation / us_lookup);
    }

    // ------------------------------------------------------------- GATE 4 --
    // The agent that learns WITHOUT a teacher.
    //
    // `compute_reward_optimal_action_for_training` -- the hand-written rule
    // that says which move is right -- is NOT called anywhere in this path.
    // The only reward is the environment's own outcome (an apple eaten, a
    // death, a step survived), and the only supervision is what the agent
    // experienced. The metric is the environment's verdict, not the agreement
    // with the rule.
    {
        AutoMappedWeightBank guard("build/test_selector_selftrain.bin");
        MappedWeightBank* bank = guard.get();
        GATE_CHECK(bank != nullptr, "bank creation failed");;
        bank->clear_master_cache();

        SnakePopcntAgent agent(bank);

        // The evaluation is averaged over several board seeds: the training is
        // seeded, so the gate itself is deterministic, and averaging removes
        // the noise of a single board layout.
        const int kEvalSeeds = 4;
        auto evaluate = [&](const SnakePopcntAgent& a, bool random_policy) {
            double apples = 0.0, steps = 0.0;
            for (int s = 0; s < kEvalSeeds; ++s) {
                const auto sc = a.evaluate_snake_policy(
                    10, 20260921ULL + static_cast<uint64_t>(s) * 7919ULL, random_policy);
                apples += sc.mean_apples;
                steps += sc.mean_steps;
            }
            return std::make_pair(apples / kEvalSeeds, steps / kEvalSeeds);
        };

        const auto base = evaluate(agent, true);

        // GATE 4a: the environment's reward plus its progress signal.
        {
            size_t pages = 0;
            SnakePopcntAgent::SelfTrainReport report;
            const double train_ms = agent.train_in_snake_gym_by_reward(
                &pages,
                3000,            // episodes
                0.60,            // initial exploration epsilon
                12345ULL,        // seed
                true,            // the environment's progress signal
                0.05,            // progress gain
                &report,
                0.02             // final exploration epsilon
            );
            const auto learned = evaluate(agent, false);

            // Measured on this host with exactly these settings: 9.1 apples
            // and 69 steps per episode (the random baseline: 0.05 apples, 4
            // steps; the hand-written teacher: 32.1 apples, 340 steps). The
            // gates are set well below the measurement so that the seed
            // variance cannot fail them.
            GATE_CHECK(learned.first >= 5.0, "the teacher-free agent did not learn to eat");;
            GATE_CHECK(learned.second >= 40.0, "the teacher-free agent did not learn to survive");;
            GATE_CHECK(learned.first > base.first * 10.0, "the teacher-free agent is not far above the random baseline");;
            GATE_CHECK(learned.second > base.second * 5.0, "the teacher-free agent does not survive longer than the random baseline");;

            std::printf("[+] GATE 4a: teacher-free training -> %.2f apples and %.0f steps per "
                        "episode (random baseline %.2f / %.0f), %.0f ms, %zu pages\n",
                        learned.first, learned.second, base.first, base.second, train_ms, pages);
        }

        // GATE 4b: the purest form -- the environment's reward ONLY, no
        // progress signal at all. It learns, but it needs more episodes.
        {
            AutoMappedWeightBank guard2("build/test_selector_selftrain_pure.bin");
            MappedWeightBank* bank2 = guard2.get();
            GATE_CHECK(bank2 != nullptr, "bank creation failed");;
            bank2->clear_master_cache();
            SnakePopcntAgent pure(bank2);

            const auto pure_before = evaluate(pure, false);
            (void)pure_before;

            size_t pages = 0;
            SnakePopcntAgent::SelfTrainReport report;
            const double train_ms = pure.train_in_snake_gym_by_reward(
                &pages,
                10000,           // episodes: the sparse reward needs more of them
                0.60,
                12349ULL,
                false,           // NO progress signal: the reward is the apple,
                0.0,             // the death and the survived step, nothing else
                &report,
                0.02
            );
            const auto learned = evaluate(pure, false);

            // Measured on this host: 7.5 apples and 117 steps per episode.
            GATE_CHECK(learned.first >= 3.0, "the agent did not learn from the bare reward");;
            GATE_CHECK(learned.second >= 60.0, "the agent did not learn to survive from the bare reward");;

            std::printf("[+] GATE 4b: bare reward only (apple / death / survived step) -> "
                        "%.2f apples and %.0f steps per episode, %.0f ms, %zu pages\n",
                        learned.first, learned.second, train_ms, pages);
        }
    }

    std::printf("[SUCCESS] All PopCount Second Brain task-training gates passed!\n");
    return 0;
}

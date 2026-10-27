#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "popcnt_brain.h"

namespace hsaa {

struct ContinuousInterpolationResult {
    std::array<float, 512> vector{};
    std::string nearest_token;
};

class SecondBrain {
public:
    explicit SecondBrain(size_t max_pages = 4096, const std::string& state_file = "build/second_brain_state.bin")
        : handle_(brain_create(max_pages, state_file.c_str())) {
        if (!handle_) {
            throw std::runtime_error("Failed to initialize PopCountBrain on path: " + state_file);
        }
    }

    ~SecondBrain() {
        if (handle_) {
            brain_destroy(handle_);
            handle_ = nullptr;
        }
    }

    SecondBrain(const SecondBrain&) = delete;
    SecondBrain& operator=(const SecondBrain&) = delete;

    SecondBrain(SecondBrain&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    SecondBrain& operator=(SecondBrain&& other) noexcept {
        if (this != &other) {
            if (handle_) brain_destroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    bool store_fact(const std::string& subject, const std::string& relation, const std::string& object) {
        return brain_store_fact(handle_, subject.c_str(), relation.c_str(), object.c_str()) != 0;
    }

    std::string query(const std::string& subject, const std::string& relation) const {
        char buf[512] = {0};
        if (brain_query_relation(handle_, subject.c_str(), relation.c_str(), buf, sizeof(buf)) != 0) {
            return std::string(buf);
        }
        return {};
    }

    bool has_conflict(const std::string& subject, const std::string& relation, const std::string& proposed_obj) const {
        return brain_check_conflict(handle_, subject.c_str(), relation.c_str(), proposed_obj.c_str()) != 0;
    }

    std::pair<size_t, size_t> get_stats() const {
        size_t pages = 0;
        size_t rules = 0;
        brain_get_stats(handle_, &pages, &rules);
        return {pages, rules};
    }

    bool save_state(const std::string& filepath) const {
        return brain_save_state(handle_, filepath.c_str()) != 0;
    }

    bool load_state(const std::string& filepath) {
        return brain_load_state(handle_, filepath.c_str()) != 0;
    }

    BrainTrainStats train_corpus(const std::string& corpus_text, int epochs = 6) {
        BrainTrainStats stats{};
        if (brain_train_corpus(handle_, corpus_text.c_str(), epochs, &stats) == 0) {
            throw std::runtime_error("brain_train_corpus failed");
        }
        return stats;
    }

    BrainTrainStats train_file(const std::string& filepath, int epochs = 6) {
        BrainTrainStats stats{};
        if (brain_train_file(handle_, filepath.c_str(), epochs, &stats) == 0) {
            throw std::runtime_error("brain_train_file failed: " + filepath);
        }
        return stats;
    }

    std::string generate(const std::string& prompt, int max_tokens = 64, float temperature = 0.0f, size_t buffer_size = 32768) {
        std::vector<char> buf(buffer_size, 0);
        int n_tok = brain_generate(handle_, prompt.c_str(), buf.data(), buf.size(), max_tokens, temperature);
        if (n_tok <= 0) return {};
        return std::string(buf.data());
    }

    std::array<float, 512> embed_continuous(const std::string& text) const {
        std::array<float, 512> vec{};
        if (brain_embed_continuous(handle_, text.c_str(), vec.data()) == 0) {
            vec.fill(0.0f);
        }
        return vec;
    }

    float continuous_similarity(const std::string& text_a, const std::string& text_b) const {
        return brain_continuous_similarity(handle_, text_a.c_str(), text_b.c_str());
    }

    ContinuousInterpolationResult interpolate_continuous(const std::string& text_a, const std::string& text_b, float alpha = 0.5f) const {
        ContinuousInterpolationResult res{};
        char tok_buf[256] = {0};
        if (brain_interpolate_continuous(handle_, text_a.c_str(), text_b.c_str(), alpha, res.vector.data(), tok_buf, sizeof(tok_buf)) > 0) {
            res.nearest_token = tok_buf;
        }
        return res;
    }

    std::pair<float, float> train_foundation_backprop(const std::string& corpus_text = "", int epochs = 8, float learning_rate = 0.08f) {
        float init_loss = 0.0f;
        float final_loss = 0.0f;
        const char* c_ptr = corpus_text.empty() ? nullptr : corpus_text.c_str();
        if (brain_train_foundation_backprop(handle_, c_ptr, epochs, learning_rate, &init_loss, &final_loss) == 0) {
            throw std::runtime_error("brain_train_foundation_backprop failed");
        }
        return {init_loss, final_loss};
    }

    std::string paraphrase_continuous(const std::string& input_text, float temperature = 0.0f, size_t buffer_size = 8192) {
        std::vector<char> buf(buffer_size, 0);
        int n = brain_paraphrase_continuous(handle_, input_text.c_str(), buf.data(), buf.size(), temperature);
        if (n <= 0) return {};
        return std::string(buf.data());
    }

    bool learn_continuous_function(
        const std::string& func_name,
        const std::vector<float>& x_samples,
        const std::vector<float>& y_samples,
        size_t num_samples,
        size_t x_dim = 1,
        size_t y_dim = 1
    ) {
        if (x_samples.size() < num_samples * x_dim || y_samples.size() < num_samples * y_dim) return false;
        return brain_learn_continuous_function(
            handle_, func_name.c_str(), x_samples.data(), y_samples.data(), num_samples, x_dim, y_dim
        ) != 0;
    }

    std::vector<float> eval_continuous_function(
        const std::string& func_name,
        const std::vector<float>& x_query,
        size_t y_dim = 1
    ) const {
        std::vector<float> out_y(y_dim, 0.0f);
        if (brain_eval_continuous_function(
                handle_, func_name.c_str(), x_query.data(), x_query.size(), out_y.data(), y_dim) == 0) {
            return {};
        }
        return out_y;
    }

    std::string infer_template_free_relation(const std::string& unstructured_corpus, const std::string& query_subject) {
        char buf[512] = {0};
        if (brain_infer_template_free_relation(handle_, unstructured_corpus.c_str(), query_subject.c_str(), buf, sizeof(buf)) != 0) {
            return std::string(buf);
        }
        return {};
    }

    std::array<float, BRAIN_DENSE_4096_DIM> embed_dense_4096(const std::string& text) const {
        std::array<float, BRAIN_DENSE_4096_DIM> vec{};
        if (brain_embed_dense_4096(handle_, text.c_str(), vec.data()) == 0) {
            vec.fill(0.0f);
        }
        return vec;
    }

    float cosine_similarity_4096(const std::string& text_a, const std::string& text_b) const {
        return brain_cosine_similarity_4096(handle_, text_a.c_str(), text_b.c_str());
    }

    BrainScaleAndHeldOutProfile profile_heldout_and_scale(const std::string& heldout_text = "") const {
        BrainScaleAndHeldOutProfile prof{};
        const char* h_ptr = heldout_text.empty() ? nullptr : heldout_text.c_str();
        if (!brain_profile_heldout_and_scale(handle_, h_ptr, &prof)) {
            throw std::runtime_error("brain_profile_heldout_and_scale failed");
        }
        return prof;
    }

    std::pair<std::string, bool> synthesize_algorithm_zeroshot(
        const std::string& spec_prompt,
        size_t buffer_size = 16384
    ) {
        std::vector<char> buf(buffer_size, 0);
        bool verified = false;
        int n = brain_synthesize_algorithm_zeroshot(handle_, spec_prompt.c_str(), buf.data(), buf.size(), &verified);
        if (n <= 0) return {{}, false};
        return {std::string(buf.data()), verified};
    }

    std::pair<std::string, double> attend_long_context(
        const std::string& long_text,
        const std::string& query_prompt,
        int max_tokens = 32,
        size_t buffer_size = 8192
    ) {
        std::vector<char> buf(buffer_size, 0);
        double max_attn = 0.0;
        int n = brain_attend_long_context(
            handle_, long_text.c_str(), query_prompt.c_str(), buf.data(), buf.size(), max_tokens, &max_attn
        );
        if (n <= 0) return {{}, 0.0};
        return {std::string(buf.data()), max_attn};
    }

    struct ManifoldScaleResult {
        size_t active_vocab = 0;
        size_t num_clusters = 0;
        double initial_loss = 0.0;
        double final_loss = 0.0;
    };

    ManifoldScaleResult scale_manifold_vocabulary(
        size_t target_vocab_size,
        int backprop_epochs = 6,
        float learning_rate = 0.12f
    ) {
        ManifoldScaleResult res{};
        if (!brain_manifold_scale_vocabulary(
                handle_, target_vocab_size, backprop_epochs, learning_rate,
                &res.active_vocab, &res.num_clusters, &res.initial_loss, &res.final_loss)) {
            throw std::runtime_error("brain_manifold_scale_vocabulary failed");
        }
        return res;
    }

    PopCountBrain* native_handle() const noexcept { return handle_; }

private:
    PopCountBrain* handle_ = nullptr;
};

} // namespace hsaa

namespace hssa = hsaa;

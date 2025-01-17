#pragma once

#include <vector>
#include <memory>
#include <queue>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace nllb {

struct BeamSearchConfig {
    int beam_size;
    int max_length;
    float length_penalty;
    float eos_penalty;
    int num_return_sequences;
    float temperature;
    float top_k;
    float top_p;
    float repetition_penalty;

    BeamSearchConfig(
        int beam_size = 5,
        int max_length = 128,
        float length_penalty = 1.0f,
        float eos_penalty = 0.9f,
        int num_return_sequences = 1,
        float temperature = 1.0f,
        float top_k = 0,
        float top_p = 0.9f,
        float repetition_penalty = 0.9f)
        : beam_size(beam_size),
          max_length(max_length),
          length_penalty(length_penalty),
          eos_penalty(eos_penalty),
          num_return_sequences(num_return_sequences),
          temperature(temperature),
          top_k(top_k),
          top_p(top_p),
          repetition_penalty(repetition_penalty) {}
};

class BeamSearchDecoder;  // 前向声明

class BeamSearchState {
public:
    explicit BeamSearchState(const BeamSearchConfig& config)
        : config_(config), finished_(false) {}

    void initialize_from_encoder_output(const std::vector<float>& encoder_output) {
        encoder_output_ = encoder_output;
        current_tokens_.clear();
        scores_.clear();
    }

    bool is_finished() const { return finished_; }
    
    const std::vector<int64_t>& get_current_tokens() const { return current_tokens_; }
    
    void set_scores(const std::vector<float>& scores) {
        scores_ = scores;
    }

    const std::vector<float>& get_scores() const { return scores_; }

    void append_token(int64_t token) {
        current_tokens_.push_back(token);
        if (current_tokens_.size() >= config_.max_length) {
            finished_ = true;
        }
    }

    std::vector<int64_t> get_best_sequence() const {
        return current_tokens_;
    }

private:
    BeamSearchConfig config_;
    std::vector<float> encoder_output_;
    std::vector<int64_t> current_tokens_;
    std::vector<float> scores_;
    bool finished_;

    friend class BeamSearchDecoder;  // 允许 BeamSearchDecoder 访问私有成员
};

class BeamSearchDecoder {
public:
    explicit BeamSearchDecoder(const BeamSearchConfig& config)
        : config_(config) {}

    void update_config(const BeamSearchConfig& config) {
        config_ = config;
    }

    int64_t generate_next_token(BeamSearchState& state) {
        // 应用温度缩放
        std::vector<float> scaled_scores = state.scores_;
        for (auto& score : scaled_scores) {
            score /= config_.temperature;
        }

        // 应用 top_k 采样
        if (config_.top_k > 0) {
            apply_top_k(scaled_scores);
        }

        // 应用 top_p 采样
        if (config_.top_p < 1.0f) {
            apply_top_p(scaled_scores);
        }

        // 应用重复惩罚
        apply_repetition_penalty(scaled_scores, state.get_current_tokens());

        // 获取最高分数的token
        auto max_it = std::max_element(scaled_scores.begin(), scaled_scores.end());
        return static_cast<int64_t>(std::distance(scaled_scores.begin(), max_it));
    }

private:
    BeamSearchConfig config_;

    void apply_top_k(std::vector<float>& scores) {
        std::vector<size_t> indices(scores.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                 [&scores](size_t i1, size_t i2) { return scores[i1] > scores[i2]; });

        std::vector<float> new_scores(scores.size(), -INFINITY);
        for (size_t i = 0; i < std::min(static_cast<size_t>(config_.top_k), scores.size()); ++i) {
            new_scores[indices[i]] = scores[indices[i]];
        }
        scores = new_scores;
    }

    void apply_top_p(std::vector<float>& scores) {
        // 计算softmax
        float max_score = *std::max_element(scores.begin(), scores.end());
        std::vector<float> probs;
        float sum = 0.0f;
        for (float score : scores) {
            float prob = std::exp(score - max_score);
            probs.push_back(prob);
            sum += prob;
        }
        for (float& prob : probs) {
            prob /= sum;
        }

        // 按概率排序
        std::vector<size_t> indices(scores.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                 [&probs](size_t i1, size_t i2) { return probs[i1] > probs[i2]; });

        // 累积概率直到达到top_p
        float cumsum = 0.0f;
        std::vector<float> new_scores(scores.size(), -INFINITY);
        for (size_t idx : indices) {
            if (cumsum >= config_.top_p) break;
            new_scores[idx] = scores[idx];
            cumsum += probs[idx];
        }
        scores = new_scores;
    }

    void apply_repetition_penalty(std::vector<float>& scores, const std::vector<int64_t>& tokens) {
        for (int64_t token : tokens) {
            if (token >= 0 && token < static_cast<int64_t>(scores.size())) {
                scores[token] /= config_.repetition_penalty;
            }
        }
    }
};

} // namespace nllb 
#pragma once

#include <vector>
#include <functional>
#include <stdexcept>
#include <random>
#include <onnxruntime_cxx_api.h>

namespace nllb {

// Beam搜索的配置参数
struct BeamSearchConfig {
    int beam_size;           // beam大小
    int max_length;         // 最大生成长度
    float length_penalty;   // 长度惩罚系数
    float eos_penalty;      // EOS标记惩罚
    int num_return_sequences; // 返回的序列数量
    float temperature;      // 温度参数，控制采样随机性
    float top_k;           // top-k采样参数
    float top_p;           // nucleus采样参数
    float repetition_penalty; // 重复惩罚系数

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
        : beam_size(beam_size)
        , max_length(max_length)
        , length_penalty(length_penalty)
        , eos_penalty(eos_penalty)
        , num_return_sequences(num_return_sequences)
        , temperature(temperature)
        , top_k(top_k)
        , top_p(top_p)
        , repetition_penalty(repetition_penalty) {
        validate();
    }

    void validate() const {
        if (beam_size <= 0) throw std::invalid_argument("beam_size must be positive");
        if (max_length <= 0) throw std::invalid_argument("max_length must be positive");
        if (length_penalty < 0) throw std::invalid_argument("length_penalty must be non-negative");
        if (eos_penalty < 0) throw std::invalid_argument("eos_penalty must be non-negative");
        if (num_return_sequences <= 0) throw std::invalid_argument("num_return_sequences must be positive");
        if (num_return_sequences > beam_size) throw std::invalid_argument("num_return_sequences cannot exceed beam_size");
        if (temperature <= 0) throw std::invalid_argument("temperature must be positive");
        if (top_k < 0) throw std::invalid_argument("top_k must be non-negative");
        if (top_p <= 0 || top_p > 1) throw std::invalid_argument("top_p must be in (0, 1]");
        if (repetition_penalty < 0) throw std::invalid_argument("repetition_penalty must be non-negative");
    }
};

// 候选序列的数据结构
struct BeamHypothesis {
    std::vector<int64_t> tokens;  // 生成的token序列
    float score;                  // 序列的累积分数
    bool is_done;                // 是否已完成（遇到EOS）

    BeamHypothesis(
        const std::vector<int64_t>& tokens = {},
        float score = 0.0f,
        bool is_done = false)
        : tokens(tokens)
        , score(score)
        , is_done(is_done) {}
};

// 缓存状态管理
class CacheState {
public:
    CacheState(int max_length, int hidden_size, int num_heads, int num_layers)
        : max_length_(max_length), hidden_size_(hidden_size), num_heads_(num_heads), num_layers_(num_layers),
          decoder_keys_(num_layers), decoder_values_(num_layers),
          encoder_keys_(num_layers), encoder_values_(num_layers) {}

    // 移动构造函数
    CacheState(CacheState&& other) noexcept = default;
    // 移动赋值运算符
    CacheState& operator=(CacheState&& other) noexcept = default;
    // 删除复制构造函数和复制赋值运算符
    CacheState(const CacheState&) = delete;
    CacheState& operator=(const CacheState&) = delete;

    Ort::Value&& get_decoder_key(int layer) { return std::move(decoder_keys_[layer]); }
    Ort::Value&& get_decoder_value(int layer) { return std::move(decoder_values_[layer]); }
    Ort::Value&& get_encoder_key(int layer) { return std::move(encoder_keys_[layer]); }
    Ort::Value&& get_encoder_value(int layer) { return std::move(encoder_values_[layer]); }

    void update_decoder_key(int layer, Ort::Value&& key) { decoder_keys_[layer] = std::move(key); }
    void update_decoder_value(int layer, Ort::Value&& value) { decoder_values_[layer] = std::move(value); }
    void update_encoder_key(int layer, Ort::Value&& key) { encoder_keys_[layer] = std::move(key); }
    void update_encoder_value(int layer, Ort::Value&& value) { encoder_values_[layer] = std::move(value); }

private:
    int max_length_;
    int hidden_size_;
    int num_heads_;
    int num_layers_;
    std::vector<Ort::Value> decoder_keys_;
    std::vector<Ort::Value> decoder_values_;
    std::vector<Ort::Value> encoder_keys_;
    std::vector<Ort::Value> encoder_values_;
};

// Beam Search解码器
class BeamSearchDecoder {
public:
    explicit BeamSearchDecoder(const BeamSearchConfig& config);

    // 主解码函数
    std::vector<BeamHypothesis> decode(
        const std::function<std::vector<float>(
            const std::vector<int64_t>&,
            const CacheState&)>& step_fn,
        int64_t bos_token_id,
        int64_t eos_token_id,
        int64_t pad_token_id);

private:
    BeamSearchConfig config_;
    std::mt19937 rng_;  // 随机数生成器

    // 计算归一化分数
    float compute_normalized_score(const BeamHypothesis& hyp) const;

    // 更新候选序列
    void update_hypotheses(
        std::vector<BeamHypothesis>& hypotheses,
        const std::vector<float>& next_scores,
        const std::vector<int64_t>& next_tokens,
        int64_t eos_token_id);

    // 应用温度和采样
    std::vector<float> apply_temperature_and_sampling(
        std::vector<float>& scores,
        const std::vector<int64_t>& tokens,
        const std::vector<int64_t>& previous_tokens) const;

    // Top-K采样
    void apply_top_k_sampling(std::vector<float>& scores) const;

    // Top-P (nucleus) 采样
    void apply_top_p_sampling(std::vector<float>& scores) const;
};

} // namespace nllb 
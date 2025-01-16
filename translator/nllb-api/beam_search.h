#pragma once

#include <vector>
#include <functional>

namespace nllb {

// Beam搜索的配置参数
struct BeamSearchConfig {
    int beam_size;           // beam大小
    int max_length;         // 最大生成长度
    float length_penalty;   // 长度惩罚系数
    float eos_penalty;      // EOS标记惩罚
    int num_return_sequences; // 返回的序列数量

    BeamSearchConfig(
        int beam_size = 5,
        int max_length = 128,
        float length_penalty = 1.0f,
        float eos_penalty = 0.9f,
        int num_return_sequences = 1)
        : beam_size(beam_size)
        , max_length(max_length)
        , length_penalty(length_penalty)
        , eos_penalty(eos_penalty)
        , num_return_sequences(num_return_sequences) {}
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
struct CacheState {
    std::vector<float> key_cache;     // key投影的缓存
    std::vector<float> value_cache;   // value投影的缓存
    int current_length;               // 当前序列长度
    int hidden_size;                 // 隐藏层大小
    int num_heads;                   // 注意力头数量

    CacheState(int max_length, int hidden_size, int num_heads)
        : key_cache(max_length * hidden_size)
        , value_cache(max_length * hidden_size)
        , current_length(0)
        , hidden_size(hidden_size)
        , num_heads(num_heads) {}

    void update(const std::vector<float>& new_key,
               const std::vector<float>& new_value) {
        // 更新缓存
        std::copy(new_key.begin(), new_key.end(),
                 key_cache.begin() + current_length * hidden_size);
        std::copy(new_value.begin(), new_value.end(),
                 value_cache.begin() + current_length * hidden_size);
        current_length++;
    }

    void clear() {
        std::fill(key_cache.begin(), key_cache.end(), 0.0f);
        std::fill(value_cache.begin(), value_cache.end(), 0.0f);
        current_length = 0;
    }
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

    // 计算归一化分数
    float compute_normalized_score(const BeamHypothesis& hyp) const;

    // 更新候选序列
    void update_hypotheses(
        std::vector<BeamHypothesis>& hypotheses,
        const std::vector<float>& next_scores,
        const std::vector<int64_t>& next_tokens,
        int64_t eos_token_id);
};

} // namespace nllb 
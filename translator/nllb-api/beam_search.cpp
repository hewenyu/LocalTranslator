#include "beam_search.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include "tensor_utils.h"

namespace nllb {

BeamSearchDecoder::BeamSearchDecoder(
    int beam_size, int max_length, float length_penalty,
    float temperature, float top_k, float top_p,
    float repetition_penalty)
    : beam_size_(beam_size),
      max_length_(max_length),
      length_penalty_(length_penalty),
      temperature_(temperature),
      top_k_(top_k),
      top_p_(top_p),
      repetition_penalty_(repetition_penalty) {}

std::vector<BeamHypothesis> BeamSearchDecoder::decode(
    Ort::Session* decoder_session,
    const Ort::MemoryInfo& memory_info,
    const std::vector<float>& encoder_output,
    CacheContainer& cache_container,
    const TokenizerResult& input_tokens,
    int eos_token_id) const {
    
    std::vector<BeamHypothesis> hypotheses(beam_size_);
    std::vector<int64_t> current_tokens(beam_size_, eos_token_id);
    bool all_done = false;
    int current_length = 0;

    while (!all_done && current_length < max_length_) {
        // 准备解码器输入
        auto input_tensor = TensorUtils::createInt64Tensor(
            memory_info, current_tokens,
            {1, static_cast<int64_t>(current_tokens.size())});

        // 运行解码器
        const char* input_names[] = {
            "input_ids",
            "encoder_hidden_states",
            "key_cache",
            "value_cache"
        };

        std::vector<Ort::Value> ort_inputs;
        ort_inputs.reserve(4);
        ort_inputs.push_back(std::move(input_tensor));
        
        auto encoder_tensor = TensorUtils::createFloatTensor(
            memory_info, encoder_output, 
            {1, static_cast<int64_t>(input_tokens.size()), 1024});
        ort_inputs.push_back(std::move(encoder_tensor));
        
        auto key_cache = cache_container.get_key_cache();
        auto value_cache = cache_container.get_value_cache();
        ort_inputs.push_back(std::move(key_cache));
        ort_inputs.push_back(std::move(value_cache));

        const char* output_names[] = {"logits", "new_key_cache", "new_value_cache"};
        
        auto outputs = decoder_session->Run(
            Ort::RunOptions{nullptr},
            input_names,
            ort_inputs.data(),
            ort_inputs.size(),
            output_names,
            3
        );

        // 更新缓存
        cache_container.update(std::move(outputs[1]), std::move(outputs[2]));

        // 处理 logits
        auto logits = TensorUtils::getTensorData<float>(outputs[0]);
        auto scores = apply_repetition_penalty(logits, current_tokens);
        scores = apply_temperature(scores);
        scores = apply_top_k_top_p(scores);

        // 为每个假设选择下一个 token
        for (int i = 0; i < beam_size_; ++i) {
            if (hypotheses[i].is_done) continue;

            // 获取当前假设的分数
            size_t offset = i * scores.size() / beam_size_;
            std::vector<float> current_scores(
                scores.begin() + offset,
                scores.begin() + offset + scores.size() / beam_size_
            );

            // 找到最佳 token
            auto max_it = std::max_element(current_scores.begin(), current_scores.end());
            int64_t next_token = std::distance(current_scores.begin(), max_it);
            float next_score = *max_it;

            // 更新假设
            hypotheses[i].tokens.push_back(next_token);
            hypotheses[i].score += next_score;
            current_tokens[i] = next_token;

            // 检查是否完成
            if (next_token == eos_token_id || 
                hypotheses[i].tokens.size() >= static_cast<size_t>(max_length_)) {
                hypotheses[i].is_done = true;
            }
        }

        // 检查是否所有假设都完成
        all_done = true;
        for (const auto& hyp : hypotheses) {
            if (!hyp.is_done) {
                all_done = false;
                break;
            }
        }

        current_length++;
    }

    // 应用长度惩罚并排序假设
    for (auto& hyp : hypotheses) {
        hyp.score = compute_sequence_score(hyp.tokens);
    }

    std::sort(hypotheses.begin(), hypotheses.end(),
              [](const BeamHypothesis& a, const BeamHypothesis& b) {
                  return a.score > b.score;
              });

    return hypotheses;
}

std::vector<float> BeamSearchDecoder::apply_repetition_penalty(
    std::vector<float>& scores,
    const std::vector<int64_t>& input_ids) const {
    
    std::vector<float> penalized_scores = scores;
    
    // 对已生成的 token 应用重复惩罚
    for (const auto& token : input_ids) {
        if (token >= 0 && static_cast<size_t>(token) < scores.size()) {
            if (scores[token] > 0) {
                penalized_scores[token] /= repetition_penalty_;
            } else {
                penalized_scores[token] *= repetition_penalty_;
            }
        }
    }
    
    return penalized_scores;
}

std::vector<float> BeamSearchDecoder::apply_temperature(
    std::vector<float>& scores) const {
    
    if (temperature_ == 0 || temperature_ == 1.0f) {
        return scores;
    }
    
    std::vector<float> scaled_scores = scores;
    for (auto& score : scaled_scores) {
        score /= temperature_;
    }
    
    return scaled_scores;
}

std::vector<float> BeamSearchDecoder::apply_top_k_top_p(
    std::vector<float>& scores) const {
    
    std::vector<float> filtered_scores = scores;
    
    // 应用 top-k 过滤
    if (top_k_ > 0 && top_k_ < static_cast<float>(scores.size())) {
        std::vector<size_t> indices(scores.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        std::partial_sort(indices.begin(), 
                         indices.begin() + static_cast<int>(top_k_),
                         indices.end(),
                         [&scores](size_t i1, size_t i2) {
                             return scores[i1] > scores[i2];
                         });
        
        for (size_t i = static_cast<size_t>(top_k_); i < indices.size(); ++i) {
            filtered_scores[indices[i]] = -INFINITY;
        }
    }
    
    // 应用 top-p (nucleus) 采样
    if (top_p_ < 1.0f) {
        std::vector<size_t> indices(filtered_scores.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        std::sort(indices.begin(), indices.end(),
                 [&filtered_scores](size_t i1, size_t i2) {
                     return filtered_scores[i1] > filtered_scores[i2];
                 });
        
        float cumsum = 0.0f;
        for (size_t i = 0; i < indices.size(); ++i) {
            cumsum += std::exp(filtered_scores[indices[i]]);
            if (cumsum > top_p_) {
                for (size_t j = i + 1; j < indices.size(); ++j) {
                    filtered_scores[indices[j]] = -INFINITY;
                }
                break;
            }
        }
    }
    
    return filtered_scores;
}

float BeamSearchDecoder::compute_sequence_score(
    const std::vector<int64_t>& sequence) const {
    
    float length_penalty = std::pow(sequence.size(), length_penalty_);
    float sequence_score = 0.0f;
    
    // 计算序列的累积分数
    for (size_t i = 1; i < sequence.size(); ++i) {
        sequence_score += std::log(sequence[i]);
    }
    
    return sequence_score / length_penalty;
}

} // namespace nllb 
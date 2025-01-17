#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "cache_container.h"
#include "tensor_utils.h"

namespace nllb {

struct BeamHypothesis {
    std::vector<int64_t> tokens;
    float score;
    bool is_done;

    BeamHypothesis(const std::vector<int64_t>& t, float s)
        : tokens(t), score(s), is_done(false) {}

    bool operator<(const BeamHypothesis& other) const {
        return score < other.score;
    }
};

class BeamSearchDecoder {
public:
    BeamSearchDecoder(int beam_size, float length_penalty, float eos_token_id)
        : beam_size_(beam_size)
        , length_penalty_(length_penalty)
        , eos_token_id_(eos_token_id) {}

    std::vector<BeamHypothesis> decode(
        Ort::Session& decoder_session,
        const Ort::MemoryInfo& memory_info,
        const std::vector<float>& encoder_output,
        const std::vector<int64_t>& encoder_shape,
        CacheContainer& cache) {
        
        std::vector<BeamHypothesis> hypotheses;
        for (int i = 0; i < beam_size_; ++i) {
            hypotheses.emplace_back(std::vector<int64_t>{}, 0.0f);
        }

        // 初始化解码状态
        std::vector<int64_t> current_tokens(beam_size_, eos_token_id_);
        bool all_done = false;

        while (!all_done) {
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
            ort_inputs.push_back(std::move(input_tensor));
            
            auto encoder_tensor = TensorUtils::createFloatTensor(
                memory_info, encoder_output, encoder_shape);
            ort_inputs.push_back(std::move(encoder_tensor));
            
            ort_inputs.push_back(cache.getKeyCache());
            ort_inputs.push_back(cache.getValueCache());

            const char* output_names[] = {"logits", "new_key_cache", "new_value_cache"};
            
            auto outputs = decoder_session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                ort_inputs.data(),
                ort_inputs.size(),
                output_names,
                3
            );

            // 更新缓存
            cache.update(std::move(outputs[1]), std::move(outputs[2]));

            // 获取 logits 并选择下一个 token
            auto logits = TensorUtils::getTensorData<float>(outputs[0]);
            advance(hypotheses, logits, current_tokens);

            // 检查是否所有假设都完成
            all_done = true;
            for (const auto& hyp : hypotheses) {
                if (!hyp.is_done) {
                    all_done = false;
                    break;
                }
            }
        }

        return hypotheses;
    }

private:
    void advance(std::vector<BeamHypothesis>& hypotheses,
                const std::vector<float>& logits,
                std::vector<int64_t>& current_tokens) {
        // 对每个假设计算下一个最佳 token
        for (size_t i = 0; i < hypotheses.size(); ++i) {
            if (hypotheses[i].is_done) continue;

            // 获取当前假设的 logits
            size_t offset = i * logits.size() / hypotheses.size();
            std::vector<float> current_logits(
                logits.begin() + offset,
                logits.begin() + offset + logits.size() / hypotheses.size()
            );

            // 应用长度惩罚
            float length_penalty = std::pow(hypotheses[i].tokens.size() + 1, length_penalty_);
            for (auto& score : current_logits) {
                score /= length_penalty;
            }

            // 找到最佳 token
            auto max_it = std::max_element(current_logits.begin(), current_logits.end());
            int64_t next_token = std::distance(current_logits.begin(), max_it);
            float next_score = *max_it;

            // 更新假设
            hypotheses[i].tokens.push_back(next_token);
            hypotheses[i].score += next_score;
            current_tokens[i] = next_token;

            // 检查是否完成
            if (next_token == eos_token_id_) {
                hypotheses[i].is_done = true;
            }
        }
    }

    int beam_size_;
    float length_penalty_;
    int64_t eos_token_id_;
};

} // namespace nllb 
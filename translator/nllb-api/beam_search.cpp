#include <algorithm>
#include <cmath>
#include <numeric>
#include "beam_search.h"

namespace nllb {

BeamSearchDecoder::BeamSearchDecoder(const BeamSearchConfig& config)
    : config_(config) {}

float BeamSearchDecoder::compute_normalized_score(const BeamHypothesis& hyp) const {
    // 应用长度惩罚
    float length_penalty = std::pow((5.0f + hyp.tokens.size()) / 6.0f, config_.length_penalty);
    return hyp.score / length_penalty;
}

void BeamSearchDecoder::update_hypotheses(
    std::vector<BeamHypothesis>& hypotheses,
    const std::vector<float>& next_scores,
    const std::vector<int64_t>& next_tokens,
    int64_t eos_token_id) {
    
    // 创建新的候选序列
    std::vector<BeamHypothesis> new_hypotheses;
    new_hypotheses.reserve(config_.beam_size * 2);  // 为可能的新序列预留空间

    // 对于每个当前的候选序列
    for (size_t i = 0; i < hypotheses.size(); ++i) {
        const auto& hyp = hypotheses[i];
        if (hyp.is_done) {
            new_hypotheses.push_back(hyp);
            continue;
        }

        // 获取当前序列的top-k个下一个token
        std::vector<std::pair<float, int64_t>> top_k;
        top_k.reserve(next_scores.size());
        for (size_t j = 0; j < next_scores.size(); ++j) {
            // 添加重复惩罚
            float score = next_scores[j];
            if (std::find(hyp.tokens.begin(), hyp.tokens.end(), next_tokens[j]) != hyp.tokens.end()) {
                score *= 0.9f;  // 重复token的惩罚因子
            }
            top_k.emplace_back(score, next_tokens[j]);
        }

        std::partial_sort(top_k.begin(), 
                         top_k.begin() + config_.beam_size,
                         top_k.end(),
                         std::greater<>());

        // 为每个top-k token创建新的候选序列
        for (int k = 0; k < config_.beam_size; ++k) {
            auto new_tokens = hyp.tokens;
            new_tokens.push_back(top_k[k].second);
            
            float new_score = hyp.score + top_k[k].first;
            bool is_done = (top_k[k].second == eos_token_id);
            
            // 如果生成了EOS，应用EOS惩罚
            if (is_done) {
                new_score *= config_.eos_penalty;
            }
            
            new_hypotheses.emplace_back(new_tokens, new_score, is_done);
        }
    }

    // 选择最好的beam_size个候选序列
    std::partial_sort(new_hypotheses.begin(),
                     new_hypotheses.begin() + config_.beam_size,
                     new_hypotheses.end(),
                     [this](const BeamHypothesis& a, const BeamHypothesis& b) {
                         return compute_normalized_score(a) > compute_normalized_score(b);
                     });

    hypotheses.clear();
    hypotheses.insert(hypotheses.end(),
                     new_hypotheses.begin(),
                     new_hypotheses.begin() + config_.beam_size);
}

std::vector<BeamHypothesis> BeamSearchDecoder::decode(
    const std::function<std::vector<float>(const std::vector<int64_t>&, const CacheState&)>& step_fn,
    int64_t bos_token_id,
    int64_t eos_token_id,
    int64_t pad_token_id) {
    
    // 初始化候选序列
    std::vector<BeamHypothesis> hypotheses;
    hypotheses.emplace_back(std::vector<int64_t>{bos_token_id}, 0.0f);

    // 初始化缓存状态
    CacheState cache(config_.max_length, 1024, 16);  // hidden_size和num_heads需要从模型配置中获取

    // 提前停止的计数器
    int no_improvement_steps = 0;
    float best_score = -std::numeric_limits<float>::infinity();

    // 主解码循环
    for (int step = 0; step < config_.max_length; ++step) {
        // 检查是否所有序列都已完成
        bool all_done = true;
        for (const auto& hyp : hypotheses) {
            if (!hyp.is_done) {
                all_done = false;
                break;
            }
        }
        if (all_done) break;

        // 批处理预测
        std::vector<std::vector<int64_t>> batch_tokens;
        std::vector<size_t> active_indices;
        for (size_t i = 0; i < hypotheses.size(); ++i) {
            if (!hypotheses[i].is_done) {
                batch_tokens.push_back(hypotheses[i].tokens);
                active_indices.push_back(i);
            }
        }

        // 如果没有活跃的序列，退出循环
        if (batch_tokens.empty()) break;

        // 对每个未完成的候选序列进行下一步预测
        std::vector<float> next_scores;
        std::vector<int64_t> next_tokens;
        
        for (const auto& tokens : batch_tokens) {
            // 使用模型进行下一步预测
            auto scores = step_fn(tokens, cache);
            
            // 获取最高分的token
            auto max_score = *std::max_element(scores.begin(), scores.end());
            next_scores.push_back(max_score);
            next_tokens.push_back(std::distance(scores.begin(),
                                              std::max_element(scores.begin(), scores.end())));
        }

        // 更新候选序列
        update_hypotheses(hypotheses, next_scores, next_tokens, eos_token_id);

        // 检查是否有更好的分数
        float current_best = -std::numeric_limits<float>::infinity();
        for (const auto& hyp : hypotheses) {
            float score = compute_normalized_score(hyp);
            current_best = std::max(current_best, score);
        }

        if (current_best > best_score) {
            best_score = current_best;
            no_improvement_steps = 0;
        } else {
            no_improvement_steps++;
        }

        // 如果连续5步没有改善，提前停止
        if (no_improvement_steps >= 5) {
            break;
        }
    }

    // 对所有未完成的序列添加EOS标记
    for (auto& hyp : hypotheses) {
        if (!hyp.is_done) {
            hyp.tokens.push_back(eos_token_id);
            hyp.is_done = true;
        }
    }

    // 按分数排序并返回前num_return_sequences个序列
    std::sort(hypotheses.begin(), hypotheses.end(),
              [this](const BeamHypothesis& a, const BeamHypothesis& b) {
                  return compute_normalized_score(a) > compute_normalized_score(b);
              });

    hypotheses.resize(std::min(static_cast<size_t>(config_.num_return_sequences),
                              hypotheses.size()));
    return hypotheses;
}

} // namespace nllb 
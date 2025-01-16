#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include "beam_search.h"

namespace nllb {

BeamSearchDecoder::BeamSearchDecoder(const BeamSearchConfig& config)
    : config_(config)
    , rng_(std::random_device{}()) {
    // 初始化ONNX Runtime会话
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 创建会话
    session_ = std::make_unique<Ort::Session>(nullptr, nullptr, session_options);

    // 初始化输入输出名称
    size_t num_input_nodes = session_->GetInputCount();
    input_names_.reserve(num_input_nodes);
    for (size_t i = 0; i < num_input_nodes; i++) {
        Ort::AllocatorWithDefaultOptions allocator;
        input_names_.push_back(session_->GetInputNameAllocated(i, allocator).get());
    }

    size_t num_output_nodes = session_->GetOutputCount();
    output_names_.reserve(num_output_nodes);
    for (size_t i = 0; i < num_output_nodes; i++) {
        Ort::AllocatorWithDefaultOptions allocator;
        output_names_.push_back(session_->GetOutputNameAllocated(i, allocator).get());
    }
}

BeamSearchDecoder::~BeamSearchDecoder() {
    // 清理资源
    input_names_.clear();
    output_names_.clear();
}

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

std::vector<float> BeamSearchDecoder::apply_temperature_and_sampling(
    std::vector<float>& scores,
    const std::vector<int64_t>& tokens,
    const std::vector<int64_t>& previous_tokens) const {
    
    try {
        // 应用重复惩罚
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (std::find(previous_tokens.begin(), previous_tokens.end(), tokens[i]) != previous_tokens.end()) {
                scores[i] *= config_.repetition_penalty;
            }
        }

        // 应用温度
        if (config_.temperature != 1.0f) {
            for (auto& score : scores) {
                score /= config_.temperature;
            }
        }

        // 应用Top-K采样
        if (config_.top_k > 0) {
            apply_top_k_sampling(scores);
        }

        // 应用Top-P采样
        if (config_.top_p < 1.0f) {
            apply_top_p_sampling(scores);
        }

        // 计算softmax
        float max_score = *std::max_element(scores.begin(), scores.end());
        float sum = 0.0f;
        for (auto& score : scores) {
            score = std::exp(score - max_score);
            sum += score;
        }
        for (auto& score : scores) {
            score /= sum;
        }

        return scores;
    } catch (const std::exception& e) {
        std::stringstream ss;
        ss << "Error in temperature sampling: " << e.what();
        throw std::runtime_error(ss.str());
    }
}

void BeamSearchDecoder::apply_top_k_sampling(std::vector<float>& scores) const {
    if (scores.empty()) return;

    // 找到第k大的元素
    size_t k = static_cast<size_t>(config_.top_k);
    k = std::min(k, scores.size());
    
    std::vector<float> sorted_scores = scores;
    std::nth_element(sorted_scores.begin(), 
                    sorted_scores.begin() + k - 1,
                    sorted_scores.end(),
                    std::greater<float>());
    
    float threshold = sorted_scores[k - 1];
    
    // 将低于阈值的分数设为负无穷
    for (auto& score : scores) {
        if (score < threshold) {
            score = -std::numeric_limits<float>::infinity();
        }
    }
}

void BeamSearchDecoder::apply_top_p_sampling(std::vector<float>& scores) const {
    if (scores.empty()) return;

    // 计算概率分布
    std::vector<std::pair<float, size_t>> prob_idx;
    prob_idx.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        prob_idx.emplace_back(scores[i], i);
    }

    // 按概率降序排序
    std::sort(prob_idx.begin(), prob_idx.end(),
              std::greater<std::pair<float, size_t>>());

    // 计算累积概率
    float cumsum = 0.0f;
    size_t last_idx = prob_idx.size();
    for (size_t i = 0; i < prob_idx.size(); ++i) {
        cumsum += prob_idx[i].first;
        if (cumsum > config_.top_p) {
            last_idx = i + 1;
            break;
        }
    }

    // 创建概率掩码
    std::vector<bool> mask(scores.size(), false);
    for (size_t i = 0; i < last_idx; ++i) {
        mask[prob_idx[i].second] = true;
    }

    // 应用掩码
    for (size_t i = 0; i < scores.size(); ++i) {
        if (!mask[i]) {
            scores[i] = -std::numeric_limits<float>::infinity();
        }
    }
}

std::vector<float> BeamSearchDecoder::step_fn(
    const std::vector<int64_t>& tokens,
    CacheState& cache) {
    std::vector<float> scores;
    
    try {
        // 准备输入
        std::vector<Ort::Value> inputs;
        
        // 对于每一层处理缓存
        for (int layer = 0; layer < cache.get_num_layers(); ++layer) {
            auto& decoder_key = cache.get_decoder_key(layer);
            auto& decoder_value = cache.get_decoder_value(layer);
            auto& encoder_key = cache.get_encoder_key(layer);
            auto& encoder_value = cache.get_encoder_value(layer);

            // 如果缓存中有值，使用它们
            if (decoder_key && decoder_value && encoder_key && encoder_value) {
                inputs.emplace_back(std::move(*decoder_key));
                inputs.emplace_back(std::move(*decoder_value));
                inputs.emplace_back(std::move(*encoder_key));
                inputs.emplace_back(std::move(*encoder_value));
            } else {
                // 初始化新的缓存值
                // 这里需要根据实际情况创建新的Ort::Value对象
                // ...
            }
        }

        // 运行模型
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, 
                                          input_names_.data(), 
                                          inputs.data(), 
                                          inputs.size(),
                                          output_names_.data(), 
                                          output_names_.size());

        // 更新缓存
        for (int layer = 0; layer < cache.get_num_layers(); ++layer) {
            cache.update_decoder_key(layer, std::move(output_tensors[layer * 4]));
            cache.update_decoder_value(layer, std::move(output_tensors[layer * 4 + 1]));
            cache.update_encoder_key(layer, std::move(output_tensors[layer * 4 + 2]));
            cache.update_encoder_value(layer, std::move(output_tensors[layer * 4 + 3]));
        }

        // 获取logits并转换为scores
        auto& logits = output_tensors.back();
        const float* logits_data = logits.GetTensorData<float>();
        size_t logits_size = logits.GetTensorTypeAndShapeInfo().GetElementCount();
        scores.assign(logits_data, logits_data + logits_size);

    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("ONNX Runtime error: ") + e.what());
    }

    return scores;
}

std::vector<BeamHypothesis> BeamSearchDecoder::decode(
    const std::function<std::vector<float>(const std::vector<int64_t>&, const CacheState&)>& step_fn,
    int64_t bos_token_id,
    int64_t eos_token_id,
    int64_t pad_token_id) {
    try {
        // 初始化候选序列
        std::vector<BeamHypothesis> hypotheses;
        hypotheses.emplace_back(std::vector<int64_t>{bos_token_id}, 0.0f);

        // 初始化缓存状态
        CacheState cache(config_.max_length, 1024, 16, 24);  // 使用标准的NLLB-200模型配置

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
                try {
                    // 使用模型进行下一步预测
                    auto scores = step_fn(tokens, cache);
                    
                    // 应用温度采样
                    scores = apply_temperature_and_sampling(scores, tokens, tokens);

                    // 如果使用采样
                    if (config_.temperature > 0) {
                        // 创建分布
                        std::discrete_distribution<int64_t> dist(scores.begin(), scores.end());
                        // 采样下一个token
                        int64_t next_token = dist(rng_);
                        next_tokens.push_back(next_token);
                        next_scores.push_back(scores[next_token]);
                    } else {
                        // 使用argmax
                        auto max_it = std::max_element(scores.begin(), scores.end());
                        next_tokens.push_back(std::distance(scores.begin(), max_it));
                        next_scores.push_back(*max_it);
                    }
                } catch (const std::exception& e) {
                    std::stringstream ss;
                    ss << "Error in decoding step " << step << ": " << e.what();
                    throw std::runtime_error(ss.str());
                }
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
    } catch (const std::exception& e) {
        std::stringstream ss;
        ss << "Beam search decoding failed: " << e.what();
        throw std::runtime_error(ss.str());
    }
}

} // namespace nllb 
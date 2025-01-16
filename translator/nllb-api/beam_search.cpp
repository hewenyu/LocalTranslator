#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <spdlog/spdlog.h>
#include "beam_search.h"

namespace nllb {

CacheState::CacheState(int max_length, int hidden_size, int num_heads, int num_layers) {
    decoder_keys_.resize(num_layers);
    decoder_values_.resize(num_layers);
    encoder_keys_.resize(num_layers);
    encoder_values_.resize(num_layers);
}

void CacheState::update_decoder_key(int layer, Ort::Value&& key) {
    decoder_keys_[layer] = std::move(key);
}

void CacheState::update_decoder_value(int layer, Ort::Value&& value) {
    decoder_values_[layer] = std::move(value);
}

void CacheState::update_encoder_key(int layer, Ort::Value&& key) {
    encoder_keys_[layer] = std::move(key);
}

void CacheState::update_encoder_value(int layer, Ort::Value&& value) {
    encoder_values_[layer] = std::move(value);
}

std::optional<Ort::Value>& CacheState::get_decoder_key(int layer) {
    return decoder_keys_[layer];
}

std::optional<Ort::Value>& CacheState::get_decoder_value(int layer) {
    return decoder_values_[layer];
}

std::optional<Ort::Value>& CacheState::get_encoder_key(int layer) {
    return encoder_keys_[layer];
}

std::optional<Ort::Value>& CacheState::get_encoder_value(int layer) {
    return encoder_values_[layer];
}

BeamSearchDecoder::BeamSearchDecoder(const BeamSearchConfig& config)
    : config_(config)
    , rng_(std::random_device{}()) {
    
    spdlog::debug("Initializing beam search decoder with beam size: {}", config.beam_size);
    
    // 初始化ONNX Runtime会话
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
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

        spdlog::debug("Beam search decoder initialized successfully");
    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize beam search decoder: {}", e.what());
        throw;
    }
}

std::vector<BeamHypothesis> BeamSearchDecoder::decode(
    const std::vector<float>& encoder_output,
    const std::vector<int64_t>& input_ids,
    const std::vector<int64_t>& attention_mask) const {
    
    try {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        
        // 准备初始假设
        std::vector<BeamHypothesis> hypotheses;
        hypotheses.reserve(config_.beam_size);
        for (int i = 0; i < config_.beam_size; ++i) {
            hypotheses.emplace_back(input_ids, 0.0f);
        }

        // 主循环
        for (int step = 0; step < config_.max_length; ++step) {
            std::vector<BeamHypothesis> candidates;
            candidates.reserve(config_.beam_size * config_.beam_size);

            for (const auto& hyp : hypotheses) {
                if (hyp.is_done) {
                    candidates.push_back(hyp);
                    continue;
                }

                // 准备decoder输入
                std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(hyp.tokens.size())};
                auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memory_info,
                    const_cast<int64_t*>(hyp.tokens.data()),
                    hyp.tokens.size(), input_shape.data(), input_shape.size());

                // 运行decoder
                std::vector<Ort::Value> inputs;
                inputs.push_back(std::move(input_ids_tensor));

                auto outputs = session_->Run(
                    Ort::RunOptions{nullptr},
                    input_names_.data(),
                    inputs.data(),
                    inputs.size(),
                    output_names_.data(),
                    output_names_.size()
                );

                // 处理输出
                float* logits = outputs[0].GetTensorMutableData<float>();
                size_t vocab_size = outputs[0].GetTensorTypeAndShapeInfo().GetShape()[2];

                // 应用温度和采样
                std::vector<float> probs(logits, logits + vocab_size);
                if (config_.temperature > 0) {
                    for (auto& prob : probs) {
                        prob /= config_.temperature;
                    }
                }

                // 应用softmax
                float max_prob = *std::max_element(probs.begin(), probs.end());
                float sum = 0.0f;
                for (auto& prob : probs) {
                    prob = std::exp(prob - max_prob);
                    sum += prob;
                }
                for (auto& prob : probs) {
                    prob /= sum;
                }

                // 生成候选序列
                std::vector<size_t> indices(vocab_size);
                std::iota(indices.begin(), indices.end(), 0);
                std::partial_sort(indices.begin(), 
                                indices.begin() + config_.beam_size,
                                indices.end(),
                                [&probs](size_t i1, size_t i2) {
                                    return probs[i1] > probs[i2];
                                });

                for (int i = 0; i < config_.beam_size; ++i) {
                    auto new_tokens = hyp.tokens;
                    new_tokens.push_back(indices[i]);
                    float new_score = hyp.score + std::log(probs[indices[i]]);
                    
                    // 应用长度惩罚
                    float length_penalty = std::pow((5.0f + new_tokens.size()) / 6.0f, 
                                                  config_.length_penalty);
                    new_score /= length_penalty;
                    
                    candidates.emplace_back(new_tokens, new_score);
                }
            }

            // 选择最好的假设
            std::partial_sort(candidates.begin(),
                            candidates.begin() + config_.beam_size,
                            candidates.end(),
                            [](const BeamHypothesis& a, const BeamHypothesis& b) {
                                return a.score > b.score;
                            });

            hypotheses.clear();
            hypotheses.insert(hypotheses.begin(),
                            candidates.begin(),
                            candidates.begin() + config_.beam_size);
        }

        // 返回结果
        std::partial_sort(hypotheses.begin(),
                         hypotheses.begin() + config_.num_return_sequences,
                         hypotheses.end(),
                         [](const BeamHypothesis& a, const BeamHypothesis& b) {
                             return a.score > b.score;
                         });

        hypotheses.resize(config_.num_return_sequences);
        return hypotheses;
    } catch (const std::exception& e) {
        spdlog::error("Beam search decoding failed: {}", e.what());
        throw;
    }
}

} // namespace nllb 
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <spdlog/spdlog.h>
#include <filesystem>
#include "beam_search.h"

namespace nllb {

BeamSearchDecoder::BeamSearchDecoder(const BeamSearchConfig& config, const std::string& model_dir)
    : config_(config)
    , rng_(std::random_device{}()) {
    
    spdlog::debug("Initializing beam search decoder with beam size: {}", config.beam_size);
    
    try {
        // 检查模型目录
        if (!std::filesystem::exists(model_dir)) {
            throw std::runtime_error("Model directory not found: " + model_dir);
        }

        // 初始化ONNX Runtime会话选项
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 加载decoder模型
        std::string decoder_path = model_dir + "/NLLB_decoder.onnx";
        if (!std::filesystem::exists(decoder_path)) {
            throw std::runtime_error("Decoder model not found: " + decoder_path);
        }

        spdlog::info("Loading decoder model: {}", decoder_path);
        session_ = std::make_unique<Ort::Session>(nullptr, decoder_path.c_str(), session_options);

        // 初始化输入输出名称
        size_t num_input_nodes = session_->GetInputCount();
        input_names_.reserve(num_input_nodes);
        for (size_t i = 0; i < num_input_nodes; i++) {
            Ort::AllocatorWithDefaultOptions allocator;
            input_names_.push_back(session_->GetInputNameAllocated(i, allocator).get());
            spdlog::debug("Decoder input {}: {}", i, input_names_.back());
        }

        size_t num_output_nodes = session_->GetOutputCount();
        output_names_.reserve(num_output_nodes);
        for (size_t i = 0; i < num_output_nodes; i++) {
            Ort::AllocatorWithDefaultOptions allocator;
            output_names_.push_back(session_->GetOutputNameAllocated(i, allocator).get());
            spdlog::debug("Decoder output {}: {}", i, output_names_.back());
        }

        spdlog::info("Successfully initialized beam search decoder");
    } catch (const Ort::Exception& e) {
        spdlog::error("ONNX Runtime error: {}", e.what());
        throw std::runtime_error(std::string("ONNX Runtime error: ") + e.what());
    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize beam search decoder: {}", e.what());
        throw;
    }
}

BeamSearchDecoder::~BeamSearchDecoder() = default;

std::vector<BeamHypothesis> BeamSearchDecoder::decode(
    const std::vector<int64_t>& input_ids,
    int64_t bos_token_id,
    int64_t eos_token_id,
    int64_t pad_token_id) {
    
    try {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        
        // 准备初始假设
        std::vector<BeamHypothesis> hypotheses;
        hypotheses.reserve(config_.beam_size);
        hypotheses.emplace_back(std::vector<int64_t>{bos_token_id}, 0.0f);

        // 创建缓存状态
        CacheState cache(config_.max_length, 1024, 16, 24);  // 使用标准的NLLB-200模型配置

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

                // 准备attention mask
                std::vector<int64_t> attention_mask(hyp.tokens.size(), 1);
                auto attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(memory_info,
                    attention_mask.data(), attention_mask.size(),
                    input_shape.data(), input_shape.size());

                // 运行decoder
                std::vector<Ort::Value> inputs;
                inputs.push_back(std::move(input_ids_tensor));
                inputs.push_back(std::move(attention_mask_tensor));

                // 添加缓存状态
                for (int layer = 0; layer < cache.get_num_layers(); ++layer) {
                    if (auto& key = cache.get_decoder_key(layer)) {
                        inputs.push_back(std::move(*key));
                    }
                    if (auto& value = cache.get_decoder_value(layer)) {
                        inputs.push_back(std::move(*value));
                    }
                    if (auto& key = cache.get_encoder_key(layer)) {
                        inputs.push_back(std::move(*key));
                    }
                    if (auto& value = cache.get_encoder_value(layer)) {
                        inputs.push_back(std::move(*value));
                    }
                }

                auto outputs = session_->Run(
                    Ort::RunOptions{nullptr},
                    input_names_.data(),
                    inputs.data(),
                    inputs.size(),
                    output_names_.data(),
                    output_names_.size()
                );

                // 更新缓存状态
                for (int layer = 0; layer < cache.get_num_layers(); ++layer) {
                    cache.update_decoder_key(layer, std::move(outputs[layer * 4]));
                    cache.update_decoder_value(layer, std::move(outputs[layer * 4 + 1]));
                    cache.update_encoder_key(layer, std::move(outputs[layer * 4 + 2]));
                    cache.update_encoder_value(layer, std::move(outputs[layer * 4 + 3]));
                }

                // 处理logits输出
                float* logits = outputs.back().GetTensorMutableData<float>();
                size_t vocab_size = outputs.back().GetTensorTypeAndShapeInfo().GetShape()[2];

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
                    new_tokens.push_back(static_cast<int64_t>(indices[i]));
                    float new_score = hyp.score + std::log(probs[indices[i]]);
                    
                    // 应用长度惩罚
                    float length_penalty = std::pow((5.0f + static_cast<float>(new_tokens.size())) / 6.0f, 
                                                  config_.length_penalty);
                    new_score /= length_penalty;
                    
                    bool is_done = (indices[i] == static_cast<size_t>(eos_token_id));
                    if (is_done) {
                        new_score *= config_.eos_penalty;
                    }
                    candidates.emplace_back(new_tokens, new_score, is_done);
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
            hypotheses.insert(hypotheses.end(),
                            candidates.begin(),
                            candidates.begin() + config_.beam_size);

            // 检查是否所有序列都已完成
            bool all_done = std::all_of(hypotheses.begin(), hypotheses.end(),
                                      [](const BeamHypothesis& h) { return h.is_done; });
            if (all_done) break;
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
    } catch (const Ort::Exception& e) {
        spdlog::error("ONNX Runtime error in decoding: {}", e.what());
        throw std::runtime_error(std::string("ONNX Runtime error in decoding: ") + e.what());
    } catch (const std::exception& e) {
        spdlog::error("Beam search decoding failed: {}", e.what());
        throw;
    }
}

} // namespace nllb 
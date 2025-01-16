#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <chrono>
#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>
#include "translator/nllb-api/nllb_translator.h"
#include "translator/nllb-api/beam_search.h"
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

namespace nllb {

// 模型配置结构
struct ModelConfig {
    int hidden_size;
    int num_heads;
    int vocab_size;
    int max_position_embeddings;
    int encoder_layers;
    int decoder_layers;

    static ModelConfig load_from_yaml(const std::string& config_path) {
        try {
            YAML::Node config = YAML::LoadFile(config_path);
            ModelConfig model_config;
            model_config.hidden_size = config["hidden_size"].as<int>();
            model_config.num_heads = config["num_heads"].as<int>();
            model_config.vocab_size = config["vocab_size"].as<int>();
            model_config.max_position_embeddings = config["max_position_embeddings"].as<int>();
            model_config.encoder_layers = config["encoder_layers"].as<int>();
            model_config.decoder_layers = config["decoder_layers"].as<int>();
            return model_config;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load model config: " + std::string(e.what()));
        }
    }
};

NLLBTranslator::NLLBTranslator(const common::TranslatorConfig& config) 
    : ort_env_(ORT_LOGGING_LEVEL_WARNING, "nllb_translator") {
    model_dir_ = config.nllb.model_dir;
    target_lang_ = config.nllb.target_lang;
    params_ = config.nllb.params;
    
    spdlog::info("Initializing NLLB translator with model dir: {}", model_dir_);
    
    try {
        // 加载模型配置
        std::string config_path = model_dir_ + "/model_config.yaml";
        model_config_ = ModelConfig::load_from_yaml(config_path);
        spdlog::info("Loaded model config: hidden_size={}, num_heads={}", 
                    model_config_.hidden_size, model_config_.num_heads);

        // 初始化组件
        initialize_language_codes();
        load_models();
        
        // 初始化tokenizer
        std::string vocab_path = model_dir_ + "/" + config.nllb.model_files.tokenizer_vocab;
        tokenizer_ = std::make_unique<Tokenizer>(vocab_path);
        spdlog::info("Initialized tokenizer with vocab: {}", vocab_path);

        // 初始化beam search配置
        beam_config_ = BeamSearchConfig(
            params_.beam_size,
            params_.max_length,
            params_.length_penalty,
            0.9f,  // Default EOS penalty
            1,     // Default num return sequences
            1.0f,  // Default temperature
            0,     // Default top_k (disabled)
            0.9f,  // Default top_p
            0.9f   // Default repetition penalty
        );
        
        // 创建beam search解码器
        beam_search_decoder_ = std::make_unique<BeamSearchDecoder>(beam_config_);
        
        spdlog::info("NLLB translator initialization completed successfully");
    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize NLLB translator: {}", e.what());
        throw;
    }
}

NLLBTranslator::~NLLBTranslator() = default;

void NLLBTranslator::initialize_language_codes() {
    spdlog::debug("Loading language codes...");
    std::string lang_file = model_dir_ + "/nllb_languages.yaml";
    try {
        YAML::Node config = YAML::LoadFile(lang_file);
        for (const auto& lang : config["languages"]) {
            std::string code = lang["code"].as<std::string>();
            std::string nllb_code = lang["code_NLLB"].as<std::string>();
            nllb_language_codes_[code] = nllb_code;
        }
        spdlog::info("Loaded {} language codes", nllb_language_codes_.size());
    } catch (const std::exception& e) {
        spdlog::error("Failed to load language codes: {}", e.what());
        throw std::runtime_error("Failed to load language codes: " + std::string(e.what()));
    }
}

void NLLBTranslator::load_models() {
    spdlog::debug("Loading ONNX models...");
    try {
        Ort::SessionOptions session_opts;
        session_opts.SetIntraOpNumThreads(params_.num_threads);
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        // Load encoder
        std::string encoder_path = model_dir_ + "/NLLB_encoder.onnx";
        encoder_session_ = std::make_unique<Ort::Session>(ort_env_, 
            encoder_path.c_str(), session_opts);
        spdlog::info("Loaded encoder model: {}", encoder_path);

        // Load decoder
        std::string decoder_path = model_dir_ + "/NLLB_decoder.onnx";
        decoder_session_ = std::make_unique<Ort::Session>(ort_env_,
            decoder_path.c_str(), session_opts);
        spdlog::info("Loaded decoder model: {}", decoder_path);

        // Load embed and lm head
        std::string embed_path = model_dir_ + "/NLLB_embed_and_lm_head.onnx";
        embed_lm_head_session_ = std::make_unique<Ort::Session>(ort_env_,
            embed_path.c_str(), session_opts);
        spdlog::info("Loaded embedding model: {}", embed_path);

        // Load cache initializer if using cache
        if (params_.use_cache) {
            std::string cache_path = model_dir_ + "/NLLB_cache_initializer.onnx";
            cache_init_session_ = std::make_unique<Ort::Session>(ort_env_,
                cache_path.c_str(), session_opts);
            spdlog::info("Loaded cache initializer: {}", cache_path);
        }
    } catch (const std::exception& e) {
        spdlog::error("Failed to load models: {}", e.what());
        throw std::runtime_error("Failed to load models: " + std::string(e.what()));
    }
}

std::vector<float> NLLBTranslator::run_encoder(const Tokenizer::TokenizerOutput& tokens) const {
    spdlog::debug("Running encoder with input length: {}", tokens.input_ids.size());
    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        // Prepare input tensors
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(tokens.input_ids.size())};
        
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
            
        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            tokens.input_ids.data(),
            tokens.input_ids.size(),
            input_shape.data(),
            input_shape.size()
        );
            
        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            tokens.attention_mask.data(),
            tokens.attention_mask.size(),
            input_shape.data(),
            input_shape.size()
        );

        // Run encoder
        const char* input_names[] = {"input_ids", "attention_mask"};
        const Ort::Value input_values[] = {input_ids_tensor, attention_mask_tensor};
        const char* output_names[] = {"encoder_output"};
        
        auto encoder_outputs = encoder_session_->Run(
            Ort::RunOptions{nullptr},
            input_names, input_values, 2,
            output_names, 1
        );

        // Get output data
        float* output_data = encoder_outputs[0].GetTensorMutableData<float>();
        size_t output_size = encoder_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        spdlog::debug("Encoder completed in {} ms", duration.count());

        return std::vector<float>(output_data, output_data + output_size);
    } catch (const std::exception& e) {
        spdlog::error("Encoder failed: {}", e.what());
        throw;
    }
}

std::vector<int64_t> NLLBTranslator::run_decoder(
    const std::vector<float>& encoder_output,
    const std::string& target_lang) const {
    
    spdlog::debug("Running decoder for target language: {}", target_lang);
    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        // 创建缓存状态
        CacheState cache(params_.max_length, model_config_.hidden_size, model_config_.num_heads);

        // 定义单步解码函数
        auto step_function = [this, &encoder_output](
            const std::vector<int64_t>& tokens,
            const CacheState& cache) -> std::vector<float> {
            
            // 准备输入张量
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

            // 准备decoder输入
            std::vector<int64_t> input_shape = {1, static_cast<int64_t>(tokens.size())};
            Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, tokens.data(), tokens.size(), 
                input_shape.data(), input_shape.size());

            // 准备encoder输出
            std::vector<int64_t> encoder_shape = {1, static_cast<int64_t>(encoder_output.size())};
            Ort::Value encoder_tensor = Ort::Value::CreateTensor<float>(
                memory_info, encoder_output.data(), encoder_output.size(),
                encoder_shape.data(), encoder_shape.size());

            // 运行decoder
            const char* input_names[] = {"input_ids", "encoder_output"};
            const char* output_names[] = {"logits"};

            auto output_tensors = decoder_session_->Run(
                Ort::RunOptions{nullptr},
                input_names,
                {input_ids_tensor, encoder_tensor},
                2,
                output_names,
                1
            );

            // 获取logits
            float* logits_data = static_cast<float*>(output_tensors[0].GetTensorMutableData());
            size_t vocab_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[2];
            
            // 计算最后一个token的概率分布
            std::vector<float> scores(logits_data + (tokens.size() - 1) * vocab_size,
                                    logits_data + tokens.size() * vocab_size);
            
            // 应用softmax
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
        };

        // 初始化候选序列
        std::vector<BeamHypothesis> hypotheses;
        hypotheses.emplace_back(std::vector<int64_t>{tokenizer_->bos_id()}, 0.0f);

        // 主解码循环
        for (int step = 0; step < params_.max_length; ++step) {
            // 检查是否所有序列都已完成
            bool all_done = true;
            for (const auto& hyp : hypotheses) {
                if (!hyp.is_done) {
                    all_done = false;
                    break;
                }
            }
            if (all_done) break;

            // 获取活跃的候选序列
            std::vector<BeamHypothesis> active_hypotheses;
            for (const auto& hyp : hypotheses) {
                if (!hyp.is_done) {
                    active_hypotheses.push_back(hyp);
                }
            }

            // 对每个活跃的候选序列进行预测
            std::vector<std::vector<float>> next_token_scores;
            for (const auto& hyp : active_hypotheses) {
                auto scores = step_function(hyp.tokens, cache);
                
                // 应用重复惩罚
                for (size_t i = 0; i < scores.size(); ++i) {
                    if (std::find(hyp.tokens.begin(), hyp.tokens.end(), i) != hyp.tokens.end()) {
                        scores[i] *= params_.repetition_penalty;
                    }
                }

                // 应用温度
                if (params_.temperature > 0) {
                    for (auto& score : scores) {
                        score /= params_.temperature;
                    }
                }

                // 应用top-k采样
                if (params_.top_k > 0) {
                    std::vector<std::pair<float, size_t>> score_idx;
                    score_idx.reserve(scores.size());
                    for (size_t i = 0; i < scores.size(); ++i) {
                        score_idx.emplace_back(scores[i], i);
                    }
                    std::partial_sort(score_idx.begin(), 
                                    score_idx.begin() + params_.top_k,
                                    score_idx.end(),
                                    std::greater<>());
                    std::fill(scores.begin(), scores.end(), 0.0f);
                    for (int i = 0; i < params_.top_k; ++i) {
                        scores[score_idx[i].second] = score_idx[i].first;
                    }
                }

                // 应用top-p采样
                if (params_.top_p < 1.0f) {
                    std::vector<std::pair<float, size_t>> score_idx;
                    score_idx.reserve(scores.size());
                    for (size_t i = 0; i < scores.size(); ++i) {
                        score_idx.emplace_back(scores[i], i);
                    }
                    std::sort(score_idx.begin(), score_idx.end(), std::greater<>());
                    float cumsum = 0.0f;
                    size_t last_idx = score_idx.size();
                    for (size_t i = 0; i < score_idx.size(); ++i) {
                        cumsum += score_idx[i].first;
                        if (cumsum > params_.top_p) {
                            last_idx = i + 1;
                            break;
                        }
                    }
                    std::fill(scores.begin(), scores.end(), 0.0f);
                    for (size_t i = 0; i < last_idx; ++i) {
                        scores[score_idx[i].second] = score_idx[i].first;
                    }
                }

                next_token_scores.push_back(scores);
            }

            // 为每个候选序列选择最佳的下一个token
            std::vector<BeamHypothesis> new_hypotheses;
            for (size_t i = 0; i < active_hypotheses.size(); ++i) {
                const auto& hyp = active_hypotheses[i];
                const auto& scores = next_token_scores[i];

                // 获取top-k个token
                std::vector<std::pair<float, int64_t>> top_k;
                top_k.reserve(scores.size());
                for (size_t j = 0; j < scores.size(); ++j) {
                    if (scores[j] > 0) {
                        top_k.emplace_back(scores[j], j);
                    }
                }
                std::partial_sort(top_k.begin(),
                                top_k.begin() + params_.beam_size,
                                top_k.end(),
                                std::greater<>());

                // 为每个top-k token创建新的候选序列
                for (int k = 0; k < params_.beam_size && k < top_k.size(); ++k) {
                    auto new_tokens = hyp.tokens;
                    new_tokens.push_back(top_k[k].second);
                    
                    float new_score = hyp.score + std::log(top_k[k].first);
                    bool is_done = (top_k[k].second == tokenizer_->eos_id());
                    
                    // 应用长度惩罚
                    float length_penalty = std::pow((5.0f + new_tokens.size()) / 6.0f, 
                                                  params_.length_penalty);
                    new_score /= length_penalty;
                    
                    // 如果生成了EOS，应用EOS惩罚
                    if (is_done) {
                        new_score *= 0.9f;  // EOS penalty
                    }
                    
                    new_hypotheses.emplace_back(new_tokens, new_score, is_done);
                }
            }

            // 添加已完成的序列
            for (const auto& hyp : hypotheses) {
                if (hyp.is_done) {
                    new_hypotheses.push_back(hyp);
                }
            }

            // 选择最好的beam_size个候选序列
            std::partial_sort(new_hypotheses.begin(),
                            new_hypotheses.begin() + params_.beam_size,
                            new_hypotheses.end(),
                            [](const BeamHypothesis& a, const BeamHypothesis& b) {
                                return a.score > b.score;
                            });

            hypotheses.clear();
            hypotheses.insert(hypotheses.end(),
                            new_hypotheses.begin(),
                            new_hypotheses.begin() + std::min(static_cast<size_t>(params_.beam_size),
                                                            new_hypotheses.size()));

            // 检查是否需要提前停止
            if (step > 0 && step % 10 == 0) {
                float best_score = hypotheses[0].score;
                bool no_improvement = true;
                for (size_t i = 1; i < hypotheses.size(); ++i) {
                    if (hypotheses[i].score > best_score * 0.9f) {
                        no_improvement = false;
                        break;
                    }
                }
                if (no_improvement) {
                    break;
                }
            }
        }

        // 对所有未完成的序列添加EOS标记
        for (auto& hyp : hypotheses) {
            if (!hyp.is_done) {
                hyp.tokens.push_back(tokenizer_->eos_id());
                hyp.is_done = true;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        spdlog::debug("Decoder completed in {} ms", duration.count());

        // 返回最好的序列
        return hypotheses[0].tokens;
    } catch (const std::exception& e) {
        spdlog::error("Decoder failed: {}", e.what());
        throw;
    }
}

std::string NLLBTranslator::translate(
    const std::string& text,
    const std::string& source_lang) const {
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        spdlog::info("Starting translation from {} to {}", source_lang, target_lang_);
        spdlog::debug("Input text: {}", text);

        // Convert language codes
        std::string nllb_source = get_nllb_language_code(source_lang);
        std::string nllb_target = get_nllb_language_code(target_lang_);

        // Tokenize input
        auto tokens = tokenizer_->encode(text, nllb_source, nllb_target);
        spdlog::debug("Tokenized input length: {}", tokens.input_ids.size());

        // Run encoder
        auto encoder_output = run_encoder(tokens);

        // Run decoder
        auto output_ids = run_decoder(encoder_output, nllb_target);

        // Decode output tokens
        auto result = tokenizer_->decode(output_ids);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        spdlog::info("Translation completed in {} ms", duration.count());
        spdlog::debug("Output text: {}", result);

        return result;
    } catch (const std::exception& e) {
        spdlog::error("Translation failed: {}", e.what());
        throw std::runtime_error("Translation failed: " + std::string(e.what()));
    }
}

std::string NLLBTranslator::get_target_language() const {
    return target_lang_;
}

} // namespace nllb 
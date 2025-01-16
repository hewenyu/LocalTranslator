#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>
#include "translator/nllb-api/nllb_translator.h"
#include "translator/nllb-api/beam_search.h"
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <windows.h>
#include <tinyxml2.h>
#include <numeric>

namespace nllb {

// Helper function to convert string to wstring
static std::wstring to_wstring(const std::string& str) {
    if (str.empty()) return std::wstring();
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstr[0], size_needed);
    return wstr;
}

ModelConfig ModelConfig::load_from_yaml(const std::string& config_path) {
    try {
        YAML::Node config = YAML::LoadFile(config_path);
        ModelConfig model_config;
        model_config.hidden_size = config["hidden_size"].as<int>();
        model_config.num_heads = config["num_heads"].as<int>();
        model_config.num_layers = config["decoder_layers"].as<int>();
        model_config.vocab_size = config["vocab_size"].as<int>();
        model_config.max_position_embeddings = config["max_position_embeddings"].as<int>();
        model_config.encoder_layers = config["encoder_layers"].as<int>();
        model_config.decoder_layers = config["decoder_layers"].as<int>();
        return model_config;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load model config: " + std::string(e.what()));
    }
}

NLLBTranslator::NLLBTranslator(const common::TranslatorConfig& config) 
    : ort_env_(ORT_LOGGING_LEVEL_WARNING, "nllb_translator") {
    try {
        // 检查并设置模型目录
        model_dir_ = std::filesystem::absolute(config.nllb.model_dir).string();
        if (!std::filesystem::exists(model_dir_)) {
            throw std::runtime_error("Model directory not found: " + model_dir_);
        }
        spdlog::info("Using model directory: {}", model_dir_);

        target_lang_ = config.nllb.target_lang;
        params_ = config.nllb.params;
        
        spdlog::info("Initializing NLLB translator with model dir: {}", model_dir_);
        
        // 加载模型配置
        std::string config_path = model_dir_ + "/model_config.yaml";
        if (!std::filesystem::exists(config_path)) {
            throw std::runtime_error("Model config file not found: " + config_path);
        }
        
        model_config_ = ModelConfig::load_from_yaml(config_path);
        spdlog::info("Loaded model config: hidden_size={}, num_heads={}, num_layers={}", 
                    model_config_.hidden_size, model_config_.num_heads, model_config_.num_layers);

        // 初始化组件
        initialize_language_codes();
        load_models();
        
        // 初始化tokenizer
        std::string vocab_path = model_dir_ + "/" + config.nllb.model_files.tokenizer_vocab;
        if (!std::filesystem::exists(vocab_path)) {
            throw std::runtime_error("Tokenizer vocabulary file not found: " + vocab_path);
        }
        
        try {
            tokenizer_ = std::make_unique<Tokenizer>(vocab_path);
            spdlog::info("Successfully initialized tokenizer with vocab: {}", vocab_path);
        } catch (const std::exception& e) {
            spdlog::error("Failed to initialize tokenizer: {}", e.what());
            throw;
        }

        // 初始化beam search配置
        beam_config_ = BeamSearchConfig(
            params_.beam_size,
            params_.max_length,
            params_.length_penalty,
            0.9f,  // Default EOS penalty
            1,     // Default num return sequences
            params_.temperature,
            0,     // Default top_k (disabled)
            0.9f,  // Default top_p
            0.9f   // Default repetition penalty
        );
        
        // 创建beam search解码器
        try {
            beam_search_decoder_ = std::make_unique<BeamSearchDecoder>(beam_config_);
            spdlog::info("Successfully initialized beam search decoder");
        } catch (const std::exception& e) {
            spdlog::error("Failed to initialize beam search decoder: {}", e.what());
            throw;
        }
        
        spdlog::info("NLLB translator initialization completed successfully");
    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize NLLB translator: {}", e.what());
        throw;
    }
}

NLLBTranslator::~NLLBTranslator() = default;

void NLLBTranslator::initialize_language_codes() {
    spdlog::debug("Loading language codes...");
    std::string lang_file = model_dir_ + "/nllb_supported_languages.xml";
    
    if (!std::filesystem::exists(lang_file)) {
        throw std::runtime_error("Language codes file not found: " + lang_file);
    }

    try {
        tinyxml2::XMLDocument doc;
        auto result = doc.LoadFile(lang_file.c_str());
        if (result != tinyxml2::XML_SUCCESS) {
            std::string error_msg = "Failed to load XML file: " + lang_file;
            spdlog::error(error_msg);
            throw std::runtime_error(error_msg);
        }

        auto root = doc.FirstChildElement("languages");
        if (!root) {
            throw std::runtime_error("Invalid XML format: missing root element");
        }

        int count = 0;
        for (auto lang = root->FirstChildElement("language"); lang; lang = lang->NextSiblingElement("language")) {
            auto code = lang->FirstChildElement("code");
            auto nllb_code = lang->FirstChildElement("code_NLLB");
            
            if (!code || !nllb_code) {
                spdlog::warn("Skipping invalid language entry (missing code or NLLB code)");
                continue;
            }

            const char* code_text = code->GetText();
            const char* nllb_code_text = nllb_code->GetText();
            
            if (!code_text || !nllb_code_text) {
                spdlog::warn("Skipping invalid language entry (empty code or NLLB code)");
                continue;
            }

            nllb_language_codes_[code_text] = nllb_code_text;
            count++;
            spdlog::debug("Added language mapping: {} -> {}", code_text, nllb_code_text);
        }

        if (count == 0) {
            throw std::runtime_error("No valid language codes found in file");
        }

        spdlog::info("Successfully loaded {} language codes", count);
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

        // Load embedding and lm head model first
        std::string embed_lm_path = model_dir_ + "/NLLB_embed_and_lm_head.onnx";
        std::wstring w_embed_lm_path = to_wstring(embed_lm_path);
        embed_lm_head_session_ = std::make_unique<Ort::Session>(ort_env_,
            w_embed_lm_path.c_str(), session_opts);
        spdlog::info("Loaded embedding and lm head model: {}", embed_lm_path);

        // Print embedding model info
        Ort::AllocatorWithDefaultOptions allocator;
        size_t embed_input_nodes = embed_lm_head_session_->GetInputCount();
        spdlog::info("Embedding model input nodes: {}", embed_input_nodes);
        for (size_t i = 0; i < embed_input_nodes; i++) {
            auto input_name = embed_lm_head_session_->GetInputNameAllocated(i, allocator);
            spdlog::info("Embedding input {}: {}", i, input_name.get());
        }

        size_t embed_output_nodes = embed_lm_head_session_->GetOutputCount();
        spdlog::info("Embedding model output nodes: {}", embed_output_nodes);
        for (size_t i = 0; i < embed_output_nodes; i++) {
            auto output_name = embed_lm_head_session_->GetOutputNameAllocated(i, allocator);
            spdlog::info("Embedding output {}: {}", i, output_name.get());
        }

        // Load encoder
        std::string encoder_path = model_dir_ + "/NLLB_encoder.onnx";
        std::wstring w_encoder_path = to_wstring(encoder_path);
        encoder_session_ = std::make_unique<Ort::Session>(ort_env_, 
            w_encoder_path.c_str(), session_opts);
        spdlog::info("Loaded encoder model: {}", encoder_path);

        // Print encoder info
        size_t num_input_nodes = encoder_session_->GetInputCount();
        spdlog::info("Encoder input nodes: {}", num_input_nodes);
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = encoder_session_->GetInputNameAllocated(i, allocator);
            spdlog::info("Encoder input {}: {}", i, input_name.get());
        }

        size_t num_output_nodes = encoder_session_->GetOutputCount();
        spdlog::info("Encoder output nodes: {}", num_output_nodes);
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = encoder_session_->GetOutputNameAllocated(i, allocator);
            spdlog::info("Encoder output {}: {}", i, output_name.get());
        }

        // Load decoder
        std::string decoder_path = model_dir_ + "/NLLB_decoder.onnx";
        std::wstring w_decoder_path = to_wstring(decoder_path);
        decoder_session_ = std::make_unique<Ort::Session>(ort_env_,
            w_decoder_path.c_str(), session_opts);
        spdlog::info("Loaded decoder model: {}", decoder_path);

        // Print decoder info
        size_t decoder_input_nodes = decoder_session_->GetInputCount();
        spdlog::info("Decoder input nodes: {}", decoder_input_nodes);
        for (size_t i = 0; i < decoder_input_nodes; i++) {
            auto input_name = decoder_session_->GetInputNameAllocated(i, allocator);
            spdlog::info("Decoder input {}: {}", i, input_name.get());
        }

        size_t decoder_output_nodes = decoder_session_->GetOutputCount();
        spdlog::info("Decoder output nodes: {}", decoder_output_nodes);
        for (size_t i = 0; i < decoder_output_nodes; i++) {
            auto output_name = decoder_session_->GetOutputNameAllocated(i, allocator);
            spdlog::info("Decoder output {}: {}", i, output_name.get());
        }

        // Load cache initializer if using cache
        if (params_.use_cache) {
            std::string cache_path = model_dir_ + "/NLLB_cache_initializer.onnx";
            std::wstring w_cache_path = to_wstring(cache_path);
            cache_init_session_ = std::make_unique<Ort::Session>(ort_env_,
                w_cache_path.c_str(), session_opts);
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
        // 1. 获取embeddings
        auto embeddings = run_embedding(tokens.input_ids);
        spdlog::debug("Generated embeddings");
        
        // 2. 准备encoder输入
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(tokens.input_ids.size())};
        
        // 创建input_ids tensor
        auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memory_info,
            const_cast<int64_t*>(tokens.input_ids.data()),
            tokens.input_ids.size(), input_shape.data(), input_shape.size());
            
        // 创建attention_mask tensor
        auto attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(memory_info,
            const_cast<int64_t*>(tokens.attention_mask.data()),
            tokens.attention_mask.size(), input_shape.data(), input_shape.size());
            
        // 创建embeddings tensor
        std::array<int64_t, 3> embed_shape{1, static_cast<int64_t>(tokens.input_ids.size()), model_config_.hidden_size};
        auto embed_tensor = Ort::Value::CreateTensor<float>(memory_info,
            embeddings.data(), embeddings.size(), embed_shape.data(), embed_shape.size());
        
        // 3. 运行encoder
        const char* encoder_input_names[] = {"input_ids", "attention_mask", "embed_matrix"};
        std::vector<Ort::Value> encoder_inputs;
        encoder_inputs.push_back(std::move(input_ids_tensor));
        encoder_inputs.push_back(std::move(attention_mask_tensor));
        encoder_inputs.push_back(std::move(embed_tensor));
        
        const char* encoder_output_names[] = {"last_hidden_state"};
        auto encoder_outputs = encoder_session_->Run(
            Ort::RunOptions{nullptr},
            encoder_input_names,
            encoder_inputs.data(),
            encoder_inputs.size(),
            encoder_output_names,
            1
        );

        // 获取输出数据
        const float* output_data = encoder_outputs[0].GetTensorData<float>();
        size_t output_size = encoder_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<float> result(output_data, output_data + output_size);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        spdlog::debug("Encoder completed in {} ms", duration.count());
        
        return result;
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
        CacheState cache(
            params_.max_length,
            model_config_.hidden_size,
            model_config_.num_heads,
            model_config_.num_layers
        );
        
        // 准备encoder输出tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::array<int64_t, 3> encoder_shape{1, static_cast<int64_t>(encoder_output.size() / model_config_.hidden_size), model_config_.hidden_size};
        auto encoder_tensor = Ort::Value::CreateTensor<float>(memory_info,
            const_cast<float*>(encoder_output.data()),
            encoder_output.size(), encoder_shape.data(), encoder_shape.size());
            
        // 初始化beam search
        std::vector<BeamHypothesis> hypotheses;
        hypotheses.emplace_back(std::vector<int64_t>{tokenizer_->bos_id()}, 0.0f);
        
        // 主循环
        for (int step = 0; step < params_.max_length; ++step) {
            std::vector<BeamHypothesis> new_hypotheses;
            
            for (const auto& hyp : hypotheses) {
                if (hyp.is_done) {
                    new_hypotheses.push_back(hyp);
                    continue;
                }
                
                // 准备decoder输入
                std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(hyp.tokens.size())};
                auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memory_info,
                    const_cast<int64_t*>(hyp.tokens.data()),
                    hyp.tokens.size(), input_shape.data(), input_shape.size());

                // 创建encoder_attention_mask tensor
                std::vector<int64_t> attention_mask(encoder_output.size() / model_config_.hidden_size, 1);
                std::array<int64_t, 2> attention_shape{1, static_cast<int64_t>(attention_mask.size())};
                auto attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(memory_info,
                    attention_mask.data(), attention_mask.size(),
                    attention_shape.data(), attention_shape.size());

                // 运行decoder
                std::vector<const char*> decoder_input_names = {
                    "embed_matrix", "encoder_attention_mask", "input_ids"
                };
                
                std::vector<const char*> decoder_output_names = {
                    "pre_logits",
                    "present.0.decoder.key", "present.0.decoder.value",
                    "present.0.encoder.key", "present.0.encoder.value",
                    "present.1.decoder.key", "present.1.decoder.value",
                    "present.1.encoder.key", "present.1.encoder.value",
                    "present.2.decoder.key", "present.2.decoder.value",
                    "present.2.encoder.key", "present.2.encoder.value",
                    "present.3.decoder.key", "present.3.decoder.value",
                    "present.3.encoder.key", "present.3.encoder.value",
                    "present.4.decoder.key", "present.4.decoder.value",
                    "present.4.encoder.key", "present.4.encoder.value",
                    "present.5.decoder.key", "present.5.decoder.value",
                    "present.5.encoder.key", "present.5.encoder.value",
                    "present.6.decoder.key", "present.6.decoder.value",
                    "present.6.encoder.key", "present.6.encoder.value",
                    "present.7.decoder.key", "present.7.decoder.value",
                    "present.7.encoder.key", "present.7.encoder.value",
                    "present.8.decoder.key", "present.8.decoder.value",
                    "present.8.encoder.key", "present.8.encoder.value",
                    "present.9.decoder.key", "present.9.decoder.value",
                    "present.9.encoder.key", "present.9.encoder.value",
                    "present.10.decoder.key", "present.10.decoder.value",
                    "present.10.encoder.key", "present.10.encoder.value",
                    "present.11.decoder.key", "present.11.decoder.value",
                    "present.11.encoder.key", "present.11.encoder.value"
                };

                // 添加past_key_values到输入名称
                std::vector<std::string> input_name_strings;
                input_name_strings.reserve(model_config_.num_layers * 4);
                for (int layer = 0; layer < model_config_.num_layers; ++layer) {
                    for (const auto& suffix : {".decoder.key", ".decoder.value", ".encoder.key", ".encoder.value"}) {
                        input_name_strings.push_back("past_key_values." + std::to_string(layer) + suffix);
                        decoder_input_names.push_back(input_name_strings.back().c_str());
                    }
                }
                
                std::vector<Ort::Value> decoder_inputs;
                decoder_inputs.push_back(std::move(encoder_tensor));
                decoder_inputs.push_back(std::move(attention_mask_tensor));
                decoder_inputs.push_back(std::move(input_ids_tensor));

                // 添加past_key_values
                for (int layer = 0; layer < model_config_.num_layers; ++layer) {
                    if (step == 0) {
                        // 第一次迭代：decoder key/value为空，encoder key/value来自encoder输出
                        std::array<int64_t, 4> kv_shape{1, model_config_.num_heads, 0, model_config_.hidden_size / model_config_.num_heads};
                        std::vector<float> empty_kv;
                        
                        // 创建decoder key tensor
                        auto decoder_key = Ort::Value::CreateTensor<float>(memory_info,
                            empty_kv.data(), empty_kv.size(), kv_shape.data(), kv_shape.size());
                        cache.update_decoder_key(layer, std::move(decoder_key));
                        decoder_inputs.push_back(std::move(cache.get_decoder_key(layer).value()));
                        
                        // 创建decoder value tensor
                        auto decoder_value = Ort::Value::CreateTensor<float>(memory_info,
                            empty_kv.data(), empty_kv.size(), kv_shape.data(), kv_shape.size());
                        cache.update_decoder_value(layer, std::move(decoder_value));
                        decoder_inputs.push_back(std::move(cache.get_decoder_value(layer).value()));
                        
                        // 使用encoder的present值
                        std::array<int64_t, 4> encoder_kv_shape{1, model_config_.num_heads, 
                            static_cast<int64_t>(encoder_output.size() / model_config_.hidden_size),
                            model_config_.hidden_size / model_config_.num_heads};
                        
                        std::vector<float> encoder_kv(encoder_kv_shape[1] * encoder_kv_shape[2] * encoder_kv_shape[3], 0.0f);
                        
                        // 创建encoder key tensor
                        auto encoder_key = Ort::Value::CreateTensor<float>(memory_info,
                            encoder_kv.data(), encoder_kv.size(), encoder_kv_shape.data(), encoder_kv_shape.size());
                        cache.update_encoder_key(layer, std::move(encoder_key));
                        decoder_inputs.push_back(std::move(cache.get_encoder_key(layer).value()));
                        
                        // 创建encoder value tensor
                        auto encoder_value = Ort::Value::CreateTensor<float>(memory_info,
                            encoder_kv.data(), encoder_kv.size(), encoder_kv_shape.data(), encoder_kv_shape.size());
                        cache.update_encoder_value(layer, std::move(encoder_value));
                        decoder_inputs.push_back(std::move(cache.get_encoder_value(layer).value()));
                    } else {
                        // 后续迭代：使用上一次的present值
                        decoder_inputs.push_back(std::move(cache.get_decoder_key(layer).value()));
                        decoder_inputs.push_back(std::move(cache.get_decoder_value(layer).value()));
                        decoder_inputs.push_back(std::move(cache.get_encoder_key(layer).value()));
                        decoder_inputs.push_back(std::move(cache.get_encoder_value(layer).value()));
                    }
                }
                
                auto decoder_outputs = decoder_session_->Run(
                    Ort::RunOptions{nullptr},
                    decoder_input_names.data(),
                    decoder_inputs.data(),
                    decoder_inputs.size(),
                    decoder_output_names.data(),
                    decoder_output_names.size()
                );
                
                // 更新cache
                if (step == 0) {
                    for (int layer = 0; layer < model_config_.num_layers; ++layer) {
                        cache.update_decoder_key(layer, std::move(decoder_outputs[1 + layer * 4]));
                        cache.update_decoder_value(layer, std::move(decoder_outputs[2 + layer * 4]));
                        cache.update_encoder_key(layer, std::move(decoder_outputs[3 + layer * 4]));
                        cache.update_encoder_value(layer, std::move(decoder_outputs[4 + layer * 4]));
                    }
                }

                // 获取logits
                float* logits_data = decoder_outputs[0].GetTensorMutableData<float>();
                size_t vocab_size = decoder_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[2];
                
                // 计算最后一个token的概率分布
                std::vector<float> scores(logits_data + (hyp.tokens.size() - 1) * vocab_size,
                                        logits_data + hyp.tokens.size() * vocab_size);
                
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

                // 为每个token创建新的候选序列
                for (size_t i = 0; i < scores.size(); ++i) {
                    auto new_tokens = hyp.tokens;
                    new_tokens.push_back(i);
                    
                    float new_score = hyp.score + std::log(scores[i]);
                    bool is_done = (i == tokenizer_->eos_id());
                    
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

            // 选择最好的beam_size个候选序列
            std::partial_sort(new_hypotheses.begin(),
                            new_hypotheses.begin() + params_.beam_size,
                            new_hypotheses.end(),
                            [](const BeamHypothesis& a, const BeamHypothesis& b) {
                                return a.score > b.score;
                            });

            hypotheses.clear();
            auto copy_size = std::min<size_t>(params_.beam_size, new_hypotheses.size());
            hypotheses.insert(
                hypotheses.begin(),
                new_hypotheses.begin(),
                new_hypotheses.begin() + copy_size
            );

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

std::string NLLBTranslator::translate(const std::string& text, const std::string& source_lang) const {
    spdlog::info("Starting translation from {} to {}", source_lang, target_lang_);
    try {
        // 获取 NLLB 语言代码
        std::string src_lang_nllb = get_nllb_language_code(source_lang);
        std::string tgt_lang_nllb = get_nllb_language_code(target_lang_);

        // 对输入文本进行分词
        auto tokens = tokenizer_->encode(text, src_lang_nllb, tgt_lang_nllb);
        
        // 运行编码器
        auto encoder_output = run_encoder(tokens);
        
        // 运行解码器
        auto output_ids = run_decoder(encoder_output, tgt_lang_nllb);
        
        // 将输出转换为文本
        return tokenizer_->decode(output_ids);
    } catch (const std::exception& e) {
        spdlog::error("Translation failed: {}", e.what());
        throw std::runtime_error("Translation failed: " + std::string(e.what()));
    }
}

std::string NLLBTranslator::get_nllb_language_code(const std::string& lang_code) const {
    if (lang_code.empty()) {
        throw std::runtime_error("Empty language code provided");
    }

    spdlog::debug("Looking up NLLB code for language code: {}", lang_code);
    
    if (nllb_language_codes_.empty()) {
        spdlog::error("Language code mapping is empty. Did you call initialize_language_codes()?");
        throw std::runtime_error("Language code mapping is not initialized");
    }

    auto it = nllb_language_codes_.find(lang_code);
    if (it == nllb_language_codes_.end()) {
        spdlog::error("Unsupported language code: {}", lang_code);
        spdlog::error("Available language codes:");
        for (const auto& pair : nllb_language_codes_) {
            spdlog::error("  {} -> {}", pair.first, pair.second);
        }
        throw std::runtime_error("Unsupported language code: " + lang_code);
    }

    spdlog::debug("Found NLLB code: {} -> {}", lang_code, it->second);
    return it->second;
}

std::string NLLBTranslator::get_target_language() const {
    return target_lang_;
}

std::vector<float> NLLBTranslator::run_embedding(const std::vector<int64_t>& input_ids) const {
    try {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(input_ids.size())};
        
        // 创建input_ids tensor
        auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, 
            const_cast<int64_t*>(input_ids.data()),
            input_ids.size(), input_shape.data(), input_shape.size());
            
        // 创建use_lm_head tensor (标量，值为false表示只要embeddings)
        std::array<int64_t, 1> scalar_shape{1};
        bool use_lm_head_value = false;
        auto use_lm_head_tensor = Ort::Value::CreateTensor<bool>(memory_info,
            &use_lm_head_value, 1, scalar_shape.data(), scalar_shape.size());
            
        // pre_logits为空tensor，因为我们只需要embeddings
        // 注意：pre_logits需要是3维tensor: [batch_size, sequence_length, hidden_size]
        std::array<int64_t, 3> empty_shape{1, 1, model_config_.hidden_size};
        std::vector<float> pre_logits_data(model_config_.hidden_size, 0.0f);
        auto pre_logits_tensor = Ort::Value::CreateTensor<float>(memory_info,
            pre_logits_data.data(), pre_logits_data.size(), empty_shape.data(), empty_shape.size());
            
        // 运行embedding模型
        const char* embed_input_names[] = {"use_lm_head", "input_ids", "pre_logits"};
        std::vector<Ort::Value> embed_inputs;
        embed_inputs.push_back(std::move(use_lm_head_tensor));
        embed_inputs.push_back(std::move(input_ids_tensor));
        embed_inputs.push_back(std::move(pre_logits_tensor));
        
        const char* embed_output_names[] = {"embed_matrix"};
        auto embed_outputs = embed_lm_head_session_->Run(
            Ort::RunOptions{nullptr},
            embed_input_names,
            embed_inputs.data(),
            embed_inputs.size(),
            embed_output_names,
            1
        );
        
        // 获取输出
        const float* output_data = embed_outputs[0].GetTensorData<float>();
        size_t output_size = embed_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        return std::vector<float>(output_data, output_data + output_size);
        
    } catch (const std::exception& e) {
        spdlog::error("Embedding failed: {}", e.what());
        throw;
    }
}

} // namespace nllb 
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
        model_config.support_low_quality_languages = config["support_low_quality_languages"].as<bool>();
        model_config.eos_penalty = config["eos_penalty"].as<float>();
        model_config.max_batch_size = config["max_batch_size"].as<int>();
        return model_config;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load model config: " + std::string(e.what()));
    }
}

NLLBTranslator::NLLBTranslator(const common::TranslatorConfig& config) 
    : ort_env_(ORT_LOGGING_LEVEL_WARNING, "nllb_translator"),
      is_initialized_(false),
      is_translating_(false) {
    try {
        model_dir_ = config.nllb.model_dir;
        target_lang_ = config.nllb.target_lang;
        params_ = config.nllb.params;
        
        spdlog::info("Initializing NLLB translator with model dir: {}", model_dir_);
        
        // 加载模型配置
        std::string config_path = model_dir_ + "/model_config.yaml";
        model_config_ = ModelConfig::load_from_yaml(config_path);
        spdlog::info("Loaded model config: hidden_size={}, num_heads={}", 
                    model_config_.hidden_size, model_config_.num_heads);

        // 初始化组件
        initialize_language_codes();
        load_models();
        load_supported_languages();
        
        // 初始化tokenizer
        std::string vocab_path = model_dir_ + "/" + config.nllb.model_files.tokenizer_vocab;
        tokenizer_ = std::make_unique<Tokenizer>(vocab_path);
        spdlog::info("Initialized tokenizer with vocab: {}", vocab_path);

        // 初始化语言检测器
        if (!initialize_language_detector()) {
            spdlog::warn("Language detector initialization failed, will use user-provided language codes");
        }

        // 初始化beam search配置
        beam_config_ = BeamSearchConfig(
            params_.beam_size,
            params_.max_length,
            params_.length_penalty,
            0.9f,  // Default EOS penalty
            1,     // Default num return sequences
            params_.temperature,
            params_.top_k,
            params_.top_p,
            params_.repetition_penalty
        );
        
        // 创建beam search解码器
        beam_search_decoder_ = std::make_unique<BeamSearchDecoder>(beam_config_);
        
        is_initialized_ = true;
        spdlog::info("NLLB translator initialization completed successfully");
    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize NLLB translator: {}", e.what());
        throw;
    }
}

NLLBTranslator::~NLLBTranslator() = default;

void NLLBTranslator::initialize_language_codes() {
    try {
        // 加载语言代码映射文件
        tinyxml2::XMLDocument doc;
        std::string xml_path = model_dir_ + "/nllb_supported_languages.xml";
        if (doc.LoadFile(xml_path.c_str()) != tinyxml2::XML_SUCCESS) {
            throw std::runtime_error("Failed to load language codes XML file");
        }

        // 解析XML文件
        auto root = doc.FirstChildElement("languages");
        if (!root) {
            throw std::runtime_error("Invalid XML format: missing root element");
        }

        for (auto lang = root->FirstChildElement("language"); lang; lang = lang->NextSiblingElement("language")) {
            auto code = lang->FirstChildElement("code");
            auto nllb_code = lang->FirstChildElement("code_NLLB");
            
            if (code && nllb_code) {
                std::string display_code = code->GetText();
                std::string nllb_code_str = nllb_code->GetText();
                
                nllb_language_codes_[display_code] = nllb_code_str;
                display_language_codes_[nllb_code_str] = display_code;
                
                if (!model_config_.support_low_quality_languages) {
                    supported_languages_.push_back(display_code);
                } else {
                    // 检查是否为低质量语言
                    auto quality = lang->FirstChildElement("quality");
                    if (quality && std::string(quality->GetText()) == "low") {
                        low_quality_languages_.push_back(display_code);
                    } else {
                        supported_languages_.push_back(display_code);
                    }
                }
            }
        }
        
        spdlog::info("Loaded {} supported languages and {} low quality languages", 
                    supported_languages_.size(), low_quality_languages_.size());
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize language codes: " + std::string(e.what()));
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
    try {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        
        // 准备输入张量
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(tokens.input_ids.size())};
        auto input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, tokens.input_ids.data(), tokens.input_ids.size(), input_shape.data(), input_shape.size());

        // 运行编码器
        auto start_time = std::chrono::high_resolution_clock::now();
        auto output_tensors = encoder_session_->Run(
            Ort::RunOptions{nullptr},
            {"input_ids"},
            &input_tensor,
            1,
            {"encoder_outputs"}
        );
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        spdlog::debug("Encoder completed in {} ms", duration.count());

        // 获取输出
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        return std::vector<float>(output_data, output_data + output_size);
    } catch (const Ort::Exception& e) {
        spdlog::error("ONNX Runtime error in encoder: {}", e.what());
        throw std::runtime_error("Encoder failed: " + std::string(e.what()));
    }
}

std::vector<int64_t> NLLBTranslator::run_decoder(
    const std::vector<float>& encoder_output,
    const std::string& target_lang) const {
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 初始化beam search
        BeamSearchState state(beam_config_);
        state.initialize_from_encoder_output(encoder_output);
        
        // 生成序列
        while (!state.is_finished()) {
            auto next_token = beam_search_decoder_->generate_next_token(state);
            state.append_token(next_token);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        spdlog::debug("Decoder completed in {} ms", duration.count());
        
        return state.get_best_sequence();
    } catch (const std::exception& e) {
        spdlog::error("Decoder failed: {}", e.what());
        throw;
    }
}

std::string NLLBTranslator::translate(const std::string& text, const std::string& source_lang) const {
    if (!is_initialized_) {
        throw std::runtime_error("Translator not initialized");
    }

    if (text.empty()) {
        return "";
    }

    // 检查源语言是否需要翻译
    if (!needs_translation(source_lang)) {
        return text;
    }

    // 获取翻译锁
    std::lock_guard<std::mutex> lock(translation_mutex_);
    
    // 使用局部变量来处理原子操作
    bool expected = false;
    if (!is_translating_.compare_exchange_strong(expected, true)) {
        throw std::runtime_error("Translation already in progress");
    }

    try {
        spdlog::info("Starting translation from {} to {}", source_lang, target_lang_);
        
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
        std::string result = tokenizer_->decode(output_ids);
        
        is_translating_.store(false);
        return result;
    } catch (const std::exception& e) {
        is_translating_.store(false);
        spdlog::error("Translation failed: {}", e.what());
        throw std::runtime_error("Translation failed: " + std::string(e.what()));
    }
}

std::string NLLBTranslator::get_nllb_language_code(const std::string& lang_code) const {
    auto it = nllb_language_codes_.find(lang_code);
    if (it != nllb_language_codes_.end()) {
        return it->second;
    }
    return lang_code; // 如果找不到映射，返回原始代码
}

std::string NLLBTranslator::get_display_language_code(const std::string& nllb_code) const {
    auto it = display_language_codes_.find(nllb_code);
    if (it != display_language_codes_.end()) {
        return it->second;
    }
    return nllb_code; // 如果找不到映射，返回原始代码
}

void NLLBTranslator::set_support_low_quality_languages(bool support) {
    if (model_config_.support_low_quality_languages != support) {
        model_config_.support_low_quality_languages = support;
        // 重新加载语言支持
        supported_languages_.clear();
        low_quality_languages_.clear();
        initialize_language_codes();
    }
}

bool NLLBTranslator::get_support_low_quality_languages() const {
    return model_config_.support_low_quality_languages;
}

void NLLBTranslator::set_eos_penalty(float penalty) {
    model_config_.eos_penalty = penalty;
    if (beam_search_decoder_) {
        beam_config_.eos_penalty = penalty;
        beam_search_decoder_->update_config(beam_config_);
    }
}

float NLLBTranslator::get_eos_penalty() const {
    return model_config_.eos_penalty;
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

bool NLLBTranslator::initialize_language_detector() {
    try {
        std::string model_path = model_dir_ + "/language_detector.onnx";
        language_detector_ = std::make_unique<LanguageDetector>(model_path);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize language detector: {}", e.what());
        return false;
    }
}

void NLLBTranslator::load_supported_languages() {
    supported_languages_.clear();
    for (const auto& pair : nllb_language_codes_) {
        supported_languages_.push_back(pair.first);
    }
}

std::string NLLBTranslator::normalize_language_code(const std::string& lang_code) const {
    // 移除区域标识符，只保留主要语言代码
    std::string normalized = lang_code;
    size_t hyphen_pos = normalized.find('-');
    if (hyphen_pos != std::string::npos) {
        normalized = normalized.substr(0, hyphen_pos);
    }
    
    size_t underscore_pos = normalized.find('_');
    if (underscore_pos != std::string::npos) {
        normalized = normalized.substr(0, underscore_pos);
    }
    
    // 转换为小写
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    
    return normalized;
}

std::string NLLBTranslator::detect_language(const std::string& text) const {
    if (!language_detector_) {
        throw std::runtime_error("Language detector not initialized");
    }
    return language_detector_->detect_language(text);
}

bool NLLBTranslator::needs_translation(const std::string& source_lang) const {
    try {
        auto normalized_source = normalize_language_code(source_lang);
        auto normalized_target = normalize_language_code(target_lang_);
        return normalized_source != normalized_target;
    } catch (const std::exception& e) {
        spdlog::error("Error checking translation need: {}", e.what());
        return true;
    }
}

std::vector<std::string> NLLBTranslator::get_supported_languages() const {
    return supported_languages_;
}

bool NLLBTranslator::is_language_supported(const std::string& lang_code) const {
    auto normalized_code = normalize_language_code(lang_code);
    return nllb_language_codes_.find(normalized_code) != nllb_language_codes_.end();
}

std::vector<std::string> NLLBTranslator::translate_batch(
    const std::vector<std::string>& texts, const std::string& source_lang) const {
    if (!is_initialized_) {
        throw std::runtime_error("Translator not initialized");
    }

    if (texts.empty()) {
        return std::vector<std::string>();
    }

    // 检查源语言是否需要翻译
    if (!needs_translation(source_lang)) {
        return texts;
    }

    // 获取翻译锁
    std::lock_guard<std::mutex> lock(translation_mutex_);
    bool expected = false;
    if (!is_translating_.compare_exchange_strong(expected, true)) {
        throw std::runtime_error("Translation already in progress");
    }

    try {
        spdlog::info("Starting batch translation from {} to {}, batch size: {}", 
                     source_lang, target_lang_, texts.size());
        
        std::string src_lang_nllb = get_nllb_language_code(source_lang);
        std::string tgt_lang_nllb = get_nllb_language_code(target_lang_);
        
        std::vector<std::string> results;
        results.reserve(texts.size());

        // 批量处理
        for (const auto& text : texts) {
            auto tokens = tokenizer_->encode(text, src_lang_nllb, tgt_lang_nllb);
            auto encoder_output = run_encoder(tokens);
            auto output_ids = run_decoder(encoder_output, tgt_lang_nllb);
            results.push_back(tokenizer_->decode(output_ids));
        }

        is_translating_.store(false);
        return results;
    } catch (const std::exception& e) {
        is_translating_.store(false);
        spdlog::error("Batch translation failed: {}", e.what());
        throw std::runtime_error("Batch translation failed: " + std::string(e.what()));
    }
}

} // namespace nllb 
#include "nllb_translator.h"
#include "beam_search.h"
#include "tokenizer.h"
#include "tensor_utils.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <future>
#include <thread>
#include <cctype>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace nllb {

NLLBTranslator::NLLBTranslator(const common::TranslatorConfig& config)
    : ort_env_(ORT_LOGGING_LEVEL_WARNING, "NLLBTranslator")
    , memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    , model_dir_(config.nllb.model_dir)
    , target_lang_(config.nllb.target_lang)
    , is_initialized_(false) {
    
    try {
        // Initialize ONNX Runtime options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(config.nllb.params.num_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Load models using absolute paths
        encoder_session_ = std::make_unique<Ort::Session>(ort_env_, 
            config.nllb.model_files.encoder.c_str(), session_options);
        decoder_session_ = std::make_unique<Ort::Session>(ort_env_,
            config.nllb.model_files.decoder.c_str(), session_options);
        embed_lm_head_session_ = std::make_unique<Ort::Session>(ort_env_,
            config.nllb.model_files.embed_lm_head.c_str(), session_options);
        cache_initializer_session_ = std::make_unique<Ort::Session>(ort_env_,
            config.nllb.model_files.cache_initializer.c_str(), session_options);
            
        // Initialize tokenizer
        tokenizer_ = std::make_unique<Tokenizer>(config.nllb.model_files.tokenizer_vocab);
        
        // Load language codes and supported languages
        initialize_language_codes();
        load_supported_languages();
        
        // Set model parameters
        model_params_.beam_size = config.nllb.params.beam_size;
        model_params_.max_length = config.nllb.params.max_length;
        model_params_.length_penalty = config.nllb.params.length_penalty;
        model_params_.temperature = config.nllb.params.temperature;
        model_params_.top_k = config.nllb.params.top_k;
        model_params_.top_p = config.nllb.params.top_p;
        model_params_.repetition_penalty = config.nllb.params.repetition_penalty;
        model_params_.num_threads = config.nllb.params.num_threads;
        model_params_.support_low_quality_languages = config.nllb.params.support_low_quality_languages;
        
        is_initialized_ = true;
    } catch (const Ort::Exception& e) {
        set_error(translator::TranslatorError::ERROR_INIT, e.what());
    } catch (const std::exception& e) {
        set_error(translator::TranslatorError::ERROR_INIT, e.what());
    }
}

NLLBTranslator::~NLLBTranslator() = default;

std::string NLLBTranslator::translate(
    const std::string& text,
    const std::string& source_lang) const {
    
    if (!is_initialized_) {
        set_error(translator::TranslatorError::ERROR_INIT, "Translator not initialized");
        return "";
    }
    
    try {
        if (!needs_translation(source_lang)) {
            return text;
        }
        
        // Tokenize input text
        auto tokens = tokenizer_->encode(text, source_lang, target_lang_);
        
        // Run encoder
        auto encoder_output = run_encoder(tokens);
        
        // Initialize beam search
        BeamSearchDecoder beam_search(
            model_params_.beam_size,
            model_params_.length_penalty,
            tokenizer_->eos_id());
            
        // Initialize cache
        CacheContainer cache;
        
        // Run beam search
        auto hypotheses = beam_search.decode(
            *decoder_session_,
            *embed_lm_head_session_,
            memory_info_,
            encoder_output,
            {static_cast<int64_t>(tokens.input_ids.size())},
            cache,
            model_params_);
            
        // Get best hypothesis
        auto& best_hyp = hypotheses[0];
        
        // Decode tokens to text
        return tokenizer_->decode(best_hyp.tokens);
        
    } catch (const std::exception& e) {
        set_error(translator::TranslatorError::ERROR_TRANSLATION, e.what());
        return "";
    }
}

std::string NLLBTranslator::get_target_language() const {
    return target_lang_;
}

void NLLBTranslator::initialize_language_codes() {
    // Load language codes from config
    std::string config_path = model_dir_ + "/model_config.yaml";
    YAML::Node config = YAML::LoadFile(config_path);
    
    if (config["language_codes"]) {
        auto codes = config["language_codes"];
        for (const auto& code : codes) {
            auto nllb_code = code.first.as<std::string>();
            auto display_code = code.second.as<std::string>();
            nllb_language_codes_[display_code] = nllb_code;
            display_language_codes_[nllb_code] = display_code;
        }
    }
}

void NLLBTranslator::load_supported_languages() {
    supported_languages_.clear();
    std::string config_path = model_dir_ + "/model_config.yaml";
    YAML::Node config = YAML::LoadFile(config_path);
    
    if (config["supported_languages"]) {
        auto langs = config["supported_languages"];
        for (const auto& lang : langs) {
            supported_languages_.push_back(lang.as<std::string>());
        }
    }
    
    if (config["low_quality_languages"]) {
        auto langs = config["low_quality_languages"];
        for (const auto& lang : langs) {
            low_quality_languages_.push_back(lang.as<std::string>());
        }
    }
}

std::string NLLBTranslator::normalize_language_code(const std::string& lang_code) const {
    std::string normalized = lang_code;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    normalized.erase(
        std::remove_if(normalized.begin(), normalized.end(), ::isspace),
        normalized.end()
    );
    return normalized;
}

bool NLLBTranslator::is_language_supported(const std::string& lang_code) const {
    auto normalized_code = normalize_language_code(lang_code);
    auto it = std::find(supported_languages_.begin(), supported_languages_.end(), normalized_code);
    return it != supported_languages_.end() || 
           (model_params_.support_low_quality_languages && 
            std::find(low_quality_languages_.begin(), low_quality_languages_.end(), normalized_code) != low_quality_languages_.end());
}

bool NLLBTranslator::needs_translation(const std::string& source_lang) const {
    auto normalized_source = normalize_language_code(source_lang);
    auto normalized_target = normalize_language_code(target_lang_);
    return normalized_source != normalized_target;
}

void NLLBTranslator::set_support_low_quality_languages(bool support) {
    model_params_.support_low_quality_languages = support;
    load_supported_languages();
}

bool NLLBTranslator::get_support_low_quality_languages() const {
    return model_params_.support_low_quality_languages;
}

std::vector<std::string> NLLBTranslator::get_supported_languages() const {
    return supported_languages_;
}

Ort::Value NLLBTranslator::create_tensor(const std::vector<int64_t>& data,
                                        const std::vector<int64_t>& shape) const {
    return Ort::Value::CreateTensor<int64_t>(
        memory_info_,
        const_cast<int64_t*>(data.data()),
        data.size(),
        shape.data(),
        shape.size()
    );
}

Ort::Value NLLBTranslator::create_tensor(const std::vector<float>& data,
                                        const std::vector<int64_t>& shape) const {
    return Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(data.data()),
        data.size(),
        shape.data(),
        shape.size()
    );
}

std::vector<float> NLLBTranslator::run_encoder(const TokenizerResult& tokens) const {
    // Prepare input tensors
    std::vector<Ort::Value> input_tensors;
    
    // Input IDs tensor
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(tokens.input_ids.size())};
    input_tensors.push_back(create_tensor(tokens.input_ids, input_shape));
        
    // Attention mask tensor
    input_tensors.push_back(create_tensor(tokens.attention_mask, input_shape));
    
    // Get input/output names
    const char* input_names[] = {"input_ids", "attention_mask"};
    const char* output_names[] = {"encoder_outputs"};
        
    // Run encoder
    auto output_tensors = encoder_session_->Run(
        Ort::RunOptions{nullptr},
        input_names,
        input_tensors.data(),
        input_tensors.size(),
        output_names,
        1);
        
    // Get encoder output
    auto* output_data = output_tensors[0].GetTensorData<float>();
    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    
    return std::vector<float>(output_data, output_data + output_size);
}

std::vector<float> NLLBTranslator::run_embed_lm_head(const std::vector<int64_t>& input_ids) const {
    // Create input tensor
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};
    auto input_tensor = create_tensor(input_ids, input_shape);
    
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));
    
    // Get input/output names
    const char* input_names[] = {"input_ids"};
    const char* output_names[] = {"logits"};
    
    // Run model
    auto output_tensors = embed_lm_head_session_->Run(
        Ort::RunOptions{nullptr},
        input_names,
        input_tensors.data(),
        input_tensors.size(),
        output_names,
        1);
        
    // Get output
    auto* output_data = output_tensors[0].GetTensorData<float>();
    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    
    return std::vector<float>(output_data, output_data + output_size);
}

void NLLBTranslator::set_error(translator::TranslatorError error, const std::string& message) const {
    last_error_ = error;
    error_message_ = message;
    
    translator::TranslatorErrorInfo error_info{
        error,
        message,
        "NLLB Translator Error"
    };
    
    notify_error(error_info);
}

translator::TranslatorError NLLBTranslator::get_last_error() const {
    return last_error_;
}

std::string NLLBTranslator::get_error_message() const {
    return error_message_;
}

void NLLBTranslator::set_error_callback(std::shared_ptr<translator::TranslatorErrorCallback> callback) {
    error_callback_ = callback;
}

void NLLBTranslator::notify_error(const translator::TranslatorErrorInfo& error) const {
    if (error_callback_) {
        error_callback_->onError(error);
    }
}

void NLLBTranslator::reset_cache() {
    try {
        cache_container_.reset();
    } catch (const std::exception& e) {
        set_error(translator::TranslatorError::ERROR_CACHE, 
                 std::string("Failed to reset cache: ") + e.what());
    }
}

void NLLBTranslator::clear_cache() {
    try {
        cache_container_.clear();
    } catch (const std::exception& e) {
        set_error(translator::TranslatorError::ERROR_CACHE, 
                 std::string("Failed to clear cache: ") + e.what());
    }
}

// Parameter settings
void NLLBTranslator::set_beam_size(int size) {
    if (size <= 0) {
        set_error(translator::TranslatorError::ERROR_INVALID_PARAM, 
                 "Beam size must be positive");
        return;
    }
    model_params_.beam_size = size;
}

void NLLBTranslator::set_max_length(int length) {
    if (length <= 0) {
        set_error(translator::TranslatorError::ERROR_INVALID_PARAM, 
                 "Max length must be positive");
        return;
    }
    model_params_.max_length = length;
}

void NLLBTranslator::set_length_penalty(float penalty) {
    model_params_.length_penalty = penalty;
}

void NLLBTranslator::set_temperature(float temp) {
    if (temp <= 0) {
        set_error(translator::TranslatorError::ERROR_INVALID_PARAM, 
                 "Temperature must be positive");
        return;
    }
    model_params_.temperature = temp;
}

void NLLBTranslator::set_top_k(int k) {
    if (k < 0) {
        set_error(translator::TranslatorError::ERROR_INVALID_PARAM, 
                 "Top-k must be non-negative");
        return;
    }
    model_params_.top_k = k;
}

void NLLBTranslator::set_top_p(float p) {
    if (p <= 0 || p > 1) {
        set_error(translator::TranslatorError::ERROR_INVALID_PARAM, 
                 "Top-p must be in (0, 1]");
        return;
    }
    model_params_.top_p = p;
}

void NLLBTranslator::set_repetition_penalty(float penalty) {
    if (penalty < 1) {
        set_error(translator::TranslatorError::ERROR_INVALID_PARAM, 
                 "Repetition penalty must be >= 1");
        return;
    }
    model_params_.repetition_penalty = penalty;
}

void NLLBTranslator::set_num_threads(int threads) {
    if (threads <= 0) {
        set_error(translator::TranslatorError::ERROR_INVALID_PARAM, 
                 "Number of threads must be positive");
        return;
    }
    model_params_.num_threads = threads;
}

} // namespace nllb 
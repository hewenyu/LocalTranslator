#include "nllb_translator.h"
#include "translator/nllb-api/beam_search.h"
#include "translator/nllb-api/tokenizer.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <tinyxml2.h>
#include <future>
#include <thread>

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
        cache_init_session_ = std::make_unique<Ort::Session>(ort_env_, 
            config.nllb.model_files.cache_initializer.c_str(), session_options);
        embed_lm_head_session_ = std::make_unique<Ort::Session>(ort_env_, 
            config.nllb.model_files.embed_lm_head.c_str(), session_options);
        
        // Initialize tokenizer with absolute path
        tokenizer_ = std::make_unique<Tokenizer>(config.nllb.model_files.tokenizer_vocab);
        
        // Initialize beam search
        beam_search_decoder_ = std::make_unique<BeamSearchDecoder>(
            config.nllb.params.beam_size,
            config.nllb.params.length_penalty,
            tokenizer_->eos_id()
        );
        
        // Initialize cache container
        cache_container_ = std::make_unique<CacheContainer>();
        
        // Load language codes and supported languages
        initialize_language_codes();
        load_supported_languages();
        
        // Copy model parameters
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
        set_error(TranslatorError::ERROR_INIT, e.what());
    } catch (const std::exception& e) {
        set_error(TranslatorError::ERROR_INIT, e.what());
    }
}

NLLBTranslator::~NLLBTranslator() = default;

std::string NLLBTranslator::translate(
    const std::string& text,
    const std::string& source_lang) const {
    
    if (!is_initialized_) {
        set_error(TranslatorError::ERROR_INIT, "Translator not initialized");
        return "";
    }
    
    std::lock_guard<std::mutex> lock(translation_mutex_);
    is_translating_ = true;
    
    try {
        // 1. Tokenization
        auto tokens = tokenizer_->encode(text, source_lang, target_lang_);
        
        // 2. Encoder processing
        auto encoder_output = run_encoder(tokens);
        std::vector<int64_t> encoder_shape = {1, static_cast<int64_t>(tokens.input_ids.size()),
                                            static_cast<int64_t>(model_params_.hidden_size)};
        
        // 3. Initialize cache
        cache_container_->initialize(*cache_init_session_, memory_info_,
                                  encoder_output, encoder_shape);
        
        // 4. Beam Search decoding
        auto hypotheses = beam_search_decoder_->decode(
            *decoder_session_,
            *embed_lm_head_session_,
            memory_info_,
            encoder_output,
            encoder_shape,
            *cache_container_,
            model_params_
        );
        
        // 5. Select best hypothesis
        auto best_hypothesis = std::max_element(
            hypotheses.begin(),
            hypotheses.end(),
            [](const auto& a, const auto& b) { return a.score < b.score; }
        );
        
        // 6. Decode result
        auto result = tokenizer_->decode(best_hypothesis->tokens);
        is_translating_ = false;
        return result;
        
    } catch (const std::exception& e) {
        set_error(TranslatorError::ERROR_DECODE, e.what());
        is_translating_ = false;
        return "";
    }
}

std::vector<float> NLLBTranslator::run_encoder(
    const Tokenizer::TokenizerOutput& tokens) const {
    
    try {
        // 1. 创建输入 tensor
        auto input_tensor = TensorUtils::createInt64Tensor(
            memory_info_,
            tokens.input_ids,
            {1, static_cast<int64_t>(tokens.input_ids.size())}
        );
        
        auto attention_mask = TensorUtils::createInt64Tensor(
            memory_info_,
            tokens.attention_mask,
            {1, static_cast<int64_t>(tokens.attention_mask.size())}
        );
        
        // 2. 准备输入
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(input_tensor));
        ort_inputs.push_back(std::move(attention_mask));
        
        // 3. 运行编码器
        const char* input_names[] = {"input_ids", "attention_mask"};
        const char* output_names[] = {"encoder_output"};
        
        auto outputs = encoder_session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            ort_inputs.data(),
            ort_inputs.size(),
            output_names,
            1
        );
        
        // 4. 获取输出
        return TensorUtils::getTensorData<float>(outputs[0]);
        
    } catch (const Ort::Exception& e) {
        set_error(TranslatorError::ERROR_ENCODE, e.what());
        return {};
    }
}

void NLLBTranslator::set_error(TranslatorError error, const std::string& message) const {
    last_error_ = error;
    error_message_ = message;
}

TranslatorError NLLBTranslator::get_last_error() const {
    return last_error_;
}

std::string NLLBTranslator::get_error_message() const {
    return error_message_;
}

void NLLBTranslator::initialize_language_codes() {
    try {
        tinyxml2::XMLDocument doc;
        std::string xml_path = model_dir_ + "/nllb_supported_languages.xml";
        
        if (doc.LoadFile(xml_path.c_str()) != tinyxml2::XML_SUCCESS) {
            throw std::runtime_error("Failed to load language codes XML file: " + xml_path);
        }

        auto root = doc.FirstChildElement("languages");
        if (!root) {
            throw std::runtime_error("Invalid XML format: missing languages element");
        }

        for (auto lang = root->FirstChildElement("language"); 
             lang; 
             lang = lang->NextSiblingElement("language")) {
            
            auto code = lang->FirstChildElement("code");
            auto code_nllb = lang->FirstChildElement("code_NLLB");
            
            if (code && code_nllb) {
                std::string lang_code = code->GetText();
                std::string nllb_code = code_nllb->GetText();
                
                nllb_language_codes_[lang_code] = nllb_code;
                display_language_codes_[nllb_code] = lang_code;
            }
        }
        
        if (nllb_language_codes_.empty()) {
            throw std::runtime_error("No language codes loaded from XML");
        }
        
    } catch (const std::exception& e) {
        set_error(TranslatorError::ERROR_INIT, 
                 "Failed to initialize language codes: " + std::string(e.what()));
    }
}

void NLLBTranslator::load_supported_languages() {
    supported_languages_.clear();
    
    // 添加所有标准语言
    for (const auto& [lang_code, nllb_code] : nllb_language_codes_) {
        if (!model_params_.support_low_quality_languages) {
            // 如果不支持低质量语言，跳过它们
            if (std::find(low_quality_languages_.begin(), 
                         low_quality_languages_.end(), 
                         lang_code) != low_quality_languages_.end()) {
                continue;
            }
        }
        supported_languages_.push_back(lang_code);
    }
    
    // 按字母顺序排序
    std::sort(supported_languages_.begin(), supported_languages_.end());
}

std::string NLLBTranslator::normalize_language_code(const std::string& lang_code) const {
    std::string normalized = lang_code;
    
    // 转换为小写
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    
    // 移除区域标识符
    size_t separator_pos = normalized.find_first_of("-_");
    if (separator_pos != std::string::npos) {
        normalized = normalized.substr(0, separator_pos);
    }
    
    return normalized;
}

std::string NLLBTranslator::get_nllb_language_code(const std::string& lang_code) const {
    auto normalized = normalize_language_code(lang_code);
    auto it = nllb_language_codes_.find(normalized);
    
    if (it == nllb_language_codes_.end()) {
        set_error(TranslatorError::ERROR_INVALID_PARAM, 
                 "Unsupported language code: " + lang_code);
        return "";
    }
    
    return it->second;
}

std::string NLLBTranslator::get_display_language_code(const std::string& nllb_code) const {
    auto it = display_language_codes_.find(nllb_code);
    
    if (it == display_language_codes_.end()) {
        set_error(TranslatorError::ERROR_INVALID_PARAM, 
                 "Unknown NLLB code: " + nllb_code);
        return nllb_code;
    }
    
    return it->second;
}

bool NLLBTranslator::is_language_supported(const std::string& lang_code) const {
    auto normalized = normalize_language_code(lang_code);
    return std::find(supported_languages_.begin(), 
                    supported_languages_.end(), 
                    normalized) != supported_languages_.end();
}

std::vector<std::string> NLLBTranslator::get_supported_languages() const {
    return supported_languages_;
}

std::string NLLBTranslator::get_target_language() const {
    return target_lang_;
}

bool NLLBTranslator::needs_translation(const std::string& source_lang) const {
    return normalize_language_code(source_lang) != normalize_language_code(target_lang_);
}

// 配置管理方法实现
void NLLBTranslator::set_support_low_quality_languages(bool support) {
    model_params_.support_low_quality_languages = support;
    load_supported_languages();
}

bool NLLBTranslator::get_support_low_quality_languages() const {
    return model_params_.support_low_quality_languages;
}

void NLLBTranslator::set_beam_size(int size) {
    if (size <= 0) {
        set_error(TranslatorError::ERROR_INVALID_PARAM, "Beam size must be positive");
        return;
    }
    model_params_.beam_size = size;
    beam_search_decoder_ = std::make_unique<BeamSearchDecoder>(
        size,
        model_params_.length_penalty,
        tokenizer_->eos_id()
    );
}

void NLLBTranslator::set_max_length(int length) {
    if (length <= 0) {
        set_error(TranslatorError::ERROR_INVALID_PARAM, "Max length must be positive");
        return;
    }
    model_params_.max_length = length;
}

void NLLBTranslator::set_length_penalty(float penalty) {
    model_params_.length_penalty = penalty;
    beam_search_decoder_ = std::make_unique<BeamSearchDecoder>(
        model_params_.beam_size,
        penalty,
        tokenizer_->eos_id()
    );
}

void NLLBTranslator::set_temperature(float temp) {
    if (temp <= 0) {
        set_error(TranslatorError::ERROR_INVALID_PARAM, "Temperature must be positive");
        return;
    }
    model_params_.temperature = temp;
}

void NLLBTranslator::set_top_k(float k) {
    if (k < 0) {
        set_error(TranslatorError::ERROR_INVALID_PARAM, "Top-k must be non-negative");
        return;
    }
    model_params_.top_k = k;
}

void NLLBTranslator::set_top_p(float p) {
    if (p < 0 || p > 1) {
        set_error(TranslatorError::ERROR_INVALID_PARAM, "Top-p must be between 0 and 1");
        return;
    }
    model_params_.top_p = p;
}

void NLLBTranslator::set_repetition_penalty(float penalty) {
    if (penalty < 0) {
        set_error(TranslatorError::ERROR_INVALID_PARAM, "Repetition penalty must be non-negative");
        return;
    }
    model_params_.repetition_penalty = penalty;
}

void NLLBTranslator::set_num_threads(int threads) {
    if (threads <= 0) {
        set_error(TranslatorError::ERROR_INVALID_PARAM, "Number of threads must be positive");
        return;
    }
    model_params_.num_threads = threads;
}

std::vector<std::string> NLLBTranslator::translate_batch(
    const std::vector<std::string>& texts,
    const std::string& source_lang) const {
    
    if (!is_initialized_) {
        set_error(TranslatorError::ERROR_INIT, "Translator not initialized");
        return {};
    }
    
    if (texts.empty()) {
        return {};
    }
    
    std::lock_guard<std::mutex> lock(translation_mutex_);
    is_translating_ = true;
    
    try {
        // Tokenize all texts
        std::vector<Tokenizer::TokenizerOutput> batch_tokens;
        for (const auto& text : texts) {
            batch_tokens.push_back(tokenizer_->encode(text, source_lang, target_lang_));
        }
        
        // Run encoder for batch
        auto batch_encoder_output = run_encoder_batch(batch_tokens);
        
        // Process each sequence in parallel
        std::vector<std::string> results(texts.size());
        std::vector<std::unique_ptr<CacheContainer>> thread_caches;
        
        // Create cache containers for each thread
        for (size_t i = 0; i < texts.size(); i++) {
            thread_caches.push_back(std::make_unique<CacheContainer>());
        }
        
        // Process sequences in parallel
        std::vector<std::future<void>> futures;
        for (size_t i = 0; i < texts.size(); i++) {
            futures.push_back(std::async(std::launch::async,
                [this, i, &batch_encoder_output, &results, &thread_caches, &batch_tokens]() {
                    // Extract encoder output for this sequence
                    std::vector<float> encoder_output(
                        batch_encoder_output.begin() + i * model_params_.hidden_size,
                        batch_encoder_output.begin() + (i + 1) * model_params_.hidden_size
                    );
                    
                    std::vector<int64_t> encoder_shape = {
                        1,
                        static_cast<int64_t>(batch_tokens[i].input_ids.size()),
                        static_cast<int64_t>(model_params_.hidden_size)
                    };
                    
                    // Initialize cache for this sequence
                    thread_caches[i]->initialize(*cache_init_session_, memory_info_,
                                              encoder_output, encoder_shape);
                    
                    // Beam search decode
                    auto hypotheses = beam_search_decoder_->decode(
                        *decoder_session_,
                        *embed_lm_head_session_,
                        memory_info_,
                        encoder_output,
                        encoder_shape,
                        *thread_caches[i],
                        model_params_
                    );
                    
                    // Select best hypothesis
                    auto best_hypothesis = std::max_element(
                        hypotheses.begin(),
                        hypotheses.end(),
                        [](const auto& a, const auto& b) { return a.score < b.score; }
                    );
                    
                    // Store result
                    results[i] = tokenizer_->decode(best_hypothesis->tokens);
                }
            ));
        }
        
        // Wait for all sequences to complete
        for (auto& future : futures) {
            future.get();
        }
        
        is_translating_ = false;
        return results;
        
    } catch (const std::exception& e) {
        set_error(TranslatorError::ERROR_DECODE, e.what());
        is_translating_ = false;
        return {};
    }
}

std::vector<float> NLLBTranslator::run_encoder_batch(
    const std::vector<Tokenizer::TokenizerOutput>& batch_tokens) const {
    
    try {
        // Prepare batch input tensors
        std::vector<int64_t> batch_input_ids;
        std::vector<int64_t> batch_attention_mask;
        size_t max_length = 0;
        
        // Find max sequence length in batch
        for (const auto& tokens : batch_tokens) {
            max_length = std::max(max_length, tokens.input_ids.size());
        }
        
        // Pad sequences to max length
        for (const auto& tokens : batch_tokens) {
            batch_input_ids.insert(batch_input_ids.end(), 
                                 tokens.input_ids.begin(), 
                                 tokens.input_ids.end());
            batch_input_ids.insert(batch_input_ids.end(),
                                 max_length - tokens.input_ids.size(),
                                 tokenizer_->pad_id());
                                 
            batch_attention_mask.insert(batch_attention_mask.end(),
                                      tokens.attention_mask.begin(),
                                      tokens.attention_mask.end());
            batch_attention_mask.insert(batch_attention_mask.end(),
                                      max_length - tokens.attention_mask.size(),
                                      0);
        }
        
        // Create input tensors
        auto input_tensor = TensorUtils::createInt64Tensor(
            memory_info_,
            batch_input_ids,
            {static_cast<int64_t>(batch_tokens.size()), 
             static_cast<int64_t>(max_length)}
        );
        
        auto attention_mask = TensorUtils::createInt64Tensor(
            memory_info_,
            batch_attention_mask,
            {static_cast<int64_t>(batch_tokens.size()), 
             static_cast<int64_t>(max_length)}
        );
        
        // Run encoder
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(input_tensor));
        ort_inputs.push_back(std::move(attention_mask));
        
        const char* input_names[] = {"input_ids", "attention_mask"};
        const char* output_names[] = {"encoder_output"};
        
        auto outputs = encoder_session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            ort_inputs.data(),
            ort_inputs.size(),
            output_names,
            1
        );
        
        return TensorUtils::getTensorData<float>(outputs[0]);
        
    } catch (const Ort::Exception& e) {
        set_error(TranslatorError::ERROR_ENCODE, e.what());
        return {};
    }
}

std::vector<float> NLLBTranslator::run_embed_lm_head(
    const std::vector<int64_t>& input_ids) const {
    
    try {
        // 1. Create input tensor
        auto input_tensor = TensorUtils::createInt64Tensor(
            memory_info_,
            input_ids,
            {1, static_cast<int64_t>(input_ids.size())}
        );
        
        // 2. Prepare inputs
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(input_tensor));
        
        // 3. Run embed_lm_head model
        const char* input_names[] = {"input_ids"};
        const char* output_names[] = {"logits"};
        
        auto outputs = embed_lm_head_session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            ort_inputs.data(),
            ort_inputs.size(),
            output_names,
            1
        );
        
        // 4. Get output
        return TensorUtils::getTensorData<float>(outputs[0]);
        
    } catch (const Ort::Exception& e) {
        set_error(TranslatorError::ERROR_ENCODE, e.what());
        return {};
    }
}

} // namespace nllb 
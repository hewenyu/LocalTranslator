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
        // 初始化 ONNX Runtime 选项
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(config.nllb.params.num_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // 加载模型
        std::string encoder_path = model_dir_ + "/encoder.onnx";
        std::string decoder_path = model_dir_ + "/decoder.onnx";
        std::string cache_init_path = model_dir_ + "/cache_init.onnx";
        
        std::wstring wencoder_path(encoder_path.begin(), encoder_path.end());
        std::wstring wdecoder_path(decoder_path.begin(), decoder_path.end());
        std::wstring wcache_init_path(cache_init_path.begin(), cache_init_path.end());
        
        encoder_session_ = std::make_unique<Ort::Session>(ort_env_, wencoder_path.c_str(), session_options);
        decoder_session_ = std::make_unique<Ort::Session>(ort_env_, wdecoder_path.c_str(), session_options);
        cache_init_session_ = std::make_unique<Ort::Session>(ort_env_, wcache_init_path.c_str(), session_options);
        
        // 初始化分词器
        std::string vocab_path = model_dir_ + "/sentencepiece.model";
        tokenizer_ = std::make_unique<Tokenizer>(vocab_path);
        
        // 初始化 Beam Search
        beam_search_decoder_ = std::make_unique<BeamSearchDecoder>(
            config.nllb.params.beam_size,
            config.nllb.params.length_penalty,
            tokenizer_->eos_id()
        );
        
        // 初始化缓存容器
        cache_container_ = std::make_unique<CacheContainer>();
        
        // 加载语言代码和支持的语言
        initialize_language_codes();
        load_supported_languages();
        
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
        // 1. 分词
        auto tokens = tokenizer_->encode(text, source_lang, target_lang_);
        
        // 2. 编码器处理
        auto encoder_output = run_encoder(tokens);
        std::vector<int64_t> encoder_shape = {1, static_cast<int64_t>(tokens.input_ids.size()),
                                            static_cast<int64_t>(model_config_.hidden_size)};
        
        // 3. 初始化缓存
        cache_container_->initialize(*cache_init_session_, memory_info_,
                                  encoder_output, encoder_shape);
        
        // 4. Beam Search 解码
        auto hypotheses = beam_search_decoder_->decode(
            *decoder_session_,
            memory_info_,
            encoder_output,
            encoder_shape,
            *cache_container_
        );
        
        // 5. 选择最佳假设
        auto best_hypothesis = std::max_element(
            hypotheses.begin(),
            hypotheses.end(),
            [](const auto& a, const auto& b) { return a.score < b.score; }
        );
        
        // 6. 解码结果
        return tokenizer_->decode(best_hypothesis->tokens);
        
    } catch (const std::exception& e) {
        set_error(TranslatorError::ERROR_DECODE, e.what());
        return "";
    }
    
    is_translating_ = false;
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
        if (!model_config_.support_low_quality_languages) {
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
    auto src_normalized = normalize_language_code(source_lang);
    auto tgt_normalized = normalize_language_code(target_lang_);
    return src_normalized != tgt_normalized;
}

// 配置管理方法实现
void NLLBTranslator::set_support_low_quality_languages(bool support) {
    if (model_config_.support_low_quality_languages != support) {
        model_config_.support_low_quality_languages = support;
        load_supported_languages();
    }
}

bool NLLBTranslator::get_support_low_quality_languages() const {
    return model_config_.support_low_quality_languages;
}

void NLLBTranslator::set_beam_size(int size) {
    if (size <= 0) {
        set_error(TranslatorError::ERROR_INVALID_PARAM, "Beam size must be positive");
        return;
    }
    model_config_.beam_size = size;
    beam_search_decoder_ = std::make_unique<BeamSearchDecoder>(
        size,
        model_config_.length_penalty,
        tokenizer_->eos_id()
    );
}

void NLLBTranslator::set_max_length(int length) {
    if (length <= 0) {
        set_error(TranslatorError::ERROR_INVALID_PARAM, "Max length must be positive");
        return;
    }
    model_config_.max_length = length;
}

void NLLBTranslator::set_length_penalty(float penalty) {
    model_config_.length_penalty = penalty;
    beam_search_decoder_ = std::make_unique<BeamSearchDecoder>(
        model_config_.beam_size,
        penalty,
        tokenizer_->eos_id()
    );
}

void NLLBTranslator::set_temperature(float temp) {
    if (temp <= 0) {
        set_error(TranslatorError::ERROR_INVALID_PARAM, "Temperature must be positive");
        return;
    }
    model_config_.temperature = temp;
}

void NLLBTranslator::set_top_k(float k) {
    if (k < 0) {
        set_error(TranslatorError::ERROR_INVALID_PARAM, "Top-k must be non-negative");
        return;
    }
    model_config_.top_k = k;
}

void NLLBTranslator::set_top_p(float p) {
    if (p < 0 || p > 1) {
        set_error(TranslatorError::ERROR_INVALID_PARAM, "Top-p must be between 0 and 1");
        return;
    }
    model_config_.top_p = p;
}

void NLLBTranslator::set_repetition_penalty(float penalty) {
    if (penalty < 0) {
        set_error(TranslatorError::ERROR_INVALID_PARAM, "Repetition penalty must be non-negative");
        return;
    }
    model_config_.repetition_penalty = penalty;
}

void NLLBTranslator::set_num_threads(int threads) {
    if (threads <= 0) {
        set_error(TranslatorError::ERROR_INVALID_PARAM, "Number of threads must be positive");
        return;
    }
    model_config_.num_threads = threads;
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
    
    // 检查是否需要翻译
    if (!needs_translation(source_lang)) {
        return texts;
    }
    
    std::lock_guard<std::mutex> lock(translation_mutex_);
    is_translating_ = true;
    
    try {
        // 1. 批量分词
        std::vector<Tokenizer::TokenizerOutput> batch_tokens;
        batch_tokens.reserve(texts.size());
        
        for (const auto& text : texts) {
            batch_tokens.push_back(tokenizer_->encode(text, source_lang, target_lang_));
        }
        
        // 2. 批量编码
        auto batch_encoder_output = run_encoder_batch(batch_tokens);
        
        // 3. 并行解码
        std::vector<std::string> results(texts.size());
        std::vector<std::future<void>> futures;
        
        const int num_threads = std::min(
            static_cast<size_t>(std::thread::hardware_concurrency()),
            texts.size()
        );
        
        std::vector<std::unique_ptr<CacheContainer>> thread_caches(num_threads);
        for (int i = 0; i < num_threads; ++i) {
            thread_caches[i] = std::make_unique<CacheContainer>();
        }
        
        // 分配工作给线程
        for (size_t i = 0; i < texts.size(); i += num_threads) {
            for (int t = 0; t < num_threads && (i + t) < texts.size(); ++t) {
                futures.push_back(std::async(std::launch::async,
                    [this, i, t, &batch_encoder_output, &results, &thread_caches, &batch_tokens]() {
                        const size_t idx = i + t;
                        const auto& tokens = batch_tokens[idx];
                        
                        // 获取当前文本的编码器输出
                        std::vector<float> encoder_output(
                            batch_encoder_output.begin() + idx * model_config_.hidden_size,
                            batch_encoder_output.begin() + (idx + 1) * model_config_.hidden_size
                        );
                        
                        std::vector<int64_t> encoder_shape = {1, static_cast<int64_t>(tokens.input_ids.size()),
                                            static_cast<int64_t>(model_config_.hidden_size)};
                        
                        // 初始化缓存
                        thread_caches[t]->initialize(*cache_init_session_, memory_info_,
                                                  encoder_output, encoder_shape);
                        
                        // Beam Search 解码
                        auto hypotheses = beam_search_decoder_->decode(
                            *decoder_session_,
                            memory_info_,
                            encoder_output,
                            encoder_shape,
                            *thread_caches[t]
                        );
                        
                        // 选择最佳假设
                        auto best_hypothesis = std::max_element(
                            hypotheses.begin(),
                            hypotheses.end(),
                            [](const auto& a, const auto& b) { return a.score < b.score; }
                        );
                        
                        // 保存结果
                        results[idx] = tokenizer_->decode(best_hypothesis->tokens);
                    }
                ));
            }
            
            // 等待当前批次完成
            for (auto& future : futures) {
                future.wait();
            }
            futures.clear();
        }
        
        is_translating_ = false;
        return results;
        
    } catch (const std::exception& e) {
        set_error(TranslatorError::ERROR_BATCH_PROCESSING, e.what());
        is_translating_ = false;
        return {};
    }
}

std::vector<float> NLLBTranslator::run_encoder_batch(
    const std::vector<Tokenizer::TokenizerOutput>& batch_tokens) const {
    
    try {
        // 1. 计算最大序列长度
        size_t max_length = 0;
        for (const auto& tokens : batch_tokens) {
            max_length = std::max(max_length, tokens.input_ids.size());
        }
        
        // 2. 准备批处理输入
        const size_t batch_size = batch_tokens.size();
        std::vector<int64_t> batch_input_ids;
        std::vector<int64_t> batch_attention_mask;
        
        batch_input_ids.reserve(batch_size * max_length);
        batch_attention_mask.reserve(batch_size * max_length);
        
        // 填充输入
        for (const auto& tokens : batch_tokens) {
            // 复制当前序列的 input_ids
            batch_input_ids.insert(batch_input_ids.end(),
                                 tokens.input_ids.begin(),
                                 tokens.input_ids.end());
            
            // 填充到最大长度
            batch_input_ids.insert(batch_input_ids.end(),
                                 max_length - tokens.input_ids.size(),
                                 tokenizer_->pad_id());
            
            // 复制当前序列的 attention_mask
            batch_attention_mask.insert(batch_attention_mask.end(),
                                     tokens.attention_mask.begin(),
                                     tokens.attention_mask.end());
            
            // 填充 attention_mask
            batch_attention_mask.insert(batch_attention_mask.end(),
                                     max_length - tokens.attention_mask.size(),
                                     0);
        }
        
        // 3. 创建输入 tensors
        auto input_tensor = TensorUtils::createInt64Tensor(
            memory_info_,
            batch_input_ids,
            {static_cast<int64_t>(batch_size),
             static_cast<int64_t>(max_length)}
        );
        
        auto attention_mask = TensorUtils::createInt64Tensor(
            memory_info_,
            batch_attention_mask,
            {static_cast<int64_t>(batch_size),
             static_cast<int64_t>(max_length)}
        );
        
        // 4. 运行编码器
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
        
        // 5. 获取输出
        return TensorUtils::getTensorData<float>(outputs[0]);
        
    } catch (const Ort::Exception& e) {
        set_error(TranslatorError::ERROR_ENCODE, e.what());
        return {};
    }
}

} // namespace nllb 
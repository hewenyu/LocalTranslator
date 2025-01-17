#pragma once

#include <string>
#include <memory>
#include <map>
#include <vector>
#include <mutex>
#include <atomic>
#include "translator/translator.h"
#include "translator/nllb-api/tokenizer.h"
#include "translator/nllb-api/beam_search.h"
#include "translator/nllb-api/language_detector.h"
#include <onnxruntime_cxx_api.h>

namespace nllb {

// Error codes matching RTranslator
enum class TranslatorError {
    OK = 0,
    ERROR_INIT = -1,
    ERROR_TOKENIZE = -2,
    ERROR_ENCODE = -3,
    ERROR_DECODE = -4,
    ERROR_MEMORY = -5,
    ERROR_INVALID_PARAM = -6
};

struct ModelConfig {
    int hidden_size;
    int num_heads;
    int num_layers;
    int vocab_size;
    int max_position_embeddings;
    int encoder_layers;
    int decoder_layers;
    bool support_low_quality_languages;
    float eos_penalty;
    int max_batch_size;
    
    // Parameters matching RTranslator
    int beam_size = 5;
    int max_length = 128;
    float length_penalty = 1.0f;
    float temperature = 1.0f;
    float top_k = 0;  // 0 means disabled
    float top_p = 0.9f;  // nucleus sampling threshold
    float repetition_penalty = 0.9f;
    int num_threads = 4;
    bool use_cache = true;
    
    // Model type
    enum class ModelType {
        NLLB,
        NLLB_CACHE,
        MADLAD,
        MADLAD_CACHE
    } model_type = ModelType::NLLB;
};

class NLLBTranslator : public translator::ITranslator {
public:
    explicit NLLBTranslator(const common::TranslatorConfig& config);
    ~NLLBTranslator() override;

    // Core translation functionality
    std::string translate(const std::string& text, const std::string& source_lang) const override;
    std::string get_target_language() const override;

    // Language detection and management
    std::string detect_language(const std::string& text) const;
    bool needs_translation(const std::string& source_lang) const;
    std::vector<std::string> get_supported_languages() const;
    bool is_language_supported(const std::string& lang_code) const;
    
    // Batch translation
    std::vector<std::string> translate_batch(const std::vector<std::string>& texts, 
                                           const std::string& source_lang) const;

    // Language code conversion
    std::string get_nllb_language_code(const std::string& lang_code) const;
    std::string get_display_language_code(const std::string& nllb_code) const;
    
    // Configuration management
    void set_support_low_quality_languages(bool support);
    bool get_support_low_quality_languages() const;
    void set_eos_penalty(float penalty);
    float get_eos_penalty() const;

    // New configuration methods
    void set_beam_size(int size);
    void set_max_length(int length);
    void set_length_penalty(float penalty);
    void set_temperature(float temp);
    void set_top_k(float k);
    void set_top_p(float p);
    void set_repetition_penalty(float penalty);

    // Error handling
    TranslatorError get_last_error() const;
    std::string get_error_message() const;

private:
    // ONNX Runtime sessions
    Ort::Env ort_env_;
    std::unique_ptr<Ort::Session> encoder_session_;
    std::unique_ptr<Ort::Session> decoder_session_;
    std::unique_ptr<Ort::Session> embed_lm_head_session_;
    std::unique_ptr<Ort::Session> cache_init_session_;
    std::unique_ptr<Ort::Session> embed_session_;

    // Tokenizer and language detector
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<LanguageDetector> language_detector_;

    // Configuration
    std::string model_dir_;
    std::string target_lang_;
    common::NLLBConfig::Parameters params_;
    ModelConfig model_config_;
    BeamSearchConfig beam_config_;
    std::unique_ptr<BeamSearchDecoder> beam_search_decoder_;
    
    // Language support
    std::map<std::string, std::string> nllb_language_codes_;
    std::map<std::string, std::string> display_language_codes_;
    std::vector<std::string> supported_languages_;
    std::vector<std::string> low_quality_languages_;
    
    // State management
    bool is_initialized_;
    mutable std::mutex translation_mutex_;
    mutable std::atomic<bool> is_translating_{false};
    mutable TranslatorError last_error_{TranslatorError::OK};
    mutable std::string error_message_;

    // Initialization methods
    void initialize_language_codes();
    bool initialize_language_detector();
    void load_models();
    void load_supported_languages();
    
    // Helper methods
    std::string normalize_language_code(const std::string& lang_code) const;
    void set_error(TranslatorError error, const std::string& message) const;
    
    // Model inference methods
    std::vector<float> run_encoder(const Tokenizer::TokenizerOutput& tokens) const;
    std::vector<float> run_embedding(const std::vector<int64_t>& input_ids) const;
    std::vector<int64_t> run_decoder(const std::vector<float>& encoder_output,
                                   const std::string& target_lang) const;
};

} // namespace nllb 
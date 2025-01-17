#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "translator/translator.h"
#include "translator/translator_error.h"
#include "common/common.h"
#include "cache_container.h"
#include "tokenizer.h"
#include "beam_search.h"

namespace nllb {

class NLLBTranslator : public translator::ITranslator {
public:
    explicit NLLBTranslator(const common::TranslatorConfig& config);
    ~NLLBTranslator() override;

    // Core translation methods
    std::string translate(const std::string& text, 
                         const std::string& source_lang) const override;
    std::string get_target_language() const override;
    
    // Language support methods
    bool is_language_supported(const std::string& lang_code) const;
    bool needs_translation(const std::string& source_lang) const;
    void set_support_low_quality_languages(bool support);
    bool get_support_low_quality_languages() const;
    std::vector<std::string> get_supported_languages() const;

    // Batch translation
    std::vector<std::string> translate_batch(
        const std::vector<std::string>& texts,
        const std::string& source_lang) const;

    // Cache management
    void reset_cache();
    
    // Error handling
    translator::TranslatorError get_last_error() const;
    std::string get_error_message() const;

    // Parameter settings
    void set_beam_size(int size);
    void set_max_length(int length);
    void set_length_penalty(float penalty);
    void set_temperature(float temp);
    void set_top_k(float k);
    void set_top_p(float p);
    void set_repetition_penalty(float penalty);
    void set_num_threads(int threads);

private:
    // ONNX Runtime environment
    Ort::Env ort_env_;
    Ort::MemoryInfo memory_info_;
    std::unique_ptr<Ort::Session> encoder_session_;
    std::unique_ptr<Ort::Session> decoder_session_;
    std::unique_ptr<Ort::Session> cache_initializer_session_;
    std::unique_ptr<Ort::Session> embed_lm_head_session_;

    // Core components
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<BeamSearchDecoder> beam_search_decoder_;
    mutable std::unique_ptr<CacheContainer> cache_container_;

    // Configuration
    std::string model_dir_;
    std::string target_lang_;
    bool is_initialized_;
    mutable std::mutex translation_mutex_;
    mutable std::atomic<bool> is_translating_{false};
    mutable translator::TranslatorError last_error_{translator::TranslatorError::OK};
    mutable std::string error_message_;

    // Language mapping
    std::unordered_map<std::string, std::string> nllb_language_codes_;
    std::unordered_map<std::string, std::string> display_language_codes_;
    std::vector<std::string> supported_languages_;
    std::vector<std::string> low_quality_languages_;

    // Model parameters
    struct ModelParams {
        int beam_size{5};
        int max_length{128};
        float length_penalty{1.0f};
        float temperature{1.0f};
        float top_k{0};
        float top_p{0.9f};
        float repetition_penalty{0.9f};
        int num_threads{4};
        int hidden_size{1024};
        bool support_low_quality_languages{false};
    } model_params_;

    // Helper methods
    std::vector<float> run_encoder(const TokenizerResult& tokens) const;
    std::vector<float> run_embed_lm_head(const std::vector<int64_t>& input_ids) const;
    void set_error(translator::TranslatorError error, const std::string& message) const;
    void initialize_language_codes();
    void load_supported_languages();
    std::string normalize_language_code(const std::string& lang_code) const;
    std::string get_nllb_language_code(const std::string& lang_code) const;
    std::string get_display_language_code(const std::string& nllb_code) const;
    Ort::Value create_tensor(const std::vector<int64_t>& data, 
                            const std::vector<int64_t>& shape) const;
    Ort::Value create_tensor(const std::vector<float>& data, 
                            const std::vector<int64_t>& shape) const;
};

} // namespace nllb 
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
#include "model_params.h"

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

    // Error handling
    translator::TranslatorError get_last_error() const;
    std::string get_error_message() const;
    void set_error_callback(std::shared_ptr<translator::TranslatorErrorCallback> callback);

    // Parameter settings
    void set_beam_size(int size);
    void set_max_length(int length);
    void set_length_penalty(float penalty);
    void set_temperature(float temp);
    void set_top_k(int k);
    void set_top_p(float p);
    void set_repetition_penalty(float penalty);
    void set_num_threads(int threads);

    // Cache management
    void reset_cache();
    void clear_cache();

private:
    // ONNX Runtime components
    Ort::Env ort_env_;
    Ort::MemoryInfo memory_info_;
    std::unique_ptr<Ort::Session> encoder_session_;
    std::unique_ptr<Ort::Session> decoder_session_;
    std::unique_ptr<Ort::Session> cache_init_session_;
    std::unique_ptr<Ort::Session> embed_lm_head_session_;
    std::unique_ptr<Ort::Session> embed_session_;

    // Tokenizer and cache
    std::unique_ptr<Tokenizer> tokenizer_;
    mutable CacheContainer cache_container_;

    // Configuration
    std::string model_dir_;
    std::string target_lang_;
    ModelParams model_params_;
    bool is_initialized_;

    // Language support
    std::vector<std::string> supported_languages_;
    std::vector<std::string> low_quality_languages_;

    // Error handling
    mutable translator::TranslatorError last_error_;
    mutable std::string error_message_;
    std::shared_ptr<translator::TranslatorErrorCallback> error_callback_;

    // Helper methods
    void load_language_config();
    std::string normalize_language_code(const std::string& lang_code) const;
    void set_error(translator::TranslatorError error, const std::string& message) const;
    void notify_error(const translator::TranslatorErrorInfo& error) const;

    // Translation pipeline methods
    Ort::Value run_encoder(const std::vector<int64_t>& input_ids) const;
    std::vector<int64_t> run_decoder(const Ort::Value& encoder_output,
                                   const std::vector<int64_t>& encoder_shape) const;
    std::string get_display_language_code(const std::string& nllb_code) const;
    
    // Tensor utilities
    Ort::Value create_tensor(const std::vector<int64_t>& data, 
                            const std::vector<int64_t>& shape) const;
    Ort::Value create_tensor(const std::vector<float>& data, 
                            const std::vector<int64_t>& shape) const;
};

} // namespace nllb 
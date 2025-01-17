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

struct ModelConfig {
    int hidden_size;
    int num_heads;
    int num_layers;
    int vocab_size;
    int max_position_embeddings;
    int encoder_layers;
    int decoder_layers;

    ModelConfig(int hidden_size = 1024, int num_heads = 16, int num_layers = 24)
        : hidden_size(hidden_size), num_heads(num_heads), num_layers(num_layers),
          vocab_size(256200), max_position_embeddings(1024),
          encoder_layers(24), decoder_layers(24) {}

    static ModelConfig load_from_yaml(const std::string& config_path);
};

class NLLBTranslator : public translator::ITranslator {
public:
    explicit NLLBTranslator(const common::TranslatorConfig& config);
    ~NLLBTranslator() override;

    // 核心翻译功能
    std::string translate(const std::string& text, const std::string& source_lang) const override;
    std::string get_target_language() const override;

    // 语言检测和管理
    std::string detect_language(const std::string& text) const;
    bool needs_translation(const std::string& source_lang) const;
    std::vector<std::string> get_supported_languages() const;
    bool is_language_supported(const std::string& lang_code) const;
    
    // 批量翻译
    std::vector<std::string> translate_batch(const std::vector<std::string>& texts, 
                                           const std::string& source_lang) const;

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
    std::vector<std::string> supported_languages_;
    
    // State management
    bool is_initialized_;
    mutable std::mutex translation_mutex_;
    std::atomic<bool> is_translating_;

    // Initialization methods
    void initialize_language_codes();
    bool initialize_language_detector();
    void load_models();
    void load_supported_languages();
    
    // Helper methods
    std::string get_nllb_language_code(const std::string& lang_code) const;
    std::string normalize_language_code(const std::string& lang_code) const;
    
    // Model inference methods
    std::vector<float> run_encoder(const Tokenizer::TokenizerOutput& tokens) const;
    std::vector<float> run_embedding(const std::vector<int64_t>& input_ids) const;
    std::vector<int64_t> run_decoder(const std::vector<float>& encoder_output,
                                   const std::string& target_lang) const;
};

} // namespace nllb 
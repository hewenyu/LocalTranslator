#pragma once

#include <string>
#include <memory>
#include <map>
#include "translator/translator.h"
#include <onnxruntime_cxx_api.h>

namespace nllb {

class NLLBTranslator : public translator::ITranslator {
public:
    explicit NLLBTranslator(const common::TranslatorConfig& config);
    ~NLLBTranslator() override;

    std::string translate(const std::string& text, const std::string& source_lang) const override;
    std::string get_target_language() const override;

private:
    // ONNX Runtime environment and sessions
    Ort::Env ort_env_;
    std::unique_ptr<Ort::Session> encoder_session_;
    std::unique_ptr<Ort::Session> decoder_session_;
    std::unique_ptr<Ort::Session> embed_lm_head_session_;
    std::unique_ptr<Ort::Session> cache_init_session_;

    // Model paths
    std::string model_dir_;
    std::string target_lang_;
    
    // Language code mappings
    std::map<std::string, std::string> nllb_language_codes_;

    // Helper methods
    void initialize_language_codes();
    std::string get_nllb_language_code(const std::string& lang_code) const;
    void load_models();
    
    // Tokenization and model inference
    struct TokenizerOutput {
        std::vector<int64_t> input_ids;
        std::vector<int64_t> attention_mask;
    };
    
    TokenizerOutput tokenize(const std::string& text, 
                           const std::string& source_lang, 
                           const std::string& target_lang) const;
                           
    std::vector<float> run_encoder(const TokenizerOutput& tokens) const;
    std::string run_decoder(const std::vector<float>& encoder_output,
                          const std::string& target_lang) const;
};

} // namespace nllb 
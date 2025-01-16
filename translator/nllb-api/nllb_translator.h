#pragma once

#include <string>
#include <memory>
#include <map>
#include "translator/translator.h"
#include "translator/nllb-api/tokenizer.h"
#include "translator/nllb-api/beam_search.h"
#include <onnxruntime_cxx_api.h>

namespace nllb {

struct ModelConfig {
    int hidden_size;
    int num_heads;
    int vocab_size;
    int max_position_embeddings;
    int encoder_layers;
    int decoder_layers;

    static ModelConfig load_from_yaml(const std::string& config_path);
};

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
    std::unique_ptr<Ort::Session> embed_session_;

    // Tokenizer
    std::unique_ptr<Tokenizer> tokenizer_;

    // Model paths and config
    std::string model_dir_;
    std::string target_lang_;
    common::NLLBConfig::Parameters params_;
    ModelConfig model_config_;
    BeamSearchConfig beam_config_;
    std::unique_ptr<BeamSearchDecoder> beam_search_decoder_;
    
    // Language code mappings
    std::map<std::string, std::string> nllb_language_codes_;

    // Helper methods
    void initialize_language_codes();
    std::string get_nllb_language_code(const std::string& lang_code) const;
    void load_models();
    
    // Model inference
    std::vector<float> run_encoder(const Tokenizer::TokenizerOutput& tokens) const;
    std::vector<float> run_embedding(const std::vector<int64_t>& input_ids) const;
    std::vector<int64_t> run_decoder(const std::vector<float>& encoder_output,
                                   const std::string& target_lang) const;
};

} // namespace nllb 
#pragma once

#include <string>
#include <memory>
#include <map>
#include "translator/translator.h"
#include "translator/nllb-api/tokenizer.h"
#include "translator/nllb-api/beam_search.h"
#include <onnxruntime_cxx_api.h>

namespace nllb {

class CacheState {
public:
    CacheState(int max_length, int hidden_size, int num_heads, int num_layers)
        : max_length_(max_length), hidden_size_(hidden_size), num_heads_(num_heads),
          decoder_keys_(num_layers), decoder_values_(num_layers),
          encoder_keys_(num_layers), encoder_values_(num_layers) {}

    Ort::Value get_decoder_key(int layer) const { return std::move(decoder_keys_[layer]); }
    Ort::Value get_decoder_value(int layer) const { return std::move(decoder_values_[layer]); }
    Ort::Value get_encoder_key(int layer) const { return std::move(encoder_keys_[layer]); }
    Ort::Value get_encoder_value(int layer) const { return std::move(encoder_values_[layer]); }

    void update_decoder_key(int layer, Ort::Value key) { decoder_keys_[layer] = std::move(key); }
    void update_decoder_value(int layer, Ort::Value value) { decoder_values_[layer] = std::move(value); }
    void update_encoder_key(int layer, Ort::Value key) { encoder_keys_[layer] = std::move(key); }
    void update_encoder_value(int layer, Ort::Value value) { encoder_values_[layer] = std::move(value); }

private:
    int max_length_;
    int hidden_size_;
    int num_heads_;
    std::vector<Ort::Value> decoder_keys_;
    std::vector<Ort::Value> decoder_values_;
    std::vector<Ort::Value> encoder_keys_;
    std::vector<Ort::Value> encoder_values_;
};

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
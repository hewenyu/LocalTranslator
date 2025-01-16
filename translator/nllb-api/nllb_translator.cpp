#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include "translator/nllb-api/nllb_translator.h"

namespace nllb {

NLLBTranslator::NLLBTranslator(const common::TranslatorConfig& config) 
    : ort_env_(ORT_LOGGING_LEVEL_WARNING, "nllb_translator") {
    model_dir_ = config.nllb.model_dir;
    target_lang_ = config.nllb.target_lang;
    
    // Initialize
    initialize_language_codes();
    load_models();
}

NLLBTranslator::~NLLBTranslator() = default;

void NLLBTranslator::initialize_language_codes() {
    // Load language codes from YAML file
    std::string lang_file = model_dir_ + "/nllb_languages.yaml";
    try {
        YAML::Node config = YAML::LoadFile(lang_file);
        for (const auto& lang : config["languages"]) {
            std::string code = lang["code"].as<std::string>();
            std::string nllb_code = lang["code_NLLB"].as<std::string>();
            nllb_language_codes_[code] = nllb_code;
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load language codes: " + std::string(e.what()));
    }
}

void NLLBTranslator::load_models() {
    Ort::SessionOptions session_opts;
    session_opts.SetIntraOpNumThreads(1);
    session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    // Load encoder
    std::string encoder_path = model_dir_ + "/NLLB_encoder.onnx";
    encoder_session_ = std::make_unique<Ort::Session>(ort_env_, 
        encoder_path.c_str(), session_opts);

    // Load decoder
    std::string decoder_path = model_dir_ + "/NLLB_decoder.onnx";
    decoder_session_ = std::make_unique<Ort::Session>(ort_env_,
        decoder_path.c_str(), session_opts);

    // Load embed and lm head
    std::string embed_path = model_dir_ + "/NLLB_embed_and_lm_head.onnx";
    embed_lm_head_session_ = std::make_unique<Ort::Session>(ort_env_,
        embed_path.c_str(), session_opts);

    // Load cache initializer
    std::string cache_path = model_dir_ + "/NLLB_cache_initializer.onnx";
    cache_init_session_ = std::make_unique<Ort::Session>(ort_env_,
        cache_path.c_str(), session_opts);
}

std::string NLLBTranslator::get_nllb_language_code(const std::string& lang_code) const {
    auto it = nllb_language_codes_.find(lang_code);
    if (it == nllb_language_codes_.end()) {
        throw std::runtime_error("Unsupported language code: " + lang_code);
    }
    return it->second;
}

NLLBTranslator::TokenizerOutput NLLBTranslator::tokenize(
    const std::string& text,
    const std::string& source_lang,
    const std::string& target_lang) const {
    // TODO: Implement SentencePiece tokenization
    // This is a placeholder implementation
    TokenizerOutput output;
    return output;
}

std::vector<float> NLLBTranslator::run_encoder(const TokenizerOutput& tokens) const {
    // Prepare input tensors
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(tokens.input_ids.size())};
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        
    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, tokens.input_ids.data(), tokens.input_ids.size(), 
        input_shape.data(), input_shape.size());
        
    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, tokens.attention_mask.data(), tokens.attention_mask.size(),
        input_shape.data(), input_shape.size());

    // Run encoder
    const char* input_names[] = {"input_ids", "attention_mask"};
    const char* output_names[] = {"encoder_output"};
    
    auto output_tensors = encoder_session_->Run(
        Ort::RunOptions{nullptr},
        input_names, 
        {input_ids_tensor, attention_mask_tensor},
        2,
        output_names,
        1);

    // Get output data
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    
    return std::vector<float>(output_data, output_data + output_size);
}

std::string NLLBTranslator::run_decoder(
    const std::vector<float>& encoder_output,
    const std::string& target_lang) const {
    // TODO: Implement decoder logic
    // This is a placeholder implementation
    return "";
}

std::string NLLBTranslator::translate(
    const std::string& text,
    const std::string& source_lang) const {
    try {
        // Convert language codes
        std::string nllb_source = get_nllb_language_code(source_lang);
        std::string nllb_target = get_nllb_language_code(target_lang_);

        // Tokenize input
        auto tokens = tokenize(text, nllb_source, nllb_target);

        // Run encoder
        auto encoder_output = run_encoder(tokens);

        // Run decoder
        return run_decoder(encoder_output, nllb_target);
    } catch (const std::exception& e) {
        throw std::runtime_error("Translation failed: " + std::string(e.what()));
    }
}

std::string NLLBTranslator::get_target_language() const {
    return target_lang_;
}

} // namespace nllb 
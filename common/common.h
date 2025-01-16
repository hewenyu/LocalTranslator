#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include <iostream>

namespace common {
// DeepLX translator Config
struct DeepLXConfig {
    std::string url;
    std::string token;
    std::string target_lang = "ZH";  // Default target language is Chinese
};

// NLLB translator Config
struct NLLBConfig {
    std::string model_dir;  // Directory containing all NLLB model files
    std::string target_lang = "ZH";  // Default target language is Chinese
    
    // Model file names (relative to model_dir)
    struct ModelFiles {
        std::string encoder = "NLLB_encoder.onnx";
        std::string decoder = "NLLB_decoder.onnx";
        std::string embed_lm_head = "NLLB_embed_and_lm_head.onnx";
        std::string cache_initializer = "NLLB_cache_initializer.onnx";
        std::string tokenizer_vocab = "sentencepiece_bpe.model";
        std::string language_config = "nllb_languages.yaml";
    } model_files;

    // Model parameters
    struct Parameters {
        int beam_size = 5;  // Beam search size
        int max_length = 128;  // Maximum sequence length
        float length_penalty = 1.0;  // Length penalty for beam search
        bool use_cache = true;  // Whether to use decoder cache
        int num_threads = 1;  // Number of threads for model inference
    } params;
};

// struct for translator config
struct TranslatorConfig {
    // type of translator
    std::string type;  // DeepLX, NLLB, None
    common::DeepLXConfig deeplx;
    common::NLLBConfig nllb;

    static TranslatorConfig LoadFromFile(const std::string& config_path) {
        TranslatorConfig config;
        try {
            // yaml load file
            YAML::Node yaml_config = YAML::LoadFile(config_path);
            
            // Load type
            if (yaml_config["type"]) {
                config.type = yaml_config["type"].as<std::string>();
            }
            
            // Load DeepLX config
            if (yaml_config["deeplx"]) {
                auto deeplx = yaml_config["deeplx"];
                if (deeplx["url"]) config.deeplx.url = deeplx["url"].as<std::string>();
                if (deeplx["token"]) config.deeplx.token = deeplx["token"].as<std::string>();
                if (deeplx["target_lang"]) config.deeplx.target_lang = deeplx["target_lang"].as<std::string>();
            }
            
            // Load NLLB config
            if (yaml_config["nllb"]) {
                auto nllb = yaml_config["nllb"];
                // Basic settings
                if (nllb["model_dir"]) config.nllb.model_dir = nllb["model_dir"].as<std::string>();
                if (nllb["target_lang"]) config.nllb.target_lang = nllb["target_lang"].as<std::string>();
                
                // Model files
                if (nllb["model_files"]) {
                    auto files = nllb["model_files"];
                    if (files["encoder"]) config.nllb.model_files.encoder = files["encoder"].as<std::string>();
                    if (files["decoder"]) config.nllb.model_files.decoder = files["decoder"].as<std::string>();
                    if (files["embed_lm_head"]) config.nllb.model_files.embed_lm_head = files["embed_lm_head"].as<std::string>();
                    if (files["cache_initializer"]) config.nllb.model_files.cache_initializer = files["cache_initializer"].as<std::string>();
                    if (files["tokenizer_vocab"]) config.nllb.model_files.tokenizer_vocab = files["tokenizer_vocab"].as<std::string>();
                    if (files["language_config"]) config.nllb.model_files.language_config = files["language_config"].as<std::string>();
                }
                
                // Model parameters
                if (nllb["parameters"]) {
                    auto params = nllb["parameters"];
                    if (params["beam_size"]) config.nllb.params.beam_size = params["beam_size"].as<int>();
                    if (params["max_length"]) config.nllb.params.max_length = params["max_length"].as<int>();
                    if (params["length_penalty"]) config.nllb.params.length_penalty = params["length_penalty"].as<float>();
                    if (params["use_cache"]) config.nllb.params.use_cache = params["use_cache"].as<bool>();
                    if (params["num_threads"]) config.nllb.params.num_threads = params["num_threads"].as<int>();
                }
            }
            
            return config;
        } catch (const YAML::Exception& e) {
            std::cerr << "Error parsing config file: " << e.what() << std::endl;
            return config;
        }
    }
};

} // namespace common

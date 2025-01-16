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
    std::string model_path;
    std::string target_lang = "ZH";
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
                if (nllb["model_path"]) config.nllb.model_path = nllb["model_path"].as<std::string>();
                if (nllb["target_lang"]) config.nllb.target_lang = nllb["target_lang"].as<std::string>();
            }
        }
        catch(const std::exception& e) {
            std::cerr << "Error loading config from file: " << e.what() << std::endl;
        }
        return config;
    }
};

} // namespace common

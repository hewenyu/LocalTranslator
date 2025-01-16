#pragma once

#include <string>

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

    static TranslatorConfig LoadFromFile(const std::string& config_path){

        try{
            // yaml load file
            YAML::Node config = YAML::LoadFile(config_path);

        }
        catch(const std::exception& e){
            std::cerr << "Error loading config from file: " << e.what() << std::endl;
            return TranslatorConfig();
        }
    };
};



} // namespace common

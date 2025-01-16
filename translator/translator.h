#pragma once

#include <memory>
#include <string>
#include <iostream>
#include <yaml-cpp/yaml.h>

namespace translator {

enum class TranslatorType {
    DeepLX,
    NLLB,
    None
};


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
    DeepLXConfig deep_lx;
    NLLBConfig nllb;

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


// Base class for all translators
class ITranslator {
public:
    virtual ~ITranslator() = default;
    virtual std::string translate(const std::string& text, const std::string& source_lang) const = 0;
    // get target language
    virtual std::string get_target_language() const = 0;
};

// Factory function to create translator
std::unique_ptr<ITranslator> CreateTranslator(TranslatorType type, const common::ModelConfig& config);

} // namespace translator


#pragma once

#include <memory>
#include <string>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include "common/common.h"

namespace translator {

enum class TranslatorType {
    DeepLX,
    NLLB,
    None
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
std::unique_ptr<ITranslator> CreateTranslator(TranslatorType type, const common::TranslatorConfig& config);

} // namespace translator


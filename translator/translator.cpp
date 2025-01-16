#include "translator.h"
#include "deeplx/deeplx_translator.h"
#include "nllb-api/nllb_translator.h"

namespace translator {

    // 根据 配置获取类型
TranslatorType GetTranslatorType(const common::TranslatorConfig& config) {
    if (config.type == "DeepLX") {
        return TranslatorType::DeepLX;
    } else if (config.type == "NLLB") {
        return TranslatorType::NLLB;
    } else {
        return TranslatorType::None;
    }
};


std::unique_ptr<ITranslator> CreateTranslator(TranslatorType type, const common::TranslatorConfig& config) {
    switch (type) {
        case TranslatorType::DeepLX:
            return std::make_unique<deeplx::DeepLXTranslator>(config);
        case TranslatorType::NLLB:
            return std::make_unique<nllb::NLLBTranslator>(config);
        case TranslatorType::None:
        default:
            return nullptr;
    }
}

} // namespace translator 
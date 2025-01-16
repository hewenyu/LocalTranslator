#include "translator.h"
#include "deeplx/deeplx_translator.h"

namespace translator {

std::unique_ptr<ITranslator> CreateTranslator(TranslatorType type, const common::TranslatorConfig& config) {
    switch (type) {
        case TranslatorType::DeepLX:
            return std::make_unique<deeplx::DeepLXTranslator>(config);
        case TranslatorType::None:
        default:
            return nullptr;
    }
}

} // namespace translator 
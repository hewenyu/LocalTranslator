#pragma once

namespace translator {

enum class TranslatorError {
    OK = 0,
    ERROR_INIT,
    ERROR_ENCODE,
    ERROR_DECODE,
    ERROR_INVALID_PARAM,
    ERROR_UNSUPPORTED_LANGUAGE,
    ERROR_NETWORK,
    ERROR_UNKNOWN
};

} // namespace translator 
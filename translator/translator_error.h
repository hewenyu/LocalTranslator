#pragma once

namespace translator {

enum class TranslatorError {
    NO_ERROR = 0,
    ERROR_INIT = 1,
    ERROR_ENCODE = 2,
    ERROR_DECODE = 3,
    ERROR_INVALID_PARAM = 4,
    ERROR_TOKENIZE = 5,
    ERROR_UNSUPPORTED_LANGUAGE = 6
};

} // namespace translator 
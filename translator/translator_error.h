#pragma once

namespace translator {

enum class TranslatorError {
    OK = 0,
    ERROR_INIT = 1,
    ERROR_ENCODE = 2,
    ERROR_DECODE = 3,
    ERROR_INVALID_PARAM = 4,
    ERROR_MEMORY = 5,
    ERROR_UNKNOWN = 6
};

} // namespace translator 
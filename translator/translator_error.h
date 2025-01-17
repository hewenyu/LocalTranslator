#pragma once

namespace translator {

enum class TranslatorError {
    OK = 0,                         // No error
    ERROR_INIT = 1,                 // Initialization error
    ERROR_ENCODE = 2,               // Encoding error
    ERROR_DECODE = 3,               // Decoding error
    ERROR_INVALID_PARAM = 4,        // Invalid parameter
    ERROR_TOKENIZE = 5,             // Tokenization error
    ERROR_UNSUPPORTED_LANGUAGE = 6  // Unsupported language
};

} // namespace translator 
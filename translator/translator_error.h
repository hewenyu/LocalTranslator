#pragma once

namespace translator {

enum class TranslatorError {
    OK = 0,                         // No error
    ERROR_INIT = 1,                 // Initialization error
    ERROR_ENCODE = 2,               // Encoding error
    ERROR_DECODE = 3,               // Decoding error
    ERROR_INVALID_PARAM = 4,        // Invalid parameter
    ERROR_TOKENIZE = 5,             // Tokenization error
    ERROR_UNSUPPORTED_LANGUAGE = 6, // Unsupported language
    ERROR_LOADING_MODEL = 7,        // Model loading error
    ERROR_EXECUTING_MODEL = 8,      // Model execution error
    ERROR_MEMORY = 9,              // Memory allocation error
    ERROR_CACHE = 10,              // Cache operation error
    ERROR_BEAM_SEARCH = 11,        // Beam search error
    ERROR_EMPTY_INPUT = 12,        // Empty input error
    ERROR_FILE_NOT_FOUND = 13,     // File not found error
    ERROR_INVALID_MODEL = 14,      // Invalid model error
    ERROR_TRANSLATION = 15,        // Translation error
    ERROR_UNKNOWN = 99             // Unknown error
};

// 错误信息结构体
struct TranslatorErrorInfo {
    TranslatorError error_code;
    std::string error_message;
    std::string error_details;
};

// 错误回调接口
class TranslatorErrorCallback {
public:
    virtual ~TranslatorErrorCallback() = default;
    virtual void onError(const TranslatorErrorInfo& error) = 0;
};

} // namespace translator 
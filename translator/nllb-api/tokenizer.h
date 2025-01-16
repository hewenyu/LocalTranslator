#pragma once

#include <string>
#include <vector>
#include <memory>
#include <array>
#include <sentencepiece_processor.h>

namespace nllb {

class Tokenizer {
public:
    explicit Tokenizer(const std::string& model_path);
    ~Tokenizer() = default;

    struct TokenizerOutput {
        std::vector<int64_t> input_ids;
        std::vector<int64_t> attention_mask;
    };

    // Tokenize text for translation
    TokenizerOutput encode(const std::string& text,
                         const std::string& source_lang,
                         const std::string& target_lang) const;

    // Decode tokens back to text
    std::string decode(const std::vector<int64_t>& tokens) const;

    // Special token IDs
    int64_t pad_id() const { return pad_id_; }
    int64_t eos_id() const { return eos_id_; }
    int64_t bos_id() const { return bos_id_; }
    int64_t unk_id() const { return unk_id_; }

    // Get language ID
    int64_t get_language_id(const std::string& language) const;

private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_;
    
    // Special token IDs
    int64_t pad_id_ = 0;
    int64_t eos_id_ = 2;
    int64_t bos_id_ = 1;
    int64_t unk_id_ = 3;

    // Dictionary length
    static constexpr int DICTIONARY_LENGTH = 256000;

    // NLLB supported languages
    static constexpr size_t NLLB_LANGUAGES_COUNT = 200;  // 实际语言数量
    static const std::array<const char*, NLLB_LANGUAGES_COUNT> NLLB_LANGUAGES;

    // Helper methods
    std::string add_language_tokens(const std::string& text,
                                  const std::string& source_lang,
                                  const std::string& target_lang) const;
};

} // namespace nllb 
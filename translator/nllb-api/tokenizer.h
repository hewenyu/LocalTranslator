#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <sentencepiece_processor.h>

namespace nllb {

struct TokenizerResult {
    std::vector<int64_t> token_ids;
    std::vector<int64_t> attention_mask;
    std::vector<std::string> tokens;
};

struct TokenizerOutput {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::vector<std::string> tokens;
    int64_t pad_token_id;
    int64_t eos_token_id;
    int64_t bos_token_id;
};

class Tokenizer {
public:
    explicit Tokenizer(const std::string& model_path);
    ~Tokenizer() = default;

    // Single text tokenization
    TokenizerResult encode(const std::string& text, const std::string& lang_code) const;
    std::string decode(const std::vector<int64_t>& token_ids) const;

    // Batch tokenization
    std::vector<TokenizerResult> encode_batch(
        const std::vector<std::string>& texts,
        const std::string& lang_code) const;

    // Special token handling
    int64_t get_pad_token_id() const { return pad_token_id_; }
    int64_t get_eos_token_id() const { return eos_token_id_; }
    int64_t get_bos_token_id() const { return bos_token_id_; }
    int64_t get_lang_id(const std::string& lang_code) const;

    // Language code handling
    std::string normalize_language_code(const std::string& lang_code) const;
    bool is_language_supported(const std::string& lang_code) const;

private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_;
    std::unordered_map<std::string, int64_t> lang_token_ids_;
    int64_t pad_token_id_;
    int64_t eos_token_id_;
    int64_t bos_token_id_;

    void initialize_special_tokens();
    void load_language_tokens();
};

} // namespace nllb 
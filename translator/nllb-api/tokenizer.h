#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <sentencepiece_processor.h>

namespace nllb {

// Result of tokenization operation
struct TokenizerResult {
    std::vector<int64_t> input_ids;      // Token IDs
    std::vector<int64_t> attention_mask;  // Attention mask (1 for tokens, 0 for padding)
};

class Tokenizer {
public:
    // Special token constants
    static constexpr const char* PAD_TOKEN = "<pad>";
    static constexpr const char* BOS_TOKEN = "<s>";
    static constexpr const char* EOS_TOKEN = "</s>";
    static constexpr const char* UNK_TOKEN = "<unk>";

    explicit Tokenizer(const std::string& model_path);
    ~Tokenizer() = default;

    // Core tokenization methods
    TokenizerResult encode(const std::string& text, 
                         const std::string& src_lang,
                         const std::string& tgt_lang) const;
    std::string decode(const std::vector<int64_t>& token_ids) const;

    // Batch processing
    std::vector<TokenizerResult> encode_batch(
        const std::vector<std::string>& texts,
        const std::string& src_lang,
        const std::string& tgt_lang) const;

    // Special token access
    int64_t pad_id() const { return pad_token_id_; }
    int64_t bos_id() const { return bos_token_id_; }
    int64_t eos_id() const { return eos_token_id_; }
    int64_t unk_id() const { return unk_token_id_; }

    // Language handling
    std::string normalize_language_code(const std::string& lang_code) const;
    bool is_language_supported(const std::string& lang_code) const;
    int64_t get_lang_id(const std::string& lang_code) const;

    // Vocabulary size
    size_t vocab_size() const;

private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_;
    std::unordered_map<std::string, int64_t> lang_token_ids_;
    
    // Special token IDs
    int64_t pad_token_id_;
    int64_t bos_token_id_;
    int64_t eos_token_id_;
    int64_t unk_token_id_;

    // Initialization helpers
    void initialize_special_tokens();
    void load_language_tokens();
    
    // Helper methods
    std::string add_language_tokens(const std::string& text,
                                  const std::string& src_lang,
                                  const std::string& tgt_lang) const;
};

} // namespace nllb 
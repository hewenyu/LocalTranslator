#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "sentencepiece_processor.h"

namespace nllb {

class TokenizerResult {
public:
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    
    TokenizerResult() = default;
    TokenizerResult(std::vector<int64_t>&& ids, std::vector<int64_t>&& mask)
        : input_ids(std::move(ids)), attention_mask(std::move(mask)) {}
};

// Memory pool for tokenizer results
class TokenizerMemoryPool {
public:
    TokenizerResult acquire() {
        if (pool_.empty()) {
            return TokenizerResult();
        }
        auto result = std::move(pool_.back());
        pool_.pop_back();
        return result;
    }

    void release(TokenizerResult&& result) {
        if (pool_.size() < max_pool_size_) {
            result.input_ids.clear();
            result.attention_mask.clear();
            pool_.push_back(std::move(result));
        }
    }

private:
    static constexpr size_t max_pool_size_ = 32;
    std::vector<TokenizerResult> pool_;
};

class Tokenizer {
public:
    explicit Tokenizer(const std::string& model_path);
    ~Tokenizer() = default;

    // Disable copy
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // Enable move
    Tokenizer(Tokenizer&&) = default;
    Tokenizer& operator=(Tokenizer&&) = default;

    TokenizerResult encode(const std::string& text, 
                         const std::string& source_lang,
                         const std::string& target_lang) const;

    std::string decode(const std::vector<int64_t>& ids) const;
    
    int64_t eos_id() const;
    int64_t pad_id() const;
    int64_t bos_id() const;

private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_processor_;
    mutable TokenizerMemoryPool memory_pool_;

    // Cache for special tokens
    int64_t eos_token_id_{0};
    int64_t pad_token_id_{1};
    int64_t bos_token_id_{2};

    // Language code cache
    mutable std::unordered_map<std::string, std::string> lang_code_cache_;
};

} // namespace nllb 
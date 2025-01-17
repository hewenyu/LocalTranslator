#include "tokenizer.h"
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace nllb {

Tokenizer::Tokenizer(const std::string& model_path) 
    : sp_processor_(std::make_unique<sentencepiece::SentencePieceProcessor>()) {
    
    auto status = sp_processor_->Load(model_path);
    if (!status.ok()) {
        throw std::runtime_error("Failed to load SentencePiece model: " + 
                               std::string(status.message()));
    }

    // Initialize special tokens
    eos_token_id_ = sp_processor_->eos_id();
    pad_token_id_ = sp_processor_->pad_id();
    bos_token_id_ = sp_processor_->bos_id();

    spdlog::info("Tokenizer initialized with model: {}", model_path);
}

TokenizerResult Tokenizer::encode(
    const std::string& text,
    const std::string& source_lang,
    const std::string& target_lang) const {
    
    try {
        // Get a TokenizerResult from pool
        auto result = memory_pool_.acquire();
        
        // Prepare input text with language tokens
        std::string processed_text = source_lang + " " + text + " " + target_lang;
        
        // Encode text
        std::vector<int> tmp_ids;
        auto status = sp_processor_->Encode(processed_text, &tmp_ids);
        if (!status.ok()) {
            throw std::runtime_error("Failed to encode text: " + 
                                   std::string(status.message()));
        }
        
        // Convert to int64_t and prepare attention mask
        result.input_ids.resize(tmp_ids.size());
        result.attention_mask.resize(tmp_ids.size(), 1);
        
        for (size_t i = 0; i < tmp_ids.size(); ++i) {
            result.input_ids[i] = static_cast<int64_t>(tmp_ids[i]);
        }
        
        return result;
    } catch (const std::exception& e) {
        spdlog::error("Error in tokenizer encode: {}", e.what());
        throw;
    }
}

std::string Tokenizer::decode(const std::vector<int64_t>& ids) const {
    try {
        // Convert int64_t to int for SentencePiece
        std::vector<int> tmp_ids;
        tmp_ids.reserve(ids.size());
        
        for (const auto& id : ids) {
            if (id != pad_token_id_) {  // Skip padding tokens
                tmp_ids.push_back(static_cast<int>(id));
            }
        }
        
        std::string result;
        auto status = sp_processor_->Decode(tmp_ids, &result);
        if (!status.ok()) {
            throw std::runtime_error("Failed to decode tokens: " + 
                                   std::string(status.message()));
        }
        
        return result;
    } catch (const std::exception& e) {
        spdlog::error("Error in tokenizer decode: {}", e.what());
        throw;
    }
}

int64_t Tokenizer::eos_id() const { return eos_token_id_; }
int64_t Tokenizer::pad_id() const { return pad_token_id_; }
int64_t Tokenizer::bos_id() const { return bos_token_id_; }

} // namespace nllb 
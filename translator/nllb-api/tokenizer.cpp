#include "tokenizer.h"
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace nllb {

Tokenizer::Tokenizer(const std::string& model_path) 
    : sp_(std::make_unique<sentencepiece::SentencePieceProcessor>()) {
    
    auto status = sp_->Load(model_path);
    if (!status.ok()) {
        throw std::runtime_error("Failed to load tokenizer model: " + status.ToString());
    }
    
    initialize_special_tokens();
    load_language_tokens();
}

void Tokenizer::initialize_special_tokens() {
    pad_token_id_ = sp_->PieceToId(PAD_TOKEN);
    bos_token_id_ = sp_->PieceToId(BOS_TOKEN);
    eos_token_id_ = sp_->PieceToId(EOS_TOKEN);
    unk_token_id_ = sp_->PieceToId(UNK_TOKEN);
}

void Tokenizer::load_language_tokens() {
    // Load language tokens from the vocabulary
    for (int i = 0; i < sp_->GetPieceSize(); ++i) {
        auto piece = sp_->IdToPiece(i);
        if (piece.find("__") == 0) { // Language tokens start with __
            lang_token_ids_[piece.substr(2)] = i; // Remove __ prefix
        }
    }
}

TokenizerResult Tokenizer::encode(
    const std::string& text,
    const std::string& src_lang,
    const std::string& tgt_lang) const {
    
    auto processed_text = add_language_tokens(text, src_lang, tgt_lang);
    std::vector<int> ids;
    auto status = sp_->Encode(processed_text, &ids);
    if (!status.ok()) {
        throw std::runtime_error("Failed to encode text: " + status.ToString());
    }

    TokenizerResult result;
    result.input_ids.assign(ids.begin(), ids.end());
    result.attention_mask.resize(result.input_ids.size(), 1);
    return result;
}

std::string Tokenizer::decode(const std::vector<int64_t>& token_ids) const {
    std::vector<int> ids(token_ids.begin(), token_ids.end());
    std::string result;
    auto status = sp_->Decode(ids, &result);
    if (!status.ok()) {
        throw std::runtime_error("Failed to decode tokens: " + status.ToString());
    }
    return result;
}

std::vector<TokenizerResult> Tokenizer::encode_batch(
    const std::vector<std::string>& texts,
    const std::string& src_lang,
    const std::string& tgt_lang) const {
    
    std::vector<TokenizerResult> results;
    results.reserve(texts.size());
    for (const auto& text : texts) {
        results.push_back(encode(text, src_lang, tgt_lang));
    }
    return results;
}

std::string Tokenizer::normalize_language_code(const std::string& lang_code) const {
    std::string normalized = lang_code;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    normalized.erase(
        std::remove_if(normalized.begin(), normalized.end(), ::isspace),
        normalized.end()
    );
    return normalized;
}

bool Tokenizer::is_language_supported(const std::string& lang_code) const {
    auto normalized = normalize_language_code(lang_code);
    return lang_token_ids_.find(normalized) != lang_token_ids_.end();
}

int64_t Tokenizer::get_lang_id(const std::string& lang_code) const {
    auto normalized = normalize_language_code(lang_code);
    auto it = lang_token_ids_.find(normalized);
    if (it == lang_token_ids_.end()) {
        throw std::runtime_error("Unsupported language code: " + lang_code);
    }
    return it->second;
}

size_t Tokenizer::vocab_size() const {
    return sp_->GetPieceSize();
}

std::string Tokenizer::add_language_tokens(
    const std::string& text,
    const std::string& src_lang,
    const std::string& tgt_lang) const {
    
    auto src_normalized = normalize_language_code(src_lang);
    auto tgt_normalized = normalize_language_code(tgt_lang);
    
    if (!is_language_supported(src_normalized) || !is_language_supported(tgt_normalized)) {
        throw std::runtime_error("Unsupported language code");
    }
    
    return "__" + src_normalized + " " + text + " __" + tgt_normalized;
}

} // namespace nllb 
#include "tokenizer.h"
#include <stdexcept>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace nllb {

Tokenizer::Tokenizer(const std::string& model_path) {
    sp_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    auto status = sp_->Load(model_path);
    if (!status.ok()) {
        throw std::runtime_error("Failed to load tokenizer model: " + status.ToString());
    }
    spdlog::info("Loaded tokenizer model from: {}", model_path);
}

Tokenizer::TokenizerOutput Tokenizer::encode(
    const std::string& text,
    const std::string& source_lang,
    const std::string& target_lang) const {
    
    try {
        // 添加语言标记
        std::string processed_text = add_language_tokens(text, source_lang, target_lang);
        
        // 使用SentencePiece进行分词
        std::vector<int> piece_ids;
        auto status = sp_->Encode(processed_text, &piece_ids);
        if (!status.ok()) {
            throw std::runtime_error("Tokenization failed: " + status.ToString());
        }

        // 转换为int64_t
        std::vector<int64_t> input_ids;
        input_ids.reserve(piece_ids.size());
        for (int id : piece_ids) {
            input_ids.push_back(static_cast<int64_t>(id));
        }

        // 创建attention mask
        std::vector<int64_t> attention_mask(input_ids.size(), 1);

        return {input_ids, attention_mask};
    } catch (const std::exception& e) {
        throw std::runtime_error("Encoding failed: " + std::string(e.what()));
    }
}

std::string Tokenizer::decode(const std::vector<int64_t>& tokens) const {
    try {
        // 转换为int
        std::vector<int> piece_ids;
        piece_ids.reserve(tokens.size());
        for (int64_t id : tokens) {
            piece_ids.push_back(static_cast<int>(id));
        }

        // 使用SentencePiece进行解码
        std::string text;
        auto status = sp_->Decode(piece_ids, &text);
        if (!status.ok()) {
            throw std::runtime_error("Decoding failed: " + status.ToString());
        }

        return text;
    } catch (const std::exception& e) {
        throw std::runtime_error("Decoding failed: " + std::string(e.what()));
    }
}

int64_t Tokenizer::get_language_id(const std::string& language) const {
    // 在NLLB_LANGUAGES中查找语言代码
    auto it = std::find_if(NLLB_LANGUAGES.begin(), NLLB_LANGUAGES.end(),
                          [&language](const char* lang) {
                              return language == lang;
                          });
    
    if (it == NLLB_LANGUAGES.end()) {
        throw std::runtime_error("Unsupported language: " + language);
    }
    
    return static_cast<int64_t>(std::distance(NLLB_LANGUAGES.begin(), it));
}

std::string Tokenizer::add_language_tokens(
    const std::string& text,
    const std::string& source_lang,
    const std::string& target_lang) const {
    
    return "__" + source_lang + "__ " + text + " __" + target_lang + "__";
}

// 定义支持的语言列表
const std::array<const char*, Tokenizer::NLLB_LANGUAGES_COUNT> Tokenizer::NLLB_LANGUAGES = {
    "ace_Arab", "ace_Latn", "acm_Arab", "acq_Arab", "aeb_Arab", "afr_Latn", "ajp_Arab",
    "aka_Latn", "amh_Ethi", "apc_Arab", "arb_Arab", "ars_Arab", "ary_Arab", "arz_Arab",
    // ... 添加更多语言代码
};

} // namespace nllb 
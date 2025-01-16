#include "tokenizer.h"
#include <stdexcept>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <filesystem>
#include <fstream>

namespace nllb {

Tokenizer::Tokenizer(const std::string& model_path) {
    // 检查文件是否存在
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Tokenizer model file not found: " + model_path);
    }

    // 检查文件大小
    auto file_size = std::filesystem::file_size(model_path);
    spdlog::info("Tokenizer model file size: {} bytes", file_size);
    if (file_size == 0) {
        throw std::runtime_error("Tokenizer model file is empty: " + model_path);
    }

    // 尝试打开文件
    std::ifstream file(model_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Failed to open tokenizer model file: " + model_path);
    }
    file.close();

    // 初始化SentencePiece处理器
    sp_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    spdlog::info("Attempting to load tokenizer model from: {}", model_path);
    
    auto status = sp_->Load(model_path);
    if (!status.ok()) {
        spdlog::error("Failed to load tokenizer model: {}", status.ToString());
        spdlog::error("Model path: {}", model_path);
        throw std::runtime_error("Failed to load tokenizer model: " + status.ToString());
    }

    // 验证模型加载后的状态
    if (sp_->GetPieceSize() == 0) {
        throw std::runtime_error("Loaded model has no vocabulary");
    }

    spdlog::info("Successfully loaded tokenizer model from: {}", model_path);
    spdlog::info("Vocabulary size: {}", sp_->GetPieceSize());
}

Tokenizer::TokenizerOutput Tokenizer::encode(
    const std::string& text,
    const std::string& source_lang,
    const std::string& target_lang) const {
    
    try {
        if (text.empty()) {
            throw std::runtime_error("Input text is empty");
        }

        // 添加语言标记
        std::string processed_text = add_language_tokens(text, source_lang, target_lang);
        spdlog::debug("Processed text with language tokens: {}", processed_text);
        
        // 使用SentencePiece进行分词
        std::vector<int> piece_ids;
        auto status = sp_->Encode(processed_text, &piece_ids);
        if (!status.ok()) {
            spdlog::error("Tokenization failed: {}", status.ToString());
            spdlog::error("Input text: {}", text);
            throw std::runtime_error("Tokenization failed: " + status.ToString());
        }

        if (piece_ids.empty()) {
            throw std::runtime_error("Tokenization produced no tokens");
        }

        // 转换为int64_t
        std::vector<int64_t> input_ids;
        input_ids.reserve(piece_ids.size());
        for (int id : piece_ids) {
            input_ids.push_back(static_cast<int64_t>(id));
        }

        // 创建attention mask
        std::vector<int64_t> attention_mask(input_ids.size(), 1);

        spdlog::debug("Encoded {} tokens", input_ids.size());
        return {input_ids, attention_mask};
    } catch (const std::exception& e) {
        spdlog::error("Encoding failed: {}", e.what());
        throw std::runtime_error("Encoding failed: " + std::string(e.what()));
    }
}

std::string Tokenizer::decode(const std::vector<int64_t>& tokens) const {
    try {
        if (tokens.empty()) {
            throw std::runtime_error("Token sequence is empty");
        }

        // 转换为int类型
        std::vector<int> piece_ids;
        piece_ids.reserve(tokens.size());
        for (int64_t id : tokens) {
            piece_ids.push_back(static_cast<int>(id));
        }

        std::string result;
        auto status = sp_->Decode(piece_ids, &result);
        if (!status.ok()) {
            spdlog::error("Decoding failed: {}", status.ToString());
            throw std::runtime_error("Decoding failed: " + status.ToString());
        }

        spdlog::debug("Decoded {} tokens to text", tokens.size());
        return result;
    } catch (const std::exception& e) {
        spdlog::error("Decoding failed: {}", e.what());
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
    "asm_Beng", "ast_Latn", "awa_Deva", "ayr_Latn", "azb_Arab", "azj_Latn", "bak_Cyrl",
    "bam_Latn", "ban_Latn", "bel_Cyrl", "bem_Latn", "ben_Beng", "bho_Deva", "bjn_Arab",
    "bjn_Latn", "bod_Tibt", "bos_Latn", "bug_Latn", "bul_Cyrl", "cat_Latn", "ceb_Latn",
    "ces_Latn", "cjk_Latn", "ckb_Arab", "crh_Latn", "cym_Latn", "dan_Latn", "deu_Latn",
    "dik_Latn", "dyu_Latn", "dzo_Tibt", "ell_Grek", "eng_Latn", "epo_Latn", "est_Latn",
    "eus_Latn", "ewe_Latn", "fao_Latn", "pes_Arab", "fij_Latn", "fin_Latn", "fon_Latn",
    "fra_Latn", "fur_Latn", "fuv_Latn", "gla_Latn", "gle_Latn", "glg_Latn", "grn_Latn",
    "guj_Gujr", "hat_Latn", "hau_Latn", "heb_Hebr", "hin_Deva", "hne_Deva", "hrv_Latn",
    "hun_Latn", "hye_Armn", "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn", "ita_Latn",
    "jav_Latn", "jpn_Jpan", "kab_Latn", "kac_Latn", "kam_Latn", "kan_Knda", "kas_Arab",
    "kas_Deva", "kat_Geor", "knc_Arab", "knc_Latn", "kaz_Cyrl", "kbp_Latn", "kea_Latn",
    "khm_Khmr", "kik_Latn", "kin_Latn", "kir_Cyrl", "kmb_Latn", "kon_Latn", "kor_Hang",
    "kmr_Latn", "lao_Laoo", "lvs_Latn", "lij_Latn", "lim_Latn", "lin_Latn", "lit_Latn",
    "lmo_Latn", "ltg_Latn", "ltz_Latn", "lua_Latn", "lug_Latn", "luo_Latn", "lus_Latn",
    "mag_Deva", "mai_Deva", "mal_Mlym", "mar_Deva", "min_Latn", "mkd_Cyrl", "plt_Latn",
    "mlt_Latn", "mni_Beng", "khk_Cyrl", "mos_Latn", "mri_Latn", "zsm_Latn", "mya_Mymr",
    "nld_Latn", "nno_Latn", "nob_Latn", "npi_Deva", "nso_Latn", "nus_Latn", "nya_Latn",
    "oci_Latn", "gaz_Latn", "ory_Orya", "pag_Latn", "pan_Guru", "pap_Latn", "pol_Latn",
    "por_Latn", "prs_Arab", "pbt_Arab", "quy_Latn", "ron_Latn", "run_Latn", "rus_Cyrl",
    "sag_Latn", "san_Deva", "sat_Beng", "scn_Latn", "shn_Mymr", "sin_Sinh", "slk_Latn",
    "slv_Latn", "smo_Latn", "sna_Latn", "snd_Arab", "som_Latn", "sot_Latn", "spa_Latn",
    "als_Latn", "srd_Latn", "srp_Cyrl", "ssw_Latn", "sun_Latn", "swe_Latn", "swh_Latn",
    "szl_Latn", "tam_Taml", "tat_Cyrl", "tel_Telu", "tgk_Cyrl", "tgl_Latn", "tha_Thai",
    "tir_Ethi", "taq_Latn", "taq_Tfng", "tpi_Latn", "tsn_Latn", "tso_Latn", "tuk_Latn",
    "tum_Latn", "tur_Latn", "twi_Latn", "tzm_Tfng", "uig_Arab", "ukr_Cyrl", "umb_Latn",
    "urd_Arab", "uzn_Latn", "vec_Latn", "vie_Latn", "war_Latn", "wol_Latn", "xho_Latn",
    "ydd_Hebr", "yor_Latn", "yue_Hant", "zho_Hans", "zho_Hant", "zul_Latn"
};

} // namespace nllb 
#include <stdexcept>
#include "tokenizer.h"

namespace nllb {

// Define the static language array
const std::array<const char*, Tokenizer::NLLB_LANGUAGES_COUNT> Tokenizer::NLLB_LANGUAGES = {
    "ace_Arab", "ace_Latn", "acm_Arab", "acq_Arab", "aeb_Arab", "afr_Latn", "ajp_Arab", "aka_Latn", 
    "amh_Ethi", "apc_Arab", "arb_Arab", "ars_Arab", "ary_Arab", "arz_Arab", "asm_Beng", "ast_Latn", 
    "awa_Deva", "ayr_Latn", "azb_Arab", "azj_Latn", "bak_Cyrl", "bam_Latn", "ban_Latn", "bel_Cyrl", 
    "bem_Latn", "ben_Beng", "bho_Deva", "bjn_Arab", "bjn_Latn", "bod_Tibt", "bos_Latn", "bug_Latn", 
    "bul_Cyrl", "cat_Latn", "ceb_Latn", "ces_Latn", "cjk_Latn", "ckb_Arab", "crh_Latn", "cym_Latn", 
    "dan_Latn", "deu_Latn", "dik_Latn", "dyu_Latn", "dzo_Tibt", "ell_Grek", "eng_Latn", "epo_Latn", 
    "est_Latn", "eus_Latn", "ewe_Latn", "fao_Latn", "pes_Arab", "fij_Latn", "fin_Latn", "fon_Latn", 
    "fra_Latn", "fur_Latn", "fuv_Latn", "gla_Latn", "gle_Latn", "glg_Latn", "grn_Latn", "guj_Gujr", 
    "hat_Latn", "hau_Latn", "heb_Hebr", "hin_Deva", "hne_Deva", "hrv_Latn", "hun_Latn", "hye_Armn", 
    "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn", "ita_Latn", "jav_Latn", "jpn_Jpan", "kab_Latn", 
    "kac_Latn", "kam_Latn", "kan_Knda", "kas_Arab", "kas_Deva", "kat_Geor", "knc_Arab", "knc_Latn", 
    "kaz_Cyrl", "kbp_Latn", "kea_Latn", "khm_Khmr", "kik_Latn", "kin_Latn", "kir_Cyrl", "kmb_Latn", 
    "kon_Latn", "kor_Hang", "kmr_Latn", "lao_Laoo", "lvs_Latn", "lij_Latn", "lim_Latn", "lin_Latn", 
    "lit_Latn", "lmo_Latn", "ltg_Latn", "ltz_Latn", "lua_Latn", "lug_Latn", "luo_Latn", "lus_Latn", 
    "mag_Deva", "mai_Deva", "mal_Mlym", "mar_Deva", "min_Latn", "mkd_Cyrl", "plt_Latn", "mlt_Latn", 
    "mni_Beng", "khk_Cyrl", "mos_Latn", "mri_Latn", "zsm_Latn", "mya_Mymr", "nld_Latn", "nno_Latn", 
    "nob_Latn", "npi_Deva", "nso_Latn", "nus_Latn", "nya_Latn", "oci_Latn", "gaz_Latn", "ory_Orya", 
    "pag_Latn", "pan_Guru", "pap_Latn", "pol_Latn", "por_Latn", "prs_Arab", "pbt_Arab", "quy_Latn", 
    "ron_Latn", "run_Latn", "rus_Cyrl", "sag_Latn", "san_Deva", "sat_Beng", "scn_Latn", "shn_Mymr", 
    "sin_Sinh", "slk_Latn", "slv_Latn", "smo_Latn", "sna_Latn", "snd_Arab", "som_Latn", "sot_Latn", 
    "spa_Latn", "als_Latn", "srd_Latn", "srp_Cyrl", "ssw_Latn", "sun_Latn", "swe_Latn", "swh_Latn", 
    "szl_Latn", "tam_Taml", "tat_Cyrl", "tel_Telu", "tgk_Cyrl", "tgl_Latn", "tha_Thai", "tir_Ethi", 
    "taq_Latn", "taq_Tfng", "tpi_Latn", "tsn_Latn", "tso_Latn", "tuk_Latn", "tum_Latn", "tur_Latn", 
    "twi_Latn", "tzm_Tfng", "uig_Arab", "ukr_Cyrl", "umb_Latn", "urd_Arab", "uzn_Latn", "vec_Latn", 
    "vie_Latn", "war_Latn", "wol_Latn", "xho_Latn", "ydd_Hebr", "yor_Latn", "yue_Hant", "zho_Hans", 
    "zho_Hant", "zul_Latn"
};

Tokenizer::Tokenizer(const std::string& model_path) {
    sp_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    const auto status = sp_->Load(model_path);
    if (!status.ok()) {
        throw std::runtime_error("Failed to load SentencePiece model: " + status.ToString());
    }
}

int64_t Tokenizer::get_language_id(const std::string& language) const {
    // Find language in the array
    for (size_t i = 0; i < NLLB_LANGUAGES_COUNT; ++i) {
        if (language == NLLB_LANGUAGES[i]) {
            return DICTIONARY_LENGTH + i + 1;
        }
    }
    throw std::runtime_error("Unsupported language code: " + language);
}

std::string Tokenizer::add_language_tokens(
    const std::string& text,
    const std::string& source_lang,
    const std::string& target_lang) const {
    // NLLB format: "__lang1__ __lang2__ text"
    return "__" + source_lang + "__ __" + target_lang + "__ " + text;
}

Tokenizer::TokenizerOutput Tokenizer::encode(
    const std::string& text,
    const std::string& source_lang,
    const std::string& target_lang) const {
    TokenizerOutput output;
    
    // Add language tokens to the text
    std::string processed_text = add_language_tokens(text, source_lang, target_lang);
    
    // Tokenize the text
    std::vector<int> piece_ids;
    const auto status = sp_->Encode(processed_text, &piece_ids);
    if (!status.ok()) {
        throw std::runtime_error("Failed to encode text: " + status.ToString());
    }
    
    // Convert piece_ids to int64_t and add special tokens
    output.input_ids.push_back(bos_id_);  // Add BOS token
    for (const auto& id : piece_ids) {
        // For NLLB, we need to adjust the token IDs
        // The first 4 IDs (0-3) have special mapping, and others need +1
        int64_t adjusted_id = id;
        if (adjusted_id >= 0 && adjusted_id <= 3) {
            switch (adjusted_id) {
                case 0: adjusted_id = 3; break;  // UNK
                case 1: adjusted_id = 0; break;  // PAD
                case 2: adjusted_id = 2; break;  // EOS
                case 3: adjusted_id = 1; break;  // BOS
            }
        } else {
            adjusted_id += 1;
        }
        output.input_ids.push_back(adjusted_id);
    }
    output.input_ids.push_back(eos_id_);  // Add EOS token
    
    // Create attention mask (1 for all tokens)
    output.attention_mask = std::vector<int64_t>(output.input_ids.size(), 1);
    
    return output;
}

std::string Tokenizer::decode(const std::vector<int64_t>& tokens) const {
    // Convert tokens to pieces, skipping special tokens and adjusting IDs back
    std::vector<int> piece_ids;
    for (const auto& token : tokens) {
        if (token != pad_id_ && token != bos_id_ && token != eos_id_ && token < DICTIONARY_LENGTH) {
            // Adjust the ID back for SentencePiece
            int adjusted_id = token;
            if (adjusted_id > 3) {
                adjusted_id -= 1;
            }
            piece_ids.push_back(adjusted_id);
        }
    }
    
    // Decode the pieces
    std::string text;
    const auto status = sp_->Decode(piece_ids, &text);
    if (!status.ok()) {
        throw std::runtime_error("Failed to decode tokens: " + status.ToString());
    }
    
    // Remove language tokens if present
    size_t text_start = text.find("__ ");
    if (text_start != std::string::npos) {
        text_start = text.find("__ ", text_start + 3);
        if (text_start != std::string::npos) {
            text = text.substr(text_start + 3);
        }
    }
    
    return text;
}

} // namespace nllb 
#pragma once

#include <string>
#include <memory>
#include <curl/curl.h>
#include "translator/translator.h"

namespace nllb {

class NLLBTranslator : public translator::ITranslator {
public:
    explicit NLLBTranslator(const common::TranslatorConfig& config);
    ~NLLBTranslator() override;

    std::string translate(const std::string& text, const std::string& source_lang) const override;
    std::string get_target_language() const override;

private:
    struct HttpResponse {
        int status_code;
        std::string body;
    };

    HttpResponse send_post_request(const std::string& json_data) const;
    std::string make_http_request(const std::string& host, int port, 
                                const std::string& path, const std::string& data) const;
    bool needs_translation(const std::string& source_lang) const;
    
    // NLLB specific language code conversion
    std::string convert_to_nllb_lang_code(const std::string& lang_code) const;

    std::string url_;
    std::string target_lang_;
    std::string host_;
    std::string path_;
    int port_;
    mutable CURL* curl_;
};

} // namespace nllb 
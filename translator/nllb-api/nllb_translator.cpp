#include "nllb_translator.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace nllb {

using json = nlohmann::json;

namespace {
    size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
        userp->append((char*)contents, size * nmemb);
        return size * nmemb;
    }
}

NLLBTranslator::NLLBTranslator(const common::TranslatorConfig& config) {
    url_ = config.url;
    target_lang_ = config.target_lang;
    
    // Parse URL into components
    size_t protocol_end = url_.find("://");
    if (protocol_end != std::string::npos) {
        url_ = url_.substr(protocol_end + 3);
    }
    
    size_t path_start = url_.find('/');
    if (path_start != std::string::npos) {
        host_ = url_.substr(0, path_start);
        path_ = url_.substr(path_start);
    } else {
        host_ = url_;
        path_ = "/";
    }
    
    size_t port_start = host_.find(':');
    if (port_start != std::string::npos) {
        port_ = std::stoi(host_.substr(port_start + 1));
        host_ = host_.substr(0, port_start);
    } else {
        port_ = 80;
    }
    
    curl_ = curl_easy_init();
    if (!curl_) {
        throw std::runtime_error("Failed to initialize CURL");
    }
}

NLLBTranslator::~NLLBTranslator() {
    if (curl_) {
        curl_easy_cleanup(curl_);
    }
}

std::string NLLBTranslator::translate(const std::string& text, const std::string& source_lang) const {
    if (!needs_translation(source_lang)) {
        return text;
    }

    json request_data = {
        {"text", text},
        {"source_lang", convert_to_nllb_lang_code(source_lang)},
        {"target_lang", convert_to_nllb_lang_code(target_lang_)}
    };

    auto response = send_post_request(request_data.dump());
    
    if (response.status_code != 200) {
        throw std::runtime_error("Translation request failed with status code: " + 
                               std::to_string(response.status_code));
    }

    try {
        json response_json = json::parse(response.body);
        return response_json["translation"].get<std::string>();
    } catch (const json::exception& e) {
        throw std::runtime_error("Failed to parse translation response: " + std::string(e.what()));
    }
}

std::string NLLBTranslator::get_target_language() const {
    return target_lang_;
}

bool NLLBTranslator::needs_translation(const std::string& source_lang) const {
    return source_lang != target_lang_;
}

NLLBTranslator::HttpResponse NLLBTranslator::send_post_request(const std::string& json_data) const {
    std::string response_data;
    long response_code = 0;

    curl_easy_reset(curl_);
    
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl_, CURLOPT_URL, (host_ + path_).c_str());
    curl_easy_setopt(curl_, CURLOPT_PORT, port_);
    curl_easy_setopt(curl_, CURLOPT_POST, 1L);
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, json_data.c_str());
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response_data);
    
    CURLcode res = curl_easy_perform(curl_);
    curl_slist_free_all(headers);
    
    if (res != CURLE_OK) {
        throw std::runtime_error(std::string("CURL request failed: ") + curl_easy_strerror(res));
    }
    
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &response_code);
    
    return HttpResponse{static_cast<int>(response_code), response_data};
}

std::string NLLBTranslator::convert_to_nllb_lang_code(const std::string& lang_code) const {
    // NLLB uses specific language codes, here's a basic mapping
    // You may need to expand this based on NLLB's supported languages
    static const std::unordered_map<std::string, std::string> lang_map = {
        {"en", "eng_Latn"},
        {"zh", "zho_Hans"},
        {"fr", "fra_Latn"},
        {"de", "deu_Latn"},
        {"es", "spa_Latn"},
        {"ru", "rus_Cyrl"},
        {"ja", "jpn_Jpan"},
        {"ko", "kor_Hang"}
        // Add more mappings as needed
    };

    auto it = lang_map.find(lang_code);
    if (it != lang_map.end()) {
        return it->second;
    }
    
    // If no mapping found, return original code
    // You might want to throw an exception instead
    return lang_code;
}

} // namespace nllb 
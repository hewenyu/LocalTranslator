#include <regex>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "translator/deeplx/deeplx_translator.h"

using json = nlohmann::json;

namespace {

size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

} // namespace

namespace deeplx {

DeepLXTranslator::DeepLXTranslator(const common::TranslatorConfig& config) {
    url_ = config.deeplx.url;   
    token_ = config.deeplx.token;
    target_lang_ = config.deeplx.target_lang;
    // Parse URL to get host, port, and path
    std::regex url_regex("^(https?://)?([^/:]+)(?::(\\d+))?(/.*)?$");
    std::smatch matches;
    if (std::regex_match(url_, matches, url_regex)) {
        host_ = matches[2].str();
        port_ = matches[3].length() > 0 ? std::stoi(matches[3].str()) : 80;
        path_ = matches[4].length() > 0 ? matches[4].str() : "/";
    } else {
        throw std::runtime_error("Invalid URL format");
    }

    curl_ = curl_easy_init();
    if (!curl_) {
        throw std::runtime_error("Failed to initialize CURL");
    }
}

DeepLXTranslator::~DeepLXTranslator() {
    if (curl_) {
        curl_easy_cleanup(curl_);
    }
}

bool DeepLXTranslator::needs_translation(const std::string& source_lang) const {
    std::string target_upper = target_lang_;
    std::string source_upper = source_lang;
    std::transform(target_upper.begin(), target_upper.end(), target_upper.begin(), ::toupper);
    std::transform(source_upper.begin(), source_upper.end(), source_upper.begin(), ::toupper);
    return source_upper != target_upper;
}

// get target language
std::string DeepLXTranslator::get_target_language() const {
    return target_lang_;
}


std::string DeepLXTranslator::make_http_request(const std::string& host, int port,
                                              const std::string& path, const std::string& data) const {
    std::string response;
    std::string url = "http://" + host + ":" + std::to_string(port) + path;

    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_, CURLOPT_POST, 1L);
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, data.c_str());
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    if (!token_.empty()) {
        std::string auth_header = "Authorization: Bearer " + token_;
        headers = curl_slist_append(headers, auth_header.c_str());
    }
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);

    CURLcode res = curl_easy_perform(curl_);
    curl_slist_free_all(headers);

    if (res != CURLE_OK) {
        throw std::runtime_error(std::string("CURL request failed: ") + curl_easy_strerror(res));
    }

    return response;
}

DeepLXTranslator::HttpResponse DeepLXTranslator::send_post_request(const std::string& json_data) const {
    std::string response_str = make_http_request(host_, port_, path_, json_data);
    
    HttpResponse response;
    try {
        json responseJson = json::parse(response_str);
        response.status_code = responseJson["code"].get<int>();
        response.body = response_str;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to parse response: ") + e.what());
    }
    
    return response;
}

std::string DeepLXTranslator::translate(const std::string& text, const std::string& source_lang) const {
    if (!needs_translation(source_lang)) {
        return text;
    }

    json requestJson = {
        {"text", text},
        {"source_lang", to_upper(source_lang)},
        {"target_lang", to_upper(target_lang_)}
    };

    std::string jsonStr = requestJson.dump();

    try {
        auto response = send_post_request(jsonStr);
        json responseJson = json::parse(response.body);

        if (responseJson["code"].get<int>() != 200) {
            throw std::runtime_error("Translation API returned error code: " +
                                   std::to_string(responseJson["code"].get<int>()));
        }

        return responseJson["data"].get<std::string>();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Translation failed: ") + e.what());
    }
}

} // namespace deeplx 
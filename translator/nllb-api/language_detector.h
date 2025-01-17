#pragma once

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace nllb {

class LanguageDetector {
public:
    explicit LanguageDetector(const std::string& model_path);
    ~LanguageDetector();

    // 检测文本语言
    std::string detect_language(const std::string& text) const;
    
    // 获取检测置信度
    float get_confidence() const;
    
    // 获取支持的语言列表
    std::vector<std::string> get_supported_languages() const;

private:
    Ort::Env ort_env_;
    std::unique_ptr<Ort::Session> session_;
    float confidence_threshold_;
    std::vector<std::string> supported_languages_;
    
    // 预处理文本
    std::vector<float> preprocess_text(const std::string& text) const;
    
    // 后处理模型输出
    std::string postprocess_output(const std::vector<float>& output) const;
};

} // namespace nllb 
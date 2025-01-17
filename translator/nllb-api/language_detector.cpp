#include "language_detector.h"
#include <stdexcept>
#include <algorithm>
#include <spdlog/spdlog.h>

namespace nllb {

LanguageDetector::LanguageDetector(const std::string& model_path)
    : ort_env_(ORT_LOGGING_LEVEL_WARNING, "language_detector"),
      confidence_threshold_(0.8f) {
    try {
        // 初始化 ONNX Runtime session
        Ort::SessionOptions session_opts;
        session_opts.SetIntraOpNumThreads(1);
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        
        session_ = std::make_unique<Ort::Session>(ort_env_, 
            std::wstring(model_path.begin(), model_path.end()).c_str(), 
            session_opts);
            
        spdlog::info("Language detector initialized with model: {}", model_path);
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("Failed to initialize language detector: " + std::string(e.what()));
    }
}

LanguageDetector::~LanguageDetector() = default;

std::string LanguageDetector::detect_language(const std::string& text) const {
    try {
        // 预处理文本
        auto input_tensor = preprocess_text(text);
        
        // 准备输入
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(input_tensor.size())};
        
        auto input_tensor_value = Ort::Value::CreateTensor<float>(memory_info,
            input_tensor.data(), input_tensor.size(),
            input_shape.data(), input_shape.size());
            
        // 运行推理
        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};
        
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor_value,
            1,
            output_names,
            1
        );
        
        // 获取输出
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<float> output(output_data, 
            output_data + output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount());
            
        // 后处理输出
        return postprocess_output(output);
    } catch (const std::exception& e) {
        spdlog::error("Language detection failed: {}", e.what());
        throw;
    }
}

float LanguageDetector::get_confidence() const {
    return confidence_threshold_;
}

std::vector<std::string> LanguageDetector::get_supported_languages() const {
    return supported_languages_;
}

std::vector<float> LanguageDetector::preprocess_text(const std::string& text) const {
    // 简单的字符级特征提取
    std::vector<float> features;
    features.reserve(text.length());
    
    for (char c : text) {
        features.push_back(static_cast<float>(c) / 255.0f);
    }
    
    return features;
}

std::string LanguageDetector::postprocess_output(const std::vector<float>& output) const {
    // 找到最大概率的语言
    auto max_it = std::max_element(output.begin(), output.end());
    size_t max_idx = std::distance(output.begin(), max_it);
    
    if (*max_it < confidence_threshold_) {
        throw std::runtime_error("Language detection confidence too low");
    }
    
    return supported_languages_[max_idx];
}

} // namespace nllb 
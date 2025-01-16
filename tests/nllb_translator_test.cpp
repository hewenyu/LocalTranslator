#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <filesystem>
#include "translator/nllb-api/nllb_translator.h"

namespace {
using namespace testing;
namespace fs = std::filesystem;

class NLLBTranslatorTest : public Test {
protected:
    void SetUp() override {
        // 设置测试配置
        common::TranslatorConfig config;
        config.nllb.model_dir = "test_models/nllb";
        config.nllb.target_lang = "eng_Latn";
        config.nllb.params.beam_size = 5;
        config.nllb.params.max_length = 128;
        config.nllb.params.length_penalty = 1.0f;
        config.nllb.params.temperature = 1.0f;
        config.nllb.params.num_threads = 4;
        config.nllb.params.use_cache = true;
        config.nllb.model_files.tokenizer_vocab = "tokenizer.model";
        
        // 确保测试目录存在
        fs::create_directories(config.nllb.model_dir);
        
        // 创建必要的配置文件
        create_test_configs(config.nllb.model_dir);
        
        translator_ = std::make_unique<nllb::NLLBTranslator>(config);
    }
    
    void TearDown() override {
        // 清理测试文件
        if (fs::exists("test_models")) {
            fs::remove_all("test_models");
        }
    }
    
    void create_test_configs(const std::string& model_dir) {
        // 创建model_config.yaml
        std::ofstream model_config(model_dir + "/model_config.yaml");
        model_config << R"(
hidden_size: 1024
num_heads: 16
vocab_size: 256200
max_position_embeddings: 1024
encoder_layers: 24
decoder_layers: 24
)";
        model_config.close();
        
        // 创建language_codes.yaml
        std::ofstream lang_config(model_dir + "/nllb_languages.yaml");
        lang_config << R"(
languages:
  - code: "eng"
    code_NLLB: "eng_Latn"
  - code: "fra"
    code_NLLB: "fra_Latn"
  - code: "cmn"
    code_NLLB: "zho_Hans"
)";
        lang_config.close();
        
        // 创建空的模型文件
        std::ofstream(model_dir + "/NLLB_encoder.onnx").close();
        std::ofstream(model_dir + "/NLLB_decoder.onnx").close();
        std::ofstream(model_dir + "/NLLB_embed_and_lm_head.onnx").close();
        std::ofstream(model_dir + "/NLLB_cache_initializer.onnx").close();
        std::ofstream(model_dir + "/tokenizer.model").close();
    }

    std::unique_ptr<nllb::NLLBTranslator> translator_;
};

// 测试初始化
TEST_F(NLLBTranslatorTest, InitializationSucceeds) {
    EXPECT_THAT(translator_->get_target_language(), Eq("eng_Latn"));
}

// 测试语言代码转换
TEST_F(NLLBTranslatorTest, LanguageCodeConversion) {
    EXPECT_THROW(translator_->translate("Hello", "invalid_lang"), std::runtime_error);
    EXPECT_NO_THROW(translator_->translate("Hello", "fra"));
}

// 测试配置验证
TEST_F(NLLBTranslatorTest, ConfigValidation) {
    common::TranslatorConfig invalid_config;
    invalid_config.nllb.model_dir = "nonexistent_dir";
    EXPECT_THROW(nllb::NLLBTranslator(invalid_config), std::runtime_error);
}

// 测试翻译功能
TEST_F(NLLBTranslatorTest, Translation) {
    // 由于我们使用了空的模型文件，这里只测试API调用是否正确
    EXPECT_THROW(translator_->translate("Bonjour", "fra"), std::runtime_error);
}

// 测试Beam Search配置
TEST_F(NLLBTranslatorTest, BeamSearchConfig) {
    common::TranslatorConfig config;
    config.nllb.model_dir = "test_models/nllb";
    config.nllb.target_lang = "eng_Latn";
    
    // 测试无效的beam size
    config.nllb.params.beam_size = 0;
    EXPECT_THROW(nllb::NLLBTranslator(config), std::invalid_argument);
    
    // 测试无效的temperature
    config.nllb.params.beam_size = 5;
    config.nllb.params.temperature = -1.0f;
    EXPECT_THROW(nllb::NLLBTranslator(config), std::invalid_argument);
}

// 测试缓存状态
TEST_F(NLLBTranslatorTest, CacheState) {
    common::TranslatorConfig config;
    config.nllb.model_dir = "test_models/nllb";
    config.nllb.target_lang = "eng_Latn";
    
    // 测试禁用缓存
    config.nllb.params.use_cache = false;
    auto translator_no_cache = std::make_unique<nllb::NLLBTranslator>(config);
    EXPECT_NO_THROW(translator_no_cache->translate("Hello", "eng"));
}

// 测试性能监控
TEST_F(NLLBTranslatorTest, PerformanceMonitoring) {
    // 这里我们只能测试API是否正确，因为没有真实的模型
    auto start = std::chrono::high_resolution_clock::now();
    EXPECT_THROW(translator_->translate("Hello", "eng"), std::runtime_error);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    EXPECT_GT(duration.count(), 0);
}

} // namespace 
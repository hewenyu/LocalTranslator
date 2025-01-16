#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "translator/nllb-api/nllb_translator.h"

namespace local_translator {
namespace {

class NLLBTranslatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建必要的目录
        std::filesystem::create_directories("models");
        
        // 创建配置文件
        std::ofstream config_file("models/model_config.yaml");
        config_file << "hidden_size: 1024\n"
                   << "num_heads: 16\n"
                   << "vocab_size: 256200\n"
                   << "max_position_embeddings: 1024\n"
                   << "encoder_layers: 24\n"
                   << "decoder_layers: 24\n";
        config_file.close();
    }

    void TearDown() override {
        // 清理测试文件
        std::filesystem::remove("models/model_config.yaml");
    }

    // 辅助函数：检查翻译器初始化
    void CheckTranslatorInit(const std::string& target_lang, bool should_succeed) {
        NLLBTranslator translator;
        auto status = translator.Init(target_lang);
        if (should_succeed) {
            EXPECT_TRUE(status.ok()) << status.message();
        } else {
            EXPECT_FALSE(status.ok());
        }
    }
};

// 测试初始化
TEST_F(NLLBTranslatorTest, InitializationTest) {
    // 测试有效的目标语言
    CheckTranslatorInit("eng_Latn", true);
    CheckTranslatorInit("zho_Hans", true);
    
    // 测试无效的目标语言
    CheckTranslatorInit("invalid_lang", false);
    CheckTranslatorInit("", false);
}

// 测试语言代码转换
TEST_F(NLLBTranslatorTest, LanguageCodeConversionTest) {
    NLLBTranslator translator;
    
    // 测试有效的语言代码转换
    EXPECT_EQ(translator.ConvertLanguageCode("zh"), "zho_Hans");
    EXPECT_EQ(translator.ConvertLanguageCode("en"), "eng_Latn");
    
    // 测试无效的语言代码
    EXPECT_EQ(translator.ConvertLanguageCode("invalid"), "");
    EXPECT_EQ(translator.ConvertLanguageCode(""), "");
}

// 测试配置验证
TEST_F(NLLBTranslatorTest, ConfigValidationTest) {
    NLLBTranslator translator;
    
    // 测试无效的 beam size
    EXPECT_FALSE(translator.SetBeamSize(0).ok());
    EXPECT_FALSE(translator.SetBeamSize(-1).ok());
    EXPECT_TRUE(translator.SetBeamSize(5).ok());
    
    // 测试无效的温度参数
    EXPECT_FALSE(translator.SetTemperature(-0.1).ok());
    EXPECT_FALSE(translator.SetTemperature(0).ok());
    EXPECT_TRUE(translator.SetTemperature(0.8).ok());
}

// 测试翻译 API
TEST_F(NLLBTranslatorTest, TranslationTest) {
    NLLBTranslator translator;
    ASSERT_TRUE(translator.Init("eng_Latn").ok());
    
    // 测试空输入
    std::string result;
    auto status = translator.Translate("", "zho_Hans", result);
    EXPECT_FALSE(status.ok());
    EXPECT_TRUE(result.empty());
    
    // 测试无效的源语言
    status = translator.Translate("测试文本", "invalid_lang", result);
    EXPECT_FALSE(status.ok());
    
    // 测试实际翻译会抛出异常（因为没有实际的模型文件）
    try {
        status = translator.Translate("测试文本", "zho_Hans", result);
        EXPECT_FALSE(status.ok());
    } catch (const std::exception& e) {
        EXPECT_TRUE(std::string(e.what()).find("model") != std::string::npos);
    }
}

// 测试缓存状态
TEST_F(NLLBTranslatorTest, CacheStateTest) {
    NLLBTranslator translator;
    ASSERT_TRUE(translator.Init("eng_Latn").ok());
    
    // 测试禁用缓存
    EXPECT_TRUE(translator.DisableCache().ok());
    
    std::string result;
    auto status = translator.Translate("test", "zho_Hans", result);
    EXPECT_FALSE(status.ok()); // 应该失败，因为没有实际的模型文件
}

// 测试性能监控
TEST_F(NLLBTranslatorTest, PerformanceTest) {
    NLLBTranslator translator;
    ASSERT_TRUE(translator.Init("eng_Latn").ok());
    
    // 记录翻译尝试的时间
    auto start = std::chrono::high_resolution_clock::now();
    std::string result;
    translator.Translate("test", "zho_Hans", result);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    EXPECT_GT(duration.count(), 0);
}

} // namespace
} // namespace local_translator 
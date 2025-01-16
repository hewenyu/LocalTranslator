#include <gtest/gtest.h>
#include "../translator/deeplx/deeplx_translator.h"
#include "../common/common.h"

using namespace deeplx;

class DeepLXTranslatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置测试环境
        common::TranslatorConfig config;
        config.type = "DeepLX";
        config.deeplx.url = "http://localhost:1188";
        config.deeplx.target_lang = "zh";
        
        translator_ = std::make_unique<DeepLXTranslator>(config);
    }

    void TearDown() override {
        // 清理测试环境
        translator_.reset();
    }

    std::unique_ptr<DeepLXTranslator> translator_;
};

// 测试翻译器初始化
TEST_F(DeepLXTranslatorTest, Initialization) {
    ASSERT_NE(translator_, nullptr);
    EXPECT_EQ(translator_->get_target_language(), "zh");
}

// 测试相同语言翻译
TEST_F(DeepLXTranslatorTest, SameLanguageTranslation) {
    const std::string input = "你好，世界！";
    const std::string source_lang = "zh";
    
    std::string result = translator_->translate(input, source_lang);
    EXPECT_EQ(result, input); // 相同语言应该直接返回原文
}

// 测试英文到中文翻译
TEST_F(DeepLXTranslatorTest, EnglishToChineseTranslation) {
    const std::string input = "Hello, World!";
    const std::string source_lang = "en";
    
    std::string result = translator_->translate(input, source_lang);
    EXPECT_FALSE(result.empty());
    EXPECT_NE(result, input);
}

// 测试错误处理
TEST_F(DeepLXTranslatorTest, ErrorHandling) {
    // 测试空字符串
    EXPECT_TRUE(translator_->translate("", "en").empty());
    
    // 测试无效的源语言
    EXPECT_THROW(translator_->translate("test", "invalid_lang"), std::invalid_argument);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 
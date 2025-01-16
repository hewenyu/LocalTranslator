#include <gtest/gtest.h>
#include "../translator/translator.h"
#include "../common/common.h"

using namespace translator;

class TranslatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置测试环境
        common::TranslatorConfig config;
        config.type = "DeepLX";
        config.deeplx.url = "http://localhost:1188/translate";
        config.deeplx.target_lang = "zh";
        
        // 创建翻译器实例
        translator_ = CreateTranslator(TranslatorType::DeepLX, config);
    }

    void TearDown() override {
        // 清理测试环境
        translator_.reset();
    }

    std::unique_ptr<ITranslator> translator_;
};

// 测试翻译器创建
TEST_F(TranslatorTest, CreateTranslator) {
    ASSERT_NE(translator_, nullptr);
}

// 测试目标语言获取
TEST_F(TranslatorTest, GetTargetLanguage) {
    EXPECT_EQ(translator_->get_target_language(), "zh");
}

// 测试基本翻译功能
TEST_F(TranslatorTest, BasicTranslation) {
    const std::string input = "Hello, World!";
    const std::string source_lang = "en";
    
    std::string result = translator_->translate(input, source_lang);
    EXPECT_FALSE(result.empty());
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 
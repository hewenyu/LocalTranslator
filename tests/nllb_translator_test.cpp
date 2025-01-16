#include <gtest/gtest.h>
#include "translator/nllb-api/nllb_translator.h"
#include "common/common.h"

class NLLBTranslatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        config.type = "NLLB";
        config.url = "http://localhost:8080";
        config.target_lang = "zh";
    }

    common::TranslatorConfig config;
};

TEST_F(NLLBTranslatorTest, CreateTranslator) {
    EXPECT_NO_THROW({
        nllb::NLLBTranslator translator(config);
    });
}

TEST_F(NLLBTranslatorTest, GetTargetLanguage) {
    nllb::NLLBTranslator translator(config);
    EXPECT_EQ(translator.get_target_language(), "zh");
}

TEST_F(NLLBTranslatorTest, TranslateToSameLanguage) {
    nllb::NLLBTranslator translator(config);
    std::string text = "测试文本";
    EXPECT_EQ(translator.translate(text, "zh"), text);
}

// Note: This test requires a running NLLB server
TEST_F(NLLBTranslatorTest, TranslateFromEnglish) {
    nllb::NLLBTranslator translator(config);
    std::string text = "Hello, world!";
    EXPECT_NO_THROW({
        std::string translation = translator.translate(text, "en");
        EXPECT_FALSE(translation.empty());
    });
}

TEST_F(NLLBTranslatorTest, InvalidServerUrl) {
    config.url = "http://invalid-server:8080";
    nllb::NLLBTranslator translator(config);
    EXPECT_THROW({
        translator.translate("Hello", "en");
    }, std::runtime_error);
} 
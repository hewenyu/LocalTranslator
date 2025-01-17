#include <gtest/gtest.h>
#include "translator/nllb-api/nllb_translator.h"
#include "common/common.h"

using namespace nllb;

class NLLBTranslatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        try {
            // 配置翻译器
            common::TranslatorConfig config;
            config.nllb.model_dir = "models/nllb";
            config.nllb.target_lang = "ZH";
            config.nllb.params.beam_size = 5;
            config.nllb.params.length_penalty = 1.0f;
            config.nllb.params.num_threads = 4;
            
            translator = std::make_unique<NLLBTranslator>(config);
            ASSERT_TRUE(translator != nullptr);
            
        } catch (const std::exception& e) {
            FAIL() << "Failed to setup test: " << e.what();
        }
    }

    void TearDown() override {
        translator.reset();
    }

    std::unique_ptr<NLLBTranslator> translator;
};

// 测试基本翻译功能
TEST_F(NLLBTranslatorTest, BasicTranslation) {
    const std::string input = "Hello, how are you?";
    const std::string source_lang = "en";
    
    std::string result = translator->translate(input, source_lang);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
}

// 测试中英翻译
TEST_F(NLLBTranslatorTest, ChineseToEnglish) {
    const std::string input = "你好，最近过得怎么样？";
    const std::string source_lang = "zh";
    
    std::string result = translator->translate(input, source_lang);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
}

// 测试语言代码转换
TEST_F(NLLBTranslatorTest, LanguageCodeConversion) {
    // 测试标准语言代码
    EXPECT_EQ(translator->get_nllb_language_code("en"), "eng_Latn");
    EXPECT_EQ(translator->get_nllb_language_code("zh"), "zho_Hans");
    
    // 测试显示语言代码
    EXPECT_EQ(translator->get_display_language_code("eng_Latn"), "en");
    EXPECT_EQ(translator->get_display_language_code("zho_Hans"), "zh");
}

// 测试低质量语言支持
TEST_F(NLLBTranslatorTest, LowQualityLanguageSupport) {
    // 默认不支持低质量语言
    EXPECT_FALSE(translator->get_support_low_quality_languages());
    
    // 启用低质量语言支持
    translator->set_support_low_quality_languages(true);
    EXPECT_TRUE(translator->get_support_low_quality_languages());
    
    // 验证支持的语言列表已更新
    auto languages = translator->get_supported_languages();
    EXPECT_FALSE(languages.empty());
}

// 测试翻译参数
TEST_F(NLLBTranslatorTest, TranslationParameters) {
    // 测试 beam size
    translator->set_beam_size(3);
    
    // 测试最大长度
    translator->set_max_length(64);
    
    // 测试长度惩罚
    translator->set_length_penalty(0.8f);
    
    // 测试温度
    translator->set_temperature(0.7f);
    
    // 测试 top_k
    translator->set_top_k(5);
    
    // 测试 top_p
    translator->set_top_p(0.95f);
    
    // 测试重复惩罚
    translator->set_repetition_penalty(0.85f);
    
    // 验证错误状态
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
}

// 测试错误处理
TEST_F(NLLBTranslatorTest, ErrorHandling) {
    // 测试无效的语言代码
    std::string result = translator->translate("Hello", "invalid_lang");
    EXPECT_TRUE(result.empty());
    EXPECT_NE(translator->get_last_error(), TranslatorError::OK);
    
    // 测试空输入
    result = translator->translate("", "en");
    EXPECT_TRUE(result.empty());
}

// 测试长文本翻译
TEST_F(NLLBTranslatorTest, LongTextTranslation) {
    std::string long_text(1000, 'a');  // 创建1000字符的字符串
    std::string result = translator->translate(long_text, "en");
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
}

// 测试批量翻译
TEST_F(NLLBTranslatorTest, BatchTranslation) {
    std::vector<std::string> texts = {
        "Hello, how are you?",
        "Nice to meet you",
        "Have a great day"
    };
    
    auto results = translator->translate_batch(texts, "en");
    EXPECT_EQ(results.size(), texts.size());
    for (const auto& result : results) {
        EXPECT_FALSE(result.empty());
    }
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
}

// 测试特殊字符
TEST_F(NLLBTranslatorTest, SpecialCharacters) {
    const std::string input = "Hello! @#$%^&*()_+ 你好";
    std::string result = translator->translate(input, "en");
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 
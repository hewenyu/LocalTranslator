#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "translator/nllb-api/nllb_translator.h"
#include "common/common.h"

using namespace nllb;

class NLLBTranslatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 配置spdlog
        auto console = spdlog::stdout_color_mt("console");
        spdlog::set_default_logger(console);
        spdlog::set_level(spdlog::level::trace);
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");
        
        spdlog::info("Starting NLLB translator test setup...");
        
        try {
            // 配置翻译器
            common::TranslatorConfig config;
            config.nllb.model_dir = "models";  // 修改为正确的模型目录路径
            config.nllb.target_lang = "eng_Latn";   // 默认目标语言为英语
            config.nllb.params.beam_size = 5;
            config.nllb.params.max_length = 128;
            config.nllb.params.length_penalty = 1.0f;
            config.nllb.params.temperature = 1.0f;
            config.nllb.params.num_threads = 4;
            config.nllb.model_files.tokenizer_vocab = "sentencepiece_bpe.model";
            
            spdlog::info("Creating translator instance...");
            translator = std::make_unique<NLLBTranslator>(config);
            spdlog::info("Translator instance created successfully");
        } catch (const std::exception& e) {
            spdlog::error("Failed to setup test: {}", e.what());
            throw;
        }
    }

    void TearDown() override {
        spdlog::info("Cleaning up translator...");
        translator.reset();
        spdlog::info("Cleanup completed");
    }

    std::unique_ptr<NLLBTranslator> translator;
};

// 测试基本翻译功能
TEST_F(NLLBTranslatorTest, BasicTranslation) {
    spdlog::info("Running BasicTranslation test...");
    const std::string input = "Hello, how are you?";
    const std::string source_lang = "eng_Latn";
    
    std::string result = translator->translate(input, source_lang);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
    spdlog::info("Translation result: '{}'", result);
}

// 测试中文到英文的翻译
TEST_F(NLLBTranslatorTest, ChineseToEnglish) {
    const std::string input = "你好，最近过得怎么样？";
    const std::string source_lang = "zho_Hans";
    
    std::string result = translator->translate(input, source_lang);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
    spdlog::info("Chinese to English: {}", result);
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
    
    // 验证支持的语言列表是否更新
    auto languages = translator->get_supported_languages();
    EXPECT_FALSE(languages.empty());
}

// 测试翻译参数配置
TEST_F(NLLBTranslatorTest, TranslationParameters) {
    // 测试 beam size
    translator->set_beam_size(3);
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
    translator->set_beam_size(-1);
    EXPECT_EQ(translator->get_last_error(), TranslatorError::ERROR_INVALID_PARAM);
    
    // 测试 max length
    translator->set_max_length(200);
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
    translator->set_max_length(0);
    EXPECT_EQ(translator->get_last_error(), TranslatorError::ERROR_INVALID_PARAM);
    
    // 测试 temperature
    translator->set_temperature(0.8f);
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
    translator->set_temperature(-1.0f);
    EXPECT_EQ(translator->get_last_error(), TranslatorError::ERROR_INVALID_PARAM);
    
    // 测试 top_p
    translator->set_top_p(0.9f);
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
    translator->set_top_p(1.5f);
    EXPECT_EQ(translator->get_last_error(), TranslatorError::ERROR_INVALID_PARAM);
    
    // 测试 repetition penalty
    translator->set_repetition_penalty(1.2f);
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
    translator->set_repetition_penalty(-0.5f);
    EXPECT_EQ(translator->get_last_error(), TranslatorError::ERROR_INVALID_PARAM);
}

// 测试错误处理
TEST_F(NLLBTranslatorTest, ErrorHandling) {
    // 测试空输入
    std::string result = translator->translate("", "eng_Latn");
    EXPECT_TRUE(result.empty());
    
    // 测试无效的源语言
    result = translator->translate("Hello", "invalid_lang");
    EXPECT_TRUE(result.empty());
    EXPECT_EQ(translator->get_last_error(), TranslatorError::ERROR_INVALID_PARAM);
    
    // 测试未初始化的翻译器
    translator.reset();
    result = translator->translate("Hello", "eng_Latn");
    EXPECT_TRUE(result.empty());
    EXPECT_EQ(translator->get_last_error(), TranslatorError::ERROR_INIT);
}

// 测试长文本翻译
TEST_F(NLLBTranslatorTest, LongTextTranslation) {
    const std::string input = "This is a long text that needs to be translated. "
                             "It contains multiple sentences and should test the model's "
                             "ability to handle longer sequences. The translation should "
                             "maintain coherence across sentences and properly handle "
                             "context throughout the text.";
    const std::string source_lang = "eng_Latn";
    
    std::string result = translator->translate(input, source_lang);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
    spdlog::info("Long text translation result: {}", result);
}

// 测试批量翻译
TEST_F(NLLBTranslatorTest, BatchTranslation) {
    std::vector<std::string> inputs = {
        "Hello",
        "How are you?",
        "Nice to meet you"
    };
    const std::string source_lang = "eng_Latn";
    
    auto results = translator->translate_batch(inputs, source_lang);
    EXPECT_EQ(results.size(), inputs.size());
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
    for (size_t i = 0; i < results.size(); ++i) {
        EXPECT_FALSE(results[i].empty());
        spdlog::info("Batch translation result {}: {}", i, results[i]);
    }
}

// 测试特殊字符处理
TEST_F(NLLBTranslatorTest, SpecialCharacters) {
    const std::string input = "Hello! @#$%^&*()_+ 你好！";
    const std::string source_lang = "eng_Latn";
    
    std::string result = translator->translate(input, source_lang);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
    spdlog::info("Special characters translation: {}", result);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 
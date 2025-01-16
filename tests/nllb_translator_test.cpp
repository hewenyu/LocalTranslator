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
    
    try {
        spdlog::info("Attempting to translate: '{}'", input);
        std::string result = translator->translate(input, source_lang);
        EXPECT_FALSE(result.empty());
        spdlog::info("Translation result: '{}'", result);
    } catch (const std::exception& e) {
        spdlog::error("Translation failed: {}", e.what());
        FAIL() << "Translation failed: " << e.what();
    }
}

// 测试中文到英文的翻译
TEST_F(NLLBTranslatorTest, ChineseToEnglish) {
    const std::string input = "你好，最近过得怎么样？";
    const std::string source_lang = "zho_Hans";
    
    try {
        std::string result = translator->translate(input, source_lang);
        EXPECT_FALSE(result.empty());
        spdlog::info("Chinese to English: {}", result);
    } catch (const std::exception& e) {
        FAIL() << "Chinese to English translation failed: " << e.what();
    }
}

// 测试长文本翻译
TEST_F(NLLBTranslatorTest, LongTextTranslation) {
    const std::string input = "This is a long text that needs to be translated. "
                             "It contains multiple sentences and should test the model's "
                             "ability to handle longer sequences. The translation should "
                             "maintain coherence across sentences and properly handle "
                             "context throughout the text.";
    const std::string source_lang = "eng_Latn";
    
    try {
        std::string result = translator->translate(input, source_lang);
        EXPECT_FALSE(result.empty());
        spdlog::info("Long text translation result: {}", result);
    } catch (const std::exception& e) {
        FAIL() << "Long text translation failed: " << e.what();
    }
}

// 测试特殊字符处理
TEST_F(NLLBTranslatorTest, SpecialCharacters) {
    const std::string input = "Hello! @#$%^&* 你好！";
    const std::string source_lang = "eng_Latn";
    
    try {
        std::string result = translator->translate(input, source_lang);
        EXPECT_FALSE(result.empty());
        spdlog::info("Special characters translation: {}", result);
    } catch (const std::exception& e) {
        FAIL() << "Special characters translation failed: " << e.what();
    }
}

// 测试错误处理 - 无效的源语言
TEST_F(NLLBTranslatorTest, InvalidSourceLanguage) {
    const std::string input = "Hello";
    const std::string invalid_lang = "invalid_lang";
    
    EXPECT_THROW(translator->translate(input, invalid_lang), std::runtime_error);
}

// 测试空输入
TEST_F(NLLBTranslatorTest, EmptyInput) {
    const std::string input = "";
    const std::string source_lang = "eng_Latn";
    
    try {
        std::string result = translator->translate(input, source_lang);
        EXPECT_TRUE(result.empty());
    } catch (const std::exception& e) {
        FAIL() << "Empty input handling failed: " << e.what();
    }
}

// 测试多语言切换
TEST_F(NLLBTranslatorTest, MultiLanguageSwitch) {
    struct TestCase {
        std::string input;
        std::string source_lang;
        std::string description;
    };

    std::vector<TestCase> test_cases = {
        {"Hello, world!", "eng_Latn", "English"},
        {"你好，世界！", "zho_Hans", "Chinese"},
        {"Bonjour le monde!", "fra_Latn", "French"},
        {"こんにちは、世界！", "jpn_Jpan", "Japanese"}
    };

    for (const auto& test : test_cases) {
        try {
            std::string result = translator->translate(test.input, test.source_lang);
            EXPECT_FALSE(result.empty());
            spdlog::info("{} translation result: {}", test.description, result);
        } catch (const std::exception& e) {
            FAIL() << test.description << " translation failed: " << e.what();
        }
    }
}

// 测试目标语言获取
TEST_F(NLLBTranslatorTest, GetTargetLanguage) {
    std::string target_lang = translator->get_target_language();
    EXPECT_EQ(target_lang, "eng_Latn");
} 
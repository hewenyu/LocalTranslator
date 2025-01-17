#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "translator/nllb-api/nllb_translator.h"
#include "common/common.h"

using namespace nllb;

class NLLBTranslatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Configure spdlog
        auto console = spdlog::stdout_color_mt("console");
        spdlog::set_default_logger(console);
        spdlog::set_level(spdlog::level::trace);
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");
        
        spdlog::info("Starting NLLB translator test setup...");
        
        try {
            // Configure translator
            common::TranslatorConfig config;
            config.nllb.model_dir = "models";
            config.nllb.target_lang = "eng_Latn";
            config.nllb.params.beam_size = 5;
            config.nllb.params.max_length = 128;
            config.nllb.params.length_penalty = 1.0f;
            config.nllb.params.temperature = 1.0f;
            config.nllb.params.top_k = 0;
            config.nllb.params.top_p = 0.9f;
            config.nllb.params.repetition_penalty = 0.9f;
            config.nllb.params.num_threads = 4;
            config.nllb.params.use_cache = true;
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

// Test basic translation functionality
TEST_F(NLLBTranslatorTest, BasicTranslation) {
    spdlog::info("Running BasicTranslation test...");
    const std::string input = "Hello, how are you?";
    const std::string source_lang = "eng_Latn";
    
    std::string result = translator->translate(input, source_lang);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
    spdlog::info("Translation result: '{}'", result);
}

// Test Chinese to English translation
TEST_F(NLLBTranslatorTest, ChineseToEnglish) {
    const std::string input = "你好，最近过得怎么样？";
    const std::string source_lang = "zho_Hans";
    
    std::string result = translator->translate(input, source_lang);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
    spdlog::info("Chinese to English: {}", result);
}

// Test language code conversion
TEST_F(NLLBTranslatorTest, LanguageCodeConversion) {
    // Test standard language codes
    EXPECT_EQ(translator->get_nllb_language_code("en"), "eng_Latn");
    EXPECT_EQ(translator->get_nllb_language_code("zh"), "zho_Hans");
    
    // Test display language codes
    EXPECT_EQ(translator->get_display_language_code("eng_Latn"), "en");
    EXPECT_EQ(translator->get_display_language_code("zho_Hans"), "zh");
}

// Test low quality language support
TEST_F(NLLBTranslatorTest, LowQualityLanguageSupport) {
    // Default should not support low quality languages
    EXPECT_FALSE(translator->get_support_low_quality_languages());
    
    // Enable low quality language support
    translator->set_support_low_quality_languages(true);
    EXPECT_TRUE(translator->get_support_low_quality_languages());
    
    // Verify supported languages list is updated
    auto languages = translator->get_supported_languages();
    EXPECT_FALSE(languages.empty());
}

// Test translation parameters
TEST_F(NLLBTranslatorTest, TranslationParameters) {
    // Test beam size
    translator->set_beam_size(3);
    EXPECT_EQ(translator->get_beam_size(), 3);
    
    // Test max length
    translator->set_max_length(64);
    EXPECT_EQ(translator->get_max_length(), 64);
    
    // Test length penalty
    translator->set_length_penalty(0.8f);
    EXPECT_FLOAT_EQ(translator->get_length_penalty(), 0.8f);
    
    // Test temperature
    translator->set_temperature(0.7f);
    EXPECT_FLOAT_EQ(translator->get_temperature(), 0.7f);
    
    // Test top_k
    translator->set_top_k(5);
    EXPECT_FLOAT_EQ(translator->get_top_k(), 5);
    
    // Test top_p
    translator->set_top_p(0.95f);
    EXPECT_FLOAT_EQ(translator->get_top_p(), 0.95f);
    
    // Test repetition penalty
    translator->set_repetition_penalty(0.85f);
    EXPECT_FLOAT_EQ(translator->get_repetition_penalty(), 0.85f);
}

// Test error handling
TEST_F(NLLBTranslatorTest, ErrorHandling) {
    // Test invalid language code
    std::string result = translator->translate("Hello", "invalid_lang");
    EXPECT_TRUE(result.empty());
    EXPECT_NE(translator->get_last_error(), TranslatorError::OK);
    
    // Test empty input
    result = translator->translate("", "eng_Latn");
    EXPECT_TRUE(result.empty());
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
}

// Test long text translation
TEST_F(NLLBTranslatorTest, LongTextTranslation) {
    std::string long_text(1000, 'a');  // Create a 1000 character string
    std::string result = translator->translate(long_text, "eng_Latn");
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
}

// Test batch translation
TEST_F(NLLBTranslatorTest, BatchTranslation) {
    std::vector<std::string> texts = {
        "Hello, how are you?",
        "Nice to meet you",
        "Have a great day"
    };
    
    auto results = translator->translate_batch(texts, "eng_Latn");
    EXPECT_EQ(results.size(), texts.size());
    for (const auto& result : results) {
        EXPECT_FALSE(result.empty());
    }
}

// Test special characters
TEST_F(NLLBTranslatorTest, SpecialCharacters) {
    const std::string input = "Hello! @#$%^&*()_+ 你好";
    std::string result = translator->translate(input, "eng_Latn");
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 
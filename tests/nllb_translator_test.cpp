#include <gtest/gtest.h>
#include "translator/nllb-api/nllb_translator.h"
#include "common/common.h"
#include <memory>
#include <string>
#include <filesystem>

class NLLBTranslatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        common::TranslatorConfig config;
        config.type = "NLLB";
        
        // Set absolute model directory path
        std::string model_dir = "C:/Users/boringsoft/code/hewenyu/LocalTranslator/models";
        config.nllb.model_dir = model_dir;
        config.nllb.target_lang = "zho_Hans";  // 简体中文
        
        // Model files with absolute paths
        config.nllb.model_files.encoder = model_dir + "/NLLB_encoder.onnx";
        config.nllb.model_files.decoder = model_dir + "/NLLB_decoder.onnx";
        config.nllb.model_files.embed_lm_head = model_dir + "/NLLB_embed_and_lm_head.onnx";
        config.nllb.model_files.cache_initializer = model_dir + "/NLLB_cache_initializer.onnx";
        config.nllb.model_files.tokenizer_vocab = model_dir + "/sentencepiece_bpe.model";
        config.nllb.model_files.language_config = model_dir + "/nllb_languages.yaml";
        
        // Parameters
        config.nllb.params.beam_size = 5;
        config.nllb.params.max_length = 128;
        config.nllb.params.length_penalty = 1.0f;
        config.nllb.params.temperature = 1.0f;
        config.nllb.params.top_k = 0;
        config.nllb.params.top_p = 0.9f;
        config.nllb.params.repetition_penalty = 1.0f;
        config.nllb.params.num_threads = 4;
        config.nllb.params.support_low_quality_languages = false;
        
        translator = std::make_unique<nllb::NLLBTranslator>(config);
    }

    void TearDown() override {
        translator.reset();
    }

    std::unique_ptr<nllb::NLLBTranslator> translator;
};

TEST_F(NLLBTranslatorTest, BasicTranslation) {
    const std::string input = "Hello, how are you?";
    const std::string source_lang = "eng_Latn";
    
    auto result = translator->translate(input, source_lang);
    EXPECT_FALSE(result.empty());
}

TEST_F(NLLBTranslatorTest, LongTextTranslation) {
    const std::string input = "This is a longer text that should test the translator's ability to handle multiple sentences. "
                             "It includes various punctuation marks and should exercise different aspects of the translation process.";
    const std::string source_lang = "eng_Latn";
    
    auto result = translator->translate(input, source_lang);
    EXPECT_FALSE(result.empty());
}

TEST_F(NLLBTranslatorTest, ErrorHandling) {
    // Test with empty input
    auto result = translator->translate("", "eng_Latn");
    EXPECT_TRUE(result.empty());
    EXPECT_NE(translator->get_last_error(), translator::TranslatorError::OK);
    
    // Test with invalid language code
    result = translator->translate("Hello", "invalid_lang");
    EXPECT_TRUE(result.empty());
    EXPECT_NE(translator->get_last_error(), translator::TranslatorError::OK);
}

TEST_F(NLLBTranslatorTest, ParameterSettings) {
    // Test beam size
    translator->set_beam_size(3);
    auto result = translator->translate("Hello", "eng_Latn");
    EXPECT_FALSE(result.empty());
    
    // Test temperature
    translator->set_temperature(0.8f);
    result = translator->translate("Hello", "eng_Latn");
    EXPECT_FALSE(result.empty());
    
    // Test top_k
    translator->set_top_k(5);
    result = translator->translate("Hello", "eng_Latn");
    EXPECT_FALSE(result.empty());
    
    // Test top_p
    translator->set_top_p(0.95f);
    result = translator->translate("Hello", "eng_Latn");
    EXPECT_FALSE(result.empty());
}

TEST_F(NLLBTranslatorTest, BatchTranslation) {
    std::vector<std::string> inputs = {
        "Hello, how are you?",
        "The weather is nice today.",
        "I love programming."
    };
    const std::string source_lang = "eng_Latn";
    
    auto results = translator->translate_batch(inputs, source_lang);
    EXPECT_EQ(results.size(), inputs.size());
    for (const auto& result : results) {
        EXPECT_FALSE(result.empty());
    }
}

TEST_F(NLLBTranslatorTest, CacheManagement) {
    // Test translation with cache reset
    auto result1 = translator->translate("Hello", "eng_Latn");
    translator->reset_cache();
    auto result2 = translator->translate("Hello", "eng_Latn");
    
    EXPECT_FALSE(result1.empty());
    EXPECT_FALSE(result2.empty());
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 
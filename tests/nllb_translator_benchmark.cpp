#include <benchmark/benchmark.h>
#include "translator/nllb-api/nllb_translator.h"
#include "common/common.h"
#include <memory>
#include <string>
#include <vector>

// 性能测试固定设置
class NLLBTranslatorBenchmark : public benchmark::Fixture {
protected:
    void SetUp(const benchmark::State& state) override {
        // 初始化翻译器配置
        common::TranslatorConfig config;
        config.type = "NLLB";
        config.nllb.model_dir = "C:/Users/boringsoft/code/hewenyu/LocalTranslator/models";
        config.nllb.target_lang = "zho_Hans";  // 简体中文
        
        // Model files
        config.nllb.model_files.encoder = config.nllb.model_dir + "/NLLB_encoder.onnx";
        config.nllb.model_files.decoder = config.nllb.model_dir + "/NLLB_decoder.onnx";
        config.nllb.model_files.embed_lm_head = config.nllb.model_dir + "/NLLB_embed_and_lm_head.onnx";
        config.nllb.model_files.cache_initializer = config.nllb.model_dir + "/NLLB_cache_initializer.onnx";
        config.nllb.model_files.tokenizer_vocab = config.nllb.model_dir + "/sentencepiece_bpe.model";
        config.nllb.model_files.language_config = config.nllb.model_dir + "/nllb_languages.yaml";
        
        // 设置默认参数
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

    void TearDown(const benchmark::State& state) override {
        translator.reset();
    }

    std::unique_ptr<nllb::NLLBTranslator> translator;
};

// 短文本翻译性能测试
BENCHMARK_F(NLLBTranslatorBenchmark, ShortTextTranslation)(benchmark::State& state) {
    const std::string input = "Hello, how are you?";
    const std::string source_lang = "eng_Latn";
    
    for (auto _ : state) {
        auto result = translator->translate(input, source_lang);
        if (result.empty() || translator->get_last_error() != translator::TranslatorError::OK) {
            state.SkipWithError("Translation failed");
            break;
        }
    }
}

// 长文本翻译性能测试
BENCHMARK_F(NLLBTranslatorBenchmark, LongTextTranslation)(benchmark::State& state) {
    const std::string input = "This is a longer text that should test the translator's ability to handle multiple sentences. "
                             "It includes various punctuation marks and should exercise different aspects of the translation process. "
                             "The text is intentionally long to measure performance with larger inputs.";
    const std::string source_lang = "eng_Latn";
    
    for (auto _ : state) {
        auto result = translator->translate(input, source_lang);
        if (result.empty() || translator->get_last_error() != translator::TranslatorError::OK) {
            state.SkipWithError("Translation failed");
            break;
        }
    }
}

// 批量翻译性能测试
BENCHMARK_F(NLLBTranslatorBenchmark, BatchTranslation)(benchmark::State& state) {
    std::vector<std::string> inputs = {
        "Hello, how are you?",
        "The weather is nice today.",
        "I love programming.",
        "This is a test message.",
        "Goodbye!"
    };
    const std::string source_lang = "eng_Latn";
    
    for (auto _ : state) {
        auto results = translator->translate_batch(inputs, source_lang);
        if (results.size() != inputs.size() || translator->get_last_error() != translator::TranslatorError::OK) {
            state.SkipWithError("Batch translation failed");
            break;
        }
    }
}

// 不同语言翻译性能测试
BENCHMARK_F(NLLBTranslatorBenchmark, MultiLanguageTranslation)(benchmark::State& state) {
    const std::string input = "Hello, how are you?";
    const std::vector<std::string> languages = {
        "eng_Latn", "fra_Latn", "deu_Latn", "spa_Latn", "rus_Cyrl"
    };
    
    for (auto _ : state) {
        for (const auto& lang : languages) {
            auto result = translator->translate(input, lang);
            if (result.empty() || translator->get_last_error() != translator::TranslatorError::OK) {
                state.SkipWithError("Multi-language translation failed");
                break;
            }
        }
    }
}

// 缓存性能测试
BENCHMARK_F(NLLBTranslatorBenchmark, CachePerformance)(benchmark::State& state) {
    const std::string input = "Hello, how are you?";
    const std::string source_lang = "eng_Latn";
    
    for (auto _ : state) {
        // First translation (no cache)
        auto result1 = translator->translate(input, source_lang);
        if (result1.empty() || translator->get_last_error() != translator::TranslatorError::OK) {
            state.SkipWithError("First translation failed");
            break;
        }
        
        // Second translation (with cache)
        auto result2 = translator->translate(input, source_lang);
        if (result2.empty() || translator->get_last_error() != translator::TranslatorError::OK) {
            state.SkipWithError("Second translation failed");
            break;
        }
        
        // Reset cache for next iteration
        translator->reset_cache();
        if (translator->get_last_error() != translator::TranslatorError::OK) {
            state.SkipWithError("Cache reset failed");
            break;
        }
    }
}

BENCHMARK_MAIN(); 
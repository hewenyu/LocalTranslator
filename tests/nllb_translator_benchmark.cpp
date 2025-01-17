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
        config.nllb.model_files.encoder = "NLLB_encoder.onnx";
        config.nllb.model_files.decoder = "NLLB_decoder.onnx";
        config.nllb.model_files.embed_lm_head = "NLLB_embed_and_lm_head.onnx";
        config.nllb.model_files.cache_initializer = "NLLB_cache_initializer.onnx";
        config.nllb.model_files.tokenizer_vocab = "sentencepiece_bpe.model";
        config.nllb.model_files.language_config = "nllb_languages.yaml";
        
        // 设置默认参数
        config.nllb.params.beam_size = 5;
        config.nllb.params.num_threads = 4;
        config.nllb.params.length_penalty = 1.0f;
        config.nllb.params.temperature = 1.0f;
        config.nllb.params.top_k = 0;
        config.nllb.params.top_p = 0.9f;
        config.nllb.params.repetition_penalty = 1.0f;
        
        // 创建翻译器实例
        translator = std::make_unique<nllb::NLLBTranslator>(config);
    }

    void TearDown(const benchmark::State& state) override {
        translator.reset();
    }

    std::unique_ptr<nllb::NLLBTranslator> translator;
    
    // 测试用例
    const std::vector<std::string> test_sentences = {
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
        "This is a longer sentence that contains multiple clauses and more complex grammar structures.",
        "Machine translation has made significant progress in recent years.",
        "I love programming and developing new applications."
    };
};

// 基准测试：单句翻译
BENCHMARK_F(NLLBTranslatorBenchmark, SingleSentenceTest)(benchmark::State& state) {
    const std::string& text = test_sentences[state.range(0) % test_sentences.size()];
    
    for (auto _ : state) {
        auto result = translator->translate(text, "eng_Latn");
        benchmark::DoNotOptimize(result);
    }
}

// 基准测试：不同的 beam size
BENCHMARK_F(NLLBTranslatorBenchmark, BeamSizeTest)(benchmark::State& state) {
    const std::string& text = test_sentences[0];
    translator->set_beam_size(state.range(0));

    for (auto _ : state) {
        auto result = translator->translate(text, "eng_Latn");
        benchmark::DoNotOptimize(result);
    }
}

// 基准测试：不同的线程数
BENCHMARK_F(NLLBTranslatorBenchmark, ThreadCountTest)(benchmark::State& state) {
    const std::string& text = test_sentences[0];
    translator->set_num_threads(state.range(0));

    for (auto _ : state) {
        auto result = translator->translate(text, "eng_Latn");
        benchmark::DoNotOptimize(result);
    }
}

// 基准测试：批处理性能
BENCHMARK_F(NLLBTranslatorBenchmark, BatchProcessingTest)(benchmark::State& state) {
    const int batch_size = state.range(0);
    std::vector<std::string> texts;
    texts.reserve(batch_size);
    
    for (int i = 0; i < batch_size; ++i) {
        texts.push_back(test_sentences[i % test_sentences.size()]);
    }

    for (auto _ : state) {
        auto results = translator->translate_batch(texts, "eng_Latn");
        benchmark::DoNotOptimize(results);
    }
}

// 注册基准测试
BENCHMARK_REGISTER_F(NLLBTranslatorBenchmark, SingleSentenceTest)
    ->Range(0, 4)  // 测试不同的句子
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(NLLBTranslatorBenchmark, BeamSizeTest)
    ->Range(1, 5)  // beam size: 1-5
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(NLLBTranslatorBenchmark, ThreadCountTest)
    ->Range(1, 4)  // threads: 1-4
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(NLLBTranslatorBenchmark, BatchProcessingTest)
    ->Range(1, 5)  // batch size: 1-5
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN(); 
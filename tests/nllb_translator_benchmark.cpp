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
        config.nllb.target_lang = "ZH";
        config.nllb.model_files.encoder = "NLLB_encoder.onnx";
        config.nllb.model_files.decoder = "NLLB_decoder.onnx";
        config.nllb.model_files.embed_lm_head = "NLLB_embed_and_lm_head.onnx";
        config.nllb.model_files.cache_initializer = "NLLB_cache_initializer.onnx";
        config.nllb.model_files.tokenizer_vocab = "sentencepiece_bpe.model";
        config.nllb.model_files.language_config = "nllb_languages.yaml";
        config.nllb.params.beam_size = 5;
        config.nllb.params.num_threads = 4;
        config.nllb.params.length_penalty = 1.0f;
        config.nllb.params.temperature = 1.0f;
        config.nllb.params.top_k = 0;
        config.nllb.params.top_p = 0.9f;
        config.nllb.params.repetition_penalty = 0.9f;
        
        // 创建翻译器实例
        translator = std::make_unique<nllb::NLLBTranslator>(config);
    }

    void TearDown(const benchmark::State& state) override {
        translator.reset();
    }

    std::unique_ptr<nllb::NLLBTranslator> translator;
};

// 基准测试：不同的 beam size
BENCHMARK_F(NLLBTranslatorBenchmark, BeamSizeTest)(benchmark::State& state) {
    const std::string text = "This is a test sentence for benchmarking.";
    translator->set_beam_size(state.range(0));

    for (auto _ : state) {
        translator->translate(text, "en");
    }
}

// 基准测试：不同的输入长度
BENCHMARK_F(NLLBTranslatorBenchmark, InputLengthTest)(benchmark::State& state) {
    std::string text(state.range(0), 'a');
    for (auto _ : state) {
        translator->translate(text, "en");
    }
}

// 基准测试：不同的线程数
BENCHMARK_F(NLLBTranslatorBenchmark, ThreadCountTest)(benchmark::State& state) {
    const std::string text = "This is a test sentence for benchmarking.";
    translator->set_num_threads(state.range(0));

    for (auto _ : state) {
        translator->translate(text, "en");
    }
}

// 基准测试：不同的温度值
BENCHMARK_F(NLLBTranslatorBenchmark, TemperatureTest)(benchmark::State& state) {
    const std::string text = "This is a test sentence for benchmarking.";
    translator->set_temperature(state.range(0) / 10.0f);

    for (auto _ : state) {
        translator->translate(text, "en");
    }
}

// 基准测试：不同的 top_k 值
BENCHMARK_F(NLLBTranslatorBenchmark, TopKTest)(benchmark::State& state) {
    const std::string text = "This is a test sentence for benchmarking.";
    translator->set_top_k(state.range(0));

    for (auto _ : state) {
        translator->translate(text, "en");
    }
}

// 基准测试：批处理性能
BENCHMARK_F(NLLBTranslatorBenchmark, BatchProcessingTest)(benchmark::State& state) {
    std::vector<std::string> texts;
    const int batch_size = state.range(0);
    for (int i = 0; i < batch_size; ++i) {
        texts.push_back("This is test sentence " + std::to_string(i) + " for batch processing.");
    }

    for (auto _ : state) {
        translator->translate_batch(texts, "en");
    }
}

// 注册基准测试
BENCHMARK_REGISTER_F(NLLBTranslatorBenchmark, BeamSizeTest)->Range(1, 8);
BENCHMARK_REGISTER_F(NLLBTranslatorBenchmark, InputLengthTest)->Range(8, 512);
BENCHMARK_REGISTER_F(NLLBTranslatorBenchmark, ThreadCountTest)->Range(1, 8);
BENCHMARK_REGISTER_F(NLLBTranslatorBenchmark, TemperatureTest)->Range(5, 20);
BENCHMARK_REGISTER_F(NLLBTranslatorBenchmark, TopKTest)->Range(0, 100);
BENCHMARK_REGISTER_F(NLLBTranslatorBenchmark, BatchProcessingTest)->Range(1, 16);

BENCHMARK_MAIN(); 
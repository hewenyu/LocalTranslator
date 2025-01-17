#include <benchmark/benchmark.h>
#include <filesystem>
#include <fstream>
#include "translator/nllb-api/nllb_translator.h"

namespace {
namespace fs = std::filesystem;

// Benchmark fixture with fixed settings
class NLLBTranslatorBenchmark : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) {
        // Setup test configuration
        config.nllb.model_dir = "../../../models";
        config.nllb.target_lang = "zh";
        config.nllb.params.beam_size = state.range(0);
        config.nllb.params.max_length = 128;
        config.nllb.params.length_penalty = 1.0f;
        config.nllb.params.temperature = state.range(2) / 10.0f;
        config.nllb.params.top_k = state.range(3);
        config.nllb.params.top_p = 0.9f;
        config.nllb.params.repetition_penalty = 0.9f;
        config.nllb.params.num_threads = state.range(1);
        config.nllb.params.use_cache = true;
        config.nllb.model_files.tokenizer_vocab = "sentencepiece_bpe.model";
        
        translator = std::make_unique<nllb::NLLBTranslator>(config);
    }

    void TearDown(const benchmark::State&) {
        translator.reset();
    }

protected:
    common::TranslatorConfig config;
    std::unique_ptr<nllb::NLLBTranslator> translator;
};

// Benchmark different beam sizes
BENCHMARK_DEFINE_F(NLLBTranslatorBenchmark, BeamSizeTest)(benchmark::State& state) {
    const std::string text = "Hello, world! This is a test sentence for benchmarking.";
    for (auto _ : state) {
        std::string result = translator->translate(text, "en");
        if (translator->get_last_error() != nllb::TranslatorError::OK) {
            state.SkipWithError(translator->get_error_message().c_str());
            break;
        }
    }
    
    // Set custom counters
    state.counters["beam_size"] = state.range(0);
    state.counters["threads"] = state.range(1);
    state.counters["temperature"] = state.range(2) / 10.0f;
    state.counters["top_k"] = state.range(3);
    state.counters["text_length"] = text.length();
}

// Benchmark different input lengths
BENCHMARK_DEFINE_F(NLLBTranslatorBenchmark, InputLengthTest)(benchmark::State& state) {
    // Generate input text of specified length
    std::string text(state.range(4), 'a');
    for (auto _ : state) {
        std::string result = translator->translate(text, "en");
        if (translator->get_last_error() != nllb::TranslatorError::OK) {
            state.SkipWithError(translator->get_error_message().c_str());
            break;
        }
    }
    
    // Set custom counters
    state.counters["beam_size"] = state.range(0);
    state.counters["threads"] = state.range(1);
    state.counters["temperature"] = state.range(2) / 10.0f;
    state.counters["top_k"] = state.range(3);
    state.counters["text_length"] = state.range(4);
}

// Benchmark different thread counts
BENCHMARK_DEFINE_F(NLLBTranslatorBenchmark, ThreadCountTest)(benchmark::State& state) {
    const std::string text = "Hello, world! This is a test sentence for benchmarking.";
    for (auto _ : state) {
        std::string result = translator->translate(text, "en");
        if (translator->get_last_error() != nllb::TranslatorError::OK) {
            state.SkipWithError(translator->get_error_message().c_str());
            break;
        }
    }
    
    // Set custom counters
    state.counters["beam_size"] = state.range(0);
    state.counters["threads"] = state.range(1);
    state.counters["temperature"] = state.range(2) / 10.0f;
    state.counters["top_k"] = state.range(3);
    state.counters["text_length"] = text.length();
}

// Benchmark different temperature values
BENCHMARK_DEFINE_F(NLLBTranslatorBenchmark, TemperatureTest)(benchmark::State& state) {
    const std::string text = "Hello, world! This is a test sentence for benchmarking.";
    for (auto _ : state) {
        std::string result = translator->translate(text, "en");
        if (translator->get_last_error() != nllb::TranslatorError::OK) {
            state.SkipWithError(translator->get_error_message().c_str());
            break;
        }
    }
    
    // Set custom counters
    state.counters["beam_size"] = state.range(0);
    state.counters["threads"] = state.range(1);
    state.counters["temperature"] = state.range(2) / 10.0f;
    state.counters["top_k"] = state.range(3);
    state.counters["text_length"] = text.length();
}

// Benchmark different top-k values
BENCHMARK_DEFINE_F(NLLBTranslatorBenchmark, TopKTest)(benchmark::State& state) {
    const std::string text = "Hello, world! This is a test sentence for benchmarking.";
    for (auto _ : state) {
        std::string result = translator->translate(text, "en");
        if (translator->get_last_error() != nllb::TranslatorError::OK) {
            state.SkipWithError(translator->get_error_message().c_str());
            break;
        }
    }
    
    // Set custom counters
    state.counters["beam_size"] = state.range(0);
    state.counters["threads"] = state.range(1);
    state.counters["temperature"] = state.range(2) / 10.0f;
    state.counters["top_k"] = state.range(3);
    state.counters["text_length"] = text.length();
}

// Register benchmarks with different parameter combinations
BENCHMARK_REGISTER_F(NLLBTranslatorBenchmark, BeamSizeTest)
    ->Args({1, 4, 10, 0})   // beam_size=1, threads=4, temp=1.0, top_k=0
    ->Args({3, 4, 10, 0})   // beam_size=3, threads=4, temp=1.0, top_k=0
    ->Args({5, 4, 10, 0})   // beam_size=5, threads=4, temp=1.0, top_k=0
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(NLLBTranslatorBenchmark, InputLengthTest)
    ->Args({5, 4, 10, 0, 32})    // beam_size=5, threads=4, temp=1.0, top_k=0, length=32
    ->Args({5, 4, 10, 0, 64})    // beam_size=5, threads=4, temp=1.0, top_k=0, length=64
    ->Args({5, 4, 10, 0, 128})   // beam_size=5, threads=4, temp=1.0, top_k=0, length=128
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(NLLBTranslatorBenchmark, ThreadCountTest)
    ->Args({5, 1, 10, 0})   // beam_size=5, threads=1, temp=1.0, top_k=0
    ->Args({5, 2, 10, 0})   // beam_size=5, threads=2, temp=1.0, top_k=0
    ->Args({5, 4, 10, 0})   // beam_size=5, threads=4, temp=1.0, top_k=0
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(NLLBTranslatorBenchmark, TemperatureTest)
    ->Args({5, 4, 5, 0})    // beam_size=5, threads=4, temp=0.5, top_k=0
    ->Args({5, 4, 10, 0})   // beam_size=5, threads=4, temp=1.0, top_k=0
    ->Args({5, 4, 15, 0})   // beam_size=5, threads=4, temp=1.5, top_k=0
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(NLLBTranslatorBenchmark, TopKTest)
    ->Args({5, 4, 10, 0})   // beam_size=5, threads=4, temp=1.0, top_k=0
    ->Args({5, 4, 10, 5})   // beam_size=5, threads=4, temp=1.0, top_k=5
    ->Args({5, 4, 10, 10})  // beam_size=5, threads=4, temp=1.0, top_k=10
    ->Unit(benchmark::kMillisecond);

} // namespace

BENCHMARK_MAIN(); 
#include <benchmark/benchmark.h>
#include <filesystem>
#include <fstream>
#include "translator/nllb-api/nllb_translator.h"

namespace {
namespace fs = std::filesystem;

// 基准测试的固定设置
class NLLBTranslatorBenchmark : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) {
        // 设置测试配置
        config.nllb.model_dir = "test_models/nllb";
        config.nllb.target_lang = "eng_Latn";
        config.nllb.params.beam_size = state.range(0);  // 使用参数作为beam size
        config.nllb.params.max_length = 128;
        config.nllb.params.length_penalty = 1.0f;
        config.nllb.params.temperature = 1.0f;
        config.nllb.params.num_threads = static_cast<int>(state.range(1));  // 使用参数作为线程数
        config.nllb.params.use_cache = true;
        config.nllb.model_files.tokenizer_vocab = "tokenizer.model";
        
        // 确保测试目录存在
        fs::create_directories(config.nllb.model_dir);
        create_test_configs(config.nllb.model_dir);
        
        translator = std::make_unique<nllb::NLLBTranslator>(config);
    }

    void TearDown(const benchmark::State&) {
        translator.reset();
        if (fs::exists("test_models")) {
            fs::remove_all("test_models");
        }
    }

protected:
    void create_test_configs(const std::string& model_dir) {
        // 创建model_config.yaml
        std::ofstream model_config(model_dir + "/model_config.yaml");
        model_config << R"(
hidden_size: 1024
num_heads: 16
vocab_size: 256200
max_position_embeddings: 1024
encoder_layers: 24
decoder_layers: 24
)";
        model_config.close();
        
        // 创建language_codes.yaml
        std::ofstream lang_config(model_dir + "/nllb_languages.yaml");
        lang_config << R"(
languages:
  - code: "eng"
    code_NLLB: "eng_Latn"
  - code: "fra"
    code_NLLB: "fra_Latn"
  - code: "cmn"
    code_NLLB: "zho_Hans"
)";
        lang_config.close();
        
        // 创建空的模型文件
        std::ofstream(model_dir + "/NLLB_encoder.onnx").close();
        std::ofstream(model_dir + "/NLLB_decoder.onnx").close();
        std::ofstream(model_dir + "/NLLB_embed_and_lm_head.onnx").close();
        std::ofstream(model_dir + "/NLLB_cache_initializer.onnx").close();
        std::ofstream(model_dir + "/tokenizer.model").close();
    }

    common::TranslatorConfig config;
    std::unique_ptr<nllb::NLLBTranslator> translator;
};

// 测试不同beam size的性能
BENCHMARK_DEFINE_F(NLLBTranslatorBenchmark, BeamSizeTest)(benchmark::State& state) {
    const std::string text = "Hello, world! This is a test sentence for benchmarking.";
    for (auto _ : state) {
        try {
            translator->translate(text, "eng");
        } catch (const std::exception&) {
            // 忽略错误，因为我们使用的是空模型
        }
    }
    
    // 设置自定义计数器
    state.counters["beam_size"] = state.range(0);
    state.counters["threads"] = state.range(1);
    state.counters["text_length"] = text.length();
}

// 测试不同输入长度的性能
BENCHMARK_DEFINE_F(NLLBTranslatorBenchmark, InputLengthTest)(benchmark::State& state) {
    // 生成指定长度的输入文本
    std::string text(state.range(2), 'a');
    for (auto _ : state) {
        try {
            translator->translate(text, "eng");
        } catch (const std::exception&) {
            // 忽略错误，因为我们使用的是空模型
        }
    }
    
    // 设置自定义计数器
    state.counters["beam_size"] = state.range(0);
    state.counters["threads"] = state.range(1);
    state.counters["text_length"] = state.range(2);
}

// 测试不同线程数的性能
BENCHMARK_DEFINE_F(NLLBTranslatorBenchmark, ThreadCountTest)(benchmark::State& state) {
    const std::string text = "Hello, world! This is a test sentence for benchmarking.";
    for (auto _ : state) {
        try {
            translator->translate(text, "eng");
        } catch (const std::exception&) {
            // 忽略错误，因为我们使用的是空模型
        }
    }
    
    // 设置自定义计数器
    state.counters["beam_size"] = state.range(0);
    state.counters["threads"] = state.range(1);
    state.counters["text_length"] = text.length();
}

// 注册基准测试
BENCHMARK_REGISTER_F(NLLBTranslatorBenchmark, BeamSizeTest)
    ->Args({1, 4})   // beam_size=1, threads=4
    ->Args({5, 4})   // beam_size=5, threads=4
    ->Args({10, 4})  // beam_size=10, threads=4
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(NLLBTranslatorBenchmark, InputLengthTest)
    ->Args({5, 4, 10})    // beam_size=5, threads=4, length=10
    ->Args({5, 4, 100})   // beam_size=5, threads=4, length=100
    ->Args({5, 4, 1000})  // beam_size=5, threads=4, length=1000
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(NLLBTranslatorBenchmark, ThreadCountTest)
    ->Args({5, 1})  // beam_size=5, threads=1
    ->Args({5, 2})  // beam_size=5, threads=2
    ->Args({5, 4})  // beam_size=5, threads=4
    ->Args({5, 8})  // beam_size=5, threads=8
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

} // namespace

BENCHMARK_MAIN(); 
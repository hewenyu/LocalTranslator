#include <gtest/gtest.h>
#include "translator/nllb-api/nllb_translator.h"
#include "translator/nllb-api/cache_container.h"
#include "common/common.h"
#include <chrono>
#include <thread>
#include <spdlog/spdlog.h>

using namespace nllb;

class NLLBTranslatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        try {
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

    // Helper function to measure execution time
    template<typename Func>
    double measure_time(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    std::unique_ptr<NLLBTranslator> translator;
};

// 基本翻译功能测试
TEST_F(NLLBTranslatorTest, BasicTranslation) {
    const std::string input = "Hello, how are you?";
    const std::string source_lang = "en";
    
    std::string result = translator->translate(input, source_lang);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
}

// 中英翻译测试
TEST_F(NLLBTranslatorTest, ChineseToEnglish) {
    const std::string input = "你好，最近过得怎么样？";
    const std::string source_lang = "zh";
    
    std::string result = translator->translate(input, source_lang);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
}

// 内存池性能测试
TEST_F(NLLBTranslatorTest, MemoryPoolPerformance) {
    const std::string input = "Test memory pool performance";
    const std::string source_lang = "en";
    
    // 预热
    translator->translate(input, source_lang);
    
    // 测试内存池性能
    double with_pool_time = measure_time([&]() {
        for (int i = 0; i < 100; ++i) {
            translator->translate(input, source_lang);
        }
    });
    
    spdlog::info("Memory pool performance test: {} ms for 100 translations", with_pool_time);
    EXPECT_LT(with_pool_time, 5000.0); // 假设5秒是合理的性能标准
}

// KV Cache 性能测试
TEST_F(NLLBTranslatorTest, KVCachePerformance) {
    const std::vector<std::string> inputs = {
        "First sentence to test cache",
        "Second sentence with similar context",
        "Third sentence continuing the context"
    };
    
    double cache_time = measure_time([&]() {
        for (const auto& input : inputs) {
            translator->translate(input, "en");
        }
    });
    
    spdlog::info("KV Cache performance test: {} ms for sequential translation", cache_time);
    EXPECT_LT(cache_time, 3000.0); // 假设3秒是合理的性能标准
}

// 批处理性能测试
TEST_F(NLLBTranslatorTest, BatchProcessingPerformance) {
    std::vector<std::string> batch_inputs(10, "Test batch processing");
    
    double batch_time = measure_time([&]() {
        auto results = translator->translate_batch(batch_inputs, "en");
        EXPECT_EQ(results.size(), batch_inputs.size());
    });
    
    // 对比单个处理的时间
    double single_time = measure_time([&]() {
        for (const auto& input : batch_inputs) {
            translator->translate(input, "en");
        }
    });
    
    spdlog::info("Batch vs Single processing: Batch={} ms, Single={} ms", 
                 batch_time, single_time);
    EXPECT_LT(batch_time, single_time);
}

// 内存使用测试
TEST_F(NLLBTranslatorTest, MemoryUsage) {
    const std::string long_input(10000, 'a'); // 长文本输入
    
    // 测试内存使用
    translator->translate(long_input, "en");
    
    // 验证内存释放
    translator->reset_cache();
    
    // 验证后续翻译是否正常
    std::string result = translator->translate("Test after reset", "en");
    EXPECT_FALSE(result.empty());
}

// 错误恢复测试
TEST_F(NLLBTranslatorTest, ErrorRecovery) {
    // 测试无效输入
    std::string result = translator->translate("", "en");
    EXPECT_TRUE(result.empty());
    
    // 验证后续翻译是否正常
    result = translator->translate("Test recovery", "en");
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(translator->get_last_error(), TranslatorError::OK);
}

// 并发测试
TEST_F(NLLBTranslatorTest, ConcurrentTranslation) {
    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::vector<std::string> results(num_threads);
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            results[i] = translator->translate("Concurrent test " + std::to_string(i), "en");
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    for (const auto& result : results) {
        EXPECT_FALSE(result.empty());
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 
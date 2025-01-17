#include <gtest/gtest.h>
#include "../translator/translator.h"
#include "../common/common.h"
#include <chrono>
#include <thread>

using namespace translator;

class TranslatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        common::TranslatorConfig config;
        config.type = "DeepLX";
        config.deeplx.url = "http://localhost:1188/translate";
        config.deeplx.target_lang = "zh";
        
        translator_ = CreateTranslator(TranslatorType::DeepLX, config);
    }

    void TearDown() override {
        translator_.reset();
    }

    // Helper function to measure execution time
    template<typename Func>
    double measure_time(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    std::unique_ptr<ITranslator> translator_;
};

// 翻译器创建测试
TEST_F(TranslatorTest, CreateTranslator) {
    ASSERT_NE(translator_, nullptr);
}

// 目标语言测试
TEST_F(TranslatorTest, GetTargetLanguage) {
    EXPECT_EQ(translator_->get_target_language(), "zh");
}

// 基本翻译功能测试
TEST_F(TranslatorTest, BasicTranslation) {
    const std::string input = "Hello, World!";
    const std::string source_lang = "en";
    
    double translation_time = measure_time([&]() {
        std::string result = translator_->translate(input, source_lang);
        EXPECT_FALSE(result.empty());
    });
    
    spdlog::info("Basic translation time: {} ms", translation_time);
}

// 性能测试
TEST_F(TranslatorTest, PerformanceTest) {
    const std::string input = "Performance test input";
    const std::string source_lang = "en";
    
    // 预热
    translator_->translate(input, source_lang);
    
    // 测试连续翻译性能
    double total_time = measure_time([&]() {
        for (int i = 0; i < 10; ++i) {
            std::string result = translator_->translate(input, source_lang);
            EXPECT_FALSE(result.empty());
        }
    });
    
    spdlog::info("Average translation time: {} ms", total_time / 10.0);
}

// 并发测试
TEST_F(TranslatorTest, ConcurrentTest) {
    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::vector<std::string> results(num_threads);
    
    double concurrent_time = measure_time([&]() {
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&, i]() {
                results[i] = translator_->translate(
                    "Concurrent test " + std::to_string(i), "en");
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    });
    
    spdlog::info("Concurrent translation time: {} ms", concurrent_time);
    
    for (const auto& result : results) {
        EXPECT_FALSE(result.empty());
    }
}

// 错误恢复测试
TEST_F(TranslatorTest, ErrorRecoveryTest) {
    // 测试无效输入
    std::string result = translator_->translate("", "en");
    EXPECT_TRUE(result.empty());
    
    // 验证后续翻译是否正常
    result = translator_->translate("Test recovery", "en");
    EXPECT_FALSE(result.empty());
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 
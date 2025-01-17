#include "cache_container.h"
#include "tensor_utils.h"
#include <stdexcept>

namespace nllb {

CacheContainer::CacheContainer(
    const Ort::MemoryInfo& memory_info,
    const std::vector<int64_t>& cache_shape)
    : memory_info_(memory_info),
      cache_shape_(cache_shape),
      is_initialized_(false) {}

void CacheContainer::initialize(Ort::Session* cache_init_session) {
    if (!cache_init_session) {
        throw std::runtime_error("Cache initialization session is null");
    }

    // 准备输入
    std::vector<float> dummy_input(1, 0.0f);  // 仅用于触发缓存初始化
    auto input_tensor = TensorUtils::createFloatTensor(
        memory_info_, dummy_input, {1, 1, 1});

    const char* input_names[] = {"dummy_input"};
    const char* output_names[] = {"key_cache", "value_cache"};

    // 运行缓存初始化
    auto outputs = cache_init_session->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        2
    );

    // 保存初始化的缓存
    key_cache_ = std::make_unique<Ort::Value>(std::move(outputs[0]));
    value_cache_ = std::make_unique<Ort::Value>(std::move(outputs[1]));
    is_initialized_ = true;
}

void CacheContainer::update(
    Ort::Value&& new_key_cache,
    Ort::Value&& new_value_cache) {
    
    if (!is_initialized_) {
        throw std::runtime_error("Cache not initialized");
    }

    // 验证缓存形状
    auto key_shape = TensorUtils::getTensorShape(new_key_cache);
    auto value_shape = TensorUtils::getTensorShape(new_value_cache);
    
    if (key_shape != cache_shape_ || value_shape != cache_shape_) {
        throw std::runtime_error("Invalid cache shape");
    }

    // 更新缓存
    key_cache_ = std::make_unique<Ort::Value>(std::move(new_key_cache));
    value_cache_ = std::make_unique<Ort::Value>(std::move(new_value_cache));
}

void CacheContainer::reset() {
    key_cache_.reset();
    value_cache_.reset();
    is_initialized_ = false;
}

Ort::Value CacheContainer::get_key_cache() const {
    if (!is_initialized_ || !key_cache_) {
        throw std::runtime_error("Key cache not initialized");
    }
    return *key_cache_;
}

Ort::Value CacheContainer::get_value_cache() const {
    if (!is_initialized_ || !value_cache_) {
        throw std::runtime_error("Value cache not initialized");
    }
    return *value_cache_;
}

} // namespace nllb 
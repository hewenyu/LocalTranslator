#pragma once

#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace nllb {

class CacheContainer {
public:
    CacheContainer(const Ort::MemoryInfo& memory_info, 
                  const std::vector<int64_t>& cache_shape);
    ~CacheContainer() = default;

    // 禁用拷贝
    CacheContainer(const CacheContainer&) = delete;
    CacheContainer& operator=(const CacheContainer&) = delete;

    // 允许移动
    CacheContainer(CacheContainer&&) = default;
    CacheContainer& operator=(CacheContainer&&) = default;

    // 缓存管理
    void initialize(Ort::Session* cache_init_session);
    void update(Ort::Value&& new_key_cache, Ort::Value&& new_value_cache);
    void reset();

    // 获取缓存
    Ort::Value get_key_cache() const;
    Ort::Value get_value_cache() const;

private:
    const Ort::MemoryInfo& memory_info_;
    std::vector<int64_t> cache_shape_;
    std::unique_ptr<Ort::Value> key_cache_;
    std::unique_ptr<Ort::Value> value_cache_;
    bool is_initialized_;
};

} // namespace nllb 
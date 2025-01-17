#pragma once

#include <vector>
#include <onnxruntime_cxx_api.h>

namespace nllb {

class CacheContainer {
public:
    CacheContainer() = default;
    ~CacheContainer() = default;

    void initializeCache(const Ort::MemoryInfo& memory_info, 
                        const std::vector<int64_t>& shape);
    
    void updateCache(std::vector<Ort::Value>& new_cache);
    
    const std::vector<Ort::Value>& getCache() const { return cache_; }
    void clearCache() { cache_.clear(); }

private:
    std::vector<Ort::Value> cache_;
    Ort::MemoryInfo memory_info_{nullptr};
    std::vector<int64_t> shape_;
};

} // namespace nllb 
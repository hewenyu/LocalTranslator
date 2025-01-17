#pragma once

#include <vector>
#include <onnxruntime_cxx_api.h>

namespace nllb {

class CacheContainer {
public:
    CacheContainer() = default;
    ~CacheContainer() = default;

    // Initialize cache with encoder output
    void initialize(
        Ort::Session& cache_init_session,
        const Ort::MemoryInfo& memory_info,
        const std::vector<float>& encoder_output,
        const std::vector<int64_t>& encoder_shape);

    // Add cache to decoder inputs
    void add_cache_to_inputs(std::vector<Ort::Value>& inputs) const;

    // Update cache with new values
    void update_cache(const Ort::Value& new_cache);

    // Clear cache
    void clear() { cache_.clear(); }

    // Get cache size
    size_t size() const { return cache_.size(); }

    // Check if cache is empty
    bool empty() const { return cache_.empty(); }

private:
    std::vector<Ort::Value> cache_;
};

} // namespace nllb 
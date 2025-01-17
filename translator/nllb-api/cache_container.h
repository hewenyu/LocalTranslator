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
    void update_cache(std::vector<Ort::Value>&& new_cache);

    // Get current cache
    const std::vector<Ort::Value>& get_cache() const;

    // Clear cache
    void clear();

private:
    std::vector<Ort::Value> cache_values_;
};

} // namespace nllb 
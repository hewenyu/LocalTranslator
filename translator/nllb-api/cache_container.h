#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>

namespace nllb {

class CacheContainer {
public:
    CacheContainer() = default;
    ~CacheContainer() = default;

    void initialize(Ort::Session& cache_init_session,
                   const Ort::MemoryInfo& memory_info,
                   const std::vector<float>& encoder_hidden_states,
                   const std::vector<int64_t>& encoder_attention_mask);

    void addCacheToInputs(std::vector<Ort::Value>& inputs) const;
    void updateCache(std::vector<Ort::Value>& new_cache);
    void reset();

private:
    std::vector<Ort::Value> past_key_values_;
    bool is_initialized_{false};
};

} // namespace nllb 
#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>

namespace nllb {

class CacheContainer {
public:
    CacheContainer() = default;
    ~CacheContainer() = default;

    void initialize(Ort::Session& session,
                   const Ort::MemoryInfo& memory_info,
                   const std::vector<float>& encoder_output,
                   const std::vector<int64_t>& encoder_shape);

    void update(Ort::Value&& new_key_cache, Ort::Value&& new_value_cache);
    
    Ort::Value&& getKeyCache() { return std::move(key_cache_); }
    Ort::Value&& getValueCache() { return std::move(value_cache_); }

private:
    Ort::Value key_cache_{nullptr};
    Ort::Value value_cache_{nullptr};
};

} // namespace nllb 
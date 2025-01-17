#include "cache_container.h"
#include "tensor_utils.h"
#include <stdexcept>

namespace nllb {

void CacheContainer::initialize(
    Ort::Session& cache_init_session,
    const Ort::MemoryInfo& memory_info,
    const std::vector<float>& encoder_output,
    const std::vector<int64_t>& encoder_shape) {
    
    try {
        // Create input tensor
        auto input_tensor = TensorUtils::createTensor<float>(
            memory_info,
            encoder_output,
            encoder_shape
        );
        
        // Run cache initialization
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(input_tensor));
        
        const char* input_names[] = {"encoder_output"};
        const char* output_names[] = {"key_values"};
        
        auto outputs = cache_init_session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            inputs.data(),
            inputs.size(),
            output_names,
            1
        );
        
        // Store cache
        cache_values_ = std::move(outputs);
        
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("Failed to initialize cache: " + std::string(e.what()));
    }
}

void CacheContainer::add_cache_to_inputs(std::vector<Ort::Value>& inputs) const {
    if (!cache_values_.empty()) {
        for (auto& cache_value : cache_values_) {
            inputs.push_back(std::move(cache_value));
        }
    }
}

void CacheContainer::update_cache(std::vector<Ort::Value>&& new_cache) {
    cache_values_ = std::move(new_cache);
}

const std::vector<Ort::Value>& CacheContainer::get_cache() const {
    return cache_values_;
}

void CacheContainer::clear() {
    cache_values_.clear();
}

} // namespace nllb 
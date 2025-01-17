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
        auto input_tensor = TensorUtils::createFloatTensor(
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
        cache_ = std::move(outputs);
        
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("Failed to initialize cache: " + std::string(e.what()));
    }
}

void CacheContainer::add_cache_to_inputs(std::vector<Ort::Value>& inputs) const {
    if (!cache_.empty()) {
        inputs.insert(inputs.end(), cache_.begin(), cache_.end());
    }
}

void CacheContainer::update_cache(const Ort::Value& new_cache) {
    cache_.clear();
    cache_.push_back(new_cache);
}

} // namespace nllb 
#include "cache_container.h"
#include "tensor_utils.h"
#include <stdexcept>

namespace nllb {

CacheContainer::CacheContainer() : has_cache_(false) {}

CacheContainer::~CacheContainer() = default;

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

void CacheContainer::add_cache_to_inputs(std::vector<Ort::Value>& input_tensors) const {
    if (has_cache_) {
        for (const auto& cache : cache_values_) {
            input_tensors.push_back(cache);
        }
    }
}

void CacheContainer::update_cache(std::vector<Ort::Value>& output_tensors) {
    cache_values_.clear();
    for (auto& tensor : output_tensors) {
        cache_values_.push_back(std::move(tensor));
    }
    has_cache_ = true;
}

const std::vector<Ort::Value>& CacheContainer::get_cache() const {
    return cache_values_;
}

void CacheContainer::clear() {
    cache_values_.clear();
    has_cache_ = false;
}

void CacheContainer::reset() {
    clear();
}

} // namespace nllb 
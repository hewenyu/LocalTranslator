#include "cache_container.h"
#include <stdexcept>

namespace nllb {

void CacheContainer::initialize(Ort::Session& cache_init_session,
                              const Ort::MemoryInfo& memory_info,
                              const std::vector<float>& encoder_hidden_states,
                              const std::vector<int64_t>& encoder_attention_mask) {
    // Create input tensors
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info,
        encoder_hidden_states.data(),
        encoder_hidden_states.size(),
        {1, static_cast<int64_t>(encoder_attention_mask.size()), 1024}));
    
    inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
        memory_info,
        encoder_attention_mask.data(),
        encoder_attention_mask.size(),
        {1, static_cast<int64_t>(encoder_attention_mask.size())}));

    // Run cache initialization
    const char* input_names[] = {"encoder_hidden_states", "encoder_attention_mask"};
    const char* output_names[] = {"past_key_values.0.key", "past_key_values.0.value"};
    
    auto outputs = cache_init_session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        inputs.data(),
        inputs.size(),
        output_names,
        2);

    // Store cache values
    past_key_values_ = std::move(outputs);
    is_initialized_ = true;
}

void CacheContainer::addCacheToInputs(std::vector<Ort::Value>& inputs) const {
    if (!is_initialized_) {
        throw std::runtime_error("Cache not initialized");
    }
    
    for (const auto& cache : past_key_values_) {
        inputs.push_back(cache);
    }
}

void CacheContainer::updateCache(std::vector<Ort::Value>& new_cache) {
    past_key_values_ = std::move(new_cache);
}

void CacheContainer::reset() {
    past_key_values_.clear();
    is_initialized_ = false;
}

} // namespace nllb 
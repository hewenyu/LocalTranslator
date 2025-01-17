#include "cache_container.h"
#include "tensor_utils.h"

namespace nllb {

void CacheContainer::initializeCache(const Ort::MemoryInfo& memory_info,
                                   const std::vector<int64_t>& shape) {
    memory_info_ = memory_info;
    shape_ = shape;
    cache_.clear();

    // Initialize cache tensors with zeros
    std::vector<float> zeros(shape_[0] * shape_[1] * shape_[2] * shape_[3], 0.0f);
    
    // Create key and value tensors for each layer
    for (int i = 0; i < 12; ++i) { // 12 layers in NLLB model
        // Key tensor
        cache_.push_back(Ort::Value::CreateTensor<float>(
            memory_info_, zeros.data(), zeros.size(), shape_.data(), shape_.size()
        ));
        
        // Value tensor
        cache_.push_back(Ort::Value::CreateTensor<float>(
            memory_info_, zeros.data(), zeros.size(), shape_.data(), shape_.size()
        ));
    }
}

void CacheContainer::updateCache(std::vector<Ort::Value>& new_cache) {
    if (new_cache.size() != cache_.size()) {
        throw std::runtime_error("Cache size mismatch in update");
    }
    cache_ = std::move(new_cache);
}

} // namespace nllb 
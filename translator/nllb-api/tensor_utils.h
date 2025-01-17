#pragma once

#include <vector>
#include <onnxruntime_cxx_api.h>

namespace nllb {
namespace TensorUtils {

// Create int64 tensor from vector
inline Ort::Value createInt64Tensor(
    const Ort::MemoryInfo& memory_info,
    const std::vector<int64_t>& data,
    const std::vector<int64_t>& shape) {
    
    return Ort::Value::CreateTensor<int64_t>(
        memory_info,
        const_cast<int64_t*>(data.data()),
        data.size(),
        shape.data(),
        shape.size()
    );
}

// Create float tensor from vector
inline Ort::Value createFloatTensor(
    const Ort::MemoryInfo& memory_info,
    const std::vector<float>& data,
    const std::vector<int64_t>& shape) {
    
    return Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(data.data()),
        data.size(),
        shape.data(),
        shape.size()
    );
}

// Get tensor data as vector
template<typename T>
inline std::vector<T> getTensorData(const Ort::Value& tensor) {
    auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
    size_t elem_count = tensor_info.GetElementCount();
    std::vector<T> result(elem_count);
    
    std::copy_n(tensor.GetTensorData<T>(), elem_count, result.begin());
    return result;
}

// Get tensor shape
inline std::vector<int64_t> getTensorShape(const Ort::Value& tensor) {
    auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}

} // namespace TensorUtils
} // namespace nllb 
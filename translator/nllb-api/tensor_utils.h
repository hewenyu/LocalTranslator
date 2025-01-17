#pragma once

#include <vector>
#include <onnxruntime_cxx_api.h>

namespace nllb {
namespace TensorUtils {

template<typename T>
Ort::Value createTensor(
    const Ort::MemoryInfo& memory_info,
    const std::vector<T>& data,
    const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<T>(
        memory_info,
        const_cast<T*>(data.data()),
        data.size(),
        shape.data(),
        shape.size()
    );
}

// 专门用于int64_t类型的tensor创建函数
inline Ort::Value createInt64Tensor(
    const Ort::MemoryInfo& memory_info,
    const std::vector<int64_t>& data,
    const std::vector<int64_t>& shape) {
    return createTensor<int64_t>(memory_info, data, shape);
}

template<typename T>
std::vector<T> getTensorData(const Ort::Value& tensor) {
    auto* data = tensor.GetTensorData<T>();
    size_t size = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
    return std::vector<T>(data, data + size);
}

} // namespace TensorUtils
} // namespace nllb 
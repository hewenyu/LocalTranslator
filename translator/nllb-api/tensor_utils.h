#pragma once

#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace nllb {

class TensorUtils {
public:
    // 创建优化的 float tensor
    static Ort::Value createFloatTensor(
        const Ort::MemoryInfo& memory_info,
        const std::vector<float>& data,
        const std::vector<int64_t>& shape) {
            
        return Ort::Value::CreateTensor<float>(
            const_cast<Ort::MemoryInfo&>(memory_info),
            const_cast<float*>(data.data()),
            data.size(),
            shape.data(),
            shape.size()
        );
    }

    // 创建优化的 int64 tensor
    static Ort::Value createInt64Tensor(
        const Ort::MemoryInfo& memory_info,
        const std::vector<int64_t>& data,
        const std::vector<int64_t>& shape) {
            
        return Ort::Value::CreateTensor<int64_t>(
            const_cast<Ort::MemoryInfo&>(memory_info),
            const_cast<int64_t*>(data.data()),
            data.size(),
            shape.data(),
            shape.size()
        );
    }

    // 从 tensor 获取数据
    template<typename T>
    static std::vector<T> getTensorData(const Ort::Value& tensor) {
        auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
        size_t total_elements = tensor_info.GetElementCount();
        std::vector<T> result(total_elements);
        
        memcpy(result.data(), tensor.GetTensorData<T>(), 
               total_elements * sizeof(T));
               
        return result;
    }

    // 获取 tensor 的形状
    static std::vector<int64_t> getTensorShape(const Ort::Value& tensor) {
        auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
        return tensor_info.GetShape();
    }
};

} // namespace nllb 
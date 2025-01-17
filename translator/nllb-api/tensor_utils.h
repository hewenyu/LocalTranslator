#pragma once

#include <vector>
#include <onnxruntime_cxx_api.h>

namespace nllb {

class TensorUtils {
public:
    template<typename T>
    static Ort::Value createTensor(
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

    static Ort::Value createInt64Tensor(
        const Ort::MemoryInfo& memory_info,
        const std::vector<int64_t>& data,
        const std::vector<int64_t>& shape) {
        
        return createTensor<int64_t>(memory_info, data, shape);
    }

    static Ort::Value createFloatTensor(
        const Ort::MemoryInfo& memory_info,
        const std::vector<float>& data,
        const std::vector<int64_t>& shape) {
        
        return createTensor<float>(memory_info, data, shape);
    }

    template<typename T>
    static std::vector<T> getTensorData(const Ort::Value& tensor) {
        auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
        size_t total_elements = tensor_info.GetElementCount();
        std::vector<T> data(total_elements);
        
        const T* tensor_data = tensor.GetTensorData<T>();
        std::copy(tensor_data, tensor_data + total_elements, data.begin());
        
        return data;
    }

    static std::vector<int64_t> getTensorShape(const Ort::Value& tensor) {
        auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
        return tensor_info.GetShape();
    }
};

} // namespace nllb 
#pragma once

#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "tensor_utils.h"

namespace nllb {

class CacheContainer {
public:
    CacheContainer() = default;
    ~CacheContainer() = default;

    // 初始化缓存
    void initialize(Ort::Session& cache_init_session,
                   const Ort::MemoryInfo& memory_info,
                   const std::vector<float>& encoder_output,
                   const std::vector<int64_t>& encoder_shape) {
        
        // 创建编码器输出 tensor
        auto encoder_tensor = TensorUtils::createFloatTensor(
            memory_info, encoder_output, encoder_shape);

        // 准备输入
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(encoder_tensor));

        // 运行缓存初始化
        const char* input_names[] = {"encoder_output"};
        const char* output_names[] = {"key_cache", "value_cache"};

        auto output_tensors = cache_init_session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            ort_inputs.data(),
            ort_inputs.size(),
            output_names,
            2
        );

        // 保存 key 和 value cache
        key_cache = std::move(output_tensors[0]);
        value_cache = std::move(output_tensors[1]);
    }

    // 更新缓存
    void update(Ort::Value&& new_key, Ort::Value&& new_value) {
        key_cache = std::move(new_key);
        value_cache = std::move(new_value);
    }

    // 获取缓存
    const Ort::Value& getKeyCache() const { return key_cache; }
    const Ort::Value& getValueCache() const { return value_cache; }

private:
    Ort::Value key_cache{nullptr};
    Ort::Value value_cache{nullptr};
};

} // namespace nllb 
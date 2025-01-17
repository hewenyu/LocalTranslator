#include "cache_container.h"
#include "tensor_utils.h"

namespace nllb {

void CacheContainer::initialize(Ort::Session& session,
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

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        ort_inputs.data(),
        ort_inputs.size(),
        output_names,
        2
    );

    // 保存 key 和 value cache
    key_cache_ = std::move(output_tensors[0]);
    value_cache_ = std::move(output_tensors[1]);
}

void CacheContainer::update(Ort::Value&& new_key_cache, Ort::Value&& new_value_cache) {
    key_cache_ = std::move(new_key_cache);
    value_cache_ = std::move(new_value_cache);
}

} // namespace nllb 
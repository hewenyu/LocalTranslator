#pragma once

namespace nllb {

struct ModelParams {
    int beam_size{5};
    int max_length{128};
    float length_penalty{1.0f};
    float temperature{1.0f};
    float top_k{0};
    float top_p{0.9f};
    float repetition_penalty{0.9f};
    int num_threads{4};
    int hidden_size{1024};
    bool support_low_quality_languages{false};
};

} // namespace nllb 
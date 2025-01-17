#pragma once

#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "cache_container.h"

namespace nllb {

struct BeamHypothesis {
    std::vector<int64_t> tokens;
    float score;
    bool is_done;
};

struct ModelParams;  // Forward declaration

class BeamSearchDecoder {
public:
    BeamSearchDecoder(int beam_size, float length_penalty, int64_t eos_token_id);
    
    std::vector<BeamHypothesis> decode(
        Ort::Session& decoder_session,
        Ort::Session& embed_lm_head_session,
        const Ort::MemoryInfo& memory_info,
        const std::vector<float>& encoder_output,
        const std::vector<int64_t>& encoder_shape,
        CacheContainer& cache_container,
        const ModelParams& params) const;

private:
    std::vector<float> compute_next_token_scores(
        const std::vector<float>& logits,
        const std::vector<int64_t>& current_tokens,
        float temperature,
        float repetition_penalty,
        float top_k,
        float top_p) const;

    int beam_size_;
    float length_penalty_;
    int64_t eos_token_id_;
};

} // namespace nllb 
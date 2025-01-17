#pragma once

#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "cache_container.h"

namespace nllb {

struct ModelParams {
    int beam_size = 5;
    int max_length = 128;
    float length_penalty = 1.0f;
    float temperature = 1.0f;
    int top_k = 0;
    float top_p = 0.9f;
    float repetition_penalty = 1.0f;
    int num_threads = 4;
    bool support_low_quality_languages = false;
};

class BeamHypothesis {
public:
    std::vector<int64_t> tokens;
    float score;
    bool done;

    BeamHypothesis() : score(0.0f), done(false) {}
};

class BeamSearchDecoder {
public:
    BeamSearchDecoder() = default;
    ~BeamSearchDecoder() = default;

    std::vector<BeamHypothesis> decode(
        Ort::Session& decoder_session,
        Ort::Session& embed_session,
        const Ort::MemoryInfo& memory_info,
        const std::vector<float>& encoder_output,
        const std::vector<int64_t>& encoder_attention_mask,
        CacheContainer& cache,
        const ModelParams& params) const;

private:
    std::vector<float> compute_next_token_scores(
        const std::vector<int64_t>& prev_tokens,
        const std::vector<BeamHypothesis>& hypotheses,
        const ModelParams& params) const;
};

} // namespace nllb 
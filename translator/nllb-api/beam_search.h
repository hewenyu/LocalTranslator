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
    float top_k = 0;
    float top_p = 0.9f;
    float repetition_penalty = 1.0f;
    int num_threads = 4;
    int hidden_size = 1024;
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
    BeamSearchDecoder(int beam_size, float length_penalty, int64_t eos_token_id)
        : beam_size_(beam_size)
        , length_penalty_(length_penalty)
        , eos_token_id_(eos_token_id) {}
    
    ~BeamSearchDecoder() = default;

    std::vector<BeamHypothesis> decode(
        Ort::Session& decoder_session,
        Ort::Session& embed_lm_head_session,
        const Ort::MemoryInfo& memory_info,
        const std::vector<float>& encoder_output,
        const std::vector<int64_t>& encoder_shape,
        CacheContainer& cache,
        const ModelParams& params) const;

private:
    int beam_size_;
    float length_penalty_;
    int64_t eos_token_id_;

    // Helper methods
    std::vector<float> compute_next_token_scores(
        Ort::Session& embed_lm_head_session,
        const Ort::MemoryInfo& memory_info,
        const std::vector<int64_t>& prev_tokens,
        const std::vector<BeamHypothesis>& hypotheses,
        const ModelParams& params) const;

    float compute_sequence_score(
        const std::vector<int64_t>& sequence,
        float raw_score) const;

    std::vector<size_t> get_top_k_indices(
        const std::vector<float>& scores,
        size_t k) const;
};

} // namespace nllb 
#pragma once

#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "cache_container.h"
#include "tokenizer.h"

namespace nllb {

struct BeamHypothesis {
    std::vector<int64_t> tokens;
    float score;
    bool is_done;

    BeamHypothesis() : score(0.0f), is_done(false) {}
};

class BeamSearchDecoder {
public:
    BeamSearchDecoder(int beam_size, int max_length, float length_penalty,
                     float temperature, float top_k, float top_p,
                     float repetition_penalty);

    std::vector<BeamHypothesis> decode(
        Ort::Session* decoder_session,
        const Ort::MemoryInfo& memory_info,
        const std::vector<float>& encoder_output,
        CacheContainer& cache_container,
        const TokenizerResult& input_tokens,
        int eos_token_id) const;

private:
    int beam_size_;
    int max_length_;
    float length_penalty_;
    float temperature_;
    float top_k_;
    float top_p_;
    float repetition_penalty_;

    std::vector<float> apply_repetition_penalty(
        std::vector<float>& scores,
        const std::vector<int64_t>& input_ids) const;

    std::vector<float> apply_temperature(
        std::vector<float>& scores) const;

    std::vector<float> apply_top_k_top_p(
        std::vector<float>& scores) const;

    float compute_sequence_score(
        const std::vector<int64_t>& sequence) const;
};

} // namespace nllb 
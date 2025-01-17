#pragma once

#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace nllb {

struct BeamSearchConfig {
    int beam_size;
    int max_length;
    float length_penalty;
    float eos_penalty;
    int num_return_sequences;
    float temperature;
    float top_k;
    float top_p;
    float repetition_penalty;
    bool use_cache;

    BeamSearchConfig(
        int beam_size = 5,
        int max_length = 128,
        float length_penalty = 1.0f,
        float eos_penalty = 0.9f,
        int num_return_sequences = 1,
        float temperature = 1.0f,
        float top_k = 0,
        float top_p = 0.9f,
        float repetition_penalty = 0.9f,
        bool use_cache = true
    ) : beam_size(beam_size),
        max_length(max_length),
        length_penalty(length_penalty),
        eos_penalty(eos_penalty),
        num_return_sequences(num_return_sequences),
        temperature(temperature),
        top_k(top_k),
        top_p(top_p),
        repetition_penalty(repetition_penalty),
        use_cache(use_cache) {}
};

struct BeamSearchState {
    size_t source_length;
    Ort::Value encoder_output;
    Ort::Value encoder_attention_mask;
    std::vector<int64_t> current_tokens;
    std::vector<float> current_scores;
    std::vector<bool> is_finished;
    std::vector<std::vector<int64_t>> finished_sequences;
    std::vector<float> finished_scores;
    size_t num_beams;
    size_t vocab_size;
    size_t cur_len;
    bool use_cache;

    explicit BeamSearchState(const BeamSearchConfig& config)
        : num_beams(config.beam_size),
          cur_len(0),
          use_cache(config.use_cache) {}
};

struct BeamSearchResult {
    std::vector<int64_t> output_ids;
    float score;
};

class BeamSearchDecoder {
public:
    explicit BeamSearchDecoder(const BeamSearchConfig& config);
    BeamSearchResult decode(BeamSearchState& state);

private:
    BeamSearchConfig config_;
    
    void initialize_beams(BeamSearchState& state);
    void expand_beams(BeamSearchState& state);
    void finalize_beams(BeamSearchState& state);
    
    std::vector<float> compute_next_token_scores(
        const std::vector<float>& scores,
        const std::vector<int64_t>& prev_tokens
    );
    
    std::vector<size_t> get_top_k_indices(
        const std::vector<float>& scores,
        size_t k
    );
};

} // namespace nllb 
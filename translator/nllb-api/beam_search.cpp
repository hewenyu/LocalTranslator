#include "beam_search.h"
#include "tensor_utils.h"
#include <algorithm>
#include <cmath>
#include <queue>

namespace nllb {

BeamSearchDecoder::BeamSearchDecoder(
    int beam_size,
    float length_penalty,
    int64_t eos_token_id)
    : beam_size_(beam_size)
    , length_penalty_(length_penalty)
    , eos_token_id_(eos_token_id) {}

std::vector<BeamHypothesis> BeamSearchDecoder::decode(
    Ort::Session& decoder_session,
    Ort::Session& embed_lm_head_session,
    const Ort::MemoryInfo& memory_info,
    const std::vector<float>& encoder_output,
    const std::vector<int64_t>& encoder_shape,
    CacheContainer& cache_container,
    const ModelParams& params) const {
    
    std::vector<BeamHypothesis> hypotheses(beam_size_);
    std::vector<std::vector<int64_t>> beam_tokens(beam_size_, std::vector<int64_t>{});
    std::vector<float> beam_scores(beam_size_, 0.0f);
    
    // Initialize with start token
    for (int i = 0; i < beam_size_; ++i) {
        beam_tokens[i].push_back(tokenizer_->bos_id());
    }
    
    bool is_done = false;
    int cur_len = 1;
    
    while (!is_done && cur_len < params.max_length) {
        std::vector<int64_t> next_token_logits;
        
        // Run decoder step
        {
            auto input_tensor = TensorUtils::createInt64Tensor(
                memory_info,
                beam_tokens.back(),
                {1, static_cast<int64_t>(beam_tokens.back().size())}
            );
            
            std::vector<Ort::Value> decoder_inputs;
            decoder_inputs.push_back(std::move(input_tensor));
            decoder_inputs.push_back(TensorUtils::createFloatTensor(
                memory_info,
                encoder_output,
                encoder_shape
            ));
            
            cache_container.addCacheToInputs(decoder_inputs);
            
            const char* input_names[] = {"input_ids", "encoder_output", "past_key_values"};
            const char* output_names[] = {"logits", "present_key_values"};
            
            auto outputs = decoder_session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                decoder_inputs.data(),
                decoder_inputs.size(),
                output_names,
                2
            );
            
            next_token_logits = TensorUtils::getTensorData<float>(outputs[0]);
            cache_container.updateCache(outputs[1]);
        }
        
        // Get scores for next tokens
        auto next_token_scores = compute_next_token_scores(
            next_token_logits,
            beam_tokens.back(),
            params.temperature,
            params.repetition_penalty,
            params.top_k,
            params.top_p
        );
        
        // Get top-k tokens and their scores
        std::vector<std::pair<float, int64_t>> token_scores;
        for (size_t i = 0; i < next_token_scores.size(); ++i) {
            token_scores.emplace_back(next_token_scores[i], i);
        }
        
        std::partial_sort(
            token_scores.begin(),
            token_scores.begin() + beam_size_,
            token_scores.end(),
            std::greater<>()
        );
        
        // Update beam hypotheses
        std::vector<std::vector<int64_t>> next_beam_tokens;
        std::vector<float> next_beam_scores;
        
        for (int i = 0; i < beam_size_; ++i) {
            auto [score, token] = token_scores[i];
            auto new_tokens = beam_tokens[i];
            new_tokens.push_back(token);
            
            float sequence_score = beam_scores[i] + score;
            
            if (token == eos_token_id_) {
                float normalized_score = sequence_score / std::pow(new_tokens.size(), length_penalty_);
                hypotheses[i] = {new_tokens, normalized_score, true};
            } else {
                next_beam_tokens.push_back(std::move(new_tokens));
                next_beam_scores.push_back(sequence_score);
            }
        }
        
        // Check if all hypotheses are done
        is_done = std::all_of(
            hypotheses.begin(),
            hypotheses.end(),
            [](const auto& h) { return h.is_done; }
        );
        
        if (!is_done) {
            beam_tokens = std::move(next_beam_tokens);
            beam_scores = std::move(next_beam_scores);
            ++cur_len;
        }
    }
    
    // Normalize scores for incomplete sequences
    for (size_t i = 0; i < hypotheses.size(); ++i) {
        if (!hypotheses[i].is_done) {
            hypotheses[i].tokens = beam_tokens[i];
            hypotheses[i].score = beam_scores[i] / std::pow(beam_tokens[i].size(), length_penalty_);
            hypotheses[i].is_done = true;
        }
    }
    
    return hypotheses;
}

std::vector<float> BeamSearchDecoder::compute_next_token_scores(
    const std::vector<float>& logits,
    const std::vector<int64_t>& current_tokens,
    float temperature,
    float repetition_penalty,
    float top_k,
    float top_p) const {
    
    std::vector<float> scores = logits;
    
    // Apply repetition penalty
    for (auto token : current_tokens) {
        if (token >= 0 && token < scores.size()) {
            scores[token] /= repetition_penalty;
        }
    }
    
    // Apply temperature
    if (temperature != 1.0f) {
        for (auto& score : scores) {
            score /= temperature;
        }
    }
    
    // Apply softmax
    float max_score = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    
    for (auto& score : scores) {
        score = std::exp(score - max_score);
        sum += score;
    }
    
    for (auto& score : scores) {
        score /= sum;
    }
    
    // Apply top-k filtering
    if (top_k > 0 && top_k < scores.size()) {
        std::vector<float> sorted_scores = scores;
        std::nth_element(
            sorted_scores.begin(),
            sorted_scores.begin() + top_k,
            sorted_scores.end(),
            std::greater<>()
        );
        
        float min_score = sorted_scores[top_k];
        for (auto& score : scores) {
            if (score < min_score) {
                score = 0.0f;
            }
        }
    }
    
    // Apply top-p filtering
    if (top_p < 1.0f) {
        std::vector<std::pair<float, size_t>> sorted_scores;
        sorted_scores.reserve(scores.size());
        
        for (size_t i = 0; i < scores.size(); ++i) {
            if (scores[i] > 0) {
                sorted_scores.emplace_back(scores[i], i);
            }
        }
        
        std::sort(
            sorted_scores.begin(),
            sorted_scores.end(),
            std::greater<>()
        );
        
        float cumsum = 0.0f;
        size_t last_idx = sorted_scores.size();
        
        for (size_t i = 0; i < sorted_scores.size(); ++i) {
            cumsum += sorted_scores[i].first;
            if (cumsum > top_p) {
                last_idx = i + 1;
                break;
            }
        }
        
        // Zero out scores below the top-p threshold
        std::vector<float> filtered_scores(scores.size(), 0.0f);
        for (size_t i = 0; i < last_idx; ++i) {
            filtered_scores[sorted_scores[i].second] = scores[sorted_scores[i].second];
        }
        scores = std::move(filtered_scores);
    }
    
    return scores;
}

} // namespace nllb 
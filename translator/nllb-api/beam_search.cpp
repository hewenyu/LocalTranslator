#include "beam_search.h"
#include "tensor_utils.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <numeric>

namespace nllb {

std::vector<BeamHypothesis> BeamSearchDecoder::decode(
    Ort::Session& decoder_session,
    Ort::Session& embed_lm_head_session,
    const Ort::MemoryInfo& memory_info,
    const std::vector<float>& encoder_output,
    const std::vector<int64_t>& encoder_shape,
    CacheContainer& cache,
    const ModelParams& params) const {
    
    std::vector<BeamHypothesis> hypotheses;
    hypotheses.reserve(beam_size_);
    
    // Initialize hypotheses
    for (int i = 0; i < beam_size_; ++i) {
        BeamHypothesis hyp;
        hyp.tokens = std::vector<int64_t>{};
        hyp.score = 0.0f;
        hyp.done = false;
        hypotheses.push_back(std::move(hyp));
    }
    
    std::vector<std::vector<int64_t>> beam_tokens(beam_size_);
    std::vector<float> beam_scores(beam_size_, 0.0f);
    
    // Initialize beams with start token
    for (auto& tokens : beam_tokens) {
        tokens.push_back(eos_token_id_);  // Use EOS as start token
    }
    
    bool is_done = false;
    int cur_len = 1;
    
    while (!is_done && cur_len < params.max_length) {
        // Get next token scores
        auto next_scores = compute_next_token_scores(
            embed_lm_head_session,
            memory_info,
            beam_tokens.back(),
            hypotheses,
            params
        );
        
        // Get top-k tokens and their scores
        auto top_indices = get_top_k_indices(next_scores, beam_size_);
        
        // Update beam hypotheses
        std::vector<std::vector<int64_t>> next_beam_tokens;
        std::vector<float> next_beam_scores;
        
        for (size_t i = 0; i < top_indices.size(); ++i) {
            int64_t token = static_cast<int64_t>(top_indices[i]);
            float score = next_scores[token];
            
            auto new_tokens = beam_tokens[i % beam_size_];
            new_tokens.push_back(token);
            
            float sequence_score = beam_scores[i % beam_size_] + score;
            
            if (token == eos_token_id_) {
                hypotheses[i % beam_size_] = {
                    new_tokens,
                    compute_sequence_score(new_tokens, sequence_score),
                    true
                };
            } else {
                next_beam_tokens.push_back(std::move(new_tokens));
                next_beam_scores.push_back(sequence_score);
            }
        }
        
        // Check if all hypotheses are done
        is_done = std::all_of(hypotheses.begin(), hypotheses.end(),
                            [](const auto& h) { return h.done; });
        
        if (!is_done) {
            beam_tokens = std::move(next_beam_tokens);
            beam_scores = std::move(next_beam_scores);
            ++cur_len;
        }
    }
    
    // Finalize incomplete hypotheses
    for (size_t i = 0; i < hypotheses.size(); ++i) {
        if (!hypotheses[i].done) {
            hypotheses[i].tokens = beam_tokens[i];
            hypotheses[i].score = compute_sequence_score(beam_tokens[i], beam_scores[i]);
            hypotheses[i].done = true;
        }
    }
    
    return hypotheses;
}

std::vector<float> BeamSearchDecoder::compute_next_token_scores(
    Ort::Session& embed_lm_head_session,
    const Ort::MemoryInfo& memory_info,
    const std::vector<int64_t>& prev_tokens,
    const std::vector<BeamHypothesis>& hypotheses,
    const ModelParams& params) const {
    
    // Get logits from embed_lm_head model
    auto input_tensor = TensorUtils::createTensor<int64_t>(
        memory_info,
        prev_tokens,
        {1, static_cast<int64_t>(prev_tokens.size())}
    );
    
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(input_tensor));
    
    const char* input_names[] = {"input_ids"};
    const char* output_names[] = {"logits"};
    
    auto outputs = embed_lm_head_session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        inputs.data(),
        inputs.size(),
        output_names,
        1
    );
    
    auto logits = TensorUtils::getTensorData<float>(outputs[0]);
    
    // Apply temperature
    if (params.temperature != 1.0f) {
        for (auto& logit : logits) {
            logit /= params.temperature;
        }
    }
    
    // Apply repetition penalty
    for (const auto& token : prev_tokens) {
        if (token >= 0 && token < logits.size()) {
            logits[token] /= params.repetition_penalty;
        }
    }
    
    // Convert to probabilities
    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size());
    float sum = 0.0f;
    
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    
    for (auto& prob : probs) {
        prob /= sum;
    }
    
    // Apply top-k filtering
    if (params.top_k > 0) {
        size_t k = static_cast<size_t>(params.top_k);
        if (k < probs.size()) {
            std::vector<float> sorted_probs = probs;
            std::nth_element(sorted_probs.begin(),
                           sorted_probs.begin() + k,
                           sorted_probs.end(),
                           std::greater<>());
            float min_prob = sorted_probs[k];
            
            for (auto& prob : probs) {
                if (prob < min_prob) {
                    prob = 0.0f;
                }
            }
        }
    }
    
    // Apply top-p filtering
    if (params.top_p < 1.0f) {
        std::vector<std::pair<float, size_t>> sorted_pairs;
        sorted_pairs.reserve(probs.size());
        
        for (size_t i = 0; i < probs.size(); ++i) {
            if (probs[i] > 0) {
                sorted_pairs.emplace_back(probs[i], i);
            }
        }
        
        std::sort(sorted_pairs.begin(), sorted_pairs.end(),
                 std::greater<>());
        
        float cumsum = 0.0f;
        size_t last_idx = sorted_pairs.size();
        
        for (size_t i = 0; i < sorted_pairs.size(); ++i) {
            cumsum += sorted_pairs[i].first;
            if (cumsum > params.top_p) {
                last_idx = i + 1;
                break;
            }
        }
        
        std::vector<float> filtered_probs(probs.size(), 0.0f);
        for (size_t i = 0; i < last_idx; ++i) {
            filtered_probs[sorted_pairs[i].second] = probs[sorted_pairs[i].second];
        }
        probs = std::move(filtered_probs);
    }
    
    return probs;
}

float BeamSearchDecoder::compute_sequence_score(
    const std::vector<int64_t>& sequence,
    float raw_score) const {
    return raw_score / std::pow(sequence.size(), length_penalty_);
}

std::vector<size_t> BeamSearchDecoder::get_top_k_indices(
    const std::vector<float>& scores,
    size_t k) const {
    
    std::vector<size_t> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::partial_sort(
        indices.begin(),
        indices.begin() + k,
        indices.end(),
        [&scores](size_t i1, size_t i2) {
            return scores[i1] > scores[i2];
        }
    );
    
    indices.resize(k);
    return indices;
}

} // namespace nllb 
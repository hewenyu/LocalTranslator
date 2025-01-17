#include "translator/nllb-api/beam_search.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <spdlog/spdlog.h>

namespace nllb {

BeamSearchDecoder::BeamSearchDecoder(const BeamSearchConfig& config)
    : config_(config) {}

BeamSearchResult BeamSearchDecoder::decode(BeamSearchState& state) {
    initialize_beams(state);
    
    while (state.cur_len < config_.max_length) {
        expand_beams(state);
        
        // Check if all beams are finished
        bool all_finished = true;
        for (bool finished : state.is_finished) {
            if (!finished) {
                all_finished = false;
                break;
            }
        }
        if (all_finished) break;
        
        state.cur_len++;
    }
    
    finalize_beams(state);
    
    // Return best sequence
    BeamSearchResult result;
    if (!state.finished_sequences.empty()) {
        size_t best_idx = 0;
        float best_score = state.finished_scores[0];
        for (size_t i = 1; i < state.finished_scores.size(); i++) {
            if (state.finished_scores[i] > best_score) {
                best_score = state.finished_scores[i];
                best_idx = i;
            }
        }
        result.output_ids = state.finished_sequences[best_idx];
        result.score = best_score;
    } else {
        // If no sequence finished, return the current best beam
        result.output_ids = state.current_tokens;
        result.score = state.current_scores[0];
    }
    
    return result;
}

void BeamSearchDecoder::initialize_beams(BeamSearchState& state) {
    state.current_tokens.clear();
    state.current_scores.resize(state.num_beams, 0.0f);
    state.is_finished.resize(state.num_beams, false);
    state.finished_sequences.clear();
    state.finished_scores.clear();
}

void BeamSearchDecoder::expand_beams(BeamSearchState& state) {
    // Get scores for next tokens
    std::vector<float> next_token_scores = compute_next_token_scores(
        state.current_scores,
        state.current_tokens
    );
    
    // Apply temperature
    if (config_.temperature != 1.0f) {
        for (float& score : next_token_scores) {
            score /= config_.temperature;
        }
    }
    
    // Apply top-k filtering
    if (config_.top_k > 0) {
        size_t k = std::min(static_cast<size_t>(config_.top_k), next_token_scores.size());
        auto top_k_indices = get_top_k_indices(next_token_scores, k);
        std::vector<float> filtered_scores(next_token_scores.size(), -INFINITY);
        for (size_t idx : top_k_indices) {
            filtered_scores[idx] = next_token_scores[idx];
        }
        next_token_scores = filtered_scores;
    }
    
    // Apply top-p (nucleus) filtering
    if (config_.top_p < 1.0f) {
        float sum = 0.0f;
        std::vector<std::pair<float, size_t>> score_idx_pairs;
        for (size_t i = 0; i < next_token_scores.size(); i++) {
            if (next_token_scores[i] != -INFINITY) {
                score_idx_pairs.emplace_back(next_token_scores[i], i);
                sum += std::exp(next_token_scores[i]);
            }
        }
        
        std::sort(score_idx_pairs.begin(), score_idx_pairs.end(),
                 std::greater<std::pair<float, size_t>>());
        
        float cumsum = 0.0f;
        std::vector<float> filtered_scores(next_token_scores.size(), -INFINITY);
        for (const auto& pair : score_idx_pairs) {
            float prob = std::exp(pair.first) / sum;
            if (cumsum + prob > config_.top_p) break;
            filtered_scores[pair.second] = pair.first;
            cumsum += prob;
        }
        next_token_scores = filtered_scores;
    }
    
    // Apply repetition penalty
    if (config_.repetition_penalty != 1.0f) {
        for (size_t i = 0; i < state.current_tokens.size(); i++) {
            size_t token = static_cast<size_t>(state.current_tokens[i]);
            if (token < next_token_scores.size()) {
                if (next_token_scores[token] > 0) {
                    next_token_scores[token] /= config_.repetition_penalty;
                } else {
                    next_token_scores[token] *= config_.repetition_penalty;
                }
            }
        }
    }
    
    // Get top beam_size tokens
    auto top_indices = get_top_k_indices(next_token_scores, state.num_beams);
    
    // Update beams
    std::vector<int64_t> new_tokens;
    std::vector<float> new_scores;
    for (size_t idx : top_indices) {
        new_tokens.push_back(static_cast<int64_t>(idx));
        new_scores.push_back(next_token_scores[idx]);
    }
    
    state.current_tokens = new_tokens;
    state.current_scores = new_scores;
}

void BeamSearchDecoder::finalize_beams(BeamSearchState& state) {
    // Apply length penalty
    if (config_.length_penalty != 1.0f) {
        float length_penalty = std::pow((5.0f + state.cur_len) / 6.0f, config_.length_penalty);
        for (float& score : state.current_scores) {
            score /= length_penalty;
        }
    }
    
    // Add EOS penalty
    if (config_.eos_penalty != 1.0f) {
        for (size_t i = 0; i < state.current_tokens.size(); i++) {
            if (state.current_tokens[i] == 2) {  // EOS token
                state.current_scores[i] *= config_.eos_penalty;
            }
        }
    }
}

std::vector<float> BeamSearchDecoder::compute_next_token_scores(
    const std::vector<float>& scores,
    const std::vector<int64_t>& prev_tokens) {
    
    std::vector<float> next_scores(state.vocab_size, -INFINITY);
    
    // Compute logits using decoder
    // This should be implemented based on your model architecture
    
    return next_scores;
}

std::vector<size_t> BeamSearchDecoder::get_top_k_indices(
    const std::vector<float>& scores,
    size_t k) {
    
    std::vector<size_t> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::partial_sort(
        indices.begin(),
        indices.begin() + k,
        indices.end(),
        [&scores](size_t i1, size_t i2) { return scores[i1] > scores[i2]; }
    );
    
    indices.resize(k);
    return indices;
}

} // namespace nllb 
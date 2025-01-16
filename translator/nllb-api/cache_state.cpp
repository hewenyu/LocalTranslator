#include "beam_search.h"
#include <spdlog/spdlog.h>

namespace nllb {

CacheState::CacheState(int max_length, int hidden_size, int num_heads, int num_layers)
    : max_length_(max_length)
    , hidden_size_(hidden_size)
    , num_heads_(num_heads)
    , num_layers_(num_layers) {
    
    try {
        decoder_keys_.resize(num_layers);
        decoder_values_.resize(num_layers);
        encoder_keys_.resize(num_layers);
        encoder_values_.resize(num_layers);
        
        spdlog::debug("Initialized cache state with {} layers", num_layers);
    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize cache state: {}", e.what());
        throw;
    }
}

void CacheState::update_decoder_key(int layer, Ort::Value&& key) {
    if (layer < 0 || layer >= num_layers_) {
        throw std::out_of_range("Layer index out of range");
    }
    decoder_keys_[layer] = std::move(key);
}

void CacheState::update_decoder_value(int layer, Ort::Value&& value) {
    if (layer < 0 || layer >= num_layers_) {
        throw std::out_of_range("Layer index out of range");
    }
    decoder_values_[layer] = std::move(value);
}

void CacheState::update_encoder_key(int layer, Ort::Value&& key) {
    if (layer < 0 || layer >= num_layers_) {
        throw std::out_of_range("Layer index out of range");
    }
    encoder_keys_[layer] = std::move(key);
}

void CacheState::update_encoder_value(int layer, Ort::Value&& value) {
    if (layer < 0 || layer >= num_layers_) {
        throw std::out_of_range("Layer index out of range");
    }
    encoder_values_[layer] = std::move(value);
}

std::optional<Ort::Value>& CacheState::get_decoder_key(int layer) {
    if (layer < 0 || layer >= num_layers_) {
        throw std::out_of_range("Layer index out of range");
    }
    return decoder_keys_[layer];
}

std::optional<Ort::Value>& CacheState::get_decoder_value(int layer) {
    if (layer < 0 || layer >= num_layers_) {
        throw std::out_of_range("Layer index out of range");
    }
    return decoder_values_[layer];
}

std::optional<Ort::Value>& CacheState::get_encoder_key(int layer) {
    if (layer < 0 || layer >= num_layers_) {
        throw std::out_of_range("Layer index out of range");
    }
    return encoder_keys_[layer];
}

std::optional<Ort::Value>& CacheState::get_encoder_value(int layer) {
    if (layer < 0 || layer >= num_layers_) {
        throw std::out_of_range("Layer index out of range");
    }
    return encoder_values_[layer];
}

} // namespace nllb 
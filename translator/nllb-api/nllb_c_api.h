#ifndef NLLB_C_API_H
#define NLLB_C_API_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
typedef enum {
    NLLB_OK = 0,
    NLLB_ERROR_INIT = -1,
    NLLB_ERROR_TOKENIZE = -2,
    NLLB_ERROR_ENCODE = -3,
    NLLB_ERROR_DECODE = -4,
    NLLB_ERROR_MEMORY = -5,
    NLLB_ERROR_INVALID_PARAM = -6
} nllb_error_t;

// Opaque types
typedef struct nllb_translator nllb_translator_t;
typedef struct nllb_tokenizer nllb_tokenizer_t;
typedef struct nllb_beam_search nllb_beam_search_t;

// Model configuration
typedef struct {
    int hidden_size;
    int num_heads;
    int vocab_size;
    int max_position_embeddings;
    int encoder_layers;
    int decoder_layers;
} nllb_model_config_t;

// Beam search configuration
typedef struct {
    int beam_size;
    int max_length;
    float length_penalty;
    float eos_penalty;
    int num_return_sequences;
    float temperature;
    float top_k;
    float top_p;
    float repetition_penalty;
} nllb_beam_config_t;

// Tokenizer output
typedef struct {
    int64_t* input_ids;
    int64_t* attention_mask;
    size_t length;
} nllb_tokenizer_output_t;

// Translator API
nllb_translator_t* nllb_translator_create(const char* model_dir, const char* target_lang, nllb_error_t* error);
void nllb_translator_destroy(nllb_translator_t* translator);
char* nllb_translate(nllb_translator_t* translator, const char* text, const char* source_lang, nllb_error_t* error);

// Tokenizer API
nllb_tokenizer_t* nllb_tokenizer_create(const char* model_path, nllb_error_t* error);
void nllb_tokenizer_destroy(nllb_tokenizer_t* tokenizer);
nllb_tokenizer_output_t* nllb_tokenizer_encode(nllb_tokenizer_t* tokenizer, const char* text, 
                                              const char* source_lang, const char* target_lang, 
                                              nllb_error_t* error);
char* nllb_tokenizer_decode(nllb_tokenizer_t* tokenizer, const int64_t* tokens, size_t length, 
                           nllb_error_t* error);
void nllb_tokenizer_output_free(nllb_tokenizer_output_t* output);

// Beam search API
nllb_beam_search_t* nllb_beam_search_create(const nllb_beam_config_t* config, nllb_error_t* error);
void nllb_beam_search_destroy(nllb_beam_search_t* beam_search);

// Utility functions
const char* nllb_error_string(nllb_error_t error);
void nllb_free_string(char* str);

#ifdef __cplusplus
}
#endif

#endif // NLLB_C_API_H 
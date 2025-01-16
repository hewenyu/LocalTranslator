#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include "translator/nllb-api/nllb_translator.h"

namespace nllb {

NLLBTranslator::NLLBTranslator(const common::TranslatorConfig& config) 
    : ort_env_(ORT_LOGGING_LEVEL_WARNING, "nllb_translator") {
    model_dir_ = config.nllb.model_dir;
    target_lang_ = config.nllb.target_lang;
    params_ = config.nllb.params;
    
    // Initialize components
    initialize_language_codes();
    load_models();
    
    // Initialize tokenizer
    std::string vocab_path = model_dir_ + "/" + config.nllb.model_files.tokenizer_vocab;
    tokenizer_ = std::make_unique<Tokenizer>(vocab_path);
}

NLLBTranslator::~NLLBTranslator() = default;

void NLLBTranslator::initialize_language_codes() {
    // Load language codes from YAML file
    std::string lang_file = model_dir_ + "/nllb_languages.yaml";
    try {
        YAML::Node config = YAML::LoadFile(lang_file);
        for (const auto& lang : config["languages"]) {
            std::string code = lang["code"].as<std::string>();
            std::string nllb_code = lang["code_NLLB"].as<std::string>();
            nllb_language_codes_[code] = nllb_code;
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load language codes: " + std::string(e.what()));
    }
}

void NLLBTranslator::load_models() {
    Ort::SessionOptions session_opts;
    session_opts.SetIntraOpNumThreads(params_.num_threads);
    session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    // Load encoder
    std::string encoder_path = model_dir_ + "/NLLB_encoder.onnx";
    encoder_session_ = std::make_unique<Ort::Session>(ort_env_, 
        encoder_path.c_str(), session_opts);

    // Load decoder
    std::string decoder_path = model_dir_ + "/NLLB_decoder.onnx";
    decoder_session_ = std::make_unique<Ort::Session>(ort_env_,
        decoder_path.c_str(), session_opts);

    // Load embed and lm head
    std::string embed_path = model_dir_ + "/NLLB_embed_and_lm_head.onnx";
    embed_lm_head_session_ = std::make_unique<Ort::Session>(ort_env_,
        embed_path.c_str(), session_opts);

    // Load cache initializer if using cache
    if (params_.use_cache) {
        std::string cache_path = model_dir_ + "/NLLB_cache_initializer.onnx";
        cache_init_session_ = std::make_unique<Ort::Session>(ort_env_,
            cache_path.c_str(), session_opts);
    }
}

std::string NLLBTranslator::get_nllb_language_code(const std::string& lang_code) const {
    auto it = nllb_language_codes_.find(lang_code);
    if (it == nllb_language_codes_.end()) {
        throw std::runtime_error("Unsupported language code: " + lang_code);
    }
    return it->second;
}

std::vector<float> NLLBTranslator::run_encoder(const Tokenizer::TokenizerOutput& tokens) const {
    // Prepare input tensors
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(tokens.input_ids.size())};
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        
    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, tokens.input_ids.data(), tokens.input_ids.size(), 
        input_shape.data(), input_shape.size());
        
    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, tokens.attention_mask.data(), tokens.attention_mask.size(),
        input_shape.data(), input_shape.size());

    // Run encoder
    const char* input_names[] = {"input_ids", "attention_mask"};
    const char* output_names[] = {"encoder_output"};
    
    auto output_tensors = encoder_session_->Run(
        Ort::RunOptions{nullptr},
        input_names, 
        {input_ids_tensor, attention_mask_tensor},
        2,
        output_names,
        1);

    // Get output data
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    
    return std::vector<float>(output_data, output_data + output_size);
}

std::vector<int64_t> NLLBTranslator::run_decoder(
    const std::vector<float>& encoder_output,
    const std::string& target_lang) const {
    
    // 创建beam search配置
    BeamSearchConfig beam_config(
        params_.beam_size,
        params_.max_length,
        params_.length_penalty,
        0.9f,  // EOS penalty
        1      // 返回最好的一个序列
    );

    // 创建beam search解码器
    BeamSearchDecoder decoder(beam_config);

    // 定义单步解码函数
    auto step_function = [this, &encoder_output, &target_lang](
        const std::vector<int64_t>& tokens,
        const CacheState& cache) -> std::vector<float> {
        
        // 准备输入张量
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        // 准备decoder输入
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(tokens.size())};
        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, tokens.data(), tokens.size(), 
            input_shape.data(), input_shape.size());

        // 准备encoder输出
        std::vector<int64_t> encoder_shape = {1, static_cast<int64_t>(encoder_output.size())};
        Ort::Value encoder_tensor = Ort::Value::CreateTensor<float>(
            memory_info, encoder_output.data(), encoder_output.size(),
            encoder_shape.data(), encoder_shape.size());

        // 运行decoder
        const char* input_names[] = {"input_ids", "encoder_output"};
        const char* output_names[] = {"logits"};

        auto output_tensors = decoder_session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            {input_ids_tensor, encoder_tensor},
            2,
            output_names,
            1);

        // 获取logits
        float* logits_data = output_tensors[0].GetTensorMutableData<float>();
        size_t vocab_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[2];
        
        // 计算最后一个token的概率分布
        std::vector<float> scores(logits_data + (tokens.size() - 1) * vocab_size,
                                logits_data + tokens.size() * vocab_size);
        
        // 应用softmax
        float max_score = *std::max_element(scores.begin(), scores.end());
        float sum = 0.0f;
        for (auto& score : scores) {
            score = std::exp(score - max_score);
            sum += score;
        }
        for (auto& score : scores) {
            score /= sum;
        }

        return scores;
    };

    // 执行beam search
    auto hypotheses = decoder.decode(
        step_function,
        tokenizer_->bos_id(),
        tokenizer_->eos_id(),
        tokenizer_->pad_id()
    );

    // 返回最好的序列
    return hypotheses[0].tokens;
}

std::string NLLBTranslator::translate(
    const std::string& text,
    const std::string& source_lang) const {
    try {
        // Convert language codes
        std::string nllb_source = get_nllb_language_code(source_lang);
        std::string nllb_target = get_nllb_language_code(target_lang_);

        // Tokenize input
        auto tokens = tokenizer_->encode(text, nllb_source, nllb_target);

        // Run encoder
        auto encoder_output = run_encoder(tokens);

        // Run decoder
        auto output_ids = run_decoder(encoder_output, nllb_target);

        // Decode output tokens
        return tokenizer_->decode(output_ids);
    } catch (const std::exception& e) {
        throw std::runtime_error("Translation failed: " + std::string(e.what()));
    }
}

std::string NLLBTranslator::get_target_language() const {
    return target_lang_;
}

} // namespace nllb 
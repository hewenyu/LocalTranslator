add_library(translator STATIC
    translator.cpp
    deeplx/deeplx_translator.cpp
    nllb-api/nllb_translator.cpp
    nllb-api/beam_search.cpp
    nllb-api/tokenizer.cpp
    nllb-api/language_detector.cpp
)

target_include_directories(translator
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${ONNXRUNTIME_ROOT}/include
    ${CMAKE_SOURCE_DIR}/dep/sentencepiece/src
)

target_link_libraries(translator
    PUBLIC
    common
    yaml-cpp::yaml-cpp
    CURL::libcurl
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    tinyxml2::tinyxml2
    protobuf::libprotobuf
    absl::base
    absl::strings
    absl::hash
    absl::flat_hash_map
    absl::flat_hash_set
    ${ONNXRUNTIME_LIB}
    sentencepiece-static
) 
cmake_minimum_required(VERSION 3.15)
project(LocalTranslator VERSION 1.0.0 LANGUAGES CXX)

# 启用测试
enable_testing()

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# vcpkg集成
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/lib")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/lib/manual-link")

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg_installed/vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "Vcpkg toolchain file")
set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "Vcpkg target triplet")

# 查找依赖包
find_package(CURL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(tinyxml2 CONFIG REQUIRED)
find_package(protobuf CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)

# SentencePiece configuration
option(SPM_ENABLE_SHARED "Build SentencePiece shared library" OFF)
option(SPM_ENABLE_TCMALLOC "Enable TCMalloc if available" OFF)
option(SPM_USE_BUILTIN_PROTOBUF "Use built-in protobuf" OFF)
add_subdirectory(dep/sentencepiece EXCLUDE_FROM_ALL)

# Windows特定设置
if(MSVC)
    add_compile_options(/utf-8)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Ob2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "/MDd /Zi /Ob0 /Od /RTC1")
endif()

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ONNX Runtime设置
if(WIN32)
    set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime/onnxruntime-win-x64")
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    set(ONNXRUNTIME_DLL "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
else()
    set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime/onnxruntime-linux-x64")
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
endif()

# 添加子目录
add_subdirectory(common)
add_subdirectory(translator)
add_subdirectory(tests)

# 主程序
add_executable(${PROJECT_NAME} main.cpp)

# 链接库
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    translator
    common
    yaml-cpp::yaml-cpp
    CURL::libcurl
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    tinyxml2::tinyxml2
    ZLIB::ZLIB
    protobuf::libprotobuf
    absl::base
    absl::strings
    absl::hash
    absl::flat_hash_map
    absl::flat_hash_set
    sentencepiece-static
)

# 包含目录
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${ONNXRUNTIME_ROOT}/include
    ${CMAKE_SOURCE_DIR}/dep/sentencepiece/src
    ${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/include
)

# Windows下复制DLL到输出目录
if(WIN32)
    # 复制onnxruntime.dll
    if(EXISTS "${ONNXRUNTIME_DLL}")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DLL}"
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
            COMMENT "Copying onnxruntime.dll to output directory"
        )
    else()
        message(WARNING "ONNXRUNTIME_DLL not found at: ${ONNXRUNTIME_DLL}")
    endif()

    # 复制vcpkg依赖
    if(EXISTS "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/bin")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/bin"
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
            COMMENT "Copying vcpkg dependencies to output directory"
        )
    else()
        message(WARNING "vcpkg binaries directory not found at: ${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/bin")
    endif()
endif() 
cmake_minimum_required(VERSION 3.15)
project(LocalTranslator VERSION 1.0.0 LANGUAGES CXX)

# 启用测试
enable_testing()

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

# vcpkg集成
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "")
endif()

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

# 查找sentencepiece
if(WIN32)
    find_library(SENTENCEPIECE_LIB
        NAMES sentencepiece
        PATHS "$ENV{VCPKG_ROOT}/installed/x64-windows/lib"
        REQUIRED
    )
    find_path(SENTENCEPIECE_INCLUDE_DIR
        NAMES sentencepiece_processor.h
        PATHS "$ENV{VCPKG_ROOT}/installed/x64-windows/include"
        REQUIRED
    )
else()
    find_library(SENTENCEPIECE_LIB
        NAMES sentencepiece
        REQUIRED
    )
    find_path(SENTENCEPIECE_INCLUDE_DIR
        NAMES sentencepiece_processor.h
        REQUIRED
    )
endif()

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
    ${SENTENCEPIECE_LIB}
)

# 包含目录
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${ONNXRUNTIME_ROOT}/include
    ${SENTENCEPIECE_INCLUDE_DIR}
)

# Windows下复制DLL到输出目录
if(WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DLL}"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )

    # 复制其他依赖DLL
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "$ENV{VCPKG_ROOT}/installed/x64-windows/bin"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif() 
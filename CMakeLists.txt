cmake_minimum_required(VERSION 3.10)
project(LocalTranslator)

# 启用测试
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 

# 设置 UTF-8 编码
if(MSVC)
    add_compile_options(/utf-8)
endif()

# 设置 vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "Vcpkg toolchain file")

# 设置库的查找路径
if(WIN32)
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/x64-windows")
else()
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/x64-linux")
endif()

# 设置 ONNX Runtime 路径
if(WIN32)
    set(ONNXRUNTIME_ROOTDIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/onnxruntime-win-x64")
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime.lib")
    set(ONNXRUNTIME_DLL "${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime.dll")
else()
    set(ONNXRUNTIME_ROOTDIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/onnxruntime-linux-x64")
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so")
endif()
list(APPEND CMAKE_PREFIX_PATH ${ONNXRUNTIME_ROOTDIR})

# 查找依赖包
find_package(CURL REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(GTest REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(sentencepiece CONFIG REQUIRED)

# 添加源文件
add_library(translator
    translator/translator.cpp
    translator/deeplx/deeplx_translator.cpp
    translator/nllb-api/nllb_translator.cpp
)

# 设置包含目录
target_include_directories(translator
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CURL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/$<IF:$<BOOL:${WIN32}>,x64-windows,x64-linux>/include
    ${ONNXRUNTIME_ROOTDIR}/include
)

# 链接依赖库
target_link_libraries(translator
    PUBLIC
    CURL::libcurl
    yaml-cpp::yaml-cpp
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    sentencepiece
    ${ONNXRUNTIME_LIB}
)

# 添加可执行文件
add_executable(local_translator main.cpp)
target_link_libraries(local_translator PRIVATE translator)

# 复制 ONNX Runtime DLL 到输出目录 (仅 Windows)
if(WIN32)
    add_custom_command(TARGET local_translator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DLL}"
        $<TARGET_FILE_DIR:local_translator>
    )
endif()

# 添加测试子目录
add_subdirectory(tests) 
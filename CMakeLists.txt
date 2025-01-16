cmake_minimum_required(VERSION 3.10)
project(LocalTranslator)

# 启用测试
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 

# 设置 vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "Vcpkg toolchain file")

# 设置库的查找路径
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/x64-windows")

# 查找依赖包
find_package(CURL REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(GTest REQUIRED)

# 添加源文件
add_library(translator
    translator/translator.cpp
    translator/deeplx/deeplx_translator.cpp
)

# 设置包含目录
target_include_directories(translator
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CURL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/x64-windows/include
)

# 链接依赖库
target_link_libraries(translator
    PUBLIC
    CURL::libcurl
    yaml-cpp::yaml-cpp
    nlohmann_json::nlohmann_json
)

# 添加可执行文件
add_executable(local_translator main.cpp)
target_link_libraries(local_translator PRIVATE translator)

# 添加测试子目录
add_subdirectory(tests) 
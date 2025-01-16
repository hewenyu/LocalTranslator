cmake_minimum_required(VERSION 3.10)

# 启用测试
enable_testing()

# 查找 GTest 包
find_package(GTest REQUIRED)

# 添加测试可执行文件
add_executable(translator_test translator_test.cpp)
add_executable(nllb_translator_test nllb_translator_test.cpp)

# 链接 GTest 和项目库
target_link_libraries(translator_test
    PRIVATE
    GTest::GTest
    GTest::Main
    translator
)

target_link_libraries(nllb_translator_test
    PRIVATE
    translator
    GTest::GTest
    GTest::Main
    spdlog::spdlog
    yaml-cpp::yaml-cpp
)

# 复制 DLL 到输出目录
if(WIN32)
    # 设置 DLL 源目录
    set(DLL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg_installed/x64-windows/bin")
    set(DLL_DEBUG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg_installed/x64-windows/debug/bin")
    
    # 复制 Debug DLLs
    file(GLOB DEBUG_DLLS "${DLL_DEBUG_DIR}/*.dll")
    file(COPY ${DEBUG_DLLS} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/Debug")
    
    # 为测试目标设置工作目录
    set_target_properties(translator_test PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Debug"
    )
    set_target_properties(nllb_translator_test PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Debug"
    )
endif()

# 添加测试
add_test(
    NAME TranslatorTest
    COMMAND translator_test
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Debug"
)

add_test(
    NAME NLLBTranslatorTest
    COMMAND nllb_translator_test
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Debug"
) 
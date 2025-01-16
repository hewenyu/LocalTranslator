cmake_minimum_required(VERSION 3.10)

# 启用测试
enable_testing()

# 查找 GTest 包
find_package(GTest REQUIRED)

# 添加测试可执行文件
add_executable(translator_test translator_test.cpp)
add_executable(deeplx_translator_test deeplx_translator_test.cpp)

# 链接 GTest 和项目库
target_link_libraries(translator_test
    PRIVATE
    GTest::GTest
    GTest::Main
    translator
)

target_link_libraries(deeplx_translator_test
    PRIVATE
    GTest::GTest
    GTest::Main
    translator
)

# 复制 DLL 到输出目录
if(WIN32)
    add_custom_command(TARGET translator_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg_installed/x64-windows/bin"
        $<TARGET_FILE_DIR:translator_test>
    )
    add_custom_command(TARGET deeplx_translator_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg_installed/x64-windows/bin"
        $<TARGET_FILE_DIR:deeplx_translator_test>
    )
endif()

# 添加测试
add_test(NAME TranslatorTest COMMAND translator_test)
add_test(NAME DeepLXTranslatorTest COMMAND deeplx_translator_test) 
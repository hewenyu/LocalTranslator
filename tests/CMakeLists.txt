enable_testing()

# 翻译器测试
add_executable(translator_test
    translator_test.cpp
)

target_link_libraries(translator_test
    PRIVATE
    translator
    GTest::gtest
    GTest::gtest_main
    benchmark::benchmark
    ${SENTENCEPIECE_LIB}
)

target_include_directories(translator_test
    PRIVATE
    ${SENTENCEPIECE_INCLUDE_DIR}
)

# NLLB性能测试
add_executable(nllb_translator_benchmark
    nllb_translator_benchmark.cpp
)

target_link_libraries(nllb_translator_benchmark
    PRIVATE
    translator
    benchmark::benchmark
    benchmark::benchmark_main
    ${SENTENCEPIECE_LIB}
)

target_include_directories(nllb_translator_benchmark
    PRIVATE
    ${SENTENCEPIECE_INCLUDE_DIR}
)

# 添加测试
add_test(NAME translator_test COMMAND translator_test)

# Windows下复制依赖DLL到测试目录
if(WIN32)
    foreach(TARGET translator_test nllb_translator_benchmark)
        # 复制onnxruntime.dll
        if(EXISTS "${ONNXRUNTIME_DLL}")
            add_custom_command(TARGET ${TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_DLL}"
                $<TARGET_FILE_DIR:${TARGET}>
                COMMENT "Copying onnxruntime.dll to output directory"
            )
        else()
            message(WARNING "ONNXRUNTIME_DLL not found at: ${ONNXRUNTIME_DLL}")
        endif()

        # 复制vcpkg依赖
        if(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/installed/x64-windows/bin")
            add_custom_command(TARGET ${TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                "$ENV{VCPKG_ROOT}/installed/x64-windows/bin"
                $<TARGET_FILE_DIR:${TARGET}>
                COMMENT "Copying vcpkg dependencies to output directory"
            )
        else()
            message(WARNING "VCPKG binaries directory not found. Please make sure VCPKG_ROOT is set correctly.")
        endif()
    endforeach()
endif() 
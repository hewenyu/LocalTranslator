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
    sentencepiece
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
    sentencepiece
)

# 添加测试
add_test(NAME translator_test COMMAND translator_test)

# Windows下复制依赖DLL到测试目录
if(WIN32)
    foreach(TARGET translator_test nllb_translator_benchmark)
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DLL}"
            $<TARGET_FILE_DIR:${TARGET}>
        )
    endforeach()
endif() 
# 构建和测试指南

本文档描述了如何构建和测试 LocalTranslator 项目。

## 前置条件

1. Windows 环境
2. Visual Studio 2022 或更高版本
3. CMake 3.10 或更高版本
4. vcpkg 包管理器

## 安装依赖

使用 vcpkg 安装所需依赖：

```bash
# 安装依赖包
vcpkg install curl:x64-windows yaml-cpp:x64-windows nlohmann-json:x64-windows gtest:x64-windows
# C:\Users\boringsoft\code\microsoft\vcpkg\vcpkg.exe install --triplet=x64-windows
```

## 构建步骤

1. 创建构建目录并生成项目文件：
```bash
cmake -B build -S .
```
预期输出：
```
-- Building for: Visual Studio 17 2022
-- Selecting Windows SDK version 10.0.22621.0
-- The CXX compiler identification is MSVC 19.37.32822.0
...
-- Configuring done
-- Generating done
-- Build files have been written to: C:/Users/[username]/LocalTranslator/build
```

2. 编译项目：
```bash
cmake --build build
```
预期输出：
```
Microsoft (R) Build Engine version 17.7.2+d6990bcfa for .NET Framework
Copyright (C) Microsoft Corporation. All rights reserved.

  translator.cpp
  deeplx_translator.cpp
  Generating Code...
  translator.vcxproj -> C:\Users\[username]\LocalTranslator\build\Debug\translator.lib
  translator_test.cpp
  test.cpp
  Generating Code...
  translator_test.vcxproj -> C:\Users\[username]\LocalTranslator\build\tests\Debug\translator_test.exe
  deeplx_translator_test.cpp
  test.cpp
  Generating Code...
  deeplx_translator_test.vcxproj -> C:\Users\[username]\LocalTranslator\build\tests\Debug\deeplx_translator_test.exe
```

## 运行测试

在 build 目录下运行测试：
```bash
cd build
ctest -C Debug --output-on-failure
```
预期输出：
```
Test project C:/Users/[username]/LocalTranslator/build
    Start 1: TranslatorTest
1/2 Test #1: TranslatorTest .....................   Passed   0.52 sec
    Start 2: DeepLXTranslatorTest
2/2 Test #2: DeepLXTranslatorTest ..............   Passed   0.48 sec

100% tests passed, 0 tests failed out of 2
Total Test time (real) =   1.00 sec
```

## 项目结构

```
LocalTranslator/
├── CMakeLists.txt          # 主 CMake 配置文件
├── vcpkg.json             # vcpkg 依赖配置
├── main.cpp               # 主程序入口
├── translator/            # 翻译器核心代码
│   ├── translator.h
│   ├── translator.cpp
│   └── deeplx/           # DeepLX 翻译器实现
├── common/               # 通用代码
└── tests/               # 测试代码
    ├── CMakeLists.txt   # 测试 CMake 配置
    ├── translator_test.cpp
    └── deeplx_translator_test.cpp
```

## 测试说明

项目包含两组测试：

1. TranslatorTest：测试翻译器接口
   - 测试翻译器创建
   - 测试目标语言获取
   - 测试基本翻译功能

2. DeepLXTranslatorTest：测试 DeepLX 翻译器实现
   - 测试初始化
   - 测试相同语言翻译
   - 测试英文到中文翻译
   - 测试错误处理

## 常见问题

1. DLL 加载错误 (0xc0000135)
   - 确保所有依赖的 DLL 都已正确复制到测试可执行文件所在目录
   - 检查 vcpkg 依赖是否正确安装

2. 编译错误
   - 确保 Visual Studio 和 CMake 版本满足要求
   - 检查 vcpkg 集成是否正确

## 调试

可以使用 Visual Studio 调试器直接调试测试：
1. 在 Visual Studio 中打开 build/LocalTranslator.sln
2. 选择要调试的测试项目
3. 设置为启动项目
4. 按 F5 开始调试 
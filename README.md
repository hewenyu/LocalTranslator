# Windows C++ 开发环境配置指南

本指南介绍如何在 Windows 系统上配置 C++ 开发环境，使用 MSYS2/MinGW-w64 作为编译工具链。

## 1. 安装 MSYS2

1. 访问 [MSYS2 官网](https://www.msys2.org/) 下载安装程序
2. 运行安装程序，建议安装到默认位置 `C:\msys64`

## 2. 安装编译工具

1. 打开 "MSYS2 MINGW64" 终端（在开始菜单中搜索）
2. 运行以下命令更新系统并安装必要工具：
```bash
pacman -Syu
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-gdb
```

## 3. 配置环境变量

1. 按 Win + R，输入 sysdm.cpl
2. 点击"高级" -> "环境变量"
3. 在"系统变量"的 Path 中添加：`C:\msys64\mingw64\bin`
4. 点击"确定"保存

## 4. 配置 VS Code

### tasks.json
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "type": "cppbuild",
            "label": "C/C++: g++ build active file",
            "command": "g++.exe",
            "args": [
                "-fdiagnostics-color=always",
                "-g",
                "${file}",
                "-o",
                "${fileDirname}\\${fileBasenameNoExtension}.exe"
            ],
            "options": {
                "cwd": "${fileDirname}"
            },
            "problemMatcher": [
                "$gcc"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "detail": "编译器: g++"
        }
    ]
}
```

### launch.json
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "C++ Debug",
            "type": "cppdbg",
            "request": "launch",
            "program": "${fileDirname}\\${fileBasenameNoExtension}.exe",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${fileDirname}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "miDebuggerPath": "C:\\msys64\\mingw64\\bin\\gdb.exe",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "C/C++: g++ build active file"
        }
    ]
}
```

## 5. 验证安装

1. 打开新的终端，运行：
```bash
g++ --version
```
应该显示版本信息

2. 在编辑器中创建并运行一个简单的 C++ 程序：
```cpp
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
```

## 注意事项

- 如果命令提示符找不到 g++，请确保重新打开终端
- 确保所有配置文件中的路径使用双反斜杠 `\\`
- 如果使用 Cursor 编辑器，配置过程与 VS Code 相同 
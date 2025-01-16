# LocalTranslator

A local translation tool that supports multiple translation backends including DeepLX and NLLB.

## Prerequisites

### Windows
- Windows 10/11
- Visual Studio Code
- MinGW-w64 (GCC)
- vcpkg package manager

### Linux
- GCC/G++
- Visual Studio Code
- vcpkg package manager

## Setup Instructions

### 1. Install vcpkg

#### Windows
```bash
# Clone vcpkg repository
git clone https://github.com/Microsoft/vcpkg.git

# Run the bootstrap script
.\vcpkg\bootstrap-vcpkg.bat
```

#### Linux
```bash
# Clone vcpkg repository
git clone https://github.com/Microsoft/vcpkg.git

# Run the bootstrap script
./vcpkg/bootstrap-vcpkg.sh
```

### 2. Install Dependencies

There are two ways to install the required dependencies:

#### Option 1: Using vcpkg.json (Recommended)
The project includes a `vcpkg.json` manifest file that specifies all required dependencies. To install them:

```bash
# Navigate to the project directory containing vcpkg.json
cd path/to/project

# For Windows
vcpkg install --triplet x64-windows

# For Linux
vcpkg install --triplet x64-linux
```

#### Option 2: Manual Installation
If you prefer to install dependencies manually:

```bash
# Windows
vcpkg install yaml-cpp:x64-windows

# Linux
vcpkg install yaml-cpp:x64-linux
```

### 3. Environment Configuration

#### Windows
Add the following path to your system's PATH environment variable:
```
C:\Users\[YourUsername]\code\microsoft\vcpkg\installed\x64-windows\bin
```

#### Linux
Add the following to your `~/.bashrc` or `~/.zshrc`:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/vcpkg/installed/x64-linux/lib
```

### 4. VSCode Configuration

The project includes three important configuration files in the `.vscode` directory:

#### Windows Configuration

1. `settings.json`:
```json
{
    "C_Cpp.default.includePath": [
        "${workspaceFolder}/**",
        "C:/Users/[YourUsername]/code/microsoft/vcpkg/installed/x64-windows/include"
    ]
}
```

2. `tasks.json`:
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "type": "cppbuild",
            "label": "C/C++: g++ build active file",
            "command": "g++",
            "args": [
                "-fdiagnostics-color=always",
                "-g",
                "${file}",
                "-o",
                "${fileDirname}\\${fileBasenameNoExtension}.exe",
                "-I", "C:/Users/[YourUsername]/code/microsoft/vcpkg/installed/x64-windows/include",
                "-L", "C:/Users/[YourUsername]/code/microsoft/vcpkg/installed/x64-windows/lib",
                "-lyaml-cpp"
            ]
        }
    ]
}
```

#### Linux Configuration

1. `settings.json`:
```json
{
    "C_Cpp.default.includePath": [
        "${workspaceFolder}/**",
        "${HOME}/vcpkg/installed/x64-linux/include"
    ]
}
```

2. `tasks.json`:
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "type": "cppbuild",
            "label": "C/C++: g++ build active file",
            "command": "g++",
            "args": [
                "-fdiagnostics-color=always",
                "-g",
                "${file}",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}",
                "-I", "${HOME}/vcpkg/installed/x64-linux/include",
                "-L", "${HOME}/vcpkg/installed/x64-linux/lib",
                "-lyaml-cpp",
                "-Wl,-rpath,${HOME}/vcpkg/installed/x64-linux/lib"
            ]
        }
    ]
}
```

3. `launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "C++ Debug",
            "type": "cppdbg",
            "request": "launch",
            "program": "${fileDirname}/${fileBasenameNoExtension}",
            "miDebuggerPath": "/usr/bin/gdb"
        }
    ]
}
```

## Project Structure

- `translator/` - Core translation functionality
  - `translator.h` - Main translator interface and configurations
- `vcpkg.json` - Project dependencies manifest 
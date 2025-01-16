@echo off

cmd /c "rmdir /s /q build && cmake -B build -S . -DCMAKE_INSTALL_PREFIX="./vcpkg_installed/x64-windows;./onnxruntime/onnxruntime-win-x64"

echo Building project...
cmake --build build

echo Build process completed.



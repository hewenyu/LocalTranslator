@echo off

cmd /c "rmdir /s /q build && cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=C:/Users/boringsoft/code/microsoft/vcpkg/scripts/buildsystems/vcpkg.cmake"

echo Building project...
cmake --build build

echo Build process completed.


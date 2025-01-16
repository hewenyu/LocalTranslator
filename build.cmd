@echo off
echo Cleaning build directory...
if exist build rmdir /s /q build

echo Configuring project...
cmake -B build -S .

echo Building project...
cmake --build build

echo Running tests...
cd build && ctest -C Debug --output-on-failure
cd ..

echo Build process completed. 
@echo off
echo Cleaning build directory...
if exist build rmdir /s /q build

echo Creating build directory...
mkdir build

echo Configuring project...
cd build
cmake .. -G "Visual Studio 17 2022" -A x64

echo Building project...
cmake --build . --config Release --target ALL_BUILD

echo Build process completed.
cd ..


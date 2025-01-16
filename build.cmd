@echo off

cmd /c "rmdir /s /q build && cmake -B build -S .

echo Building project...
cmake --build build --config Release

echo Build process completed.



@echo off
setlocal enabledelayedexpansion

set "ROOT_DIR=%~dp0..\"
set "BUILD_DIR=%ROOT_DIR%build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

where cl.exe >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
    )
)

echo [*] Compiling Independent Challenger Benchmark Suite...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /EHsc /utf-8 /wd4068 ^
    /I"%ROOT_DIR%include" /I"%ROOT_DIR%legacy_ptq" ^
    "%ROOT_DIR%src\bridge.cpp" ^
    "%ROOT_DIR%tests\bench_bridge_challenger.cpp" ^
    /Fe:"%BUILD_DIR%\bench_bridge_challenger.exe" /Fo:"%BUILD_DIR%\\" ^
    /link /STACK:33554432

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Compilation failed.
    exit /b %ERRORLEVEL%
)

echo [+] Built successfully: "%BUILD_DIR%\bench_bridge_challenger.exe"
echo [*] Executing Challenger Benchmark Suite...
"%BUILD_DIR%\bench_bridge_challenger.exe"
exit /b %ERRORLEVEL%

@echo off
chcp 65001 >nul
setlocal

set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

if "%1"=="clean" (
    echo [*] Cleaning second_brain build artifacts...
    if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
    echo [+] Clean complete.
    exit /b 0
)

where cl.exe >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul
    )
)

echo [*] Compiling PopCount Second Brain Engine (AVX-512 VPOPCNTDQ / C++20)...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /utf-8 /wd4068 /I"%SCRIPT_DIR%." /I"%SCRIPT_DIR%..\include" "%SCRIPT_DIR%popcnt_brain.cpp" /Fe:"%BUILD_DIR%\popcnt_brain.exe" /Fo:"%BUILD_DIR%\popcnt_brain.obj" /link /STACK:33554432
if %ERRORLEVEL% NEQ 0 (
    echo [!] AVX-512 compilation failed, falling back to AVX2...
    cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX2 /openmp /EHsc /utf-8 /wd4068 /I"%SCRIPT_DIR%." /I"%SCRIPT_DIR%..\include" "%SCRIPT_DIR%popcnt_brain.cpp" /Fe:"%BUILD_DIR%\popcnt_brain.exe" /Fo:"%BUILD_DIR%\popcnt_brain.obj" /link /STACK:33554432
    if errorlevel 1 (
        echo [!] Compilation failed!
        exit /b 1
    )
)
echo [+] Build succeeded: "%BUILD_DIR%\popcnt_brain.exe"

REM Build Static Library & C-ABI Test
echo [*] Building PopCount Brain C-ABI Library Object...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /utf-8 /wd4068 /DBUILDING_POPCNT_BRAIN_LIB /I"%SCRIPT_DIR%." /I"%SCRIPT_DIR%..\include" /c "%SCRIPT_DIR%popcnt_brain.cpp" /Fo:"%BUILD_DIR%\popcnt_brain_lib.obj"
if errorlevel 1 exit /b 1
lib.exe /nologo /OUT:"%BUILD_DIR%\popcnt_brain.lib" "%BUILD_DIR%\popcnt_brain_lib.obj"
if errorlevel 1 exit /b 1
if %ERRORLEVEL% EQU 0 (
    echo [+] Static library created: "%BUILD_DIR%\popcnt_brain.lib"
)

echo [*] Compiling C-ABI Integration Test...
cl.exe /nologo /std:c++20 /O2 /EHsc /utf-8 /I"%SCRIPT_DIR%." "%SCRIPT_DIR%test_c_api.cpp" "%BUILD_DIR%\popcnt_brain.lib" /Fe:"%BUILD_DIR%\test_c_api.exe" /Fo:"%BUILD_DIR%\test_c_api.obj" /link /STACK:33554432
if errorlevel 1 exit /b 1
if %ERRORLEVEL% EQU 0 (
    echo [+] C-ABI test built: "%BUILD_DIR%\test_c_api.exe"
)

if "%1"=="run" (
    echo [*] Executing PopCount Second Brain Self-Test Suite...
    "%BUILD_DIR%\popcnt_brain.exe"
    if errorlevel 1 exit /b 1
    exit /b 0
)

if "%1"=="test" (
    echo [*] Executing PopCount Second Brain C-ABI Integration Test...
    "%BUILD_DIR%\test_c_api.exe"
    if errorlevel 1 exit /b 1
    exit /b 0
)

if "%1"=="all" (
    echo [*] Executing Full Suite...
    "%BUILD_DIR%\popcnt_brain.exe"
    if errorlevel 1 exit /b 1
    "%BUILD_DIR%\test_c_api.exe"
    if errorlevel 1 exit /b 1
    exit /b 0
)

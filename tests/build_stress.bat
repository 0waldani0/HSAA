@echo off
setlocal enabledelayedexpansion

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

set "ROOT_DIR=%~dp0..\"
set "BUILD_DIR=%ROOT_DIR%build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

echo [*] Compiling Independent Stress Test Harness (stress_model_mod.cpp)...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /utf-8 /wd4068 ^
    /I"%ROOT_DIR%include" /I"%ROOT_DIR%legacy_ptq" ^
    "%ROOT_DIR%src\vsa_db.cpp" ^
    "%ROOT_DIR%src\model_mod.cpp" ^
    "%ROOT_DIR%tests\stress_model_mod.cpp" ^
    /Fe:"%BUILD_DIR%\stress_model_mod.exe" /Fo:"%BUILD_DIR%\\" ^
    /link /STACK:33554432 psapi.lib

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Stress test harness compilation failed.
    exit /b %ERRORLEVEL%
)

echo [+] Stress test executable built: "%BUILD_DIR%\stress_model_mod.exe"

if "%1"=="run" (
    echo [*] Executing Adversarial Stress Tests...
    "%BUILD_DIR%\stress_model_mod.exe"
    exit /b %ERRORLEVEL%
)

if "%1"=="" (
    echo [*] Executing Adversarial Stress Tests...
    "%BUILD_DIR%\stress_model_mod.exe"
    exit /b %ERRORLEVEL%
)

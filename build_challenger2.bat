@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "ROOT_DIR=%~dp0"
set "BUILD_DIR=%ROOT_DIR%build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

where cl.exe >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if not exist "!VSWHERE!" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "!VSWHERE!" (
        for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
            if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" (
                call "%%i\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
            )
        )
    )
)

echo [*] Compiling Challenger 2 Stress Test Suite...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /utf-8 /wd4068 ^
    /I"%ROOT_DIR%include" /I"%ROOT_DIR%second_brain" /I"%ROOT_DIR%legacy_ptq" /I"%ROOT_DIR%tests" ^
    "%ROOT_DIR%src\bridge.cpp" ^
    "%ROOT_DIR%src\vsa_db.cpp" ^
    "%ROOT_DIR%src\model_mod.cpp" ^
    "%ROOT_DIR%tests\stress_test_challenger2.cpp" ^
    "%BUILD_DIR%\popcnt_brain.lib" ^
    /Fe:"%BUILD_DIR%\stress_test_challenger2.exe" /Fo:"%BUILD_DIR%\\" ^
    /link /STACK:33554432 psapi.lib

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Compilation failed with code %ERRORLEVEL%.
    exit /b %ERRORLEVEL%
)

echo [+] Successfully built: "%BUILD_DIR%\stress_test_challenger2.exe"
"%BUILD_DIR%\stress_test_challenger2.exe"
exit /b %ERRORLEVEL%

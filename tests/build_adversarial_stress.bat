@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "ROOT_DIR=%~dp0..\"
set "BUILD_DIR=%ROOT_DIR%build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

where cl.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 goto :msvc_configured

set "VCVARS64="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" (
            set "VCVARS64=%%i\VC\Auxiliary\Build\vcvars64.bat"
        )
    )
)

if not defined VCVARS64 (
    if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS64=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS64=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    )
)

if defined VCVARS64 (
    call "%VCVARS64%" >nul 2>nul
)

:msvc_configured

REM ============================================================================
REM 2. Ensure popcnt_brain.lib exists
REM ============================================================================
if not exist "%BUILD_DIR%\popcnt_brain.lib" (
    echo [*] Building prerequisite popcnt_brain.lib...
    cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /EHsc /utf-8 /wd4068 /DBUILDING_POPCNT_BRAIN_LIB /c ^
        "%ROOT_DIR%second_brain\popcnt_brain.cpp" /Fo:"%BUILD_DIR%\popcnt_brain_lib.obj"
    lib.exe /nologo "%BUILD_DIR%\popcnt_brain_lib.obj" /OUT:"%BUILD_DIR%\popcnt_brain.lib"
)

echo [*] Compiling Adversarial Stress Harness (adversarial_stress_hssa.cpp)...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /utf-8 /wd4068 ^
    /I"%ROOT_DIR%include" /I"%ROOT_DIR%second_brain" /I"%ROOT_DIR%legacy_ptq" /I"%ROOT_DIR%tests" ^
    "%ROOT_DIR%src\bridge.cpp" ^
    "%ROOT_DIR%src\vsa_db.cpp" ^
    "%ROOT_DIR%src\model_mod.cpp" ^
    "%ROOT_DIR%tests\adversarial_stress_hssa.cpp" ^
    "%BUILD_DIR%\popcnt_brain.lib" ^
    /Fe:"%BUILD_DIR%\adversarial_stress_hssa.exe" /Fo:"%BUILD_DIR%\\" ^
    /link /STACK:33554432 psapi.lib

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Compilation failed with code %ERRORLEVEL%.
    exit /b %ERRORLEVEL%
)

echo [*] Executing Adversarial Stress Harness on Host Hardware...
"%BUILD_DIR%\adversarial_stress_hssa.exe"
exit /b %ERRORLEVEL%

@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "ROOT_DIR=%~dp0"
set "BUILD_DIR=%ROOT_DIR%build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

REM ============================================================================
REM 1. Multi-Tiered MSVC 2022 vcvars64.bat Detection
REM ============================================================================
where cl.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [*] MSVC cl.exe already available in PATH.
    goto :msvc_configured
)

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
    ) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS64=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS64=%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS64=%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    )
)

if defined VCVARS64 (
    echo [*] Initializing MSVC x64 environment: "%VCVARS64%"
    call "%VCVARS64%" >nul 2>nul
)

where cl.exe >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] MSVC cl.exe not available in PATH. Please install Visual Studio 2022 or C++ Build Tools.
    exit /b 1
)

:msvc_configured

REM ============================================================================
REM 2. Ensure popcnt_brain.lib exists
REM ============================================================================
if not exist "%BUILD_DIR%\popcnt_brain.lib" (
    echo [*] Building prerequisite popcnt_brain.lib...
    cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /EHsc /utf-8 /wd4068 /c ^
        "%ROOT_DIR%second_brain\popcnt_brain.cpp" /Fo:"%BUILD_DIR%\popcnt_brain.obj"
    lib.exe /nologo "%BUILD_DIR%\popcnt_brain.obj" /OUT:"%BUILD_DIR%\popcnt_brain.lib"
)

REM ============================================================================
REM 3. Compile HSSA Empirical Micro-Benchmark Suite
REM ============================================================================
echo [*] Compiling HSSA Micro-Benchmark Suite (micro_bench_hssa.cpp)...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /utf-8 /wd4068 ^
    /I"%ROOT_DIR%include" /I"%ROOT_DIR%second_brain" /I"%ROOT_DIR%legacy_ptq" /I"%ROOT_DIR%tests" ^
    "%ROOT_DIR%src\bridge.cpp" ^
    "%ROOT_DIR%src\vsa_db.cpp" ^
    "%ROOT_DIR%src\model_mod.cpp" ^
    "%ROOT_DIR%tests\micro_bench_hssa.cpp" ^
    "%BUILD_DIR%\popcnt_brain.lib" ^
    /Fe:"%BUILD_DIR%\micro_bench_hssa.exe" /Fo:"%BUILD_DIR%\\" ^
    /link /STACK:33554432 psapi.lib

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Compilation of micro_bench_hssa.exe failed with code %ERRORLEVEL%.
    exit /b %ERRORLEVEL%
)

echo [+] Successfully built: "%BUILD_DIR%\micro_bench_hssa.exe"

REM ============================================================================
REM 4. Execute Benchmark Suite
REM ============================================================================
if "%1"=="build_only" (
    exit /b 0
)

echo [*] Executing HSSA Micro-Benchmark Suite on Host Hardware...
"%BUILD_DIR%\micro_bench_hssa.exe"
exit /b %ERRORLEVEL%

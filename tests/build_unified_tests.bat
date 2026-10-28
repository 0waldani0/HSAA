@echo off
chcp 65001 >nul
setlocal

set "ROOT_DIR=%~dp0.."
set "BUILD_DIR=%ROOT_DIR%\build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%BUILD_DIR%\obj_test_unified" mkdir "%BUILD_DIR%\obj_test_unified"

where cl.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 goto :msvc_ok

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" (
            call "%%i\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
            goto :msvc_ok
        )
    )
)
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
)

:msvc_ok
echo [*] Compiling test_unified_engine.exe...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /utf-8 /wd4068 ^
    /I"%ROOT_DIR%\include" /I"%ROOT_DIR%\second_brain" /I"%ROOT_DIR%\legacy_ptq" ^
    "%ROOT_DIR%\src\bridge.cpp" ^
    "%ROOT_DIR%\src\vsa_db.cpp" ^
    "%ROOT_DIR%\src\model_mod.cpp" ^
    "%ROOT_DIR%\src\hssa.cpp" ^
    "%ROOT_DIR%\src\bench_runner.cpp" ^
    "%ROOT_DIR%\src\unified_engine.cpp" ^
    "%ROOT_DIR%\tests\test_unified_engine.cpp" ^
    "%BUILD_DIR%\popcnt_brain.lib" ^
    "%BUILD_DIR%\ptq_cuda.lib" ^
    /Fe:"%BUILD_DIR%\test_unified_engine.exe" ^
    /Fo:"%BUILD_DIR%\obj_test_unified\\" ^
    /link /STACK:33554432 psapi.lib

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Compilation of test_unified_engine.exe failed.
    exit /b 1
)
echo [+] Built "%BUILD_DIR%\test_unified_engine.exe"

if "%1"=="run" (
    "%BUILD_DIR%\test_unified_engine.exe"
    exit /b %ERRORLEVEL%
)
exit /b 0

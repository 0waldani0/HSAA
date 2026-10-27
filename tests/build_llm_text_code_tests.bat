@echo off
chcp 65001 >nul
setlocal

set "ROOT_DIR=%~dp0.."
set "BUILD_DIR=%ROOT_DIR%\build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

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
if not exist "%BUILD_DIR%\popcnt_brain_lib.obj" (
    echo [*] popcnt_brain_lib.obj missing, building second_brain first...
    call "%ROOT_DIR%\build.bat" brain
    if errorlevel 1 exit /b 1
)
echo [*] Compiling test_llm_text_code_gen.exe...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /utf-8 /wd4068 /wd4996 ^
    "%ROOT_DIR%\tests\test_llm_text_code_gen.cpp" ^
    "%BUILD_DIR%\popcnt_brain_lib.obj" ^
    /Fe:"%BUILD_DIR%\test_llm_text_code_gen.exe" ^
    /Fo:"%BUILD_DIR%\test_llm_text_code_gen.obj" ^
    /link /STACK:33554432
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Compilation of test_llm_text_code_gen.exe failed.
    exit /b 1
)
echo [+] Built "%BUILD_DIR%\test_llm_text_code_gen.exe"

if "%1"=="run" (
    "%BUILD_DIR%\test_llm_text_code_gen.exe"
    exit /b %ERRORLEVEL%
)
exit /b 0

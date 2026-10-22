@echo off
chcp 65001 >nul
setlocal

set "ROOT_DIR=%~dp0..\"
set "BUILD_DIR=%ROOT_DIR%build"

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
    )
)

if defined VCVARS64 (
    call "%VCVARS64%" >nul 2>nul
)

:msvc_configured

cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /utf-8 /I"%ROOT_DIR%include" ^
    "%ROOT_DIR%src\vsa_db.cpp" ^
    "%ROOT_DIR%tests\test_cyrillic_persist.cpp" ^
    /Fe:"%BUILD_DIR%\test_cyrillic_persist.exe" /Fo:"%BUILD_DIR%\\" /link /STACK:33554432

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed
    exit /b %ERRORLEVEL%
)

"%BUILD_DIR%\test_cyrillic_persist.exe"
exit /b %ERRORLEVEL%

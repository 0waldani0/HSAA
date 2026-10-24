@echo off
chcp 65001 >nul
setlocal

rem Builds and runs the associative-fallback threshold calibration tool.
rem The tool includes ..\assoc_sketch.h -- the same header the shipped engine
rem uses -- so the sweep it prints describes shipped math, not a copy.
rem Paths derive from this script; the exe/obj are git-ignored like the other
rem measur artifacts.

set "HERE=%~dp0"
set "ROOT=%HERE%..\.."

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VCVARS64="
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS64=%%i\VC\Auxiliary\Build\vcvars64.bat"
    )
)
if not defined VCVARS64 (
    echo [FATAL] vcvars64 not found
    exit /b 90
)
call "%VCVARS64%" >nul

echo [*] building assoc_calib.exe ...
cl.exe /nologo /std:c++20 /O2 /EHsc /utf-8 /arch:AVX2 /I"%ROOT%\second_brain" ^
    "%HERE%assoc_calib.cpp" ^
    /Fe:"%HERE%assoc_calib.exe" /Fo:"%HERE%assoc_calib.obj" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL] assoc_calib build FAILED
    exit /b 91
)

"%HERE%assoc_calib.exe"
set "RC=%ERRORLEVEL%"
exit /b %RC%

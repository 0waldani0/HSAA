@echo off
chcp 65001 >nul
setlocal

rem Builds and runs the C-ABI integration probe against the engine library.
rem All paths are derived from this script's location, so the step works on
rem any checkout: it needs SECOND_BRAIN\popcnt_brain.h and BUILD\popcnt_brain.lib,
rem and it leaves capi_probe.exe/.obj next to itself (both git-ignored, the
rem layout measur/regress_par.py expects when it runs capi_probe.exe).

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

if not exist "%ROOT%\build\popcnt_brain.lib" (
    echo [FATAL] %ROOT%\build\popcnt_brain.lib not found -- run build.bat brain first
    exit /b 92
)

set "OUTDIR=%HERE%"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo [*] building capi_probe.exe ...
cl.exe /nologo /std:c++20 /O2 /EHsc /utf-8 /wd4068 /I"%ROOT%\second_brain" ^
    "%HERE%capi_probe.c" "%ROOT%\build\popcnt_brain.lib" ^
    /Fe:"%OUTDIR%\capi_probe.exe" /Fo:"%OUTDIR%\capi_probe.obj" /link /STACK:33554432 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL] capi_probe build FAILED
    exit /b 91
)
echo [+] built

cd /d "%ROOT%"
"%OUTDIR%\capi_probe.exe"
set "RC=%ERRORLEVEL%"
echo [*] probe exit=%RC%
exit /b %RC%

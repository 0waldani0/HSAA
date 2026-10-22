@echo off
chcp 65001 >nul
setlocal

REM ---------------------------------------------------------------------------
REM Builds and runs tests\test_popcnt_selector.cpp -- the task-training gates of
REM the second brain: the STE relay selector (a policy), and the relational
REM memory used as a lookup without generation.
REM ---------------------------------------------------------------------------

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VCVARS64="
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS64=%%i\VC\Auxiliary\Build\vcvars64.bat"
    )
)
if not defined VCVARS64 ( echo [FATAL] vcvars64.bat not found & exit /b 90 )
call "%VCVARS64%" >nul

set "ROOT=%~dp0.."
set "SB=%ROOT%\second_brain"
set "OUT=%ROOT%\build"

if not exist "%OUT%" mkdir "%OUT%"

cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /wd4068 /utf-8 ^
    /I"%SB%" "%ROOT%\tests\test_popcnt_selector.cpp" ^
    /Fe:"%OUT%\test_popcnt_selector.exe" /Fo:"%OUT%\test_popcnt_selector.obj" /link /STACK:33554432 2>&1
if %ERRORLEVEL% NEQ 0 ( echo [FATAL] build FAILED & exit /b 91 )

cd /d "%ROOT%"
"%OUT%\test_popcnt_selector.exe"
exit /b %ERRORLEVEL%

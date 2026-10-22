@echo off
setlocal enabledelayedexpansion
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" call "%%i\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
)

cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /EHsc /utf-8 "%~dp0tests\test_ssm_median.cpp" /Fe:"%~dp0build\test_ssm_median.exe" /Fo:"%~dp0build\\"
if %ERRORLEVEL% EQU 0 (
    "%~dp0build\test_ssm_median.exe"
)
exit /b %ERRORLEVEL%

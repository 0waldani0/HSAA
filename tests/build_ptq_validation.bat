@echo off
setlocal

where cl.exe >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul
    )
)

set "ROOT_DIR=%~dp0..\"
set "BUILD_DIR=%ROOT_DIR%build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

if not exist "%BUILD_DIR%\ptq_cuda.lib" (
    echo [*] ptq_cuda.lib missing, building it first...
    call "%ROOT_DIR%build.bat" ptq
    if errorlevel 1 exit /b 1
)

echo [*] Compiling PTQ host-validation tests...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /EHsc /utf-8 ^
    /I"%ROOT_DIR%legacy_ptq" ^
    "%~dp0test_ptq_validation.cpp" ^
    "%BUILD_DIR%\ptq_cuda.lib" ^
    /Fe:"%BUILD_DIR%\test_ptq_validation.exe" ^
    /Fo:"%BUILD_DIR%\test_ptq_validation.obj"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PTQ validation test compilation failed.
    exit /b 1
)

if not "%1"=="run" exit /b 0

echo [*] Executing PTQ host-validation tests...
"%BUILD_DIR%\test_ptq_validation.exe"
exit /b %ERRORLEVEL%

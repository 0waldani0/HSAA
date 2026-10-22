@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%..\"
set "BUILD_DIR=%ROOT_DIR%build\"
set "OUT_DIR=%BUILD_DIR%tests\"

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

REM ============================================================================
REM Detect MSVC 2022 vcvars64.bat
REM ============================================================================
where cl.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [*] MSVC cl.exe already active in environment.
    goto :msvc_ready
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
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS64=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS64=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS64=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS64=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS64=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    )
)

if not defined VCVARS64 (
    echo [!] ERROR: Could not locate MSVC 2022 vcvars64.bat
    exit /b 1
)

echo [*] Initializing MSVC x64 via %VCVARS64%...
call "%VCVARS64%" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [!] ERROR: Failed to configure MSVC x64 environment.
    exit /b %ERRORLEVEL%
)

:msvc_ready
echo [*] Compiling HSSA Milestone M2 Test Suite with AVX-512 and sm_86 CUDA support...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /utf-8 /wd4068 ^
    /I"%ROOT_DIR%include" ^
    /I"%ROOT_DIR%legacy_ptq" ^
    /I"%ROOT_DIR%tests" ^
    "%SCRIPT_DIR%test_hssa_m2.cpp" ^
    "%ROOT_DIR%src\hssa.cpp" ^
    "%ROOT_DIR%src\vsa_db.cpp" ^
    "%ROOT_DIR%src\bridge.cpp" ^
    "%ROOT_DIR%src\model_mod.cpp" ^
    "%ROOT_DIR%legacy_ptq\ptq_cuda.c" ^
    /Fe:"%OUT_DIR%test_hssa_m2.exe" ^
    /Fo"%OUT_DIR%\" ^
    /link psapi.lib /STACK:33554432

if %ERRORLEVEL% NEQ 0 (
    echo [!] Compilation failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo [+] Build succeeded: "%OUT_DIR%test_hssa_m2.exe"

if "%1"=="run" (
    echo.
    echo ================================================================================
    echo [*] Executing HSSA Milestone M2 Adversarial Stress Suite...
    echo ================================================================================
    "%OUT_DIR%test_hssa_m2.exe"
    exit /b %ERRORLEVEL%
)

exit /b 0

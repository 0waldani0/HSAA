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
set "OUT_DIR=%ROOT_DIR%build\tests\"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo [*] Compiling E2E Test Suite with MSVC C++20 /arch:AVX512...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /EHsc /wd4068 /I"%~dp0." ^
    "%~dp0e2e_test_runner.cpp" ^
    "%~dp0tier1_feature_tests.cpp" ^
    "%~dp0tier2_boundary_tests.cpp" ^
    "%~dp0tier3_pairwise_tests.cpp" ^
    "%~dp0tier4_workload_tests.cpp" ^
    /Fe:"%OUT_DIR%e2e_runner.exe" ^
    /Fo"%OUT_DIR%/" ^
    /link /STACK:33554432

if %ERRORLEVEL% EQU 0 (
    echo [+] E2E Test Runner built successfully: "%OUT_DIR%e2e_runner.exe"
) else (
    echo [!] Compilation failed with error %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

if "%1"=="run" (
    echo [*] Executing E2E Test Suite...
    "%OUT_DIR%e2e_runner.exe" %2 %3 %4 %5
)

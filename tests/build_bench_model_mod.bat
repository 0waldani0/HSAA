@echo off
setlocal

where cl.exe >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul
    )
)

set "ROOT_DIR=%~dp0..\"
set "OUT_DIR=%ROOT_DIR%build\"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo [*] Compiling bench_model_mod.cpp with MSVC C++20 /arch:AVX512 /openmp...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /utf-8 /wd4068 ^
    /I"%ROOT_DIR%include" /I"%ROOT_DIR%legacy_ptq" ^
    "%ROOT_DIR%src\vsa_db.cpp" ^
    "%ROOT_DIR%src\model_mod.cpp" ^
    "%~dp0bench_model_mod.cpp" ^
    /Fe:"%OUT_DIR%bench_model_mod.exe" ^
    /Fo:"%OUT_DIR%/" ^
    /link /STACK:33554432 psapi.lib

if %ERRORLEVEL% EQU 0 (
    echo [+] Benchmark harness built successfully: "%OUT_DIR%bench_model_mod.exe"
) else (
    echo [!] Compilation failed with error %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

if "%1"=="run" (
    echo [*] Executing bench_model_mod.exe...
    "%OUT_DIR%bench_model_mod.exe"
    exit /b %ERRORLEVEL%
)

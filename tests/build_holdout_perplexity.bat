@echo off
chcp 65001 >nul
setlocal
set "ROOT_DIR=%~dp0.."
set "BUILD_DIR=%ROOT_DIR%\build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%BUILD_DIR%\obj_holdout" mkdir "%BUILD_DIR%\obj_holdout"

where cl.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 goto :msvc_ok
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
)

:msvc_ok
rem Always rebuild the engine: reusing a cached .lib silently links stale code
rem after any edit to popcnt_brain.cpp, which would corrupt the measurement.
echo [*] Building popcnt_brain.lib...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /EHsc /utf-8 /wd4068 /DBUILDING_POPCNT_BRAIN_LIB /c ^
    "%ROOT_DIR%\second_brain\popcnt_brain.cpp" /Fo:"%BUILD_DIR%\popcnt_brain_lib.obj"
if errorlevel 1 exit /b 1
lib.exe /nologo "%BUILD_DIR%\popcnt_brain_lib.obj" /OUT:"%BUILD_DIR%\popcnt_brain.lib"
if errorlevel 1 exit /b 1

echo [*] Compiling held-out perplexity benchmark...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /utf-8 /wd4068 ^
    /I"%ROOT_DIR%\second_brain" ^
    "%ROOT_DIR%\tests\bench_holdout_perplexity.cpp" ^
    "%BUILD_DIR%\popcnt_brain.lib" ^
    /Fe:"%BUILD_DIR%\bench_holdout_perplexity.exe" /Fo:"%BUILD_DIR%\obj_holdout\\" ^
    /link psapi.lib
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Compilation failed.
    exit /b 1
)

echo [*] Running held-out perplexity benchmark...
"%BUILD_DIR%\bench_holdout_perplexity.exe" "%ROOT_DIR%"
exit /b %ERRORLEVEL%

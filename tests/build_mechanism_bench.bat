@echo off
chcp 65001 >nul
setlocal
set "ROOT_DIR=%~dp0.."
set "BUILD_DIR=%ROOT_DIR%\build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%BUILD_DIR%\obj_mech" mkdir "%BUILD_DIR%\obj_mech"

where cl.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 goto :msvc_ok
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
)

:msvc_ok
if not exist "%BUILD_DIR%\ptq_cuda.lib" (
    echo [*] ptq_cuda.lib missing; build it with "build.bat ptq" first.
    exit /b 1
)

echo [*] Compiling mechanism benchmark (all sources, no stale objects)...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /utf-8 /wd4068 ^
    /I"%ROOT_DIR%\include" /I"%ROOT_DIR%\second_brain" /I"%ROOT_DIR%\legacy_ptq" ^
    "%ROOT_DIR%\src\bridge.cpp" ^
    "%ROOT_DIR%\src\vsa_db.cpp" ^
    "%ROOT_DIR%\src\model_mod.cpp" ^
    "%ROOT_DIR%\src\hssa.cpp" ^
    "%ROOT_DIR%\tests\bench_mechanism_ternary_vsa.cpp" ^
    "%BUILD_DIR%\ptq_cuda.lib" ^
    /Fe:"%BUILD_DIR%\bench_mechanism_ternary_vsa.exe" /Fo:"%BUILD_DIR%\obj_mech\\" ^
    /link /STACK:33554432 psapi.lib
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Compilation failed.
    exit /b 1
)

echo [*] Running mechanism benchmark...
"%BUILD_DIR%\bench_mechanism_ternary_vsa.exe"
exit /b %ERRORLEVEL%

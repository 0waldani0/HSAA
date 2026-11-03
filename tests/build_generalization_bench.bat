@echo off
chcp 65001 >nul
setlocal
set "ROOT_DIR=%~dp0.."
set "BUILD_DIR=%ROOT_DIR%\build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%BUILD_DIR%\obj_gen" mkdir "%BUILD_DIR%\obj_gen"

where cl.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 goto :msvc_ok
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
)

:msvc_ok
rem Always rebuild the engine: a cached .lib silently links stale code.
echo [*] Building popcnt_brain.lib...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /EHsc /utf-8 /wd4068 /DBUILDING_POPCNT_BRAIN_LIB /c ^
    "%ROOT_DIR%\second_brain\popcnt_brain.cpp" /Fo:"%BUILD_DIR%\popcnt_brain_lib.obj"
if errorlevel 1 exit /b 1
lib.exe /nologo "%BUILD_DIR%\popcnt_brain_lib.obj" /OUT:"%BUILD_DIR%\popcnt_brain.lib"
if errorlevel 1 exit /b 1

echo [*] Compiling unseen-function benchmark...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /utf-8 /wd4068 ^
    /I"%ROOT_DIR%\second_brain" ^
    "%ROOT_DIR%\tests\bench_unseen_function.cpp" ^
    "%BUILD_DIR%\popcnt_brain.lib" ^
    /Fe:"%BUILD_DIR%\bench_unseen_function.exe" /Fo:"%BUILD_DIR%\obj_gen\\" ^
    /link psapi.lib
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] unseen-function compilation failed.
    exit /b 1
)

echo [*] Compiling zero-shot falsification benchmark...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /utf-8 /wd4068 ^
    /I"%ROOT_DIR%\second_brain" ^
    "%ROOT_DIR%\tests\bench_zeroshot_falsify.cpp" ^
    "%BUILD_DIR%\popcnt_brain.lib" ^
    /Fe:"%BUILD_DIR%\bench_zeroshot_falsify.exe" /Fo:"%BUILD_DIR%\obj_gen\\" ^
    /link psapi.lib
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] zero-shot compilation failed.
    exit /b 1
)

echo [*] Compiling rule-generalization benchmark...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /utf-8 /wd4068 ^
    /I"%ROOT_DIR%\second_brain" ^
    "%ROOT_DIR%\tests\bench_rule_generalization.cpp" ^
    "%BUILD_DIR%\popcnt_brain.lib" ^
    /Fe:"%BUILD_DIR%\bench_rule_generalization.exe" /Fo:"%BUILD_DIR%\obj_gen\\" ^
    /link psapi.lib
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] rule-generalization compilation failed.
    exit /b 1
)

echo [*] Running unseen-function benchmark...
"%BUILD_DIR%\bench_unseen_function.exe"
set "RC_FN=%ERRORLEVEL%"
echo [*] Running zero-shot falsification benchmark...
"%BUILD_DIR%\bench_zeroshot_falsify.exe"
set "RC_ZS=%ERRORLEVEL%"
echo [*] Running rule-generalization benchmark...
"%BUILD_DIR%\bench_rule_generalization.exe"
set "RC_RG=%ERRORLEVEL%"

if not "%RC_FN%"=="0" exit /b %RC_FN%
if not "%RC_ZS%"=="0" exit /b %RC_ZS%
if not "%RC_RG%"=="0" exit /b %RC_RG%
exit /b 0

@echo off
chcp 65001 >nul
setlocal

set "ROOT_DIR=%~dp0"
set "BUILD_DIR=%ROOT_DIR%build"

REM ============================================================================
REM 1. Command Router & Clean
REM ============================================================================
if "%1"=="clean" (
    echo [*] Cleaning build artifacts...
    if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
    if exist "%ROOT_DIR%second_brain\build" rmdir /s /q "%ROOT_DIR%second_brain\build"
    echo [+] Clean complete.
    exit /b 0
)

REM Ensure output build directory exists
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

REM ============================================================================
REM 2. Multi-Tiered MSVC 2022 vcvars64.bat Detection
REM ============================================================================
where cl.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [*] MSVC cl.exe already available in PATH.
    goto :msvc_configured
)

set "VCVARS64="

REM Tier 1: Query vswhere.exe for Visual Studio 2022 installation
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" (
            set "VCVARS64=%%i\VC\Auxiliary\Build\vcvars64.bat"
        )
    )
)

REM Tier 2: Priority fallback paths for standard MSVC 2022 editions
if not defined VCVARS64 (
    if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS64=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS64=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS64=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS64=%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS64=%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    )
)

if defined VCVARS64 (
    echo [*] Initializing MSVC x64 environment: "%VCVARS64%"
    call "%VCVARS64%" >nul 2>nul
)

where cl.exe >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] MSVC cl.exe not available in PATH. Please install Visual Studio 2022 or C++ Build Tools.
    exit /b 1
)

:msvc_configured

REM ============================================================================
REM 3. Compiler Flag Definitions
REM ============================================================================
set "MSVC_CXX_FLAGS=/nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /utf-8 /wd4068"
set "MSVC_C_FLAGS=/nologo /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /utf-8"
set "LINK_FLAGS=/link /STACK:33554432"

REM Route directly if specific target requested
if "%1"=="cubin" goto :build_cubin
if "%1"=="ptq" goto :build_ptq
if "%1"=="brain" goto :build_second_brain
if "%1"=="bridge" goto :build_bridge
if "%1"=="test_bridge" goto :build_bridge
if "%1"=="vsa_db" goto :build_vsa_db
if "%1"=="test_vsa_db" goto :build_vsa_db
if "%1"=="model_mod" goto :build_model_mod
if "%1"=="test_model_mod" goto :build_model_mod
if "%1"=="hssa" goto :build_hssa
if "%1"=="test_hssa" goto :build_hssa
if "%1"=="lab" goto :build_core_lab
if "%1"=="test_bench" goto :build_test_bench
if "%1"=="brain_gates" goto :build_brain_gates
if "%1"=="test_brain_gates" goto :build_brain_gates
if "%1"=="test" goto :build_test
if "%1"=="e2e" goto :build_e2e

REM ============================================================================
REM 4. Target: CUBIN Compilation (CUDA sm_86)
REM ============================================================================
:build_cubin
echo [*] Compiling sm_86 CUDA kernel into build\kernel.cubin...
nvcc -cubin -arch=sm_86 -O3 "%ROOT_DIR%legacy_ptq\kernel.cu" -o "%BUILD_DIR%\kernel.cubin"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] NVCC cubin compilation failed.
    exit /b %ERRORLEVEL%
)
echo [+] Kernel cubin built successfully: "%BUILD_DIR%\kernel.cubin"
if "%1"=="cubin" exit /b 0

REM ============================================================================
REM 5. Target: Legacy PTQ Engine
REM ============================================================================
:build_ptq
if not exist "%BUILD_DIR%\kernel.cubin" (
    echo [*] Kernel cubin missing, compiling cubin first...
    nvcc -cubin -arch=sm_86 -O3 "%ROOT_DIR%legacy_ptq\kernel.cu" -o "%BUILD_DIR%\kernel.cubin"
    if errorlevel 1 exit /b 1
)
echo [*] Compiling Legacy PTQ CUDA Driver Engine...
if exist "%ROOT_DIR%legacy_ptq\ptq_cuda.c" (
    echo [*] Compiling PTQ CUDA modular library...
    cl.exe %MSVC_C_FLAGS% /I"%ROOT_DIR%legacy_ptq" /c "%ROOT_DIR%legacy_ptq\ptq_cuda.c" /Fo:"%BUILD_DIR%\ptq_cuda.obj"
    if errorlevel 1 exit /b 1
    lib.exe /nologo /OUT:"%BUILD_DIR%\ptq_cuda.lib" "%BUILD_DIR%\ptq_cuda.obj"
    if errorlevel 1 exit /b 1
    cl.exe %MSVC_C_FLAGS% /I"%ROOT_DIR%legacy_ptq" "%ROOT_DIR%legacy_ptq\engine.c" "%BUILD_DIR%\ptq_cuda.lib" /Fe:"%BUILD_DIR%\legacy_ptq.exe" /Fo:"%BUILD_DIR%\engine.obj" %LINK_FLAGS%
    if errorlevel 1 exit /b 1
) else (
    cl.exe %MSVC_C_FLAGS% /I"%ROOT_DIR%legacy_ptq" "%ROOT_DIR%legacy_ptq\engine.c" /Fe:"%BUILD_DIR%\legacy_ptq.exe" /Fo:"%BUILD_DIR%\engine.obj" %LINK_FLAGS%
    if errorlevel 1 exit /b 1
)
echo [+] Legacy PTQ engine built: "%BUILD_DIR%\legacy_ptq.exe"
if "%1"=="ptq" exit /b 0

REM ============================================================================
REM 6. Target: Second Brain PopCount Module
REM ============================================================================
:build_second_brain
echo [*] Compiling Second Brain (AVX-512 VPOPCNTDQ)...
cl.exe %MSVC_CXX_FLAGS% /I"%ROOT_DIR%second_brain" "%ROOT_DIR%second_brain\popcnt_brain.cpp" /Fe:"%BUILD_DIR%\popcnt_brain.exe" /Fo:"%BUILD_DIR%\popcnt_brain.obj" %LINK_FLAGS%
if %ERRORLEVEL% NEQ 0 (
    echo [!] AVX-512 popcnt_brain failed, attempting AVX2 fallback...
    cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX2 /openmp /EHsc /utf-8 /wd4068 /I"%ROOT_DIR%second_brain" "%ROOT_DIR%second_brain\popcnt_brain.cpp" /Fe:"%BUILD_DIR%\popcnt_brain.exe" /Fo:"%BUILD_DIR%\popcnt_brain.obj" %LINK_FLAGS%
    if errorlevel 1 exit /b 1
)
echo [+] PopCount Second Brain built: "%BUILD_DIR%\popcnt_brain.exe"

echo [*] Creating popcnt_brain.lib static library...
cl.exe %MSVC_CXX_FLAGS% /DBUILDING_POPCNT_BRAIN_LIB /I"%ROOT_DIR%second_brain" /c "%ROOT_DIR%second_brain\popcnt_brain.cpp" /Fo:"%BUILD_DIR%\popcnt_brain_lib.obj"
if errorlevel 1 exit /b 1
lib.exe /nologo /OUT:"%BUILD_DIR%\popcnt_brain.lib" "%BUILD_DIR%\popcnt_brain_lib.obj"
if errorlevel 1 exit /b 1
if "%1"=="brain" exit /b 0

REM ============================================================================
REM 6B. Target: Hybrid Bridge & Test Harness
REM ============================================================================
:build_bridge
echo [*] Compiling Hybrid Bridge and Test Harness...
if not exist "%BUILD_DIR%\obj_bridge" mkdir "%BUILD_DIR%\obj_bridge"
cl.exe %MSVC_CXX_FLAGS% /I"%ROOT_DIR%include" /I"%ROOT_DIR%legacy_ptq" ^
    "%ROOT_DIR%src\bridge.cpp" ^
    "%ROOT_DIR%tests\test_bridge.cpp" ^
    /Fe:"%BUILD_DIR%\test_bridge.exe" /Fo:"%BUILD_DIR%\obj_bridge\\" %LINK_FLAGS%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Hybrid Bridge compilation failed.
    exit /b %ERRORLEVEL%
)
echo [+] Hybrid Bridge test executable built: "%BUILD_DIR%\test_bridge.exe"
if "%1"=="bridge" exit /b 0
if "%1"=="test_bridge" (
    echo [*] Executing Hybrid Bridge tests...
    "%BUILD_DIR%\test_bridge.exe"
    if errorlevel 1 exit /b 1
    exit /b 0
)

REM ============================================================================
REM 6C. Target: Dynamic VSA DB & Test Harness
REM ============================================================================
:build_vsa_db
echo [*] Compiling Dynamic VSA Database and Test Harness...
if not exist "%BUILD_DIR%\obj_vsa_db" mkdir "%BUILD_DIR%\obj_vsa_db"
cl.exe %MSVC_CXX_FLAGS% /I"%ROOT_DIR%include" ^
    "%ROOT_DIR%src\vsa_db.cpp" ^
    "%ROOT_DIR%tests\test_vsa_db.cpp" ^
    /Fe:"%BUILD_DIR%\test_vsa_db.exe" /Fo:"%BUILD_DIR%\obj_vsa_db\\" %LINK_FLAGS%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Dynamic VSA Database compilation failed.
    exit /b %ERRORLEVEL%
)
echo [+] Dynamic VSA Database test executable built: "%BUILD_DIR%\test_vsa_db.exe"
if "%1"=="vsa_db" exit /b 0
if "%1"=="test_vsa_db" (
    echo [*] Executing Dynamic VSA Database tests...
    "%BUILD_DIR%\test_vsa_db.exe"
    if errorlevel 1 exit /b 1
    exit /b 0
)

REM ============================================================================
REM 6D. Target: Model Modification Subsystem (M3)
REM ============================================================================
:build_model_mod
echo [*] Compiling Model Modification Subsystem and Test Harness...
if not exist "%BUILD_DIR%\obj_model_mod" mkdir "%BUILD_DIR%\obj_model_mod"
cl.exe %MSVC_CXX_FLAGS% /I"%ROOT_DIR%include" /I"%ROOT_DIR%legacy_ptq" ^
    "%ROOT_DIR%src\vsa_db.cpp" ^
    "%ROOT_DIR%src\model_mod.cpp" ^
    "%ROOT_DIR%tests\test_model_mod.cpp" ^
    /Fe:"%BUILD_DIR%\test_model_mod.exe" /Fo:"%BUILD_DIR%\obj_model_mod\\" %LINK_FLAGS%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Model Modification Subsystem compilation failed.
    exit /b %ERRORLEVEL%
)
echo [+] Model Modification test executable built: "%BUILD_DIR%\test_model_mod.exe"
if "%1"=="model_mod" exit /b 0
if "%1"=="test_model_mod" (
    echo [*] Executing Model Modification tests...
    "%BUILD_DIR%\test_model_mod.exe"
    if errorlevel 1 exit /b 1
    exit /b 0
)

REM ============================================================================
REM 6E. Target: HSSA Engine (Milestone M1)
REM ============================================================================
:build_hssa
echo [*] Compiling HSSA Engine (AVX-512)...
if not exist "%BUILD_DIR%\obj_hssa" mkdir "%BUILD_DIR%\obj_hssa"
if not exist "%BUILD_DIR%\popcnt_brain.lib" (
    echo [*] popcnt_brain.lib missing, building second_brain first...
    call :build_second_brain
    if errorlevel 1 exit /b 1
)
if not exist "%BUILD_DIR%\ptq_cuda.lib" (
    echo [*] ptq_cuda.lib missing, building ptq first...
    call :build_ptq
    if errorlevel 1 exit /b 1
)
if not exist "%BUILD_DIR%\bridge.obj" (
    echo [*] Compiling bridge.obj...
    cl.exe %MSVC_CXX_FLAGS% /I"%ROOT_DIR%include" /I"%ROOT_DIR%legacy_ptq" /c "%ROOT_DIR%src\bridge.cpp" /Fo:"%BUILD_DIR%\bridge.obj"
    if errorlevel 1 exit /b 1
)
if not exist "%BUILD_DIR%\vsa_db.obj" (
    echo [*] Compiling vsa_db.obj...
    cl.exe %MSVC_CXX_FLAGS% /I"%ROOT_DIR%include" /c "%ROOT_DIR%src\vsa_db.cpp" /Fo:"%BUILD_DIR%\vsa_db.obj"
    if errorlevel 1 exit /b 1
)
if not exist "%BUILD_DIR%\model_mod.obj" (
    echo [*] Compiling model_mod.obj...
    cl.exe %MSVC_CXX_FLAGS% /I"%ROOT_DIR%include" /I"%ROOT_DIR%legacy_ptq" /c "%ROOT_DIR%src\model_mod.cpp" /Fo:"%BUILD_DIR%\model_mod.obj"
    if errorlevel 1 exit /b 1
)
echo [*] Compiling hssa.obj...
cl.exe %MSVC_CXX_FLAGS% /I"%ROOT_DIR%include" /I"%ROOT_DIR%legacy_ptq" /c "%ROOT_DIR%src\hssa.cpp" /Fo:"%BUILD_DIR%\hssa.obj"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] HSSA Engine compilation failed.
    exit /b %ERRORLEVEL%
)
echo [+] HSSA Engine object compiled: "%BUILD_DIR%\hssa.obj"

echo [*] Linking HSSA Test Harness (test_hssa.exe)...
cl.exe %MSVC_CXX_FLAGS% /I"%ROOT_DIR%include" /I"%ROOT_DIR%second_brain" /I"%ROOT_DIR%legacy_ptq" ^
    "%ROOT_DIR%tests\test_hssa.cpp" ^
    "%BUILD_DIR%\hssa.obj" ^
    "%BUILD_DIR%\bridge.obj" ^
    "%BUILD_DIR%\vsa_db.obj" ^
    "%BUILD_DIR%\model_mod.obj" ^
    "%BUILD_DIR%\popcnt_brain.lib" ^
    "%BUILD_DIR%\ptq_cuda.lib" ^
    /Fe:"%BUILD_DIR%\test_hssa.exe" /Fo:"%BUILD_DIR%\obj_hssa\\" %LINK_FLAGS% psapi.lib
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] HSSA Test Harness build failed.
    exit /b %ERRORLEVEL%
)
echo [+] HSSA test executable built: "%BUILD_DIR%\test_hssa.exe"
if "%1"=="hssa" exit /b 0
if "%1"=="test_hssa" (
    echo [*] Executing HSSA Engine verification tests...
    "%BUILD_DIR%\test_hssa.exe"
    if errorlevel 1 exit /b 1
    exit /b 0
)

REM ============================================================================
REM 7. Target: Core Laboratory Runtime & CLI (M1-M4)
REM ============================================================================
:build_core_lab
if not exist "%BUILD_DIR%\popcnt_brain.lib" (
    echo [*] popcnt_brain.lib missing, building second_brain first...
    call :build_second_brain
    if errorlevel 1 exit /b 1
)
if not exist "%BUILD_DIR%\ptq_cuda.lib" (
    echo [*] ptq_cuda.lib missing, building ptq first...
    call :build_ptq
    if errorlevel 1 exit /b 1
)
echo [*] Compiling Native Research Laboratory CLI (upgrade_lab.exe)...
if not exist "%BUILD_DIR%\obj_lab" mkdir "%BUILD_DIR%\obj_lab"
cl.exe %MSVC_CXX_FLAGS% /I"%ROOT_DIR%include" /I"%ROOT_DIR%second_brain" /I"%ROOT_DIR%legacy_ptq" ^
    "%ROOT_DIR%src\bridge.cpp" ^
    "%ROOT_DIR%src\vsa_db.cpp" ^
    "%ROOT_DIR%src\model_mod.cpp" ^
    "%ROOT_DIR%src\hssa.cpp" ^
    "%ROOT_DIR%src\bench_runner.cpp" ^
    "%ROOT_DIR%src\main_lab.cpp" ^
    "%BUILD_DIR%\popcnt_brain.lib" ^
    "%BUILD_DIR%\ptq_cuda.lib" ^
    /Fe:"%BUILD_DIR%\upgrade_lab.exe" /Fo:"%BUILD_DIR%\obj_lab\\" %LINK_FLAGS% psapi.lib
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] upgrade_lab build failed.
    exit /b %ERRORLEVEL%
)
echo [+] upgrade_lab built: "%BUILD_DIR%\upgrade_lab.exe"
if "%1"=="lab" exit /b 0

REM ============================================================================
REM 7B. Target: Benchmark Runner Test Harness (M4)
REM ============================================================================
:build_test_bench
if not exist "%BUILD_DIR%\popcnt_brain.lib" (
    echo [*] popcnt_brain.lib missing, building second_brain first...
    call :build_second_brain
    if errorlevel 1 exit /b 1
)
echo [*] Compiling Benchmark Runner Test Harness (test_bench_runner.exe)...
if not exist "%BUILD_DIR%\obj_test_bench" mkdir "%BUILD_DIR%\obj_test_bench"
cl.exe %MSVC_CXX_FLAGS% /I"%ROOT_DIR%include" /I"%ROOT_DIR%second_brain" /I"%ROOT_DIR%legacy_ptq" ^
    "%ROOT_DIR%src\bridge.cpp" ^
    "%ROOT_DIR%src\vsa_db.cpp" ^
    "%ROOT_DIR%src\model_mod.cpp" ^
    "%ROOT_DIR%src\bench_runner.cpp" ^
    "%ROOT_DIR%tests\test_bench_runner.cpp" ^
    "%BUILD_DIR%\popcnt_brain.lib" ^
    /Fe:"%BUILD_DIR%\test_bench_runner.exe" /Fo:"%BUILD_DIR%\obj_test_bench\\" %LINK_FLAGS% psapi.lib
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] test_bench_runner compilation failed.
    exit /b %ERRORLEVEL%
)
echo [+] test_bench_runner built: "%BUILD_DIR%\test_bench_runner.exe"
echo [*] Executing Benchmark Runner tests...
"%BUILD_DIR%\test_bench_runner.exe"
exit /b %ERRORLEVEL%

REM ============================================================================
REM 7Z. Target: the second brain's own gates (task training + backoff)
REM ============================================================================
:build_brain_gates
echo [*] Building the second brain's task-training gate...
call "%ROOT_DIR%tests\build_popcnt_selector_tests.bat"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] the task-training gate failed.
    exit /b %ERRORLEVEL%
)
echo [*] Building the second brain's backoff gate...
call "%ROOT_DIR%tests\build_popcnt_backoff_tests.bat"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] the backoff gate failed.
    exit /b %ERRORLEVEL%
)
echo [+] The second brain's gates passed.
if "%1"=="brain_gates" exit /b 0
if "%1"=="test_brain_gates" exit /b 0
goto :build_e2e

REM 8. Target: Native E2E Test Suite
REM ============================================================================
:build_test
REM The second brain's own gates first: they are the newest surface and the
REM cheapest to catch a regression in. A `goto`, not a `call`: :build_brain_gates
REM ends by falling through into the e2e block below, so a `call` would run
REM the gates a second time.
goto :build_brain_gates

:build_e2e
REM The E2E suite links the real engine objects and asserts behaviour. The old
REM tier1..4 runner only inspected source text (for example, whether build.bat
REM contained the word "clean") and never executed a single engine function.
call "%ROOT_DIR%tests\build_tests.bat" run
exit /b %ERRORLEVEL%

echo [SUCCESS] Complete native laboratory built successfully in "%BUILD_DIR%".
exit /b 0

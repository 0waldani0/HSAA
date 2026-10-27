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
set "OUT_DIR=%ROOT_DIR%build\"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
if not exist "%OUT_DIR%obj_e2e" mkdir "%OUT_DIR%obj_e2e"

REM The E2E runner links the real modules, so it needs the two prebuilt
REM libraries. Build them through the top-level dispatcher if they are absent.
REM (build.bat brain / ptq exit before reaching its own E2E block, so this
REM cannot recurse.)
if not exist "%OUT_DIR%popcnt_brain.lib" (
    echo [*] popcnt_brain.lib missing, building it first...
    call "%ROOT_DIR%build.bat" brain
    if errorlevel 1 exit /b 1
)
if not exist "%OUT_DIR%ptq_cuda.lib" (
    echo [*] ptq_cuda.lib missing, building it first...
    call "%ROOT_DIR%build.bat" ptq
    if errorlevel 1 exit /b 1
)

echo [*] Compiling Native E2E Test Suite (real modules, no mocks)...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /utf-8 /EHsc /wd4068 ^
    /I"%ROOT_DIR%include" /I"%ROOT_DIR%second_brain" /I"%ROOT_DIR%legacy_ptq" ^
    "%~dp0e2e_test_runner.cpp" ^
    "%ROOT_DIR%src\bridge.cpp" ^
    "%ROOT_DIR%src\vsa_db.cpp" ^
    "%ROOT_DIR%src\model_mod.cpp" ^
    "%ROOT_DIR%src\hssa.cpp" ^
    "%ROOT_DIR%src\bench_runner.cpp" ^
    "%OUT_DIR%popcnt_brain.lib" ^
    "%OUT_DIR%ptq_cuda.lib" ^
    /Fe:"%OUT_DIR%e2e_runner.exe" ^
    /Fo"%OUT_DIR%obj_e2e\\" ^
    /link /STACK:33554432 psapi.lib

if %ERRORLEVEL% NEQ 0 (
    echo [!] E2E compilation failed with error %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo [+] E2E Test Runner built: "%OUT_DIR%e2e_runner.exe"

if not "%1"=="run" exit /b 0

echo [*] Executing E2E Test Suite...
pushd "%ROOT_DIR%"
"%OUT_DIR%e2e_runner.exe"
set "E2E_RC=%ERRORLEVEL%"
popd
exit /b %E2E_RC%

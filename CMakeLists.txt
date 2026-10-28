cmake_minimum_required(VERSION 3.20)
project(project_upgrade LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ==============================================================================
# Global Optimization & Invariant Compiler Flags
# ==============================================================================
if(MSVC)
    add_compile_options(
        /nologo
        /O2
        /Oi
        /Ot
        /arch:AVX512
        /openmp
        /EHsc
        /utf-8
        /wd4068
    )
    add_link_options(/STACK:33554432)
else()
    add_compile_options(-O3 -march=tigerlake -mavx512f -mavx512bw -mavx512vpopcntdq -fopenmp)
endif()

# Global include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/second_brain
    ${CMAKE_CURRENT_SOURCE_DIR}/legacy_ptq
)

# ==============================================================================
# CUDA sm_86 CUBIN Target (Optional Custom Command via NVCC)
# ==============================================================================
find_program(CUDA_NVCC nvcc HINTS "$ENV{CUDA_PATH}/bin" "/usr/local/cuda/bin")

set(KERNEL_CU "${CMAKE_CURRENT_SOURCE_DIR}/legacy_ptq/kernel.cu")
set(KERNEL_CUBIN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/build/kernel.cubin")
set(KERNEL_CUBIN_BIN "${CMAKE_BINARY_DIR}/kernel.cubin")

if(CUDA_NVCC)
    add_custom_command(
        OUTPUT "${KERNEL_CUBIN_BIN}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}"
        COMMAND ${CUDA_NVCC} -cubin -arch=sm_86 -O3 "${KERNEL_CU}" -o "${KERNEL_CUBIN_BIN}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${KERNEL_CUBIN_BIN}" "${KERNEL_CUBIN_ROOT}"
        DEPENDS "${KERNEL_CU}"
        COMMENT "Compiling CUDA sm_86 kernel into standalone ${KERNEL_CUBIN_BIN}..."
    )
    add_custom_target(generate_kernel_cubin ALL DEPENDS "${KERNEL_CUBIN_BIN}")
else()
    message(STATUS "NVCC not found; building with dynamic CUDA driver probe + AVX-512 CPU fallback.")
endif()

# ==============================================================================
# Legacy PTQ CUDA Engine Executable & Library
# ==============================================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/legacy_ptq/ptq_cuda.c")
    add_library(ptq_cuda STATIC legacy_ptq/ptq_cuda.c)
    target_include_directories(ptq_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/legacy_ptq)
    if(NOT MSVC)
        target_link_libraries(ptq_cuda PUBLIC dl m)
    endif()
    if(TARGET generate_kernel_cubin)
        add_dependencies(ptq_cuda generate_kernel_cubin)
    endif()

    add_executable(legacy_ptq legacy_ptq/engine.c)
    target_include_directories(legacy_ptq PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/legacy_ptq)
    target_link_libraries(legacy_ptq PRIVATE ptq_cuda)
    if(TARGET generate_kernel_cubin)
        add_dependencies(legacy_ptq generate_kernel_cubin)
    endif()
else()
    add_executable(legacy_ptq legacy_ptq/engine.c)
    target_include_directories(legacy_ptq PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/legacy_ptq)
    if(TARGET generate_kernel_cubin)
        add_dependencies(legacy_ptq generate_kernel_cubin)
    endif()
endif()

# ==============================================================================
# Second Brain PopCount Submodule
# ==============================================================================
add_subdirectory(second_brain)

# ==============================================================================
# Hybrid Bridge Test Executable
# ==============================================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bridge.cpp" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/bridge.cpp")
    add_executable(test_bridge src/bridge.cpp tests/test_bridge.cpp)
    target_include_directories(test_bridge PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/legacy_ptq
    )
    if(NOT MSVC)
        target_link_libraries(test_bridge PRIVATE dl pthread)
    endif()
    if(TARGET generate_kernel_cubin)
        add_dependencies(test_bridge generate_kernel_cubin)
    endif()
endif()

# ==============================================================================
# Core Laboratory Library & CLI Executable (Incremental inclusion)
# ==============================================================================
set(CORE_SOURCES
    src/bridge.cpp
    src/vsa_db.cpp
    src/model_mod.cpp
    src/hssa.cpp
    src/bench_runner.cpp
    src/unified_engine.cpp
)

set(ALL_CORE_EXIST TRUE)
foreach(SRC_FILE ${CORE_SOURCES})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FILE}")
        set(ALL_CORE_EXIST FALSE)
    endif()
endforeach()

if(ALL_CORE_EXIST)
    add_library(upgrade_core STATIC ${CORE_SOURCES})
    target_link_libraries(upgrade_core PUBLIC popcnt_brain_static ptq_cuda)
    if(NOT MSVC)
        target_link_libraries(upgrade_core PUBLIC pthread dl gomp)
    endif()
    if(TARGET generate_kernel_cubin)
        add_dependencies(upgrade_core generate_kernel_cubin)
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main_lab.cpp")
        add_executable(upgrade_lab src/main_lab.cpp)
        target_link_libraries(upgrade_lab PRIVATE upgrade_core)
    endif()
endif()

# ==============================================================================
# Standalone Native E2E Test Suite
# ==============================================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e_test_runner.cpp")
    # The second brain's own gates. They include the engine source directly
    # (like the shipped self-test) because the trainer and the backoff tables
    # are internal classes.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_popcnt_selector.cpp")
        add_executable(test_popcnt_selector tests/test_popcnt_selector.cpp)
        target_include_directories(test_popcnt_selector PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/second_brain
        )
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_popcnt_backoff.cpp")
        add_executable(test_popcnt_backoff tests/test_popcnt_backoff.cpp)
        target_include_directories(test_popcnt_backoff PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/second_brain
        )
    endif()

    add_executable(e2e_test_runner tests/e2e_test_runner.cpp)
    target_include_directories(e2e_test_runner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    if(TARGET upgrade_core)
        target_link_libraries(e2e_test_runner PRIVATE upgrade_core)
    else()
        target_link_libraries(e2e_test_runner PRIVATE popcnt_brain_static)
    endif()
    if(TARGET generate_kernel_cubin)
        add_dependencies(e2e_test_runner generate_kernel_cubin)
    endif()
endif()

cmake_minimum_required(VERSION 3.20)
project(popcnt_brain LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2 /Oi /Ot /EHsc /wd4068 /openmp /arch:AVX512)
else()
    add_compile_options(-O3 -march=tigerlake -mavx512f -mavx512bw -mavx512vpopcntdq -fopenmp)
endif()

# Standalone test executable
add_executable(popcnt_brain_test popcnt_brain.cpp)
target_include_directories(popcnt_brain_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Static Library for linking with other engines or LLM runtimes
add_library(popcnt_brain_static STATIC popcnt_brain.cpp)
target_include_directories(popcnt_brain_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(popcnt_brain_static PRIVATE BUILDING_POPCNT_BRAIN_LIB)

# Shared Library (DLL) for Python ctypes / C FFI bindings
add_library(popcnt_brain_shared SHARED popcnt_brain.cpp)
target_include_directories(popcnt_brain_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(popcnt_brain_shared PRIVATE BUILDING_POPCNT_BRAIN_DLL)

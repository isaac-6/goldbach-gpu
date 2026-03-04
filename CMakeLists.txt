cmake_minimum_required(VERSION 3.18)
project(GoldbachGPU LANGUAGES CXX CUDA)

# --- Options ---
option(BUILD_LEGACY "Build legacy GPU verifiers (gpu)" OFF)

# --- Global Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optimization Flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -Wall -Wextra")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -Xcompiler=-Wall,-Wextra")

# Target Architecture for RTX 3070 (Compute Capability 8.6)
# Use "all" or specific "8.6"
# set(CMAKE_CUDA_ARCHITECTURES 86-real)

# Default to building only for the GPU currently installed in the machine (fastest compile).
# HPC Users: If compiling on a head node without the target GPU, override this by running:
# cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;90" (for A100 and H100)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# --- Dependencies ---
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# Manual check for GMP (as it doesn't always have a find_package)
find_path(GMP_INCLUDE_DIR NAMES gmp.h)
find_library(GMP_LIBRARIES NAMES gmp)

if(NOT GMP_LIBRARIES)
    message(FATAL_ERROR "GMP library not found. Please install libgmp-dev.")
endif()

# --- Include Directories ---
include_directories(include ${GMP_INCLUDE_DIR})

# --- Internal Library ---
# Shared logic for bitsets and sieving
add_library(goldbach_lib 
    src/prime_bitset.cpp
    src/segmented_sieve.cpp
)
target_link_libraries(goldbach_lib PUBLIC OpenMP::OpenMP_CXX)

# --- Executables ---

# Multi-GPU
add_executable(goldbach_gpu3 src/goldbach_gpu3.cu)
target_link_libraries(goldbach_gpu3 PRIVATE goldbach_lib)
set_target_properties(goldbach_gpu3 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")


# Multi-GPU 5a (Stream-Based Optimization)
add_executable(goldbach_gpu5a src/goldbach_gpu5a.cu)
target_link_libraries(goldbach_gpu5a PRIVATE goldbach_lib)
set_target_properties(goldbach_gpu5a PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")


# Multi-GPU 5b (Stream-Based Optimization)
add_executable(goldbach_gpu5b src/goldbach_gpu5b.cu)
target_link_libraries(goldbach_gpu5b PRIVATE goldbach_lib)

# Enable GPU-vs-CPU segment sieve validation
target_compile_definitions(goldbach_gpu5b PRIVATE DEBUG_SEGMENT_SIEVE)

set_target_properties(goldbach_gpu5b PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Multi-GPU 5c (Stream-Based Optimization)
add_executable(goldbach_gpu5c src/goldbach_gpu5c.cu)
target_link_libraries(goldbach_gpu5c PRIVATE goldbach_lib)
set_target_properties(goldbach_gpu5c PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# 1. Global Bitset Verifier (Baseline)
add_executable(goldbach_gpu2 src/goldbach_gpu2.cu)
target_link_libraries(goldbach_gpu2 PRIVATE goldbach_lib)

# 2. Arbitrary Precision Checker (Requires GMP)
add_executable(big_check src/big_check.cpp)
target_link_libraries(big_check PRIVATE goldbach_lib ${GMP_LIBRARIES} OpenMP::OpenMP_CXX)

# 3. Single 64-bit Number Checker
add_executable(single_check src/single_check.cu)
target_link_libraries(single_check PRIVATE goldbach_lib)

# 4. CPU Baseline
add_executable(cpu_goldbach src/cpu_goldbach.cpp)
target_link_libraries(cpu_goldbach PRIVATE goldbach_lib OpenMP::OpenMP_CXX)

# --- Installation/Output ---
# (Optional) Set output directory to a bin folder
set_target_properties(goldbach_gpu3 goldbach_gpu2 big_check single_check cpu_goldbach 
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# --- Legacy Tools (optional) ---
if(BUILD_LEGACY)
    message(STATUS "Building legacy tools...")

    # 0a. Original GPU Verifier
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/legacy/goldbach_gpu.cu")
        add_executable(goldbach_gpu legacy/goldbach_gpu.cu)
        target_link_libraries(goldbach_gpu PRIVATE goldbach_lib)
        set_target_properties(goldbach_gpu PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/legacy")
    endif()

    # 0. Flagship Segmented Verifier
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/legacy/goldbach_gpu3a.cu")
        add_executable(goldbach_gpu3a legacy/goldbach_gpu3a.cu)
        target_link_libraries(goldbach_gpu3a PRIVATE goldbach_lib)
        set_target_properties(goldbach_gpu3a PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/legacy")
    endif()

    # 0b. Fully optimised GPU3c (byte/bitset selectable, small-bitset toggle)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/legacy/goldbach_gpu3c.cu")
        add_executable(goldbach_gpu3c legacy/goldbach_gpu3c.cu)
        target_link_libraries(goldbach_gpu3c PRIVATE goldbach_lib)
        set_target_properties(goldbach_gpu3c PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/legacy")
    endif()
endif()
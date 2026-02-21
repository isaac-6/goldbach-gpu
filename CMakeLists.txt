cmake_minimum_required(VERSION 3.18)
project(goldbach-gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Output binaries into build/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Shared sieve library used by both executables
add_library(sieve_lib STATIC src/segmented_sieve.cpp)

# CPU Goldbach executable
add_executable(goldbach src/cpu_goldbach.cpp)
target_link_libraries(goldbach PRIVATE sieve_lib)

# Sieve test executable
add_executable(test_sieve src/test_sieve.cpp)
target_link_libraries(test_sieve PRIVATE sieve_lib)

# GPU Goldbach executable
add_executable(goldbach_gpu src/goldbach_gpu.cu)
target_link_libraries(goldbach_gpu PRIVATE sieve_lib)
set_property(TARGET goldbach_gpu PROPERTY CUDA_ARCHITECTURES native)

# Single number Goldbach checker with Miller-Rabin
add_executable(single_check src/single_check.cu)
target_link_libraries(single_check PRIVATE sieve_lib)
set_property(TARGET single_check PROPERTY CUDA_ARCHITECTURES native)

# Bitset library
add_library(bitset_lib STATIC src/prime_bitset.cpp)
target_include_directories(bitset_lib PUBLIC src)

# Bitset test
add_executable(test_bitset src/test_bitset.cpp)
target_link_libraries(test_bitset PRIVATE bitset_lib)
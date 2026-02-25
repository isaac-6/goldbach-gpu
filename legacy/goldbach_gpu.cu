// goldbach_gpu.cu
// GPU Goldbach verification — one CUDA thread per even number n.
// Cross-check uses random sampling against CPU oracle (not full scan).

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <chrono>
#include <random>

// Defined in segmented_sieve.cpp
std::vector<char> segmented_sieve(uint64_t low, uint64_t high);

// -------------------------------------------------------
// CUDA error checking macro.
// Prints file, line, and error message on any CUDA failure.
// -------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " — " << cudaGetErrorString(err) << "\n";         \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

// -------------------------------------------------------
// Kernel: one thread per even number n.
// Scans p from 2 to n/2, checks if p and n-p are both prime.
// Writes 1 to d_results[tid] if a partition is found, 0 if not.
// -------------------------------------------------------
__global__ void goldbach_kernel(
    const char* d_is_prime,
    char*       d_results,
    uint64_t    start,
    uint64_t    count)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    uint64_t n = start + tid * 2;

    for (uint64_t p = 2; p <= n / 2; p++) {
        if (d_is_prime[p] && d_is_prime[n - p]) {
            d_results[tid] = 1;
            return;
        }
    }
    d_results[tid] = 0;
}

// -------------------------------------------------------
// Cross-check: compare GPU results vs CPU oracle
// for a random sample of `sample_size` even numbers.
// Returns number of mismatches found.
// -------------------------------------------------------
uint64_t cross_check(
    const std::vector<char>& results,   // GPU results (full array)
    const std::vector<char>& is_prime,  // prime lookup table
    uint64_t start,                     // first even number checked
    uint64_t count,                     // total even numbers checked
    uint64_t sample_size)               // how many to sample
{
    // Use a fixed seed so results are reproducible
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> dist(0, count - 1);

    uint64_t mismatches = 0;

    for (uint64_t s = 0; s < sample_size; s++) {
        uint64_t i = dist(rng);       // random index into results
        uint64_t n = start + i * 2;  // corresponding even number

        // CPU oracle for this n
        bool cpu_result = false;
        for (uint64_t p = 2; p <= n / 2; p++) {
            if (is_prime[p] && is_prime[n - p]) {
                cpu_result = true;
                break;
            }
        }

        bool gpu_result = (results[i] == 1);

        if (cpu_result != gpu_result) {
            std::cout << "MISMATCH at n=" << n
                      << " cpu=" << cpu_result
                      << " gpu=" << gpu_result << "\n";
            mismatches++;
        }
    }

    return mismatches;
}

int main() {
    // -------------------------------------------------------
    // Configuration
    // -------------------------------------------------------
    const uint64_t LIMIT       = 1'000'000'000;  // 10^9
    const uint64_t SAMPLE_SIZE = 1'000;          // random cross-check samples

    // -------------------------------------------------------
    // Step 1: Build prime table on CPU
    // -------------------------------------------------------
    std::cout << "Building prime table up to " << LIMIT << " on CPU...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<char> is_prime = segmented_sieve(0, LIMIT);

    auto t1 = std::chrono::high_resolution_clock::now();
    double sieve_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Sieve done in " << sieve_ms << " ms\n";

    // -------------------------------------------------------
    // Step 2: Copy prime table to GPU
    // -------------------------------------------------------
    uint64_t table_size = LIMIT + 1;

    char* d_is_prime = nullptr;
    CUDA_CHECK(cudaMalloc(&d_is_prime, table_size));
    CUDA_CHECK(cudaMemcpy(d_is_prime, is_prime.data(), table_size,
                          cudaMemcpyHostToDevice));
    std::cout << "Prime table copied to GPU ("
              << table_size / 1024 / 1024 << " MB)\n";

    // -------------------------------------------------------
    // Step 3: Allocate results array on GPU
    // -------------------------------------------------------
    uint64_t count = (LIMIT - 4) / 2 + 1;
    char* d_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_results, count));

    // -------------------------------------------------------
    // Step 4: Launch kernel
    // -------------------------------------------------------
    int      threads_per_block = 256;
    uint64_t blocks = (count + threads_per_block - 1) / threads_per_block;

    std::cout << "Launching kernel over " << count << " even numbers...\n";

    auto t2 = std::chrono::high_resolution_clock::now();

    goldbach_kernel<<<(uint32_t)blocks, threads_per_block>>>(
        d_is_prime, d_results, 4, count);

    CUDA_CHECK(cudaDeviceSynchronize());

    auto t3 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // -------------------------------------------------------
    // Step 5: Copy results back to CPU
    // -------------------------------------------------------
    std::vector<char> results(count);
    CUDA_CHECK(cudaMemcpy(results.data(), d_results, count,
                          cudaMemcpyDeviceToHost));

    // -------------------------------------------------------
    // Step 6: Scan for failures
    // -------------------------------------------------------
    uint64_t failures = 0;
    for (uint64_t i = 0; i < count; i++) {
        if (results[i] == 0) {
            std::cout << "FAIL: Goldbach fails at n = " << 4 + i * 2 << "\n";
            failures++;
        }
    }

    // -------------------------------------------------------
    // Step 7: Random cross-check vs CPU oracle
    // -------------------------------------------------------
    std::cout << "Cross-checking " << SAMPLE_SIZE
              << " random samples against CPU oracle...\n";
    uint64_t mismatches = cross_check(results, is_prime, 4, count, SAMPLE_SIZE);

    // -------------------------------------------------------
    // Step 8: Summary
    // -------------------------------------------------------
    std::cout << "\n--- Summary ---\n";
    std::cout << "Even numbers checked : " << count      << "\n";
    std::cout << "Goldbach failures    : " << failures   << "\n";
    std::cout << "CPU/GPU mismatches   : " << mismatches << "\n";
    std::cout << "Sieve time  (CPU)    : " << sieve_ms   << " ms\n";
    std::cout << "Kernel time (GPU)    : " << gpu_ms     << " ms\n";

    if (failures == 0 && mismatches == 0)
        std::cout << "\nAll even numbers up to " << LIMIT
                  << " satisfy Goldbach, CPU and GPU agree on sample. ✓\n";

    // -------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------
    CUDA_CHECK(cudaFree(d_is_prime));
    CUDA_CHECK(cudaFree(d_results));

    return (failures == 0 && mismatches == 0) ? 0 : 1;
}

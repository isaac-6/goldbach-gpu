// goldbach_gpu.cu
// First GPU implementation of Goldbach verification.
// Strategy: one CUDA thread per even number n.
// Each thread scans p from 2 to n/2 and checks if both p and n-p are prime.
// This is a correctness prototype — performance optimizations come later.

#include <cuda_runtime.h>   // CUDA runtime API (cudaMalloc, cudaMemcpy, etc.)
#include <cstdint>
#include <vector>
#include <iostream>
#include <chrono>

// Declared here, defined in segmented_sieve.cpp.
// Builds a primality lookup table for [low, high].
// Returns a vector<char> where result[i] = 1 iff (low + i) is prime.
std::vector<char> segmented_sieve(uint64_t low, uint64_t high);

// -------------------------------------------------------
// CUDA error checking macro.
// Every CUDA call can fail silently without this.
// This wraps each call and prints the error + line number if it fails.
// Usage: CUDA_CHECK(cudaMalloc(...));
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
// Kernel: goldbach_kernel
//
// Each thread handles exactly one even number n and determines
// whether it has at least one Goldbach partition (p + q = n).
//
// Parameters:
//   d_is_prime  - prime lookup table in GPU global memory
//                 d_is_prime[k] = 1 iff k is prime
//                 indexed from 0 up to LIMIT (inclusive)
//   d_results   - output array in GPU global memory
//                 d_results[tid] = 1 if Goldbach holds for this n
//                 d_results[tid] = 0 if no partition found (failure)
//   start       - the first even number to check (always 4)
//   count       - total number of even numbers to check
// -------------------------------------------------------
__global__ void goldbach_kernel(
    const char* d_is_prime,
    char*       d_results,
    uint64_t    start,
    uint64_t    count)
{
    // Compute this thread's unique index.
    // blockIdx.x  = which block this thread belongs to
    // blockDim.x  = number of threads per block
    // threadIdx.x = this thread's position within its block
    // Together they give a unique tid across the entire grid.
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard: the grid may have more threads than even numbers to check.
    // Extra threads must do nothing and exit immediately.
    if (tid >= count) return;

    // Compute the even number this thread is responsible for.
    // tid=0 → n=4, tid=1 → n=6, tid=2 → n=8, etc.
    uint64_t n = start + tid * 2;

    // Scan all p from 2 to n/2.
    // We only need p <= n/2 because pairs are unordered:
    // (p, q) and (q, p) are the same partition.
    // If p is prime and n-p is also prime, we found a partition.
    for (uint64_t p = 2; p <= n / 2; p++) {
        if (d_is_prime[p] && d_is_prime[n - p]) {
            // Found a valid partition — Goldbach holds for this n.
            d_results[tid] = 1;
            return;  // early exit, no need to keep scanning
        }
    }

    // No partition found for this n — this would be a counterexample
    // to Goldbach's conjecture (none have ever been found).
    d_results[tid] = 0;
}

int main() {
    // -------------------------------------------------------
    // Configuration
    // Start small (10^6) to validate correctness.
    // Once confirmed correct, we'll push to 10^8, 10^9, etc.
    // -------------------------------------------------------
    const uint64_t LIMIT = 100'000'000;

    // -------------------------------------------------------
    // Step 1: Build prime lookup table on CPU.
    // We use the CPU segmented sieve as our trusted oracle.
    // is_prime[k] = 1 iff k is prime, for k in [0, LIMIT].
    // -------------------------------------------------------
    std::cout << "Building prime table up to " << LIMIT << " on CPU...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<char> is_prime = segmented_sieve(0, LIMIT);

    auto t1 = std::chrono::high_resolution_clock::now();
    double sieve_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Sieve done in " << sieve_ms << " ms\n";

    // -------------------------------------------------------
    // Step 2: Copy prime table from CPU RAM to GPU VRAM.
    // cudaMalloc allocates memory on the GPU.
    // cudaMemcpy transfers data between CPU and GPU.
    // -------------------------------------------------------
    uint64_t table_size = LIMIT + 1;  // one byte per number in [0, LIMIT]

    char* d_is_prime = nullptr;  // d_ prefix = "device" (GPU) pointer
    CUDA_CHECK(cudaMalloc(&d_is_prime, table_size));
    CUDA_CHECK(cudaMemcpy(d_is_prime, is_prime.data(), table_size,
                          cudaMemcpyHostToDevice));
    std::cout << "Prime table copied to GPU ("
              << table_size / 1024 << " KB)\n";

    // -------------------------------------------------------
    // Step 3: Allocate results array on GPU.
    // One byte per even number we're checking.
    // Will be 1 (pass) or 0 (fail) after the kernel runs.
    // -------------------------------------------------------
    uint64_t count = (LIMIT - 4) / 2 + 1;  // number of even n in [4, LIMIT]
    char* d_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_results, count));

    // -------------------------------------------------------
    // Step 4: Configure and launch the kernel.
    //
    // CUDA organizes threads into blocks, and blocks into a grid.
    // We choose 256 threads per block (a standard starting point).
    // We need enough blocks to cover all `count` even numbers.
    //
    // Example for LIMIT=1,000,000:
    //   count = 499,999 even numbers
    //   blocks = ceil(499999 / 256) = 1954 blocks
    //   total threads = 1954 * 256 = 500,224
    //   (the extra 225 threads hit the `tid >= count` guard)
    // -------------------------------------------------------
    int threads_per_block = 256;
    int blocks = (int)((count + threads_per_block - 1) / threads_per_block);

    std::cout << "Launching " << blocks << " blocks x "
              << threads_per_block << " threads ("
              << (uint64_t)blocks * threads_per_block << " total threads)\n";
    std::cout << "Checking " << count << " even numbers...\n";

    auto t2 = std::chrono::high_resolution_clock::now();

    // Launch the kernel.
    // <<<blocks, threads_per_block>>> is CUDA syntax for the launch config.
    goldbach_kernel<<<blocks, threads_per_block>>>(
        d_is_prime,
        d_results,
        4,      // start: first even number to check
        count   // how many even numbers to check
    );

    // Wait for all GPU threads to finish before we read results.
    // Without this, we might read d_results before the kernel is done.
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t3 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // -------------------------------------------------------
    // Step 5: Copy results back from GPU to CPU and check them.
    // -------------------------------------------------------
    std::vector<char> results(count);
    CUDA_CHECK(cudaMemcpy(results.data(), d_results, count,
                          cudaMemcpyDeviceToHost));

    // -------------------------------------------------------
    // Step 6: Cross-check GPU results against CPU oracle.
    // For every even n, recompute on CPU and compare.
    // This catches any subtle GPU bugs before we scale up.
    // We only do this for small LIMIT — too slow for large ranges.
    // -------------------------------------------------------
    std::cout << "Cross-checking GPU results against CPU oracle...\n";
    uint64_t mismatches = 0;
    uint64_t failures   = 0;

    for (uint64_t i = 0; i < count; i++) {
        uint64_t n = 4 + i * 2;

        // CPU oracle: scan p from 2 to n/2
        bool cpu_result = false;
        for (uint64_t p = 2; p <= n / 2; p++) {
            if (is_prime[p] && is_prime[n - p]) {
                cpu_result = true;
                break;
            }
        }

        bool gpu_result = (results[i] == 1);

        // Flag any disagreement between CPU and GPU
        if (cpu_result != gpu_result) {
            std::cout << "MISMATCH at n=" << n
                      << " cpu=" << cpu_result
                      << " gpu=" << gpu_result << "\n";
            mismatches++;
        }

        // Flag any Goldbach failure (both CPU and GPU agree it failed)
        if (!gpu_result) {
            std::cout << "FAIL: Goldbach fails at n = " << n << "\n";
            failures++;
        }
    }

    // -------------------------------------------------------
    // Step 7: Print summary
    // -------------------------------------------------------
    std::cout << "\n--- Summary ---\n";
    std::cout << "Even numbers checked : " << count      << "\n";
    std::cout << "Goldbach failures    : " << failures   << "\n";
    std::cout << "CPU/GPU mismatches   : " << mismatches << "\n";
    std::cout << "Sieve time  (CPU)    : " << sieve_ms   << " ms\n";
    std::cout << "Kernel time (GPU)    : " << gpu_ms     << " ms\n";

    if (failures == 0 && mismatches == 0)
        std::cout << "\nAll even numbers up to " << LIMIT
                  << " satisfy Goldbach, CPU and GPU agree. ✓\n";

    // -------------------------------------------------------
    // Cleanup: free GPU memory
    // Always free what you allocate — good habit even at program end.
    // -------------------------------------------------------
    CUDA_CHECK(cudaFree(d_is_prime));
    CUDA_CHECK(cudaFree(d_results));

    return (failures == 0 && mismatches == 0) ? 0 : 1;
}

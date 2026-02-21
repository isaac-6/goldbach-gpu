// goldbach_gpu2.cu
// GPU Goldbach range verifier using compact prime bitset.
// 16x less VRAM than byte array — extends range to 10^11 on RTX 3070.
// Processes even numbers in batches to avoid VRAM overflow.
// Stops and reports the first counterexample if one is found.

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <chrono>
#include <climits>
#include "prime_bitset.hpp"

using namespace goldbach;

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
// Device function: primality lookup in the compact bitset.
// Returns false for any n outside [3, limit] or even n.
// -------------------------------------------------------
__device__ bool device_is_prime(
    const uint64_t* __restrict__ d_words,
    uint64_t n,
    uint64_t limit)
{
    if (n < 2)         return false;
    if (n == 2)        return true;
    if ((n & 1) == 0)  return false;  // even → not prime
    if (n > limit)     return false;  // outside bitset range

    // Encoding: bit index = (n - 3) / 2
    uint64_t bit_pos  = (n - 3) / 2;
    uint64_t word_idx = bit_pos / 64;
    uint64_t bit_idx  = bit_pos % 64;

    return (d_words[word_idx] >> bit_idx) & 1ULL;
}

// -------------------------------------------------------
// Kernel: one thread per even number n in the batch.
//
// For each n, scans p from 2 to min(n/2, limit).
// Clamping p to limit is critical — without it, p can
// exceed the bitset bounds causing out-of-range reads.
//
// Records the first failing n via atomicMin so we get
// the exact counterexample if Goldbach ever fails.
// -------------------------------------------------------
__global__ void goldbach_bitset_kernel(
    const uint64_t*     __restrict__ d_words,
    uint64_t            batch_start,
    uint64_t            batch_count,
    uint64_t            limit,
    unsigned long long* __restrict__ d_failures,
    // uint64_t*           __restrict__ d_first_fail)
    unsigned long long*           __restrict__ d_first_fail)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_count) return;

    // Early exit if a failure was already found by another thread
    // if (*d_first_fail != UINT64_MAX) return;
    if (*d_first_fail != ULLONG_MAX) return;

    uint64_t n = batch_start + tid * 2;

    // Clamp p_max to limit — never read outside the bitset.
    // Without this, p can exceed limit when n is close to LIMIT,
    // causing device_is_prime to read out-of-bounds GPU memory.
    uint64_t p_max = min(n / 2, limit);

    for (uint64_t p = 2; p <= p_max; p++) {
        if (device_is_prime(d_words, p, limit) &&
            device_is_prime(d_words, n - p, limit)) {
            return;  // partition found — Goldbach holds for n
        }
    }

    // No partition found — record failure
    atomicAdd(d_failures, 1ULL);
    // atomicMin(d_first_fail, n);  // track the smallest failing n
    atomicMin(d_first_fail, (unsigned long long)n);
}

int main() {
    // -------------------------------------------------------
    // Configuration
    // -------------------------------------------------------
    const uint64_t LIMIT      = 10'000'000'000ULL;  // 10^10
    const uint64_t BATCH_SIZE = 100'000'000ULL;      // 10^8 per batch

    std::cout << "Goldbach range verifier (GPU bitset)\n";
    std::cout << "Checking all even n in [4, " << LIMIT << "]\n\n";

    // -------------------------------------------------------
    // Step 1: Build compact prime bitset on CPU
    // -------------------------------------------------------
    std::cout << "Building prime bitset up to " << LIMIT << "...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    PrimeBitset bitset = build_prime_bitset(LIMIT);

    auto t1 = std::chrono::high_resolution_clock::now();
    double sieve_ms = std::chrono::duration<double,
                      std::milli>(t1 - t0).count();

    std::cout << "Bitset built in " << sieve_ms << " ms\n";
    std::cout << "Bitset memory : " << bitset.memory_bytes() / 1024 / 1024
              << " MB (vs " << (LIMIT + 1) / 1024 / 1024
              << " MB for byte array)\n\n";

    // -------------------------------------------------------
    // Step 2: Copy bitset to GPU once — reused for all batches
    // -------------------------------------------------------
    uint64_t bitset_bytes = bitset.word_count() * sizeof(uint64_t);
    uint64_t* d_words = nullptr;
    CUDA_CHECK(cudaMalloc(&d_words, bitset_bytes));
    CUDA_CHECK(cudaMemcpy(d_words, bitset.data(), bitset_bytes,
                          cudaMemcpyHostToDevice));
    std::cout << "Bitset copied to GPU ("
              << bitset_bytes / 1024 / 1024 << " MB)\n\n";

    // -------------------------------------------------------
    // Step 3: Allocate GPU result variables
    // d_failures   : count of even numbers with no partition
    // d_first_fail : smallest n that failed (UINT64_MAX = none)
    // -------------------------------------------------------
    unsigned long long* d_failures   = nullptr;
    // uint64_t*           d_first_fail = nullptr;
    unsigned long long* d_first_fail = nullptr;

    CUDA_CHECK(cudaMalloc(&d_failures,   sizeof(unsigned long long)));
    // CUDA_CHECK(cudaMalloc(&d_first_fail, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_first_fail, sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemset(d_failures, 0, sizeof(unsigned long long)));

    // Initialize first_fail to UINT64_MAX so atomicMin works correctly
    // uint64_t max_val = UINT64_MAX;
    // CUDA_CHECK(cudaMemcpy(d_first_fail, &max_val, sizeof(uint64_t),
                        //   cudaMemcpyHostToDevice));
    unsigned long long max_val = ULLONG_MAX;
    CUDA_CHECK(cudaMemcpy(d_first_fail, &max_val,
                          sizeof(unsigned long long),
                          cudaMemcpyHostToDevice));

    // -------------------------------------------------------
    // Step 4: Process even numbers in batches
    // Each batch launches BATCH_SIZE threads on the GPU.
    // The bitset stays in VRAM the whole time.
    // -------------------------------------------------------
    int      threads_per_block = 256;
    uint64_t total_count       = (LIMIT - 4) / 2 + 1;
    uint64_t processed         = 0;
    uint64_t batch_start       = 4;  // first even number to check

    std::cout << "Processing " << total_count
              << " even numbers in batches of " << BATCH_SIZE << "...\n";

    auto t2 = std::chrono::high_resolution_clock::now();

    while (batch_start <= LIMIT) {
        uint64_t remaining   = (LIMIT - batch_start) / 2 + 1;
        uint64_t batch_count = std::min(BATCH_SIZE, remaining);

        uint64_t blocks = (batch_count + threads_per_block - 1)
                          / threads_per_block;

        goldbach_bitset_kernel<<<(uint32_t)blocks, threads_per_block>>>(
            d_words,
            batch_start,
            batch_count,
            LIMIT,
            d_failures,
            d_first_fail);

        CUDA_CHECK(cudaDeviceSynchronize());

        processed   += batch_count;
        batch_start += batch_count * 2;

        // Progress indicator
        double pct = 100.0 * processed / total_count;
        std::cout << "  Progress: " << processed << " / "
                  << total_count << " ("
                  << pct << "%)\r" << std::flush;
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double,
                    std::milli>(t3 - t2).count();

    std::cout << "\n";

    // -------------------------------------------------------
    // Step 5: Copy results back from GPU
    // -------------------------------------------------------
    unsigned long long failures   = 0;
    // uint64_t           first_fail = 0;
    unsigned long long first_fail = 0;

    CUDA_CHECK(cudaMemcpy(&failures, d_failures,
                          sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    // CUDA_CHECK(cudaMemcpy(&first_fail, d_first_fail,
    //                       sizeof(uint64_t),
    //                       cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&first_fail, d_first_fail,
                      sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));

    // -------------------------------------------------------
    // Step 6: Summary
    // -------------------------------------------------------
    std::cout << "\n--- Summary ---\n";
    std::cout << "Even numbers checked : " << total_count << "\n";
    std::cout << "Failures             : " << failures   << "\n";

    if (failures > 0)
        std::cout << "First failure at n   : " << first_fail
                  << "  ← potential counterexample!\n";

    std::cout << "Sieve time  (CPU)    : " << sieve_ms << " ms\n";
    std::cout << "Kernel time (GPU)    : " << gpu_ms   << " ms\n";
    std::cout << "Total                : " << sieve_ms + gpu_ms
              << " ms\n";

    if (failures == 0)
        std::cout << "\nAll even numbers up to " << LIMIT
                  << " satisfy Goldbach. ✓\n";

    // -------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------
    CUDA_CHECK(cudaFree(d_words));
    CUDA_CHECK(cudaFree(d_failures));
    CUDA_CHECK(cudaFree(d_first_fail));

    return (failures == 0) ? 0 : 1;
}
// single_check.cu
// Goldbach verification for a single large even number n.
// Uses Miller-Rabin primality test on GPU.
// Processes primes in chunks to avoid RAM/VRAM overflow.
// Works for any even n up to ~10^15 on consumer hardware.

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>

// Defined in segmented_sieve.cpp
std::vector<char> segmented_sieve(uint64_t low, uint64_t high);

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
// 128-bit modular multiplication — prevents overflow
// -------------------------------------------------------
__device__ uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
    return (__uint128_t)a * b % m;
}

// -------------------------------------------------------
// Modular exponentiation: (base^exp) % mod
// -------------------------------------------------------
__device__ uint64_t powmod(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1)
            result = mulmod(result, base, mod);
        base = mulmod(base, base, mod);
        exp >>= 1;
    }
    return result;
}

// -------------------------------------------------------
// Miller-Rabin single witness test
// -------------------------------------------------------
__device__ bool miller_rabin_witness(uint64_t n, uint64_t a,
                                     uint64_t d, int r) {
    uint64_t x = powmod(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (int i = 0; i < r - 1; i++) {
        x = mulmod(x, x, n);
        if (x == n - 1) return true;
    }
    return false;
}

// -------------------------------------------------------
// Deterministic Miller-Rabin for all 64-bit integers.
// Uses 12 witnesses — no false positives for any uint64_t.
// -------------------------------------------------------
__device__ bool is_prime_miller_rabin(uint64_t n) {
    if (n < 2)  return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;

    // Write n-1 as 2^r * d with d odd
    uint64_t d = n - 1;
    int r = 0;
    while (d % 2 == 0) { d /= 2; r++; }

    // 12 witnesses sufficient for all n < 3.3 * 10^24
    const uint64_t witnesses[] =
        {2,3,5,7,11,13,17,19,23,29,31,37};

    for (int i = 0; i < 12; i++) {
        uint64_t a = witnesses[i];
        if (a >= n) continue;
        if (!miller_rabin_witness(n, a, d, r))
            return false;
    }
    return true;
}

// -------------------------------------------------------
// Kernel: each thread checks one prime p from the chunk.
// Computes q = n - p, tests q with Miller-Rabin.
// Records first valid partition found.
// -------------------------------------------------------
__global__ void goldbach_single_kernel(
    const uint64_t* __restrict__ d_primes,
    uint64_t        prime_count,
    uint64_t        n,
    int*            __restrict__ d_found,
    uint64_t*       __restrict__ d_p_out,
    uint64_t*       __restrict__ d_q_out)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= prime_count) return;

    // Coherent early exit if another thread already found a partition
    if (atomicAdd(d_found, 0) != 0) return;

    uint64_t p = d_primes[tid];
    uint64_t q = n - p;

    // Only check p <= q to avoid duplicate pairs
    if (q < p) return;

    if (is_prime_miller_rabin(q)) {
        // Atomically claim the result slot
        if (atomicExch(d_found, 1) == 0) {
            *d_p_out = p;
            *d_q_out = q;
        }
    }
}

// -------------------------------------------------------
// Extract prime list from a segmented sieve result.
// Appends primes in [low, high] to the output vector.
// -------------------------------------------------------
void extract_primes(const std::vector<char>& sieve,
                    uint64_t low,
                    std::vector<uint64_t>& out) {
    for (uint64_t i = 0; i < sieve.size(); i++) {
        if (sieve[i]) out.push_back(low + i);
    }
}

void check_single(uint64_t n) {
    if (n < 4 || n % 2 != 0) {
        std::cerr << "Error: n must be even and >= 4\n";
        return;
    }

    std::cout << "Checking Goldbach for n = " << n << "\n";

    // -------------------------------------------------------
    // Chunk size: how many numbers we sieve at once on CPU.
    // 10^8 per chunk = ~100 MB, fits easily in RAM.
    // Adjust down if you hit memory pressure.
    // -------------------------------------------------------
    const uint64_t CHUNK_SIZE    = 100'000'000ULL;  // sieve chunk
    const uint64_t MAX_PRIMES    = 10'000'000ULL;   // primes per GPU launch

    // -------------------------------------------------------
    // Allocate GPU buffers once, reuse across chunks
    // -------------------------------------------------------
    uint64_t* d_primes = nullptr;
    int*      d_found  = nullptr;
    uint64_t* d_p_out  = nullptr;
    uint64_t* d_q_out  = nullptr;

    CUDA_CHECK(cudaMalloc(&d_primes, MAX_PRIMES * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_found,  sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_p_out,  sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_q_out,  sizeof(uint64_t)));

    // Initialize found flag to 0
    CUDA_CHECK(cudaMemset(d_found, 0, sizeof(int)));

    int threads_per_block = 256;

    auto t_start = std::chrono::high_resolution_clock::now();

    double sieve_ms = 0.0;
    double gpu_ms   = 0.0;

    bool found = false;
    uint64_t p_out = 0, q_out = 0;

    // -------------------------------------------------------
    // Main loop: sieve primes in [2, n/2] in chunks,
    // launch GPU kernel on each chunk.
    // Stop as soon as a partition is found.
    // -------------------------------------------------------
    uint64_t low = 2;
    uint64_t half_n = n / 2;

    while (low <= half_n && !found) {
        uint64_t high = std::min(low + CHUNK_SIZE - 1, half_n);

        // --- Sieve this chunk on CPU ---
        auto ts0 = std::chrono::high_resolution_clock::now();
        auto sieve = segmented_sieve(low, high);
        auto ts1 = std::chrono::high_resolution_clock::now();
        sieve_ms += std::chrono::duration<double, std::milli>(ts1 - ts0).count();

        // Extract primes from sieve result
        std::vector<uint64_t> primes;
        primes.reserve(CHUNK_SIZE / 10);  // rough estimate
        extract_primes(sieve, low, primes);

        if (primes.empty()) { low = high + 1; continue; }

        // --- Process primes in GPU-sized batches ---
        // (in case a chunk has more primes than MAX_PRIMES)
        uint64_t offset = 0;
        while (offset < primes.size() && !found) {
            uint64_t batch_size = std::min(
                (uint64_t)primes.size() - offset, MAX_PRIMES);

            // Copy this batch to GPU
            CUDA_CHECK(cudaMemcpy(d_primes,
                                  primes.data() + offset,
                                  batch_size * sizeof(uint64_t),
                                  cudaMemcpyHostToDevice));

            // Launch kernel
            uint64_t blocks = (batch_size + threads_per_block - 1)
                              / threads_per_block;

            auto tg0 = std::chrono::high_resolution_clock::now();

            goldbach_single_kernel<<<(uint32_t)blocks, threads_per_block>>>(
                d_primes, batch_size, n, d_found, d_p_out, d_q_out);

            CUDA_CHECK(cudaDeviceSynchronize());

            auto tg1 = std::chrono::high_resolution_clock::now();
            gpu_ms += std::chrono::duration<double,
                       std::milli>(tg1 - tg0).count();

            // Check if partition was found
            int h_found = 0;
            CUDA_CHECK(cudaMemcpy(&h_found, d_found, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            if (h_found) {
                found = true;
                CUDA_CHECK(cudaMemcpy(&p_out, d_p_out,
                                      sizeof(uint64_t),
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&q_out, d_q_out,
                                      sizeof(uint64_t),
                                      cudaMemcpyDeviceToHost));
            }

            offset += batch_size;
        }

        low = high + 1;

        // Print progress every chunk so we know it's running
        std::cout << "  Checked primes up to " << high
                  << " (" << (100.0 * high / half_n) << "%)\r"
                  << std::flush;
    }

    std::cout << "\n";

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double,
                      std::milli>(t_end - t_start).count();

    // -------------------------------------------------------
    // Results
    // -------------------------------------------------------
    std::cout << "\n--- Result ---\n";
    if (found) {
        std::cout << n << " = " << p_out << " + " << q_out << "\n";
        std::cout << "Goldbach holds. ✓\n";
    } else {
        std::cout << "NO PARTITION FOUND — counterexample!\n";
    }

    std::cout << "\n--- Timing ---\n";
    std::cout << "Sieve time (CPU) : " << sieve_ms << " ms\n";
    std::cout << "Kernel time (GPU): " << gpu_ms   << " ms\n";
    std::cout << "Total            : " << total_ms  << " ms\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_primes));
    CUDA_CHECK(cudaFree(d_found));
    CUDA_CHECK(cudaFree(d_p_out));
    CUDA_CHECK(cudaFree(d_q_out));
}

int main(int argc, char* argv[]) {
    uint64_t n = 1'000'000'000'000ULL;  // 10^12 default

    if (argc > 1)
        n = std::stoull(argv[1]);

    check_single(n);
    return 0;
}

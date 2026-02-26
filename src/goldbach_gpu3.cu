// goldbach_gpu3.cu
// GPU Goldbach range verifier -- correct segmented design.
//
// Algorithm (double sieve over n):
//
//   For each segment [A, B] of even numbers:
//     Phase 1 (GPU): for each prime p in [2, P_SMALL],
//       mark all even n in [A, B] as verified if n-p is prime.
//       q = n-p checked via small bitset, segment bitset, or Miller-Rabin.
//     Phase 2 (CPU fallback): any n still unverified after Phase 1
//       is checked exhaustively: all odd p from 3 to n/2,
//       testing both p and n-p for primality directly.
//       This is slow but correct, and expected to never trigger.
//
// Correctness guarantee:
//   Every even n in [4, LIMIT] is verified by Phase 1 or Phase 2.
//   Phase 2 is truly exhaustive -- no bound on p.
//   The result is mathematically rigorous.

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "prime_bitset.hpp"

using namespace goldbach;

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " -- " << cudaGetErrorString(err) << "\n";        \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

// -------------------------------------------------------
// Miller-Rabin primality test for 64-bit integers on GPU.
// Deterministic for all n < 3.3 x 10^24 with these witnesses.
// -------------------------------------------------------
__device__ uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
    return (__uint128_t)a * b % m;
}

__device__ uint64_t powmod64(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = mulmod64(result, base, mod);
        base = mulmod64(base, base, mod);
        exp >>= 1;
    }
    return result;
}

__device__ bool miller_rabin_witness(uint64_t n, uint64_t a,
                                     uint64_t d, uint64_t r) {
    uint64_t x = powmod64(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (uint64_t i = 0; i < r - 1; i++) {
        x = mulmod64(x, x, n);
        if (x == n - 1) return true;
    }
    return false;
}

__device__ bool gpu_is_prime_miller_rabin(uint64_t n) {
    if (n < 2)  return false;
    if (n == 2 || n == 3) return true;
    if ((n & 1) == 0) return false;

    uint64_t d = n - 1, r = 0;
    while ((d & 1) == 0) { d >>= 1; r++; }

    const uint64_t witnesses[] =
        {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (int i = 0; i < 12; i++) {
        if (witnesses[i] >= n) continue;
        if (!miller_rabin_witness(n, witnesses[i], d, r)) return false;
    }
    return true;
}

// -------------------------------------------------------
// Device function: is q prime?
// Three cases depending on where q falls:
//   1. q <= small_high: small primes bitset
//   2. q in [seg_low, seg_high]: segment bitset
//   3. otherwise: Miller-Rabin
// -------------------------------------------------------
__device__ bool is_prime_q(
    uint64_t q,
    const uint64_t* __restrict__ d_small,
    uint64_t small_high,
    const uint64_t* __restrict__ d_seg,
    uint64_t seg_low,
    uint64_t seg_high)
{
    if (q < 2) return false;
    if (q == 2) return true;
    if ((q & 1) == 0) return false;

    if (q <= small_high) {
        uint64_t bit_pos  = (q - 3) / 2;
        uint64_t word_idx = bit_pos / 64;
        uint64_t bit_idx  = bit_pos % 64;
        return (d_small[word_idx] >> bit_idx) & 1ULL;
    }

    if (q >= seg_low && q <= seg_high) {
        uint64_t bit_pos  = (q - seg_low) / 2;
        uint64_t word_idx = bit_pos / 64;
        uint64_t bit_idx  = bit_pos % 64;
        return (d_seg[word_idx] >> bit_idx) & 1ULL;
    }

    return gpu_is_prime_miller_rabin(q);
}

// -------------------------------------------------------
// Phase 1 kernel: GPU verification with bounded p.
// One thread per even n in segment.
// For each prime in current batch, check if n-p is prime.
// -------------------------------------------------------
__global__ void goldbach_phase1_kernel(
    const uint64_t* __restrict__ d_small,
    uint64_t        small_high,
    const uint64_t* __restrict__ d_seg,
    uint64_t        seg_low,
    uint64_t        seg_high,
    uint64_t        seg_even_start,
    uint64_t        seg_even_count,
    const uint64_t* __restrict__ p_batch,
    uint64_t        p_batch_size,
    uint8_t*        __restrict__ d_verified)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= seg_even_count) return;

    // Early exit -- already verified by a previous batch
    if (d_verified[tid]) return;

    uint64_t n = seg_even_start + tid * 2;

    for (uint64_t i = 0; i < p_batch_size; i++) {
        uint64_t p = p_batch[i];

        // Batch is sorted -- once p > n/2 no point continuing
        if (p > n / 2) break;

        uint64_t q = n - p;

        if (is_prime_q(q, d_small, small_high,
                          d_seg, seg_low, seg_high)) {
            d_verified[tid] = 1;
            return;
        }
    }
}

// -------------------------------------------------------
// CPU trial division primality test.
// Used in Phase 2 only -- expected to be called rarely.
// -------------------------------------------------------
static bool cpu_is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if ((n & 1) == 0) return false;
    uint64_t sq = (uint64_t)std::sqrt((double)n);
    for (uint64_t d = 3; d <= sq; d += 2)
        if (n % d == 0) return false;
    return true;
}

// -------------------------------------------------------
// Phase 2: truly exhaustive CPU fallback.
// Tests ALL odd p from 3 to n/2, plus p=2.
// No bound on p -- mathematically rigorous.
// Expected to never be called at our target scales.
// -------------------------------------------------------
static bool cpu_exhaustive_check(uint64_t n) {
    // Try p=2 first
    if (cpu_is_prime(n - 2)) return true;

    // Try all odd p from 3 to n/2
    for (uint64_t p = 3; p <= n / 2; p += 2) {
        if (cpu_is_prime(p) && cpu_is_prime(n - p))
            return true;
    }
    return false;
}

// -------------------------------------------------------
// Build segment prime bitset for [seg_low, seg_high].
// -------------------------------------------------------
static std::vector<uint64_t> build_segment_bitset(
    uint64_t seg_low,
    uint64_t seg_high,
    const std::vector<uint64_t>& small_primes)
{
    if (seg_low % 2 == 0) seg_low++;

    uint64_t num_odds  = (seg_high - seg_low) / 2 + 1;
    uint64_t num_words = (num_odds + 63) / 64;
    std::vector<uint64_t> words(num_words, ~0ULL);

    for (uint64_t p : small_primes) {
        if (p * p > seg_high) break;

        uint64_t first = ((seg_low + p - 1) / p) * p;
        if (first % 2 == 0) first += p;
        if (first < p * p) {
            first = p * p;
            if (first % 2 == 0) first += p;
        }
        if (first > seg_high) continue;

        for (uint64_t j = first; j <= seg_high; j += 2 * p) {
            uint64_t bit_pos  = (j - seg_low) / 2;
            uint64_t word_idx = bit_pos / 64;
            uint64_t bit_idx  = bit_pos % 64;
            words[word_idx]  &= ~(1ULL << bit_idx);
        }
    }

    return words;
}

void print_usage(const char* prog) {
    std::cout << "Goldbach Conjecture Segmented Verifier (GPU)\n";
    std::cout << "Usage: " << prog << " <LIMIT> [SEG_SIZE] [P_SMALL]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  LIMIT     Max even integer to check (e.g., 1000000000)\n";
    std::cout << "  SEG_SIZE  (Optional) Even integers per segment. Default: 500,000,000\n";
    std::cout << "  P_SMALL   (Optional) GPU prime search bound. Default: 2,000,000\n\n";
    std::cout << "Flags:\n";
    std::cout << "  -h, --help  Show this help message\n";
}

void check_vram_limit(uint64_t seg_size, uint64_t small_bytes) {
    // Hardware VRAM is automatically checked against SEG_SIZE to prevent OOM.
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    uint64_t verified_bytes = seg_size;
    uint64_t segment_bitset_bytes = (seg_size + 1) / 8;
    uint64_t p_batch_bytes = 100000ULL * sizeof(uint64_t);

    uint64_t total_required =
        verified_bytes +
        segment_bitset_bytes +
        p_batch_bytes +
        small_bytes +
        (50ULL * 1024 * 1024); // 50 MB safety margin

    if (total_required > prop.totalGlobalMem) {
        std::cerr << "\n[!] ERROR: SEG_SIZE (" << seg_size << ") requires approx "
                  << total_required / (1024*1024) << " MB VRAM.\n";
        std::cerr << "[!] This GPU (" << prop.name << ") only has "
                  << prop.totalGlobalMem / (1024*1024) << " MB available.\n";
        std::cerr << "[!] Reduce SEG_SIZE or use a smaller LIMIT.\n";
        std::exit(1);
    }

    std::cout << "[Hardware] GPU: " << prop.name
              << " (" << prop.totalGlobalMem / (1024*1024) << " MB VRAM)\n";
}

int main(int argc, char** argv) {
    if (argc < 2 ||
        std::string(argv[1]) == "-h" ||
        std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        return 0;
    }

    uint64_t LIMIT, SEG_SIZE, P_SMALL;

    try {
        LIMIT    = std::stoull(argv[1]);
        SEG_SIZE = (argc > 2) ? std::stoull(argv[2]) : 500'000'000ULL;
        P_SMALL  = (argc > 3) ? std::stoull(argv[3]) : 2'000'000ULL;
    } catch (...) {
        std::cerr << "Error: Invalid numeric argument. Use -h for help.\n";
        return 1;
    }

    if (LIMIT < 4) {
        std::cerr << "Error: LIMIT must be >= 4.\n";
        return 1;
    }

    if (LIMIT % 2 != 0) LIMIT--;

    // Calculate small_high
    uint64_t small_high = std::max((uint64_t)std::sqrt((double)LIMIT) + 1, P_SMALL);
    if (small_high % 2 == 0) small_high++;

    // Calculate exact bytes needed for the small primes bitset
    uint64_t num_small_odds = (small_high - 3) / 2 + 1;
    uint64_t small_bytes = ((num_small_odds + 63) / 64) * sizeof(uint64_t);

    // Pass small_bytes into check function
    check_vram_limit(SEG_SIZE, small_bytes);

    const uint64_t P_BATCH  = 100'000ULL;       // primes per kernel launch

    std::cout << "Goldbach segmented verifier (Phase 1: GPU, Phase 2: CPU)\n";
    std::cout << "Checking all even n in [4, " << LIMIT << "]\n";
    std::cout << "P_SMALL = " << P_SMALL
              << ", exhaustive CPU fallback for any remainder\n\n";

    // -------------------------------------------------------
    // Step 1: Build small primes bitset
    // Cover both sqrt(LIMIT) for segment sieve
    // and P_SMALL for GPU primality lookups
    // -------------------------------------------------------

    std::cout << "Building small primes bitset up to "
              << small_high << "...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    PrimeBitset small_bitset = build_prime_bitset(small_high);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Built in "
              << std::chrono::duration<double, std::milli>(t1-t0).count()
              << " ms (" << small_bitset.memory_bytes() / 1024 << " KB)\n\n";

    // Collect small primes as sorted list
    std::vector<uint64_t> small_primes;
    small_primes.reserve(small_high / 10);
    if (small_bitset.is_prime(2)) small_primes.push_back(2);
    for (uint64_t i = 3; i <= small_high; i += 2)
        if (small_bitset.is_prime(i))
            small_primes.push_back(i);

    // GPU primes: only those <= P_SMALL
    std::vector<uint64_t> gpu_primes;
    for (uint64_t p : small_primes)
        if (p <= P_SMALL) gpu_primes.push_back(p);

    std::cout << "GPU primes (p <= " << P_SMALL << "): "
              << gpu_primes.size() << "\n\n";

    // -------------------------------------------------------
    // Step 2: Copy small primes bitset to GPU -- permanent
    // -------------------------------------------------------
    small_bytes = small_bitset.word_count() * sizeof(uint64_t);
    uint64_t* d_small = nullptr;
    CUDA_CHECK(cudaMalloc(&d_small, small_bytes));
    CUDA_CHECK(cudaMemcpy(d_small, small_bitset.data(), small_bytes,
                          cudaMemcpyHostToDevice));
    std::cout << "Small primes in GPU ("
              << small_bytes / 1024 << " KB)\n";

    // -------------------------------------------------------
    // Step 3: Allocate GPU buffers
    //
    // d_seg: segment prime bitset
    //   odd range is [seg_start-1, seg_end+1]
    //   width ~ 2*SEG_SIZE, so SEG_SIZE+1 odd numbers
    //
    // d_verified: one byte per even n in segment
    //   max size = SEG_SIZE
    //
    // d_p_batch: current prime batch
    // -------------------------------------------------------
    uint64_t seg_odd_count = SEG_SIZE + 1;
    uint64_t seg_words     = (seg_odd_count + 63) / 64;
    uint64_t seg_bytes     = seg_words * sizeof(uint64_t);

    uint64_t* d_seg      = nullptr;
    uint8_t*  d_verified = nullptr;
    uint64_t* d_p_batch  = nullptr;

    CUDA_CHECK(cudaMalloc(&d_seg,      seg_bytes));
    CUDA_CHECK(cudaMalloc(&d_verified, SEG_SIZE));
    CUDA_CHECK(cudaMalloc(&d_p_batch,  P_BATCH * sizeof(uint64_t)));

    std::cout << "Segment buffer: " << seg_bytes / 1024 / 1024 << " MB\n";
    std::cout << "Verified buffer: " << SEG_SIZE / 1024 / 1024 << " MB\n\n";

    // -------------------------------------------------------
    // Step 4: Main loop -- process segments
    // -------------------------------------------------------
    int      threads_per_block  = 256;
    uint64_t total_even         = (LIMIT - 4) / 2 + 1;
    uint64_t total_processed    = 0;
    uint64_t total_failures     = 0;
    uint64_t total_phase2_count = 0;

    auto t_main = std::chrono::high_resolution_clock::now();

    uint64_t seg_start = 4;

    while (seg_start <= LIMIT) {
        uint64_t seg_end = std::min(seg_start + SEG_SIZE * 2 - 2, LIMIT);

        // Odd range for segment bitset
        // Must cover all q = n-p that fall in this range
        uint64_t seg_low  = (seg_start >= 3) ? seg_start - 1 : 3;
        if (seg_low % 2 == 0) seg_low++;
        uint64_t seg_high = seg_end + 1;
        if (seg_high % 2 == 0) seg_high++;

        uint64_t seg_even_count = (seg_end - seg_start) / 2 + 1;

        // Build segment bitset on CPU and copy to GPU
        auto seg_vec = build_segment_bitset(seg_low, seg_high,
                                             small_primes);

        // Safety check -- never copy more than allocated
        uint64_t copy_words = std::min(seg_vec.size(), (size_t)seg_words);
        CUDA_CHECK(cudaMemcpy(d_seg, seg_vec.data(),
                              copy_words * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));

        // Initialize verified flags to zero
        CUDA_CHECK(cudaMemset(d_verified, 0, seg_even_count));

        // Phase 1: GPU -- process gpu_primes in batches
        uint64_t blocks = (seg_even_count + threads_per_block - 1)
                          / threads_per_block;

        for (uint64_t bi = 0; bi < gpu_primes.size(); bi += P_BATCH) {
            uint64_t bend  = std::min(bi + P_BATCH,
                                      (uint64_t)gpu_primes.size());
            uint64_t bsize = bend - bi;

            CUDA_CHECK(cudaMemcpy(d_p_batch,
                                  gpu_primes.data() + bi,
                                  bsize * sizeof(uint64_t),
                                  cudaMemcpyHostToDevice));

            goldbach_phase1_kernel<<<(uint32_t)blocks,
                                     threads_per_block>>>(
                d_small,   small_high,
                d_seg,     seg_low, seg_high,
                seg_start, seg_even_count,
                d_p_batch, bsize,
                d_verified);
            
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Copy verified flags back
        std::vector<uint8_t> verified(seg_even_count);
        CUDA_CHECK(cudaMemcpy(verified.data(), d_verified,
                              seg_even_count,
                              cudaMemcpyDeviceToHost));

        // Phase 2: CPU exhaustive fallback for any unverified n
        for (uint64_t i = 0; i < seg_even_count; i++) {
            if (verified[i]) continue;

            uint64_t n = seg_start + i * 2;
            total_phase2_count++;

            std::cout << "\n  Phase 2 fallback for n = " << n << "...\n";
            bool found = cpu_exhaustive_check(n);

            if (!found) {
                std::cout << "FAILURE: no Goldbach partition for n = "
                          << n << "\n";
                total_failures++;
            } else {
                std::cout << "  Phase 2 verified n = " << n << "\n";
            }
        }

        total_processed += seg_even_count;
        seg_start        = seg_end + 2;

        double pct = 100.0 * total_processed / total_even;
        std::cout << "  Progress: " << total_processed
                  << " / " << total_even
                  << " (" << pct << "%)\r" << std::flush;
    }

    double total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_main).count();

    std::cout << "\n";

    // -------------------------------------------------------
    // Step 5: Summary
    // -------------------------------------------------------
    std::cout << "\n--- Summary ---\n";
    std::cout << "Even numbers checked  : " << total_even         << "\n";
    std::cout << "Failures              : " << total_failures     << "\n";
    std::cout << "Phase 2 fallbacks     : " << total_phase2_count << "\n";
    std::cout << "P_SMALL               : " << P_SMALL            << "\n";
    std::cout << "Total time            : " << total_ms           << " ms\n";

    if (total_failures == 0) {
        std::cout << "\nAll even numbers up to " << LIMIT
                  << " satisfy Goldbach. ✓\n";
        if (total_phase2_count == 0)
            std::cout << "(All verified by GPU with p <= "
                      << P_SMALL << ")\n";
        else
            std::cout << "(" << total_phase2_count
                      << " required CPU exhaustive fallback)\n";
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_small));
    CUDA_CHECK(cudaFree(d_seg));
    CUDA_CHECK(cudaFree(d_verified));
    CUDA_CHECK(cudaFree(d_p_batch));

    return (total_failures == 0) ? 0 : 1;
}
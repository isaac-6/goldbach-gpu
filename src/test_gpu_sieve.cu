// Differential test: GPU segment sieve vs. the CPU segmented sieve.
// The CPU sieve in segmented_sieve.cpp is independently validated
// against known pi(n) values by test_sieve.cpp.
//
// Exit 0 = agreement, 1 = mismatch.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "sieve_kernel.cuh"
#include "prime_bitset.hpp"

std::vector<uint64_t> simple_sieve(uint64_t limit);
std::vector<char> segmented_sieve(uint64_t low, uint64_t high);

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); exit(2); } } while (0)

// Returns number of disagreements; prints the first few.
static uint64_t check_range(uint64_t q_low, uint64_t q_high,
                            const std::vector<uint64_t>& small_primes, bool verbose)
{
    if (q_low % 2 == 0) q_low++;
    if (q_high % 2 == 0) q_high++;

    uint64_t num_odds = (q_high - q_low) / 2 + 1;
    uint64_t words    = (num_odds + 63) / 64;

    uint64_t *d_bits = nullptr, *d_primes = nullptr;
    CK(cudaMalloc(&d_bits, words * sizeof(uint64_t)));
    CK(cudaMalloc(&d_primes, small_primes.size() * sizeof(uint64_t)));
    CK(cudaMemcpy(d_primes, small_primes.data(),
                  small_primes.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    uint64_t sc = sieve_split_prime_count(small_primes.data(), small_primes.size());
    launch_segment_sieve(q_low, q_high, d_primes,
                         sc, small_primes.size() - sc, d_bits, 256, 0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<uint64_t> host_bits(words);
    CK(cudaMemcpy(host_bits.data(), d_bits, words * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    auto cpu = segmented_sieve(q_low, q_high);

    uint64_t mismatches = 0, shown = 0;
    for (uint64_t i = 0; i < num_odds; i++) {
        uint64_t q = q_low + 2 * i;
        bool gpu_says = (host_bits[i / 64] >> (i % 64)) & 1ULL;
        bool cpu_says = cpu[q - q_low] != 0;
        if (gpu_says != cpu_says) {
            mismatches++;
            if (verbose && shown < 5) {
                printf("    q=%llu  gpu=%s cpu=%s\n",
                       (unsigned long long)q,
                       gpu_says ? "prime" : "composite",
                       cpu_says ? "prime" : "composite");
                shown++;
            }
        }
    }

    CK(cudaFree(d_bits));
    CK(cudaFree(d_primes));
    return mismatches;
}

int main() {
    uint64_t total = 0;

    // A prime only reaches large_prime_sieve_kernel when it is >= TILE_ODDS,
    // and only marks anything when p*p <= q_high. Since check_range is handed
    // primes up to isqrt(hi), that path is exercised only for hi >= TILE_ODDS^2
    // (~1.07e9 at the default). The ranges below 2^32 therefore cover the tiled
    // kernel alone; the high ranges are what gate the large-prime split.
    struct { uint64_t lo, hi; const char* name; } fixed[] = {
        {3,          1000000,     "from 3 (q_low edge)"},
        {999983,     2000000,     "small offset"},
        {4294967291ULL, 4295000000ULL, "straddling 2^32"},
        {1000000000ULL, 1000200000ULL, "1e9"},
        {1073741824ULL, 1074141824ULL, "TILE_ODDS^2 boundary"},
        {100000000000ULL, 100000400000ULL, "1e11"},
        {1000000000000ULL, 1000000400000ULL, "1e12"},
    };

    for (auto& t : fixed) {
        uint64_t root = 1; while ((root + 1) * (root + 1) <= t.hi) root++;
        auto primes = simple_sieve(root + 1);
        printf("  [%s] [%llu, %llu]\n", t.name,
               (unsigned long long)t.lo, (unsigned long long)t.hi);
        uint64_t m = check_range(t.lo, t.hi, primes, true);
        printf("    -> %llu mismatches\n", (unsigned long long)m);
        total += m;
    }

    std::mt19937_64 rng(20260907);
    printf("  [randomized] 50 ranges below 1e9, 25 up to 1e12\n");
    for (int i = 0; i < 75; i++) {
        // The first 50 stay low (tiled kernel only); the rest are drawn high
        // enough that the large-prime kernel has work to do.
        uint64_t lo = (i < 50) ? 3 + (rng() % 1000000000ULL)
                               : 1073741824ULL + (rng() % 1000000000000ULL);
        uint64_t hi = lo + 100000 + (rng() % 400000);
        uint64_t root = 1; while ((root + 1) * (root + 1) <= hi) root++;
        auto primes = simple_sieve(root + 1);
        total += check_range(lo, hi, primes, false);
    }

    printf("\nTOTAL MISMATCHES: %llu\n", (unsigned long long)total);
    if (total) { printf("FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}
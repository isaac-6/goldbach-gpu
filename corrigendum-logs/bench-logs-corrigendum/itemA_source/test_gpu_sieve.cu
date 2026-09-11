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

// --- D1 quantification counters ---
static unsigned long long g_total_odds      = 0;  // odd numbers compared
static unsigned long long g_total_composite = 0;  // of those, composite per CPU
static unsigned long long g_total_prime     = 0;  // of those, prime per CPU
static unsigned long long g_false_prime     = 0;  // GPU says prime, CPU says composite
static unsigned long long g_false_composite = 0;  // GPU says composite, CPU says prime
static unsigned int g_grid = 0, g_block = 0; static size_t g_shmem = 0;
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

    // v2.0.0 has no large-prime split: launch the tiled kernel directly with
    // v2.0.0's shared-memory sizing (a uint64 bitmap of TILE_ODDS bits = 4096 B).
    uint32_t num_tiles = (uint32_t)((num_odds + TILE_ODDS - 1) / TILE_ODDS);
    size_t shmem = (TILE_ODDS / 64) * sizeof(uint64_t);
    g_grid = num_tiles; g_block = 256; g_shmem = shmem;
    tiled_sieve_segment_kernel<<<num_tiles, 256, shmem>>>(
        q_low, q_high, d_primes, small_primes.size(), d_bits);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<uint64_t> host_bits(words);
    CK(cudaMemcpy(host_bits.data(), d_bits, words * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    auto cpu = segmented_sieve(q_low, q_high);

    uint64_t mismatches = 0, shown = 0;
    uint64_t r_comp = 0, r_false_prime = 0, r_false_comp = 0;
    for (uint64_t i = 0; i < num_odds; i++) {
        uint64_t q = q_low + 2 * i;
        bool gpu_says = (host_bits[i / 64] >> (i % 64)) & 1ULL;
        bool cpu_says = cpu[q - q_low] != 0;
        g_total_odds++;
        if (cpu_says) g_total_prime++; else { g_total_composite++; r_comp++; }
        if (gpu_says != cpu_says) {
            if (gpu_says && !cpu_says) { g_false_prime++; r_false_prime++; }
            else { g_false_composite++; r_false_comp++; }
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

    if (verbose)
        printf("    [range] odds=%llu composites=%llu false_prime=%llu false_composite=%llu"
               " rate_vs_composites=%.6f%%\n",
               (unsigned long long)num_odds, (unsigned long long)r_comp,
               (unsigned long long)r_false_prime, (unsigned long long)r_false_comp,
               r_comp ? 100.0*(double)r_false_prime/(double)r_comp : 0.0);
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

    printf("\n=== D1 QUANTIFICATION (v2.0.0 kernel, defect intact) ===\n");
    printf("  launch config      : grid=<per range> block=%u shared=%zu bytes\n", g_block, g_shmem);
    printf("  odd numbers compared     : %llu\n", g_total_odds);
    printf("  of which prime (CPU)     : %llu\n", g_total_prime);
    printf("  of which composite (CPU) : %llu\n", g_total_composite);
    printf("  disagreements total      : %llu\n", g_false_prime + g_false_composite);
    printf("    composite reported PRIME: %llu\n", g_false_prime);
    printf("    prime reported COMPOSITE: %llu\n", g_false_composite);
    printf("  rate vs all odds compared: %.8f%%\n", 100.0*(double)(g_false_prime+g_false_composite)/(double)g_total_odds);
    printf("  rate vs composites       : %.8f%%\n", 100.0*(double)g_false_prime/(double)g_total_composite);
    printf("\nTOTAL MISMATCHES: %llu\n", (unsigned long long)total);
    if (total) { printf("FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}
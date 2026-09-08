// Differential test: GPU Phase 1 Goldbach verification vs. a CPU reference.
//
// The GPU side runs the production kernels from sieve_kernel.cuh and
// phase1_kernel.cuh, wired up exactly as goldbach.cu wires them: sieve the
// segment, then sweep the prime batches over the even numbers.
//
// The CPU side redoes the search independently -- segmented_sieve over
// [n_low - p_small, n_high], then for each even n a straight ascending scan
// for the first p with n - p prime.
//
// Exit 0 = agreement, 1 = mismatch.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include "sieve_kernel.cuh"
#include "phase1_kernel.cuh"
#include "prime_bitset.hpp"

using namespace goldbach;

std::vector<uint64_t> simple_sieve(uint64_t limit);
std::vector<char> segmented_sieve(uint64_t low, uint64_t high);

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); exit(2); } } while (0)

static const int THREADS_PER_BLOCK = 256;
static const uint64_t P_BATCH = 2000000;

static uint64_t isqrt64(uint64_t n) {
    if (n < 2) return n;
    uint64_t r = 1;
    while (r <= n / r) r++;
    return r - 1;
}

// -------------------------------------------------------
// CPU reference: index of the first p (ascending) with n - p prime,
// or NO_P if the whole list is exhausted. Mirrors what the kernel searches
// for, by a wholly separate route.
//
// Returning the index rather than a bare verdict is what gives this test its
// resolution. Every even n in these ranges has many Goldbach partitions, so
// "verified" alone is saturated -- it stays true even if the kernel's
// primality lookups are wrong, because some later p still succeeds. The index
// pins down *which* p resolved n, so a wrong lookup has nowhere to hide.
// -------------------------------------------------------
static const uint32_t NO_P = UINT32_MAX;

static std::vector<uint32_t> cpu_phase1(uint64_t n_low, uint64_t n_high,
                                        uint64_t even_count,
                                        const std::vector<uint64_t>& gpu_primes,
                                        uint64_t p_small)
{
    // Own lower bound, chosen to cover every reachable q = n - p rather than
    // copied from the kernel's q_low -- the smallest q arises at n = n_low with
    // the largest usable p. Note this reaches below the kernel's q_low, which is
    // odd-adjusted and clamped to 3: at n = 4 the only partition is 2 + 2.
    uint64_t ref_low = (n_low > p_small + 2) ? n_low - p_small : 2;

    std::vector<char> is_prime = segmented_sieve(ref_low, n_high);
    std::vector<uint32_t> pmin(even_count, NO_P);

    for (uint64_t i = 0; i < even_count; i++) {
        uint64_t n = n_low + 2 * i;
        for (uint32_t j = 0; j < gpu_primes.size(); j++) {
            uint64_t p = gpu_primes[j];
            if (p > n / 2) break;
            uint64_t q = n - p;
            if (is_prime[q - ref_low]) { pmin[i] = j; break; }
        }
    }
    return pmin;
}

// Returns number of disagreements; prints the first few.
static uint64_t check_range(uint64_t n_low, uint64_t n_high, uint64_t p_small,
                            bool verbose)
{
    if (n_low % 2) n_low++;
    if (n_high % 2) n_high--;
    if (n_low < 4) n_low = 4;
    if (n_high < n_low) return 0;

    uint64_t even_count = (n_high - n_low) / 2 + 1;

    // Segment geometry, as goldbach.cu computes it.
    uint64_t q_low = (n_low > p_small ? n_low - p_small : 3);
    if ((q_low & 1) == 0) q_low++;
    uint64_t q_high = n_high + 1;
    if ((q_high & 1) == 0) q_high++;

    uint64_t small_high = std::max(isqrt64(n_high) + 1, p_small);
    if (small_high % 2 == 0) small_high++;

    // Host prime tables.
    PrimeBitset small_bitset = build_prime_bitset(small_high);
    std::vector<uint64_t> small_primes;
    if (small_bitset.is_prime(2)) small_primes.push_back(2);
    for (uint64_t i = 3; i <= small_high; i += 2)
        if (small_bitset.is_prime(i)) small_primes.push_back(i);

    std::vector<uint64_t> gpu_primes;
    for (uint64_t p : small_primes)
        if (p <= p_small) gpu_primes.push_back(p);

    // -------- GPU side --------
    uint64_t num_odds  = (q_high - q_low) / 2 + 1;
    uint64_t seg_words = (num_odds + 63) / 64;
    size_t small_bytes = small_bitset.word_count() * sizeof(uint64_t);

    uint64_t *d_small = nullptr, *d_small_primes = nullptr;
    uint64_t *d_seg_bits = nullptr, *d_p_batch = nullptr;
    uint8_t  *d_verified = nullptr;

    CK(cudaMalloc(&d_small, small_bytes));
    CK(cudaMalloc(&d_small_primes, small_primes.size() * sizeof(uint64_t)));
    CK(cudaMalloc(&d_seg_bits, seg_words * sizeof(uint64_t)));
    CK(cudaMalloc(&d_p_batch, std::min(P_BATCH, (uint64_t)gpu_primes.size()) * sizeof(uint64_t)));
    CK(cudaMalloc(&d_verified, even_count));

    CK(cudaMemcpy(d_small, small_bitset.data(), small_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_small_primes, small_primes.data(),
                  small_primes.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_verified, 0, even_count));

    PrimeTest primeTest = PrimeTest::BPSW;
    CK(cudaMemcpyToSymbol(g_device_prime_test, &primeTest, sizeof(PrimeTest)));

    uint32_t num_tiles = (uint32_t)((num_odds + TILE_ODDS - 1) / TILE_ODDS);
    size_t shmem = TILE_ODDS * sizeof(unsigned char);

    tiled_sieve_segment_kernel<<<num_tiles, THREADS_PER_BLOCK, shmem>>>(
        q_low, q_high, d_small_primes, small_primes.size(), d_seg_bits);
    CK(cudaGetLastError());

    // -------- CPU side --------
    auto cpu_pmin = cpu_phase1(n_low, n_high, even_count, gpu_primes, p_small);

    // Sweep prime-list prefixes. Handing the production kernel only the first
    // K primes makes its verdict mean "p_min_idx < K", so comparing verdicts
    // across several K probes the index without modifying the kernel. The full
    // list is included last, which is the plain end-to-end case.
    uint64_t prefixes[] = {1, 2, 3, 5, 10, 25, 100, 1000, (uint64_t)gpu_primes.size()};

    uint32_t blocks = (uint32_t)((even_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    uint64_t mismatches = 0, shown = 0;
    std::vector<uint8_t> gpu_verified(even_count);

    for (uint64_t K : prefixes) {
        if (K > gpu_primes.size()) continue;

        CK(cudaMemset(d_verified, 0, even_count));
        for (uint64_t bi = 0; bi < K; bi += P_BATCH) {
            uint64_t bsize = std::min(P_BATCH, K - bi);
            CK(cudaMemcpy(d_p_batch, gpu_primes.data() + bi,
                          bsize * sizeof(uint64_t), cudaMemcpyHostToDevice));

            goldbach_phase1_kernel<<<blocks, THREADS_PER_BLOCK>>>(
                d_small, small_high, d_seg_bits, q_low, q_high,
                n_low, even_count, d_p_batch, bsize, d_verified);
            CK(cudaGetLastError());
        }
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(gpu_verified.data(), d_verified, even_count, cudaMemcpyDeviceToHost));

        for (uint64_t i = 0; i < even_count; i++) {
            bool g = gpu_verified[i] != 0;
            bool c = cpu_pmin[i] != NO_P && cpu_pmin[i] < K;
            if (g != c) {
                mismatches++;
                if (verbose && shown < 5) {
                    printf("    n=%llu  gpu=%s cpu=%s  (first %llu primes",
                           (unsigned long long)(n_low + 2 * i),
                           g ? "verified" : "unverified",
                           c ? "verified" : "unverified",
                           (unsigned long long)K);
                    if (cpu_pmin[i] == NO_P) printf(", cpu found no p)\n");
                    else printf(", cpu p_min=%llu at index %u)\n",
                                (unsigned long long)gpu_primes[cpu_pmin[i]], cpu_pmin[i]);
                    shown++;
                }
            }
        }
    }

    CK(cudaFree(d_small));
    CK(cudaFree(d_small_primes));
    CK(cudaFree(d_seg_bits));
    CK(cudaFree(d_p_batch));
    CK(cudaFree(d_verified));
    return mismatches;
}

int main() {
    uint64_t total = 0;
    const uint64_t SPAN = 2 * 200000;   // ~200k even numbers
    const uint64_t P_SMALL = 1000000;

    struct { uint64_t lo, hi; const char* name; } fixed[] = {
        {4,                4 + SPAN,                "from 4 (small-n edge)"},
        {1000000000ULL,    1000000000ULL + SPAN,    "1e9"},
        {100000000000ULL,  100000000000ULL + SPAN,  "1e11"},
        // goldbach.cu walks segments of SEG_SIZE*2; this range straddles one
        // such boundary at 2e8 with the default --seg-size=200000000.
        {400000000ULL - SPAN / 2, 400000000ULL + SPAN / 2, "straddling segment boundary"},
    };

    for (auto& t : fixed) {
        printf("  [%s] [%llu, %llu]\n", t.name,
               (unsigned long long)t.lo, (unsigned long long)t.hi);
        fflush(stdout);
        uint64_t m = check_range(t.lo, t.hi, P_SMALL, true);
        printf("    -> %llu mismatches\n", (unsigned long long)m);
        total += m;
    }

    std::mt19937_64 rng(20260908);
    printf("  [randomized] 20 ranges\n");
    fflush(stdout);
    for (int i = 0; i < 20; i++) {
        uint64_t lo = 4 + (rng() % 100000000000ULL);
        uint64_t hi = lo + SPAN;
        total += check_range(lo, hi, P_SMALL, false);
    }

    printf("\nTOTAL MISMATCHES: %llu\n", (unsigned long long)total);
    if (total) { printf("FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}

// analyze_pmin.cpp
//
// Measures how much work goldbach_phase1_kernel actually does per even number,
// and how much a warp pays for it.
//
// Each GPU thread walks the prime list ascending and exits at the first p with
// n - p prime. That prime is p_min(n); its POSITION in the list is p_min_idx(n).
// A warp of 32 lanes runs until its slowest lane finishes, so the cost of 32
// consecutive even numbers is set by the maximum across the group, not the
// average.
//
// TWO DIFFERENT NUMBERS, BOTH REPORTED:
//   idx  the position in the ascending prime list (0 = the prime 2). This is
//        what determines kernel cost -- it is the loop trip count.
//   p    the prime at that position, i.e. p_min itself. This is what must be
//        compared against P_SMALL when judging headroom.
// They differ by a large and growing factor: at N = 1e14 the maximum idx is
// 286 while the corresponding prime is 1873, 6.5x larger. Reading an index as
// if it were a prime badly understates how close p_min is running to P_SMALL,
// so every statistic below prints both.
//
// Reports both distributions plus the resulting efficiency, and the resolution
// rate for candidate cut-offs K (for a bulk-pass / straggler-pass split).
//
// Usage: analyze_pmin [start] [count] [p_small]
//   start   first even number to sample      (default 1e11)
//   count   how many consecutive evens       (default 1000000)
//   p_small prime bound, as in --p-small     (default 1e6)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>

std::vector<uint64_t> simple_sieve(uint64_t limit);
std::vector<char> segmented_sieve(uint64_t low, uint64_t high);

// The prime at a given list position, or 0 if out of range.
static uint64_t prime_at(const std::vector<uint64_t>& primes, uint64_t i) {
    return (i < primes.size()) ? primes[i] : 0;
}

static uint64_t percentile(std::vector<uint32_t>& v, double p) {
    if (v.empty()) return 0;
    size_t i = (size_t)(p * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + i, v.end());
    return v[i];
}

int main(int argc, char** argv) {
    uint64_t start   = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 100000000000ULL;
    uint64_t count   = (argc > 2) ? strtoull(argv[2], nullptr, 10) : 1000000ULL;
    uint64_t p_small = (argc > 3) ? strtoull(argv[3], nullptr, 10) : 1000000ULL;

    if (start % 2) start++;

    uint64_t n_high = start + 2 * (count - 1);
    uint64_t q_low  = (start > p_small + 4) ? start - p_small : 3;
    uint64_t q_high = n_high;

    printf("range   : [%llu, %llu]  (%llu even numbers)\n",
           (unsigned long long)start, (unsigned long long)n_high,
           (unsigned long long)count);
    printf("p_small : %llu\n", (unsigned long long)p_small);
    printf("sieving q in [%llu, %llu] ...\n",
           (unsigned long long)q_low, (unsigned long long)q_high);
    fflush(stdout);

    std::vector<uint64_t> primes = simple_sieve(p_small);
    std::vector<char> is_prime   = segmented_sieve(q_low, q_high);

    printf("primes  : %zu up to p_small\n\n", primes.size());
    fflush(stdout);

    // p_min_idx per even number; UINT32_MAX marks "not resolved within p_small"
    std::vector<uint32_t> idx(count, UINT32_MAX);
    uint64_t unresolved = 0;

    for (uint64_t k = 0; k < count; k++) {
        uint64_t n = start + 2 * k;
        for (size_t i = 0; i < primes.size(); i++) {
            uint64_t p = primes[i];
            if (p > n / 2) break;
            uint64_t q = n - p;
            if (q < q_low) break;
            if (is_prime[q - q_low]) { idx[k] = (uint32_t)i; break; }
        }
        if (idx[k] == UINT32_MAX) unresolved++;
    }

    if (unresolved) {
        printf("WARNING: %llu numbers unresolved within p_small\n\n",
               (unsigned long long)unresolved);
    }

    // Per-warp maximum over 32 consecutive even numbers.
    const uint64_t W = 32;
    uint64_t warps = count / W;
    std::vector<uint32_t> warp_max(warps, 0);
    for (uint64_t w = 0; w < warps; w++) {
        uint32_t m = 0;
        for (uint64_t j = 0; j < W; j++) {
            uint32_t v = idx[w * W + j];
            if (v != UINT32_MAX && v > m) m = v;
        }
        warp_max[w] = m;
    }

    // Work accounting: iterations actually needed vs. iterations a warp pays for.
    long double needed = 0, paid = 0;
    for (uint64_t k = 0; k < count; k++)
        if (idx[k] != UINT32_MAX) needed += idx[k] + 1;
    for (uint64_t w = 0; w < warps; w++)
        paid += (long double)(warp_max[w] + 1) * W;

    std::vector<uint32_t> per_n;
    per_n.reserve(count);
    for (uint64_t k = 0; k < count; k++)
        if (idx[k] != UINT32_MAX) per_n.push_back(idx[k]);

    // idx is the loop trip count; p is p_min itself. Never conflate them.
    long double mean_idx = needed / (long double)per_n.size() - 1;
    printf("p_min per even number   (idx = position in prime list, 0 = prime 2;"
           "  p = the prime there)\n");
    printf("  mean    idx=%8.2Lf   p=%8llu   (prime at the rounded index)\n",
           mean_idx, (unsigned long long)prime_at(primes, (uint64_t)(mean_idx + 0.5L)));
    for (double p : {0.5, 0.9, 0.99, 0.999, 1.0}) {
        uint64_t i = percentile(per_n, p);
        printf("  p%-5.4g  idx=%8llu   p=%8llu\n", p * 100,
               (unsigned long long)i, (unsigned long long)prime_at(primes, i));
    }

    printf("\nmax p_min per 32-number warp   (the warp pays for its slowest lane)\n");
    for (double p : {0.5, 0.9, 0.99, 0.999, 1.0}) {
        uint64_t i = percentile(warp_max, p);
        printf("  p%-5.4g  idx=%8llu   p=%8llu\n", p * 100,
               (unsigned long long)i, (unsigned long long)prime_at(primes, i));
    }

    printf("\nwork\n");
    printf("  iterations needed  %14.0Lf\n", needed);
    printf("  iterations paid    %14.0Lf   (warp-serialized)\n", paid);
    printf("  efficiency         %13.2Lf%%\n", 100.0L * needed / paid);
    printf("  waste factor       %13.2Lf x\n", paid / needed);


    // Transposed layout: one lane owns 64 consecutive even numbers packed
    // into a uint64_t, so it iterates until the slowest of its 64 is done.
    // A warp of 32 such lanes covers 2048 numbers and pays the max over all
    // of them.
    printf("\ntransposed layout (64 numbers per lane, 2048 per warp)\n");
    for (uint64_t G : {(uint64_t)64, (uint64_t)2048}) {
        uint64_t groups = count / G;
        if (!groups) continue;
        long double sum_max = 0;
        std::vector<uint32_t> gmax(groups, 0);
        for (uint64_t g = 0; g < groups; g++) {
            uint32_t m = 0;
            for (uint64_t j = 0; j < G; j++) {
                uint32_t v = idx[g * G + j];
                if (v != UINT32_MAX && v > m) m = v;
            }
            gmax[g] = m;
            sum_max += m + 1;
        }
        long double gmean = sum_max / groups;
        printf("  group of %llu\n", (unsigned long long)G);
        printf("    mean max  idx=%8.1Lf   p=%8llu   (prime at the rounded index)\n",
               gmean, (unsigned long long)prime_at(primes, (uint64_t)(gmean + 0.5L)));
        for (double q : {0.50, 0.99, 1.0}) {
            uint64_t i = percentile(gmax, q);
            printf("    p%-5.4g    idx=%8llu   p=%8llu\n", q * 100,
                   (unsigned long long)i, (unsigned long long)prime_at(primes, i));
        }
        if (G == 2048) {
            long double transposed_paid = sum_max * 32.0L;
            printf("\n  warp-iterations per even number\n");
            printf("    transposed  %8.3Lf\n", transposed_paid / (long double)count);
            printf("    current     %8.3Lf\n", paid / (long double)count);
            printf("    speedup     %8.2Lf x\n", paid / transposed_paid);
        }
    }

    

    printf("\nresolution rate by cut-off K (bulk pass over the first K primes)\n");
    printf("  %8s %10s %14s %14s\n", "K", "largest p", "numbers done", "warps done");
    for (uint32_t K : {8u, 16u, 32u, 48u, 64u, 96u, 128u, 192u, 256u, 512u}) {
        uint64_t n_done = 0, w_done = 0;
        for (uint64_t k = 0; k < count; k++)
            if (idx[k] != UINT32_MAX && idx[k] < K) n_done++;
        for (uint64_t w = 0; w < warps; w++)
            if (warp_max[w] < K) w_done++;
        printf("  %8u %10llu %13.4f%% %13.4f%%\n", K,
               (unsigned long long)prime_at(primes, K - 1),
               100.0 * (double)n_done / (double)count,
               100.0 * (double)w_done / (double)warps);
    }

    return 0;
}
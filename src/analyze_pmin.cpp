// analyze_pmin.cpp
//
// Measures how much work goldbach_phase1_kernel actually does per even number,
// and how much a warp pays for it.
//
// Each GPU thread walks the prime list ascending and exits at the first p with
// n - p prime. Call that position p_min_idx(n). A warp of 32 lanes runs until
// its slowest lane finishes, so the cost of 32 consecutive even numbers is set
// by the maximum p_min_idx across the group, not the average.
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

    printf("p_min_idx per even number (0 = the prime 2)\n");
    printf("  mean   %8.2Lf\n", needed / (long double)per_n.size() - 1);
    for (double p : {0.5, 0.9, 0.99, 0.999, 1.0})
        printf("  p%-5.4g %8llu\n", p * 100,
               (unsigned long long)percentile(per_n, p));

    printf("\nmax p_min_idx per 32-number warp\n");
    for (double p : {0.5, 0.9, 0.99, 0.999, 1.0})
        printf("  p%-5.4g %8llu\n", p * 100,
               (unsigned long long)percentile(warp_max, p));

    printf("\nwork\n");
    printf("  iterations needed  %14.0Lf\n", needed);
    printf("  iterations paid    %14.0Lf   (warp-serialized)\n", paid);
    printf("  efficiency         %13.2Lf%%\n", 100.0L * needed / paid);
    printf("  waste factor       %13.2Lf x\n", paid / needed);

    printf("\nresolution rate by cut-off K (bulk pass over first K primes)\n");
    printf("  %8s %14s %14s\n", "K", "numbers done", "warps done");
    for (uint32_t K : {8u, 16u, 32u, 48u, 64u, 96u, 128u, 192u, 256u, 512u}) {
        uint64_t n_done = 0, w_done = 0;
        for (uint64_t k = 0; k < count; k++)
            if (idx[k] != UINT32_MAX && idx[k] < K) n_done++;
        for (uint64_t w = 0; w < warps; w++)
            if (warp_max[w] < K) w_done++;
        printf("  %8u %13.4f%% %13.4f%%\n", K,
               100.0 * (double)n_done / (double)count,
               100.0 * (double)w_done / (double)warps);
    }

    return 0;
}
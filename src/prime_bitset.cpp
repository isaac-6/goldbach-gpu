// prime_bitset.cpp
// Builds a PrimeBitset using a parallelized segmented sieve.
// Parallelization strategy: divide the OUTPUT range into segments,
// one per thread. Each thread owns its own memory region exclusively
// — no sharing, no atomics, no contention.

#include "prime_bitset.hpp"
#include <cmath>
#include <vector>
#include <omp.h>

namespace goldbach {

static inline uint64_t isqrt(uint64_t n) {
    uint64_t r = std::sqrt((double)n);
    while (r * r > n) r--;
    while ((r + 1) * (r + 1) <= n) r++;
    return r;
}

PrimeBitset build_prime_bitset(uint64_t limit) {
    if (limit < 3) return PrimeBitset(limit);

    PrimeBitset bitset(limit);

    // -------------------------------------------------------
    // Step 1: Simple sieve up to sqrt(limit) — single threaded.
    // Fast enough that parallelizing this isn't worth it.
    // -------------------------------------------------------
    uint64_t sq = isqrt(limit);
    std::vector<char> small(sq + 1, 1);
    small[0] = small[1] = 0;

    for (uint64_t i = 2; i * i <= sq; i++)
        if (small[i])
            for (uint64_t j = i * i; j <= sq; j += i)
                small[j] = 0;

    std::vector<uint64_t> base_primes;
    base_primes.reserve(sq / 10);
    for (uint64_t i = 3; i <= sq; i += 2)
        if (small[i]) base_primes.push_back(i);

    // -------------------------------------------------------
    // Step 2: Parallel marking — divide range into segments.
    //
    // Each thread gets an exclusive slice of the output bitset.
    // Thread t owns odd numbers in [seg_low_t, seg_high_t].
    //
    // Slices must be exclusive in WORDS, not just in bit index. The bitset is
    // stored as uint64_t and clear() is a non-atomic read-modify-write, so if
    // a slice boundary falls mid-word the two adjacent threads race on that
    // word and lose each other's clears. A lost clear leaves the bit SET, i.e.
    // reports a composite as prime -- silently, and in the direction that makes
    // verification wrongly succeed. Rounding odds_per_thread up to a whole
    // number of words is what makes the "no atomics" claim below actually true.
    //
    // We work in terms of odd numbers only:
    //   odd number k → bit index (k-3)/2
    // -------------------------------------------------------
    int nthreads = omp_get_max_threads();

    // Total number of odd numbers in [3, limit]
    uint64_t total_odds = (limit - 3) / 2 + 1;

    // Each thread handles roughly total_odds/nthreads odd numbers
    uint64_t odds_per_thread = (total_odds + nthreads - 1) / nthreads;

    // Round up to a whole number of 64-bit words so no word is shared.
    // This only ever grows the slice, so coverage is preserved: nthreads *
    // odds_per_thread >= total_odds still holds. Threads whose bit_start runs
    // past total_odds get bit_end < bit_start from the clamp below and mark
    // nothing, because `first` is derived from seg_low and so always exceeds
    // seg_high. The buffer is protected by that same clamp, not by this value.
    odds_per_thread = (odds_per_thread + 63) & ~63ULL;

    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int t = 0; t < nthreads; t++) {
        // Compute this thread's range of odd numbers
        // bit_start and bit_end are indices into the bitset
        uint64_t bit_start = (uint64_t)t * odds_per_thread;
        uint64_t bit_end   = std::min(bit_start + odds_per_thread - 1,
                                      total_odds - 1);

        // Convert bit indices back to odd numbers
        // odd number = 2 * bit_index + 3
        uint64_t seg_low  = 2 * bit_start + 3;
        uint64_t seg_high = 2 * bit_end   + 3;

        // For each base prime, mark its multiples in our segment
        for (uint64_t p : base_primes) {
            // First odd multiple of p in [seg_low, seg_high]
            // Must be >= p*p (smaller multiples handled by smaller primes)
            uint64_t first = ((seg_low + p - 1) / p) * p;

            // Make sure first is odd
            if (first % 2 == 0) first += p;

            // Also ensure first >= p*p
            if (first < p * p) first = p * p;
            if (first % 2 == 0) first += p;  // keep odd

            // Mark odd multiples in our segment
            // Step by 2*p to stay on odd numbers
            for (uint64_t j = first; j <= seg_high; j += 2 * p)
                bitset.clear(j);  // safe — this thread owns these whole words
        }
    }

    return bitset;
}

} // namespace goldbach
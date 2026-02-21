// prime_bitset.cpp
// Builds a PrimeBitset using a segmented sieve of Eratosthenes.
// Only odd numbers are stored — even numbers are never prime (except 2),
// so we skip them entirely, halving memory and improving cache usage.

#include "prime_bitset.hpp"
#include <cmath>
#include <vector>

namespace goldbach {

// Integer square root — avoids floating point rounding errors
static inline uint64_t isqrt(uint64_t n) {
    uint64_t r = std::sqrt((double)n);
    while (r * r > n) r--;
    while ((r + 1) * (r + 1) <= n) r++;
    return r;
}

PrimeBitset build_prime_bitset(uint64_t limit) {
    // Guard: nothing to sieve below 3
    if (limit < 3) return PrimeBitset(limit);

    PrimeBitset bitset(limit);

    // -------------------------------------------------------
    // Step 1: Simple sieve up to sqrt(limit) to get base primes.
    // We only need odd base primes — 2 is handled separately
    // and even multiples are never stored in the bitset.
    // -------------------------------------------------------
    uint64_t sq = isqrt(limit);
    std::vector<char> small(sq + 1, 1);
    small[0] = small[1] = 0;

    for (uint64_t i = 2; i * i <= sq; i++)
        if (small[i])
            for (uint64_t j = i * i; j <= sq; j += i)
                small[j] = 0;

    // Collect odd base primes only (skip 2 — even multiples
    // are not stored in the bitset anyway)
    std::vector<uint64_t> base_primes;
    base_primes.reserve(sq / 10);  // π(√n) rough estimate
    for (uint64_t i = 3; i <= sq; i += 2)
        if (small[i]) base_primes.push_back(i);

    // -------------------------------------------------------
    // Step 2: For each base prime p, mark its odd multiples
    // as composite in the bitset.
    //
    // We start at p*p because all smaller multiples of p
    // have already been marked by smaller primes.
    //
    // We step by 2*p to stay on odd numbers only —
    // even multiples of p are not stored in the bitset.
    // -------------------------------------------------------
    for (uint64_t p : base_primes) {
        uint64_t start = p * p;
        if (start > limit) break;

        // Mark odd multiples: p^2, p^2 + 2p, p^2 + 4p, ...
        for (uint64_t j = start; j <= limit; j += 2 * p)
            bitset.clear(j);
    }

    return bitset;
}

} // namespace goldbach
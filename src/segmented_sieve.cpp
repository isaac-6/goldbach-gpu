#include <cstdint>
#include <vector>
#include <cmath>

static uint64_t isqrt(uint64_t n) {
    uint64_t r = std::sqrt((double)n);
    while (r * r > n) r--;
    while ((r + 1) * (r + 1) <= n) r++;
    return r;
}

// Returns all primes up to `limit`
std::vector<uint64_t> simple_sieve(uint64_t limit) {
    std::vector<char> mark(limit + 1, 1);
    mark[0] = mark[1] = 0;

    for (uint64_t i = 2; i * i <= limit; i++)
        if (mark[i])
            for (uint64_t j = i * i; j <= limit; j += i)
                mark[j] = 0;

    std::vector<uint64_t> primes;
    for (uint64_t i = 2; i <= limit; i++)
        if (mark[i]) primes.push_back(i);
    return primes;
}

// Segmented sieve over [low, high], returns is_prime[i] for low+i
std::vector<char> segmented_sieve(uint64_t low, uint64_t high) {
    auto base_primes = simple_sieve(isqrt(high));

    std::vector<char> is_prime(high - low + 1, 1);

    // Handle 0 and 1 explicitly
    if (low == 0) is_prime[0] = 0;
    if (low <= 1 && high >= 1) is_prime[1 - low] = 0;

    for (uint64_t p : base_primes) {
        // First multiple of p in [low, high]
        uint64_t start = ((low + p - 1) / p) * p;
        // Don't mark p itself as composite
        // if (start == p) start += p;
        if (start < 2 * p) start = 2 * p;

        for (uint64_t j = start; j <= high; j += p)
            is_prime[j - low] = 0;
    }

    return is_prime;
}

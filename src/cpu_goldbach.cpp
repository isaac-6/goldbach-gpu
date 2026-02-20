#include <cstdint>
#include <vector>
#include <iostream>
#include <chrono>

// These live in segmented_sieve.cpp
std::vector<char> segmented_sieve(uint64_t low, uint64_t high);

// -------------------------------------------------------
// Goldbach verify: does n have at least one partition?
// n must be even and >= 4.
// is_prime is indexed from 0, so is_prime[k] = 1 iff k is prime.
// -------------------------------------------------------
bool goldbach_verify(uint64_t n, const std::vector<char>& is_prime) {
    // We scan p from 2 up to n/2.
    // If p is prime and n-p is prime, we found a partition.
    // We only need to go up to n/2 because pairs are unordered:
    // (p, q) and (q, p) are the same partition.
    for (uint64_t p = 2; p <= n / 2; p++) {
        if (is_prime[p] && is_prime[n - p])
            return true;
    }
    return false;
}

// -------------------------------------------------------
// Goldbach count: how many unordered partitions does n have?
// -------------------------------------------------------
uint64_t goldbach_count(uint64_t n, const std::vector<char>& is_prime) {
    uint64_t count = 0;
    for (uint64_t p = 2; p <= n / 2; p++) {
        if (is_prime[p] && is_prime[n - p])
            count++;
    }
    return count;
}

int main() {
    // -------------------------------------------------------
    // Configuration — change these to test different ranges
    // -------------------------------------------------------
    const uint64_t LIMIT = 1'000'000'000;   // check all even n up to this
    const bool PRINT_COUNTS = false;    // set true to print c(n) for every n
    const bool STOP_ON_FAIL = true;     // stop immediately if Goldbach fails

    std::cout << "Building prime table up to " << LIMIT << "...\n";

    // Time the sieve
    auto t0 = std::chrono::high_resolution_clock::now();

    // Build full primality lookup for [0, LIMIT]
    // is_prime[k] == 1 iff k is prime
    auto is_prime = segmented_sieve(0, LIMIT);

    auto t1 = std::chrono::high_resolution_clock::now();
    double sieve_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Sieve done in " << sieve_ms << " ms\n";

    // -------------------------------------------------------
    // Goldbach verification loop
    // -------------------------------------------------------
    std::cout << "Checking Goldbach for all even n in [4, " << LIMIT << "]...\n";

    auto t2 = std::chrono::high_resolution_clock::now();

    uint64_t failures = 0;
    uint64_t checked  = 0;

    for (uint64_t n = 4; n <= LIMIT; n += 2) {
        checked++;

        if (PRINT_COUNTS) {
            uint64_t c = goldbach_count(n, is_prime);
            std::cout << "c(" << n << ") = " << c << "\n";
        } else {
            if (!goldbach_verify(n, is_prime)) {
                std::cout << "FAIL: Goldbach fails at n = " << n << "\n";
                failures++;
                if (STOP_ON_FAIL) break;
            }
        }
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    double check_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // -------------------------------------------------------
    // Summary
    // -------------------------------------------------------
    std::cout << "\n--- Summary ---\n";
    std::cout << "Even numbers checked : " << checked << "\n";
    std::cout << "Failures found       : " << failures << "\n";
    std::cout << "Sieve time           : " << sieve_ms << " ms\n";
    std::cout << "Goldbach check time  : " << check_ms << " ms\n";
    std::cout << "Total time           : " << sieve_ms + check_ms << " ms\n";

    if (failures == 0)
        std::cout << "\nAll even numbers up to " << LIMIT << " satisfy Goldbach. ✓\n";

    return (failures == 0) ? 0 : 1;
}

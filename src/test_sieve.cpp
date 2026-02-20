#include <iostream>
#include <vector>
#include <cstdint>
#include <cassert>

// Forward declarations — these live in segmented_sieve.cpp
std::vector<uint64_t> simple_sieve(uint64_t limit);
std::vector<char> segmented_sieve(uint64_t low, uint64_t high);

// Count primes in [low, high] using the segmented sieve
uint64_t count_primes(uint64_t low, uint64_t high) {
    auto is_prime = segmented_sieve(low, high);
    uint64_t count = 0;
    for (char c : is_prime)
        if (c) count++;
    return count;
}

// Count primes up to n (i.e., pi(n))
uint64_t pi(uint64_t n) {
    return count_primes(0, n);
}

int main() {
    // Known values of pi(n)
    struct TestCase {
        uint64_t n;
        uint64_t expected;
    };

    TestCase cases[] = {
        {10,            4},
        {100,           25},
        {1'000,         168},
        {10'000,        1'229},
        {100'000,       9'592},
        {1'000'000,     78'498},
        {10'000'000,    664'579},
        {100'000'000,   5'761'455},
    };

    bool all_passed = true;

    for (auto& [n, expected] : cases) {
        uint64_t result = pi(n);
        bool passed = (result == expected);
        std::cout << "pi(10^?) up to " << n
                  << ": expected " << expected
                  << ", got " << result
                  << (passed ? "  PASS" : "  FAIL") << "\n";
        if (!passed) all_passed = false;
    }

    std::cout << "\n" << (all_passed ? "All tests passed." : "SOME TESTS FAILED.") << "\n";
    return all_passed ? 0 : 1;
}

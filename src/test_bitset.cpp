// test_bitset.cpp
// Validates PrimeBitset against known prime counts and values.

#include <iostream>
#include <vector>
#include <cstdint>
#include "prime_bitset.hpp"

using namespace goldbach;

int main() {
    bool all_passed = true;

    // -------------------------------------------------------
    // Test 1: Small primes by hand
    // -------------------------------------------------------
    {
        auto bs = build_prime_bitset(30);
        std::vector<uint64_t> expected = {2,3,5,7,11,13,17,19,23,29};
        std::vector<uint64_t> got;

        if (bs.is_prime(2)) got.push_back(2);
        for (uint64_t n = 3; n <= 30; n += 2)
            if (bs.is_prime(n)) got.push_back(n);

        bool pass = (got == expected);
        std::cout << "Test 1 (primes up to 30): "
                  << (pass ? "PASS" : "FAIL") << "\n";
        if (!pass) {
            std::cout << "  Expected: ";
            for (auto p : expected) std::cout << p << " ";
            std::cout << "\n  Got:      ";
            for (auto p : got) std::cout << p << " ";
            std::cout << "\n";
        }
        all_passed &= pass;
    }

    // -------------------------------------------------------
    // Test 2: Known prime counts (pi(n))
    // -------------------------------------------------------
    struct TestCase { uint64_t n; uint64_t expected; };
    TestCase cases[] = {
        {1'000,         168},
        {10'000,        1'229},
        {100'000,       9'592},
        {1'000'000,     78'498},
        {10'000'000,    664'579},
        {100'000'000,   5'761'455},
    };

    for (auto& [n, expected] : cases) {
        auto bs = build_prime_bitset(n);

        uint64_t count = bs.is_prime(2) ? 1 : 0;
        for (uint64_t k = 3; k <= n; k += 2)
            if (bs.is_prime(k)) count++;

        bool pass = (count == expected);
        std::cout << "pi(" << n << ") = " << count
                  << " expected " << expected
                  << (pass ? "  PASS" : "  FAIL") << "\n";
        all_passed &= pass;
    }

    // -------------------------------------------------------
    // Test 3: Memory usage
    // -------------------------------------------------------
    {
        auto bs = build_prime_bitset(1'000'000'000);
        std::cout << "\nMemory for bitset up to 10^9: "
                  << bs.memory_bytes() / 1024 / 1024 << " MB"
                  << " (vs 953 MB for byte array)\n";
    }

    std::cout << "\n" << (all_passed ? "All tests passed." : "SOME TESTS FAILED.") << "\n";
    return all_passed ? 0 : 1;
}
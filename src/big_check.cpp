// big_check.cpp
// Goldbach verification for arbitrarily large even numbers.
// Uses GMP (GNU Multiple Precision) for big integer arithmetic.
// Works for any even number regardless of size — no uint64_t limit.
//
// Strategy:
//   1. Generate small primes up to a chunk size on CPU
//   2. For each prime p <= n/2, compute q = n - p using GMP
//   3. Test q with GMP's Miller-Rabin primality test
//   4. Report first valid partition found
//
// Usage:
//   ./big_check 123456789012345678901234567890   (any even number as string)
//   ./big_check                                  (uses default 10^50)

#include <gmp.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdint>

// -------------------------------------------------------
// Generate all primes up to `limit` using simple sieve.
// -------------------------------------------------------
std::vector<uint64_t> generate_primes(uint64_t limit) {
    std::vector<char> is_prime(limit + 1, 1);
    is_prime[0] = is_prime[1] = 0;

    for (uint64_t i = 2; i * i <= limit; i++)
        if (is_prime[i])
            for (uint64_t j = i * i; j <= limit; j += i)
                is_prime[j] = 0;

    std::vector<uint64_t> primes;
    for (uint64_t i = 2; i <= limit; i++)
        if (is_prime[i]) primes.push_back(i);

    return primes;
}

// -------------------------------------------------------
// Check Goldbach for a single large even number n.
// n is passed as a decimal string — no size limit.
// -------------------------------------------------------
void big_check(const std::string& n_str) {
    std::cout << "Checking Goldbach for n = " << n_str << "\n";
    std::cout << "  (" << n_str.size() << " digits)\n\n";

    // -------------------------------------------------------
    // Step 1: Parse n from string into GMP integer
    // -------------------------------------------------------
    mpz_t n, n_half, q, p_mpz;
    mpz_init(n);
    mpz_init(n_half);
    mpz_init(q);
    mpz_init(p_mpz);

    if (mpz_set_str(n, n_str.c_str(), 10) != 0) {
        std::cerr << "Error: invalid number string\n";
        mpz_clear(n); mpz_clear(n_half);
        mpz_clear(q); mpz_clear(p_mpz);
        return;
    }

    // Validate: must be even and >= 4
    if (mpz_odd_p(n)) {
        std::cerr << "Error: n must be even\n";
        mpz_clear(n); mpz_clear(n_half);
        mpz_clear(q); mpz_clear(p_mpz);
        return;
    }

    if (mpz_cmp_ui(n, 4) < 0) {
        std::cerr << "Error: n must be >= 4\n";
        mpz_clear(n); mpz_clear(n_half);
        mpz_clear(q); mpz_clear(p_mpz);
        return;
    }

    // Precompute n/2 — we never need p > n/2
    // because pairs (p,q) and (q,p) are the same partition
    mpz_fdiv_q_2exp(n_half, n, 1);  // n_half = n / 2

    // -------------------------------------------------------
    // Step 2: Generate candidate primes p up to chunk size.
    // We start with primes up to 10^7 — almost always enough.
    // If no partition found, extend the search.
    // -------------------------------------------------------
    const uint64_t CHUNK = 10'000'000ULL;  // 10^7

    auto t_start = std::chrono::high_resolution_clock::now();

    bool found = false;
    uint64_t chunk_start = 0;
    uint64_t p_found = 0;
    std::string q_found_str;

    while (!found) {
        uint64_t chunk_end = chunk_start + CHUNK;

        std::cout << "Generating primes up to " << chunk_end << "...\n";
        auto primes = generate_primes(chunk_end);

        std::cout << "Testing " << primes.size()
                  << " primes as candidates for p...\n";

        for (uint64_t p : primes) {
            // Skip primes handled in previous chunks
            if (p <= chunk_start) continue;

            // Set p as GMP integer
            mpz_set_ui(p_mpz, p);

            // Break early: once p > n/2, all remaining
            // primes give q < p — duplicate pairs, skip all
            if (mpz_cmp(p_mpz, n_half) > 0) {
                found = false;  // exhausted all pairs
                goto done;
            }

            // Compute q = n - p
            mpz_sub(q, n, p_mpz);

            // q must be >= 2 to be prime
            if (mpz_cmp_ui(q, 2) < 0) continue;

            // Test if q is prime using Miller-Rabin
            // 25 rounds → error probability < 4^(-25) ≈ 10^(-15)
            // Check primality BEFORE the q>=p comparison —
            // primality test is expensive, but q<p is already
            // impossible here since p <= n/2 guarantees q >= p
            if (mpz_probab_prime_p(q, 25) > 0) {
                // Found a valid partition
                found = true;
                p_found = p;

                char* q_str = mpz_get_str(nullptr, 10, q);
                q_found_str = std::string(q_str);
                free(q_str);
                goto done;
            }
        }

        if (!found) {
            std::cout << "No partition found in primes up to "
                      << chunk_end << " — extending search...\n";
            chunk_start = chunk_end;
        }
    }

done:
    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,
                std::milli>(t_end - t_start).count();

    // -------------------------------------------------------
    // Results
    // -------------------------------------------------------
    std::cout << "\n--- Result ---\n";
    if (found) {
        std::cout << n_str << "\n  = " << p_found
                  << " + " << q_found_str << "\n";
        std::cout << "\np is " << std::to_string(p_found).size()
                  << " digits, q is " << q_found_str.size()
                  << " digits\n";
        std::cout << "Goldbach holds. ✓\n";
    } else {
        std::cout << "NO PARTITION FOUND — counterexample!\n";
    }

    std::cout << "\n--- Timing ---\n";
    std::cout << "Total time: " << ms << " ms\n";

    // Cleanup GMP
    mpz_clear(n);
    mpz_clear(n_half);
    mpz_clear(q);
    mpz_clear(p_mpz);
}

int main(int argc, char* argv[]) {
    // Default: 10^50
    std::string n_str =
        "100000000000000000000000000000000000000000000000000";

    if (argc > 1) {
        n_str = argv[1];
    }

    // Make sure n is even — if odd, subtract 1
    if ((n_str.back() - '0') % 2 != 0) {
        std::cerr << "Warning: n is odd, using n-1 instead\n";
        n_str.back() -= 1;
    }

    big_check(n_str);
    return 0;
}


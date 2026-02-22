// big_check.cpp
// Goldbach verification for arbitrarily large even numbers.
// Uses GMP for big integer arithmetic + OpenMP for parallelism.
// Each thread tests a different prime p independently.
//
// Usage:
//   ./big_check                                    (default 10^50)
//   ./big_check "1000...000"                       (any even number)

#include <gmp.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdint>
#include <atomic>
#include <omp.h>

// -------------------------------------------------------
// Generate all primes up to limit using simple sieve.
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

void big_check(const std::string& n_str) {
    std::cout << "Checking Goldbach for n = " << n_str << "\n";
    std::cout << "  (" << n_str.size() << " digits)\n\n";

    // -------------------------------------------------------
    // Step 1: Parse n into GMP integer
    // -------------------------------------------------------
    mpz_t n, n_half;
    mpz_init(n);
    mpz_init(n_half);

    if (mpz_set_str(n, n_str.c_str(), 10) != 0) {
        std::cerr << "Error: invalid number string\n";
        mpz_clear(n); mpz_clear(n_half);
        return;
    }

    if (mpz_odd_p(n)) {
        std::cerr << "Error: n must be even\n";
        mpz_clear(n); mpz_clear(n_half);
        return;
    }

    if (mpz_cmp_ui(n, 4) < 0) {
        std::cerr << "Error: n must be >= 4\n";
        mpz_clear(n); mpz_clear(n_half);
        return;
    }

    // Precompute n/2 -- never need p > n/2
    mpz_fdiv_q_2exp(n_half, n, 1);

    // -------------------------------------------------------
    // Step 2: Generate candidate primes
    // -------------------------------------------------------
    const uint64_t CHUNK = 10'000'000ULL;

    auto t_start = std::chrono::high_resolution_clock::now();

    std::cout << "Generating primes up to " << CHUNK << "...\n";
    auto primes = generate_primes(CHUNK);
    std::cout << "Testing " << primes.size()
              << " primes using "
              << omp_get_max_threads()
              << " threads...\n";

    // -------------------------------------------------------
    // Step 3: Parallel search across primes.
    //
    // Each thread gets its own GMP variables -- GMP is not
    // thread-safe when sharing mpz_t objects, so every thread
    // must have its own local copies of q and p_mpz.
    //
    // We use std::atomic<bool> found_flag to signal all threads
    // to stop as soon as any thread finds a valid partition.
    //
    // We use critical section to safely write the result.
    // -------------------------------------------------------
    std::atomic<bool> found_flag(false);
    uint64_t    p_result = 0;
    std::string q_result_str;

    #pragma omp parallel
    {
        // Each thread has its own GMP variables
        mpz_t q_local, p_mpz_local;
        mpz_init(q_local);
        mpz_init(p_mpz_local);

        #pragma omp for schedule(dynamic, 64)
        for (int64_t i = 0; i < (int64_t)primes.size(); i++) {
            // Early exit if another thread found a partition
            if (found_flag.load(std::memory_order_relaxed)) continue;

            uint64_t p = primes[i];

            // Set p as GMP integer
            mpz_set_ui(p_mpz_local, p);

            // Break early: p > n/2 means q < p, duplicate pair
            if (mpz_cmp(p_mpz_local, n_half) > 0) {
                found_flag.store(true);  // signal exhaustion
                continue;
            }

            // Compute q = n - p
            mpz_sub(q_local, n, p_mpz_local);

            // q must be >= 2
            if (mpz_cmp_ui(q_local, 2) < 0) continue;

            // Test primality with Miller-Rabin (25 rounds)
            if (mpz_probab_prime_p(q_local, 25) > 0) {
                // Found a partition -- record it
                if (!found_flag.exchange(true)) {
                    // Only first thread to find writes the result
                    #pragma omp critical
                    {
                        p_result = p;
                        char* q_str = mpz_get_str(nullptr, 10, q_local);
                        q_result_str = std::string(q_str);
                        free(q_str);
                    }
                }
            }
        }

        // Clean up thread-local GMP variables
        mpz_clear(q_local);
        mpz_clear(p_mpz_local);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,
                std::milli>(t_end - t_start).count();

    // -------------------------------------------------------
    // Results
    // -------------------------------------------------------
    std::cout << "\n--- Result ---\n";
    if (p_result > 0) {
        std::cout << n_str << "\n  = " << p_result
                  << " + " << q_result_str << "\n";
        std::cout << "\np is " << std::to_string(p_result).size()
                  << " digits, q is " << q_result_str.size()
                  << " digits\n";
        std::cout << "Goldbach holds. ✓\n";
    } else {
        std::cout << "NO PARTITION FOUND -- counterexample!\n";
    }

    std::cout << "\n--- Timing ---\n";
    std::cout << "Threads used : " << omp_get_max_threads() << "\n";
    std::cout << "Total time   : " << ms << " ms\n";

    mpz_clear(n);
    mpz_clear(n_half);
}

int main(int argc, char* argv[]) {
    std::string n_str =
        "100000000000000000000000000000000000000000000000000";

    if (argc > 1) n_str = argv[1];

    if ((n_str.back() - '0') % 2 != 0) {
        std::cerr << "Warning: n is odd, using n-1 instead\n";
        n_str.back() -= 1;
    }

    big_check(n_str);
    return 0;
}
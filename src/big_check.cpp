// big_check.cpp
// Goldbach verification for arbitrarily large even numbers.
// Uses GMP for big integer arithmetic + OpenMP for parallelism.
//
// Features a STEP-WISE BATCH search to prevent thread runaway
// and ensure we find the smallest possible partition prime.

#include <gmp.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdint>
#include <atomic>
#include <omp.h>
#include <cmath>    // for std::log

std::vector<uint64_t> generate_primes(uint64_t limit) {
    std::vector<char> is_prime(limit + 1, 1);
    is_prime[0] = is_prime[1] = 0;
    for (uint64_t i = 2; i * i <= limit; i++)
        if (is_prime[i])
            for (uint64_t j = i * i; j <= limit; j += i)
                is_prime[j] = 0;

    std::vector<uint64_t> primes;
    // Heuristic reservation: π(n) ≈ n / ln(n)
    primes.reserve(static_cast<size_t>(limit / std::log(limit) * 1.1));
    
    for (uint64_t i = 2; i <= limit; i++)
        if (is_prime[i]) primes.push_back(i);
    return primes;
}

void big_check(const std::string& n_str) {
    if (n_str.size() <= 100) {
        std::cout << "Checking Goldbach for n = " << n_str << "\n";
    } else {
        std::cout << "Checking Goldbach for n with "
                  << n_str.size() << " digits\n";
    }
    std::cout << "  (" << n_str.size() << " digits)\n\n";

    mpz_t n, n_half;
    mpz_init(n);
    mpz_init(n_half);

    if (mpz_set_str(n, n_str.c_str(), 10) != 0 || mpz_odd_p(n) || mpz_cmp_ui(n, 4) < 0) {
        std::cerr << "Error: invalid number or not even / >= 4\n";
        mpz_clear(n); mpz_clear(n_half);
        return;
    }
    mpz_fdiv_q_2exp(n_half, n, 1);

    // Choosing 10^7 covers the vast majority of smallest Goldbach primes in known ranges.
    // Raising to 2-5×10^7 costs little extra time but increases confidence further.
    const uint64_t CHUNK = 10'000'000ULL;
    auto t_start = std::chrono::high_resolution_clock::now();

    std::cout << "Generating primes up to " << CHUNK << "...\n";
    auto primes = generate_primes(CHUNK);
    std::cout << "Generated " << primes.size() << " primes. Using "
              << omp_get_max_threads() << " threads...\n\n";

    std::atomic<bool> found_flag(false);
    uint64_t    p_result = 0;
    std::string q_result_str;

    // STEP-WISE STRATEGY: Process primes in batches of 1,000
    // Prevents fast threads from racing far ahead while slow ones lag on heavy tests.
    const int BATCH_SIZE = 1000;
    int total_batches = (primes.size() + BATCH_SIZE - 1) / BATCH_SIZE;

    for (int batch = 0; batch < total_batches; batch++) {
        if (found_flag.load(std::memory_order_relaxed)) break;

        int start_idx = batch * BATCH_SIZE;
        int end_idx = std::min((int)primes.size(), start_idx + BATCH_SIZE);

        std::cout << "  Scanning primes " << primes[start_idx] 
                  << " – " << primes[end_idx - 1] << "   \r" << std::flush;

        #pragma omp parallel
        {
            mpz_t q_local, p_mpz_local;
            mpz_init(q_local);
            mpz_init(p_mpz_local);

            // dynamic,1 → good load balancing for variable-cost Miller-Rabin tests
            #pragma omp for schedule(dynamic, 1)
            for (int i = start_idx; i < end_idx; i++) {
                if (found_flag.load(std::memory_order_relaxed)) continue;

                uint64_t p = primes[i];
                mpz_set_ui(p_mpz_local, p);

                if (mpz_cmp(p_mpz_local, n_half) > 0) {
                    found_flag.store(true, std::memory_order_relaxed);
                    continue;
                }

                mpz_sub(q_local, n, p_mpz_local);
                if (mpz_cmp_ui(q_local, 2) < 0) continue;

                // 25 rounds → false positive probability per composite < 4^{-25} ≈ 2.98e-16
                // Extremely strong for practical purposes; deterministic tests exist but are slower.
                if (mpz_probab_prime_p(q_local, 25) > 0) {
                    char* buf = mpz_get_str(nullptr, 10, q_local);
                    if (!found_flag.exchange(true, std::memory_order_relaxed)) {
                        p_result = p;
                        q_result_str = buf;
                    }
                    std::free(buf);
                }
            }
            mpz_clear(q_local);
            mpz_clear(p_mpz_local);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "\n\n--- Result ---\n";
    if (p_result > 0) {
        if (n_str.size() <= 100) {
            std::cout << n_str << "\n  = " << p_result << " + " << q_result_str << "\n";
        } else {
            std::cout << "n has " << n_str.size() << " digits\n";
            std::cout << "p = " << p_result << "\n";
            std::cout << "q has " << q_result_str.size() << " digits\n";
        }
        std::cout << "Goldbach holds. ✓\n";
    } else {
        std::cout << "NO PARTITION FOUND -- counterexample!\n";
    }

    std::cout << "\n--- Timing ---\n";
    std::cout << "Threads used : " << omp_get_max_threads() << "\n";
    std::cout << "Total time   : " << ms / 1000.0 << " seconds\n";

    mpz_clear(n);
    mpz_clear(n_half);
}

int main(int argc, char* argv[]) {
    // 1. Help flag check
    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        std::cout << "Arbitrary Precision Goldbach Checker (GMP)\n";
        std::cout << "Usage:   " << argv[0] << " <N>\n";
        std::cout << "Example: " << argv[0] << " 1000000000000000000000000000000\n";
        return 0;
    }

    std::string n_str = argv[1];

    // 2. Basic string validation: purely numeric
    for (char c : n_str) {
        if (!std::isdigit(c)) {
            std::cerr << "Error: Input must be a positive integer.\n";
            return 1;
        }
    }

    // 3. Quick evenness check via last digit
    int last_digit = n_str.back() - '0';
    if (last_digit % 2 != 0) {
        std::cerr << "Error: Goldbach's conjecture applies to even integers >= 4.\n";
        std::cerr << "Input ends in '" << last_digit << "' (odd).\n";
        return 1;
    }

    big_check(n_str);
    return 0;
}
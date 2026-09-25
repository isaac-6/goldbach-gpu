// big_check.cpp
// Goldbach verification for arbitrarily large even numbers.
// GMP for big-integer arithmetic, OpenMP for parallelism.
//
// Searches for the SMALLEST prime p <= L with n - p a (probable) prime, where
// L = min(p_max, floor(n/2)). The result is independent of thread count and
// scheduling: see the ordering argument above the search loop.
//
// Exit codes:
//   0  a partition was found
//   3  no partition with p <= L; this is a search-limit result, not a
//      counterexample to Goldbach
//   1  invalid input

#include <gmp.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <limits>
#include <omp.h>
#include <cmath>

static const uint64_t DEFAULT_P_MAX = 10'000'000ULL;
static const int      BATCH_SIZE    = 1000;
static const size_t   NO_IDX        = std::numeric_limits<size_t>::max();

std::vector<uint64_t> generate_primes(uint64_t limit) {
    std::vector<uint64_t> primes;
    if (limit < 2) return primes;

    std::vector<char> is_prime(limit + 1, 1);
    is_prime[0] = is_prime[1] = 0;
    for (uint64_t i = 2; i * i <= limit; i++)
        if (is_prime[i])
            for (uint64_t j = i * i; j <= limit; j += i)
                is_prime[j] = 0;

    // Heuristic reservation: pi(n) ~ n / ln(n)
    if (limit >= 3)
        primes.reserve(static_cast<size_t>(limit / std::log((double)limit) * 1.1));
    for (uint64_t i = 2; i <= limit; i++)
        if (is_prime[i]) primes.push_back(i);
    return primes;
}

// -------------------------------------------------------
// Input parsing
// -------------------------------------------------------
// Accepts a plain decimal string, or a^b, a^b+c, a^b-c, evaluated with GMP so
// that the appendix numbers are reproducible directly from their command lines
// (e.g. "2^33220" rather than a 10000-digit literal).
static bool parse_n(const std::string& s, mpz_t out) {
    if (s.empty()) return false;

    size_t caret = s.find('^');
    if (caret == std::string::npos)
        return mpz_set_str(out, s.c_str(), 10) == 0;

    std::string a_str = s.substr(0, caret);
    std::string rest  = s.substr(caret + 1);
    if (a_str.empty() || rest.empty()) return false;

    // An operator at position 0 would mean an empty exponent.
    size_t oppos = rest.find_first_of("+-");
    if (oppos == 0) return false;

    std::string b_str = (oppos == std::string::npos) ? rest : rest.substr(0, oppos);
    char        op    = (oppos == std::string::npos) ? 0    : rest[oppos];
    std::string c_str = (oppos == std::string::npos) ? ""   : rest.substr(oppos + 1);
    if (b_str.empty() || (op && c_str.empty())) return false;

    for (char ch : b_str) if (!isdigit((unsigned char)ch)) return false;

    mpz_t a, c;
    mpz_init(a); mpz_init(c);
    bool ok = (mpz_set_str(a, a_str.c_str(), 10) == 0);
    if (ok && op) ok = (mpz_set_str(c, c_str.c_str(), 10) == 0);

    if (ok) {
        errno = 0;
        char* endp = nullptr;
        unsigned long b = std::strtoul(b_str.c_str(), &endp, 10);
        if (errno != 0 || *endp != '\0') ok = false;
        else {
            mpz_pow_ui(out, a, b);
            if (op == '+') mpz_add(out, out, c);
            else if (op == '-') mpz_sub(out, out, c);
        }
    }
    mpz_clear(a); mpz_clear(c);
    return ok;
}

// Full decimal string when q is short, otherwise digit count plus the leading
// and trailing 20 digits.
static void print_q(const mpz_t q, size_t digits) {
    char* buf = mpz_get_str(nullptr, 10, q);
    std::string s(buf);
    std::free(buf);
    if (digits <= 100) {
        std::cout << "q = " << s << "\n";
    } else {
        std::cout << "q has " << digits << " digits\n";
        std::cout << "q = " << s.substr(0, 20) << "..." << s.substr(s.size() - 20) << "\n";
    }
}

struct Outcome {
    bool     found     = false;
    uint64_t p         = 0;
    size_t   q_digits  = 0;
    size_t   candidates = 0;   // index of p, plus one
    uint64_t prp_calls = 0;
    int      prp_status = 0;   // 2 = proven prime, 1 = probable prime (BPSW)
};

static Outcome search(const mpz_t n, const std::vector<uint64_t>& primes, bool quiet) {
    Outcome out;

    // best_idx is the smallest index known to have a prime complement. Threads
    // CAS it downwards only, and skip any index above it, so no index below the
    // eventual winner is ever skipped. Batches run in order and the search stops
    // after the first batch containing a hit, so the winner is the smallest
    // index overall -- independent of thread count and scheduling.
    std::atomic<size_t>   best_idx(NO_IDX);
    std::atomic<uint64_t> prp_calls(0);

    size_t total = primes.size();
    size_t batches = (total + BATCH_SIZE - 1) / BATCH_SIZE;

    for (size_t batch = 0; batch < batches; batch++) {
        size_t start_idx = batch * BATCH_SIZE;
        size_t end_idx   = std::min(total, start_idx + (size_t)BATCH_SIZE);

        if (!quiet)
            std::cout << "  Scanning primes " << primes[start_idx]
                      << " - " << primes[end_idx - 1] << "   \r" << std::flush;

        #pragma omp parallel
        {
            mpz_t q_local, p_local;
            mpz_init(q_local);
            mpz_init(p_local);
            uint64_t local_calls = 0;

            #pragma omp for schedule(dynamic, 1)
            for (long long ii = (long long)start_idx; ii < (long long)end_idx; ii++) {
                size_t i = (size_t)ii;
                if (i > best_idx.load(std::memory_order_relaxed)) continue;

                mpz_set_ui(p_local, primes[i]);
                mpz_sub(q_local, n, p_local);
                if (mpz_cmp_ui(q_local, 2) < 0) continue;

                // GMP 6.3: mpz_probab_prime_p(q, reps) runs trial division, then
                // a BPSW probable-prime test, then reps-24 Miller-Rabin rounds.
                // With reps = 25 that is one Miller-Rabin round beyond BPSW.
                // Returns 2 for a proven prime, 1 for a probable prime, 0 for a
                // composite.
                local_calls++;
                if (mpz_probab_prime_p(q_local, 25) > 0) {
                    size_t cur = best_idx.load(std::memory_order_relaxed);
                    while (i < cur &&
                           !best_idx.compare_exchange_weak(cur, i,
                                                           std::memory_order_relaxed))
                        ;   // cur is refreshed by compare_exchange_weak on failure
                }
            }
            prp_calls.fetch_add(local_calls, std::memory_order_relaxed);
            mpz_clear(q_local);
            mpz_clear(p_local);
        }

        if (best_idx.load(std::memory_order_relaxed) != NO_IDX) break;
    }

    out.prp_calls = prp_calls.load();
    size_t win = best_idx.load();
    if (win == NO_IDX) {
        out.found = false;
        out.candidates = total;   // every candidate below the bound was examined
        return out;
    }

    out.found      = true;
    out.p          = primes[win];
    out.candidates = win + 1;

    mpz_t q, p;
    mpz_init(q); mpz_init(p);
    mpz_set_ui(p, out.p);
    mpz_sub(q, n, p);
    out.q_digits  = mpz_sizeinbase(q, 10);
    out.prp_status = mpz_probab_prime_p(q, 25);
    out.prp_calls++;
    if (!quiet) {
        std::cout << "\n\n--- Result ---\n";
        std::cout << "p = " << out.p << "\n";
        print_q(q, out.q_digits);
        std::cout << "n - p is "
                  << (out.prp_status == 2 ? "prime" : "probable prime (BPSW)") << "\n";
        std::cout << "Goldbach holds for this n. \n";
    }
    mpz_clear(q); mpz_clear(p);
    return out;
}

static void usage(const char* prog) {
    std::cout << "Arbitrary-precision Goldbach checker (GMP)\n";
    std::cout << "Usage: " << prog << " <N> [--p-max=<B>] [--quiet]\n";
    std::cout << "  <N>          even integer >= 4, as a decimal string or a^b, a^b+c, a^b-c\n";
    std::cout << "  --p-max=<B>  search primes p <= B (default " << DEFAULT_P_MAX << ")\n";
    std::cout << "  --quiet      print only the RESULT line\n";
    std::cout << "Exit: 0 found, 3 no partition with p <= L, 1 invalid input\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        usage(argv[0]);
        return argc < 2 ? 1 : 0;
    }

    std::string n_str;
    uint64_t    p_max = DEFAULT_P_MAX;
    bool        quiet = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--quiet") { quiet = true; }
        else if (a.rfind("--p-max=", 0) == 0) {
            std::string v = a.substr(8);
            if (v.empty()) { std::cerr << "Error: --p-max needs a value\n"; return 1; }
            for (char c : v) if (!isdigit((unsigned char)c)) {
                std::cerr << "Error: --p-max must be a non-negative integer\n"; return 1;
            }
            errno = 0; char* endp = nullptr;
            p_max = std::strtoull(v.c_str(), &endp, 10);
            if (errno != 0 || *endp != '\0') {
                std::cerr << "Error: --p-max out of range\n"; return 1;
            }
        }
        else if (a.rfind("--", 0) == 0) {
            std::cerr << "Error: unknown option " << a << "\n"; return 1;
        }
        else if (n_str.empty()) { n_str = a; }
        else { std::cerr << "Error: unexpected argument " << a << "\n"; return 1; }
    }

    if (n_str.empty()) { std::cerr << "Error: no N given\n"; return 1; }

    mpz_t n, n_half;
    mpz_init(n); mpz_init(n_half);

    if (!parse_n(n_str, n)) {
        std::cerr << "Error: could not parse N (expected a decimal string or a^b, a^b+c, a^b-c)\n";
        mpz_clear(n); mpz_clear(n_half);
        return 1;
    }
    // Evenness and magnitude are checked on the parsed value, not on the text.
    if (mpz_cmp_ui(n, 4) < 0 || mpz_odd_p(n)) {
        std::cerr << "Error: N must be an even integer >= 4\n";
        mpz_clear(n); mpz_clear(n_half);
        return 1;
    }
    mpz_fdiv_q_2exp(n_half, n, 1);

    // L = min(p_max, floor(n/2)). Restricting the candidate list to p <= L is
    // what removes the old "p > n/2" special case from the inner loop.
    uint64_t L = p_max;
    if (mpz_cmp_ui(n_half, p_max) < 0) L = mpz_get_ui(n_half);

    size_t n_digits = mpz_sizeinbase(n, 10);
    auto t_start = std::chrono::high_resolution_clock::now();

    if (!quiet) {
        std::cout << "Checking Goldbach for n";
        if (n_digits <= 100) {
            char* nb = mpz_get_str(nullptr, 10, n);
            std::cout << " = " << nb;
            std::free(nb);
        }
        std::cout << " (" << n_digits << " digits)\n";
        std::cout << "Search limit L = min(p-max, n/2) = " << L << "\n";
        std::cout << "Generating primes up to " << L << "...\n";
    }

    std::vector<uint64_t> primes = generate_primes(L);

    if (!quiet)
        std::cout << "Generated " << primes.size() << " primes. Using "
                  << omp_get_max_threads() << " threads...\n\n";

    Outcome r = search(n, primes, quiet);

    auto t_end = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t_end - t_start).count();

    if (!r.found && !quiet) {
        std::cout << "\n\n--- Result ---\n";
        std::cout << "No Goldbach partition with p <= " << L << ".\n";
        std::cout << "This is a search-limit result, not a statement about Goldbach:\n";
        std::cout << "raising --p-max continues the search.\n";
    }

    std::cout << "RESULT"
              << " status=" << (r.found ? "found" : "bound")
              << " p=" << r.p
              << " q_digits=" << r.q_digits
              << " candidates=" << r.candidates
              << " prp_calls=" << r.prp_calls
              << " threads=" << omp_get_max_threads()
              << " time_s=" << std::fixed << std::setprecision(3) << secs
              << "\n";

    mpz_clear(n); mpz_clear(n_half);
    return r.found ? 0 : 3;
}

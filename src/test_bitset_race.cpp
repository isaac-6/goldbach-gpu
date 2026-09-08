// Regression test for a data race in build_prime_bitset.
//
// build_prime_bitset splits the output bitset into one slice per OpenMP
// thread. The slices are exclusive in BIT index, but storage is 64-bit words:
// if odds_per_thread is not a multiple of 64, every slice boundary falls
// mid-word and the two adjacent threads both read-modify-write that word
// through the non-atomic PrimeBitset::clear(). Lost clears leave a bit SET,
// so a composite is reported prime -- the direction that makes Goldbach
// verification wrongly succeed.
//
// The test builds the bitset many times and diffs each build against a
// single-threaded reference. Two limits are used:
//
//   MISALIGNED  odds_per_thread % 64 != 0  -- reproduces the race
//   ALIGNED     odds_per_thread % 64 == 0  -- must pass even unfixed
//
// The aligned case is the control: it isolates the fault to boundary
// alignment rather than to the sieve logic itself.
//
// Exit 0 = every build matched, 1 = at least one build was wrong.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include "prime_bitset.hpp"

using namespace goldbach;

// Deliberately self-contained: an independent reference, not shared with any
// code under test.
static std::vector<char> reference_sieve(uint64_t limit) {
    std::vector<char> is_prime(limit + 1, 1);
    is_prime[0] = 0;
    if (limit >= 1) is_prime[1] = 0;
    for (uint64_t i = 2; i * i <= limit; i++)
        if (is_prime[i])
            for (uint64_t j = i * i; j <= limit; j += i)
                is_prime[j] = 0;
    return is_prime;
}

struct Result { uint64_t builds, bad_builds, total_bad_bits; };

static Result run_case(const char* label, uint64_t limit, int builds)
{
    std::vector<char> ref = reference_sieve(limit);

    int      nthreads    = omp_get_max_threads();
    uint64_t total_odds  = (limit - 3) / 2 + 1;
    uint64_t opt         = (total_odds + nthreads - 1) / nthreads;

    printf("  [%s] limit=%llu  threads=%d  total_odds=%llu\n",
           label, (unsigned long long)limit, nthreads,
           (unsigned long long)total_odds);
    printf("        odds_per_thread=%llu  %% 64 = %llu  -> slice boundaries %s\n",
           (unsigned long long)opt, (unsigned long long)(opt % 64),
           (opt % 64) ? "fall MID-WORD" : "are word-aligned");

    Result r{ (uint64_t)builds, 0, 0 };
    uint64_t shown = 0;

    for (int b = 0; b < builds; b++) {
        PrimeBitset bs = build_prime_bitset(limit);
        uint64_t bad = 0;

        for (uint64_t q = 3; q <= limit; q += 2) {
            bool got = bs.is_prime(q);
            bool want = ref[q] != 0;
            if (got == want) continue;

            bad++;
            r.total_bad_bits++;
            if (shown < 6) {
                uint64_t bit = (q - 3) / 2;
                // Distance to the nearest slice boundary, signed-ish.
                uint64_t m = bit % opt;
                uint64_t dist = (m < opt - m) ? m : opt - m;
                printf("        q=%llu bit=%llu  reported %s, actually %s"
                       "  (bit %% odds_per_thread = %llu, %llu from a boundary)\n",
                       (unsigned long long)q, (unsigned long long)bit,
                       got ? "PRIME" : "composite",
                       want ? "prime" : "COMPOSITE",
                       (unsigned long long)m, (unsigned long long)dist);
                shown++;
            }
        }
        if (bad) r.bad_builds++;
    }

    printf("        -> %llu/%llu builds wrong (%llu bad bits total)\n\n",
           (unsigned long long)r.bad_builds, (unsigned long long)r.builds,
           (unsigned long long)r.total_bad_bits);
    return r;
}

int main(int argc, char** argv) {
    int builds = (argc > 1) ? atoi(argv[1]) : 250;

    // total_odds = 499999; ceil/4 = 125000, ceil/8 = 62500, ceil/16 = 31250 --
    // none a multiple of 64, so this is misaligned at 4, 8 and 16 threads.
    const uint64_t MISALIGNED = 1000000;

    // total_odds = 499712 = 1024 * 488; /4, /8 and /16 are all multiples of 64.
    const uint64_t ALIGNED = 999425;

    printf("build_prime_bitset race test (%d builds per case)\n\n", builds);

    Result mis = run_case("misaligned", MISALIGNED, builds);
    Result ali = run_case("aligned (control)", ALIGNED, builds);

    uint64_t bad = mis.bad_builds + ali.bad_builds;
    printf("TOTAL BAD BUILDS: %llu\n", (unsigned long long)bad);
    if (bad) { printf("FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}

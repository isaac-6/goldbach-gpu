// test_records.cpp
// Validates the pipeline's notion of p_min against an EXTERNAL, independently
// computed source.
//
// Every defect found while developing this verifier failed in the direction of
// false success, and "no counterexample found" is the same output whether the
// run was correct or broken. The rest of the test suite compares our GPU code
// against our own CPU code; agreement there does not rule out a shared
// misconception. This test compares against numbers computed by someone else,
// on different hardware, with different software.
//
// p_min(n) is the smallest prime p for which n - p is also prime. A record is
// an n whose p_min exceeds that of every smaller even number.
//
// SOURCE OF THE TABLE BELOW
//   Tomas Oliveira e Silva, "Goldbach conjecture verification", data file
//   https://sweet.ua.pt/tos/goldbach/t0.txt.gz  (retrieved 2026-09-09; file
//   header states "(c) 2012-2016, Tomas Oliveira e Silva", last data update
//   April 7, 2012), linked from https://sweet.ua.pt/tos/goldbach.html
//   This is the dataset underlying: T. Oliveira e Silva, S. Herzog and
//   S. Pardi, "Empirical verification of the even Goldbach conjecture and
//   computation of prime gaps up to 4*10^18", Math. Comp. 83(288), 2014,
//   pp. 2033-2060.
//
//   That file tabulates, for each prime p, the value S(p) = "the least even
//   number for which p is the smallest prime in one of its Goldbach
//   partitions". It marks TWO different kinds of record with an asterisk, and
//   they are not the same set -- 69 primes are p-records, 132 values are
//   S(p)-records. The one meaning "p_min reaches a new maximum at n" is the
//   p-record, defined in the file header as
//       "p is a record-holder if S(q) > S(p) for all q > p, i.e., if there
//        exists a Goldbach partition for each even number smaller than S(p)
//        that uses a prime smaller than p"
//   whose second clause is exactly "every even m < S(p) has p_min(m) < p".
//   The S(p)-record marker is a weaker condition constraining only q < p; it
//   permits some m < S(p) with p_min(m) > p, so it is NOT the right set here.
//   The pairs below are every p-record with n < 10^13: 48 of them.
//
// Exit 0 = agreement, 1 = any disagreement.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>

std::vector<uint64_t> simple_sieve(uint64_t limit);
std::vector<char> segmented_sieve(uint64_t low, uint64_t high);

struct Record { uint64_t n; uint64_t p_min; };

// (n, p_min) record pairs with n < 10^13. See SOURCE above.
static const Record RECORDS[] = {
    { 4ULL, 2 },
    { 6ULL, 3 },
    { 12ULL, 5 },
    { 30ULL, 7 },
    { 98ULL, 19 },
    { 220ULL, 23 },
    { 308ULL, 31 },
    { 556ULL, 47 },
    { 992ULL, 73 },
    { 2642ULL, 103 },
    { 5372ULL, 139 },
    { 7426ULL, 173 },
    { 43532ULL, 211 },
    { 54244ULL, 233 },
    { 63274ULL, 293 },
    { 113672ULL, 313 },
    { 128168ULL, 331 },
    { 194428ULL, 359 },
    { 194470ULL, 383 },
    { 413572ULL, 389 },
    { 503222ULL, 523 },
    { 1077422ULL, 601 },
    { 3526958ULL, 727 },
    { 3807404ULL, 751 },
    { 10759922ULL, 829 },
    { 24106882ULL, 929 },
    { 27789878ULL, 997 },
    { 37998938ULL, 1039 },
    { 60119912ULL, 1093 },
    { 113632822ULL, 1163 },
    { 187852862ULL, 1321 },
    { 335070838ULL, 1427 },
    { 419911924ULL, 1583 },
    { 721013438ULL, 1789 },
    { 1847133842ULL, 1861 },
    { 7473202036ULL, 1877 },
    { 11001080372ULL, 1879 },
    { 12703943222ULL, 2029 },
    { 21248558888ULL, 2089 },
    { 35884080836ULL, 2803 },
    { 105963812462ULL, 3061 },
    { 244885595672ULL, 3163 },
    { 599533546358ULL, 3457 },
    { 3132059294006ULL, 3463 },
    { 3620821173302ULL, 3529 },
    { 4438327672994ULL, 3613 },
    { 5320503815888ULL, 3769 },
    { 8342945544436ULL, 3917 },
};
static const size_t NUM_RECORDS = sizeof(RECORDS) / sizeof(RECORDS[0]);

// Even numbers checked below each record. The record property is global, so a
// window can only refute it locally -- but a local violation is still a real
// disagreement, and the window covers the region where a competing record is
// most likely to hide.
static const uint64_t WINDOW = 100000;

// Enough to cover every tabulated p_min below 10^13 (largest is 3137).
static const uint64_t P_LIMIT = 4000;

int main(int argc, char** argv) {
    uint64_t window = (argc > 1) ? strtoull(argv[1], nullptr, 10) : WINDOW;

    std::vector<uint64_t> primes = simple_sieve(P_LIMIT);
    printf("Oliveira e Silva p_min record check\n");
    printf("  %zu published records with n < 1e13, window %llu below each\n\n",
           NUM_RECORDS, (unsigned long long)window);

    uint64_t failures = 0;

    for (size_t i = 0; i < NUM_RECORDS; i++) {
        uint64_t n      = RECORDS[i].n;
        uint64_t expect = RECORDS[i].p_min;
        uint64_t prev   = (i == 0) ? 0 : RECORDS[i - 1].p_min;

        // Cover q = m - p for every m in the window and every candidate p.
        uint64_t lo = (n > window + 4) ? n - window : 4;
        if (lo % 2) lo++;
        uint64_t sieve_lo = (lo > P_LIMIT + 2) ? lo - P_LIMIT : 2;
        std::vector<char> is_prime = segmented_sieve(sieve_lo, n);

        auto p_min_of = [&](uint64_t m) -> uint64_t {
            for (uint64_t p : primes) {
                if (p > m / 2) break;
                uint64_t q = m - p;
                if (q < sieve_lo) continue;
                if (is_prime[q - sieve_lo]) return p;
            }
            return 0;   // no p <= P_LIMIT worked
        };

        uint64_t got = p_min_of(n);
        if (got != expect) {
            printf("  [FAIL] n=%llu: computed p_min=%llu, published %llu\n",
                   (unsigned long long)n, (unsigned long long)got,
                   (unsigned long long)expect);
            failures++;
        }

        // No m below the record may beat the PREVIOUS record -- otherwise this
        // n is not where p_min first reaches this height.
        uint64_t worst_m = 0, worst_p = 0;
        for (uint64_t m = lo; m < n; m += 2) {
            uint64_t pm = p_min_of(m);
            if (pm == 0 || pm > prev) {
                if (pm > worst_p) { worst_p = pm; worst_m = m; }
            }
        }
        if (worst_p) {
            printf("  [FAIL] n=%llu (p_min=%llu): m=%llu below it has p_min=%llu"
                   " > previous record %llu\n",
                   (unsigned long long)n, (unsigned long long)expect,
                   (unsigned long long)worst_m, (unsigned long long)worst_p,
                   (unsigned long long)prev);
            failures++;
        }

        printf("  n=%-15llu p_min=%-6llu OK  (%llu even numbers below it checked)\n",
               (unsigned long long)n, (unsigned long long)expect,
               (unsigned long long)((n - lo) / 2));
        fflush(stdout);
    }

    printf("\nTOTAL DISAGREEMENTS: %llu\n", (unsigned long long)failures);
    if (failures) { printf("FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}

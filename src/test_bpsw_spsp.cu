// Negative and positive controls for the primality tests Phase 2 uses:
// Baillie-PSW and 12-base Miller-Rabin, on device and host.
//
// Negative controls, every one of which must be rejected by all four tests:
//   - every base-2 strong pseudoprime below 2^32 (2,314). A strong base-2 test
//     alone accepts them; BPSW's strong Lucas step and MR's other bases are
//     what reject them. test_primality samples random inputs, which almost
//     never include one, so a Lucas step that always passed went unnoticed
//     there.
//   - the first ten strong Lucas pseudoprimes for Method A* parameters, as
//     published in Baillie, Fiori and Wagstaff, "Strengthening the
//     Baillie-PSW primality test" (arXiv:2006.14425), Section 2.4. The strong
//     Lucas step alone accepts them; BPSW's base-2 step is what rejects them,
//     so a BPSW that skipped that step passes the first set but not this one.
//
// The pseudoprimes below 2^32 are generated here, independently of
// primality.cuh: an odd-only segmented sieve decides compositeness, and a
// 32-bit Montgomery strong base-2 test decides pseudoprimality. Neither
// shares code with the implementations under test.
//
// Guards against a vacuous pass:
//   - the generator must find exactly EXPECTED_SPSP, and the known first five
//     (2047 = 23*89, 3277 = 29*113, 4033 = 37*109, 4681 = 31*151, 8321 =
//     53*157);
//   - all four tests must ACCEPT every prime below 2^20 and every prime in the
//     last 10^6 below 2^32, so a test that rejected everything would also fail.
//
// Exit 0 = all composites rejected and all primes accepted, 1 = otherwise.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include "primality.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); exit(2); } } while (0)

// Found by this generator and, separately, by the project's own strong
// base-2 test filtered through 12-base Miller-Rabin on the GPU.
static const uint64_t EXPECTED_SPSP = 2314;
static const uint64_t LIMIT = 1ULL << 32;

// -------------------------------------------------------
// Independent generator
// -------------------------------------------------------
// Odd-only sieve of [3, 2^32): bit i <-> 2i + 3, set = composite.
static std::vector<uint64_t> composite_bits() {
    const uint64_t odds = (LIMIT - 3) / 2 + 1;
    std::vector<uint64_t> bits((odds + 63) / 64, 0);

    std::vector<uint32_t> base;                 // odd primes below 2^16
    std::vector<char> small(1 << 16, 1);
    for (uint32_t i = 3; i < (1u << 16); i += 2) {
        if (!small[i]) continue;
        base.push_back(i);
        for (uint32_t j = i * i; j < (1u << 16); j += 2 * i) small[j] = 0;
    }

    const uint64_t SEG = 1ULL << 21;            // odd indices per segment, a multiple of 64
    #pragma omp parallel for schedule(dynamic)
    for (long long s = 0; s < (long long)((odds + SEG - 1) / SEG); s++) {
        uint64_t lo = (uint64_t)s * SEG, hi = std::min(lo + SEG, odds);   // bit range [lo, hi)
        uint64_t n_lo = 2 * lo + 3;
        for (uint32_t p : base) {
            uint64_t pp = (uint64_t)p * p;
            uint64_t m = std::max(pp, (n_lo + p - 1) / p * p);
            if (!(m & 1)) m += p;
            for (uint64_t b = (m - 3) / 2; b < hi; b += p) bits[b >> 6] |= 1ULL << (b & 63);
        }
    }
    return bits;
}

static inline bool is_composite(const std::vector<uint64_t>& bits, uint64_t n) {
    uint64_t b = (n - 3) / 2;
    return (bits[b >> 6] >> (b & 63)) & 1;
}

// Montgomery arithmetic mod an odd n < 2^32, R = 2^32.
struct Mont {
    uint32_t n, ninv;   // ninv = -n^-1 mod 2^32
    uint64_t r2;        // R^2 mod n
    explicit Mont(uint32_t n_) : n(n_) {
        uint32_t inv = n;                                 // Newton: 5 steps give 32 bits
        for (int i = 0; i < 5; i++) inv *= 2 - n * inv;
        ninv = (uint32_t)(0u - inv);
        uint64_t r = (uint64_t)((((unsigned __int128)1) << 64) % n);
        r2 = r;
    }
    // REDC of t < n * 2^32: t * R^-1 mod n.
    uint32_t redc(uint64_t t) const {
        uint32_t m  = (uint32_t)t * ninv;
        uint64_t mn = (uint64_t)m * n;
        // (t + mn) is divisible by 2^32; add the high halves and the carry of
        // the low halves, which is 1 unless the low half of t is 0.
        uint64_t u = (t >> 32) + (mn >> 32) + ((uint32_t)t != 0);
        return (uint32_t)(u >= n ? u - n : u);
    }
    uint32_t to(uint32_t a)   const { return redc((uint64_t)a * (uint32_t)r2); }
    uint32_t mul(uint32_t a, uint32_t b) const { return redc((uint64_t)a * b); }
};

// Strong probable prime to base 2, for odd n > 2.
static bool strong_base2(uint32_t n) {
    Mont M(n);
    uint32_t d = n - 1; int s = 0;
    while (!(d & 1)) { d >>= 1; s++; }
    uint32_t one = M.to(1), minus_one = M.to(n - 1);
    uint32_t x = one, base = M.to(2);
    for (uint32_t e = d; e; e >>= 1) {
        if (e & 1) x = M.mul(x, base);
        base = M.mul(base, base);
    }
    if (x == one || x == minus_one) return true;
    for (int i = 1; i < s; i++) {
        x = M.mul(x, x);
        if (x == minus_one) return true;
    }
    return false;
}

// -------------------------------------------------------
// Production code under test
// -------------------------------------------------------
// out[i] bit 0 = device BPSW verdict, bit 1 = device Miller-Rabin verdict.
__global__ void bpsw_kernel(const uint64_t* n, uint64_t count, unsigned char* out) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = (gpu_is_prime_bpsw(n[i]) ? 1 : 0)
                          | (gpu_is_prime_miller_rabin(n[i]) ? 2 : 0);
}

static std::vector<unsigned char> device_tests(const std::vector<uint64_t>& v) {
    std::vector<unsigned char> out(v.size());
    if (v.empty()) return out;
    uint64_t* d_n; unsigned char* d_out;
    CK(cudaMalloc(&d_n, v.size() * sizeof(uint64_t)));
    CK(cudaMalloc(&d_out, v.size()));
    CK(cudaMemcpy(d_n, v.data(), v.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    bpsw_kernel<<<(uint32_t)((v.size() + 255) / 256), 256>>>(d_n, v.size(), d_out);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out.data(), d_out, v.size(), cudaMemcpyDeviceToHost));
    CK(cudaFree(d_n)); CK(cudaFree(d_out));
    return out;
}

// Runs device and host BPSW and Miller-Rabin over v and counts, per test, the
// values whose verdict differs from expect_prime. Returns the total.
static uint64_t check_set(const char* label, const std::vector<uint64_t>& v, bool expect_prime) {
    std::vector<unsigned char> dev = device_tests(v);
    uint64_t wrong[4] = {0, 0, 0, 0}, shown = 0;   // device BPSW, device MR, host BPSW, host MR
    for (size_t i = 0; i < v.size(); i++) {
        bool got[4] = {(dev[i] & 1) != 0, (dev[i] & 2) != 0,
                       cpu_is_prime_bpsw(v[i]), cpu_miller_rabin(v[i])};
        bool any = false;
        for (int t = 0; t < 4; t++) if (got[t] != expect_prime) { wrong[t]++; any = true; }
        if (any && shown++ < 5)
            printf("  [FAIL] %llu should be %s: device BPSW %d, device MR %d, host BPSW %d, host MR %d\n",
                   (unsigned long long)v[i], expect_prime ? "prime" : "composite",
                   got[0], got[1], got[2], got[3]);
    }
    printf("  %s: %zu checked, %s: device BPSW %llu, device MR %llu, host BPSW %llu, host MR %llu (must be 0)\n",
           label, v.size(), expect_prime ? "rejected" : "accepted",
           (unsigned long long)wrong[0], (unsigned long long)wrong[1],
           (unsigned long long)wrong[2], (unsigned long long)wrong[3]);
    return wrong[0] + wrong[1] + wrong[2] + wrong[3];
}

int main() {
    uint64_t failures = 0;

    printf("Sieving odd numbers below 2^32 ...\n"); fflush(stdout);
    std::vector<uint64_t> comp = composite_bits();

    // Self-check of the Montgomery test before trusting it.
    {
        const uint32_t known_spsp[] = {2047, 3277, 4033, 4681, 8321};
        const uint32_t known_non[]  = {2049, 3279, 4035, 9, 15};
        for (uint32_t n : known_spsp) if (!strong_base2(n)) { printf("  [FAIL] generator rejects spsp %u\n", n); failures++; }
        for (uint32_t n : known_non)  if (strong_base2(n))  { printf("  [FAIL] generator accepts %u\n", n); failures++; }
        for (uint32_t p : {3u, 5u, 65537u, 4294967291u}) if (!strong_base2(p)) { printf("  [FAIL] generator rejects prime %u\n", p); failures++; }
    }

    printf("Strong base-2 test over every odd composite below 2^32 ...\n"); fflush(stdout);
    std::vector<std::vector<uint64_t>> per_thread(omp_get_max_threads());
    #pragma omp parallel for schedule(dynamic, 1 << 16)
    for (long long b = 0; b < (long long)((LIMIT - 3) / 2 + 1); b++) {
        uint64_t n = 2 * (uint64_t)b + 3;
        if (is_composite(comp, n) && strong_base2((uint32_t)n))
            per_thread[omp_get_thread_num()].push_back(n);
    }
    std::vector<uint64_t> spsp;
    for (auto& v : per_thread) spsp.insert(spsp.end(), v.begin(), v.end());
    std::sort(spsp.begin(), spsp.end());

    printf("  found %zu base-2 strong pseudoprimes (expected %llu); first: %llu %llu %llu %llu %llu\n",
           spsp.size(), (unsigned long long)EXPECTED_SPSP,
           spsp.size() > 0 ? (unsigned long long)spsp[0] : 0ULL,
           spsp.size() > 1 ? (unsigned long long)spsp[1] : 0ULL,
           spsp.size() > 2 ? (unsigned long long)spsp[2] : 0ULL,
           spsp.size() > 3 ? (unsigned long long)spsp[3] : 0ULL,
           spsp.size() > 4 ? (unsigned long long)spsp[4] : 0ULL);
    const uint64_t first5[] = {2047, 3277, 4033, 4681, 8321};
    if (spsp.size() != EXPECTED_SPSP || !std::equal(first5, first5 + 5, spsp.begin())) {
        printf("  [FAIL] generator did not reproduce the expected pseudoprime list\n");
        failures++;
    }

    // Negative controls: all must be rejected.
    failures += check_set("base-2 strong pseudoprimes", spsp, false);
    const std::vector<uint64_t> slpsp = {5459, 5777, 10877, 16109, 18971,
                                         22499, 24569, 25199, 40309, 58519};
    failures += check_set("strong Lucas pseudoprimes (A*)", slpsp, false);

    // Positive control: primes must be accepted.
    std::vector<uint64_t> primes = {2, 3};
    for (uint64_t n = 5; n < (1ULL << 20); n += 2) if (!is_composite(comp, n)) primes.push_back(n);
    for (uint64_t n = LIMIT - 1000001; n < LIMIT; n += 2) if (!is_composite(comp, n)) primes.push_back(n);
    failures += check_set("primes", primes, true);

    printf("\nTOTAL FAILURES: %llu\n", (unsigned long long)failures);
    if (failures) { printf("FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}

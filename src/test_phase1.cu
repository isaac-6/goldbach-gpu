// Differential test: GPU Phase 1 Goldbach verification vs. a CPU reference.
//
// The GPU side runs the production kernels from sieve_kernel.cuh and
// phase1_kernel.cuh, wired up as goldbach.cu wires them: the range is walked
// in segments using the verifier's own segment_geometry(), each segment is
// sieved, its padding word zeroed, and the prime batches swept over its even
// numbers.
//
// The CPU side redoes the search independently -- segmented_sieve over
// [n_low - p_small, n_high], then for each even n a straight ascending scan
// for the first p with n - p prime. It knows nothing about segments.
//
// Coverage beyond the single-segment ranges:
//   - ranges that span many segments, including a partial last segment and
//     one where consecutive segments switch from the scalar to the transposed
//     kernel;
//   - batch sizes 1, 7 and 1000, so the verified state carries across many
//     Phase 1 launches per segment.
//
// Exit 0 = agreement, 1 = mismatch.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include "sieve_kernel.cuh"
#include "phase1_kernel.cuh"
#include "segment_geometry.hpp"
#include "prime_bitset.hpp"

using namespace goldbach;

std::vector<uint64_t> simple_sieve(uint64_t limit);
std::vector<char> segmented_sieve(uint64_t low, uint64_t high);

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); exit(2); } } while (0)

static const int THREADS_PER_BLOCK = 256;
static const uint64_t P_BATCH = 2000000;

// Launches queued between host synchronizations, at most. compute-sanitizer
// racecheck keeps ~2.2 KB per thread of every launch queued since the last
// synchronize and frees it only then. At batch sizes 1 and 7 this loop queued
// up to 78,498 and 11,214 launches of ~1,000-5,000 threads per prime prefix:
// 15 GiB for the "switch" range at batch 1, over 24 GiB at batch 7 on the 1e9
// range, and an out-of-memory kill of the machine's whole allowance at batch 1
// there (measured; ablation-logs/reruns/v320/memprobe). A synchronize every
// 64 launches bounds that to about 64 * 5,120 * 2.2 KB ~= 0.7 GiB and changes
// nothing the test checks: launches on one stream run in order either way.
static const uint64_t SYNC_EVERY_LAUNCHES = 64;

static uint64_t isqrt64(uint64_t n) {
    if (n < 2) return n;
    uint64_t r = 1;
    while (r <= n / r) r++;
    return r - 1;
}

// -------------------------------------------------------
// CPU reference: index of the first p (ascending) with n - p prime,
// or NO_P if the whole list is exhausted. Mirrors what the kernel searches
// for, by a wholly separate route.
//
// Returning the index rather than a bare verdict is what gives this test its
// resolution. Every even n in these ranges has many Goldbach partitions, so
// "verified" alone is saturated -- it stays true even if the kernel's
// primality lookups are wrong, because some later p still succeeds. The index
// pins down *which* p resolved n, so a wrong lookup has nowhere to hide.
// -------------------------------------------------------
static const uint32_t NO_P = UINT32_MAX;

static std::vector<uint32_t> cpu_phase1(uint64_t n_low, uint64_t n_high,
                                        uint64_t even_count,
                                        const std::vector<uint64_t>& gpu_primes,
                                        uint64_t p_small)
{
    // Own lower bound, chosen to cover every reachable q = n - p rather than
    // copied from the kernel's q_low -- the smallest q arises at n = n_low with
    // the largest usable p. Note this reaches below the kernel's q_low, which is
    // odd-adjusted and clamped to 3: at n = 4 the only partition is 2 + 2.
    uint64_t ref_low = (n_low > p_small + 2) ? n_low - p_small : 2;

    std::vector<char> is_prime = segmented_sieve(ref_low, n_high);
    std::vector<uint32_t> pmin(even_count, NO_P);

    for (uint64_t i = 0; i < even_count; i++) {
        uint64_t n = n_low + 2 * i;
        for (uint32_t j = 0; j < gpu_primes.size(); j++) {
            uint64_t p = gpu_primes[j];
            if (p > n / 2) break;
            uint64_t q = n - p;
            if (is_prime[q - ref_low]) { pmin[i] = j; break; }
        }
    }
    return pmin;
}

// Checks every even n in [n_low, n_high], walked in segments of seg_size even
// numbers as goldbach.cu walks them from START = n_low to LIMIT = n_high, with
// batch primes per Phase 1 launch. Returns the number of disagreements and
// prints the first few.
static uint64_t check_range(uint64_t n_low, uint64_t n_high, uint64_t p_small,
                            uint64_t seg_size, uint64_t batch, bool verbose)
{
    if (n_low % 2) n_low++;
    if (n_high % 2) n_high--;
    if (n_low < 4) n_low = 4;
    if (n_high < n_low) return 0;

    uint64_t even_count = (n_high - n_low) / 2 + 1;

    uint64_t small_high = std::max(isqrt64(n_high) + 1, p_small);
    if (small_high % 2 == 0) small_high++;

    // Host prime tables.
    PrimeBitset small_bitset = build_prime_bitset(small_high);
    std::vector<uint64_t> small_primes;
    if (small_bitset.is_prime(2)) small_primes.push_back(2);
    for (uint64_t i = 3; i <= small_high; i += 2)
        if (small_bitset.is_prime(i)) small_primes.push_back(i);

    std::vector<uint64_t> gpu_primes;
    for (uint64_t p : small_primes)
        if (p <= p_small) gpu_primes.push_back(p);

    // -------- CPU side --------
    auto cpu_pmin = cpu_phase1(n_low, n_high, even_count, gpu_primes, p_small);

    // -------- GPU side --------
    // Buffers sized for the widest segment, as goldbach.cu sizes them.
    uint64_t max_odds = 0;
    for (uint64_t s = n_low; s <= n_high; s += 2 * seg_size)
        max_odds = std::max(max_odds, segment_geometry(s, seg_size, n_high, p_small).num_odds);
    uint64_t seg_words      = (max_odds + 63) / 64;
    uint64_t max_ver_words  = (std::min(seg_size, even_count) + 63) / 64;
    uint64_t p_batch_alloc  = std::max<uint64_t>(1, std::min(batch, (uint64_t)gpu_primes.size()));
    size_t   small_bytes    = small_bitset.word_count() * sizeof(uint64_t);

    uint64_t *d_small = nullptr, *d_small_primes = nullptr;
    uint64_t *d_seg_bits = nullptr, *d_p_batch = nullptr;
    uint64_t *d_verified = nullptr;   // bitset: 1 bit per even number

    CK(cudaMalloc(&d_small, small_bytes));
    CK(cudaMalloc(&d_small_primes, small_primes.size() * sizeof(uint64_t)));
    // +1 padding word for the transposed kernel's one-word overread.
    CK(cudaMalloc(&d_seg_bits, (seg_words + 1) * sizeof(uint64_t)));
    CK(cudaMalloc(&d_p_batch, p_batch_alloc * sizeof(uint64_t)));
    CK(cudaMalloc(&d_verified, max_ver_words * sizeof(uint64_t)));

    CK(cudaMemcpy(d_small, small_bitset.data(), small_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_small_primes, small_primes.data(),
                  small_primes.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    uint64_t sc = sieve_split_prime_count(small_primes.data(), small_primes.size());

    // Sweep prime-list prefixes. Handing the production kernel only the first
    // K primes makes its verdict mean "p_min_idx < K", so comparing verdicts
    // across several K probes the index without modifying the kernel. The full
    // list is included last, which is the plain end-to-end case.
    uint64_t prefixes[] = {1, 2, 3, 5, 10, 25, 100, 1000, (uint64_t)gpu_primes.size()};

    uint64_t mismatches = 0, shown = 0;
    std::vector<uint64_t> gpu_verified(max_ver_words);

    for (uint64_t seg_start = n_low; seg_start <= n_high; seg_start += 2 * seg_size) {
        const SegmentGeometry geo = segment_geometry(seg_start, seg_size, n_high, p_small);
        uint64_t ver_words = (geo.seg_even_count + 63) / 64;
        uint64_t first_i   = (seg_start - n_low) / 2;   // index into cpu_pmin

        // Sieve the segment, then zero the word past its bits: the same stale
        // padding goldbach.cu clears before the transposed kernel reads it.
        CK(cudaMemset(d_seg_bits, 0xFF, (seg_words + 1) * sizeof(uint64_t)));  // stale junk
        launch_segment_sieve(geo.q_low, geo.q_high, d_small_primes,
                             sc, small_primes.size() - sc, d_seg_bits,
                             THREADS_PER_BLOCK, 0);
        CK(cudaGetLastError());
        CK(cudaMemset(d_seg_bits + (geo.num_odds + 63) / 64, 0, sizeof(uint64_t)));

        for (uint64_t K : prefixes) {
            if (K > gpu_primes.size()) continue;

            CK(cudaMemset(d_verified, 0, ver_words * sizeof(uint64_t)));
            uint64_t queued = 0;   // launches since the last synchronize
            for (uint64_t bi = 0; bi < K; bi += batch) {
                uint64_t bsize = std::min(batch, K - bi);
                CK(cudaMemcpy(d_p_batch, gpu_primes.data() + bi,
                              bsize * sizeof(uint64_t), cudaMemcpyHostToDevice));

                launch_goldbach_phase1(
                    d_small, small_high, d_seg_bits, geo.q_low, geo.q_high,
                    seg_start, geo.seg_even_count, d_p_batch, bsize, d_verified,
                    p_small, THREADS_PER_BLOCK, 0);
                CK(cudaGetLastError());
                if (++queued == SYNC_EVERY_LAUNCHES) {   // see SYNC_EVERY_LAUNCHES
                    CK(cudaStreamSynchronize(0));
                    queued = 0;
                }
            }
            CK(cudaDeviceSynchronize());
            unsigned int q_range_error = 0;
            CK(cudaMemcpyFromSymbol(&q_range_error, g_phase1_q_range_error, sizeof(unsigned int)));
            if (q_range_error) {
                // A q outside both bitsets: the segment geometry is wrong.
                printf("    q outside both bitsets (segment at %llu, first %llu primes)\n",
                       (unsigned long long)seg_start, (unsigned long long)K);
                mismatches++;
                unsigned int zero = 0;
                CK(cudaMemcpyToSymbol(g_phase1_q_range_error, &zero, sizeof(unsigned int)));
            }
            CK(cudaMemcpy(gpu_verified.data(), d_verified,
                          ver_words * sizeof(uint64_t), cudaMemcpyDeviceToHost));

            for (uint64_t i = 0; i < geo.seg_even_count; i++) {
                bool g = ((gpu_verified[i >> 6] >> (i & 63)) & 1ULL) != 0;
                uint32_t ref = cpu_pmin[first_i + i];
                bool c = ref != NO_P && ref < K;
                if (g != c) {
                    mismatches++;
                    if (verbose && shown < 5) {
                        printf("    n=%llu  gpu=%s cpu=%s  (segment at %llu, first %llu primes, batch %llu",
                               (unsigned long long)(seg_start + 2 * i),
                               g ? "verified" : "unverified",
                               c ? "verified" : "unverified",
                               (unsigned long long)seg_start,
                               (unsigned long long)K, (unsigned long long)batch);
                        if (ref == NO_P) printf(", cpu found no p)\n");
                        else printf(", cpu p_min=%llu at index %u)\n",
                                    (unsigned long long)gpu_primes[ref], ref);
                        shown++;
                    }
                }
            }
        }
    }

    CK(cudaFree(d_small));
    CK(cudaFree(d_small_primes));
    CK(cudaFree(d_seg_bits));
    CK(cudaFree(d_p_batch));
    CK(cudaFree(d_verified));
    return mismatches;
}

// One segment covering the whole range: the even count rounded up to even,
// as SEG_SIZE must be.
static uint64_t one_segment(uint64_t lo, uint64_t hi) {
    uint64_t evens = (hi - lo) / 2 + 2;
    return evens + (evens & 1);
}

static uint64_t segments_in(uint64_t lo, uint64_t hi, uint64_t seg_size) {
    if (lo % 2) lo++;
    if (lo < 4) lo = 4;
    return ((hi - lo) / 2 + 1 + seg_size - 1) / seg_size;
}

int main() {
    uint64_t total = 0;
    const uint64_t SPAN = 2 * 200000;   // ~200k even numbers
    const uint64_t P_SMALL = 1000000;

    struct { uint64_t lo, hi; const char* name; } fixed[] = {
        {4,                4 + SPAN,                "from 4 (small-n edge)"},
        {1000000000ULL,    1000000000ULL + SPAN,    "1e9"},
        {100000000000ULL,  100000000000ULL + SPAN,  "1e11"},
    };

    for (auto& t : fixed) {
        printf("  [%s] [%llu, %llu]\n", t.name,
               (unsigned long long)t.lo, (unsigned long long)t.hi);
        fflush(stdout);
        uint64_t m = check_range(t.lo, t.hi, P_SMALL, one_segment(t.lo, t.hi), P_BATCH, true);
        printf("    -> %llu mismatches\n", (unsigned long long)m);
        total += m;
    }

    // -------------------------------------------------------
    // Scalar-kernel coverage of the segment-bitset lookup.
    // -------------------------------------------------------
    // launch_goldbach_phase1 routes a segment to the scalar kernel only when
    // seg_even_start <= 2*p_small + 128, so with P_SMALL = 1e6 the only range
    // above that takes the scalar path is [4, 4+SPAN]. There every complement
    // q = n - p stays below small_high = max(isqrt(n_high)+1, p_small) = 1000001,
    // so is_prime_q always answers from d_small and its segment-bitset branch
    // is never entered -- a corruption of that branch goes undetected.
    //
    // Lowering p_small to 1e4 keeps the range on the scalar path
    // (4 <= 2*10^4 + 128) while dropping small_high to 10001, so the complements
    // up to ~400001 now exceed it and resolve against the segment bitset.
    const uint64_t P_SMALL_SCALAR = 10000;
    {
        uint64_t lo = 4, hi = 4 + SPAN;
        printf("  [from 4, p_small=1e4 (scalar segment-bitset path)] [%llu, %llu]\n",
               (unsigned long long)lo, (unsigned long long)hi);
        fflush(stdout);
        uint64_t m = check_range(lo, hi, P_SMALL_SCALAR, one_segment(lo, hi), P_BATCH, true);
        printf("    -> %llu mismatches\n", (unsigned long long)m);
        total += m;
    }

    // -------------------------------------------------------
    // Many segments, and many batches per segment.
    // -------------------------------------------------------
    // "switch": p_small = 1e4 and 5000-number segments from 4. Segments
    //   starting at 4, 10004 and 20004 are <= 2*p_small + 128 and take the
    //   scalar kernel; from 30004 on they take the transposed kernel, so the
    //   routing changes between consecutive segments. The last is partial.
    //   Every q still lies in the small bitset or the segment bitset, as it
    //   must (see is_prime_q); the out-of-range flag is checked after every
    //   segment.
    // "1e9 multi": 50002-number segments (not a multiple of 64) over ~400k
    //   even numbers at 1e9, all transposed, last one partial.
    // Each runs with batch sizes 1, 7 and 1000 as well as the default, so the
    // verified bits carry across up to one launch per prime.
    struct { uint64_t lo, hi, p_small, seg_size; const char* name; } multi[] = {
        {4,             4 + 2 * 60001,       P_SMALL_SCALAR, 5000,  "switch scalar->transposed"},
        {1000000000ULL, 1000000000ULL + 2 * SPAN, P_SMALL,   50002, "1e9 multi-segment"},
    };
    const uint64_t batches[] = {P_BATCH, 1000, 7, 1};

    for (auto& t : multi) {
        for (uint64_t b : batches) {
            // Batch 1 with the full 1e6 list is ~80k launches per segment;
            // two segments of it are enough to show the carry-over.
            uint64_t hi = t.hi;
            if (b == 1 && t.p_small == P_SMALL) hi = t.lo + 4 * t.seg_size - 2 + 1000;
            printf("  [%s, batch %llu] [%llu, %llu], %llu segments of %llu\n", t.name,
                   (unsigned long long)b, (unsigned long long)t.lo, (unsigned long long)hi,
                   (unsigned long long)segments_in(t.lo, hi, t.seg_size),
                   (unsigned long long)t.seg_size);
            fflush(stdout);
            uint64_t m = check_range(t.lo, hi, t.p_small, t.seg_size, b, true);
            printf("    -> %llu mismatches\n", (unsigned long long)m);
            total += m;
        }
    }

    // -------------------------------------------------------
    // Negative control for the out-of-range flag.
    // -------------------------------------------------------
    // The scalar kernel called directly with a segment range that starts above
    // every q it will meet (n in [1000, 1018], p = 3, q_low = 1000001): no
    // bitset covers q, so is_prime_q must raise g_phase1_q_range_error and
    // must not verify anything. The verifier's own geometry never does this.
    {
        printf("  [negative control: q outside both bitsets]\n");
        uint64_t *d_small = nullptr, *d_seg = nullptr, *d_p = nullptr, *d_ver = nullptr;
        CK(cudaMalloc(&d_small, 64)); CK(cudaMalloc(&d_seg, 64));
        CK(cudaMalloc(&d_p, 8));      CK(cudaMalloc(&d_ver, 8));
        CK(cudaMemset(d_small, 0xFF, 64)); CK(cudaMemset(d_seg, 0xFF, 64)); CK(cudaMemset(d_ver, 0, 8));
        uint64_t p3 = 3;
        CK(cudaMemcpy(d_p, &p3, 8, cudaMemcpyHostToDevice));
        launch_goldbach_phase1(d_small, 5, d_seg, 1000001, 1000101, 1000, 10, d_p, 1, d_ver,
                               1000, THREADS_PER_BLOCK, 0);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        unsigned int flag = 0; uint64_t ver = 0;
        CK(cudaMemcpyFromSymbol(&flag, g_phase1_q_range_error, sizeof(unsigned int)));
        CK(cudaMemcpy(&ver, d_ver, 8, cudaMemcpyDeviceToHost));
        bool ok = flag == 1 && ver == 0;
        printf("    flag=%u verified bits=%llu -> %s\n", flag, (unsigned long long)ver,
               ok ? "ok" : "FAIL (flag must be 1, nothing verified)");
        if (!ok) total++;
        unsigned int zero = 0;
        CK(cudaMemcpyToSymbol(g_phase1_q_range_error, &zero, sizeof(unsigned int)));
        CK(cudaFree(d_small)); CK(cudaFree(d_seg)); CK(cudaFree(d_p)); CK(cudaFree(d_ver));
    }

    std::mt19937_64 rng(20260908);
    printf("  [randomized] 20 ranges\n");
    fflush(stdout);
    for (int i = 0; i < 20; i++) {
        uint64_t lo = 4 + (rng() % 100000000000ULL);
        uint64_t hi = lo + SPAN;
        total += check_range(lo, hi, P_SMALL, one_segment(lo, hi), P_BATCH, false);
    }

    printf("\nTOTAL MISMATCHES: %llu\n", (unsigned long long)total);
    if (total) { printf("FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}

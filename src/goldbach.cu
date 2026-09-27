// goldbach.cu
// v2.0.0 -- 2026-03-05
//
// GPU Goldbach range verifier
// This function gets updated with the best version of the GPU code.
//
// Algorithm:
//
//   Build small primes up to small_high (>= max(P_SMALL, sqrt(LIMIT))).
//
//   For each segment [A, B] of even numbers:
//     1) GPU sieve odd q in [q_low, q_high], where:
//          q_low  = max(3, A - P_SMALL), odd
//          q_high = B + 1, odd
//     2) Phase 1 (GPU): for each prime p in [2, P_SMALL],
//          mark all even n in [A, B] as verified if n-p is prime.
//          q = n-p checked via:
//            - small bitset (q <= small_high)
//            - segment bitset (q_low <= q <= q_high)
//            - Miller-Rabin otherwise
//     3) Phase 2 (CPU fallback): any n still unverified after Phase 1
//          is checked using optimized sieve (up to 10^8) + Miller-Rabin.
//
// Correctness guarantee:
//   Every even n in [4, LIMIT] is verified by Phase 1 or Phase 2.
//
// CRITICAL LIMITS:
//   - P_SMALL must be <= 4,000,000,000 (~4 billion) to prevent p*p overflow
//   - LIMIT is theoretically up to 2^64-1, but practical limits:
//     * GPU VRAM constrains SEG_SIZE 
//     * Integer sqrt computed exactly using binary search (no double loss)
//     * Phase 2 now uses sieve + Miller-Rabin
// 
// MULTI-GPU ARCHITECTURE:
//   - Lock-free work queue for dynamic load balancing
//   - Each GPU processes independent segments
//   - Thread-safe logging and failure detection
//   - Exception-safe resource cleanup
//
// PERFORMANCE CHARACTERISTICS:
//   - Phase 1: 10^12 in 36.5 seconds on RTX 5090. With 2x 5090, it took 19 seconds.
//   - Phase 2: never reached on tested inputs due to effective Phase 1 filtering
//   - Memory: ~200 MB with  --seg-size=200000000 --p-small=1000000 --batch-size=2000000
//
// RANGE:
//   This implementation is mathematically sound for
//   verification from 4 to 1.8 * 10^19 (limited by time).

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <fstream>
#include <cctype>
#include "prime_bitset.hpp"
#include "sieve_kernel.cuh"
#include "phase1_kernel.cuh"
#include "primality.cuh"

using namespace goldbach;

inline std::chrono::high_resolution_clock::time_point now() {
    return std::chrono::high_resolution_clock::now();
}

// ------------------------------------------------------------
// Thread-Safe Global State & Logging
// ------------------------------------------------------------
static std::atomic<bool>     g_failure{false};
static std::atomic<bool>     g_system_error{false}; // For CUDA/Runtime errors
static std::atomic<uint64_t> g_failure_n{0};
static std::atomic<uint64_t> g_next_segment_start{0};
static std::atomic<uint64_t> g_total_phase2_count{0};
static std::atomic<uint64_t> g_total_processed{0};

static std::mutex g_log_mutex;

// --record-check: running global maximum of p_min and the n attaining it.
// Guarded by its own mutex because segments may finish out of order when more
// than one GPU is in use. Under multi-GPU the printed sequence is not merely
// reordered: a worker that finishes a later segment first raises the running
// maximum, permanently suppressing genuine records from earlier segments that
// have not been processed yet. Which records survive is therefore scheduling-
// dependent and not reproducible across runs. Use a single GPU when the
// record sequence is being used for validation.
static std::mutex    g_record_mutex;
static uint64_t      g_record_p = 0;

// --count-primes: running total of primes counted over all segments, and the
// per-segment counts for --count-file. Segments may complete out of order under
// multi-GPU; the counts are sorted by range before they are written, and the
// total is order-independent.
struct SegmentCount { uint64_t lo, hi, count; };
static std::atomic<uint64_t>     g_prime_count_total{0};
static std::mutex                g_count_mutex;
static std::vector<SegmentCount> g_segment_counts;

template<typename... Args>
void safe_log(Args... args) {
    std::ostringstream oss;
    (oss << ... << args);
    std::lock_guard<std::mutex> lock(g_log_mutex);
    std::cout << oss.str() << "\n";
}

// -------------------------------------------------------
// Configuration & Macros
// -------------------------------------------------------

static const int THREADS_PER_BLOCK = 256;
// Headroom for allocator fragmentation only. The CUDA context is NOT covered
// by this: it is accounted for implicitly by comparing against the free memory
// reported by cudaMemGetInfo after context creation, rather than against
// totalGlobalMem. The old 50 MB margin was an order of magnitude smaller than
// the ~505 MiB context it was implicitly being asked to cover.
static const uint64_t FRAGMENTATION_MARGIN_BYTES = 64ULL * 1024 * 1024;


struct Options {
    uint64_t batchSize = 100000;
    bool showProgress = false;
    PrimeTest primeTest = PrimeTest::BPSW; 
    bool recordCheck = false;
    bool countPrimes = false;
    std::string countFile;
};

// Throw exception instead of exit(1) for graceful multi-thread shutdown
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::ostringstream err_msg;                                     \
            err_msg << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                    << " -- " << cudaGetErrorString(err);                   \
            throw std::runtime_error(err_msg.str());                        \
        }                                                                   \
    } while (0)



// -------------------------------------------------------
// GPU Kernels
// -------------------------------------------------------
// One thread per word of the d_verified bitset. Counts BOTH the verified and
// the unverified numbers of the segment into d_counts[COUNT_VERIFIED] and
// d_counts[COUNT_UNVERIFIED]. The host requires the two to sum to
// seg_even_count and treats anything else as a system error. That makes this
// gate fail closed: a launch that did not run, a grid that missed words, or a
// counter that wrapped all leave the sum short of seg_even_count, rather than
// reading as "0 unverified" and skipping Phase 2.
//
// The counters are 64-bit. The unverified count used to be a uint32_t, which
// wrapped once a segment held 2^32 unverified numbers and then read as 0:
// Phase 2 was skipped and success printed with nothing checked.
//
// Only the segment's own bits are counted. The final word's bits past
// seg_even_count are masked off here rather than assumed either way: the
// transposed kernel leaves them set, the scalar kernel leaves them clear.
//
// One atomic per block and counter: each warp reduces with shuffles, the warp
// sums meet in shared memory, and thread 0 adds the block total. A per-warp
// atomic on the verified count, which is nearly every word, cost ~36 us per
// 2e8-number segment in contention on the one address; per block it costs
// nothing measurable. Every lane reaches the shuffles and the barrier, so
// there is no early return, and blockDim.x must be a multiple of 32 (at most
// 1024); the launch uses 256.
static const int COUNT_VERIFIED   = 0;
static const int COUNT_UNVERIFIED = 1;

__global__ void count_unverified_kernel(
    const uint64_t* __restrict__ d_verified,
    uint64_t seg_even_count,
    unsigned long long* __restrict__ d_counts)
{
    uint64_t w = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t verified_words = (seg_even_count + 63) / 64;

    uint32_t verified = 0, unverified = 0;
    if (w < verified_words) {
        uint64_t valid = seg_even_count - w * 64;   // >= 1, since w < verified_words
        uint64_t mask  = (valid < 64) ? ~(~0ULL << valid) : ~0ULL;
        uint64_t word  = d_verified[w];
        verified   = (uint32_t)__popcll(word & mask);
        unverified = (uint32_t)__popcll(~word & mask);
    }
    for (int off = 16; off > 0; off >>= 1) {
        verified   += __shfl_down_sync(0xffffffffu, verified, off);
        unverified += __shfl_down_sync(0xffffffffu, unverified, off);
    }

    __shared__ uint32_t warp_verified[32], warp_unverified[32];   // <= 2048 each
    if ((threadIdx.x & 31) == 0) {
        warp_verified[threadIdx.x >> 5]   = verified;
        warp_unverified[threadIdx.x >> 5] = unverified;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long block_verified = 0, block_unverified = 0;
        for (unsigned i = 0; i < blockDim.x / 32; i++) {
            block_verified   += warp_verified[i];
            block_unverified += warp_unverified[i];
        }
        if (block_verified)   atomicAdd(&d_counts[COUNT_VERIFIED],   block_verified);
        if (block_unverified) atomicAdd(&d_counts[COUNT_UNVERIFIED], block_unverified);
    }
}

// --count-primes only: number of set bits in d_seg_bits over the bit indices
// [i_lo, i_hi], i.e. the primes among the odd q = q_low + 2i in that range.
// Launched after both sieve kernels and never on the default path, so the
// kernels above are compiled exactly as without it.
__global__ void count_segment_primes_kernel(
    const uint64_t* __restrict__ d_seg_bits,
    uint64_t i_lo, uint64_t i_hi,
    unsigned long long* __restrict__ d_prime_count)
{
    uint64_t w = (i_lo >> 6) + (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = 0;
    if (w <= (i_hi >> 6)) {
        uint64_t word = d_seg_bits[w];
        if (w == (i_lo >> 6)) word &= ~0ULL << (i_lo & 63);
        if (w == (i_hi >> 6) && (i_hi & 63) != 63) word &= ~(~0ULL << ((i_hi & 63) + 1));
        n = (uint32_t)__popcll(word);
    }
    // One atomic per warp rather than per word.
    for (int off = 16; off > 0; off >>= 1) n += __shfl_down_sync(0xffffffffu, n, off);
    if ((threadIdx.x & 31) == 0 && n) atomicAdd(d_prime_count, (unsigned long long)n);
}

// -------------------------------------------------------

// -------------------------------------------------------
// Phase 2 (CPU Fallback) logic
// -------------------------------------------------------
static const uint64_t PHASE2_SIEVE_LIMIT = 100'000'000ULL;

static std::vector<uint64_t> generate_cpu_primes(uint64_t limit) {
    if (limit < 2) return {};
    std::vector<bool> is_prime(limit + 1, true);
    is_prime[0] = is_prime[1] = false;
    for (uint64_t i = 2; i * i <= limit; i++) {
        if (is_prime[i]) {
            for (uint64_t j = i * i; j <= limit; j += i) is_prime[j] = false;
        }
    }
    std::vector<uint64_t> primes;
    primes.reserve(limit / 10);
    for (uint64_t i = 2; i <= limit; i++) if (is_prime[i]) primes.push_back(i);
    return primes;
}

// Returns p_min(n), the smallest prime p <= min(n/2, PHASE2_SIEVE_LIMIT) with
// n - p prime, or 0 if there is none. cpu_primes ascends and each test is
// exact below 2^64, so the first hit is p_min; --record-check relies on that.
// 0 is not a counterexample: it only means no partition with p <= 10^8. For
// n <= 2 * 10^8 the scan covers every p <= n/2 and is exhaustive.
static uint64_t cpu_optimized_check(uint64_t n, const std::vector<uint64_t>& cpu_primes,
                                    PrimeTest primeTest) {
    for (uint64_t p : cpu_primes) {
        if (p > n / 2) break;
        uint64_t q = n - p;
        if (q <= PHASE2_SIEVE_LIMIT) {
            if (std::binary_search(cpu_primes.begin(), cpu_primes.end(), q)) return p;
        } else {
            bool q_prime = (primeTest == PrimeTest::BPSW)
                           ? cpu_is_prime_bpsw(q)
                           : cpu_miller_rabin(q);
            if (q_prime) return p;
        }
    }
    return 0;
}

// -------------------------------------------------------
// GPU Worker Thread (Dynamic Load Balancing)
// -------------------------------------------------------
void run_gpu_worker(
    int device_id, uint64_t LIMIT, uint64_t SEG_SIZE, uint64_t P_SMALL, uint64_t P_BATCH,
    uint64_t small_high, size_t small_bytes,
    const PrimeBitset& small_bitset,
    const std::vector<uint64_t>& small_primes,
    const std::vector<uint64_t>& gpu_primes,
    const std::vector<uint64_t>& cpu_primes,
    PrimeTest primeTest,
    bool recordCheck,
    bool countPrimes,
    uint64_t COUNT_HIGH
)
{
    try {
        CUDA_CHECK(cudaSetDevice(device_id));
        // Copy primality-test selector to this device's constant memory
        CUDA_CHECK(cudaMemcpyToSymbol(g_device_prime_test, &primeTest, sizeof(PrimeTest)));
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Memory Allocations
        uint64_t* d_small = nullptr;
        CUDA_CHECK(cudaMalloc(&d_small, small_bytes));
        CUDA_CHECK(cudaMemcpyAsync(d_small, small_bitset.data(), small_bytes, cudaMemcpyHostToDevice, stream));

        uint64_t small_prime_count = small_primes.size();
        // Split the prime list at TILE_ODDS once: primes below it are worth a
        // per-tile scan, the rest are not (see large_prime_sieve_kernel).
        uint64_t sieve_small_count =
            sieve_split_prime_count(small_primes.data(), small_prime_count);
        uint64_t sieve_large_count = small_prime_count - sieve_small_count;

        uint64_t* d_small_primes = nullptr;
        CUDA_CHECK(cudaMalloc(&d_small_primes, small_prime_count * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyAsync(d_small_primes, small_primes.data(), small_prime_count * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));

        uint64_t max_q_span = 2 * SEG_SIZE + P_SMALL;
        uint64_t max_odds = (max_q_span + 1) / 2;
        uint64_t seg_words = (max_odds + 63) / 64;
        // +1 padding word: the transposed Phase 1 kernel reads one word past
        // the word holding i_base.
        size_t seg_bytes = (seg_words + 1) * sizeof(uint64_t);

        uint64_t* d_seg_bits = nullptr;
        uint64_t* d_verified = nullptr;   // bitset: 1 bit per even number
        uint64_t* d_p_batch  = nullptr;
        unsigned long long* d_counts = nullptr;   // [COUNT_VERIFIED], [COUNT_UNVERIFIED]
        unsigned long long* d_record = nullptr;
        unsigned long long* d_prime_count = nullptr;

        CUDA_CHECK(cudaMalloc(&d_seg_bits, seg_bytes));
        // d_verified holds one bit per even number; SEG_SIZE is the largest
        // seg_even_count any segment can have.
        uint64_t max_verified_words = (SEG_SIZE + 63) / 64;
        CUDA_CHECK(cudaMalloc(&d_verified, max_verified_words * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_p_batch, P_BATCH * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_counts, 2 * sizeof(unsigned long long)));
        if (recordCheck) CUDA_CHECK(cudaMalloc(&d_record, sizeof(unsigned long long)));
        if (countPrimes) CUDA_CHECK(cudaMalloc(&d_prime_count, sizeof(unsigned long long)));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Main Work Loop
        while (!g_failure.load(std::memory_order_relaxed) && !g_system_error.load(std::memory_order_relaxed)) {
            
            uint64_t seg_start = g_next_segment_start.fetch_add(SEG_SIZE * 2, std::memory_order_relaxed);
            if (seg_start > LIMIT) break;

            uint64_t seg_end = std::min(seg_start + SEG_SIZE * 2 - 2, LIMIT);
            uint64_t seg_even_count = (seg_end - seg_start) / 2 + 1;

            uint64_t q_low = (seg_start > P_SMALL ? seg_start - P_SMALL : 3);
            if ((q_low & 1) == 0) q_low++;
            uint64_t q_high = (seg_end < UINT64_MAX - 1) ? seg_end + 1 : seg_end;
            if ((q_high & 1) == 0) q_high++;

            uint64_t num_odds = (q_high - q_low) / 2 + 1;

            // A. Sieve Segment
            launch_segment_sieve(q_low, q_high, d_small_primes,
                                 sieve_small_count, sieve_large_count,
                                 d_seg_bits, THREADS_PER_BLOCK, stream);
            CUDA_CHECK(cudaGetLastError());

            // --count-primes: count this segment's own odd q, [seg_start + 1,
            // seg_end + 1] clipped to COUNT_HIGH. The sieved span reaches
            // P_SMALL further down, but everything below seg_start - 1 is the
            // previous segment's q_high and already counted there, so these
            // ranges tile [START + 1, COUNT_HIGH] exactly. Runs on the final
            // bitset: same stream, after both sieve kernels.
            uint64_t count_lo = seg_start + 1;
            uint64_t count_hi = std::min(q_high, COUNT_HIGH);
            bool count_this = countPrimes && count_lo <= count_hi;
            if (count_this) {
                uint64_t i_lo = (count_lo - q_low) / 2;
                uint64_t i_hi = (count_hi - q_low) / 2;
                uint64_t words = (i_hi >> 6) - (i_lo >> 6) + 1;
                CUDA_CHECK(cudaMemsetAsync(d_prime_count, 0, sizeof(unsigned long long), stream));
                count_segment_primes_kernel<<<(uint32_t)((words + 255) / 256), 256, 0, stream>>>(
                    d_seg_bits, i_lo, i_hi, d_prime_count);
                CUDA_CHECK(cudaGetLastError());
            }

            // Zero the word just past this segment's bits, so the transposed
            // kernel's one-word overread sees 0 rather than the previous
            // segment's leftovers.
            uint64_t cur_seg_words = (num_odds + 63) / 64;
            CUDA_CHECK(cudaMemsetAsync(d_seg_bits + cur_seg_words, 0,
                                       sizeof(uint64_t), stream));

            // B. Phase 1 Verification Batches
            uint64_t verified_words = (seg_even_count + 63) / 64;
            CUDA_CHECK(cudaMemsetAsync(d_verified, 0,
                                       verified_words * sizeof(uint64_t), stream));
            if (recordCheck)
                CUDA_CHECK(cudaMemsetAsync(d_record, 0, sizeof(unsigned long long), stream));
            for (uint64_t bi = 0; bi < gpu_primes.size(); bi += P_BATCH) {
                uint64_t bsize = std::min(P_BATCH, (uint64_t)gpu_primes.size() - bi);
                CUDA_CHECK(cudaMemcpyAsync(d_p_batch, gpu_primes.data() + bi, bsize * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));

                launch_goldbach_phase1(
                    d_small, small_high, d_seg_bits, q_low, q_high,
                    seg_start, seg_even_count, d_p_batch, bsize, d_verified,
                    P_SMALL, THREADS_PER_BLOCK, stream, d_record);
                CUDA_CHECK(cudaGetLastError());
            }

            // C. Count verified and unverified numbers
            unsigned long long counts[2] = {0, 0};
            CUDA_CHECK(cudaMemsetAsync(d_counts, 0, 2 * sizeof(unsigned long long), stream));

            uint32_t count_blocks = (uint32_t)((verified_words + 255) / 256);
            count_unverified_kernel<<<count_blocks, 256, 0, stream>>>(d_verified, seg_even_count, d_counts);
            // A failed launch is reported only here: the copy and the sync
            // below would both return success and leave the counts at 0.
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(counts, d_counts, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));

            // Rides along with the unverified-count readback: no extra sync.
            unsigned long long record_enc = 0;
            if (recordCheck)
                CUDA_CHECK(cudaMemcpyAsync(&record_enc, d_record, sizeof(unsigned long long),
                                           cudaMemcpyDeviceToHost, stream));
            unsigned long long seg_prime_count = 0;
            if (count_this)
                CUDA_CHECK(cudaMemcpyAsync(&seg_prime_count, d_prime_count, sizeof(unsigned long long),
                                           cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Fail closed: every number of the segment must have been counted
            // exactly once, as verified or as unverified. A shortfall means the
            // count did not happen as launched, and "0 unverified" could then
            // be false, so it is a system error, never a success.
            const uint64_t verified_count   = counts[COUNT_VERIFIED];
            const uint64_t unverified_count = counts[COUNT_UNVERIFIED];
            if (verified_count + unverified_count != seg_even_count) {
                std::ostringstream err_msg;
                err_msg << "segment count invariant failed at seg_start=" << seg_start
                        << ": verified " << verified_count << " + unverified "
                        << unverified_count << " != " << seg_even_count;
                throw std::runtime_error(err_msg.str());
            }

            if (count_this) {
                g_prime_count_total.fetch_add(seg_prime_count, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lk(g_count_mutex);
                g_segment_counts.push_back({count_lo, count_hi, seg_prime_count});
            }

            // --record-check: this segment's largest p_min and the smallest n
            // attaining it. Phase 1's candidate is decoded here; Phase 2 below
            // may replace it, so the record is published only after Phase 2.
            uint64_t rec_p = 0, rec_n = 0;
            if (recordCheck && record_enc) {
                rec_p = (uint64_t)(record_enc >> RECORD_IDX_BITS);
                rec_n = seg_start + 2 * (RECORD_IDX_MASK - (record_enc & RECORD_IDX_MASK));
            }
            bool seg_failed = false;

            // D. CPU Phase 2 Processing
            if (unverified_count > 0) {
                std::vector<uint64_t> verified(verified_words);
                CUDA_CHECK(cudaMemcpy(verified.data(), d_verified,
                                      verified_words * sizeof(uint64_t), cudaMemcpyDeviceToHost));

                // Bounded by seg_even_count, so the final word's tail bits are
                // never examined regardless of how they were left.
                for (uint64_t i = 0; i < seg_even_count; i++) {
                    if (!((verified[i >> 6] >> (i & 63)) & 1ULL)) {
                        uint64_t n = seg_start + i * 2;
                        g_total_phase2_count.fetch_add(1, std::memory_order_relaxed);
                        // safe_log("[GPU ", device_id, "] Phase 2 fallback for n = ", n, "...");
                        
                        uint64_t p2 = cpu_optimized_check(n, cpu_primes, primeTest);
                        if (p2 == 0) {
                            g_failure.store(true, std::memory_order_relaxed);
                            g_failure_n.store(n, std::memory_order_relaxed);
                            seg_failed = true;
                            break;
                        }
                        // Every number that reaches Phase 2 has p_min > P_SMALL,
                        // above any p_min Phase 1 found, so records must include
                        // these: omitting them printed false records and missed
                        // the maximum. n ascends, so a strict > keeps the
                        // smallest n attaining the segment maximum.
                        if (p2 > rec_p) { rec_p = p2; rec_n = n; }
                    }
                }
            }

            if (recordCheck && rec_p && !seg_failed) {
                std::lock_guard<std::mutex> lk(g_record_mutex);
                if (rec_p > g_record_p) {
                    g_record_p = rec_p;
                    safe_log("[record] n=", rec_n, " p_min=", rec_p);
                }
            }
            g_total_processed.fetch_add(seg_even_count, std::memory_order_relaxed);
        }

        // Cleanup
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(d_small));
        CUDA_CHECK(cudaFree(d_small_primes));
        CUDA_CHECK(cudaFree(d_seg_bits));
        CUDA_CHECK(cudaFree(d_verified));
        CUDA_CHECK(cudaFree(d_p_batch));
        CUDA_CHECK(cudaFree(d_counts));
        if (d_record) CUDA_CHECK(cudaFree(d_record));
        if (d_prime_count) CUDA_CHECK(cudaFree(d_prime_count));

    } catch (const std::exception& e) {
        safe_log("[!] FATAL ERROR in GPU ", device_id, " Worker: ", e.what());
        g_system_error.store(true, std::memory_order_relaxed);
    }
}

// -------------------------------------------------------
// Initialization & Hardware Check
// -------------------------------------------------------
// Bytes this configuration will allocate on each device, excluding the CUDA
// context (which cudaMemGetInfo already excludes from `free`).
static uint64_t device_alloc_bytes(uint64_t SEG_SIZE, uint64_t P_SMALL, uint64_t P_BATCH,
                                   size_t small_bytes, uint64_t small_prime_count)
{
    uint64_t verified_bytes     = ((SEG_SIZE + 63) / 64) * sizeof(uint64_t);
    uint64_t p_batch_bytes      = P_BATCH * sizeof(uint64_t);
    uint64_t max_q_span         = 2 * SEG_SIZE + P_SMALL;
    uint64_t max_odds           = (max_q_span + 1) / 2;
    uint64_t seg_words          = (max_odds + 63) / 64;
    uint64_t seg_bytes          = (seg_words + 1) * sizeof(uint64_t);
    uint64_t small_primes_bytes = small_prime_count * sizeof(uint64_t);  // was unaccounted
    return verified_bytes + p_batch_bytes + seg_bytes + small_bytes + small_primes_bytes
         + 2 * sizeof(unsigned long long);
}

// Largest SEG_SIZE whose allocations stay within about a quarter of free VRAM.
// The two SEG_SIZE-dependent buffers (d_verified, d_seg_bits) are each roughly
// SEG_SIZE/8 bytes, so the variable part is ~SEG_SIZE/4.
static uint64_t derive_seg_size(size_t free_bytes, uint64_t P_SMALL, uint64_t P_BATCH,
                                size_t small_bytes, uint64_t small_prime_count)
{
    const uint64_t CAP = 200'000'000ULL;   // never exceed the documented default
    const uint64_t FLOOR = 1'000'000ULL;

    // P_SMALL widens the sieved q span by a constant, costing ~P_SMALL/16 bytes
    // in d_seg_bits regardless of SEG_SIZE, so it belongs in the fixed term.
    uint64_t fixed = P_BATCH * sizeof(uint64_t) + small_bytes
                   + small_prime_count * sizeof(uint64_t)
                   + P_SMALL / 16 + FRAGMENTATION_MARGIN_BYTES;
    uint64_t budget = (uint64_t)free_bytes / 4;
    if (budget <= fixed) return FLOOR;

    uint64_t seg = 4 * (budget - fixed);
    if (seg > CAP)   seg = CAP;
    if (seg < FLOOR) seg = FLOOR;
    if (seg & 1ULL)  seg--;               // SEG_SIZE must be even
    return seg;
}

void validate_hardware_and_limits(int use_gpus, uint64_t SEG_SIZE, uint64_t P_SMALL,
                                  uint64_t P_BATCH, size_t small_bytes,
                                  uint64_t small_prime_count) {
    uint64_t max_q_span = 2 * SEG_SIZE + P_SMALL;
    uint64_t max_odds   = (max_q_span + 1) / 2;

    uint64_t total_required =
        device_alloc_bytes(SEG_SIZE, P_SMALL, P_BATCH, small_bytes, small_prime_count)
        + FRAGMENTATION_MARGIN_BYTES;

    // Validate CUDA Grid Sizes
    uint64_t num_tiles = (max_odds + TILE_ODDS - 1) / TILE_ODDS;
    uint64_t blocks = (SEG_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Grid dimensions are validated per device against maxGridSize[0] below.
    // Thread indices are computed as (uint64_t)blockIdx.x * blockDim.x, so the
    // old 2^32 total-thread ceiling no longer applies; the binding limits are
    // the device's grid extent and free memory.

    // Validate shared memory and VRAM for all selected devices
    for (int i = 0; i < use_gpus; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        // Grid extent: the widest launch is the scalar Phase 1 path, one
        // thread per even number. maxGridSize[0] is the real ceiling (2^31-1
        // on current hardware), not UINT32_MAX.
        if (blocks > (uint64_t)prop.maxGridSize[0] || num_tiles > (uint64_t)prop.maxGridSize[0]) {
            std::cerr << "\n[!] ERROR: GPU " << i << " (" << prop.name
                      << "): segment size exceeds the device grid limit.\n";
            std::cerr << "    Phase 1 blocks needed: " << blocks
                      << " | sieve tiles needed: " << num_tiles << "\n";
            std::cerr << "    Device maxGridSize[0]: " << prop.maxGridSize[0] << "\n";
            std::cerr << "    Maximum --seg-size on this device: "
                      << (uint64_t)prop.maxGridSize[0] * THREADS_PER_BLOCK << "\n";
            std::exit(1);
        }

        // Shared memory: the tiled sieve asks for TILE_ODDS bytes per block.
        // Fail here, naming both numbers, rather than at launch with the
        // opaque "invalid argument". Never silently reduce TILE_ODDS -- that
        // would change what is being measured without saying so.
        size_t tile_shared = (size_t)TILE_ODDS * sizeof(unsigned char);
        if (tile_shared > prop.sharedMemPerBlock) {
            std::cerr << "\n[!] ERROR: GPU " << i << " (" << prop.name
                      << ") cannot provide the shared memory this build requires.\n";
            std::cerr << "    TILE_ODDS = " << (uint64_t)TILE_ODDS
                      << " needs " << tile_shared << " bytes per block\n";
            std::cerr << "    Device sharedMemPerBlock = " << prop.sharedMemPerBlock << " bytes\n";
            std::cerr << "    Rebuild with -DTILE_ODDS=" << (prop.sharedMemPerBlock / 2)
                      << " or smaller (a power of two).\n";
            std::exit(1);
        }

        // Free memory, queried after cudaSetDevice so the context already
        // exists and is therefore excluded from `free` -- this is what makes
        // the estimate comparable to reality.
        CUDA_CHECK(cudaSetDevice(i));
        size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

        std::cout << "[vram] GPU " << i << " (" << prop.name << "): need "
                  << total_required / (1024*1024) << " MB, free "
                  << free_bytes / (1024*1024) << " MB of "
                  << total_bytes / (1024*1024) << " MB total\n";

        if (total_required > free_bytes) {
            std::cerr << "\n[!] ERROR: GPU " << i << " (" << prop.name << ") has insufficient free VRAM.\n";
            std::cerr << "    Required: " << total_required / (1024*1024)
                      << " MB (including " << FRAGMENTATION_MARGIN_BYTES / (1024*1024)
                      << " MB fragmentation margin)\n";
            std::cerr << "    Free:     " << free_bytes / (1024*1024)
                      << " MB of " << total_bytes / (1024*1024) << " MB total\n";
            std::cerr << "    Reduce --seg-size or --batch-size, or free GPU memory.\n";
            std::exit(1);
        }
        std::cout << "[Hardware] GPU " << i << ": " << prop.name 
                  << " (" << prop.totalGlobalMem / (1024*1024) << " MB VRAM)\n";
    }
}

void print_usage(const char* prog) {
    std::cout << "Goldbach Multi-GPU Verifier\n\n"
              << "Usage:\n"
              << "  " << prog << " <LIMIT> [SEG_SIZE] [P_SMALL]\n"
              << "  " << prog << " <LIMIT> [--seg-size=N] [--p-small=N] [--gpus=N] [options]\n\n"
              << "Required:\n"
              << "  LIMIT            Max integer to check, 4 <= LIMIT <= 18446744065119617024\n"
              << "                   (2^64 - 2^33); an odd LIMIT checks up to LIMIT - 1\n\n"
              << "Optional:\n"
              << "  --seg-size=N     Even integers per segment, even, 2 <= N < 2^32\n"
              << "                   (default: derived from free VRAM)\n"
              << "  --record-check   Print each new maximum p_min as [record] n=... p_min=...\n"
              << "                   (requires the default --start=4; at most one per segment)\n"
              << "  --count-primes   Also count the primes up to LIMIT from the segment sieve;\n"
              << "                   prints pi(LIMIT) = ...\n"
              << "  --count-file=F   With --count-primes, write one line per segment to F:\n"
              << "                   \"A B count\", count = number of primes in [A, B]\n"
              << "  --p-small=N      GPU prime search bound, 3 <= N <= 4,000,000,000\n"
              << "                   (default: 1000000)\n"
              << "  --batch-size=N   Primes per GPU batch, 1 <= N <= 2^32 (default: 100000)\n"
              << "  --gpus=N         Number of GPUs to use (default: 1 | all: -1)\n"
              << "  --start=N        First even number to verify, START <= LIMIT (default: 4)\n"
              << "  --primetest=X    Primality test: BPSW (default) or MR\n"
              << "  --progress       Show real-time progress updates\n"
              << "  -h, --help       Show this help message\n"
              << "\nWith G GPUs, LIMIT + 2*SEG_SIZE*(G+1) must also stay below 2^64.\n";
}

// -------------------------------------------------------
// Parameter limits
// -------------------------------------------------------
// Every limit below is derived from the arithmetic it protects.
//
// MAX_LIMIT = 2^64 - 2^33. With SEG_SIZE < 2^32 a segment spans 2*SEG_SIZE - 2
// < 2^33, so for every seg_start <= LIMIT the values seg_start + 2*SEG_SIZE,
// seg_end + 1 (= q_high) and q + 1 (the strong Lucas test works on n + 1) stay
// below 2^64. The same 2^33 headroom exceeds every sieving prime (at most
// isqrt(2^64) + 1 = 2^32 + 1), so even the sieve's former first-multiple
// formula q_low + p - 1 could not overflow; the sieve no longer forms that sum
// at all.
//
// The segment counter is a separate bound, checked once the GPU count G and
// the final SEG_SIZE are known: each of the G workers claims one segment start
// past LIMIT before it stops, so the counter reaches at most
// LIMIT + 2*SEG_SIZE*(G+1), and that must not wrap.
//
// MAX_SEG_SIZE < 2^32 and MAX_BATCH_SIZE = 2^32, with P_SMALL <= 4e9, keep
// every device-memory byte computation below 2^36.
static const uint64_t MAX_LIMIT      = 18446744065119617024ULL;   // 2^64 - 2^33
static const uint64_t MAX_SEG_SIZE   = 4294967294ULL;             // largest even < 2^32
static const uint64_t MIN_P_SMALL    = 3;
static const uint64_t MAX_P_SMALL    = 4'000'000'000ULL;
static const uint64_t MAX_BATCH_SIZE = 4294967296ULL;             // 2^32

// Strict decimal parse: digits only -- no sign, no whitespace, no suffix --
// and no silent wrap. std::stoull accepted "-1" as 2^64 - 1.
static uint64_t parse_u64(const std::string& text, const std::string& what) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(what + " must be a non-negative decimal integer, got '"
                                    + text + "'");
    uint64_t v = 0;
    for (char c : text) {
        uint64_t d = (uint64_t)(c - '0');
        if (v > (UINT64_MAX - d) / 10)
            throw std::invalid_argument(what + " is out of range: '" + text + "'");
        v = v * 10 + d;
    }
    return v;
}

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 0; }

    Options opt;
    uint64_t LIMIT = 0;
    uint64_t SEG_SIZE = 10'000'000ULL;
    bool seg_size_explicit = false;
    uint64_t P_SMALL = 1'000'000ULL;
    uint64_t START = 4; // Default starting point
    int requested_gpus = 1;

    try {
        std::vector<std::string> positional;
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
            if (arg == "--progress") { opt.showProgress = true; continue; }
            if (arg.rfind("--batch-size=", 0) == 0) { opt.batchSize = parse_u64(arg.substr(13), "--batch-size"); continue; }
            if (arg.rfind("--gpus=", 0) == 0) {
                std::string v = arg.substr(7);
                if (v == "-1") { requested_gpus = -1; continue; }
                uint64_t g = parse_u64(v, "--gpus");
                if (g == 0 || g > 1024)
                    throw std::invalid_argument("--gpus must be -1 (all) or between 1 and 1024, got '" + v + "'");
                requested_gpus = (int)g;
                continue;
            }
            if (arg.rfind("--seg-size=", 0) == 0) { SEG_SIZE = parse_u64(arg.substr(11), "--seg-size"); seg_size_explicit = true; continue; }
            if (arg.rfind("--p-small=", 0) == 0) { P_SMALL = parse_u64(arg.substr(10), "--p-small"); continue; }
            if (arg == "--record-check") { opt.recordCheck = true; continue; }
            if (arg == "--count-primes") { opt.countPrimes = true; continue; }
            if (arg.rfind("--count-file=", 0) == 0) { opt.countFile = arg.substr(13); continue; }
            if (arg.rfind("--start=", 0) == 0) { START = parse_u64(arg.substr(8), "--start"); continue; }
            if (arg.rfind("--primetest=", 0) == 0) {
                std::string mode = arg.substr(12);
                if (mode == "MR" || mode == "mr") {
                    opt.primeTest = PrimeTest::MillerRabin;
                } else if (mode == "BPSW" || mode == "bpsw") {
                    opt.primeTest = PrimeTest::BPSW;
                } else {
                    throw std::invalid_argument("unknown --primetest value '" + mode + "' (use MR or BPSW)");
                }
                continue;
            }
            if (arg.size() > 1 && arg[0] == '-') {
                if (isdigit((unsigned char)arg[1]))
                    throw std::invalid_argument("positional arguments must be non-negative decimal integers, got '"
                                                + arg + "'");
                throw std::invalid_argument("unknown option '" + arg + "'");
            }
            positional.push_back(arg);
        }

        if (positional.empty())
            throw std::invalid_argument("LIMIT is required");
        if (positional.size() > 3)
            throw std::invalid_argument("too many positional arguments (expected <LIMIT> [SEG_SIZE] [P_SMALL]), got '"
                                        + positional[3] + "'");
        LIMIT = parse_u64(positional[0], "LIMIT");
        if (positional.size() >= 2) { SEG_SIZE = parse_u64(positional[1], "SEG_SIZE"); seg_size_explicit = true; }
        if (positional.size() >= 3) P_SMALL = parse_u64(positional[2], "P_SMALL");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    if (LIMIT < 4) { std::cerr << "Error: LIMIT must be >= 4.\n"; return 1; }
    if (LIMIT > MAX_LIMIT) {
        std::cerr << "Error: LIMIT must be <= " << MAX_LIMIT << " (2^64 - 2^33).\n";
        return 1;
    }
    // --count-primes counts up to the LIMIT as given: an odd LIMIT is itself
    // the last q the final segment sieves (q_high = LIMIT - 1 + 1).
    const uint64_t COUNT_HIGH = LIMIT;
    if (LIMIT % 2 != 0) LIMIT--;
    if (!opt.countFile.empty() && !opt.countPrimes) {
        std::cerr << "Error: --count-file requires --count-primes.\n"; return 1;
    }
    if (seg_size_explicit && (SEG_SIZE == 0 || SEG_SIZE % 2 != 0 || SEG_SIZE > MAX_SEG_SIZE)) {
        std::cerr << "Error: SEG_SIZE must be even, > 0 and < 2^32 (at most "
                  << MAX_SEG_SIZE << "), got " << SEG_SIZE << ".\n";
        return 1;
    }
    if (P_SMALL < MIN_P_SMALL || P_SMALL > MAX_P_SMALL) {
        std::cerr << "Error: P_SMALL must be between " << MIN_P_SMALL << " and "
                  << MAX_P_SMALL << ", got " << P_SMALL << ".\n";
        return 1;
    }
    if (opt.batchSize == 0 || opt.batchSize > MAX_BATCH_SIZE) {
        std::cerr << "Error: --batch-size must be between 1 and " << MAX_BATCH_SIZE
                  << ", got " << opt.batchSize << ".\n";
        return 1;
    }
    if (P_SMALL > LIMIT) P_SMALL = LIMIT;   // LIMIT >= 4, so P_SMALL stays >= 3

    // START is compared before it is rounded: LIMIT <= MAX_LIMIT is even, so
    // an odd START <= LIMIT rounds up to at most LIMIT and cannot wrap.
    if (START > LIMIT) {
        std::cerr << "Error: START must be <= LIMIT.\n";
        return 1;
    }
    if (START < 4) START = 4;
    if (START % 2 != 0) START++; // Force it to be even

    // A record is an n whose p_min exceeds that of every smaller even number
    // from 4. Starting elsewhere would print maxima relative to START, which
    // look like records but are not.
    if (opt.recordCheck && START != 4) {
        std::cerr << "Error: --record-check requires --start=4 (got START=" << START
                  << "): records are defined relative to 4.\n";
        return 1;
    }

    // Performance warning
    if (P_SMALL < 1'000'000ULL) {
        std::cerr << "\n[!] WARNING: P_SMALL = " << P_SMALL << " is very small.\n";
        std::cerr << "    This may cause excessive Phase 2 fallbacks.\n";
        std::cerr << "    Recommended: P_SMALL >= 10^7 for numbers > 10^12\n";
        std::cerr << "                 P_SMALL >= 10^8 for numbers > 10^18\n\n";
    }

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) { std::cerr << "No CUDA devices found.\n"; return 1; }

    if (requested_gpus > device_count) {
        std::cerr << "Requested " << requested_gpus
                << " GPUs, but only " << device_count
                << " available. Using " << device_count << ".\n";
    }

    int use_gpus = (requested_gpus <= 0 || requested_gpus > device_count) ? device_count : requested_gpus;

    // Integer sqrt to avoid precision loss
    uint64_t sqrt_limit = 0;
    if (LIMIT >= 4) {
        uint64_t low = 1, high = LIMIT;
        if (high > (1ULL << 32)) high = (1ULL << 32); 
        while (low <= high) {
            uint64_t mid = low + (high - low) / 2;
            if (mid <= LIMIT / mid) { sqrt_limit = mid; low = mid + 1; }
            else { high = mid - 1; }
        }
    }

    uint64_t small_high = std::max(sqrt_limit + 1, P_SMALL);
    if (small_high % 2 == 0) small_high++;

    uint64_t num_small_odds = (small_high - 3) / 2 + 1;
    size_t small_bytes = ((num_small_odds + 63) / 64) * sizeof(uint64_t);

    // Establish the primary device's context up front and read free memory
    // from it. Doing this before the init timer keeps context-creation cost
    // out of the reported initialization time.
    CUDA_CHECK(cudaSetDevice(0));
    size_t free_bytes0 = 0, total_bytes0 = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes0, &total_bytes0));

    // An explicit --seg-size always wins; otherwise scale to free VRAM.
    // Uses an upper bound on the prime count (pi(x) < 1.3 x / ln x for x >= 17)
    // because the real list is not built yet; it only affects a fixed term.
    if (!seg_size_explicit) {
        uint64_t pi_bound = (small_high < 17) ? 8
                          : (uint64_t)(1.3 * (double)small_high / std::log((double)small_high));
        SEG_SIZE = derive_seg_size(free_bytes0, P_SMALL, opt.batchSize, small_bytes, pi_bound);
        std::cout << "[auto] --seg-size not given; chose " << SEG_SIZE
                  << " from " << free_bytes0 / (1024*1024) << " MB free VRAM\n";
    }

    // Segment counter headroom (see MAX_LIMIT above): LIMIT + 2*SEG_SIZE*(G+1)
    // must not wrap, or a worker would claim a small seg_start again.
    if ((unsigned __int128)LIMIT + (unsigned __int128)2 * SEG_SIZE * (unsigned)(use_gpus + 1)
            > (unsigned __int128)UINT64_MAX) {
        std::cerr << "Error: LIMIT + 2*SEG_SIZE*(GPUs+1) exceeds 2^64 - 1, so the segment counter\n"
                  << "       would wrap (LIMIT=" << LIMIT << ", SEG_SIZE=" << SEG_SIZE
                  << ", GPUs=" << use_gpus << "). Reduce LIMIT, --seg-size or --gpus.\n";
        return 1;
    }

    // std::cout << "\nGoldbach Multi-GPU Verifier (Limit: " << LIMIT << ")\n";
    std::cout << "Building small primes bitset up to " << small_high << "...\n";
    auto t0 = now();
    
    PrimeBitset small_bitset = build_prime_bitset(small_high);
    
    std::vector<uint64_t> small_primes;
    small_primes.reserve(small_high / 10);
    if (small_bitset.is_prime(2)) small_primes.push_back(2);
    for (uint64_t i = 3; i <= small_high; i += 2) {
        if (small_bitset.is_prime(i)) small_primes.push_back(i);
    }

    std::vector<uint64_t> gpu_primes;
    for (uint64_t p : small_primes) {
        if (p <= P_SMALL) gpu_primes.push_back(p);
    }

    // Both Phase 1 kernels return on the FIRST prime that works, so the prime
    // they report is p_min only if this list ascends. Verification itself does
    // not depend on the order, but --record-check does, and a silently
    // mis-ordered list would give plausible-looking wrong records.
    if (!std::is_sorted(gpu_primes.begin(), gpu_primes.end())) {
        std::cerr << "[!] ERROR: gpu_primes is not sorted ascending.\n";
        std::cerr << "    Phase 1 returns on the first hit, so p_min and --record-check\n";
        std::cerr << "    both depend on ascending order.\n";
        return 1;
    }
    if (opt.recordCheck) {
        // The record encoding packs p_min and the segment index into 32 bits each.
        if (SEG_SIZE > RECORD_IDX_MASK || P_SMALL > RECORD_IDX_MASK) {
            std::cerr << "[!] ERROR: --record-check needs --seg-size and --p-small below "
                      << RECORD_IDX_MASK << ".\n";
            return 1;
        }
        std::cout << "[record] tracking enabled (" << gpu_primes.size()
                  << " primes, ascending)\n";
    }

    // --count-primes: segments count only q in [START + 1, COUNT_HIGH], so the
    // primes up to START (2 and 3 at the default START = 4) are added here.
    uint64_t primes_below_start = 0;
    if (opt.countPrimes) {
        if (START > small_high) {
            std::cerr << "[!] ERROR: --count-primes needs --start <= " << small_high << ".\n";
            return 1;
        }
        primes_below_start = (uint64_t)(std::upper_bound(small_primes.begin(),
                                        small_primes.end(), START) - small_primes.begin());
        std::cout << "[count] counting primes; " << primes_below_start
                  << " at or below START=" << START << " added on the host\n";
    }

    // Fail-Fast Validations. Placed here so small_primes.size() is the real
    // count rather than an estimate, but still ahead of the ~200 ms Phase 2
    // prime table below.
    validate_hardware_and_limits(use_gpus, SEG_SIZE, P_SMALL, opt.batchSize,
                                 small_bytes, small_primes.size());

    std::cout << "Pre-generating CPU primes up to " << PHASE2_SIEVE_LIMIT << "...\n";
    std::vector<uint64_t> cpu_primes = generate_cpu_primes(PHASE2_SIEVE_LIMIT);

    auto t1 = now();
    std::cout << "Initialization completed in " 
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms.\n";

    // Benchmark instrumentation: prime counts actually used, and how the list
    // splits at SPLIT_THRESHOLD. Printed after the init timer so it cannot
    // perturb it.
    {
        uint64_t split = sieve_split_prime_count(small_primes.data(), small_primes.size());
        std::cout << "[counts] small_high=" << small_high
                  << " small_prime_count=" << small_primes.size()
                  << " gpu_primes=" << gpu_primes.size()
                  << " | SPLIT_THRESHOLD=" << (uint64_t)SPLIT_THRESHOLD
                  << " tiled=" << split
                  << " large=" << (small_primes.size() - split) << "\n\n";
    }

    // ========================================================================
    // INITIALIZE ATOMIC COUNTER & ADJUST TOTALS
    // ========================================================================
    
    // 1. Seed the global atomic counter with the starting value.
    // Every GPU worker will execute a fetch_add() against this exact variable
    // to grab its first segment. It must be stored before any thread is created.
    g_next_segment_start.store(START);

    // 2. Adjust the total workload calculation for accurate statistics and logging.
    // Instead of (LIMIT - 4), we use (LIMIT - START).
    uint64_t total_even_to_check = (LIMIT - START) / 2 + 1;

    std::cout << "--- Launching Multi-GPU Verifier ---\n";
    std::cout << "Checking range : [" << START << ", " << LIMIT << "]\n";
    std::cout << "Total numbers  : " << total_even_to_check << "\n\n";

    auto t_main_start = now();


    // Progress Monitor (optional, controlled by --progress flag)
    std::thread progress_thread;
    std::atomic<bool> progress_running{false};

    if (opt.showProgress) {
        progress_running.store(true);

        // total_even_to_check is captured by value. This block used to declare
        // its own copy, which the by-reference capture then read after the
        // block had ended, while the thread was still running.
        progress_thread = std::thread([&, total_even_to_check]() {
            auto start_time = now();
            auto last_update = start_time;
            
            while (progress_running.load() && 
                !g_failure.load() && 
                !g_system_error.load()) {
                
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                
                auto current_time = now();
                uint64_t processed = g_total_processed.load(std::memory_order_relaxed);
                
                if (processed >= total_even_to_check) break;
                
                // Update every 0.5 seconds
                auto elapsed_since_update = std::chrono::duration<double>(
                    current_time - last_update).count();
                
                if (elapsed_since_update >= 0.5) {
                    double total_elapsed = std::chrono::duration<double>(
                        current_time - start_time).count();
                    
                    double pct = 100.0 * processed / total_even_to_check;
                    uint64_t rate = (total_elapsed > 0) ? 
                        (uint64_t)(processed / total_elapsed) : 0;
                    
                    // Estimate time remaining
                    uint64_t remaining = total_even_to_check - processed;
                    uint64_t eta_seconds = (rate > 0) ? remaining / rate : 0;
                    
                    std::cout << "\r[Progress] " 
                            << processed << " / " << total_even_to_check
                            << " (" << std::fixed << std::setprecision(2) << pct << "%) "
                            << "| " << std::scientific << std::setprecision(2) 
                            << (double)rate << " numbers/sec "
                            << "| ETA: " << eta_seconds << "s          "
                            << std::flush;
                    
                    last_update = current_time;
                }
            }
            
            // Clear progress line when done
            if (progress_running.load()) {
                std::cout << "\r" << std::string(80, ' ') << "\r" << std::flush;
            }
        });
    }

    // Launch Worker Threads
    std::vector<std::thread> workers;
    for (int g = 0; g < use_gpus; ++g) {
        workers.emplace_back(
            run_gpu_worker, g, LIMIT, SEG_SIZE, P_SMALL, opt.batchSize,
            small_high, small_bytes, std::cref(small_bitset),
            std::cref(small_primes), std::cref(gpu_primes), std::cref(cpu_primes),
            opt.primeTest, opt.recordCheck, opt.countPrimes, COUNT_HIGH
        );
    }


    for (auto& t : workers) {
        if (t.joinable()) t.join();
    }

    auto t_main_end = now();
    double total_ms = std::chrono::duration<double, std::milli>(t_main_end - t_main_start).count();

    // Stop progress thread
    if (opt.showProgress) {
        progress_running.store(false);
        if (progress_thread.joinable()) {
            progress_thread.join();
        }
    }

    if (g_system_error.load()) {
        std::cerr << "\n[!] Program aborted due to internal hardware/CUDA errors.\n";
        return 1;
    }

    if (g_failure.load()) {
        std::cout << "\n[!] no partition with p ≤ 10^8 found for n = " << g_failure_n.load() << "\n";
        return 1;
    }

    std::cout << "\n--- Verification Complete ---\n";
    std::cout << "All even numbers from " << START << " up to " << LIMIT << " satisfy Goldbach. ✓\n";
    std::cout << "Total computation time : " << (total_ms / 1000.0) << " seconds\n";
    std::cout << "Phase 2 fallbacks      : " << g_total_phase2_count.load() << "\n";

    if (opt.countPrimes) {
        std::cout << "pi(" << COUNT_HIGH << ") = "
                  << primes_below_start + g_prime_count_total.load() << "\n";
        if (!opt.countFile.empty()) {
            std::sort(g_segment_counts.begin(), g_segment_counts.end(),
                      [](const SegmentCount& a, const SegmentCount& b) { return a.lo < b.lo; });
            std::ofstream out(opt.countFile);
            for (const auto& c : g_segment_counts)
                out << c.lo << " " << c.hi << " " << c.count << "\n";
            out.close();
            if (!out) {
                std::cerr << "[!] ERROR: could not write --count-file " << opt.countFile << "\n";
                return 1;
            }
            std::cout << "[count] " << g_segment_counts.size()
                      << " segment counts written to " << opt.countFile << "\n";
        }
    }

    return 0;
}
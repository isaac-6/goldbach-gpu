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
static const uint64_t VRAM_SAFETY_MARGIN_BYTES = 50ULL * 1024 * 1024;


struct Options {
    bool async = false;
    uint64_t batchSize = 100000;
    bool showProgress = false;
    PrimeTest primeTest = PrimeTest::BPSW; 
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
// One thread per word of the d_verified bitset; unverified = 64 - popcount.
//
// The final word's bits past seg_even_count are masked to 1 here rather than
// assumed set. The transposed kernel does leave them set, but the scalar
// kernel sets bits one at a time and leaves them clear -- an unguarded
// popcount would then report those as unverified and trigger a spurious
// Phase 2 fallback on every scalar-path segment.
__global__ void count_unverified_kernel(
    const uint64_t* __restrict__ d_verified,
    uint64_t seg_even_count,
    uint32_t* __restrict__ d_unverified_count)
{
    uint64_t w = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t verified_words = (seg_even_count + 63) / 64;
    if (w >= verified_words) return;

    uint64_t word = d_verified[w];

    uint64_t valid = seg_even_count - w * 64;   // >= 1, since w < verified_words
    if (valid < 64) word |= (~0ULL << valid);

    uint32_t unverified = 64u - (uint32_t)__popcll(word);
    if (unverified) atomicAdd(d_unverified_count, unverified);
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

static bool cpu_optimized_check(uint64_t n, const std::vector<uint64_t>& cpu_primes,
                                 PrimeTest primeTest) {
    for (uint64_t p : cpu_primes) {
        if (p > n / 2) break;
        uint64_t q = n - p;
        if (q <= PHASE2_SIEVE_LIMIT) {
            if (std::binary_search(cpu_primes.begin(), cpu_primes.end(), q)) return true;
        } else {
            bool q_prime = (primeTest == PrimeTest::BPSW)
                           ? cpu_is_prime_bpsw(q)
                           : cpu_miller_rabin(q);
            if (q_prime) return true;
        }
    }
    return false;
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
    PrimeTest primeTest
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
        uint32_t* d_unverified_count = nullptr;

        CUDA_CHECK(cudaMalloc(&d_seg_bits, seg_bytes));
        // d_verified holds one bit per even number; SEG_SIZE is the largest
        // seg_even_count any segment can have.
        uint64_t max_verified_words = (SEG_SIZE + 63) / 64;
        CUDA_CHECK(cudaMalloc(&d_verified, max_verified_words * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_p_batch, P_BATCH * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_unverified_count, sizeof(uint32_t)));

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
            for (uint64_t bi = 0; bi < gpu_primes.size(); bi += P_BATCH) {
                uint64_t bsize = std::min(P_BATCH, (uint64_t)gpu_primes.size() - bi);
                CUDA_CHECK(cudaMemcpyAsync(d_p_batch, gpu_primes.data() + bi, bsize * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));

                launch_goldbach_phase1(
                    d_small, small_high, d_seg_bits, q_low, q_high,
                    seg_start, seg_even_count, d_p_batch, bsize, d_verified,
                    P_SMALL, THREADS_PER_BLOCK, stream);
                CUDA_CHECK(cudaGetLastError());
            }

            // C. Count Unverified
            uint32_t unverified_count = 0;
            CUDA_CHECK(cudaMemsetAsync(d_unverified_count, 0, sizeof(uint32_t), stream));

            uint32_t count_blocks = (uint32_t)((verified_words + 255) / 256);
            count_unverified_kernel<<<count_blocks, 256, 0, stream>>>(d_verified, seg_even_count, d_unverified_count);
            CUDA_CHECK(cudaMemcpyAsync(&unverified_count, d_unverified_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

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
                        
                        if (!cpu_optimized_check(n, cpu_primes, primeTest)) {
                            g_failure.store(true, std::memory_order_relaxed);
                            g_failure_n.store(n, std::memory_order_relaxed);
                            break;
                        }
                    }
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
        CUDA_CHECK(cudaFree(d_unverified_count));

    } catch (const std::exception& e) {
        safe_log("[!] FATAL ERROR in GPU ", device_id, " Worker: ", e.what());
        g_system_error.store(true, std::memory_order_relaxed);
    }
}

// -------------------------------------------------------
// Initialization & Hardware Check
// -------------------------------------------------------
void validate_hardware_and_limits(int use_gpus, uint64_t SEG_SIZE, uint64_t P_SMALL, uint64_t P_BATCH, size_t small_bytes) {
    uint64_t verified_bytes = ((SEG_SIZE + 63) / 64) * sizeof(uint64_t);
    uint64_t p_batch_bytes  = P_BATCH * sizeof(uint64_t);
    uint64_t max_q_span   = 2 * SEG_SIZE + P_SMALL;
    uint64_t max_odds     = (max_q_span + 1) / 2;
    uint64_t seg_words    = (max_odds + 63) / 64;
    uint64_t seg_bytes    = (seg_words + 1) * sizeof(uint64_t);

    uint64_t total_required = verified_bytes + p_batch_bytes + seg_bytes + small_bytes + VRAM_SAFETY_MARGIN_BYTES;

    // Validate CUDA Grid Sizes
    uint64_t num_tiles = (max_odds + TILE_ODDS - 1) / TILE_ODDS;
    uint64_t blocks = (SEG_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    if (num_tiles > UINT32_MAX || blocks > UINT32_MAX) {
        std::cerr << "[!] ERROR: Segment size too large. Grid dimensions exceed uint32_t limit.\n";
        std::cerr << "    num_tiles: " << num_tiles << " | blocks: " << blocks << "\n";
        std::cerr << "    Reduce SEG_SIZE or P_SMALL.\n";
        std::exit(1);
    }

    // Validate VRAM for all selected devices
    for (int i = 0; i < use_gpus; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        if (total_required > prop.totalGlobalMem) {
            std::cerr << "\n[!] ERROR: GPU " << i << " (" << prop.name << ") has insufficient VRAM.\n";
            std::cerr << "    Required: " << total_required / (1024*1024) << " MB\n";
            std::cerr << "    Available: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
            std::cerr << "    Reduce SEG_SIZE or use a smaller LIMIT.\n";
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
              << "  " << prog << " <LIMIT> [--seg-size=N] [--p-small=N][--gpus=N]\n\n"
              << "Required:\n"
              << "  LIMIT            Max even integer to check\n\n"
              << "Optional:\n"
              << "  --seg-size=N     Even integers per segment (default: 10,000,000)\n"
              << "  --p-small=N      GPU prime search bound (max: 4,000,000,000)\n"
              << "  --batch-size=N   Primes per GPU batch (default: 100000)\n"
              << "  --gpus=N         Number of GPUs to use (default: 1 | all: -1)\n"
              << "  --start=N        Starting number for verification (default: 4)\n"
              << "  --primetest=X    Primality test: BPSW (default) or MR\n"
              << "  --progress       Show real-time progress updates\n"
              << "  -h, --help       Show this help message\n";
}

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 0; }

    Options opt;
    uint64_t LIMIT = 0;
    uint64_t SEG_SIZE = 10'000'000ULL;
    uint64_t P_SMALL = 1'000'000ULL;
    uint64_t START = 4; // Default starting point
    int requested_gpus = 1;

    std::vector<std::string> positional;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
        if (arg == "--async") { opt.async = true; continue; }
        if (arg == "--progress") { opt.showProgress = true; continue; }
        if (arg.rfind("--batch-size=", 0) == 0) { opt.batchSize = std::stoull(arg.substr(13)); continue; }
        if (arg.rfind("--gpus=", 0) == 0) { requested_gpus = std::stoi(arg.substr(7)); continue; }
        if (arg.rfind("--seg-size=", 0) == 0) { SEG_SIZE = std::stoull(arg.substr(11)); continue; }
        if (arg.rfind("--p-small=", 0) == 0) { P_SMALL = std::stoull(arg.substr(10)); continue; }
        if (arg.rfind("--start=", 0) == 0) { START = std::stoull(arg.substr(8)); continue; }
        if (arg.rfind("--primetest=", 0) == 0) {
            std::string mode = arg.substr(12);
            if (mode == "MR" || mode == "mr") {
                opt.primeTest = PrimeTest::MillerRabin;
            } else if (mode == "BPSW" || mode == "bpsw") {
                opt.primeTest = PrimeTest::BPSW;
            } else {
                std::cerr << "Unknown --primetest value: " << mode
                          << "  (use MR or BPSW)\n";
                return 1;
            }
            continue;
        }
        positional.push_back(arg);
    }

    try {
        if (positional.size() >= 1) LIMIT = std::stoull(positional[0]);
        if (positional.size() >= 2) SEG_SIZE = std::stoull(positional[1]);
        if (positional.size() >= 3) P_SMALL = std::stoull(positional[2]);
    } catch (...) {
        std::cerr << "Error: Invalid numeric argument.\n"; return 1;
    }

    if (LIMIT < 4) { std::cerr << "Error: LIMIT must be >= 4.\n"; return 1; }
    if (LIMIT % 2 != 0) LIMIT--;
    if (SEG_SIZE == 0 || SEG_SIZE % 2 != 0) { std::cerr << "Error: SEG_SIZE must be even and > 0.\n"; return 1; }
    
    const uint64_t MAX_P_SMALL = 4'000'000'000ULL;
    if (P_SMALL > MAX_P_SMALL) {
        std::cerr << "Error: P_SMALL must be <= " << MAX_P_SMALL << " to prevent mathematical overflow.\n";
        return 1;
    }
    if (P_SMALL > LIMIT) P_SMALL = LIMIT;

    if (START < 4) START = 4;
    if (START % 2 != 0) START++; // Force it to be even
    if (START >= LIMIT) {
        std::cerr << "Error: START must be less than LIMIT.\n";
        return 1;
    }

    // Performance warnings
    if (P_SMALL > LIMIT) {
        std::cout << "[Info] P_SMALL (" << P_SMALL << ") > LIMIT (" << LIMIT 
                << "), adjusting P_SMALL to LIMIT.\n";
        P_SMALL = LIMIT;
    }
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

    // Fail-Fast Validations
    validate_hardware_and_limits(use_gpus, SEG_SIZE, P_SMALL, opt.batchSize, small_bytes);

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

    std::cout << "Pre-generating CPU primes up to " << PHASE2_SIEVE_LIMIT << "...\n";
    std::vector<uint64_t> cpu_primes = generate_cpu_primes(PHASE2_SIEVE_LIMIT);

    auto t1 = now();
    std::cout << "Initialization completed in " 
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms.\n\n";

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
        uint64_t total_even_to_check = (LIMIT - START) / 2 + 1;
        progress_running.store(true);
        
        progress_thread = std::thread([&]() {
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
            opt.primeTest
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
        std::cout << "\n[!] Goldbach FAILED at n = " << g_failure_n.load() << "\n";
        return 1;
    }

    std::cout << "\n--- Verification Complete ---\n";
    std::cout << "All even numbers from " << START << " up to " << LIMIT << " satisfy Goldbach. ✓\n";
    std::cout << "Total computation time : " << (total_ms / 1000.0) << " seconds\n";
    std::cout << "Phase 2 fallbacks      : " << g_total_phase2_count.load() << "\n";

    return 0;
}
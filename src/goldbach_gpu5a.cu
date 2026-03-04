// goldbach_multi_gpu_prod.cu
// Production-grade Multi-GPU Goldbach range verifier.
// Mathematically sound up to 10^19.

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>
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
static std::atomic<uint64_t> g_next_segment_start{4};
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
#define TILE_ODDS 32768
static const int THREADS_PER_BLOCK = 256;
static const uint64_t VRAM_SAFETY_MARGIN_BYTES = 50ULL * 1024 * 1024;

struct Options {
    bool async = false;
    uint64_t batchSize = 100000;
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
// GPU Kernel Device Functions (Mathematically verified)
// -------------------------------------------------------
__device__ uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)((__uint128_t)a * b % m);
}

__device__ uint64_t powmod64(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = mulmod64(result, base, mod);
        base = mulmod64(base, base, mod);
        exp >>= 1;
    }
    return result;
}

__device__ bool miller_rabin_witness(uint64_t n, uint64_t a, uint64_t d, uint64_t r) {
    uint64_t x = powmod64(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (uint64_t i = 0; i < r - 1; i++) {
        x = mulmod64(x, x, n);
        if (x == n - 1) return true;
    }
    return false;
}

__device__ bool gpu_is_prime_miller_rabin(uint64_t n) {
    if (n < 2)  return false;
    if (n == 2 || n == 3) return true;
    if ((n & 1) == 0) return false;

    uint64_t d = n - 1, r = 0;
    while ((d & 1) == 0) { d >>= 1; r++; }

    const uint64_t witnesses[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (int i = 0; i < 12; i++) {
        if (witnesses[i] >= n) continue;
        if (!miller_rabin_witness(n, witnesses[i], d, r)) return false;
    }
    return true;
}

__device__ bool is_prime_q(
    uint64_t q, const uint64_t* __restrict__ d_small, uint64_t small_high,
    const uint64_t* __restrict__ d_seg_bits, uint64_t q_low, uint64_t q_high)
{
    if (q < 2)  return false;
    if (q == 2) return true;
    if ((q & 1) == 0) return false;

    if (q <= small_high) {
        uint64_t bit_pos  = (q - 3) / 2;
        return (d_small[bit_pos / 64] >> (bit_pos % 64)) & 1ULL;
    }

    if (q >= q_low && q <= q_high) {
        uint64_t bit_pos  = (q - q_low) / 2;
        return (d_seg_bits[bit_pos / 64] >> (bit_pos % 64)) & 1ULL;
    }

    return gpu_is_prime_miller_rabin(q);
}

// -------------------------------------------------------
// GPU Kernels
// -------------------------------------------------------
__global__ void count_unverified_kernel(
    const uint8_t* __restrict__ d_verified,
    uint64_t seg_even_count,
    uint32_t* __restrict__ d_unverified_count)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= seg_even_count) return;

    if (d_verified[tid] == 0) {
        atomicAdd(d_unverified_count, 1u);
    }
}

// RESTORED 5c EXACT: Overflow-safe tiled sieve
__global__ void tiled_sieve_segment_kernel(
    uint64_t        q_low,
    uint64_t        q_high,
    const uint64_t* __restrict__ d_small_primes,
    uint64_t        small_prime_count,
    uint64_t*       __restrict__ d_seg_bits)
{
    extern __shared__ uint64_t sh_tile[]; 

    uint64_t num_odds = (q_high - q_low) / 2 + 1;
    uint64_t num_tiles = (num_odds + TILE_ODDS - 1) / TILE_ODDS;

    uint64_t tile_id = blockIdx.x;
    if (tile_id >= num_tiles) return;

    uint64_t tile_bit_offset = tile_id * TILE_ODDS;
    uint64_t tile_odd_start = tile_bit_offset;
    uint64_t tile_odd_end   = min(tile_odd_start + TILE_ODDS, num_odds);
    uint64_t tile_odd_count = tile_odd_end - tile_odd_start;
    uint64_t tile_word_count = (tile_odd_count + 63) / 64;

    for (uint64_t w = threadIdx.x; w < tile_word_count; w += blockDim.x) {
        sh_tile[w] = ~0ULL;
    }
    __syncthreads();

    for (uint64_t pi = threadIdx.x; pi < small_prime_count; pi += blockDim.x) {
        uint64_t p = d_small_primes[pi];
        if (p < 3) continue;          
        if (p > q_high / p) continue; // OVERFLOW-SAFE

        uint64_t first = (q_low + p - 1) / p * p;
        if ((first & 1) == 0) first += p;
        
        if (p <= q_high / p && first < p * p) first = p * p;
        if ((first & 1) == 0) first += p;
        if (first > q_high) continue;

        uint64_t first_bit_offset = first - q_low;
        int64_t first_bit = (int64_t)(first_bit_offset / 2);
        if (first_bit >= (int64_t)tile_odd_end) continue;

        if (first_bit < (int64_t)tile_odd_start) {
            int64_t delta_bits = tile_odd_start - first_bit;
            int64_t steps = (delta_bits + (int64_t)p - 1) / (int64_t)p;
            // OVERFLOW-SAFE: Prevent INT64_MAX overflow during step calc
            if (steps > 0 && (int64_t)p > INT64_MAX / steps) continue;
            first_bit += steps * (int64_t)p;
        }

        for (int64_t bit = first_bit; bit < (int64_t)tile_odd_end; bit += (int64_t)p) {
            uint64_t local_bit = (uint64_t)(bit - tile_odd_start);
            sh_tile[local_bit / 64] &= ~(1ULL << (local_bit % 64));
        }
    }
    __syncthreads();

    uint64_t global_word_offset = tile_odd_start / 64;
    for (uint64_t w = threadIdx.x; w < tile_word_count; w += blockDim.x) {
        d_seg_bits[global_word_offset + w] = sh_tile[w];
    }
}

__global__ void goldbach_phase1_kernel(
    const uint64_t* __restrict__ d_small, uint64_t small_high,
    const uint64_t* __restrict__ d_seg_bits, uint64_t q_low, uint64_t q_high,
    uint64_t seg_even_start, uint64_t seg_even_count,
    const uint64_t* __restrict__ p_batch, uint64_t p_batch_size,
    uint8_t* __restrict__ d_verified)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= seg_even_count) return;
    if (d_verified[tid]) return; // Early return monotonic safety

    uint64_t n = seg_even_start + tid * 2;

    for (uint64_t i = 0; i < p_batch_size; i++) {
        uint64_t p = p_batch[i];
        if (p > n / 2) break; 
        uint64_t q = n - p;

        if (is_prime_q(q, d_small, small_high, d_seg_bits, q_low, q_high)) {
            d_verified[tid] = 1;
            return;
        }
    }
}

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

static bool cpu_miller_rabin(uint64_t n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;
    
    uint64_t d = n - 1, r = 0;
    while ((d & 1) == 0) { d >>= 1; r++; }
    
    const uint64_t witnesses[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (int i = 0; i < 12; i++) {
        if (witnesses[i] >= n) continue;
        uint64_t x = 1, base = witnesses[i] % n;
        uint64_t exp = d;
        while (exp > 0) {
            if (exp & 1) x = (uint64_t)((__uint128_t)x * base % n);
            base = (uint64_t)((__uint128_t)base * base % n);
            exp >>= 1;
        }
        if (x == 1 || x == n - 1) continue;
        bool witness = false;
        for (uint64_t j = 0; j < r - 1; j++) {
            x = (uint64_t)((__uint128_t)x * x % n);
            if (x == n - 1) { witness = true; break; }
        }
        if (!witness) return false;
    }
    return true;
}

static bool cpu_optimized_check(uint64_t n, const std::vector<uint64_t>& cpu_primes) {
    for (uint64_t p : cpu_primes) {
        if (p > n / 2) break;
        uint64_t q = n - p;
        if (q <= PHASE2_SIEVE_LIMIT) {
            if (std::binary_search(cpu_primes.begin(), cpu_primes.end(), q)) return true;
        } else {
            if (cpu_miller_rabin(q)) return true;
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
    const Options& opt)
{
    try {
        CUDA_CHECK(cudaSetDevice(device_id));
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Memory Allocations
        uint64_t* d_small = nullptr;
        CUDA_CHECK(cudaMalloc(&d_small, small_bytes));
        CUDA_CHECK(cudaMemcpyAsync(d_small, small_bitset.data(), small_bytes, cudaMemcpyHostToDevice, stream));

        uint64_t small_prime_count = small_primes.size();
        uint64_t* d_small_primes = nullptr;
        CUDA_CHECK(cudaMalloc(&d_small_primes, small_prime_count * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyAsync(d_small_primes, small_primes.data(), small_prime_count * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));

        uint64_t max_q_span = 2 * SEG_SIZE + P_SMALL;
        uint64_t max_odds = (max_q_span + 1) / 2;
        uint64_t seg_words = (max_odds + 63) / 64;
        size_t seg_bytes = seg_words * sizeof(uint64_t);

        uint64_t* d_seg_bits = nullptr;
        uint8_t*  d_verified = nullptr;
        uint64_t* d_p_batch  = nullptr;
        uint32_t* d_unverified_count = nullptr;

        CUDA_CHECK(cudaMalloc(&d_seg_bits, seg_bytes));
        CUDA_CHECK(cudaMalloc(&d_verified, SEG_SIZE));
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
            uint32_t num_tiles = (uint32_t)((num_odds + TILE_ODDS - 1) / TILE_ODDS);
            size_t shared_bytes = (TILE_ODDS / 64) * sizeof(uint64_t);

            // A. Sieve Segment
            tiled_sieve_segment_kernel<<<num_tiles, THREADS_PER_BLOCK, shared_bytes, stream>>>(
                q_low, q_high, d_small_primes, small_prime_count, d_seg_bits);
            CUDA_CHECK(cudaGetLastError());

            // B. Phase 1 Verification Batches
            CUDA_CHECK(cudaMemsetAsync(d_verified, 0, seg_even_count, stream));
            uint32_t blocks = (uint32_t)((seg_even_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

            for (uint64_t bi = 0; bi < gpu_primes.size(); bi += P_BATCH) {
                uint64_t bsize = std::min(P_BATCH, (uint64_t)gpu_primes.size() - bi);
                CUDA_CHECK(cudaMemcpyAsync(d_p_batch, gpu_primes.data() + bi, bsize * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));

                goldbach_phase1_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    d_small, small_high, d_seg_bits, q_low, q_high,
                    seg_start, seg_even_count, d_p_batch, bsize, d_verified);
                CUDA_CHECK(cudaGetLastError());
            }

            // C. Count Unverified
            uint32_t unverified_count = 0;
            CUDA_CHECK(cudaMemsetAsync(d_unverified_count, 0, sizeof(uint32_t), stream));

            uint32_t count_blocks = (uint32_t)((seg_even_count + 255) / 256);
            count_unverified_kernel<<<count_blocks, 256, 0, stream>>>(d_verified, seg_even_count, d_unverified_count);
            CUDA_CHECK(cudaMemcpyAsync(&unverified_count, d_unverified_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // D. CPU Phase 2 Processing
            if (unverified_count > 0) {
                std::vector<uint8_t> verified(seg_even_count);
                CUDA_CHECK(cudaMemcpy(verified.data(), d_verified, seg_even_count, cudaMemcpyDeviceToHost));

                for (uint64_t i = 0; i < seg_even_count; i++) {
                    if (!verified[i]) {
                        uint64_t n = seg_start + i * 2;
                        g_total_phase2_count.fetch_add(1, std::memory_order_relaxed);
                        safe_log("[GPU ", device_id, "] Phase 2 fallback for n = ", n, "...");
                        
                        if (!cpu_optimized_check(n, cpu_primes)) {
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
    uint64_t verified_bytes = SEG_SIZE;
    uint64_t p_batch_bytes  = P_BATCH * sizeof(uint64_t);
    uint64_t max_q_span   = 2 * SEG_SIZE + P_SMALL;
    uint64_t max_odds     = (max_q_span + 1) / 2;
    uint64_t seg_words    = (max_odds + 63) / 64;
    uint64_t seg_bytes    = seg_words * sizeof(uint64_t);

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
              << "  --gpus=N         Number of GPUs to use (default: all available)\n"
              << "  -h, --help       Show this help message\n";
}

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 0; }

    Options opt;
    uint64_t LIMIT = 0;
    uint64_t SEG_SIZE = 10'000'000ULL;
    uint64_t P_SMALL = 1'000'000ULL;
    int requested_gpus = -1;

    std::vector<std::string> positional;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
        if (arg == "--async") { opt.async = true; continue; }
        if (arg.rfind("--batch-size=", 0) == 0) { opt.batchSize = std::stoull(arg.substr(13)); continue; }
        if (arg.rfind("--gpus=", 0) == 0) { requested_gpus = std::stoi(arg.substr(7)); continue; }
        if (arg.rfind("--seg-size=", 0) == 0) { SEG_SIZE = std::stoull(arg.substr(11)); continue; }
        if (arg.rfind("--p-small=", 0) == 0) { P_SMALL = std::stoull(arg.substr(10)); continue; }
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

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) { std::cerr << "No CUDA devices found.\n"; return 1; }

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

    std::cout << "\nGoldbach Multi-GPU Verifier (Limit: " << LIMIT << ")\n";
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

    auto t_main_start = now();

    // Launch Worker Threads
    std::vector<std::thread> workers;
    for (int g = 0; g < use_gpus; ++g) {
        workers.emplace_back(
            run_gpu_worker, g, LIMIT, SEG_SIZE, P_SMALL, opt.batchSize,
            small_high, small_bytes, std::cref(small_bitset),
            std::cref(small_primes), std::cref(gpu_primes), std::cref(cpu_primes),
            std::cref(opt)
        );
    }

    // // Progress Monitor (Optional enhancement, runs on main thread)
    // uint64_t total_even_to_check = (LIMIT - 4) / 2 + 1;
    // while (!g_failure.load() && !g_system_error.load()) {
    //     uint64_t processed = g_total_processed.load(std::memory_order_relaxed);
    //     if (processed >= total_even_to_check) break;
    //     std::this_thread::sleep_for(std::chrono::milliseconds(500));
    // }

    for (auto& t : workers) {
        if (t.joinable()) t.join();
    }

    auto t_main_end = now();
    double total_ms = std::chrono::duration<double, std::milli>(t_main_end - t_main_start).count();

    if (g_system_error.load()) {
        std::cerr << "\n[!] Program aborted due to internal hardware/CUDA errors.\n";
        return 1;
    }

    if (g_failure.load()) {
        std::cout << "\n[!] Goldbach FAILED at n = " << g_failure_n.load() << "\n";
        return 1;
    }

    std::cout << "\n--- Verification Complete ---\n";
    std::cout << "All even numbers up to " << LIMIT << " satisfy Goldbach. ✓\n";
    std::cout << "Total computation time : " << (total_ms / 1000.0) << " seconds\n";
    std::cout << "Phase 2 fallbacks      : " << g_total_phase2_count.load() << "\n";

    return 0;
}
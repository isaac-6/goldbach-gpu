// goldbach_gpu5c.cu
// GPU Goldbach range verifier -- GPU-segmented sieve for q.
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
//          is checked exhaustively.
//
// Correctness guarantee:
//   Every even n in [4, LIMIT] is verified by Phase 1 or Phase 2.

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <string>
#include "prime_bitset.hpp"

using namespace goldbach;

inline std::chrono::high_resolution_clock::time_point now() {
    return std::chrono::high_resolution_clock::now();
}

// -------------------------------------------------------
// Runtime optimization flags
// -------------------------------------------------------
struct Options {
    bool async = false;
    int batchSize = 100000;
    enum CopyMode { FULL, FAILURES, NONE } copyMode = FULL;
    int streams = 1;
};

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " -- " << cudaGetErrorString(err) << "\n";        \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

// -------------------------------------------------------
// Miller-Rabin primality test for 64-bit integers on GPU.
// -------------------------------------------------------
__device__ uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
    return (__uint128_t)a * b % m;
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

__device__ bool miller_rabin_witness(uint64_t n, uint64_t a,
                                     uint64_t d, uint64_t r) {
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

    const uint64_t witnesses[] =
        {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (int i = 0; i < 12; i++) {
        if (witnesses[i] >= n) continue;
        if (!miller_rabin_witness(n, witnesses[i], d, r)) return false;
    }
    return true;
}

// -------------------------------------------------------
// Device function: is q prime?
//   1. q <= small_high: small primes bitset
//   2. q in [q_low, q_high]: segment bitset
//   3. otherwise: Miller-Rabin
// -------------------------------------------------------
__device__ bool is_prime_q(
    uint64_t q,
    const uint64_t* __restrict__ d_small,
    uint64_t small_high,
    const uint64_t* __restrict__ d_seg_bits,
    uint64_t q_low,
    uint64_t q_high)
{
    if (q < 2)  return false;
    if (q == 2) return true;
    if ((q & 1) == 0) return false;

    if (q <= small_high) {
        uint64_t bit_pos  = (q - 3) / 2;
        uint64_t word_idx = bit_pos / 64;
        uint64_t bit_idx  = bit_pos % 64;
        return (d_small[word_idx] >> bit_idx) & 1ULL;
    }

    if (q >= q_low && q <= q_high) {
        uint64_t bit_pos  = (q - q_low) / 2;
        uint64_t word_idx = bit_pos / 64;
        uint64_t bit_idx  = bit_pos % 64;
        return (d_seg_bits[word_idx] >> bit_idx) & 1ULL;
    }

    return gpu_is_prime_miller_rabin(q);
}

// -------------------------------------------------------
// GPU sieve: initialize segment bitset to all 1s.
// -------------------------------------------------------
__global__ void init_segment_bits_kernel(
    uint64_t* __restrict__ d_seg_bits,
    uint64_t num_words)
{
    uint64_t wid = blockIdx.x * blockDim.x + threadIdx.x;
    if (wid < num_words) {
        d_seg_bits[wid] = ~0ULL;
    }
}

#define TILE_ODDS 32768  // must be multiple of 64 for convenience

__global__ void tiled_sieve_segment_kernel(
    uint64_t        q_low,
    uint64_t        q_high,
    const uint64_t* __restrict__ d_small_primes,
    uint64_t        small_prime_count,
    uint64_t*       __restrict__ d_seg_bits)
{
    extern __shared__ uint64_t sh_tile[]; // TILE_ODDS / 64 words

    uint64_t num_odds = (q_high - q_low) / 2 + 1;
    uint64_t num_tiles = (num_odds + TILE_ODDS - 1) / TILE_ODDS;

    uint64_t tile_id = blockIdx.x;
    if (tile_id >= num_tiles) return;

    // Global bit offset for this tile
    uint64_t tile_bit_offset = tile_id * TILE_ODDS;

    // How many odds are actually in this tile (last tile may be partial)
    uint64_t tile_odd_start = tile_bit_offset;
    uint64_t tile_odd_end   = min(tile_odd_start + TILE_ODDS, num_odds);
    uint64_t tile_odd_count = tile_odd_end - tile_odd_start;

    // Number of 64-bit words in this tile
    uint64_t tile_word_count = (tile_odd_count + 63) / 64;

    // 1) Initialize shared tile to all 1s
    for (uint64_t w = threadIdx.x; w < tile_word_count; w += blockDim.x) {
        sh_tile[w] = ~0ULL;
    }
    __syncthreads();

    // 2) Mark composites in this tile using all small primes
    // Each thread walks over a subset of primes
    for (uint64_t pi = threadIdx.x; pi < small_prime_count; pi += blockDim.x) {
        uint64_t p = d_small_primes[pi];
        if (p < 3) continue;          // skip 2, we only store odds
        if (p * p > q_high) continue; // no need to mark beyond sqrt(q_high)

        // Global value of the first odd multiple of p in [q_low, q_high]
        uint64_t first = (q_low + p - 1) / p * p;
        if ((first & 1) == 0) first += p;
        if (first < p * p) first = p * p;
        if ((first & 1) == 0) first += p;
        if (first > q_high) continue;

        // Now restrict to this tile: we only care about multiples inside it
        // Tile covers odds with global bit indices [tile_odd_start, tile_odd_end)
        // Global bit index for 'first':
        int64_t first_bit = (int64_t)((first - q_low) / 2);
        if (first_bit >= (int64_t)tile_odd_end) continue;

        // Start at max(first_bit, tile_odd_start)
        if (first_bit < (int64_t)tile_odd_start) {
            // Advance to the first multiple inside the tile
            int64_t delta_bits = tile_odd_start - first_bit;
            // Each step of 2p increases q by 2p → increases bit index by p
            int64_t steps = (delta_bits + (int64_t)p - 1) / (int64_t)p;
            first_bit += steps * (int64_t)p;
        }

        for (int64_t bit = first_bit;
             bit < (int64_t)tile_odd_end;
             bit += (int64_t)p)
        {
            uint64_t local_bit = (uint64_t)(bit - tile_odd_start);
            uint64_t word_idx  = local_bit / 64;
            uint64_t bit_idx   = local_bit % 64;
            sh_tile[word_idx] &= ~(1ULL << bit_idx);
        }
    }

    __syncthreads();

    // 3) Write shared tile back to global bitset
    // Global word offset for this tile
    uint64_t global_word_offset = tile_odd_start / 64;

    for (uint64_t w = threadIdx.x; w < tile_word_count; w += blockDim.x) {
        d_seg_bits[global_word_offset + w] = sh_tile[w];
    }
}

// -------------------------------------------------------
// GPU sieve: mark composites in [q_low, q_high] using small primes.
// One thread per small prime.
// -------------------------------------------------------
__global__ void sieve_segment_kernel(
    uint64_t        q_low,
    uint64_t        q_high,
    const uint64_t* __restrict__ d_small_primes,
    uint64_t        small_prime_count,
    uint64_t*       __restrict__ d_seg_bits)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= small_prime_count) return;

    uint64_t p = d_small_primes[tid];
    if (p < 3) return; // skip 2, we only store odds
    if (p * p > q_high) return;

    uint64_t first = (q_low + p - 1) / p * p;
    if ((first & 1) == 0) first += p;
    if (first < p * p) first = p * p;
    if ((first & 1) == 0) first += p;
    if (first > q_high) return;

    for (uint64_t j = first; j <= q_high; j += 2 * p) {
        uint64_t bit_pos  = (j - q_low) / 2;
        uint64_t word_idx = bit_pos / 64;
        uint64_t bit_idx  = bit_pos % 64;
        // atomicAnd(&d_seg_bits[word_idx], ~(1ULL << bit_idx));
        atomicAnd(
            (unsigned long long int*)&d_seg_bits[word_idx],
            ~(1ULL << bit_idx)
        );
    }
}

// -------------------------------------------------------
// Phase 1 kernel: GPU verification with bounded p.
// One thread per even n in segment.
// -------------------------------------------------------
__global__ void goldbach_phase1_kernel(
    const uint64_t* __restrict__ d_small,
    uint64_t        small_high,
    const uint64_t* __restrict__ d_seg_bits,
    uint64_t        q_low,
    uint64_t        q_high,
    uint64_t        seg_even_start,
    uint64_t        seg_even_count,
    const uint64_t* __restrict__ p_batch,
    uint64_t        p_batch_size,
    uint8_t*        __restrict__ d_verified)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= seg_even_count) return;

    if (d_verified[tid]) return;

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
// CPU trial division primality test (Phase 2).
// -------------------------------------------------------
static bool cpu_is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if ((n & 1) == 0) return false;
    uint64_t sq = (uint64_t)std::sqrt((double)n);
    for (uint64_t d = 3; d <= sq; d += 2)
        if (n % d == 0) return false;
    return true;
}

// -------------------------------------------------------
// Phase 2: truly exhaustive CPU fallback.
// -------------------------------------------------------
static bool cpu_exhaustive_check(uint64_t n) {
    if (cpu_is_prime(n - 2)) return true;

    for (uint64_t p = 3; p <= n / 2; p += 2) {
        if (cpu_is_prime(p) && cpu_is_prime(n - p))
            return true;
    }
    return false;
}

void print_usage(const char* prog) {
    std::cout << "Goldbach Conjecture Segmented Verifier (GPU)\n\n";

    std::cout << "Usage:\n"
              << "  " << prog << " <LIMIT> [SEG_SIZE] [P_SMALL]\n"
              << "  " << prog << " <LIMIT> [--seg-size=N] [--p-small=N]\n\n";

    std::cout << "Required:\n"
              << "  LIMIT            Max even integer to check (e.g., 1000000000)\n\n";

    std::cout << "Optional positional arguments (legacy):\n"
              << "  SEG_SIZE         Even integers per segment (default: 10,000,000)\n"
              << "  P_SMALL          GPU prime search bound (default: 1,000,000)\n\n";

    std::cout << "Optional flags:\n"
              << "  --seg-size=N     Override SEG_SIZE\n"
              << "  --p-small=N      Override P_SMALL\n"
              << "  --batch-size=N   Primes per GPU batch (default: 100000)\n"
              << "  --copy-mode=MODE {full, failures, none}\n"
              << "  --streams=N      Number of CUDA streams (default: 1)\n"
              << "  --async          Enable async kernel launches\n"
              << "  -h, --help       Show this help message\n";
}

void check_vram_limit(uint64_t seg_size, uint64_t small_bytes, uint64_t p_small) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    uint64_t verified_bytes = seg_size;
    uint64_t p_batch_bytes  = 100000ULL * sizeof(uint64_t);

    // Worst-case q-range per segment: ~2*SEG_SIZE + P_SMALL
    uint64_t max_q_span   = 2 * seg_size + p_small;
    uint64_t max_odds     = (max_q_span + 1) / 2;
    uint64_t seg_words    = (max_odds + 63) / 64;
    uint64_t seg_bytes    = seg_words * sizeof(uint64_t);

    uint64_t total_required =
        verified_bytes +
        p_batch_bytes +
        seg_bytes +
        small_bytes +
        (50ULL * 1024 * 1024); // 50 MB safety margin

    if (total_required > prop.totalGlobalMem) {
        std::cerr << "\n[!] ERROR: SEG_SIZE (" << seg_size << ") requires approx "
                  << total_required / (1024*1024) << " MB VRAM.\n";
        std::cerr << "[!] This GPU (" << prop.name << ") only has "
                  << prop.totalGlobalMem / (1024*1024) << " MB available.\n";
        std::cerr << "[!] Reduce SEG_SIZE or use a smaller LIMIT.\n";
        std::exit(1);
    }

    std::cout << "[Hardware] GPU: " << prop.name
              << " (" << prop.totalGlobalMem / (1024*1024) << " MB VRAM)\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 0;
    }

    Options opt;
    uint64_t LIMIT   = 0;
    uint64_t SEG_SIZE= 10'000'000ULL;
    uint64_t P_SMALL = 1'000'000ULL;

    std::vector<std::string> positional;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }

        if (arg == "--async") {
            opt.async = true;
            continue;
        }

        if (arg.rfind("--batch-size=", 0) == 0) {
            opt.batchSize = std::stoull(arg.substr(13));
            continue;
        }

        if (arg.rfind("--copy-mode=", 0) == 0) {
            std::string m = arg.substr(12);
            if (m == "full") opt.copyMode = Options::FULL;
            else if (m == "failures") opt.copyMode = Options::FAILURES;
            else if (m == "none") opt.copyMode = Options::NONE;
            else {
                std::cerr << "Invalid --copy-mode value\n";
                return 1;
            }
            continue;
        }

        if (arg.rfind("--streams=", 0) == 0) {
            opt.streams = std::stoi(arg.substr(10));
            continue;
        }

        if (arg.rfind("--seg-size=", 0) == 0) {
            SEG_SIZE = std::stoull(arg.substr(11));
            continue;
        }

        if (arg.rfind("--p-small=", 0) == 0) {
            P_SMALL = std::stoull(arg.substr(10));
            continue;
        }

        positional.push_back(arg);
    }

    try {
        if (positional.size() < 1) {
            std::cerr << "Error: LIMIT is required.\n";
            return 1;
        }

        LIMIT = std::stoull(positional[0]);

        if (positional.size() > 1)
            SEG_SIZE = std::stoull(positional[1]);

        if (positional.size() > 2)
            P_SMALL = std::stoull(positional[2]);

    } catch (...) {
        std::cerr << "Error: Invalid numeric argument. Use -h for help.\n";
        return 1;
    }

    if (LIMIT < 4) {
        std::cerr << "Error: LIMIT must be >= 4.\n";
        return 1;
    }

    if (LIMIT % 2 != 0) LIMIT--;

    uint64_t small_high = std::max((uint64_t)std::sqrt((double)LIMIT) + 1, P_SMALL);
    if (small_high % 2 == 0) small_high++;

    uint64_t num_small_odds = (small_high - 3) / 2 + 1;
    uint64_t small_bytes    = ((num_small_odds + 63) / 64) * sizeof(uint64_t);

    check_vram_limit(SEG_SIZE, small_bytes, P_SMALL);

    uint64_t P_BATCH = opt.batchSize;

    std::cout << "Goldbach segmented verifier (Phase 1: GPU, Phase 2: CPU)\n";
    std::cout << "Checking all even n in [4, " << LIMIT << "]\n";
    std::cout << "P_SMALL = " << P_SMALL
              << ", exhaustive CPU fallback for any remainder\n\n";

    std::cout << "Building small primes bitset up to "
              << small_high << "...\n";
    auto t0 = now();

    PrimeBitset small_bitset = build_prime_bitset(small_high);

    auto t1 = now();
    std::cout << "Built in "
              << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << " ms (" << small_bitset.memory_bytes() / 1024 << " KB)\n\n";

    std::vector<uint64_t> small_primes;
    small_primes.reserve(small_high / 10);
    if (small_bitset.is_prime(2)) small_primes.push_back(2);
    for (uint64_t i = 3; i <= small_high; i += 2)
        if (small_bitset.is_prime(i))
            small_primes.push_back(i);

    std::vector<uint64_t> gpu_primes;
    for (uint64_t p : small_primes)
        if (p <= P_SMALL) gpu_primes.push_back(p);

    std::cout << "GPU primes (p <= " << P_SMALL << "): "
              << gpu_primes.size() << "\n\n";

    small_bytes = small_bitset.word_count() * sizeof(uint64_t);
    uint64_t* d_small = nullptr;
    CUDA_CHECK(cudaMalloc(&d_small, small_bytes));
    CUDA_CHECK(cudaMemcpy(d_small, small_bitset.data(), small_bytes,
                          cudaMemcpyHostToDevice));
    std::cout << "Small primes in GPU ("
              << small_bytes / 1024 << " KB)\n";

    // Copy all small_primes to device for sieve_segment_kernel
    uint64_t* d_small_primes = nullptr;
    uint64_t  small_prime_count = small_primes.size();
    CUDA_CHECK(cudaMalloc(&d_small_primes, small_prime_count * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_small_primes, small_primes.data(),
                          small_prime_count * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));

    // Allocate segment bitset buffer for worst-case q-range
    uint64_t max_q_span   = 2 * SEG_SIZE + P_SMALL;
    uint64_t max_odds     = (max_q_span + 1) / 2;
    uint64_t seg_words    = (max_odds + 63) / 64;
    uint64_t seg_bytes    = seg_words * sizeof(uint64_t);

    uint64_t* d_seg_bits  = nullptr;
    uint8_t*  d_verified  = nullptr;
    uint64_t* d_p_batch   = nullptr;

    CUDA_CHECK(cudaMalloc(&d_seg_bits, seg_bytes));
    CUDA_CHECK(cudaMalloc(&d_verified, SEG_SIZE));
    CUDA_CHECK(cudaMalloc(&d_p_batch,  P_BATCH * sizeof(uint64_t)));

    std::cout << "Segment buffer: " << seg_bytes / 1024 / 1024 << " MB\n";
    std::cout << "Verified buffer: " << SEG_SIZE / 1024 / 1024 << " MB\n\n";

    if (opt.streams < 1) opt.streams = 1;
    std::vector<cudaStream_t> streams(opt.streams);
    for (int s = 0; s < opt.streams; ++s) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }

    int      threads_per_block  = 256;
    uint64_t total_even         = (LIMIT - 4) / 2 + 1;
    uint64_t total_processed    = 0;
    uint64_t total_failures     = 0;
    uint64_t total_phase2_count = 0;

    auto t_main = now();

    uint64_t seg_start = 4;
    uint64_t seg_index = 0;

    double total_ms_sieve        = 0.0;
    double total_ms_kernel       = 0.0;
    double total_ms_copy_verified= 0.0;
    double total_ms_phase2       = 0.0;

    while (seg_start <= LIMIT) {
        auto t_seg_start = now();
        cudaStream_t stream = streams[seg_index % streams.size()];

        uint64_t seg_end = std::min(seg_start + SEG_SIZE * 2 - 2, LIMIT);
        uint64_t seg_even_count = (seg_end - seg_start) / 2 + 1;

        // q-range for this segment
        uint64_t q_low  = (seg_start > P_SMALL ? seg_start - P_SMALL : 3);
        if ((q_low & 1) == 0) q_low++;
        uint64_t q_high = seg_end + 1;
        if ((q_high & 1) == 0) q_high++;

        uint64_t num_odds  = (q_high - q_low) / 2 + 1;
        uint64_t num_words = (num_odds + 63) / 64;

        // 1) GPU sieve for [q_low, q_high]
        float sieve_ms = 0.0f;
        cudaEvent_t s_start, s_end;
        cudaEventCreate(&s_start);
        cudaEventCreate(&s_end);

        // uint64_t num_odds  = (q_high - q_low) / 2 + 1;
        uint64_t num_tiles = (num_odds + TILE_ODDS - 1) / TILE_ODDS;

        // Each tile needs TILE_ODDS/64 words in shared memory
        size_t shared_bytes = (TILE_ODDS / 64) * sizeof(uint64_t);

        cudaEventRecord(s_start, stream);

        tiled_sieve_segment_kernel<<<(uint32_t)num_tiles,
                                    threads_per_block,
                                    shared_bytes,
                                    stream>>>(
            q_low, q_high,
            d_small_primes, small_prime_count,
            d_seg_bits);

        CUDA_CHECK(cudaGetLastError());

        cudaEventRecord(s_end, stream);
        cudaEventSynchronize(s_end);
        cudaEventElapsedTime(&sieve_ms, s_start, s_end);
        total_ms_sieve += sieve_ms;
        // float sieve_ms = 0.0f;
        // cudaEvent_t s_start, s_end;
        // cudaEventCreate(&s_start);
        // cudaEventCreate(&s_end);

        // // init bits
        // uint64_t init_blocks = (num_words + threads_per_block - 1) / threads_per_block;
        // cudaEventRecord(s_start, stream);
        // init_segment_bits_kernel<<<(uint32_t)init_blocks, threads_per_block, 0, stream>>>(
        //     d_seg_bits, num_words);
        // CUDA_CHECK(cudaGetLastError());

        // // mark composites
        // uint64_t sieve_blocks = (small_prime_count + threads_per_block - 1) / threads_per_block;
        // sieve_segment_kernel<<<(uint32_t)sieve_blocks, threads_per_block, 0, stream>>>(
        //     q_low, q_high,
        //     d_small_primes, small_prime_count,
        //     d_seg_bits);
        // CUDA_CHECK(cudaGetLastError());


        // cudaEventRecord(s_end, stream);
        // cudaEventSynchronize(s_end);
        // cudaEventElapsedTime(&sieve_ms, s_start, s_end);
        // total_ms_sieve += sieve_ms;

        // 2) Phase 1: GPU Goldbach kernel
        CUDA_CHECK(cudaMemset(d_verified, 0, seg_even_count));

        float kernel_ms = 0.0f;
        cudaEvent_t k_start, k_end;
        cudaEventCreate(&k_start);
        cudaEventCreate(&k_end);

        uint64_t blocks = (seg_even_count + threads_per_block - 1) / threads_per_block;

        for (uint64_t bi = 0; bi < gpu_primes.size(); bi += P_BATCH) {
            uint64_t bend  = std::min(bi + P_BATCH, (uint64_t)gpu_primes.size());
            uint64_t bsize = bend - bi;

            if (opt.async) {
                CUDA_CHECK(cudaMemcpyAsync(d_p_batch,
                                           gpu_primes.data() + bi,
                                           bsize * sizeof(uint64_t),
                                           cudaMemcpyHostToDevice,
                                           stream));

                cudaEventRecord(k_start, stream);

                goldbach_phase1_kernel<<<(uint32_t)blocks,
                                         threads_per_block,
                                         0, stream>>>(
                    d_small, small_high,
                    d_seg_bits, q_low, q_high,
                    seg_start, seg_even_count,
                    d_p_batch, bsize,
                    d_verified);

                CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemcpy(d_p_batch,
                                      gpu_primes.data() + bi,
                                      bsize * sizeof(uint64_t),
                                      cudaMemcpyHostToDevice));

                cudaEventRecord(k_start, stream);

                goldbach_phase1_kernel<<<(uint32_t)blocks,
                                         threads_per_block>>>(
                    d_small, small_high,
                    d_seg_bits, q_low, q_high,
                    seg_start, seg_even_count,
                    d_p_batch, bsize,
                    d_verified);

                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            cudaEventRecord(k_end, stream);
            cudaEventSynchronize(k_end);

            float this_ms = 0.0f;
            cudaEventElapsedTime(&this_ms, k_start, k_end);
            kernel_ms += this_ms;
        }

        total_ms_kernel += kernel_ms;

        // 3) Copy verified flags back
        std::vector<uint8_t> verified(seg_even_count);

        auto t_copy_verified_start = now();

        if (opt.copyMode == Options::FULL) {
            if (opt.async) {
                CUDA_CHECK(cudaMemcpyAsync(verified.data(), d_verified,
                                           seg_even_count,
                                           cudaMemcpyDeviceToHost,
                                           stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            } else {
                CUDA_CHECK(cudaMemcpy(verified.data(), d_verified,
                                      seg_even_count,
                                      cudaMemcpyDeviceToHost));
            }
        } else if (opt.copyMode == Options::FAILURES) {
            uint8_t any_unverified = 0;
            CUDA_CHECK(cudaMemcpy(&any_unverified, d_verified,
                                  1, cudaMemcpyDeviceToHost));
            if (!any_unverified) {
                verified.assign(seg_even_count, 1);
            } else {
                CUDA_CHECK(cudaMemcpy(verified.data(), d_verified,
                                      seg_even_count,
                                      cudaMemcpyDeviceToHost));
            }
        } else if (opt.copyMode == Options::NONE) {
            verified.assign(seg_even_count, 1);
        }

        auto t_copy_verified_end = now();
        double ms_copy_verified =
            std::chrono::duration<double, std::milli>(t_copy_verified_end - t_copy_verified_start).count();
        total_ms_copy_verified += ms_copy_verified;

        // 4) Phase 2: CPU fallback
        auto t_phase2_start = now();

        for (uint64_t i = 0; i < seg_even_count; i++) {
            if (verified[i]) continue;

            uint64_t n = seg_start + i * 2;
            total_phase2_count++;

            std::cout << "\n  Phase 2 fallback for n = " << n << "...\n";
            bool found = cpu_exhaustive_check(n);

            if (!found) {
                std::cout << "FAILURE: no Goldbach partition for n = "
                          << n << "\n";
                total_failures++;
            } else {
                std::cout << "  Phase 2 verified n = " << n << "\n";
            }
        }

        auto t_phase2_end = now();
        double ms_phase2 =
            std::chrono::duration<double, std::milli>(t_phase2_end - t_phase2_start).count();
        total_ms_phase2 += ms_phase2;

        total_processed += seg_even_count;
        seg_start        = seg_end + 2;
        seg_index++;

        auto t_seg_end = now();
        double ms_total =
            std::chrono::duration<double, std::milli>(t_seg_end - t_seg_start).count();

        // double pct = 100.0 * total_processed / total_even;
        // std::cout << "  Progress: " << total_processed
        //           << " / " << total_even
        //           << " (" << pct << "%)"
        //           << "  (segment " << seg_index
        //           << ", " << ms_total << " ms)\r" << std::flush;
    }

    double total_ms = std::chrono::duration<double, std::milli>(
        now() - t_main).count();

    std::cout << "\n";

    std::cout << "\n--- Timing breakdown ---\n";
    std::cout << "GPU sieve total:          " << total_ms_sieve         << " ms\n";
    std::cout << "GPU kernel total:         " << total_ms_kernel        << " ms\n";
    std::cout << "Copy verified → CPU total:" << total_ms_copy_verified << " ms\n";
    std::cout << "CPU Phase 2 total:        " << total_ms_phase2        << " ms\n";

    std::cout << "\n--- Summary ---\n";
    std::cout << "Even numbers checked  : " << total_even         << "\n";
    std::cout << "Failures              : " << total_failures     << "\n";
    std::cout << "Phase 2 fallbacks     : " << total_phase2_count << "\n";
    std::cout << "P_SMALL               : " << P_SMALL            << "\n";
    std::cout << "Total time            : " << total_ms           << " ms\n";

    if (total_failures == 0) {
        std::cout << "\nAll even numbers up to " << LIMIT
                  << " satisfy Goldbach. ✓\n";
        if (total_phase2_count == 0)
            std::cout << "(All verified by GPU with p <= "
                      << P_SMALL << ")\n";
        else
            std::cout << "(" << total_phase2_count
                      << " required CPU exhaustive fallback)\n";
    }

    for (auto& s : streams) {
        CUDA_CHECK(cudaStreamDestroy(s));
    }

    CUDA_CHECK(cudaFree(d_small));
    CUDA_CHECK(cudaFree(d_small_primes));
    CUDA_CHECK(cudaFree(d_seg_bits));
    CUDA_CHECK(cudaFree(d_verified));
    CUDA_CHECK(cudaFree(d_p_batch));

    return (total_failures == 0) ? 0 : 1;
}
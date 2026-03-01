// goldbach_gpu3c_multi.cu
// Multi-worker, multi-GPU Goldbach segmented verifier.
//
// Architecture
// ------------
// Each worker thread owns one GPU device and a private set of device
// buffers (d_seg, d_p_batch, d_verified_*). Workers pull segments from
// a shared work queue and process them independently. No device memory
// is shared between workers -- this is the fundamental correctness
// requirement that the previous version violated.
//
// On a single GPU (laptop/desktop):
//   Use --workers=1 (default). One worker fully saturates a GPU.
//   --workers=2 on one GPU gives no speedup -- CUDA serialises kernels
//   from different host threads on the same device anyway, and the
//   additional CPU-side concurrency just causes contention.
//
// On multiple GPUs (cloud, e.g. 2/4/8 x A100):
//   Use --gpus=N (or --gpus=-1 for all available).
//   --workers is set automatically to match --gpus.
//   Each worker is bound to a different device via cudaSetDevice(worker_id).
//   Segments are distributed dynamically across workers via the work queue,
//   so load imbalance is handled automatically.
//
// CLI
// ---
//   goldbach_gpu3c_multi <LIMIT> [SEG_SIZE] [P_SMALL]
//                        [--gpus=N]      use N GPUs (default: 1; -1 = all)
//                        [--workers=N]   override worker count (advanced)
//   -h / --help

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <atomic>
#include <sstream>

#include "prime_bitset.hpp"

using namespace goldbach;

// ============================================================
// CUDA error checking
// ============================================================
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " -- " << cudaGetErrorString(err) << "\n";         \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// ============================================================
// Work queue: thread-safe, single producer / multi consumer
// ============================================================
struct SegmentTask {
    uint64_t seg_start;
    uint64_t seg_end;
};

class WorkQueue {
public:
    void push(const SegmentTask& t) {
        std::lock_guard<std::mutex> lock(m_);
        q_.push(t);
        cv_.notify_one();
    }

    // Returns nullopt when queue is empty AND done_ is set.
    std::optional<SegmentTask> pop() {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [this]{ return !q_.empty() || done_; });
        if (q_.empty()) return std::nullopt;
        SegmentTask t = q_.front();
        q_.pop();
        return t;
    }

    void set_done() {
        std::lock_guard<std::mutex> lock(m_);
        done_ = true;
        cv_.notify_all();
    }

private:
    std::queue<SegmentTask> q_;
    std::mutex m_;
    std::condition_variable cv_;
    bool done_ = false;
};

// ============================================================
// Result aggregation (thread-safe)
// ============================================================
struct SegmentResult {
    uint64_t evens_checked    = 0;
    uint64_t failures         = 0;
    uint64_t phase2_fallbacks = 0;
};

struct GlobalResults {
    std::atomic<uint64_t> evens_checked    {0};
    std::atomic<uint64_t> failures         {0};
    std::atomic<uint64_t> phase2_fallbacks {0};

    void merge(const SegmentResult& r) {
        evens_checked    += r.evens_checked;
        failures         += r.failures;
        phase2_fallbacks += r.phase2_fallbacks;
    }
};

// ============================================================
// GPU kernels (same as gpu3c)
// ============================================================
__device__ __forceinline__ uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)((__uint128_t)a * b % m);
}

__device__ __forceinline__ uint64_t powmod64(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = mulmod64(result, base, mod);
        base = mulmod64(base, base, mod);
        exp >>= 1;
    }
    return result;
}

__device__ __forceinline__ bool miller_rabin_witness(uint64_t n, uint64_t a,
                                                      uint64_t d, uint64_t r) {
    uint64_t x = powmod64(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (uint64_t i = 0; i < r - 1; i++) {
        x = mulmod64(x, x, n);
        if (x == n - 1) return true;
    }
    return false;
}

__device__ __forceinline__ bool gpu_is_prime_miller_rabin(uint64_t n) {
    uint64_t d = n - 1, r = 0;
    while ((d & 1) == 0) { d >>= 1; r++; }
    const uint64_t witnesses[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (int i = 0; i < 12; i++) {
        uint64_t a = witnesses[i];
        if (a >= n) continue;
        if (!miller_rabin_witness(n, a, d, r)) return false;
    }
    return true;
}

__device__ __forceinline__ bool is_prime_q(
    uint64_t q,
    const uint64_t* __restrict__ d_small,
    uint64_t small_high,
    bool use_small_bitset,
    const uint64_t* __restrict__ d_seg,
    uint64_t seg_low,
    uint64_t seg_high)
{
    if (q < 2) return false;
    if (q == 2) return true;
    if ((q & 1) == 0) return false;

    if (use_small_bitset && q <= small_high) {
        uint64_t bit_pos  = (q - 3) >> 1;
        return (d_small[bit_pos >> 6] >> (bit_pos & 63)) & 1ULL;
    }

    if (q >= seg_low && q <= seg_high) {
        uint64_t bit_pos  = (q - seg_low) >> 1;
        return (d_seg[bit_pos >> 6] >> (bit_pos & 63)) & 1ULL;
    }

    return gpu_is_prime_miller_rabin(q);
}

__global__ void goldbach_phase1_kernel_byte(
    const uint64_t* __restrict__ d_small,
    uint64_t        small_high,
    bool            use_small_bitset,
    const uint64_t* __restrict__ d_seg,
    uint64_t        seg_low,
    uint64_t        seg_high,
    uint64_t        seg_even_start,
    uint64_t        seg_even_count,
    const uint64_t* __restrict__ p_batch,
    uint64_t        p_batch_size,
    uint8_t*        __restrict__ d_verified)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= seg_even_count) return;
    if (d_verified[tid]) return;

    uint64_t n = seg_even_start + (tid << 1);

    for (uint64_t i = 0; i < p_batch_size; i++) {
        uint64_t p = p_batch[i];
        if (p > (n >> 1)) break;
        uint64_t q = n - p;
        if (is_prime_q(q, d_small, small_high, use_small_bitset,
                       d_seg, seg_low, seg_high)) {
            d_verified[tid] = 1;
            return;
        }
    }
}

__global__ void goldbach_phase1_kernel_bit(
    const uint64_t* __restrict__ d_small,
    uint64_t        small_high,
    bool            use_small_bitset,
    const uint64_t* __restrict__ d_seg,
    uint64_t        seg_low,
    uint64_t        seg_high,
    uint64_t        seg_even_start,
    uint64_t        seg_even_count,
    const uint64_t* __restrict__ p_batch,
    uint64_t        p_batch_size,
    unsigned long long* __restrict__ d_verified_words)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= seg_even_count) return;

    uint64_t word_idx = tid >> 6;
    uint64_t bit_idx  = tid & 63ULL;
    unsigned long long mask = 1ULL << bit_idx;

    if (d_verified_words[word_idx] & mask) return;

    uint64_t n = seg_even_start + (tid << 1);

    for (uint64_t i = 0; i < p_batch_size; i++) {
        uint64_t p = p_batch[i];
        if (p > (n >> 1)) break;
        uint64_t q = n - p;
        if (is_prime_q(q, d_small, small_high, use_small_bitset,
                       d_seg, seg_low, seg_high)) {
            atomicOr(&d_verified_words[word_idx], mask);
            return;
        }
    }
}

// ============================================================
// CPU helpers
// ============================================================
static bool cpu_is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if ((n & 1) == 0) return false;
    uint64_t sq = (uint64_t)std::sqrt((double)n);
    for (uint64_t d = 3; d <= sq; d += 2)
        if (n % d == 0) return false;
    return true;
}

static bool cpu_exhaustive_check(uint64_t n) {
    if (cpu_is_prime(n - 2)) return true;
    for (uint64_t p = 3; p <= n / 2; p += 2)
        if (cpu_is_prime(p) && cpu_is_prime(n - p))
            return true;
    return false;
}

static std::vector<uint64_t> build_segment_bitset(
    uint64_t seg_low,
    uint64_t seg_high,
    const std::vector<uint64_t>& small_primes)
{
    if ((seg_low & 1ULL) == 0) seg_low++;
    uint64_t num_odds  = (seg_high - seg_low) / 2 + 1;
    uint64_t num_words = (num_odds + 63) / 64;
    std::vector<uint64_t> words(num_words, ~0ULL);

    for (uint64_t p : small_primes) {
        if (p * p > seg_high) break;
        uint64_t first = ((seg_low + p - 1) / p) * p;
        if ((first & 1ULL) == 0) first += p;
        if (first < p * p) {
            first = p * p;
            if ((first & 1ULL) == 0) first += p;
        }
        if (first > seg_high) continue;
        for (uint64_t j = first; j <= seg_high; j += 2 * p) {
            uint64_t bit_pos = (j - seg_low) >> 1;
            words[bit_pos >> 6] &= ~(1ULL << (bit_pos & 63ULL));
        }
    }
    return words;
}

// ============================================================
// Shared configuration (read-only after construction)
// ============================================================
struct SegmentConfig {
    uint64_t P_SMALL;
    uint64_t small_high;
    bool     use_bit_verified;
    bool     use_small_bitset;
    int      threads_per_block;
    uint64_t SEG_SIZE;              // max even numbers per segment
    const std::vector<uint64_t>* small_primes;
    const std::vector<uint64_t>* gpu_primes;
    // small bitset on host (for per-worker GPU upload)
    const uint64_t* small_data;
    uint64_t         small_bytes;
};

// ============================================================
// Per-worker GPU buffers
// Allocated inside the worker thread so each worker owns its memory.
// ============================================================
struct WorkerBuffers {
    uint64_t*           d_small         = nullptr;
    uint64_t*           d_seg           = nullptr;
    uint64_t*           d_p_batch       = nullptr;
    uint8_t*            d_verified_bytes = nullptr;
    unsigned long long* d_verified_words = nullptr;

    void allocate(const SegmentConfig& cfg) {
        // Small primes bitset (read-only per worker, but each device needs its own copy)
        if (cfg.use_small_bitset) {
            CUDA_CHECK(cudaMalloc(&d_small, cfg.small_bytes));
            CUDA_CHECK(cudaMemcpy(d_small, cfg.small_data,
                                  cfg.small_bytes, cudaMemcpyHostToDevice));
        }

        // Segment sieve bitset
        uint64_t seg_odd_count = cfg.SEG_SIZE + 1;
        uint64_t seg_words     = (seg_odd_count + 63) / 64;
        CUDA_CHECK(cudaMalloc(&d_seg,     seg_words * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_p_batch, 100'000ULL * sizeof(uint64_t)));

        // Verified buffer
        if (cfg.use_bit_verified) {
            uint64_t vw = (cfg.SEG_SIZE + 63) / 64;
            CUDA_CHECK(cudaMalloc(&d_verified_words,
                                  vw * sizeof(unsigned long long)));
        } else {
            CUDA_CHECK(cudaMalloc(&d_verified_bytes, cfg.SEG_SIZE));
        }
    }

    void free_all() {
        if (d_small)          { cudaFree(d_small);          d_small          = nullptr; }
        if (d_seg)            { cudaFree(d_seg);            d_seg            = nullptr; }
        if (d_p_batch)        { cudaFree(d_p_batch);        d_p_batch        = nullptr; }
        if (d_verified_bytes) { cudaFree(d_verified_bytes); d_verified_bytes = nullptr; }
        if (d_verified_words) { cudaFree(d_verified_words); d_verified_words = nullptr; }
    }

    ~WorkerBuffers() { free_all(); }

    WorkerBuffers() = default;
    WorkerBuffers(const WorkerBuffers&) = delete;
    WorkerBuffers& operator=(const WorkerBuffers&) = delete;
};

// ============================================================
// run_segment: process one segment on the current device
// ============================================================
static SegmentResult run_segment(
    const SegmentTask&  task,
    const SegmentConfig& cfg,
    WorkerBuffers&       buf)
{
    SegmentResult result{};

    uint64_t seg_start = task.seg_start;
    uint64_t seg_end   = task.seg_end;

    uint64_t seg_low  = (seg_start >= 3) ? seg_start - 1 : 3;
    if ((seg_low  & 1ULL) == 0) seg_low++;
    uint64_t seg_high = seg_end + 1;
    if ((seg_high & 1ULL) == 0) seg_high++;

    uint64_t seg_even_count = (seg_end - seg_start) / 2 + 1;
    result.evens_checked = seg_even_count;

    // Build segment sieve on CPU and upload
    auto seg_vec    = build_segment_bitset(seg_low, seg_high, *cfg.small_primes);
    uint64_t seg_words = (uint64_t)seg_vec.size();
    CUDA_CHECK(cudaMemcpy(buf.d_seg, seg_vec.data(),
                          seg_words * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));

    // Clear verified buffer for this segment
    if (cfg.use_bit_verified) {
        uint64_t vw = (seg_even_count + 63) / 64;
        CUDA_CHECK(cudaMemset(buf.d_verified_words, 0,
                              vw * sizeof(unsigned long long)));
    } else {
        CUDA_CHECK(cudaMemset(buf.d_verified_bytes, 0, seg_even_count));
    }

    uint64_t blocks   = (seg_even_count + cfg.threads_per_block - 1)
                        / cfg.threads_per_block;
    const uint64_t P_BATCH = 100'000ULL;

    // Phase 1: GPU
    for (uint64_t bi = 0; bi < cfg.gpu_primes->size(); bi += P_BATCH) {
        uint64_t bend  = std::min(bi + P_BATCH,
                                  (uint64_t)cfg.gpu_primes->size());
        uint64_t bsize = bend - bi;

        CUDA_CHECK(cudaMemcpy(buf.d_p_batch,
                              cfg.gpu_primes->data() + bi,
                              bsize * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));

        if (cfg.use_bit_verified) {
            goldbach_phase1_kernel_bit<<<(uint32_t)blocks,
                                         cfg.threads_per_block>>>(
                buf.d_small, cfg.small_high, cfg.use_small_bitset,
                buf.d_seg,   seg_low, seg_high,
                seg_start,   seg_even_count,
                buf.d_p_batch, bsize,
                buf.d_verified_words);
        } else {
            goldbach_phase1_kernel_byte<<<(uint32_t)blocks,
                                          cfg.threads_per_block>>>(
                buf.d_small, cfg.small_high, cfg.use_small_bitset,
                buf.d_seg,   seg_low, seg_high,
                seg_start,   seg_even_count,
                buf.d_p_batch, bsize,
                buf.d_verified_bytes);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Phase 2: CPU fallback for any unverified n
    if (cfg.use_bit_verified) {
        uint64_t vw = (seg_even_count + 63) / 64;
        std::vector<unsigned long long> verified(vw);
        CUDA_CHECK(cudaMemcpy(verified.data(), buf.d_verified_words,
                              vw * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        for (uint64_t i = 0; i < seg_even_count; i++) {
            if ((verified[i >> 6] >> (i & 63)) & 1ULL) continue;
            result.phase2_fallbacks++;
            if (!cpu_exhaustive_check(seg_start + (i << 1)))
                result.failures++;
        }
    } else {
        std::vector<uint8_t> verified(seg_even_count);
        CUDA_CHECK(cudaMemcpy(verified.data(), buf.d_verified_bytes,
                              seg_even_count, cudaMemcpyDeviceToHost));
        for (uint64_t i = 0; i < seg_even_count; i++) {
            if (verified[i]) continue;
            result.phase2_fallbacks++;
            if (!cpu_exhaustive_check(seg_start + (i << 1)))
                result.failures++;
        }
    }

    return result;
}

// ============================================================
// Worker thread
// Each worker: binds to its device, allocates its own buffers,
// pulls segments from the queue, processes them, merges results.
// ============================================================
struct WorkerConfig {
    int            device_id;
    WorkQueue*     queue;
    GlobalResults* global;
    SegmentConfig  seg_cfg;   // read-only shared config (no device ptrs)
};

static void worker_thread(WorkerConfig cfg) {
    // Bind this thread to its assigned GPU
    CUDA_CHECK(cudaSetDevice(cfg.device_id));

    // Each worker allocates its own private device buffers
    WorkerBuffers buf;
    buf.allocate(cfg.seg_cfg);

    // Process segments until the queue is exhausted
    while (true) {
        auto task_opt = cfg.queue->pop();
        if (!task_opt.has_value()) break;

        SegmentResult r = run_segment(*task_opt, cfg.seg_cfg, buf);
        cfg.global->merge(r);
    }

    // Free this worker's device buffers before the thread exits
    buf.free_all();
}

// ============================================================
// Utility
// ============================================================
static void print_usage(const char* prog) {
    std::cout << "Goldbach Conjecture Segmented Verifier (multi-GPU)\n";
    std::cout << "Usage: " << prog
              << " <LIMIT> [SEG_SIZE] [P_SMALL] [OPTIONS]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  LIMIT       Max even integer to check\n";
    std::cout << "  SEG_SIZE    Even integers per segment (default: 10,000,000)\n";
    std::cout << "  P_SMALL     GPU prime search bound   (default: 1,000,000)\n\n";
    std::cout << "Options:\n";
    std::cout << "  --gpus=N    Number of GPUs to use (default: 1; -1 = all available)\n";
    std::cout << "  --workers=N Override worker count (default: matches --gpus)\n";
    std::cout << "  -h, --help  Show this message\n\n";
    std::cout << "Notes:\n";
    std::cout << "  On a single GPU, --workers > 1 gives no speedup.\n";
    std::cout << "  On N GPUs, use --gpus=N for near-linear scaling.\n";
    std::cout << "  Each worker is bound to a different GPU device.\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog << " 1000000000000\n";
    std::cout << "  " << prog << " 1000000000000 500000000 2000000 --gpus=4\n";
    std::cout << "  " << prog << " 1000000000000 --gpus=-1\n";
}

// ============================================================
// main()
// ============================================================
int main(int argc, char** argv) {
    if (argc < 2 ||
        std::string(argv[1]) == "-h" ||
        std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        return 0;
    }

    uint64_t LIMIT    = 0;
    uint64_t SEG_SIZE = 10'000'000ULL;
    uint64_t P_SMALL  = 1'000'000ULL;
    int      gpus_requested = 1;
    int      workers_override = -1;   // -1 = not set by user

    int positional = 0;
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg.rfind("--gpus=", 0) == 0) {
            gpus_requested = std::stoi(arg.substr(7));
        } else if (arg.rfind("--workers=", 0) == 0) {
            workers_override = std::stoi(arg.substr(10));
        } else if (arg.rfind("--", 0) == 0) {
            std::cerr << "Unknown option: " << arg
                      << ". Use -h for help.\n";
            return 1;
        } else {
            try {
                uint64_t val = std::stoull(arg);
                if      (positional == 0) LIMIT    = val;
                else if (positional == 1) SEG_SIZE = val;
                else if (positional == 2) P_SMALL  = val;
            } catch (...) {
                std::cerr << "Error: invalid argument '" << arg
                          << "'. Use -h for help.\n";
                return 1;
            }
            positional++;
        }
    }

    if (LIMIT < 4) {
        std::cerr << "Error: LIMIT must be >= 4.\n";
        return 1;
    }
    if (LIMIT % 2 != 0) LIMIT--;

    // --- Determine available and requested GPU count ---
    int num_available_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_available_gpus));

    if (num_available_gpus == 0) {
        std::cerr << "Error: no CUDA devices found.\n";
        return 1;
    }

    int num_gpus;
    if (gpus_requested == -1) {
        num_gpus = num_available_gpus;
    } else {
        num_gpus = std::min(gpus_requested, num_available_gpus);
        if (gpus_requested > num_available_gpus) {
            std::cerr << "[!] Warning: requested " << gpus_requested
                      << " GPUs but only " << num_available_gpus
                      << " available. Using " << num_gpus << ".\n";
        }
    }

    // Workers default to 1 per GPU. On a single GPU, >1 worker gives
    // no speedup (CUDA serialises kernels from different threads on the
    // same device) and risks contention. On N GPUs, N workers is ideal.
    int num_workers = (workers_override > 0) ? workers_override : num_gpus;

    // Warn if the user has set workers > gpus on a single-GPU machine
    if (num_workers > num_gpus && num_gpus == 1) {
        std::cerr << "[!] Warning: --workers=" << num_workers
                  << " on a single GPU gives no speedup and may cause "
                     "contention. Resetting to 1.\n";
        num_workers = 1;
    }

    // Print GPU info
    std::cout << "[Hardware] " << num_available_gpus
              << " GPU(s) available, using " << num_gpus
              << ", workers = " << num_workers << "\n";
    for (int d = 0; d < num_gpus; d++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, d);
        std::cout << "  [GPU " << d << "] " << prop.name
                  << " (" << prop.totalGlobalMem / (1024*1024) << " MB VRAM)\n";
    }
    std::cout << "\n";

    // --- Build shared read-only data ---
    uint64_t small_high = std::max(
        (uint64_t)std::sqrt((double)LIMIT) + 1, P_SMALL);
    if ((small_high & 1ULL) == 0) small_high++;

    std::cout << "Building small primes bitset up to " << small_high << "...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    PrimeBitset small_bitset = build_prime_bitset(small_high);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Built in "
              << std::chrono::duration<double, std::milli>(t1-t0).count()
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
              << gpu_primes.size() << "\n";

    // Segment/verified settings
    bool use_bit_verified = false;   // byte verified is faster at SEG_SIZE ~1e7
    bool use_small_bitset = true;

    uint64_t small_bytes = small_bitset.word_count() * sizeof(uint64_t);

    // Shared config passed to all workers (read-only, no device pointers)
    SegmentConfig seg_cfg;
    seg_cfg.P_SMALL           = P_SMALL;
    seg_cfg.small_high        = small_high;
    seg_cfg.use_bit_verified  = use_bit_verified;
    seg_cfg.use_small_bitset  = use_small_bitset;
    seg_cfg.threads_per_block = 256;
    seg_cfg.SEG_SIZE          = SEG_SIZE;
    seg_cfg.small_primes      = &small_primes;
    seg_cfg.gpu_primes        = &gpu_primes;
    seg_cfg.small_data        = small_bitset.data();
    seg_cfg.small_bytes       = small_bytes;

    // --- Build work queue (all segments, pre-loaded) ---
    WorkQueue queue;
    {
        uint64_t seg_start = 4;
        while (seg_start <= LIMIT) {
            uint64_t seg_end = std::min(seg_start + SEG_SIZE * 2 - 2, LIMIT);
            queue.push(SegmentTask{seg_start, seg_end});
            seg_start = seg_end + 2;
        }
    }
    // Signal that no more tasks will be pushed
    queue.set_done();

    GlobalResults global;

    // --- Launch workers ---
    std::vector<std::thread> workers;
    workers.reserve(num_workers);

    auto t_main = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_workers; i++) {
        WorkerConfig cfg;
        cfg.device_id = i % num_gpus;   // round-robin if workers > gpus
        cfg.queue     = &queue;
        cfg.global    = &global;
        cfg.seg_cfg   = seg_cfg;
        workers.emplace_back(worker_thread, cfg);
    }

    for (auto& t : workers) t.join();

    double total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_main).count();

    // --- Summary ---
    std::cout << "\n--- Summary ---\n";
    std::cout << "Even numbers checked  : " << global.evens_checked.load()    << "\n";
    std::cout << "Failures              : " << global.failures.load()         << "\n";
    std::cout << "Phase 2 fallbacks     : " << global.phase2_fallbacks.load() << "\n";
    std::cout << "GPUs used             : " << num_gpus                       << "\n";
    std::cout << "Workers               : " << num_workers                    << "\n";
    std::cout << "Total time            : " << total_ms                       << " ms\n";

    if (global.failures.load() == 0) {
        std::cout << "\nAll even numbers up to " << LIMIT
                  << " satisfy Goldbach. ✓\n";
        if (global.phase2_fallbacks.load() == 0)
            std::cout << "(All verified by GPU with p <= " << P_SMALL << ")\n";
        else
            std::cout << "(" << global.phase2_fallbacks.load()
                      << " required CPU exhaustive fallback)\n";
    }

    return (global.failures.load() == 0) ? 0 : 1;
}

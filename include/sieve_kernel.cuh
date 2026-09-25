// sieve_kernel.cuh
// Shared device-side segment sieve, included by both the verifier and its tests.
// Extracted from goldbach.cu so tests exercise the production kernel, not a copy.

#pragma once
#include <cstdint>
#include <algorithm>

// Odd numbers per sieve tile. Shared memory per block is TILE_ODDS bytes
// (one byte per odd number), so this trades occupancy against the fixed
// per-tile cost of looping over every small prime.
//
// Overridable at compile time: -DTILE_ODDS=<value>. Note a block cannot
// exceed the device's default 48 KB shared-memory cap unless
// cudaFuncSetAttribute raises it, so values above 49152 will fail to launch.
// 32768 is the largest power of two that fits under that cap.
//
// Chosen by measurement on v3.0.0 code, with large_prime_sieve_kernel present,
// on an RTX 5090 (sm_120). Figures are variable cost: mean wall clock at N minus
// mean wall clock at 1e6, which removes the fixed setup. Five runs at 1e11 and
// 1e12, three at 1e13, builds interleaved, 0 Phase 2 fallbacks throughout.
//
//   TILE_ODDS      1e11       1e12       1e13
//        4096         --    10.760s         --
//        8192     0.726s     7.946s    83.969s
//       16384     0.702s     7.398s    78.670s   <- best at every limit
//       32768     0.800s     8.418s    88.957s
//
// 16384 wins by 13-14% over 32768 at all three limits, a margin stable across
// two decades of N rather than a crossover.
//
// This reverses an earlier default. A sweep at 1e11 recorded here previously
// (4096 -> 4.019s, 8192 -> 2.201s, 16384 -> 1.379s, 32768 -> 1.284s) put 32768
// ahead, and that measurement was sound for the code it was taken on: commits
// 6daef62 and 75a03b5 differ only in this constant and measure 1.383s and
// 1.303s at 1e11, matching those figures. What changed is 29878c3, which moved
// large primes out of the tiled kernel into large_prime_sieve_kernel. That
// removed the per-tile work a wider tile was amortising, and the tradeoff
// inverted. Do not reinstate 32768 on the strength of the old numbers.
//
// Occupancy does not predict the optimum. cudaOccupancyMaxActiveBlocksPerMulti-
// processor on tiled_sieve_segment_kernel at 256 threads gives 6, 6, 5 and 3
// resident blocks per SM for 4096, 8192, 16384 and 32768 (36 registers in every
// case, so shared memory alone sets it). Occupancy falls monotonically with
// tile width while runtime is U-shaped; the fastest setting is the second
// lowest occupancy of the four.
#ifndef TILE_ODDS
#define TILE_ODDS 16384
#endif

// Prime-list partition point: primes below this go to the tiled kernel, the
// rest to large_prime_sieve_kernel. Independent of TILE_ODDS -- tile width is
// a shared-memory/occupancy question, this is a per-tile-division vs global-
// atomic tradeoff. Both kernels are correct for any split point.
//
// Chosen at N = 1e13, the limit the manuscript reports, with TILE_ODDS=16384
// on an RTX 5090 (sm_120). Variable cost: mean wall clock at 1e13 minus mean
// wall clock at 1e6. Three runs at 1e13, five at 1e6, builds interleaved,
// 0 Phase 2 fallbacks throughout:
//
//   SPLIT_THRESHOLD   variable cost   vs best
//             32768         85.991s    +9.28%
//             65536         78.686s      best
//            131072         81.369s    +3.41%
//
// 65536 remains the best of the three after the tile width dropped to 16384,
// so the two constants did not have to be retuned together. The curve keeps the
// shape it had at TILE_ODDS=32768: one step below is the expensive mistake
// (+9.28%) and one step above the cheap one (+3.41%).
//
// The remaining sieve time is atomic-bound rather than division-bound: primes
// marking at most one byte per tile are still cheaper left in the tiled kernel,
// paying full per-tile divisions, than moved to global atomicAnd.
//
// The optimum depends on how many sieving primes there are, which grows as
// sqrt(N) above 1e12, so a different limit may prefer a different value. Only
// powers of two were sampled and only at 1e13; the true optimum lies somewhere
// in [32768, 131072].
#ifndef SPLIT_THRESHOLD
#define SPLIT_THRESHOLD 65536
#endif

// Overflow-safe tiled sieve
__global__ void tiled_sieve_segment_kernel(
    uint64_t        q_low,
    uint64_t        q_high,
    const uint64_t* __restrict__ d_small_primes,
    uint64_t        small_prime_count,
    uint64_t*       __restrict__ d_seg_bits)
{
    extern __shared__ unsigned char sh_tile[];

    uint64_t num_odds = (q_high - q_low) / 2 + 1;
    uint64_t num_tiles = (num_odds + TILE_ODDS - 1) / TILE_ODDS;

    uint64_t tile_id = blockIdx.x;
    if (tile_id >= num_tiles) return;

    uint64_t tile_bit_offset = tile_id * TILE_ODDS;
    uint64_t tile_odd_start = tile_bit_offset;
    uint64_t tile_odd_end   = min(tile_odd_start + TILE_ODDS, num_odds);
    uint64_t tile_odd_count = tile_odd_end - tile_odd_start;
    uint64_t tile_word_count = (tile_odd_count + 63) / 64;

    for (uint64_t i = threadIdx.x; i < tile_odd_count; i += blockDim.x) {
        sh_tile[i] = 1;
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
            sh_tile[bit - tile_odd_start] = 0;
        }
    }
    __syncthreads();

    uint64_t global_word_offset = tile_odd_start / 64;
    for (uint64_t w = threadIdx.x; w < tile_word_count; w += blockDim.x) {
        uint64_t word = 0;
        uint64_t base = w * 64;
        for (uint64_t b = 0; b < 64; b++) {
            if (base + b < tile_odd_count && sh_tile[base + b]) {
                word |= (1ULL << b);
            }
        }
        d_seg_bits[global_word_offset + w] = word;
    }
}

// -------------------------------------------------------
// Large-prime sieve: one thread per prime, straight to global memory.
// -------------------------------------------------------
// A prime p >= TILE_ODDS marks at most one byte in any given tile, yet the
// tiled kernel above pays three 64-bit divisions for it in every tile --
// and 64-bit division is emulated on NVIDIA hardware. Splitting those primes
// out moves that cost from once-per-tile-per-prime to once-per-segment-per-
// prime.
//
// The marking is sparse: over a ~4e8 wide q span a prime near 1e6 marks ~200
// positions and one near 1e5 marks ~2000, scattered across the whole segment
// bitset rather than concentrated in one tile.
//
// The atomic is required for the same reason the shared-memory version needed
// one: clearing a bit is a read-modify-write, and different primes land on
// different bits of the same 64-bit word. Contention is low because the
// writes are spread over the entire bitset.
//
// ORDERING: this must run after tiled_sieve_segment_kernel, which writes whole
// words (overwriting, not merging) and would otherwise erase these marks.
// Launching both on the same stream is sufficient.
//
// The prime list is sorted ascending, so threads in a warp get adjacent primes
// and therefore near-identical iteration counts.
__global__ void large_prime_sieve_kernel(
    uint64_t        q_low,
    uint64_t        q_high,
    const uint64_t* __restrict__ d_large_primes,
    uint64_t        large_prime_count,
    uint64_t*       d_seg_bits)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= large_prime_count) return;

    uint64_t p = d_large_primes[idx];
    if (p < 3) return;
    if (p > q_high / p) return;   // OVERFLOW-SAFE, and p*p > q_high marks nothing

    uint64_t first = (q_low + p - 1) / p * p;
    if ((first & 1) == 0) first += p;
    // p*p is odd for odd p, so this needs no second parity fixup.
    if (first < p * p) first = p * p;
    if (first > q_high) return;

    // Step in bit-index space: q advances by 2p, so i advances by p. Iterating
    // on the index rather than on q keeps the loop free of any overflow risk
    // near the top of the uint64_t range.
    uint64_t num_odds = (q_high - q_low) / 2 + 1;
    for (uint64_t i = (first - q_low) >> 1; i < num_odds; i += p) {
        atomicAnd(reinterpret_cast<unsigned long long*>(&d_seg_bits[i >> 6]),
                  ~(1ULL << (i & 63)));
    }
}

// Split point for a sorted prime list: the number of primes below
// SPLIT_THRESHOLD. Those go to the tiled kernel, the rest to
// large_prime_sieve_kernel.
static inline uint64_t sieve_split_prime_count(const uint64_t* primes,
                                               uint64_t count)
{
    return (uint64_t)(std::lower_bound(primes, primes + count,
                                       (uint64_t)SPLIT_THRESHOLD) - primes);
}

// Runs the full segment sieve: tiled kernel over the small primes, then the
// large-prime kernel over the rest. Shared by the verifier and its tests so
// both exercise the same pipeline.
static inline void launch_segment_sieve(
    uint64_t q_low, uint64_t q_high,
    const uint64_t* d_primes, uint64_t small_count, uint64_t large_count,
    uint64_t* d_seg_bits, int threads_per_block, cudaStream_t stream)
{
    uint64_t num_odds  = (q_high - q_low) / 2 + 1;
    uint32_t num_tiles = (uint32_t)((num_odds + TILE_ODDS - 1) / TILE_ODDS);

    tiled_sieve_segment_kernel<<<num_tiles, threads_per_block,
                                 TILE_ODDS * sizeof(unsigned char), stream>>>(
        q_low, q_high, d_primes, small_count, d_seg_bits);

    if (large_count > 0) {
        uint32_t blocks =
            (uint32_t)((large_count + threads_per_block - 1) / threads_per_block);
        large_prime_sieve_kernel<<<blocks, threads_per_block, 0, stream>>>(
            q_low, q_high, d_primes + small_count, large_count, d_seg_bits);
    }
}

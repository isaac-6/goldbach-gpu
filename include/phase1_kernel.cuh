// phase1_kernel.cuh
// Shared device-side Goldbach Phase 1 verification, included by both the
// verifier and its tests.
// Extracted from goldbach.cu so tests exercise the production kernel, not a copy.

#pragma once
#include <cstdint>
#include "primality.cuh"

enum class PrimeTest {
    MillerRabin,
    BPSW
};

// Passed to GPU once per device via cudaMemcpyToSymbol
__constant__ PrimeTest g_device_prime_test;

// -------------------------------------------------------
// GPU Kernel Device Functions
// -------------------------------------------------------


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

    if (g_device_prime_test == PrimeTest::BPSW)
        return gpu_is_prime_bpsw(q);
    return gpu_is_prime_miller_rabin(q);
}

// Phase 1 Kernel: GPU Goldbach Verification
// -------------------------------------------------------
// One thread per even number in segment.
//
// d_verified is a bitset: bit j of word w is the even number
// seg_even_start + 2*(64*w + j). One thread per even number means 64 threads
// share a word, so the set must be an atomicOr. This path runs only for the
// first segment or two, so the atomic costs nothing that matters.
//
// CRITICAL INVARIANTS:
// 1. p_batch MUST be sorted in ascending order (allows early termination)
// 2. d_verified bits are monotonic: once set, they stay set forever
// 3. Kernel may be called multiple times with different p_batch slices
// 4. Early return on d_verified[tid] == 1 is SAFE because:
//    - We only need ONE valid Goldbach partition (p + q = n)
//    - Finding one partition proves n satisfies the conjecture
//    - Additional partitions are unnecessary
// 5. p > n/2 termination is SAFE because:
//    - If p > n/2, then q = n - p < n/2 < p
//    - This would be a duplicate of the partition (q, p)
//    - All unique partitions have p <= n/2
//
// THREAD SAFETY:
// - Multiple threads may write d_verified[tid] = 1 concurrently (idempotent)
// - No thread ever writes d_verified[tid] = 0 after initialization
// - No race conditions or data corruption possible
//
// OVERFLOW SAFETY:
// - p > n/2 uses division (safe for all uint64_t values)
// - n - p cannot underflow because p <= n/2 < n
// -------------------------------------------------------
__global__ void goldbach_phase1_kernel(
    const uint64_t* __restrict__ d_small, uint64_t small_high,
    const uint64_t* __restrict__ d_seg_bits, uint64_t q_low, uint64_t q_high,
    uint64_t seg_even_start, uint64_t seg_even_count,
    const uint64_t* __restrict__ p_batch, uint64_t p_batch_size,
    uint64_t* d_verified)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= seg_even_count) return;
    // Early return monotonic safety. This read is not atomic while other lanes
    // atomicOr the same word, but bits only ever go 0 -> 1, so a stale read
    // costs redundant work and never a wrong answer.
    if ((d_verified[tid >> 6] >> (tid & 63)) & 1ULL) return;

    uint64_t n = seg_even_start + tid * 2;

    for (uint64_t i = 0; i < p_batch_size; i++) {
        uint64_t p = p_batch[i];
        if (p > n / 2) break; 
        uint64_t q = n - p;

        if (is_prime_q(q, d_small, small_high, d_seg_bits, q_low, q_high)) {
            atomicOr(reinterpret_cast<unsigned long long*>(&d_verified[tid >> 6]),
                     1ULL << (tid & 63));
            return;
        }
    }
}

// -------------------------------------------------------
// Phase 1 Kernel, transposed: 64 even numbers per thread.
// -------------------------------------------------------
// For 64 consecutive even numbers n_j = n_base + 2j (j = 0..63) and a fixed
// odd prime p, the complements q_j = n_base - p + 2j are 64 consecutive odd
// numbers -- exactly a 64-bit window of the segment bitset. One shifted load
// and one OR resolves all 64 numbers against that prime.
//
// Indexing: bit i of d_seg_bits is the odd number q_low + 2i, so for
// q_base = n_base - p the window starts at bit i_base = (q_base - q_low)/2.
//
// WHY THE SEGMENT BITSET ALONE SUFFICES (no is_prime_q, no Miller-Rabin):
// every q = n - p with n in the segment and p <= P_SMALL lies within
// [q_low, q_high], which the sieve has already resolved exactly. The scalar
// kernel above keeps is_prime_q for the small-n fallback path.
//
// PRECONDITION (enforced by launch_goldbach_phase1): the caller only routes
// segments with seg_even_start > 2*P_SMALL + 128 here. That gives two things:
//   - n_base - p >= q_low for every p in the batch, so i_base never underflows
//   - p <= P_SMALL < n/2 always, so the scalar kernel's "p > n/2" break can
//     never trigger and the two kernels agree exactly on every n
//
// READS ONE WORD PAST i_base's word: d_seg_bits must be allocated with one
// padding word beyond the segment's own words.
// -------------------------------------------------------
__global__ void goldbach_phase1_transposed_kernel(
    const uint64_t* __restrict__ d_seg_bits, uint64_t q_low,
    uint64_t seg_even_start, uint64_t seg_even_count,
    const uint64_t* __restrict__ p_batch, uint64_t p_batch_size,
    uint64_t* __restrict__ d_verified)
{
    uint64_t t           = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t group_start = t * 64;                 // index of this thread's first even number
    if (group_start >= seg_even_count) return;

    uint64_t valid = seg_even_count - group_start;
    if (valid > 64) valid = 64;

    uint64_t n_base = seg_even_start + 2 * group_start;

    // Thread t owns word t of the d_verified bitset outright -- no other thread
    // writes it, so the accumulate is a plain load and store, not an atomic.
    //
    // Tail masking: on the final partial group the high bits cover numbers past
    // the range. Seed them to 1 so verified_word can still reach ~0ULL and take
    // the early exit. They are stored, so on this path the tail ends up set;
    // count_unverified_kernel does not rely on that (the scalar path leaves the
    // same bits clear) and masks them itself.
    uint64_t tail_mask     = (valid == 64) ? 0ULL : (~0ULL << valid);
    uint64_t verified_word = d_verified[t] | tail_mask;

    for (uint64_t i = 0; i < p_batch_size && verified_word != ~0ULL; i++) {
        uint64_t p = p_batch[i];
        // p = 2 gives an even q, which the odd-only segment bitset cannot
        // represent. q even is prime only at q = 2 (n = 4), which lives in the
        // small-n range and is handled by the scalar kernel.
        if (p == 2) continue;

        uint64_t i_base   = (n_base - p - q_low) >> 1;
        uint64_t word_idx = i_base >> 6;
        uint32_t shift    = (uint32_t)(i_base & 63);

        uint64_t window;
        if (shift == 0) {
            // Shifting a 64-bit value by 64 is undefined behaviour, not zero.
            window = d_seg_bits[word_idx];
        } else {
            window = (d_seg_bits[word_idx] >> shift)
                   | (d_seg_bits[word_idx + 1] << (64 - shift));
        }
        verified_word |= window;
    }

    d_verified[t] = verified_word;
}

// -------------------------------------------------------
// Host-side dispatch between the two Phase 1 kernels.
// -------------------------------------------------------
// The transposed kernel requires n_base - p >= q_low for every p in the batch.
// When seg_even_start <= P_SMALL the segment's q_low is clamped to 3 and that
// fails, so those segments take the scalar kernel.
//
// The threshold is 2*P_SMALL + 128, not P_SMALL + 128: above it every p is
// also < n/2, so the scalar kernel's "p > n/2" break is unreachable and both
// kernels compute identically. (Between P_SMALL and 2*P_SMALL the transposed
// kernel would still be correct -- a partition with p > q is still a partition
// -- but it would verify numbers the scalar kernel declines to, and the two
// paths would no longer agree.)
//
// NOTE: this deliberately does not test seg_even_start against
// q_low + P_SMALL + 128. Since q_low is itself seg_start - P_SMALL, that
// comparison reduces to seg_start < seg_start + 128 and is true for every
// segment, which would route all work to the scalar kernel.
// -------------------------------------------------------
static inline void launch_goldbach_phase1(
    const uint64_t* d_small, uint64_t small_high,
    const uint64_t* d_seg_bits, uint64_t q_low, uint64_t q_high,
    uint64_t seg_even_start, uint64_t seg_even_count,
    const uint64_t* d_p_batch, uint64_t p_batch_size,
    uint64_t* d_verified, uint64_t p_small,
    int threads_per_block, cudaStream_t stream)
{
    if (seg_even_start > 2 * p_small + 128) {
        uint64_t groups = (seg_even_count + 63) / 64;
        uint32_t blocks = (uint32_t)((groups + threads_per_block - 1) / threads_per_block);
        goldbach_phase1_transposed_kernel<<<blocks, threads_per_block, 0, stream>>>(
            d_seg_bits, q_low, seg_even_start, seg_even_count,
            d_p_batch, p_batch_size, d_verified);
    } else {
        uint32_t blocks =
            (uint32_t)((seg_even_count + threads_per_block - 1) / threads_per_block);
        goldbach_phase1_kernel<<<blocks, threads_per_block, 0, stream>>>(
            d_small, small_high, d_seg_bits, q_low, q_high,
            seg_even_start, seg_even_count, d_p_batch, p_batch_size, d_verified);
    }
}

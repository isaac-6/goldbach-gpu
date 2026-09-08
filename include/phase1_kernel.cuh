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
// CRITICAL INVARIANTS:
// 1. p_batch MUST be sorted in ascending order (allows early termination)
// 2. d_verified[tid] is monotonic: once set to 1, stays 1 forever
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

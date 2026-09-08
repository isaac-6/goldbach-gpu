// sieve_kernel.cuh
// Shared device-side segment sieve, included by both the verifier and its tests.
// Extracted from goldbach.cu so tests exercise the production kernel, not a copy.

#pragma once
#include <cstdint>

// Odd numbers per sieve tile. Shared memory per block is TILE_ODDS bytes
// (one byte per odd number), so this trades occupancy against the fixed
// per-tile cost of looping over every small prime.
//
// Overridable at compile time: -DTILE_ODDS=<value>. Note a block cannot
// exceed the device's default 48 KB shared-memory cap unless
// cudaFuncSetAttribute raises it, so values above 49152 will fail to launch.
// 32768 is the largest power of two that fits under that cap.
//
// Chosen by measurement at 1e11 (RTX 5090): 4096 -> 4.019s, 8192 -> 2.201s,
// 16384 -> 1.379s, 32768 -> 1.284s.
#ifndef TILE_ODDS
#define TILE_ODDS 32768
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
// Cross-checks BPSW against the 12-base deterministic Miller-Rabin.
// MR is sound across the full 64-bit range, so it serves as oracle.
// Emphasis on n > 2^63, where the Lucas halving steps overflow.
//
// Exit 0 = agreement, 1 = disagreement.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "primality.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); exit(2); } } while (0)

__global__ void compare_kernel(const uint64_t* n, int count,
                               unsigned char* mr, unsigned char* bpsw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    mr[i]   = gpu_is_prime_miller_rabin(n[i]) ? 1 : 0;
    bpsw[i] = gpu_is_prime_bpsw(n[i])         ? 1 : 0;
}

static uint64_t run_batch(const std::vector<uint64_t>& vals, const char* label) {
    int count = (int)vals.size();
    uint64_t *d_n; unsigned char *d_mr, *d_bpsw;
    CK(cudaMalloc(&d_n, count * sizeof(uint64_t)));
    CK(cudaMalloc(&d_mr, count));
    CK(cudaMalloc(&d_bpsw, count));
    CK(cudaMemcpy(d_n, vals.data(), count * sizeof(uint64_t), cudaMemcpyHostToDevice));

    compare_kernel<<<(count + 255) / 256, 256>>>(d_n, count, d_mr, d_bpsw);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<unsigned char> mr(count), bpsw(count);
    CK(cudaMemcpy(mr.data(), d_mr, count, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(bpsw.data(), d_bpsw, count, cudaMemcpyDeviceToHost));

    uint64_t bad = 0, shown = 0;
    for (int i = 0; i < count; i++) {
        // Host BPSW must agree too - it has the same halving bug.
        bool host_bpsw = cpu_is_prime_bpsw(vals[i]);
        bool host_mr   = cpu_miller_rabin(vals[i]);
        if (mr[i] != bpsw[i] || host_bpsw != host_mr || (mr[i] != 0) != host_mr) {
            bad++;
            if (shown < 5) {
                printf("    n=%llu  gpu_mr=%d gpu_bpsw=%d cpu_mr=%d cpu_bpsw=%d\n",
                       (unsigned long long)vals[i], mr[i], bpsw[i],
                       (int)host_mr, (int)host_bpsw);
                shown++;
            }
        }
    }
    printf("  [%s] %d values -> %llu disagreements\n", label, count, (unsigned long long)bad);
    CK(cudaFree(d_n)); CK(cudaFree(d_mr)); CK(cudaFree(d_bpsw));
    return bad;
}

int main() {
    uint64_t total = 0;
    std::mt19937_64 rng(20260907);

    // Below 2^63: expected to pass even before the fix.
    {
        std::vector<uint64_t> v;
        for (int i = 0; i < 200000; i++) v.push_back((rng() % (1ULL << 62)) | 1ULL);
        total += run_batch(v, "odd n < 2^62");
    }

    // Above 2^63: the overflow region.
    {
        std::vector<uint64_t> v;
        for (int i = 0; i < 200000; i++)
            v.push_back(((1ULL << 63) + (rng() % ((1ULL << 63) - 1))) | 1ULL);
        total += run_batch(v, "odd n > 2^63");
    }

    // Just above the boundary, where wrapping first appears.
    {
        std::vector<uint64_t> v;
        for (uint64_t k = 1; k < 100000; k += 2) v.push_back((1ULL << 63) + k);
        total += run_batch(v, "n just above 2^63");
    }

    // Known primes near the top of the range.
    {
        std::vector<uint64_t> v = {
            18446744073709551557ULL, 18446744073709551533ULL,
            18446744073709551521ULL, 18446744073709551437ULL,
            9223372036854775837ULL,  9223372036854775907ULL
        };
        total += run_batch(v, "known large primes");
    }

    printf("\nTOTAL DISAGREEMENTS: %llu\n", (unsigned long long)total);
    if (total) { printf("FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}
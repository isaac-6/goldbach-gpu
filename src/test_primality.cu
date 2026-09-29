// Cross-checks BPSW against the 12-base deterministic Miller-Rabin.
// MR is sound across the full 64-bit range, so it serves as oracle.
// Emphasis on n > 2^63, where the Lucas halving steps once overflowed.
// Random inputs almost never include a base-2 strong pseudoprime, so this
// barely exercises the strong Lucas step; test_bpsw_spsp does.
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

// Counts values on which the four tests (device and host, MR and BPSW) do not
// all agree. With expect, each value must also match its known verdict
// (1 = prime, 0 = composite), so a fault shared by both tests still shows.
// Lucas conditions of [BFW] for given (n, D, P, Q), on the device.
__global__ void lucas_kernel(const uint64_t* n, const int64_t* dpq, int count, unsigned char* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    LucasConditions c = gpu_lucas_conditions(n[i], dpq[3 * i], dpq[3 * i + 1], dpq[3 * i + 2]);
    out[i] = (c.strong ? 1 : 0) | (c.v ? 2 : 0) | (c.euler ? 4 : 0);
}

// The V_{n+1} = 2Q check (BPSW step 4) only ever rejects a composite that
// already passed steps 1-3, and none is known, so no primality verdict can
// show that it is computed. This pins it to Baillie, Fiori and Wagstaff
// (arXiv:2006.14425): the five Lucas-V pseudoprimes of their Table 2 must
// satisfy it with the Method A* parameters they state, and n = 323, their
// Section 2.4 example (V_324 = 135, 2Q = 10), must not. None is a strong Lucas
// probable prime. Device and host.
static uint64_t run_lucas_v() {
    struct { uint64_t n; int64_t D, P, Q; bool v; } c[] = {
        {323ULL,             5, 5, 5, false},
        {913ULL,             5, 5, 5, true},
        {150267335403ULL,    5, 5, 5, true},
        {430558874533ULL,    5, 5, 5, true},
        {14760229232131ULL, -7, 1, 2, true},
        {936916995253453ULL, 5, 5, 5, true},
    };
    const int count = sizeof(c) / sizeof(c[0]);
    std::vector<uint64_t> n(count); std::vector<int64_t> dpq(3 * count);
    for (int i = 0; i < count; i++) { n[i] = c[i].n; dpq[3*i] = c[i].D; dpq[3*i+1] = c[i].P; dpq[3*i+2] = c[i].Q; }
    uint64_t* d_n; int64_t* d_dpq; unsigned char* d_out;
    CK(cudaMalloc(&d_n, count * sizeof(uint64_t))); CK(cudaMalloc(&d_dpq, 3 * count * sizeof(int64_t)));
    CK(cudaMalloc(&d_out, count));
    CK(cudaMemcpy(d_n, n.data(), count * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dpq, dpq.data(), 3 * count * sizeof(int64_t), cudaMemcpyHostToDevice));
    lucas_kernel<<<1, 32>>>(d_n, d_dpq, count, d_out);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<unsigned char> dev(count);
    CK(cudaMemcpy(dev.data(), d_out, count, cudaMemcpyDeviceToHost));
    uint64_t bad = 0;
    for (int i = 0; i < count; i++) {
        LucasConditions h = cpu_lucas_conditions(c[i].n, c[i].D, c[i].P, c[i].Q);
        bool ok = h.v == c[i].v && ((dev[i] & 2) != 0) == c[i].v && !h.strong && !(dev[i] & 1);
        if (!ok) {
            bad++;
            printf("    n=%llu: V_{n+1}=2Q host %d device %d (expected %d); strong host %d device %d (expected 0)\n",
                   (unsigned long long)c[i].n, (int)h.v, (dev[i] & 2) != 0, (int)c[i].v, (int)h.strong, dev[i] & 1);
        }
    }
    printf("  [Lucas-V condition: BFW Table 2 and n = 323] %d values -> %llu disagreements\n",
           count, (unsigned long long)bad);
    CK(cudaFree(d_n)); CK(cudaFree(d_dpq)); CK(cudaFree(d_out));
    return bad;
}

static uint64_t run_batch(const std::vector<uint64_t>& vals, const char* label,
                          const std::vector<int>* expect = nullptr) {
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
        // Host BPSW must agree too: it shares the Lucas halving step that once
        // overflowed above 2^63.
        bool host_bpsw = cpu_is_prime_bpsw(vals[i]);
        bool host_mr   = cpu_miller_rabin(vals[i]);
        bool wrong = expect && (host_mr != ((*expect)[i] != 0));
        if (mr[i] != bpsw[i] || host_bpsw != host_mr || (mr[i] != 0) != host_mr || wrong) {
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

    // Below 2^63, where the old halving overflow could not occur.
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

    // Every prime below 2^32 whose Selfridge search needs more than 40 values
    // of D, with the number it needs. BPSW rejected all 16 while the search was
    // capped at 40 tries. Perfect squares are the other side of that search
    // (no D exists) and must stay composite, including 1093^2, a base-2 strong
    // pseudoprime, and 4294967291^2, the largest square of a prime below 2^64.
    {
        std::vector<uint64_t> v = {
            452980999ULL,  505313251ULL,  1143791191ULL, 1272463669ULL,   // 43 49 43 47
            1373861479ULL, 1582819291ULL, 2055693949ULL, 2283397141ULL,   // 43 43 49 43
            2287905811ULL, 2366713651ULL, 2410622971ULL, 3441877651ULL,   // 43 43 47 49
            3703811101ULL, 3823259311ULL, 4131973231ULL, 4294405021ULL,   // 43 43 43 47
            1194649ULL, 12327121ULL,                     // 1093^2, 3511^2
            4294836225ULL, 18446744030759878681ULL       // 65535^2, 4294967291^2
        };
        total += run_batch(v, "long Selfridge search, and squares");
    }

    // Known verdicts, each confirmed independently with sympy 1.14.0
    // (isprime, factorint, and jacobi_symbol for the try counts):
    //   - 3825123056546413051 = 149491 * 747451 * 34233211 is a strong
    //     pseudoprime to every prime base up to 31; base 37 is what makes the
    //     12-base Miller-Rabin deterministic below 2^64, so this pins it.
    //   - five primes above 2^63 whose Selfridge search needs 68 to 82 values
    //     of D (the counts on the right), far past the 43-49 needed below 2^32,
    //     so a search capped anywhere below 82 rejects at least one of them.
    {
        std::vector<uint64_t> v = {
            3825123056546413051ULL,
            15749200944221826181ULL, 9586010881482168169ULL,           // 82 80
            9586402395203680489ULL,  9491614064492025181ULL,           // 73 73
            9624417432061496821ULL                                     // 68
        };
        std::vector<int> expect = {0, 1, 1, 1, 1, 1};
        total += run_batch(v, "known verdicts: psi_11 and long searches near 2^64", &expect);
    }

    total += run_lucas_v();

    printf("\nTOTAL DISAGREEMENTS: %llu\n", (unsigned long long)total);
    if (total) { printf("FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}
// primality.cuh
// Primality tests shared by the verifier and its tests.
// Extracted from goldbach.cu so tests exercise the production code.

#pragma once
#include <cstdint>
#include <vector>

// ---- device ----

__device__ uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)((__uint128_t)a * b % m);
}

// (a + n) / 2 for odd a, odd n, computed without overflow when n > 2^63.
__host__ __device__ __forceinline__ uint64_t halve_mod(uint64_t a, uint64_t n) {
    return (uint64_t)(((__uint128_t)a + n) >> 1);
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

// ===================================================================
// BPSW DEVICE — Jacobi, sprp-base-2, strong Lucas, full test
// ===================================================================

// Jacobi symbol (a/n), n must be odd positive.
// a is small (|D| typically < 200); n is uint64_t to handle full 64-bit range.
__device__ int gpu_jacobi(int64_t a_signed, uint64_t n) {
    if (n == 1) return 1;
    // Reduce a mod n safely with signed input
    uint64_t a;
    if (a_signed >= 0) {
        a = (uint64_t)a_signed % n;
    } else {
        uint64_t mag = (uint64_t)(-(a_signed + 1)) + 1ULL;
        uint64_t r   = mag % n;
        a = (r == 0) ? 0 : n - r;
    }
    int result = 1;
    while (a != 0) {
        while ((a & 1) == 0) {
            a >>= 1;
            uint64_t nm8 = n & 7;
            if (nm8 == 3 || nm8 == 5) result = -result;
        }
        uint64_t tmp = a; a = n; n = tmp;   // swap
        if ((a & 3) == 3 && (n & 3) == 3) result = -result;
        a %= n;
    }
    return (n == 1) ? result : 0;
}

// Strong base-2 test (uses existing mulmod64/powmod64)
__device__ bool gpu_sprp_base2(uint64_t n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if ((n & 1) == 0) return false;
    uint64_t d = n - 1, s = 0;
    while ((d & 1) == 0) { d >>= 1; s++; }
    uint64_t x = powmod64(2, d, n);
    if (x == 1 || x == n - 1) return true;
    for (uint64_t i = 1; i < s; i++) {
        x = mulmod64(x, x, n);
        if (x == n - 1) return true;
    }
    return false;
}

// Strong Lucas test — all products go through mulmod64 to avoid overflow
__device__ bool gpu_strong_lucas_prp(uint64_t n,
                                     int64_t D, int64_t P, int64_t Q) {
    uint64_t d = n + 1, s = 0;
    while ((d & 1) == 0) { d >>= 1; s++; }

    // Helper: signed int64 → uint64 reduced mod n
    auto smod = [&](int64_t x) -> uint64_t {
        if (x >= 0) return (uint64_t)x % n;
        uint64_t mag = (uint64_t)(-(x + 1)) + 1ULL;
        uint64_t r = mag % n;
        return (r == 0) ? 0 : n - r;
    };

    uint64_t uU = 0, uV = 2 % n, uQk = 1 % n;
    uint64_t uP = smod(P), uD = smod(D), uQ_base = smod(Q);

    int lead = 63 - __clzll(d);
    for (int bit = lead; bit >= 0; bit--) {
        // Double
        uint64_t U2  = mulmod64(uU, uV, n);
        uint64_t VV  = mulmod64(uV, uV, n);
        uint64_t twoQk = mulmod64(2, uQk, n);
        uint64_t V2  = (uint64_t)(((__uint128_t)VV + n - twoQk) % n);
        uint64_t Q2  = mulmod64(uQk, uQk, n);
        uU = U2; uV = V2; uQk = Q2;

        if ((d >> bit) & 1) {
            // Step
            uint64_t tU = (uint64_t)(((__uint128_t)mulmod64(uP, uU, n) + uV) % n);
            tU = (tU & 1) ? halve_mod(tU, n) : (tU >> 1);

            uint64_t tV = (uint64_t)(((__uint128_t)mulmod64(uD, uU, n) + mulmod64(uP, uV, n)) % n);
            tV = (tV & 1) ? halve_mod(tV, n) : (tV >> 1);

            uU = tU;
            uV = tV;
            uQk = mulmod64(uQk, uQ_base, n);
        }
    }

    if (uU == 0) return true;
    if (uV == 0) return true;
    for (uint64_t r = 0; r < s-1; r++) {
        uint64_t twoQk = mulmod64(2, uQk, n);
        uV  = (uint64_t)(((__uint128_t)mulmod64(uV, uV, n) + n - twoQk) % n);
        uQk = mulmod64(uQk, uQk, n);
        if (uV == 0) return true;
    }
    return false;
}

__device__ bool gpu_is_prime_bpsw(uint64_t n) {
    if (n < 2)  return false;
    if (n == 2 || n == 3) return true;
    if ((n & 1) == 0) return false;
    if (!gpu_sprp_base2(n)) return false;

    // Find D via Method A* with proper termination
    int64_t D = 5;
    int step = 2;
    for (int i = 0; i < 40; i++) {
        int j = gpu_jacobi(D, n);
        if (j == -1) break;
        if (j == 0) {
            uint64_t absD = (D >= 0) ? (uint64_t)D : (uint64_t)(-D);
            if (absD % n != 0) return false;  // genuine factor: 1 < gcd(|D|,n) < n
            // else n divides |D| (only happens for tiny n like n=5 where D=n)
            // skip this D and try the next one
        }
        D = -D - step;
        step = -step;
    }
    // After 40 tries with no -1: n is a perfect square → composite
    // (a perfect square is never prime; this guards the infinite loop)
    if (gpu_jacobi(D, n) != -1) return false;

    int64_t P = 1, Q = (1 - D) / 4;
    if (Q == -1) { P = 5; Q = 5; }

    return gpu_strong_lucas_prp(n, D, P, Q);
}

// ---- host ----

inline bool cpu_miller_rabin(uint64_t n) {
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

inline bool cpu_sprp_base2(uint64_t n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if ((n & 1) == 0) return false;
    uint64_t d = n - 1, s = 0;
    while ((d & 1) == 0) { d >>= 1; s++; }
    auto mm = [&](uint64_t a, uint64_t b) {
        return (uint64_t)((__uint128_t)a * b % n);
    };
    uint64_t x = 1, base = 2;
    for (uint64_t e = d; e; e >>= 1) {
        if (e & 1) x = mm(x, base);
        base = mm(base, base);
    }
    if (x == 1 || x == n - 1) return true;
    for (uint64_t i = 1; i < s; i++) {
        x = mm(x, x);
        if (x == n - 1) return true;
    }
    return false;
}

inline bool cpu_strong_lucas_prp(uint64_t n,
                                  int64_t D, int64_t P, int64_t Q) {
    uint64_t d = n + 1, s = 0;
    while ((d & 1) == 0) { d >>= 1; s++; }

    auto mm = [&](uint64_t a, uint64_t b) -> uint64_t {
        return (uint64_t)((__uint128_t)a * b % n);
    };
    auto smod = [&](int64_t x) -> uint64_t {
        if (x >= 0) return (uint64_t)x % n;
        uint64_t mag = (uint64_t)(-(x + 1)) + 1ULL;
        uint64_t r = mag % n;
        return (r == 0) ? 0 : n - r;
    };

    uint64_t uU = 0, uV = 2 % n, uQk = 1 % n;
    uint64_t uP = smod(P), uD = smod(D), uQ_base = smod(Q);

    int lead = 63 - __builtin_clzll(d);
    for (int bit = lead; bit >= 0; bit--) {
        uint64_t U2  = mm(uU, uV);
        uint64_t V2  = (uint64_t)(((__uint128_t)mm(uV, uV) + n - mm(2, uQk)) % n);
        uint64_t Q2  = mm(uQk, uQk);
        uU = U2; uV = V2; uQk = Q2;

        if ((d >> bit) & 1) {
            uint64_t tU = (uint64_t)(((__uint128_t)mm(uP, uU) + uV) % n);
            tU = (tU & 1) ? halve_mod(tU, n) : (tU >> 1);

            uint64_t tV = (uint64_t)(((__uint128_t)mm(uD, uU) + mm(uP, uV)) % n);
            tV = (tV & 1) ? halve_mod(tV, n) : (tV >> 1);

            uU = tU; uV = tV;
            uQk = mm(uQk, uQ_base);
        }
    }

    if (uU == 0) return true;
    if (uV == 0) return true;
    for (uint64_t r = 0; r < s-1; r++) {
        uV  = (uint64_t)(((__uint128_t)mm(uV, uV) + n - mm(2, uQk)) % n);
        uQk = mm(uQk, uQk);
        if (uV == 0) return true;
    }
    return false;
}

// Jacobi symbol (a/n), host-side — needed by cpu_is_prime_bpsw below.
inline int cpu_jacobi(int64_t a_signed, uint64_t n) {
    if (n == 1) return 1;
    uint64_t a;
    if (a_signed >= 0) {
        a = (uint64_t)a_signed % n;
    } else {
        uint64_t mag = (uint64_t)(-(a_signed + 1)) + 1ULL;
        uint64_t r   = mag % n;
        a = (r == 0) ? 0 : n - r;
    }
    int result = 1;
    while (a != 0) {
        while ((a & 1) == 0) {
            a >>= 1;
            uint64_t nm8 = n & 7;
            if (nm8 == 3 || nm8 == 5) result = -result;
        }
        uint64_t tmp = a; a = n; n = tmp;
        if ((a & 3) == 3 && (n & 3) == 3) result = -result;
        a %= n;
    }
    return (n == 1) ? result : 0;
}

inline bool cpu_is_prime_bpsw(uint64_t n) {
    if (n < 2)  return false;
    if (n == 2 || n == 3) return true;
    if ((n & 1) == 0) return false;
    if (!cpu_sprp_base2(n)) return false;

    int64_t D = 5;
    int step = 2;
    for (int i = 0; i < 40; i++) {
        int j = cpu_jacobi(D, n);
        if (j == -1) break;
        if (j == 0) {
            uint64_t absD = (D >= 0) ? (uint64_t)D : (uint64_t)(-D);
            if (absD % n != 0) return false;  // genuine factor: 1 < gcd(|D|,n) < n
            // else n divides |D| (only happens for tiny n like n=5 where D=n)
            // skip this D and try the next one
        }
        D = -D - step;
        step = -step;
    }
    if (cpu_jacobi(D, n) != -1) return false;

    int64_t P = 1, Q = (1 - D) / 4;
    if (Q == -1) { P = 5; Q = 5; }

    return cpu_strong_lucas_prp(n, D, P, Q);
}

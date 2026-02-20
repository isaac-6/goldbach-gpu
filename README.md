# goldbach-gpu

GPU-accelerated verification of Goldbach's conjecture using CUDA.

## What this does
Verifies that every even integer n >= 4 up to a configurable limit
can be expressed as the sum of two primes (Goldbach's conjecture).

## Current local Hardware
- CPU: used for prime sieve and as correctness oracle
- GPU: NVIDIA RTX 3070 (8GB VRAM), used for parallel Goldbach checking

## Results so far
| Limit | CPU time | GPU time | Speedup |
|-------|----------|----------|---------|
| 10^6  | 25ms     | 0.91ms   | 27x     |
| 10^8  | 3,819ms  | 25ms     | 153x    |

Expectedly, no counterexamples found up to 10^8 :)

## Build
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Run
```bash
./goldbach      # CPU version
./goldbach_gpu  # GPU version
./test_sieve    # Sieve validation tests
```

## Project structure
```
src/
  segmented_sieve.cpp   # Segmented sieve of Eratosthenes (CPU)
  cpu_goldbach.cpp      # CPU Goldbach verifier (correctness oracle)
  goldbach_gpu.cu       # GPU Goldbach verifier (CUDA)
  test_sieve.cpp        # Prime count validation tests
```

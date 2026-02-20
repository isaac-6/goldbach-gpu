# goldbach-gpu

GPU-accelerated verification of Goldbach's conjecture using CUDA.

**Goldbach's conjecture** states that every even integer greater than 2
can be expressed as the sum of two prime numbers. First proposed in 1742,
it remains one of the oldest unsolved problems in mathematics — verified
computationally up to 4×10^18 but never formally proved.

This project implements a GPU-accelerated verification engine using CUDA,
combining a segmented Sieve of Eratosthenes with a massively parallel
Goldbach checker. On an NVIDIA RTX 3070, we achieve ~200x speedup over
a CPU baseline.

## Hardware
- CPU: prime sieve and correctness oracle
- GPU: NVIDIA RTX 3070 (8GB VRAM), parallel Goldbach checking

## Range verification results
| Limit | CPU time | GPU time | Speedup |
|-------|----------|----------|---------|
| 10^6  | 25ms     | 0.91ms   | 27x     |
| 10^8  | 3,819ms  | 25ms     | 153x    |
| 10^9  | 42,936ms | 206ms    | 208x    |

No counterexamples found up to 10^9.

## Single number verification
Using Miller-Rabin primality test on GPU, individual large numbers
can be checked without storing a prime table in memory.

| n | Partition found | Time |
|---|----------------|------|
| 10^12 | 194267 + 999999805733 | 1.4s |
| 10^15 | 152639 + 999999999847361 | 1.4s |
| 10^18 | 14831 + 999999999999985169 | 1.5s |
| 10^19 | 226283 + 9999999999999773717 | 1.5s |

Timing is essentially constant — partitions are always found quickly
with small primes. Limited to uint64_t max (~1.8 x 10^19).

## Build
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage
```bash
./goldbach          # CPU range verifier (set LIMIT inside source)
./goldbach_gpu      # GPU range verifier (set LIMIT inside source)
./test_sieve        # Sieve validation tests
./single_check      # Check single number (default 10^12)
./single_check 999999999999999998   # Check any even number up to 1.8*10^19
```

## Project structure
```
src/
  segmented_sieve.cpp   # Segmented sieve of Eratosthenes (CPU)
  cpu_goldbach.cpp      # CPU Goldbach range verifier (correctness oracle)
  goldbach_gpu.cu       # GPU Goldbach range verifier
  single_check.cu       # GPU single number checker (Miller-Rabin)
  test_sieve.cpp        # Prime count validation tests
```

## License
MIT

# goldbach-gpu

GPU-accelerated verification of Goldbach's conjecture using CUDA and GMP.

**Goldbach's conjecture** states that every even integer greater than 2
can be expressed as the sum of two prime numbers. First proposed in 1742,
it remains one of the oldest unsolved problems in mathematics. Verified
computationally up to 4x10^18 but never formally proved.

This project implements multiple verification engines of increasing capability:
a CPU baseline, a GPU range verifier with compact bitset, a GPU single-number
checker using Miller-Rabin, and an arbitrary precision checker using GMP.
No counterexamples have been found in any computation.

## Hardware used
- CPU: Intel i7-12700H, 20 threads (WSL2, Ubuntu 24)
- GPU: NVIDIA RTX 3070 (8GB VRAM)
- RAM: 32 GB

## Results summary

### Range verification (all even n in [4, LIMIT])
| Limit | Sieve (CPU) | Kernel (GPU) | Total | Failures |
|-------|-------------|--------------|-------|----------|
| 10^6  | 6ms         | 0.91ms       | 31ms  | 0 |
| 10^8  | 1,042ms     | 25ms         | 4,861ms | 0 |
| 10^9  | 12,928ms    | 206ms        | 57,848ms | 0 |
| 10^10 | 31,238ms    | 7,893ms      | 39,131ms | 0 |
| 10^11 | 380,746ms   | 95,123ms     | 475,869ms | 0 |

GPU speedup over CPU baseline: up to 208x.

### Single number verification (GPU Miller-Rabin, uint64_t)
| n | Partition | Time |
|---|-----------|------|
| 10^18 | 14,831 + 999,999,999,999,985,169 | 1.5s |
| 10^19 | 226,283 + 9,999,999,999,999,773,717 | 1.5s |

Limited to uint64_t max (~1.8x10^19).

### Big integer verification (GMP + OpenMP, no size limit)
| n | p found | Time | Threads |
|---|---------|------|---------|
| 10^50   | 383    | 43ms    | 1  |
| 10^1000 | 26,981 | 2,299ms | 1  |
| 10^1000 | unknown| 363ms   | 20 |
| 10^10000| 47,717 | 231,051ms | 20 |

## Build
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Requires: CUDA toolkit, GMP (sudo apt install libgmp-dev), OpenMP.

## Tools and usage

### CPU range verifier
```bash
./goldbach
```
Edit LIMIT in src/cpu_goldbach.cpp to change range.

### GPU range verifier (byte array, up to 10^9)
```bash
./goldbach_gpu
```
Edit LIMIT in src/goldbach_gpu.cu to change range.

### GPU range verifier (compact bitset, up to 10^11)
```bash
./goldbach_gpu2
```
Edit LIMIT in src/goldbach_gpu2.cu to change range.

### GPU single number checker (up to 1.8x10^19)
```bash
./single_check                        # default 10^12
./single_check 999999999999999998     # any even number up to 1.8x10^19
```

### Big integer checker (any size, GMP + OpenMP)
```bash
./big_check                           # default 10^50
./big_check "123456789012345678901234567890"   # any even number as string

# Powers of 10 using Python to generate the string
./big_check "$(python3 -c "print('1' + '0'*1000)")"    # 10^1000
./big_check "$(python3 -c "print('1' + '0'*10000)")"   # 10^10000
```

### Sieve validation tests
```bash
./test_bitset     # validates prime bitset correctness
./test_sieve      # validates segmented sieve correctness
```

## Project structure
```
src/
  cpu_goldbach.cpp      # CPU range verifier (baseline)
  goldbach_gpu.cu       # GPU range verifier (byte array)
  goldbach_gpu2.cu      # GPU range verifier (compact bitset)
  single_check.cu       # GPU single number checker (Miller-Rabin)
  big_check.cpp         # Arbitrary precision checker (GMP + OpenMP)
  prime_bitset.hpp      # Compact prime bitset (1 bit per odd number)
  prime_bitset.cpp      # Bitset construction (segmented sieve + OpenMP)
  segmented_sieve.cpp   # Segmented sieve of Eratosthenes
  test_bitset.cpp       # Bitset correctness tests
  test_sieve.cpp        # Sieve correctness tests
```

## License
MIT
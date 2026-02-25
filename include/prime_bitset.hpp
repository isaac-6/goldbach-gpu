// prime_bitset.hpp
// Compact bitset representation for prime numbers.
// Stores 1 bit per odd number, saving 8x memory vs byte array.
// 2 is handled as a special case — all other even numbers are composite.
//
// Encoding:
//   bit index i  →  odd number (2*i + 3)
//   odd number n →  bit index  (n - 3) / 2
//
// Memory usage: ~n/16 bytes for primes up to n
//   10^9  →  ~60 MB
//   10^10 →  ~600 MB
//   10^11 →  ~6 GB

#pragma once
#include <cstdint>
#include <vector>
#include <cassert>

namespace goldbach {

// -------------------------------------------------------
// PrimeBitset: stores primality for odd numbers in [3, limit].
// Backed by a vector of uint64_t words.
// Note: limit must be >= 3. Even numbers are never stored —
// they are handled explicitly in is_prime().
// -------------------------------------------------------
class PrimeBitset {
public:
    // Construct a bitset covering odd numbers up to `limit`.
    // All bits initialised to 1 (all assumed prime).
    // The sieve (build_prime_bitset) will clear composite bits.
    explicit PrimeBitset(uint64_t limit)
        : limit_(limit)
        , words_((num_bits(limit) + 63) / 64, ~0ULL)
    {}

    // -------------------------------------------------------
    // Core bit operations — only valid for odd n >= 3
    // -------------------------------------------------------

    // Mark odd number n as prime
    void set(uint64_t n) {
        assert(n % 2 == 1 && n >= 3 && "set: n must be odd and >= 3");
        words_[word_idx(n)] |= (1ULL << bit_idx(n));
    }

    // Mark odd number n as composite
    void clear(uint64_t n) {
        assert(n % 2 == 1 && n >= 3 && "clear: n must be odd and >= 3");
        words_[word_idx(n)] &= ~(1ULL << bit_idx(n));
    }

    // Test if odd number n is prime
    bool test(uint64_t n) const {
        assert(n % 2 == 1 && n >= 3 && "test: n must be odd and >= 3");
        return (words_[word_idx(n)] >> bit_idx(n)) & 1ULL;
    }

    // -------------------------------------------------------
    // General primality test (handles 2 and even numbers safely)
    // This is the main public interface — use this, not test()
    // -------------------------------------------------------
    bool is_prime(uint64_t n) const {
        if (n < 2)  return false;
        if (n == 2) return true;
        if (n % 2 == 0) return false;
        if (n > limit_) return false;
        return test(n);
    }

    // -------------------------------------------------------
    // Accessors
    // -------------------------------------------------------
    uint64_t limit()       const { return limit_; }
    uint64_t word_count()  const { return words_.size(); }
    const uint64_t* data() const { return words_.data(); }
    uint64_t* data()             { return words_.data(); }

    // Memory usage in bytes
    uint64_t memory_bytes() const {
        return words_.size() * sizeof(uint64_t);
    }

private:
    uint64_t limit_;
    std::vector<uint64_t> words_;

    // Number of bits needed to cover odd numbers up to limit
    static inline constexpr uint64_t num_bits(uint64_t limit) {
        if (limit < 3) return 0;
        return (limit - 3) / 2 + 1;
    }

    // Which uint64_t word holds the bit for odd number n
    static inline constexpr uint64_t word_idx(uint64_t n) {
        return ((n - 3) / 2) / 64;
    }

    // Which bit within that word for odd number n
    static inline constexpr uint64_t bit_idx(uint64_t n) {
        return ((n - 3) / 2) % 64;
    }
};

// -------------------------------------------------------
// Build a PrimeBitset up to `limit` using segmented sieve.
// Returns a fully populated bitset ready for is_prime() queries.
// -------------------------------------------------------
PrimeBitset build_prime_bitset(uint64_t limit);

} // namespace goldbach
// prime_bitset.hpp
// Compact bitset representation for prime numbers.
// Stores 1 bit per odd number, saving ~8x memory vs byte array.
// 2 is handled as a special case — all other even numbers are composite.
//
// Encoding:
//   bit index i  →  odd number (2*i + 3)
//   odd number n →  bit index  (n - 3) / 2
//
// Memory usage examples:
//   10^9  →  ~ 60 MB
//   10^10 → ~ 600 MB
//   10^11 → ~ 6   GB
//   10^12 → ~60   GB

#pragma once
#include <cstdint>
#include <vector>
#include <cassert>

namespace goldbach {

// -------------------------------------------------------
// PrimeBitset: stores primality for odd numbers in [3, limit].
// limit may be even or odd — bitset always covers odd numbers up to the
// largest odd ≤ limit (i.e. limit if odd, limit-1 if even).
// Backed by a vector of uint64_t words.
// -------------------------------------------------------
class PrimeBitset {
public:
    // Construct a bitset covering odd numbers up to `limit`.
    // All bits initialised to 1 (assume prime); sieve will clear composites.
    explicit PrimeBitset(uint64_t limit)
        : limit_(limit)
    {
        auto bits = num_bits(limit);
        words_.reserve((bits + 63) / 64);
        words_.assign((bits + 63) / 64, ~0ULL);
    }

    // -------------------------------------------------------
    // General primality test — main public interface
    // Handles 2, evens, numbers > limit safely.
    // -------------------------------------------------------
    [[nodiscard]] bool is_prime(uint64_t n) const {
        if (n < 2)  return false;
        if (n == 2) return true;
        if (n % 2 == 0) return false;
        if (n > limit_) return false;
        return test(n);
    }

    // -------------------------------------------------------
    // Core bit operations — only for odd n >= 3
    // -------------------------------------------------------
    void set(uint64_t n) {
        assert(n % 2 == 1 && n >= 3 && "set: n must be odd and >= 3");
        words_[word_idx(n)] |= (1ULL << bit_idx(n));
    }

    void clear(uint64_t n) {
        assert(n % 2 == 1 && n >= 3 && "clear: n must be odd and >= 3");
        words_[word_idx(n)] &= ~(1ULL << bit_idx(n));
    }

    bool test(uint64_t n) const {
        assert(n % 2 == 1 && n >= 3 && "test: n must be odd and >= 3");
        return (words_[word_idx(n)] >> bit_idx(n)) & 1ULL;
    }

    // -------------------------------------------------------
    // Accessors
    // -------------------------------------------------------
    uint64_t limit()       const { return limit_; }
    uint64_t word_count()  const { return words_.size(); }
    const uint64_t* data() const { return words_.data(); }
    uint64_t* data()             { return words_.data(); }

    uint64_t memory_bytes() const {
        return words_.size() * sizeof(uint64_t);
    }

    // Number of bits needed to cover odd numbers up to limit
    // Useful for memory estimation, GPU allocation planning, etc.
    [[nodiscard]] static constexpr uint64_t num_bits(uint64_t limit) {
        if (limit < 3) return 0;
        return (limit - 3) / 2 + 1;
    }

private:
    uint64_t limit_;
    std::vector<uint64_t> words_;

    static inline constexpr uint64_t word_idx(uint64_t n) {
        return ((n - 3) / 2) / 64;
    }

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
// parse_u64.hpp
// Strict decimal parsing of command-line numbers, shared by goldbach and
// single_check.

#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>

// Digits only -- no sign, no whitespace, no prefix or suffix -- and no silent
// wrap: std::stoull accepted "-1" as 2^64 - 1 and "12abc" as 12. A malformed
// value throws std::invalid_argument; a value above 2^64 - 1 throws
// std::out_of_range. Both messages name the argument (`what`).
static inline uint64_t parse_u64(const std::string& text, const std::string& what) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(what + " must be a non-negative decimal integer, got '"
                                    + text + "'");
    uint64_t v = 0;
    for (char c : text) {
        uint64_t d = (uint64_t)(c - '0');
        if (v > (UINT64_MAX - d) / 10)
            throw std::out_of_range(what + " is out of range: '" + text + "'");
        v = v * 10 + d;
    }
    return v;
}

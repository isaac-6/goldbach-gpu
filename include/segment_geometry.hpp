// segment_geometry.hpp
// Where a segment's even numbers and sieved odd complements lie. Shared by the
// verifier and test_phase1, so the test walks segments exactly as the verifier
// does rather than a copy of that arithmetic.

#pragma once
#include <cstdint>
#include <algorithm>

struct SegmentGeometry {
    uint64_t seg_start;       // first even number of the segment
    uint64_t seg_end;         // last even number, clipped to LIMIT
    uint64_t seg_even_count;  // (seg_end - seg_start) / 2 + 1
    uint64_t q_low;           // first odd number sieved: seg_start - P_SMALL, odd, >= 3
    uint64_t q_high;          // last odd number sieved: seg_end + 1
    uint64_t num_odds;        // odd numbers in [q_low, q_high]
};

// seg_start is even and <= limit; seg_size is the segment's even-number count.
// Overflow-free for limit <= 2^64 - 2^33 and seg_size < 2^32, which the
// verifier enforces (see MAX_LIMIT in goldbach.cu).
static inline SegmentGeometry segment_geometry(uint64_t seg_start, uint64_t seg_size,
                                               uint64_t limit, uint64_t p_small)
{
    SegmentGeometry g;
    g.seg_start      = seg_start;
    g.seg_end        = std::min(seg_start + seg_size * 2 - 2, limit);
    g.seg_even_count = (g.seg_end - seg_start) / 2 + 1;

    g.q_low = (seg_start > p_small ? seg_start - p_small : 3);
    if ((g.q_low & 1) == 0) g.q_low++;
    g.q_high = (g.seg_end < UINT64_MAX - 1) ? g.seg_end + 1 : g.seg_end;
    if ((g.q_high & 1) == 0) g.q_high++;

    g.num_odds = (g.q_high - g.q_low) / 2 + 1;
    return g;
}

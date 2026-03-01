#pragma once
#include <cstdint>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

struct SegmentTask {
    uint64_t seg_start;
    uint64_t seg_end;
};

class WorkQueue {
public:
    void push(const SegmentTask& t) {
        std::lock_guard<std::mutex> lock(m_);
        q_.push(t);
        cv_.notify_one();
    }

    std::optional<SegmentTask> pop() {
        std::unique_lock<std::mutex> lock(m_);
        while (q_.empty() && !done_) {
            cv_.wait(lock);
        }
        if (q_.empty()) return std::nullopt;
        SegmentTask t = q_.front();
        q_.pop();
        return t;
    }

    void set_done() {
        std::lock_guard<std::mutex> lock(m_);
        done_ = true;
        cv_.notify_all();
    }

private:
    std::queue<SegmentTask> q_;
    std::mutex m_;
    std::condition_variable cv_;
    bool done_ = false;
};

struct SegmentResult {
    uint64_t evens_checked = 0;
    uint64_t failures = 0;
    uint64_t phase2_fallbacks = 0;
};

struct GlobalResults {
    uint64_t evens_checked = 0;
    uint64_t failures = 0;
    uint64_t phase2_fallbacks = 0;
    std::mutex m;

    void merge(const SegmentResult& r) {
        std::lock_guard<std::mutex> lock(m);
        evens_checked      += r.evens_checked;
        failures           += r.failures;
        phase2_fallbacks   += r.phase2_fallbacks;
    }
};
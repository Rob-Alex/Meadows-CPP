#pragma once
#include <chrono>
#include <vector>
#include <cstddef>
#include <cstdio>
#include <cmath>
#include <string>
#include "memory.hpp"

namespace bm {

// High-resolution wall-clock timer
struct Timer {
  using clock = std::chrono::high_resolution_clock;
  clock::time_point _start;

  Timer() : _start(clock::now()) {}

  double elapsed_ms() const {
    return std::chrono::duration<double, std::milli>(
        clock::now() - _start).count();
  }
};

// Prevent the compiler from optimising away a value
template<typename T>
void do_not_optimise(T const& val) {
  asm volatile("" : : "r,m"(val) : "memory");
}

// Statistics from a vector of measurements
struct Stats {
  double mean_ms  = 0.0;
  double stddev_ms = 0.0;
  double min_ms   = 0.0;
  double max_ms   = 0.0;
};

inline Stats compute_stats(const std::vector<double>& times) {
  Stats s;
  s.min_ms = times[0];
  s.max_ms = times[0];
  for (double t : times) {
    s.mean_ms += t;
    if (t < s.min_ms) s.min_ms = t;
    if (t > s.max_ms) s.max_ms = t;
  }
  s.mean_ms /= static_cast<double>(times.size());
  for (double t : times) {
    double d = t - s.mean_ms;
    s.stddev_ms += d * d;
  }
  s.stddev_ms = std::sqrt(s.stddev_ms / static_cast<double>(times.size()));
  return s;
}

// Run f() warmup times (discarded), then repeats times (recorded)
template<typename F>
Stats time_repeated(F&& f, int warmup = 1, int repeats = 5) {
  for (int i = 0; i < warmup; ++i) f();
  std::vector<double> times(static_cast<size_t>(repeats));
  for (int i = 0; i < repeats; ++i) {
    Timer t;
    f();
    times[static_cast<size_t>(i)] = t.elapsed_ms();
  }
  return compute_stats(times);
}

// Memory state snapshot (uses track_allocations directly)
struct MemSnapshot {
  size_t live_count;       // currently live allocations
  size_t live_bytes;       // currently live bytes
  size_t event_count;      // alloc events since last reset_event_counters()
  size_t total_bytes_ever; // bytes allocated since last reset
};

inline MemSnapshot capture_memory() {
  return {
    track_allocations::get_active_count(),
    track_allocations::get_total_bytes(),
    track_allocations::event_count().load(std::memory_order_relaxed),
    track_allocations::total_bytes_ever().load(std::memory_order_relaxed)
  };
}

inline void reset_memory_events() {
  track_allocations::reset_event_counters();
}

// Pretty-print helpers
inline void print_separator(int width = 80) {
  for (int i = 0; i < width; ++i) std::putchar('=');
  std::putchar('\n');
}

inline void print_header(const char* title) {
  print_separator();
  std::printf("%s\n", title);
  print_separator();
}

inline std::string human_bytes(size_t bytes) {
  char buf[32];
  if (bytes >= 1024ULL * 1024 * 1024)
    std::snprintf(buf, sizeof(buf), "%.2f GB",
                  static_cast<double>(bytes) / (1024.0 * 1024 * 1024));
  else if (bytes >= 1024ULL * 1024)
    std::snprintf(buf, sizeof(buf), "%.2f MB",
                  static_cast<double>(bytes) / (1024.0 * 1024));
  else if (bytes >= 1024ULL)
    std::snprintf(buf, sizeof(buf), "%.2f KB",
                  static_cast<double>(bytes) / 1024.0);
  else
    std::snprintf(buf, sizeof(buf), "%zu B", bytes);
  return std::string(buf);
}

} // namespace bm

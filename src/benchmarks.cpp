#include "elliptical_solve_bm.hpp"
#include "memory_bm.hpp"
#include <cstring>

int main(int argc, char** argv) {
  const char* suite = (argc > 1) ? argv[1] : "all";

  if (std::strcmp(suite, "mms") == 0 || std::strcmp(suite, "all") == 0)
    bm::run_mms_benchmarks();

  if (std::strcmp(suite, "scale") == 0 || std::strcmp(suite, "all") == 0)
    bm::run_scaling_benchmarks();

  if (std::strcmp(suite, "memory") == 0 || std::strcmp(suite, "all") == 0) {
    bm::run_allocator_microbenchmarks();
    bm::run_memory_profile_benchmarks();
  }

  if (std::strcmp(suite, "threads") == 0 || std::strcmp(suite, "all") == 0)
    bm::run_thread_scaling_benchmarks();

  if (std::strcmp(suite, "quick") == 0) {
    bm::run_mms_benchmarks(false);
    bm::run_scaling_benchmarks(false);
  }

  return 0;
}

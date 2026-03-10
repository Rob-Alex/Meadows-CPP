/*
  meadows-cpp benchmark driver.

  Runs the full benchmark suite for the geometric multigrid solver:
    1. Convergence profile (residual vs iteration, convergence factors)
    2. Grid scaling (solve time vs N, DOF/s throughput)
    3. Solution space complexity (O(N) verification, asymptotic conv rates)
    4. Memory profile (allocations and bytes vs N)
    5. Per-level timing breakdown (smoother, restriction, etc.)
    6. Strong scaling / thread scaling (fixed problem, vary threads)
    7. Weak scaling (fixed DOF/thread, grow problem with threads)

  Usage:
    ./build/benchmark                   — run all benchmarks
    ./build/benchmark convergence [N]   — convergence profile (default N=128)
    ./build/benchmark scaling           — grid scaling only
    ./build/benchmark solution          — solution space complexity
    ./build/benchmark memory            — memory profile only
    ./build/benchmark levels [N]        — per-level breakdown (default N=256)
    ./build/benchmark threads [N]       — strong scaling (requires OMP)
    ./build/benchmark weak [N]          — weak scaling (requires OMP)
    ./build/benchmark quick             — fast sanity check (small grids)

  All results are written as CSV to build/benchmarks/ for
  plotting with matplotlib, pgfplots, or similar tools.

  Compile:
    make benchmark-build            — single-threaded
    make benchmark-build OMP=1      — with OpenMP
  Run:
    make benchmark                  — build + run all
    make benchmark BENCH_MODE=quick — build + run quick
*/
#include <cstdio>
#include <cstring>
#include "benchmark.hpp"

int main(int argc, char* argv[]) {
  using Suite = BenchmarkSuite<double, 2>;
  Suite::Config cfg;
  cfg.output_dir = "build/benchmarks";
  cfg.n_warmup = 1;
  cfg.n_repeat = 5;

  Suite suite(cfg);

  const char* mode = (argc > 1) ? argv[1] : "all";

  if (std::strcmp(mode, "all") == 0) {
    suite.run_all();
  }
  else if (std::strcmp(mode, "convergence") == 0) {
    int N = (argc > 2) ? std::atoi(argv[2]) : 128;
    suite.run_convergence_profile(N);
  }
  else if (std::strcmp(mode, "scaling") == 0) {
    suite.run_grid_scaling({16, 32, 64, 128, 256, 512});
  }
  else if (std::strcmp(mode, "solution") == 0) {
    suite.run_solution_space({16, 32, 64, 128, 256, 512});
  }
  else if (std::strcmp(mode, "memory") == 0) {
    suite.run_memory_profile({16, 32, 64, 128, 256, 512});
  }
  else if (std::strcmp(mode, "levels") == 0) {
    int N = (argc > 2) ? std::atoi(argv[2]) : 256;
    suite.run_level_profile(N);
  }
  else if (std::strcmp(mode, "threads") == 0) {
    int N = (argc > 2) ? std::atoi(argv[2]) : 256;
    suite.run_thread_scaling(N, {1, 2, 4, 8});
  }
  else if (std::strcmp(mode, "weak") == 0) {
    int N = (argc > 2) ? std::atoi(argv[2]) : 128;
    suite.run_weak_scaling(N, {1, 2, 4, 8});
  }
  else if (std::strcmp(mode, "quick") == 0) {
    cfg.n_warmup = 0;
    cfg.n_repeat = 1;
    Suite quick_suite(cfg);
    quick_suite.run_convergence_profile(64);
    quick_suite.run_grid_scaling({16, 32, 64});
    quick_suite.run_solution_space({16, 32, 64});
    quick_suite.run_memory_profile({16, 32, 64});
  }
  else {
    std::printf("Unknown mode: %s\n", mode);
    std::printf("Usage: %s [all|convergence|scaling|solution|memory|levels|threads|weak|quick] [N]\n",
                argv[0]);
    return 1;
  }

  return 0;
}

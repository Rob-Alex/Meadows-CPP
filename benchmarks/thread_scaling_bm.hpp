#pragma once
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include "solver.hpp"
#include "bm_utils.hpp"
#include "mms_bm.hpp"  // for n_levels_for

#ifdef _OPENMP
#include <omp.h>
#endif

namespace bm {

#ifdef _OPENMP

// Run solver `repeats` times, return median elapsed ms
template<typename SolverT>
static double median_solve_ms(SolverT& solver, int repeats) {
  std::vector<double> times;
  times.reserve(static_cast<size_t>(repeats));
  for (int r = 0; r < repeats; ++r) {
    Timer t;
    solver.solve_impl();
    times.push_back(t.elapsed_ms());
  }
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

// -------------------------------------------------------------------------
// Strong scaling: fixed 2D 1024x1024 problem
// -------------------------------------------------------------------------
inline void run_strong_scaling() {
  const int N = 1024;
  const double h = 1.0 / N;
  const int n_levels = n_levels_for(N);
  const int max_threads = omp_get_max_threads();

  std::printf("\n=== Strong Scaling: 2D %dx%d ===\n", N, N);
  std::printf("  Threads  solve_ms   speedup  efficiency\n");
  std::printf("  -------  ---------  -------  ----------\n");

  using Alloc = HostAllocator<double, no_tracking>;
  GridGeometry<double, 2> geom({h / 2.0, h / 2.0}, {h, h}, {N, N}, 0, 1);

  GridHierarchy<double, 2, Alloc> hier;
  int phi_idx = hier.register_component("phi");
  int rhs_idx = hier.register_component("rhs");
  int res_idx = hier.register_component("res");
  hier.build(geom, n_levels);

  {
    const double pi = M_PI, pi2 = pi * pi;
    auto rhs_acc = hier.finest().accessor(rhs_idx);
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < N; ++i) {
        double x = (i + 0.5) * h, y = (j + 0.5) * h;
        rhs_acc(i, j) = -2.0 * pi2 * std::sin(pi * x) * std::sin(pi * y);
      }
  }

  BCRegistry<double, 2> bc;
  DirichletBC<double, 2> d0; d0.value = 0.0;
  for (int dim = 0; dim < 2; ++dim) { bc.set(dim, 0, d0); bc.set(dim, 1, d0); }

  EllipticSolverGMG<double, 2, Alloc> solver(
      hier, bc, phi_idx, rhs_idx, res_idx,
      2, 2, 20, 50, 1e-10, false);

  double t1 = 0.0;

  const int thread_counts[] = {1, 2, 4, 8, 16, 32};
  for (int tc : thread_counts) {
    if (tc > max_threads) break;
    omp_set_num_threads(tc);

    double ms = median_solve_ms(solver, 3);
    if (tc == 1) t1 = ms;

    double speedup    = t1 / ms;
    double efficiency = speedup / static_cast<double>(tc) * 100.0;

    std::printf("  %7d  %9.2f  %6.2fx  %9.0f%%\n",
                tc, ms, speedup, efficiency);
  }

  omp_set_num_threads(max_threads);
}

// -------------------------------------------------------------------------
// Weak scaling: 128x128 DOFs per thread (total grows with thread count)
// -------------------------------------------------------------------------
inline void run_weak_scaling() {
  const int base = 128;
  const int max_threads = omp_get_max_threads();

  std::printf("\n=== Weak Scaling: 2D (%dx%d per thread) ===\n", base, base);
  std::printf("  Threads  DOFs/thrd    total_DOFs   solve_ms  efficiency\n");
  std::printf("  -------  -----------  -----------  --------  ----------\n");

  double t1 = 0.0;

  const int thread_counts[] = {1, 2, 4, 8, 16, 32};
  for (int tc : thread_counts) {
    if (tc > max_threads) break;
    omp_set_num_threads(tc);

    // N s.t. N^2 ≈ base^2 * tc; round to nearest multiple of 32
    // (32 = 2^(n_levels-1) = divisibility requirement for 6-level hierarchy)
    int N;
    {
      int target = static_cast<int>(std::round(base * std::sqrt(static_cast<double>(tc))));
      N = std::max(base, (target / 32) * 32);
      if (N == 0) N = 32;
    }
    double h = 1.0 / N;
    int n_levels = n_levels_for(N);
    long long total_dofs      = static_cast<long long>(N) * N;
    long long dofs_per_thread = total_dofs / tc;

    using Alloc = HostAllocator<double, no_tracking>;
    GridGeometry<double, 2> geom({h / 2.0, h / 2.0}, {h, h}, {N, N}, 0, 1);

    GridHierarchy<double, 2, Alloc> hier;
    int phi_idx = hier.register_component("phi");
    int rhs_idx = hier.register_component("rhs");
    int res_idx = hier.register_component("res");
    hier.build(geom, n_levels);

    {
      const double pi = M_PI, pi2 = pi * pi;
      auto rhs_acc = hier.finest().accessor(rhs_idx);
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
          double x = (i + 0.5) * h, y = (j + 0.5) * h;
          rhs_acc(i, j) = -2.0 * pi2 * std::sin(pi * x) * std::sin(pi * y);
        }
    }

    BCRegistry<double, 2> bc;
    DirichletBC<double, 2> d0; d0.value = 0.0;
    for (int dim = 0; dim < 2; ++dim) { bc.set(dim, 0, d0); bc.set(dim, 1, d0); }

    EllipticSolverGMG<double, 2, Alloc> solver(
        hier, bc, phi_idx, rhs_idx, res_idx,
        2, 2, 20, 50, 1e-10, false);

    double ms = median_solve_ms(solver, 3);
    // Efficiency: normalise by work done so varying DOFs/thread don't skew it
    // efficiency = (t1 / dofs_per_thread_1) / (ms / dofs_per_thread) * 100
    static double t1_per_dof = 0.0;
    if (tc == 1) { t1 = ms; t1_per_dof = ms / static_cast<double>(dofs_per_thread); }
    double efficiency = t1_per_dof / (ms / static_cast<double>(dofs_per_thread)) * 100.0;

    std::printf("  %7d  %11lld  %11lld  %8.2f  %9.0f%%\n",
                tc, dofs_per_thread, total_dofs, ms, efficiency);
  }

  omp_set_num_threads(max_threads);
}

inline void run_thread_scaling_benchmarks() {
  print_header("OpenMP Thread Scaling Benchmarks");
  std::printf("  Max threads available: %d\n", omp_get_max_threads());
  run_strong_scaling();
  run_weak_scaling();
  std::printf("\n");
}

#else  // !_OPENMP

inline void run_thread_scaling_benchmarks() {
  std::printf("\n[Thread scaling skipped: compiled without -fopenmp]\n\n");
}

#endif  // _OPENMP

} // namespace bm

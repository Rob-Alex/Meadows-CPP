#pragma once
#include <cstdio>
#include <cmath>
#include "solver.hpp"
#include "bm_utils.hpp"
#include "mms_bm.hpp"  // for n_levels_for

namespace bm {

// -------------------------------------------------------------------------
// 1D scaling benchmark: wall time vs N for Poisson solve
// -------------------------------------------------------------------------

inline void run_scaling_1d() {
  std::printf("\n=== Scaling Benchmark: 1D ===\n");
  std::printf("   N          DOFs         solve_ms   ±stddev   DOFs/s        ms/MDOF\n");
  std::printf("   ---------  -----------  ---------  --------  ------------  -------\n");

  const int sizes[] = {64, 256, 1024, 4096, 16384, 65536, 262144};

  for (int N : sizes) {
    double h = 1.0 / N;
    int n_levels = n_levels_for(N);
    long long dofs = static_cast<long long>(N);

    using Alloc = HostAllocator<double, no_tracking>;
    GridGeometry<double, 1> geom({h / 2.0}, {h}, {N}, 0, 1);

    GridHierarchy<double, 1, Alloc> hier;
    int phi_idx = hier.register_component("phi");
    int rhs_idx = hier.register_component("rhs");
    int res_idx = hier.register_component("res");
    hier.build(geom, n_levels);

    // RHS = -pi^2 * sin(pi*x) (any smooth function works for timing)
    {
      const double pi = M_PI, pi2 = pi * pi;
      auto rhs_acc = hier.finest().accessor(rhs_idx);
      for (int i = 0; i < N; ++i) {
        double x = (i + 0.5) * h;
        rhs_acc(i) = -pi2 * std::sin(pi * x);
      }
    }

    BCRegistry<double, 1> bc;
    DirichletBC<double, 1> d0; d0.value = 0.0;
    bc.set(0, 0, d0); bc.set(0, 1, d0);

    EllipticSolverGMG<double, 1, Alloc> solver(
        hier, bc, phi_idx, rhs_idx, res_idx,
        2, 2, 20, 50, 1e-10, false);

    // Fewer repeats for large grids to keep total time reasonable
    int warmup  = (N <= 16384)  ? 1 : 1;
    int repeats = (N <= 4096)   ? 5 :
                  (N <= 65536)  ? 3 : 2;

    Stats s = time_repeated([&]() { solver.solve_impl(); }, warmup, repeats);

    double dofs_per_sec = static_cast<double>(dofs) / (s.mean_ms / 1000.0);
    double ms_per_mdof  = s.mean_ms / (static_cast<double>(dofs) / 1.0e6);

    std::printf("  %9d  %11lld  %9.3f  %8.3f  %12.3e  %7.1f\n",
                N, dofs, s.mean_ms, s.stddev_ms, dofs_per_sec, ms_per_mdof);
  }
}

// -------------------------------------------------------------------------
// 2D scaling benchmark
// -------------------------------------------------------------------------

inline void run_scaling_2d() {
  std::printf("\n=== Scaling Benchmark: 2D ===\n");
  std::printf("   N      DOFs         solve_ms   ±stddev   DOFs/s        ms/MDOF\n");
  std::printf("   -----  -----------  ---------  --------  ------------  -------\n");

  const int sizes[] = {32, 64, 128, 256, 512, 1024, 2048};

  for (int N : sizes) {
    double h = 1.0 / N;
    int n_levels = n_levels_for(N);
    long long dofs = static_cast<long long>(N) * N;

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

    int warmup  = 1;
    int repeats = (N <= 128)  ? 5 :
                  (N <= 512)  ? 3 :
                  (N <= 1024) ? 2 : 1;

    Stats s = time_repeated([&]() { solver.solve_impl(); }, warmup, repeats);

    double dofs_per_sec = static_cast<double>(dofs) / (s.mean_ms / 1000.0);
    double ms_per_mdof  = s.mean_ms / (static_cast<double>(dofs) / 1.0e6);

    std::printf("  %5d  %11lld  %9.3f  %8.3f  %12.3e  %7.1f\n",
                N, dofs, s.mean_ms, s.stddev_ms, dofs_per_sec, ms_per_mdof);
  }
}

// -------------------------------------------------------------------------
// 3D scaling benchmark
// -------------------------------------------------------------------------

inline void run_scaling_3d() {
  std::printf("\n=== Scaling Benchmark: 3D ===\n");
  std::printf("   N      DOFs         solve_ms   ±stddev   DOFs/s        ms/MDOF\n");
  std::printf("   -----  -----------  ---------  --------  ------------  -------\n");

  const int sizes[] = {16, 32, 64, 128, 256};

  for (int N : sizes) {
    double h = 1.0 / N;
    int n_levels = n_levels_for(N);
    long long dofs = static_cast<long long>(N) * N * N;

    using Alloc = HostAllocator<double, no_tracking>;
    GridGeometry<double, 3> geom(
        {h / 2.0, h / 2.0, h / 2.0}, {h, h, h}, {N, N, N}, 0, 1);

    GridHierarchy<double, 3, Alloc> hier;
    int phi_idx = hier.register_component("phi");
    int rhs_idx = hier.register_component("rhs");
    int res_idx = hier.register_component("res");
    hier.build(geom, n_levels);

    {
      const double pi = M_PI, pi2 = pi * pi;
      auto rhs_acc = hier.finest().accessor(rhs_idx);
      for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
          for (int i = 0; i < N; ++i) {
            double x=(i+0.5)*h, y=(j+0.5)*h, z=(k+0.5)*h;
            rhs_acc(i,j,k) = -3.0 * pi2 *
                std::sin(pi*x) * std::sin(pi*y) * std::sin(pi*z);
          }
    }

    BCRegistry<double, 3> bc;
    DirichletBC<double, 3> d0; d0.value = 0.0;
    for (int dim = 0; dim < 3; ++dim) { bc.set(dim, 0, d0); bc.set(dim, 1, d0); }

    EllipticSolverGMG<double, 3, Alloc> solver(
        hier, bc, phi_idx, rhs_idx, res_idx,
        2, 2, 20, 50, 1e-10, false);

    int warmup  = 1;
    int repeats = (N <= 32)  ? 3 :
                  (N <= 64)  ? 2 : 1;

    Stats s = time_repeated([&]() { solver.solve_impl(); }, warmup, repeats);

    double dofs_per_sec = static_cast<double>(dofs) / (s.mean_ms / 1000.0);
    double ms_per_mdof  = s.mean_ms / (static_cast<double>(dofs) / 1.0e6);

    std::printf("  %5d  %11lld  %9.3f  %8.3f  %12.3e  %7.1f\n",
                N, dofs, s.mean_ms, s.stddev_ms, dofs_per_sec, ms_per_mdof);
  }
}

// Top-level: runs 1D, 2D, and optionally 3D
inline void run_scaling_benchmarks(bool do_3d = true) {
  print_header("Grid-Size Scaling Benchmarks (O(N) validation)");
  run_scaling_1d();
  run_scaling_2d();
  if (do_3d) run_scaling_3d();
  std::printf("\n");
}

} // namespace bm

#pragma once
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include "solver.hpp"
#include "bm_utils.hpp"

namespace bm {

// n_levels_for: choose hierarchy depth so coarsest has >= 4 cells; cap at 6
inline int n_levels_for(int N) {
  return std::min(6, std::max(2, static_cast<int>(std::log2(static_cast<double>(N))) - 1));
}

// -------------------------------------------------------------------------
// MMS 1D: phi_exact = sin(pi*x), RHS = -pi^2 * sin(pi*x)
// BCs: Dirichlet 0 at x=0 and x=1
// -------------------------------------------------------------------------

inline void run_mms_1d() {
  const double pi = M_PI;
  const double pi2 = pi * pi;

  std::printf("\n=== MMS Benchmark: 1D ===\n");
  std::printf("   N      h           max_error   L2_error    rate   solve_ms  vcycles  conv_fac\n");
  std::printf("   -----  ----------  ----------  ----------  -----  --------  -------  --------\n");

  const int sizes[] = {8, 16, 32, 64, 128, 256, 512, 1024};

  double prev_max_err = -1.0;

  for (int N : sizes) {
    double h = 1.0 / N;
    int n_levels = n_levels_for(N);

    using Alloc = HostAllocator<double, no_tracking>;
    GridGeometry<double, 1> finest_geom(
        {h / 2.0}, {h}, {N}, 0, 1);

    GridHierarchy<double, 1, Alloc> hier;
    int phi_idx = hier.register_component("phi");
    int rhs_idx = hier.register_component("rhs");
    int res_idx = hier.register_component("res");
    hier.build(finest_geom, n_levels);

    // Set RHS on finest level
    auto rhs_acc = hier.finest().accessor(rhs_idx);
    for (int i = 0; i < N; ++i) {
      double x = (i + 0.5) * h;
      rhs_acc(i) = -pi2 * std::sin(pi * x);
    }

    // Boundary conditions: Dirichlet 0 everywhere
    BCRegistry<double, 1> bc;
    DirichletBC<double, 1> d0; d0.value = 0.0;
    bc.set(0, 0, d0);
    bc.set(0, 1, d0);

    EllipticSolverGMG<double, 1, Alloc> solver(
        hier, bc, phi_idx, rhs_idx, res_idx,
        2, 2, 30, 100, 1e-10, false);

    // Track convergence info via callback
    int vcycles = 0;
    double prev_rn = 1.0, conv_factor = 0.0;
    bool have_two = false;

    Timer timer;
    solver.solve_impl([&](int cycle, double rn) {
      if (cycle > 0) {
        vcycles = cycle;
        if (have_two) conv_factor = rn / prev_rn;
        prev_rn = rn;
        have_two = true;
      }
    });
    double solve_ms = timer.elapsed_ms();

    // Compute max and L2 errors
    auto phi_acc = hier.finest().accessor(phi_idx);
    double max_err = 0.0, sum_sq = 0.0;
    for (int i = 0; i < N; ++i) {
      double x = (i + 0.5) * h;
      double exact = std::sin(pi * x);
      double diff = std::abs(phi_acc(i) - exact);
      max_err = std::max(max_err, diff);
      sum_sq += diff * diff;
    }
    double l2_err = std::sqrt(h * sum_sq);

    // Convergence rate
    double rate = (prev_max_err > 0.0)
                  ? std::log2(prev_max_err / max_err)
                  : 0.0;

    if (prev_max_err < 0.0) {
      std::printf("  %5d  %10.3e  %10.3e  %10.3e  ---    %8.2f  %7d  %8.2e\n",
                  N, h, max_err, l2_err, solve_ms, vcycles, conv_factor);
    } else {
      std::printf("  %5d  %10.3e  %10.3e  %10.3e  %5.2f  %8.2f  %7d  %8.2e\n",
                  N, h, max_err, l2_err, rate, solve_ms, vcycles, conv_factor);
    }

    prev_max_err = max_err;
  }
}

// -------------------------------------------------------------------------
// MMS 2D: phi_exact = sin(pi*x)*sin(pi*y), RHS = -2*pi^2 * phi_exact
// BCs: Dirichlet 0 on all four faces
// -------------------------------------------------------------------------

inline void run_mms_2d() {
  const double pi = M_PI;
  const double pi2 = pi * pi;

  std::printf("\n=== MMS Benchmark: 2D ===\n");
  std::printf("   N      h           max_error   L2_error    rate   solve_ms  vcycles  conv_fac\n");
  std::printf("   -----  ----------  ----------  ----------  -----  --------  -------  --------\n");

  const int sizes[] = {8, 16, 32, 64, 128, 256, 512};

  double prev_max_err = -1.0;

  for (int N : sizes) {
    double h = 1.0 / N;
    int n_levels = n_levels_for(N);

    using Alloc = HostAllocator<double, no_tracking>;
    GridGeometry<double, 2> finest_geom(
        {h / 2.0, h / 2.0}, {h, h}, {N, N}, 0, 1);

    GridHierarchy<double, 2, Alloc> hier;
    int phi_idx = hier.register_component("phi");
    int rhs_idx = hier.register_component("rhs");
    int res_idx = hier.register_component("res");
    hier.build(finest_geom, n_levels);

    // Set RHS on finest level
    auto rhs_acc = hier.finest().accessor(rhs_idx);
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < N; ++i) {
        double x = (i + 0.5) * h;
        double y = (j + 0.5) * h;
        rhs_acc(i, j) = -2.0 * pi2 * std::sin(pi * x) * std::sin(pi * y);
      }
    }

    // Boundary conditions: Dirichlet 0 everywhere
    BCRegistry<double, 2> bc;
    DirichletBC<double, 2> d0; d0.value = 0.0;
    for (int dim = 0; dim < 2; ++dim) {
      bc.set(dim, 0, d0);
      bc.set(dim, 1, d0);
    }

    EllipticSolverGMG<double, 2, Alloc> solver(
        hier, bc, phi_idx, rhs_idx, res_idx,
        2, 2, 30, 100, 1e-10, false);

    int vcycles = 0;
    double prev_rn = 1.0, conv_factor = 0.0;
    bool have_two = false;

    Timer timer;
    solver.solve_impl([&](int cycle, double rn) {
      if (cycle > 0) {
        vcycles = cycle;
        if (have_two) conv_factor = rn / prev_rn;
        prev_rn = rn;
        have_two = true;
      }
    });
    double solve_ms = timer.elapsed_ms();

    // Compute errors
    auto phi_acc = hier.finest().accessor(phi_idx);
    double max_err = 0.0, sum_sq = 0.0;
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < N; ++i) {
        double x = (i + 0.5) * h;
        double y = (j + 0.5) * h;
        double exact = std::sin(pi * x) * std::sin(pi * y);
        double diff = std::abs(phi_acc(i, j) - exact);
        max_err = std::max(max_err, diff);
        sum_sq += diff * diff;
      }
    }
    double l2_err = std::sqrt(h * h * sum_sq);

    double rate = (prev_max_err > 0.0)
                  ? std::log2(prev_max_err / max_err)
                  : 0.0;

    if (prev_max_err < 0.0) {
      std::printf("  %5d  %10.3e  %10.3e  %10.3e  ---    %8.2f  %7d  %8.2e\n",
                  N, h, max_err, l2_err, solve_ms, vcycles, conv_factor);
    } else {
      std::printf("  %5d  %10.3e  %10.3e  %10.3e  %5.2f  %8.2f  %7d  %8.2e\n",
                  N, h, max_err, l2_err, rate, solve_ms, vcycles, conv_factor);
    }

    prev_max_err = max_err;
  }
}

// -------------------------------------------------------------------------
// MMS 3D: phi_exact = sin(pi*x)*sin(pi*y)*sin(pi*z), RHS = -3*pi^2 * phi_exact
// BCs: Dirichlet 0 on all six faces
// -------------------------------------------------------------------------

inline void run_mms_3d() {
  const double pi = M_PI;
  const double pi2 = pi * pi;

  std::printf("\n=== MMS Benchmark: 3D ===\n");
  std::printf("   N      h           max_error   L2_error    rate   solve_ms  vcycles  conv_fac\n");
  std::printf("   -----  ----------  ----------  ----------  -----  --------  -------  --------\n");

  const int sizes[] = {8, 16, 32, 64, 128};

  double prev_max_err = -1.0;

  for (int N : sizes) {
    double h = 1.0 / N;
    int n_levels = n_levels_for(N);

    using Alloc = HostAllocator<double, no_tracking>;
    GridGeometry<double, 3> finest_geom(
        {h / 2.0, h / 2.0, h / 2.0}, {h, h, h}, {N, N, N}, 0, 1);

    GridHierarchy<double, 3, Alloc> hier;
    int phi_idx = hier.register_component("phi");
    int rhs_idx = hier.register_component("rhs");
    int res_idx = hier.register_component("res");
    hier.build(finest_geom, n_levels);

    // Set RHS on finest level
    auto rhs_acc = hier.finest().accessor(rhs_idx);
    for (int k = 0; k < N; ++k) {
      for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
          double x = (i + 0.5) * h;
          double y = (j + 0.5) * h;
          double z = (k + 0.5) * h;
          rhs_acc(i, j, k) =
              -3.0 * pi2 * std::sin(pi * x) * std::sin(pi * y) * std::sin(pi * z);
        }
      }
    }

    // Boundary conditions: Dirichlet 0 everywhere
    BCRegistry<double, 3> bc;
    DirichletBC<double, 3> d0; d0.value = 0.0;
    for (int dim = 0; dim < 3; ++dim) {
      bc.set(dim, 0, d0);
      bc.set(dim, 1, d0);
    }

    EllipticSolverGMG<double, 3, Alloc> solver(
        hier, bc, phi_idx, rhs_idx, res_idx,
        2, 2, 30, 100, 1e-10, false);

    int vcycles = 0;
    double prev_rn = 1.0, conv_factor = 0.0;
    bool have_two = false;

    Timer timer;
    solver.solve_impl([&](int cycle, double rn) {
      if (cycle > 0) {
        vcycles = cycle;
        if (have_two) conv_factor = rn / prev_rn;
        prev_rn = rn;
        have_two = true;
      }
    });
    double solve_ms = timer.elapsed_ms();

    // Compute errors
    auto phi_acc = hier.finest().accessor(phi_idx);
    double max_err = 0.0, sum_sq = 0.0;
    for (int k = 0; k < N; ++k) {
      for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
          double x = (i + 0.5) * h;
          double y = (j + 0.5) * h;
          double z = (k + 0.5) * h;
          double exact = std::sin(pi * x) * std::sin(pi * y) * std::sin(pi * z);
          double diff = std::abs(phi_acc(i, j, k) - exact);
          max_err = std::max(max_err, diff);
          sum_sq += diff * diff;
        }
      }
    }
    double l2_err = std::sqrt(h * h * h * sum_sq);

    double rate = (prev_max_err > 0.0)
                  ? std::log2(prev_max_err / max_err)
                  : 0.0;

    if (prev_max_err < 0.0) {
      std::printf("  %5d  %10.3e  %10.3e  %10.3e  ---    %8.2f  %7d  %8.2e\n",
                  N, h, max_err, l2_err, solve_ms, vcycles, conv_factor);
    } else {
      std::printf("  %5d  %10.3e  %10.3e  %10.3e  %5.2f  %8.2f  %7d  %8.2e\n",
                  N, h, max_err, l2_err, rate, solve_ms, vcycles, conv_factor);
    }

    prev_max_err = max_err;
  }
}

// Top-level: runs 1D, 2D, and optionally 3D
inline void run_mms_benchmarks(bool do_3d = true) {
  print_header("MMS Accuracy Benchmarks");
  run_mms_1d();
  run_mms_2d();
  if (do_3d) run_mms_3d();
  std::printf("\n");
}

} // namespace bm

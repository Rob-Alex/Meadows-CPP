#pragma once
#include <cstdio>
#include <cmath>
#include "solver.hpp"
#include "bm_utils.hpp"
#include "mms_bm.hpp"  // for n_levels_for

namespace bm {

// -------------------------------------------------------------------------
// Memory profile per (Dim, N): measures allocations in build vs solve phase.
// Uses HostAllocator<double, track_allocations> explicitly to capture events.
// -------------------------------------------------------------------------

inline void run_memory_profile_1d() {
  std::printf("\n=== Memory Profile: 1D ===\n");
  std::printf("   N          DOFs         live_allocs  live_bytes    bytes/DOF  "
              "solve_allocs  solve_bytes\n");
  std::printf("   ---------  -----------  -----------  ------------  ---------  "
              "------------  -----------\n");

  const int sizes[] = {32, 64, 128, 256, 512, 1024};

  for (int N : sizes) {
    double h = 1.0 / N;
    int n_levels = n_levels_for(N);
    long long dofs = static_cast<long long>(N);

    using Alloc = HostAllocator<double, track_allocations>;

    reset_memory_events();
    // ---- build phase ----
    GridGeometry<double, 1> geom({h / 2.0}, {h}, {N}, 0, 1);

    GridHierarchy<double, 1, Alloc> hier;
    int phi_idx = hier.register_component("phi");
    int rhs_idx = hier.register_component("rhs");
    int res_idx = hier.register_component("res");
    hier.build(geom, n_levels);
    // ---- end build ----
    auto snap_post_build = capture_memory();

    // Set RHS
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

    // Use no_tracking for the solver internals (only grid data matters)
    EllipticSolverGMG<double, 1, Alloc> solver(
        hier, bc, phi_idx, rhs_idx, res_idx,
        2, 2, 30, 100, 1e-10, false);

    // ---- solve phase ----
    reset_memory_events();
    solver.solve_impl();
    auto snap_post_solve = capture_memory();
    // ---- end solve ----

    double bytes_per_dof = (dofs > 0)
        ? static_cast<double>(snap_post_build.live_bytes) / static_cast<double>(dofs)
        : 0.0;

    std::printf("  %9d  %11lld  %11zu  %12s  %9.1f  %12zu  %11zu\n",
                N, dofs,
                snap_post_build.live_count,
                human_bytes(snap_post_build.live_bytes).c_str(),
                bytes_per_dof,
                snap_post_solve.event_count,
                snap_post_solve.total_bytes_ever);
  }
}

inline void run_memory_profile_2d() {
  std::printf("\n=== Memory Profile: 2D ===\n");
  std::printf("   N      DOFs         live_allocs  live_bytes    bytes/DOF  "
              "solve_allocs  solve_bytes\n");
  std::printf("   -----  -----------  -----------  ------------  ---------  "
              "------------  -----------\n");

  const int sizes[] = {32, 64, 128, 256, 512};

  for (int N : sizes) {
    double h = 1.0 / N;
    int n_levels = n_levels_for(N);
    long long dofs = static_cast<long long>(N) * N;

    using Alloc = HostAllocator<double, track_allocations>;

    reset_memory_events();
    GridGeometry<double, 2> geom({h / 2.0, h / 2.0}, {h, h}, {N, N}, 0, 1);

    GridHierarchy<double, 2, Alloc> hier;
    int phi_idx = hier.register_component("phi");
    int rhs_idx = hier.register_component("rhs");
    int res_idx = hier.register_component("res");
    hier.build(geom, n_levels);
    auto snap_post_build = capture_memory();

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
        2, 2, 30, 100, 1e-10, false);

    reset_memory_events();
    solver.solve_impl();
    auto snap_post_solve = capture_memory();

    double bytes_per_dof = (dofs > 0)
        ? static_cast<double>(snap_post_build.live_bytes) / static_cast<double>(dofs)
        : 0.0;

    std::printf("  %5d  %11lld  %11zu  %12s  %9.1f  %12zu  %11zu\n",
                N, dofs,
                snap_post_build.live_count,
                human_bytes(snap_post_build.live_bytes).c_str(),
                bytes_per_dof,
                snap_post_solve.event_count,
                snap_post_solve.total_bytes_ever);
  }
}

inline void run_memory_profile_3d() {
  std::printf("\n=== Memory Profile: 3D ===\n");
  std::printf("   N      DOFs         live_allocs  live_bytes    bytes/DOF  "
              "solve_allocs  solve_bytes\n");
  std::printf("   -----  -----------  -----------  ------------  ---------  "
              "------------  -----------\n");

  const int sizes[] = {16, 32, 64, 128};

  for (int N : sizes) {
    double h = 1.0 / N;
    int n_levels = n_levels_for(N);
    long long dofs = static_cast<long long>(N) * N * N;

    using Alloc = HostAllocator<double, track_allocations>;

    reset_memory_events();
    GridGeometry<double, 3> geom(
        {h / 2.0, h / 2.0, h / 2.0}, {h, h, h}, {N, N, N}, 0, 1);

    GridHierarchy<double, 3, Alloc> hier;
    int phi_idx = hier.register_component("phi");
    int rhs_idx = hier.register_component("rhs");
    int res_idx = hier.register_component("res");
    hier.build(geom, n_levels);
    auto snap_post_build = capture_memory();

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
        2, 2, 30, 100, 1e-10, false);

    reset_memory_events();
    solver.solve_impl();
    auto snap_post_solve = capture_memory();

    double bytes_per_dof = (dofs > 0)
        ? static_cast<double>(snap_post_build.live_bytes) / static_cast<double>(dofs)
        : 0.0;

    std::printf("  %5d  %11lld  %11zu  %12s  %9.1f  %12zu  %11zu\n",
                N, dofs,
                snap_post_build.live_count,
                human_bytes(snap_post_build.live_bytes).c_str(),
                bytes_per_dof,
                snap_post_solve.event_count,
                snap_post_solve.total_bytes_ever);
  }
}

// Top-level memory profile benchmark
inline void run_memory_profile_benchmarks() {
  print_header("Memory Profile Benchmarks");
  run_memory_profile_1d();
  run_memory_profile_2d();
  run_memory_profile_3d();
  std::printf("\n");
}

} // namespace bm

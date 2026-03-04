#pragma once
#include "test_harness.hpp"
#include "solver.hpp"
#include <cmath>
#include <vector>

TestResults run_elliptic_tests() {
  TestResults results;
  std::printf("[elliptic]\n");

  const double pi = M_PI;

  // --- V-cycle reduces residual much faster than RBGS alone ---
  {
    const int N = 32;
    const double dx = 1.0 / N;
    GridGeometry<double, 2> finest_geom(
      {dx / 2.0, dx / 2.0}, {dx, dx}, {N, N}, 0, 1);

    GridHierarchy<double, 2> hierarchy;
    int phi_idx = hierarchy.register_component("phi");
    int rhs_idx = hierarchy.register_component("rhs");
    int res_idx = hierarchy.register_component("res");
    hierarchy.build(finest_geom, 4);  // 4 levels: 4x4, 8x8, 16x16, 32x32

    // set RHS on finest level: rhs = -2*pi^2*sin(pi*x)*sin(pi*y)
    auto rhs_acc = hierarchy.finest().accessor(rhs_idx);
    const auto& geom = hierarchy.finest().geometry();
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        double x = (i + 0.5) * dx;
        double y = (j + 0.5) * dx;
        rhs_acc(i, j) = -2.0 * pi * pi * std::sin(pi * x) * std::sin(pi * y);
      }
    }

    // Dirichlet zero BCs
    DirichletBC<double, 2> bc_zero;
    bc_zero.value = 0.0;
    BCRegistry<double, 2> bc_reg;
    for (int dim = 0; dim < 2; ++dim) {
      for (int side = 0; side < 2; ++side) {
        bc_reg.set(dim, side, bc_zero);
      }
    }

    EllipticSolverGMG<double, 2> solver(
      hierarchy, bc_reg, phi_idx, rhs_idx, res_idx,
      2, 2, 20, 50, 1e-12, false);

    auto phi_acc = hierarchy.finest().accessor(phi_idx);
    auto res_acc = hierarchy.finest().accessor(res_idx);
    auto fill_bc = [&](FieldAccessor<double, 2>& acc) {
      bc_reg.fill_all(acc, geom);
    };

    // run 3 V-cycles, measure residual after each
    // the 2nd and 3rd cycles should show good asymptotic convergence
    double res_norms[3];
    for (int c = 0; c < 3; ++c) {
      solver.V_cycle(hierarchy.finest_level());
      fill_bc(phi_acc);
      res_norms[c] = residual<StandardFlux, double, 2>::compute(
        phi_acc, rhs_acc, res_acc, geom);
    }

    // asymptotic convergence factor: ratio between consecutive residuals
    // after 1-2 cycles should be < 0.5 (i.e. >2x reduction per cycle)
    double factor = res_norms[2] / res_norms[1];
    TEST_ASSERT(factor < 0.5, "V-cycle asymptotic convergence factor < 0.5");
  }

  // --- Full solve: FMG + V-cycles converge to tolerance ---
  {
    const int N = 32;
    const double dx = 1.0 / N;
    GridGeometry<double, 2> finest_geom(
      {dx / 2.0, dx / 2.0}, {dx, dx}, {N, N}, 0, 1);

    GridHierarchy<double, 2> hierarchy;
    int phi_idx = hierarchy.register_component("phi");
    int rhs_idx = hierarchy.register_component("rhs");
    int res_idx = hierarchy.register_component("res");
    hierarchy.build(finest_geom, 4);

    auto rhs_acc = hierarchy.finest().accessor(rhs_idx);
    const auto& geom = hierarchy.finest().geometry();
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        double x = (i + 0.5) * dx;
        double y = (j + 0.5) * dx;
        rhs_acc(i, j) = -2.0 * pi * pi * std::sin(pi * x) * std::sin(pi * y);
      }
    }

    DirichletBC<double, 2> bc_zero;
    bc_zero.value = 0.0;
    BCRegistry<double, 2> bc_reg;
    for (int dim = 0; dim < 2; ++dim) {
      for (int side = 0; side < 2; ++side) {
        bc_reg.set(dim, side, bc_zero);
      }
    }

    double tol = 1e-8;
    EllipticSolverGMG<double, 2> solver(
      hierarchy, bc_reg, phi_idx, rhs_idx, res_idx,
      2, 2, 20, 50, tol, false);

    solver.solve();

    // check residual is below tolerance
    auto phi_acc = hierarchy.finest().accessor(phi_idx);
    auto res_acc = hierarchy.finest().accessor(res_idx);
    auto fill_bc = [&](FieldAccessor<double, 2>& acc) {
      bc_reg.fill_all(acc, geom);
    };
    fill_bc(phi_acc);
    double final_res = residual<StandardFlux, double, 2>::compute(
      phi_acc, rhs_acc, res_acc, geom);
    TEST_ASSERT(final_res < tol, "Full solve converges below tolerance");
  }

  // --- Manufactured solution: phi matches analytic to O(h^2) ---
  {
    const int N = 32;
    const double dx = 1.0 / N;
    GridGeometry<double, 2> finest_geom(
      {dx / 2.0, dx / 2.0}, {dx, dx}, {N, N}, 0, 1);

    GridHierarchy<double, 2> hierarchy;
    int phi_idx = hierarchy.register_component("phi");
    int rhs_idx = hierarchy.register_component("rhs");
    int res_idx = hierarchy.register_component("res");
    hierarchy.build(finest_geom, 4);

    auto rhs_acc = hierarchy.finest().accessor(rhs_idx);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        double x = (i + 0.5) * dx;
        double y = (j + 0.5) * dx;
        rhs_acc(i, j) = -2.0 * pi * pi * std::sin(pi * x) * std::sin(pi * y);
      }
    }

    DirichletBC<double, 2> bc_zero;
    bc_zero.value = 0.0;
    BCRegistry<double, 2> bc_reg;
    for (int dim = 0; dim < 2; ++dim) {
      for (int side = 0; side < 2; ++side) {
        bc_reg.set(dim, side, bc_zero);
      }
    }

    EllipticSolverGMG<double, 2> solver(
      hierarchy, bc_reg, phi_idx, rhs_idx, res_idx,
      2, 2, 20, 50, 1e-10, false);

    solver.solve();

    // compare solution to exact: phi = sin(pi*x)*sin(pi*y)
    auto phi_acc = hierarchy.finest().accessor(phi_idx);
    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        double x = (i + 0.5) * dx;
        double y = (j + 0.5) * dx;
        double exact = std::sin(pi * x) * std::sin(pi * y);
        double err = std::fabs(phi_acc(i, j) - exact);
        if (err > max_err) max_err = err;
      }
    }
    // 2nd-order on 32x32: expect max error ~ O(h^2) ~ 0.001
    TEST_ASSERT(max_err < 0.005, "Manufactured solution max error < 0.005");
  }

  // --- Laplace equation with inhomogeneous BCs (parallel plate capacitor) ---
  // Solves ∇²φ = 0 on [0,1]² with φ=0 at x=0, φ=1 at x=1,
  // zero Neumann on y-faces. Exact solution: φ = x.
  // This tests that the V-cycle correctly uses homogeneous BCs on
  // correction levels — without this fix, the solver diverges.
  {
    const int N = 32;
    const double dx = 1.0 / N;
    GridGeometry<double, 2> finest_geom(
      {dx / 2.0, dx / 2.0}, {dx, dx}, {N, N}, 0, 1);

    GridHierarchy<double, 2> hierarchy;
    int phi_idx = hierarchy.register_component("phi");
    int rhs_idx = hierarchy.register_component("rhs");
    int res_idx = hierarchy.register_component("res");
    hierarchy.build(finest_geom, 4);

    // RHS = 0 (Laplace equation)
    // (already zero from allocation)

    // BCs: Dirichlet 0 at x=0, Dirichlet 1 at x=1, Neumann 0 on y-faces
    DirichletBC<double, 2> bc_lo;
    bc_lo.value = 0.0;
    DirichletBC<double, 2> bc_hi;
    bc_hi.value = 1.0;
    NeumannBC<double, 2> bc_neumann;
    bc_neumann.flux = 0.0;

    BCRegistry<double, 2> bc_reg;
    bc_reg.set(0, 0, bc_lo);      // x-low: φ = 0
    bc_reg.set(0, 1, bc_hi);      // x-high: φ = 1
    bc_reg.set(1, 0, bc_neumann); // y-low: ∂φ/∂n = 0
    bc_reg.set(1, 1, bc_neumann); // y-high: ∂φ/∂n = 0

    double tol = 1e-8;
    EllipticSolverGMG<double, 2> solver(
      hierarchy, bc_reg, phi_idx, rhs_idx, res_idx,
      2, 2, 20, 50, tol, false);

    solver.solve();

    // check residual converged
    auto phi_acc = hierarchy.finest().accessor(phi_idx);
    auto rhs_acc = hierarchy.finest().accessor(rhs_idx);
    auto res_acc = hierarchy.finest().accessor(res_idx);
    const auto& geom = hierarchy.finest().geometry();
    auto fill_bc = [&](FieldAccessor<double, 2>& acc) {
      bc_reg.fill_all(acc, geom);
    };
    fill_bc(phi_acc);
    double final_res = residual<StandardFlux, double, 2>::compute(
      phi_acc, rhs_acc, res_acc, geom);
    TEST_ASSERT(final_res < tol, "Parallel plate: residual converges below tolerance");

    // check solution matches φ = x
    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        double x = (i + 0.5) * dx;
        double exact = x;  // linear solution φ = x
        double err = std::fabs(phi_acc(i, j) - exact);
        if (err > max_err) max_err = err;
      }
    }
    TEST_ASSERT(max_err < 0.005, "Parallel plate: solution matches φ = x to O(h^2)");
  }

  // --- Residual monotonically decreases (inhomogeneous BCs, parallel plate) ---
  // This is the regression test for the homogeneous BC fix.
  // Before the fix, the residual would GROW on coarser correction levels
  // because original (non-zero) BCs were applied to the correction variable.
  // Track residual at each V-cycle and assert strict monotonic decrease.
  {
    const int N = 32;
    const double dx = 1.0 / N;
    GridGeometry<double, 2> finest_geom(
      {dx / 2.0, dx / 2.0}, {dx, dx}, {N, N}, 0, 1);

    GridHierarchy<double, 2> hierarchy;
    int phi_idx = hierarchy.register_component("phi");
    int rhs_idx = hierarchy.register_component("rhs");
    int res_idx = hierarchy.register_component("res");
    hierarchy.build(finest_geom, 4);

    DirichletBC<double, 2> bc_lo;
    bc_lo.value = 0.0;
    DirichletBC<double, 2> bc_hi;
    bc_hi.value = 1.0;
    NeumannBC<double, 2> bc_neumann;
    bc_neumann.flux = 0.0;

    BCRegistry<double, 2> bc_reg;
    bc_reg.set(0, 0, bc_lo);
    bc_reg.set(0, 1, bc_hi);
    bc_reg.set(1, 0, bc_neumann);
    bc_reg.set(1, 1, bc_neumann);

    std::vector<double> res_history;
    EllipticSolverGMG<double, 2> solver(
      hierarchy, bc_reg, phi_idx, rhs_idx, res_idx,
      2, 2, 20, 15, 1e-12, false);

    solver.solve_impl([&](int /*step*/, double res_norm) {
      res_history.push_back(res_norm);
    });

    // every residual should be <= previous (monotonic decrease)
    bool monotonic = true;
    for (size_t i = 1; i < res_history.size(); ++i) {
      if (res_history[i] > res_history[i - 1] * 1.01) {  // 1% tolerance for noise
        monotonic = false;
        break;
      }
    }
    TEST_ASSERT(monotonic, "Inhomogeneous BCs: residual monotonically decreases");

    // should have reduced by at least 1e4 overall
    double reduction = res_history.front() / res_history.back();
    TEST_ASSERT(reduction > 1e4, "Inhomogeneous BCs: >1e4 total residual reduction");
  }

  // --- Grid-independent convergence factor (N-scaling) ---
  // A correct multigrid should have a convergence factor that does NOT
  // degrade as N grows. We run at N=16, 32, 64 and check the asymptotic
  // convergence factor (ratio of consecutive residuals) is similar.
  // This tests both Poisson (rhs != 0) and Laplace (inhomogeneous BCs).
  {
    // Test 1: Poisson with zero Dirichlet
    double factors_poisson[3];
    int grids[] = {16, 32, 64};
    int levels[] = {3, 4, 5};

    for (int run = 0; run < 3; ++run) {
      const int N = grids[run];
      const double dx = 1.0 / N;
      GridGeometry<double, 2> finest_geom(
        {dx / 2.0, dx / 2.0}, {dx, dx}, {N, N}, 0, 1);

      GridHierarchy<double, 2> hierarchy;
      int phi_idx = hierarchy.register_component("phi");
      int rhs_idx = hierarchy.register_component("rhs");
      int res_idx = hierarchy.register_component("res");
      hierarchy.build(finest_geom, levels[run]);

      auto rhs_acc = hierarchy.finest().accessor(rhs_idx);
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          double x = (i + 0.5) * dx;
          double y = (j + 0.5) * dx;
          rhs_acc(i, j) = -2.0 * pi * pi * std::sin(pi * x) * std::sin(pi * y);
        }
      }

      DirichletBC<double, 2> bc_zero;
      bc_zero.value = 0.0;
      BCRegistry<double, 2> bc_reg;
      for (int dim = 0; dim < 2; ++dim) {
        for (int side = 0; side < 2; ++side) {
          bc_reg.set(dim, side, bc_zero);
        }
      }

      // collect residuals per V-cycle
      std::vector<double> res_hist;
      EllipticSolverGMG<double, 2> solver(
        hierarchy, bc_reg, phi_idx, rhs_idx, res_idx,
        2, 2, 20, 6, 1e-14, false);

      solver.solve_impl([&](int /*step*/, double res_norm) {
        res_hist.push_back(res_norm);
      });

      // asymptotic factor from last two V-cycles
      size_t n = res_hist.size();
      factors_poisson[run] = res_hist[n - 1] / res_hist[n - 2];
    }

    // all convergence factors should be < 0.5 (good multigrid)
    for (int run = 0; run < 3; ++run) {
      char name[80];
      std::snprintf(name, sizeof(name),
        "Poisson N=%d: convergence factor < 0.5 (got %.3f)",
        grids[run], factors_poisson[run]);
      TEST_ASSERT(factors_poisson[run] < 0.5, name);
    }

    // factors should be similar across grids (grid-independent)
    // max/min ratio should be < 2 (they should all be ~0.1)
    double max_f = factors_poisson[0], min_f = factors_poisson[0];
    for (int i = 1; i < 3; ++i) {
      if (factors_poisson[i] > max_f) max_f = factors_poisson[i];
      if (factors_poisson[i] < min_f) min_f = factors_poisson[i];
    }
    TEST_ASSERT(max_f / min_f < 2.0, "Poisson: grid-independent convergence factor");

    // Test 2: Laplace with inhomogeneous Dirichlet (parallel plate)
    double factors_laplace[3];

    for (int run = 0; run < 3; ++run) {
      const int N = grids[run];
      const double dx = 1.0 / N;
      GridGeometry<double, 2> finest_geom(
        {dx / 2.0, dx / 2.0}, {dx, dx}, {N, N}, 0, 1);

      GridHierarchy<double, 2> hierarchy;
      int phi_idx = hierarchy.register_component("phi");
      int rhs_idx = hierarchy.register_component("rhs");
      int res_idx = hierarchy.register_component("res");
      hierarchy.build(finest_geom, levels[run]);

      // RHS = 0 (Laplace)

      DirichletBC<double, 2> bc_lo;
      bc_lo.value = 0.0;
      DirichletBC<double, 2> bc_hi;
      bc_hi.value = 1.0;
      NeumannBC<double, 2> bc_neumann;
      bc_neumann.flux = 0.0;

      BCRegistry<double, 2> bc_reg;
      bc_reg.set(0, 0, bc_lo);
      bc_reg.set(0, 1, bc_hi);
      bc_reg.set(1, 0, bc_neumann);
      bc_reg.set(1, 1, bc_neumann);

      std::vector<double> res_hist;
      EllipticSolverGMG<double, 2> solver(
        hierarchy, bc_reg, phi_idx, rhs_idx, res_idx,
        2, 2, 20, 6, 1e-14, false);

      solver.solve_impl([&](int /*step*/, double res_norm) {
        res_hist.push_back(res_norm);
      });

      size_t n = res_hist.size();
      factors_laplace[run] = res_hist[n - 1] / res_hist[n - 2];
    }

    for (int run = 0; run < 3; ++run) {
      char name[80];
      std::snprintf(name, sizeof(name),
        "Laplace N=%d: convergence factor < 0.5 (got %.3f)",
        grids[run], factors_laplace[run]);
      TEST_ASSERT(factors_laplace[run] < 0.5, name);
    }

    max_f = factors_laplace[0]; min_f = factors_laplace[0];
    for (int i = 1; i < 3; ++i) {
      if (factors_laplace[i] > max_f) max_f = factors_laplace[i];
      if (factors_laplace[i] < min_f) min_f = factors_laplace[i];
    }
    TEST_ASSERT(max_f / min_f < 2.0, "Laplace: grid-independent convergence factor");
  }

  // --- Convergence rate: error drops ~4x when grid doubles (2nd order) ---
  {
    double errors[2];
    for (int run = 0; run < 2; ++run) {
      const int N = (run == 0) ? 16 : 32;
      const double dx = 1.0 / N;
      // need enough levels so coarsest is at least 2x2
      int n_levels = (run == 0) ? 3 : 4;

      GridGeometry<double, 2> finest_geom(
        {dx / 2.0, dx / 2.0}, {dx, dx}, {N, N}, 0, 1);

      GridHierarchy<double, 2> hierarchy;
      int phi_idx = hierarchy.register_component("phi");
      int rhs_idx = hierarchy.register_component("rhs");
      int res_idx = hierarchy.register_component("res");
      hierarchy.build(finest_geom, n_levels);

      auto rhs_acc = hierarchy.finest().accessor(rhs_idx);
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          double x = (i + 0.5) * dx;
          double y = (j + 0.5) * dx;
          rhs_acc(i, j) = -2.0 * pi * pi * std::sin(pi * x) * std::sin(pi * y);
        }
      }

      DirichletBC<double, 2> bc_zero;
      bc_zero.value = 0.0;
      BCRegistry<double, 2> bc_reg;
      for (int dim = 0; dim < 2; ++dim) {
        for (int side = 0; side < 2; ++side) {
          bc_reg.set(dim, side, bc_zero);
        }
      }

      EllipticSolverGMG<double, 2> solver(
        hierarchy, bc_reg, phi_idx, rhs_idx, res_idx,
        2, 2, 20, 50, 1e-10, false);

      solver.solve();

      auto phi_acc = hierarchy.finest().accessor(phi_idx);
      double max_err = 0.0;
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          double x = (i + 0.5) * dx;
          double y = (j + 0.5) * dx;
          double exact = std::sin(pi * x) * std::sin(pi * y);
          double err = std::fabs(phi_acc(i, j) - exact);
          if (err > max_err) max_err = err;
        }
      }
      errors[run] = max_err;
    }

    double rate = errors[0] / errors[1];
    // 2nd order → ratio should be ~4
    TEST_ASSERT(rate > 3.5, "Solver 2nd-order convergence (error ratio > 3.5)");
  }

  return results;
}

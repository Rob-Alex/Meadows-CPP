#pragma once
#include "test_harness.hpp"
#include "operators.hpp"
#include "interlevel.hpp"
#include "boundary_conditions.hpp"
#include <cmath>

TestResults run_operators_tests() {
  TestResults results;
  std::printf("[operators]\n");

  const double pi = M_PI;

  // --- 1D Laplacian: phi(x) = sin(pi*x) on [0,1], L(phi) = -pi^2 sin(pi*x)
  {
    const int N = 64;
    const double dx = 1.0 / N;
    GridGeometry<double, 1> geom({dx / 2.0}, {dx}, {N}, 0, 2);

    Box<double, 1> box(geom, 2);  // phi=0, Lphi=1
    auto phi  = box.accessor(0);
    auto Lphi = box.accessor(1);

    for (int i = 0; i < N; ++i) {
      double x = (i + 0.5) * dx;
      std::array<int, 1> idx = {i};
      phi[idx] = std::sin(pi * x);
    }

    // Dirichlet BCs: sin(pi*0)=0, sin(pi*1)=0
    DirichletBC<double, 1> bc_zero;
    bc_zero.value = 0.0;
    BCRegistry<double, 1> bc_reg;
    bc_reg.set(0, 0, bc_zero);
    bc_reg.set(0, 1, bc_zero);
    bc_reg.fill_all(phi, geom);

    laplacian<StandardFlux, double, 1>::apply(phi, Lphi, geom);

    // Check all interior cells against analytic -pi^2 sin(pi*x)
    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
      double x = (i + 0.5) * dx;
      double exact = -pi * pi * std::sin(pi * x);
      std::array<int, 1> idx = {i};
      double err = std::fabs(Lphi[idx] - exact);
      if (err > max_err) max_err = err;
    }
    // 2nd-order stencil on N=64 → error ~ (pi*dx)^2 / 12 ≈ 0.02
    TEST_ASSERT(max_err < 0.05, "1D Laplacian max error < 0.05");
  }

  // --- 2D Laplacian: phi(x,y) = sin(pi*x)*sin(pi*y), L(phi) = -2*pi^2*phi
  {
    const int N = 32;
    const double dx = 1.0 / N;
    GridGeometry<double, 2> geom({dx / 2.0, dx / 2.0}, {dx, dx}, {N, N}, 0, 2);

    Box<double, 2> box(geom, 2);
    auto phi  = box.accessor(0);
    auto Lphi = box.accessor(1);

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        double x = (i + 0.5) * dx;
        double y = (j + 0.5) * dx;
        phi(i, j) = std::sin(pi * x) * std::sin(pi * y);
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
    bc_reg.fill_all(phi, geom);

    laplacian<StandardFlux, double, 2>::apply(phi, Lphi, geom);

    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        double x = (i + 0.5) * dx;
        double y = (j + 0.5) * dx;
        double exact = -2.0 * pi * pi * std::sin(pi * x) * std::sin(pi * y);
        double err = std::fabs(Lphi(i, j) - exact);
        if (err > max_err) max_err = err;
      }
    }
    TEST_ASSERT(max_err < 0.1, "2D Laplacian max error < 0.1");
  }

  // --- 2D convergence rate: error should drop by ~4x when dx halves (2nd order)
  {
    double errors[2];
    for (int run = 0; run < 2; ++run) {
      const int N = (run == 0) ? 32 : 64;
      const double dx = 1.0 / N;
      GridGeometry<double, 2> geom({dx / 2.0, dx / 2.0}, {dx, dx}, {N, N}, 0, 2);

      Box<double, 2> box(geom, 2);
      auto phi  = box.accessor(0);
      auto Lphi = box.accessor(1);

      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          double x = (i + 0.5) * dx;
          double y = (j + 0.5) * dx;
          phi(i, j) = std::sin(pi * x) * std::sin(pi * y);
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
      bc_reg.fill_all(phi, geom);

      laplacian<StandardFlux, double, 2>::apply(phi, Lphi, geom);

      double max_err = 0.0;
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          double x = (i + 0.5) * dx;
          double y = (j + 0.5) * dx;
          double exact = -2.0 * pi * pi * std::sin(pi * x) * std::sin(pi * y);
          double err = std::fabs(Lphi(i, j) - exact);
          if (err > max_err) max_err = err;
        }
      }
      errors[run] = max_err;
    }
    double rate = errors[0] / errors[1];
    // 2nd order → ratio should be ~4. Accept 3.5+ to allow for discretisation effects.
    TEST_ASSERT(rate > 3.5, "2D Laplacian 2nd-order convergence (ratio > 3.5)");
  }

  // --- l2_norm: known field, check against analytic
  {
    const int N = 4;
    const double dx = 1.0 / N;
    GridGeometry<double, 2> geom({dx / 2.0, dx / 2.0}, {dx, dx}, {N, N}, 0, 1);

    Box<double, 2> box(geom, 1);
    auto phi = box.accessor(0);

    // Set all interior cells to 3.0 → L2 = sqrt(sum(9) / 16) = sqrt(9) = 3.0
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        phi(i, j) = 3.0;
      }
    }

    double norm = l2_norm<double, 2>::compute(phi, geom);
    TEST_ASSERT_NEAR(norm, 3.0, 1e-12, "l2_norm of constant field == 3.0");
  }

  // --- l2_norm: zero field
  {
    const int N = 8;
    const double dx = 1.0 / N;
    GridGeometry<double, 2> geom({dx / 2.0, dx / 2.0}, {dx, dx}, {N, N}, 0, 1);

    Box<double, 2> box(geom, 1);
    auto phi = box.accessor(0);
    // already zero-initialised

    double norm = l2_norm<double, 2>::compute(phi, geom);
    TEST_ASSERT_NEAR(norm, 0.0, 1e-15, "l2_norm of zero field == 0.0");
  }

  // --- residual: if phi is exact solution, residual should be small
  {
    const int N = 32;
    const double dx = 1.0 / N;
    GridGeometry<double, 2> geom({dx / 2.0, dx / 2.0}, {dx, dx}, {N, N}, 0, 2);

    Box<double, 2> box(geom, 3);  // phi=0, rhs=1, res=2
    auto phi = box.accessor(0);
    auto rhs = box.accessor(1);
    auto res = box.accessor(2);

    // Set phi = sin(pi*x)*sin(pi*y), rhs = -2*pi^2*sin(pi*x)*sin(pi*y)
    // Then res = rhs - L(phi) should be ~ 0 (up to discretisation error)
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        double x = (i + 0.5) * dx;
        double y = (j + 0.5) * dx;
        phi(i, j) = std::sin(pi * x) * std::sin(pi * y);
        rhs(i, j) = -2.0 * pi * pi * std::sin(pi * x) * std::sin(pi * y);
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
    bc_reg.fill_all(phi, geom);

    double res_norm = residual<StandardFlux, double, 2>::compute(phi, rhs, res, geom);
    // Residual should be small (discretisation error only, ~O(h^2))
    TEST_ASSERT(res_norm < 0.1, "residual norm small for exact manufactured solution");
  }

  // --- RBGS: smoothing reduces the residual
  {
    const int N = 16;
    const double dx = 1.0 / N;
    GridGeometry<double, 2> geom({dx / 2.0, dx / 2.0}, {dx, dx}, {N, N}, 0, 1);

    Box<double, 2> box(geom, 3);  // phi=0, rhs=1, res=2
    auto phi = box.accessor(0);
    auto rhs = box.accessor(1);
    auto res = box.accessor(2);

    // rhs = -2*pi^2*sin(pi*x)*sin(pi*y)  (Poisson source for known solution)
    // phi starts at 0 — the smoother should drive it toward the solution
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        double x = (i + 0.5) * dx;
        double y = (j + 0.5) * dx;
        rhs(i, j) = -2.0 * pi * pi * std::sin(pi * x) * std::sin(pi * y);
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

    auto fill_bc = [&](FieldAccessor<double, 2>& acc) {
      bc_reg.fill_all(acc, geom);
    };

    // measure residual before smoothing
    fill_bc(phi);
    double res_before = residual<StandardFlux, double, 2>::compute(phi, rhs, res, geom);

    // run 50 RBGS sweeps — enough to see meaningful reduction even on
    // low-frequency modes. RBGS alone converges slowly on smooth error
    // (that's why we need multigrid), but 50 sweeps should still help.
    RBGS<double, 2> smoother;
    smoother.smooth(phi, rhs, geom, 50, fill_bc);

    // measure residual after smoothing
    double res_after = residual<StandardFlux, double, 2>::compute(phi, rhs, res, geom);

    TEST_ASSERT(res_after < res_before, "RBGS reduces residual after 50 sweeps");
    TEST_ASSERT(res_after < 0.5 * res_before, "RBGS halves residual in 50 sweeps");
  }

  // --- RBGS: many sweeps converge toward the exact solution
  {
    const int N = 16;
    const double dx = 1.0 / N;
    GridGeometry<double, 2> geom({dx / 2.0, dx / 2.0}, {dx, dx}, {N, N}, 0, 1);

    Box<double, 2> box(geom, 2);  // phi=0, rhs=1
    auto phi = box.accessor(0);
    auto rhs = box.accessor(1);

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        double x = (i + 0.5) * dx;
        double y = (j + 0.5) * dx;
        rhs(i, j) = -2.0 * pi * pi * std::sin(pi * x) * std::sin(pi * y);
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

    auto fill_bc = [&](FieldAccessor<double, 2>& acc) {
      bc_reg.fill_all(acc, geom);
    };

    // run 500 sweeps — on a 16x16 grid, RBGS alone should converge
    RBGS<double, 2> smoother;
    fill_bc(phi);
    smoother.smooth(phi, rhs, geom, 500, fill_bc);

    // compare against exact solution phi = sin(pi*x)*sin(pi*y)
    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        double x = (i + 0.5) * dx;
        double y = (j + 0.5) * dx;
        double exact = std::sin(pi * x) * std::sin(pi * y);
        double err = std::fabs(phi(i, j) - exact);
        if (err > max_err) max_err = err;
      }
    }
    // should be close to discretisation error level
    TEST_ASSERT(max_err < 0.01, "RBGS 500 sweeps converges to solution (max err < 0.01)");
  }

  // ================================================================
  // Inter-level transfer operators (Restriction & Prolongation)
  // ================================================================

  // --- Restriction: constant field should restrict exactly
  {
    const int N_fine = 8;
    const int ratio = 2;
    const int N_coarse = N_fine / ratio;
    const double dx_fine = 1.0 / N_fine;
    const double dx_coarse = 1.0 / N_coarse;

    GridGeometry<double, 2> fine_geom(
      {dx_fine / 2.0, dx_fine / 2.0}, {dx_fine, dx_fine}, {N_fine, N_fine}, 1, 1);
    GridGeometry<double, 2> coarse_geom(
      {dx_coarse / 2.0, dx_coarse / 2.0}, {dx_coarse, dx_coarse}, {N_coarse, N_coarse}, 0, 1);

    Box<double, 2> fine_box(fine_geom, 1);
    Box<double, 2> coarse_box(coarse_geom, 1);
    auto fine_acc = fine_box.accessor(0);
    auto coarse_acc = coarse_box.accessor(0);

    // set fine field to constant 5.0
    for (int i = 0; i < N_fine; ++i) {
      for (int j = 0; j < N_fine; ++j) {
        fine_acc(i, j) = 5.0;
      }
    }

    Restriction<double, 2>::apply(fine_acc, coarse_acc, coarse_geom, ratio);

    // every coarse cell should be exactly 5.0
    double max_err = 0.0;
    for (int i = 0; i < N_coarse; ++i) {
      for (int j = 0; j < N_coarse; ++j) {
        double err = std::fabs(coarse_acc(i, j) - 5.0);
        if (err > max_err) max_err = err;
      }
    }
    TEST_ASSERT(max_err < 1e-14, "Restriction preserves constant field");
  }

  // --- Restriction: smooth field is averaged correctly
  {
    const int N_fine = 8;
    const int ratio = 2;
    const int N_coarse = N_fine / ratio;
    const double dx_fine = 1.0 / N_fine;
    const double dx_coarse = 1.0 / N_coarse;

    GridGeometry<double, 2> fine_geom(
      {dx_fine / 2.0, dx_fine / 2.0}, {dx_fine, dx_fine}, {N_fine, N_fine}, 1, 1);
    GridGeometry<double, 2> coarse_geom(
      {dx_coarse / 2.0, dx_coarse / 2.0}, {dx_coarse, dx_coarse}, {N_coarse, N_coarse}, 0, 1);

    Box<double, 2> fine_box(fine_geom, 1);
    Box<double, 2> coarse_box(coarse_geom, 1);
    auto fine_acc = fine_box.accessor(0);
    auto coarse_acc = coarse_box.accessor(0);

    // set fine field to linear: phi = x + y
    for (int i = 0; i < N_fine; ++i) {
      for (int j = 0; j < N_fine; ++j) {
        double x = (i + 0.5) * dx_fine;
        double y = (j + 0.5) * dx_fine;
        fine_acc(i, j) = x + y;
      }
    }

    Restriction<double, 2>::apply(fine_acc, coarse_acc, coarse_geom, ratio);

    // for a linear field, the average of the 4 children should equal
    // the value at the coarse cell centre
    double max_err = 0.0;
    for (int i = 0; i < N_coarse; ++i) {
      for (int j = 0; j < N_coarse; ++j) {
        double x_c = (i + 0.5) * dx_coarse;
        double y_c = (j + 0.5) * dx_coarse;
        double expected = x_c + y_c;
        double err = std::fabs(coarse_acc(i, j) - expected);
        if (err > max_err) max_err = err;
      }
    }
    TEST_ASSERT(max_err < 1e-14, "Restriction of linear field == coarse cell-centre value");
  }

  // --- Prolongation: constant correction added correctly
  {
    const int N_fine = 8;
    const int ratio = 2;
    const int N_coarse = N_fine / ratio;
    const double dx_fine = 1.0 / N_fine;
    const double dx_coarse = 1.0 / N_coarse;

    GridGeometry<double, 2> fine_geom(
      {dx_fine / 2.0, dx_fine / 2.0}, {dx_fine, dx_fine}, {N_fine, N_fine}, 1, 1);
    GridGeometry<double, 2> coarse_geom(
      {dx_coarse / 2.0, dx_coarse / 2.0}, {dx_coarse, dx_coarse}, {N_coarse, N_coarse}, 0, 1);

    Box<double, 2> fine_box(fine_geom, 1);
    Box<double, 2> coarse_box(coarse_geom, 1);
    auto fine_acc = fine_box.accessor(0);
    auto coarse_acc = coarse_box.accessor(0);

    // fine starts at 1.0, coarse correction is 3.0
    for (int i = 0; i < N_fine; ++i) {
      for (int j = 0; j < N_fine; ++j) {
        fine_acc(i, j) = 1.0;
      }
    }
    for (int i = 0; i < N_coarse; ++i) {
      for (int j = 0; j < N_coarse; ++j) {
        coarse_acc(i, j) = 3.0;
      }
    }

    Prolongation<double, 2>::apply(coarse_acc, fine_acc, fine_geom, ratio);

    // fine should now be 1.0 + 3.0 = 4.0 everywhere
    double max_err = 0.0;
    for (int i = 0; i < N_fine; ++i) {
      for (int j = 0; j < N_fine; ++j) {
        double err = std::fabs(fine_acc(i, j) - 4.0);
        if (err > max_err) max_err = err;
      }
    }
    TEST_ASSERT(max_err < 1e-14, "Prolongation += constant correction");
  }

  // --- Prolongation uses += not =
  {
    const int N_fine = 4;
    const int ratio = 2;
    const int N_coarse = N_fine / ratio;
    const double dx_fine = 1.0 / N_fine;
    const double dx_coarse = 1.0 / N_coarse;

    GridGeometry<double, 2> fine_geom(
      {dx_fine / 2.0, dx_fine / 2.0}, {dx_fine, dx_fine}, {N_fine, N_fine}, 1, 1);
    GridGeometry<double, 2> coarse_geom(
      {dx_coarse / 2.0, dx_coarse / 2.0}, {dx_coarse, dx_coarse}, {N_coarse, N_coarse}, 0, 1);

    Box<double, 2> fine_box(fine_geom, 1);
    Box<double, 2> coarse_box(coarse_geom, 1);
    auto fine_acc = fine_box.accessor(0);
    auto coarse_acc = coarse_box.accessor(0);

    // fine = 10, coarse = 7 → after prolongation fine should be 17, not 7
    for (int i = 0; i < N_fine; ++i) {
      for (int j = 0; j < N_fine; ++j) {
        fine_acc(i, j) = 10.0;
      }
    }
    for (int i = 0; i < N_coarse; ++i) {
      for (int j = 0; j < N_coarse; ++j) {
        coarse_acc(i, j) = 7.0;
      }
    }

    Prolongation<double, 2>::apply(coarse_acc, fine_acc, fine_geom, ratio);

    TEST_ASSERT_NEAR(fine_acc(0, 0), 17.0, 1e-14, "Prolongation += semantics (10 + 7 = 17)");
  }

  // --- Restrict then prolongate a constant: round-trip preserves value
  {
    const int N_fine = 8;
    const int ratio = 2;
    const int N_coarse = N_fine / ratio;
    const double dx_fine = 1.0 / N_fine;
    const double dx_coarse = 1.0 / N_coarse;

    GridGeometry<double, 2> fine_geom(
      {dx_fine / 2.0, dx_fine / 2.0}, {dx_fine, dx_fine}, {N_fine, N_fine}, 1, 1);
    GridGeometry<double, 2> coarse_geom(
      {dx_coarse / 2.0, dx_coarse / 2.0}, {dx_coarse, dx_coarse}, {N_coarse, N_coarse}, 0, 1);

    Box<double, 2> fine_box(fine_geom, 1);
    Box<double, 2> coarse_box(coarse_geom, 1);
    Box<double, 2> result_box(fine_geom, 1);
    auto fine_acc = fine_box.accessor(0);
    auto coarse_acc = coarse_box.accessor(0);
    auto result_acc = result_box.accessor(0);

    // set fine to constant 4.0
    for (int i = 0; i < N_fine; ++i) {
      for (int j = 0; j < N_fine; ++j) {
        fine_acc(i, j) = 4.0;
      }
    }

    // restrict fine → coarse, then prolongate coarse → result (which starts at 0)
    Restriction<double, 2>::apply(fine_acc, coarse_acc, coarse_geom, ratio);
    Prolongation<double, 2>::apply(coarse_acc, result_acc, fine_geom, ratio);

    // result should be 4.0 everywhere
    double max_err = 0.0;
    for (int i = 0; i < N_fine; ++i) {
      for (int j = 0; j < N_fine; ++j) {
        double err = std::fabs(result_acc(i, j) - 4.0);
        if (err > max_err) max_err = err;
      }
    }
    TEST_ASSERT(max_err < 1e-14, "Restrict→Prolongate round-trip preserves constant");
  }

  // --- Restriction with GridHierarchy: test across levels
  {
    GridGeometry<double, 2> finest_geom({0.05, 0.05}, {0.1, 0.1}, {8, 8}, 0, 1);

    GridHierarchy<double, 2> hierarchy;
    int phi_idx = hierarchy.register_component("phi");
    hierarchy.build(finest_geom, 3);  // 3 levels: 2x2, 4x4, 8x8

    // set finest level to constant 6.0
    auto finest_acc = hierarchy.finest().accessor(phi_idx);
    const auto& finest_g = hierarchy.finest().geometry();
    for (int i = 0; i < finest_g._n_interior[0]; ++i) {
      for (int j = 0; j < finest_g._n_interior[1]; ++j) {
        finest_acc(i, j) = 6.0;
      }
    }

    // restrict finest → level 1
    auto lvl1_acc = hierarchy.level(1).accessor(phi_idx);
    const auto& lvl1_g = hierarchy.level(1).geometry();
    Restriction<double, 2>::apply(finest_acc, lvl1_acc, lvl1_g, hierarchy.ref_ratio());

    // restrict level 1 → level 0
    auto lvl0_acc = hierarchy.level(0).accessor(phi_idx);
    const auto& lvl0_g = hierarchy.level(0).geometry();
    Restriction<double, 2>::apply(lvl1_acc, lvl0_acc, lvl0_g, hierarchy.ref_ratio());

    // coarsest should be 6.0
    TEST_ASSERT_NEAR(lvl0_acc(0, 0), 6.0, 1e-14, "Hierarchy restriction preserves constant");
    TEST_ASSERT_NEAR(lvl0_acc(1, 1), 6.0, 1e-14, "Hierarchy restriction preserves constant (1,1)");
  }

  return results;
}

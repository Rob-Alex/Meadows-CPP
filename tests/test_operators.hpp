#pragma once
#include "test_harness.hpp"
#include "operators.hpp"
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

  return results;
}

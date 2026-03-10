/*
  meadows-cpp: 2D Poisson solve with Gaussian charge + E-field export.

  Solves ∇²φ = -ρ/ε₀ on [0,1]² with:
    - Gaussian charge blob at centre: ρ(r) = Q·exp(-r²/2σ²)
    - Dirichlet φ = 0 on all boundaries (grounded box)

  Exports: phi, Ex, Ey, E_mag to HDF5/XDMF for ParaView/VisIt.
*/
#include <cstdio>
#include <cmath>
#include "solver.hpp"
#include "exporter.hpp"

int main() {
  const int N = 128;
  const double dx = 1.0 / N;
  const int n_levels = static_cast<int>(std::log2(N)) - 1;

  // charge parameters
  const double sigma = 0.05;            // blob width [m]
  const double Q     = 1.0;             // charge amplitude (arbitrary units)
  const double cx    = 0.5, cy = 0.5;   // blob centre

  std::printf("meadows-cpp: 2D Gaussian charge — Poisson + E-field\n");
  std::printf("  N = %d, sigma = %.3f, %d multigrid levels\n\n", N, sigma, n_levels);

  // --- grid hierarchy ---
  GridGeometry<double, 2> finest_geom(
    {dx / 2.0, dx / 2.0}, {dx, dx}, {N, N}, 0, 1);

  GridHierarchy<double, 2> hierarchy;
  int phi_idx  = hierarchy.register_component("phi");
  int rhs_idx  = hierarchy.register_component("rhs");
  int res_idx  = hierarchy.register_component("res");
  int ex_idx   = hierarchy.register_component("Ex");
  int ey_idx   = hierarchy.register_component("Ey");
  int emag_idx = hierarchy.register_component("E_mag");
  hierarchy.build(finest_geom, n_levels);

  // --- RHS: Gaussian charge blob ---
  auto rhs_acc = hierarchy.finest().accessor(rhs_idx);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      double x = (i + 0.5) * dx;
      double y = (j + 0.5) * dx;
      double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      rhs_acc(i, j) = -Q * std::exp(-r2 / (2.0 * sigma * sigma));
    }
  }

  // --- BCs: grounded box (Dirichlet zero on all faces) ---
  DirichletBC<double, 2> bc_zero;
  bc_zero.value = 0.0;
  BCRegistry<double, 2> bc_reg;
  for (int dim = 0; dim < 2; ++dim) {
    for (int side = 0; side < 2; ++side) {
      bc_reg.set(dim, side, bc_zero);
    }
  }

  // --- solve Poisson ---
  EllipticSolverGMG<double, 2> solver(
    hierarchy, bc_reg, phi_idx, rhs_idx, res_idx,
    2, 2, 20, 50, 1e-10, true);
  solver.solve();

  // --- compute E = -∇φ ---
  auto phi_acc  = hierarchy.finest().accessor(phi_idx);
  auto ex_acc   = hierarchy.finest().accessor(ex_idx);
  auto ey_acc   = hierarchy.finest().accessor(ey_idx);
  auto emag_acc = hierarchy.finest().accessor(emag_idx);

  bc_reg.fill_all(phi_acc, finest_geom);

  std::array<FieldAccessor<double, 2>, 2> E_comps = {ex_acc, ey_acc};
  gradient<double, 2>::compute(phi_acc, E_comps, emag_acc, finest_geom);

  // --- print field ranges ---
  double phi_min = 1e30, phi_max = -1e30;
  double emag_min = 1e30, emag_max = -1e30;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      phi_min  = std::min(phi_min,  phi_acc(i, j));
      phi_max  = std::max(phi_max,  phi_acc(i, j));
      emag_min = std::min(emag_min, emag_acc(i, j));
      emag_max = std::max(emag_max, emag_acc(i, j));
    }
  }
  std::printf("\nField ranges:\n");
  std::printf("  Potential: \t [%.6e, %.6e]\n", phi_min, phi_max);
  std::printf("  E Magnitude: \t [%.6e, %.6e]\n", emag_min, emag_max);

  // --- export ---
  GridExporter<double, 2> exporter("build/outputs", "charge");
  exporter.write(hierarchy, {"phi", "rhs", "Ex", "Ey", "E_mag"}, 0);

  std::printf("\nDone. Open build/outputs/charge_000000.xmf in ParaView.\n");

  return 0;
}

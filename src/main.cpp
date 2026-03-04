/*
  meadows-cpp: Parallel plate capacitor demo with per-iteration HDF5/XDMF export.

  Solves ∇²φ = 0 on [0,1]² with:
    φ = 0 at x = 0  (grounded plate)
    φ = 1 at x = 1  (charged plate)
    ∂φ/∂n = 0 on y-faces (insulating walls)

  Exact solution: φ = x (linear potential gradient).

  Exports phi, rhs, and residual at each V-cycle iteration to
  build/outputs/ for visualisation in VisIt or ParaView.
*/
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include "solver.hpp"
#include "exporter.hpp"

int main() {
  std::printf("meadows-cpp: 2D parallel plate capacitor (Laplace) with GMG\n\n");

  const int N = 128;
  const double dx = 1.0 / N;

  // --- build grid hierarchy ---
  GridGeometry<double, 3> finest_geom(
    {dx / 2.0, dx / 2.0, dx / 2.0}, {dx, dx, dx}, {N, N, N}, 0, 2);

  GridHierarchy<double, 3> hierarchy;
  int phi_idx = hierarchy.register_component("phi");
  int rhs_idx = hierarchy.register_component("rhs");
  int res_idx = hierarchy.register_component("res");
  hierarchy.build(finest_geom, 4); 

  std::printf("Grid: %dx%dx%d - %d levels (coarsest %dx%dx%d)\n",
    N, N, N, hierarchy.num_levels(),
    hierarchy.level(0).geometry()._n_interior[0],
    hierarchy.level(0).geometry()._n_interior[1],
    hierarchy.level(0).geometry()._n_interior[2] );
    

  // --- RHS = 0 (Laplace equation) ---
  // already zero from allocation

  // --- boundary conditions: parallel plate capacitor ---
  DirichletBC<double, 3> bc_lo;
  bc_lo.value = 0.0;   // grounded plate at x=0
  DirichletBC<double, 3> bc_hi;
  bc_hi.value = 20000.0;   // charged plate at x=1
  NeumannBC<double, 3> bc_neumann;
  bc_neumann.flux = 0.0;  // insulating walls on y-faces

  BCRegistry<double, 3> bc_reg;
  bc_reg.set(0, 0, bc_lo);      // x-low: φ = 0
  bc_reg.set(0, 1, bc_hi);      // x-high: φ = 1
  bc_reg.set(1, 0, bc_neumann); // y-low: ∂φ/∂n = 0
  bc_reg.set(1, 1, bc_neumann); // y-high: ∂φ/∂n = 0
  bc_reg.set(2, 0, bc_neumann); // z-low: ∂φ/∂n = 0
  bc_reg.set(2, 1, bc_neumann); // z-high: ∂φ/∂n = 0

  // --- set up exporter ---
  GridExporter<double, 3> exporter("build/outputs", "capacitor");
  std::vector<std::string> export_comps = {"phi", "rhs", "res"};
  int total_steps = 0;

  // --- solve with per-iteration export ---
  EllipticSolverGMG<double, 3> solver(
    hierarchy, bc_reg, phi_idx, rhs_idx, res_idx,
    10, 10, 50,    // pre/post/coarse smooth
    100,          // max V-cycles
    1e-10,       // tolerance
    true);       // verbose

  std::printf("\nSolving...\n");
  solver.solve_impl([&](int step, double res_norm) {
    exporter.write(hierarchy, export_comps, step);
    total_steps = step + 1;
    (void)res_norm;  // already printed by verbose mode
  });

  // --- write time-series master file ---
  exporter.write_time_series(total_steps, finest_geom, export_comps);

  // --- verify against exact solution: φ = x ---
  /*
  auto phi_acc = hierarchy.finest().accessor(phi_idx);
  double max_err = 0.0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      double x = (i + 0.5) * dx;
      double exact = x;
      double err = std::fabs(phi_acc(i, j) - exact);
      if (err > max_err) max_err = err;
    }
  }
  */

  //std::printf("\nMax error vs exact solution (φ = x): %.6e\n", max_err);
  std::printf("Output: build/outputs/capacitor.xmf (open in VisIt/ParaView)\n");

  return 0;
}

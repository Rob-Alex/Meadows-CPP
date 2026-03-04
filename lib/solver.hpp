/*
  solver.hpp
  Robbie Alexander
  Geometric multigrid solver for elliptic equations (Poisson / Helmholtz).
  Uses CRTP for the solver base, wires together:
    - RBGS smoother (operators.hpp)
    - Laplacian + residual (operators.hpp)
    - Restriction + Prolongation (interlevel.hpp)
    - GridHierarchy (grid.hpp)
    - BCRegistry (boundary_conditions.hpp)

  Key design: the V-cycle uses the ORIGINAL (inhomogeneous) BCs on the
  entry level and HOMOGENEOUS BCs on all coarser correction levels.
  The correction equation L(e) = r has e = 0 at boundaries because
  the fine-level BCs are already satisfied — the error is zero there.
*/
#pragma once
#include "grid.hpp"
#include "boundary_conditions.hpp"
#include "operators.hpp"
#include "interlevel.hpp"
#include <cstdio>
#include <functional>

template<class Derived, typename T, int Dims>
class SolverBase {
public:
  void solve() {
    static_cast<Derived*>(this)->solve_impl();
  }
protected:
  SolverBase() = default;
};

// EllipticSolverGMG: geometric multigrid for L(phi) = rhs.
//
// Does NOT own the GridHierarchy or BCRegistry — references them.
// The caller is responsible for:
//   1. Registering components (phi, rhs, res) on the hierarchy
//   2. Building the hierarchy
//   3. Setting the RHS on the finest level before calling solve()
//
// The solver caches component indices and smoothing parameters.

template<typename T, int Dims, typename Alloc = HostAllocator<T, default_tracking<T>>>
class EllipticSolverGMG : public SolverBase<EllipticSolverGMG<T, Dims, Alloc>, T, Dims> {
private:
  GridHierarchy<T, Dims, Alloc>& _hierarchy;
  const BCRegistry<T, Dims>& _bc_registry;
  BCRegistry<T, Dims> _homogeneous_bc;
  RBGS<T, Dims> _smoother;

  // cached component indices
  int _phi;
  int _rhs;
  int _res;

  // solver parameters
  int _n_pre_smooth;
  int _n_post_smooth;
  int _n_coarse_smooth;
  int _max_vcycles;
  T _tol;
  bool _verbose;

public:
  EllipticSolverGMG(GridHierarchy<T, Dims, Alloc>& hierarchy,
                    const BCRegistry<T, Dims>& bc_registry,
                    int phi_comp, int rhs_comp, int res_comp,
                    int n_pre_smooth = 2,
                    int n_post_smooth = 2,
                    int n_coarse_smooth = 20,
                    int max_vcycles = 50,
                    T tol = T{1e-10},
                    bool verbose = false)
    : _hierarchy(hierarchy),
      _bc_registry(bc_registry),
      _homogeneous_bc(bc_registry.make_homogeneous()),
      _phi(phi_comp), _rhs(rhs_comp), _res(res_comp),
      _n_pre_smooth(n_pre_smooth),
      _n_post_smooth(n_post_smooth),
      _n_coarse_smooth(n_coarse_smooth),
      _max_vcycles(max_vcycles),
      _tol(tol),
      _verbose(verbose) {}

  // V-cycle: recursive multigrid correction scheme.
  // level = index into hierarchy (0 = coarsest).
  // is_correction: when true, use homogeneous BCs (correction equation).
  //   The entry level of the V-cycle uses original (inhomogeneous) BCs.
  //   All coarser levels solve L(e) = r where e = 0 at boundaries.
  void V_cycle(int level, bool is_correction = false) {
    auto& lvl = _hierarchy.level(level);
    const auto& geom = lvl.geometry();
    auto phi_acc = lvl.accessor(_phi);
    auto rhs_acc = lvl.accessor(_rhs);

    // select BC registry: original for the entry level, homogeneous for correction
    const auto& bc = is_correction ? _homogeneous_bc : _bc_registry;

    auto fill_bc = [&](FieldAccessor<T, Dims>& acc) {
      bc.fill_all(acc, geom);
    };

    // --- base case: coarsest level, just smooth a lot ---
    if (level == 0) {
      fill_bc(phi_acc);
      _smoother.smooth(phi_acc, rhs_acc, geom, _n_coarse_smooth, fill_bc);
      return;
    }

    // --- pre-smooth ---
    fill_bc(phi_acc);
    _smoother.smooth(phi_acc, rhs_acc, geom, _n_pre_smooth, fill_bc);

    // --- compute residual: res = rhs - L(phi) ---
    fill_bc(phi_acc);
    auto res_acc = lvl.accessor(_res);
    residual<StandardFlux, T, Dims>::compute(phi_acc, rhs_acc, res_acc, geom);

    // --- restrict residual to coarser level's RHS ---
    auto& coarse_lvl = _hierarchy.level(level - 1);
    const auto& coarse_geom = coarse_lvl.geometry();
    auto coarse_rhs = coarse_lvl.accessor(_rhs);
    Restriction<T, Dims>::apply(res_acc, coarse_rhs, coarse_geom,
                                _hierarchy.ref_ratio());

    // --- zero the coarse correction ---
    auto coarse_phi = coarse_lvl.accessor(_phi);
    zero_field(coarse_phi, coarse_geom);

    // --- recurse: coarser levels always solve the correction equation ---
    V_cycle(level - 1, true);

    // --- prolongate coarse correction back and add to fine phi ---
    Prolongation<T, Dims>::apply(coarse_phi, phi_acc, geom,
                                 _hierarchy.ref_ratio());

    // --- post-smooth ---
    fill_bc(phi_acc);
    _smoother.smooth(phi_acc, rhs_acc, geom, _n_post_smooth, fill_bc);
  }

  // Full Multigrid: provides an O(h^2) initial guess before V-cycling.
  //
  // 1. Restrict RHS from finest down to all coarser levels
  // 2. Solve coarsest with original BCs (full problem)
  // 3. For each level upward: prolongate solution as initial guess,
  //    then run a V-cycle with original BCs to improve it
  void full_multigrid_cycle() {
    int finest = _hierarchy.finest_level();

    // --- restrict RHS from finest down to coarsest ---
    for (int lvl = finest; lvl >= 1; --lvl) {
      auto fine_rhs = _hierarchy.level(lvl).accessor(_rhs);
      auto coarse_rhs = _hierarchy.level(lvl - 1).accessor(_rhs);
      const auto& coarse_geom = _hierarchy.level(lvl - 1).geometry();
      Restriction<T, Dims>::apply(fine_rhs, coarse_rhs, coarse_geom,
                                  _hierarchy.ref_ratio());
    }

    // --- solve coarsest level with original BCs (full problem) ---
    auto& coarse_lvl = _hierarchy.level(0);
    const auto& coarse_geom = coarse_lvl.geometry();
    auto coarse_phi = coarse_lvl.accessor(_phi);
    auto coarse_rhs = coarse_lvl.accessor(_rhs);
    zero_field(coarse_phi, coarse_geom);

    auto fill_bc_coarse = [&](FieldAccessor<T, Dims>& acc) {
      _bc_registry.fill_all(acc, coarse_geom);
    };
    fill_bc_coarse(coarse_phi);
    _smoother.smooth(coarse_phi, coarse_rhs, coarse_geom,
                     _n_coarse_smooth, fill_bc_coarse);

    // --- work upward: prolongate + V-cycle on each level ---
    // Each level solves the full problem (not correction), so
    // V_cycle is called with is_correction=false.
    for (int lvl = 1; lvl <= finest; ++lvl) {
      auto& level = _hierarchy.level(lvl);
      const auto& geom = level.geometry();
      auto phi_acc = level.accessor(_phi);

      // prolongate coarser solution as initial guess
      // first zero phi, then += from coarse (so it's a pure interpolation)
      zero_field(phi_acc, geom);
      auto coarser_phi = _hierarchy.level(lvl - 1).accessor(_phi);
      Prolongation<T, Dims>::apply(coarser_phi, phi_acc, geom,
                                   _hierarchy.ref_ratio());

      // V-cycle to improve the guess (full problem, not correction)
      V_cycle(lvl, false);
    }
  }

  // solve(): top-level entry point.
  // FMG for initial guess, then iterate V-cycles until convergence.
  // Optional iteration_callback is called after each V-cycle with
  // (cycle_number, residual_norm). Use this for per-iteration export.
  template<typename IterCallback = decltype([](int, T){})>
  void solve_impl(IterCallback&& on_iteration = {}) {
    int finest = _hierarchy.finest_level();

    // FMG for a good initial guess
    full_multigrid_cycle();

    // export the FMG initial guess as step 0
    {
      auto& lvl = _hierarchy.level(finest);
      const auto& geom = lvl.geometry();
      auto phi_acc = lvl.accessor(_phi);
      auto rhs_acc = lvl.accessor(_rhs);
      auto res_acc = lvl.accessor(_res);

      auto fill_bc = [&](FieldAccessor<T, Dims>& acc) {
        _bc_registry.fill_all(acc, geom);
      };
      fill_bc(phi_acc);

      T res_norm = residual<StandardFlux, T, Dims>::compute(
        phi_acc, rhs_acc, res_acc, geom);

      if (_verbose) {
        std::printf("  FMG initial: ||res|| = %.6e\n", static_cast<double>(res_norm));
      }
      on_iteration(0, res_norm);
    }

    // iterate V-cycles until converged
    // Entry-level V-cycle uses original BCs; correction levels use homogeneous
    for (int cycle = 0; cycle < _max_vcycles; ++cycle) {
      V_cycle(finest, false);

      // compute residual norm on finest level
      auto& lvl = _hierarchy.level(finest);
      const auto& geom = lvl.geometry();
      auto phi_acc = lvl.accessor(_phi);
      auto rhs_acc = lvl.accessor(_rhs);
      auto res_acc = lvl.accessor(_res);

      auto fill_bc = [&](FieldAccessor<T, Dims>& acc) {
        _bc_registry.fill_all(acc, geom);
      };
      fill_bc(phi_acc);

      T res_norm = residual<StandardFlux, T, Dims>::compute(
        phi_acc, rhs_acc, res_acc, geom);

      if (_verbose) {
        std::printf("  V-cycle %d: ||res|| = %.6e\n", cycle + 1, static_cast<double>(res_norm));
      }

      on_iteration(cycle + 1, res_norm);

      if (res_norm < _tol) {
        if (_verbose) {
          std::printf("  Converged after %d V-cycles\n", cycle + 1);
        }
        return;
      }
    }

    if (_verbose) {
      std::printf("  Warning: did not converge in %d V-cycles\n", _max_vcycles);
    }
  }

private:
  // zero all interior cells of a field
  static void zero_field(FieldAccessor<T, Dims>& acc,
                         const GridGeometry<T, Dims>& geom) {
    std::array<int, Dims> idx{};
    InteriorLoop<Dims, 0>::run(geom._n_interior, idx,
      [&](const std::array<int, Dims>& cell) {
        acc[cell] = T{0};
      });
  }
};

/*
TODO: Implement hyperbolic solver
template<typename T, int Dims>
class HyperbolicSolver : public SolverBase<HyperbolicSolver<T, Dims>, T, Dims> {

};
*/

/*
  Operators.hpp
  Author: Me
  this contains all the relevant stencil operators, etc
*/
#pragma once
#include "grid.hpp"
#include "parallel_loop.hpp"
#include <cmath>

// InteriorLoop: compile-time nested loop over all Dims dimensions.
// Same recursive template pattern as TransverseLoop (used by BCs)
// but iterates every dimension — no skip.
// D walks from 0 to Dims. Base case at D == Dims calls the lambda.

template<int Dims, int D>
struct InteriorLoop {
  template<typename Func>
  static void run(const std::array<int, Dims>& ni,
                  std::array<int, Dims>& idx, Func&& f) {
    for (int i = 0; i < ni[D]; ++i) {
      idx[D] = i;
      InteriorLoop<Dims, D + 1>::run(ni, idx, f);
    }
  }
};

template<int Dims>
struct InteriorLoop<Dims, Dims> {
  template<typename Func>
  static void run(const std::array<int, Dims>&,
                  std::array<int, Dims>& idx, Func&& f) {
    f(idx);
  }
};

// StandardFlux: default flux policy for the Laplacian.
// Accumulates the standard 2nd-order central difference contribution
// for one dimension into the running Laplacian value.

struct StandardFlux {
  template<typename T, int Dims, typename IdxArray>
  static void accumulate(const FieldAccessor<T, Dims>& phi,
                         const IdxArray& idx,
                         int d, T inv_dx2, T& lap) {
    auto idx_p = idx;  idx_p[d] += 1;
    auto idx_m = idx;  idx_m[d] -= 1;
    lap += (phi[idx_p] + phi[idx_m] - T{2} * phi[idx]) * inv_dx2;
  }
};

// Laplacian: applies L(phi) = sum_d d^2(phi)/dx_d^2 over all interior cells.
// FluxPolicy controls per-face contribution (StandardFlux now, EBFlux later).

template<typename FluxPolicy, typename T, int Dims, typename Alloc = HostAllocator<T, default_tracking<T>>>
struct laplacian {
  static void apply(FieldAccessor<T, Dims>& phi,
                    FieldAccessor<T, Dims>& Lphi,
                    const GridGeometry<T, Dims>& geom) {

    std::array<T, Dims> inv_dx2;
    for (int d = 0; d < Dims; ++d) {
      inv_dx2[d] = T{1} / (geom._spacing[d] * geom._spacing[d]);
    }

    parallel_for_interior<T, Dims>(geom,
      [&](const std::array<int, Dims>& cell) {
        T lap = T{0};
        for (int d = 0; d < Dims; ++d) {
          FluxPolicy::accumulate(phi, cell, d, inv_dx2[d], lap);
        }
        Lphi[cell] = lap;
      });
  }
};

// l2_norm: sqrt(sum(phi^2) / N) over interior cells.

template<typename T, int Dims, typename Alloc = HostAllocator<T, default_tracking<T>>>
struct l2_norm {
  static T compute(FieldAccessor<T, Dims>& phi,
                   const GridGeometry<T, Dims>& geom) {
    T sum = parallel_reduce_interior<T, Dims>(geom, T{0},
      [&](const std::array<int, Dims>& cell) -> T {
        return phi[cell] * phi[cell];
      });
    return std::sqrt(sum / static_cast<T>(geom.total_interior_cells()));
  }
};

// residual: computes res = rhs - L(phi), returns L2 norm of res.
// Fused into a single pass — avoids a separate Lphi buffer.

template<typename FluxPolicy, typename T, int Dims, typename Alloc = HostAllocator<T, default_tracking<T>>>
struct residual {
  static T compute(FieldAccessor<T, Dims>& phi,
                   FieldAccessor<T, Dims>& rhs,
                   FieldAccessor<T, Dims>& res,
                   const GridGeometry<T, Dims>& geom) {

    std::array<T, Dims> inv_dx2;
    for (int d = 0; d < Dims; ++d) {
      inv_dx2[d] = T{1} / (geom._spacing[d] * geom._spacing[d]);
    }

    T sum = parallel_reduce_interior<T, Dims>(geom, T{0},
      [&](const std::array<int, Dims>& cell) -> T {
        T lap = T{0};
        for (int d = 0; d < Dims; ++d) {
          FluxPolicy::accumulate(phi, cell, d, inv_dx2[d], lap);
        }
        res[cell] = rhs[cell] - lap;
        return res[cell] * res[cell];
      });

    return std::sqrt(sum / static_cast<T>(geom.total_interior_cells()));
  }
};

// SmootherBase: CRTP base for smoothers.
// Derived must implement:
//   smooth_impl(phi, rhs, geom, n_sweeps, fill_bc_fn)
// fill_bc_fn is a callable: void(FieldAccessor<T, Dims>&)
// so the smoother can fill BCs between colour sweeps without
// coupling to BCRegistry.

template<class Derived, typename T, int Dims>
struct SmootherBase {
  template<typename FillBCFn>
  void smooth(FieldAccessor<T, Dims>& phi,
              FieldAccessor<T, Dims>& rhs,
              const GridGeometry<T, Dims>& geom,
              int n_sweeps,
              FillBCFn&& fill_bc_fn) {
    static_cast<Derived*>(this)->smooth_impl(
      phi, rhs, geom, n_sweeps, std::forward<FillBCFn>(fill_bc_fn));
  }
};

// RBGS: Red-Black Gauss-Seidel smoother for the Laplacian.
//
// Cell colouring: (i + j + ...) % 2
//   red = 0, black = 1
//
// Each sweep:
//   1. Update all red cells   (neighbours are all black → no conflicts)
//   2. Fill BCs               (ghost values may have changed)
//   3. Update all black cells (neighbours are all red → no conflicts)
//
// Update formula (derived from solving L(phi)=rhs locally for phi[cell]):
//   off_diagonal = sum_d (phi[i+1,d] + phi[i-1,d]) * inv_dx2[d]
//   diagonal     = sum_d (-2 * inv_dx2[d])
//   phi[cell]    = (rhs[cell] - off_diagonal) / diagonal
//
// In-place — no scratch buffer. Each colour's neighbours are the
// other colour, so reads are always from the "old" values.
// GPU: each colour sweep is embarrassingly parallel → parallel_for.

template<typename T, int Dims>
struct RBGS : SmootherBase<RBGS<T, Dims>, T, Dims> {

  template<typename FillBCFn>
  void smooth_impl(FieldAccessor<T, Dims>& phi,
                   FieldAccessor<T, Dims>& rhs,
                   const GridGeometry<T, Dims>& geom,
                   int n_sweeps,
                   FillBCFn&& fill_bc_fn) {

    // precompute 1/dx^2 per dimension and the diagonal coefficient
    std::array<T, Dims> inv_dx2;
    T diag = T{0};
    for (int d = 0; d < Dims; ++d) {
      inv_dx2[d] = T{1} / (geom._spacing[d] * geom._spacing[d]);
      diag += T{-2} * inv_dx2[d];
    }

    for (int sweep = 0; sweep < n_sweeps; ++sweep) {
      // two passes per sweep: colour 0 (red) then colour 1 (black)
      for (int colour = 0; colour < 2; ++colour) {
        parallel_for_interior_colour<T, Dims>(geom, colour,
          [&](const std::array<int, Dims>& cell) {
            // accumulate off-diagonal: sum of neighbour contributions
            T off_diag = T{0};
            for (int d = 0; d < Dims; ++d) {
              auto idx_p = cell;  idx_p[d] += 1;
              auto idx_m = cell;  idx_m[d] -= 1;
              off_diag += (phi[idx_p] + phi[idx_m]) * inv_dx2[d];
            }

            phi[cell] = (rhs[cell] - off_diag) / diag;
          });

        // fill BCs after each colour pass so the next colour
        // sees correct ghost values
        fill_bc_fn(phi);
      }
    }
  }
};

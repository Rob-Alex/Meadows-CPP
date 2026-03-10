/*
  Meadows CPP
  Robbie Alexander
  operators.hpp

  Stencil operators for the elliptic solver.
  Hot-path operators (RBGS, residual) use raw pointer arithmetic
  in the inner loop for maximum throughput:
    - No per-cell index array copies
    - No per-cell stride computation
    - Precomputed reciprocal for division -> multiplication
    - Inner dimension contiguous for SIMD auto-vectorisation
*/
#pragma once
#include "grid.hpp"
#include "parallel_loop.hpp"
#include <cmath>
#include <cstddef>

// InteriorLoop: compile-time nested loop over all Dims dimensions.
// Used by non-hot-path code (zero_field, BC fill, etc.)

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

// StandardFlux: flux policy for the Laplacian (kept for non-hot-path use).

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

// laplacian: L(phi) over all interior cells.
// Not on the critical path — uses the lambda-based parallel loop.

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

// ============================================================================
// residual: res = rhs - L(phi), returns L2 norm.
// ============================================================================
// Raw pointer inner loop: the innermost dimension runs contiguously
// with direct pointer offsets. No index arrays, no stride loops.
// The FluxPolicy parameter is retained for API compatibility;
// the implementation uses the standard 2nd-order central difference.

template<typename FluxPolicy, typename T, int Dims, typename Alloc = HostAllocator<T, default_tracking<T>>>
struct residual {
  static T compute(FieldAccessor<T, Dims>& phi,
                   FieldAccessor<T, Dims>& rhs,
                   FieldAccessor<T, Dims>& res,
                   const GridGeometry<T, Dims>& geom) {

    const auto& ni = geom._n_interior;

    std::array<T, Dims> inv_dx2;
    T diag = T{0};
    for (int d = 0; d < Dims; ++d) {
      inv_dx2[d] = T{1} / (geom._spacing[d] * geom._spacing[d]);
      diag += T{-2} * inv_dx2[d];
    }

    // Raw data pointers (pre-offset to interior start by accessor)
    T* phi_ptr = phi._data;
    T* rhs_ptr = rhs._data;
    T* res_ptr = res._data;

    // Signed strides for safe negative indexing into ghost cells
    std::array<ptrdiff_t, Dims> s;
    for (int d = 0; d < Dims; ++d) {
      s[d] = static_cast<ptrdiff_t>(phi._strides[d]);
    }

    int outer_total = 1;
    for (int d = 0; d < Dims - 1; ++d) {
      outer_total *= ni[d];
    }
    const int ni_inner = ni[Dims - 1];

    T total_sum = T{0};

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(+:total_sum) if(outer_total >= OMP_MIN_OUTER)
#endif
    for (int outer = 0; outer < outer_total; ++outer) {
      // Unflatten outer -> base offset (amortised once per row)
      int rem = outer;
      ptrdiff_t base = 0;
      for (int d = Dims - 2; d >= 0; --d) {
        int idx = rem % ni[d];
        rem /= ni[d];
        base += idx * s[d];
      }

      T row_sum = T{0};
      for (int j = 0; j < ni_inner; ++j) {
        ptrdiff_t c = base + j;

        // Stencil: L(phi) at cell c
        T center = phi_ptr[c];
        T lap = (phi_ptr[c + 1] + phi_ptr[c - 1]) * inv_dx2[Dims - 1];
        for (int d = 0; d < Dims - 1; ++d) {
          lap += (phi_ptr[c + s[d]] + phi_ptr[c - s[d]]) * inv_dx2[d];
        }
        lap += diag * center;  // diagonal: -2 * sum(inv_dx2) * phi[c]

        // Wait: the full Laplacian already includes the -2*center term
        // per dimension inside each (phi+ + phi- - 2*phi) * inv_dx2.
        // By separating: neighbor_sum * inv_dx2 + diag * center.
        // diag = sum(-2 * inv_dx2[d]) so: lap = neighbor_weighted_sum + diag*center. Correct.

        T r = rhs_ptr[c] - lap;
        res_ptr[c] = r;
        row_sum += r * r;
      }
      total_sum += row_sum;
    }

    return std::sqrt(total_sum / static_cast<T>(geom.total_interior_cells()));
  }
};

// gradient: E = -nabla(phi) at cell centres.
// Not on the critical path — uses the lambda-based parallel loop.

template<typename T, int Dims, typename Alloc = HostAllocator<T, default_tracking<T>>>
struct gradient {
  static void compute(FieldAccessor<T, Dims>& phi,
                      std::array<FieldAccessor<T, Dims>, Dims>& E,
                      FieldAccessor<T, Dims>& E_mag,
                      const GridGeometry<T, Dims>& geom) {

    std::array<T, Dims> inv_2dx;
    for (int d = 0; d < Dims; ++d) {
      inv_2dx[d] = T{1} / (T{2} * geom._spacing[d]);
    }

    parallel_for_interior<T, Dims>(geom,
      [&](const std::array<int, Dims>& cell) {
        T mag_sq = T{0};
        for (int d = 0; d < Dims; ++d) {
          auto idx_p = cell;  idx_p[d] += 1;
          auto idx_m = cell;  idx_m[d] -= 1;
          T Ed = -(phi[idx_p] - phi[idx_m]) * inv_2dx[d];
          E[d][cell] = Ed;
          mag_sq += Ed * Ed;
        }
        E_mag[cell] = std::sqrt(mag_sq);
      });
  }
};

// SmootherBase: CRTP base.

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

// ============================================================================
// RBGS: Red-Black Gauss-Seidel with raw pointer inner loop.
// ============================================================================
// The inner loop uses direct pointer arithmetic with signed strides:
//   phi_ptr[c ± 1]      — inner dimension neighbours (contiguous)
//   phi_ptr[c ± s[d]]   — outer dimension neighbours
//   inv_diag             — precomputed reciprocal (multiply, not divide)
//
// The parallel loop is inlined: OpenMP on the outer row loop, serial
// on the inner column loop. This gives one fork/join per colour pass
// and the inner loop is fully vectorisable.

template<typename T, int Dims>
struct RBGS : SmootherBase<RBGS<T, Dims>, T, Dims> {

  template<typename FillBCFn>
  void smooth_impl(FieldAccessor<T, Dims>& phi,
                   FieldAccessor<T, Dims>& rhs,
                   const GridGeometry<T, Dims>& geom,
                   int n_sweeps,
                   FillBCFn&& fill_bc_fn) {

    const auto& ni = geom._n_interior;

    // Precompute coefficients
    std::array<T, Dims> inv_dx2;
    T diag = T{0};
    for (int d = 0; d < Dims; ++d) {
      inv_dx2[d] = T{1} / (geom._spacing[d] * geom._spacing[d]);
      diag += T{-2} * inv_dx2[d];
    }
    T inv_diag = T{1} / diag;  // multiply is ~20x faster than divide

    // Raw data pointers
    T* phi_ptr = phi._data;
    const T* rhs_ptr = rhs._data;

    // Signed strides for safe negative indexing into ghost cells
    std::array<ptrdiff_t, Dims> s;
    for (int d = 0; d < Dims; ++d) {
      s[d] = static_cast<ptrdiff_t>(phi._strides[d]);
    }

    // Outer loop size (all dimensions except innermost)
    int outer_total = 1;
    for (int d = 0; d < Dims - 1; ++d) {
      outer_total *= ni[d];
    }
    const int ni_inner = ni[Dims - 1];

    for (int sweep = 0; sweep < n_sweeps; ++sweep) {
      for (int colour = 0; colour < 2; ++colour) {

#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(outer_total >= OMP_MIN_OUTER)
#endif
        for (int outer = 0; outer < outer_total; ++outer) {
          // Unflatten outer indices and compute row base offset
          int rem = outer;
          int outer_sum = 0;
          ptrdiff_t base = 0;
          for (int d = Dims - 2; d >= 0; --d) {
            int idx = rem % ni[d];
            rem /= ni[d];
            outer_sum += idx;
            base += idx * s[d];
          }

          int j_start = colour ^ (outer_sum & 1);

          // Tight inner loop: raw pointer arithmetic, no index arrays
          for (int j = j_start; j < ni_inner; j += 2) {
            ptrdiff_t c = base + j;

            // Off-diagonal: weighted sum of all neighbours
            T off_diag = (phi_ptr[c + 1] + phi_ptr[c - 1]) * inv_dx2[Dims - 1];
            for (int d = 0; d < Dims - 1; ++d) {
              off_diag += (phi_ptr[c + s[d]] + phi_ptr[c - s[d]]) * inv_dx2[d];
            }

            phi_ptr[c] = (rhs_ptr[c] - off_diag) * inv_diag;
          }
        }

        fill_bc_fn(phi);
      }
    }
  }
};

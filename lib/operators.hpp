/*
  Operators.hpp
  Author: Me
  this contains all the relevant stencil operators, etc
*/
#pragma once
#include "grid.hpp"
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

    std::array<int, Dims> idx{};
    InteriorLoop<Dims, 0>::run(geom._n_interior, idx,
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
    T sum = T{0};
    std::array<int, Dims> idx{};
    InteriorLoop<Dims, 0>::run(geom._n_interior, idx,
      [&](const std::array<int, Dims>& cell) {
        sum += phi[cell] * phi[cell];
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

    T sum = T{0};
    std::array<int, Dims> idx{};
    InteriorLoop<Dims, 0>::run(geom._n_interior, idx,
      [&](const std::array<int, Dims>& cell) {
        T lap = T{0};
        for (int d = 0; d < Dims; ++d) {
          FluxPolicy::accumulate(phi, cell, d, inv_dx2[d], lap);
        }
        res[cell] = rhs[cell] - lap;
        sum += res[cell] * res[cell];
      });

    return std::sqrt(sum / static_cast<T>(geom.total_interior_cells()));
  }
};

/*
  Meadows CPP
  Robbie Alexander
  interlevel.hpp

  Inter-level transfer operators for geometric multigrid.
  Restriction (fine -> coarse) and Prolongation (coarse -> fine).

  Restriction uses a compile-time ChildLoop to iterate fine children
  without any integer division or modulo (replaces the old flat+unflatten).

  Prolongation uses raw pointer arithmetic in the inner loop with
  integer division only for the parent index (which the compiler
  optimises to a right-shift for power-of-2 ratios).
*/
#pragma once
#include "grid.hpp"
#include "operators.hpp"
#include "parallel_loop.hpp"
#include <cstddef>

// ============================================================================
// ChildLoop: compile-time nested loop over ratio^Dims children.
// ============================================================================
// Generates one loop per dimension (like InteriorLoop) but each
// dimension iterates [0, ratio) and computes the fine cell index
// as coarse_cell[D] * ratio + k. No integer division or modulo.
// The compiler fully unrolls this for small ratio (2 or 4).

template<int Dims, int D>
struct ChildLoop {
  template<typename Func>
  static void run(std::array<int, Dims>& fine_cell,
                  const std::array<int, Dims>& coarse_cell,
                  int ratio, Func&& f) {
    int base = coarse_cell[D] * ratio;
    for (int k = 0; k < ratio; ++k) {
      fine_cell[D] = base + k;
      ChildLoop<Dims, D + 1>::run(fine_cell, coarse_cell, ratio, f);
    }
  }
};

template<int Dims>
struct ChildLoop<Dims, Dims> {
  template<typename Func>
  static void run(std::array<int, Dims>& fine_cell,
                  const std::array<int, Dims>&,
                  int, Func&& f) {
    f(fine_cell);
  }
};

// ============================================================================
// Restriction: fine -> coarse via cell averaging.
// ============================================================================
// Each coarse cell = mean of ratio^Dims fine children.
// Uses ChildLoop for the child iteration: zero integer division,
// fully unrollable by the compiler.

template<typename T, int Dims>
struct Restriction {
  static void apply(FieldAccessor<T, Dims>& fine,
                    FieldAccessor<T, Dims>& coarse,
                    const GridGeometry<T, Dims>& coarse_geom,
                    int ratio) {

    int n_children = 1;
    for (int d = 0; d < Dims; ++d) {
      n_children *= ratio;
    }
    T inv_children = T{1} / static_cast<T>(n_children);

    parallel_for_interior<T, Dims>(coarse_geom,
      [&](const std::array<int, Dims>& coarse_cell) {
        T sum = T{0};
        std::array<int, Dims> fine_cell;
        ChildLoop<Dims, 0>::run(fine_cell, coarse_cell, ratio,
          [&](const std::array<int, Dims>& fc) {
            sum += fine[fc];
          });
        coarse[coarse_cell] = sum * inv_children;
      });
  }
};

// ============================================================================
// Prolongation: coarse -> fine via piecewise-constant injection.
// ============================================================================
// Uses += (not =): adds the coarse correction to the existing fine solution.
//
// Raw pointer inner loop: the innermost dimension runs contiguously.
// Parent index = fine_index / ratio, which the compiler optimises to
// a right-shift for power-of-2 ratios (the common case).

template<typename T, int Dims>
struct Prolongation {
  static void apply(FieldAccessor<T, Dims>& coarse,
                    FieldAccessor<T, Dims>& fine,
                    const GridGeometry<T, Dims>& fine_geom,
                    int ratio) {

    const auto& ni = fine_geom._n_interior;

    T* fine_ptr = fine._data;
    const T* coarse_ptr = coarse._data;

    // Signed strides for both grids
    std::array<ptrdiff_t, Dims> fs, cs;
    for (int d = 0; d < Dims; ++d) {
      fs[d] = static_cast<ptrdiff_t>(fine._strides[d]);
      cs[d] = static_cast<ptrdiff_t>(coarse._strides[d]);
    }

    int outer_total = 1;
    for (int d = 0; d < Dims - 1; ++d) {
      outer_total *= ni[d];
    }
    const int ni_inner = ni[Dims - 1];

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(outer_total >= OMP_MIN_OUTER)
#endif
    for (int outer = 0; outer < outer_total; ++outer) {
      // Unflatten fine outer indices, compute base offsets for both grids
      int rem = outer;
      ptrdiff_t fine_base = 0;
      ptrdiff_t coarse_base = 0;
      for (int d = Dims - 2; d >= 0; --d) {
        int fi = rem % ni[d];
        rem /= ni[d];
        fine_base += fi * fs[d];
        coarse_base += (fi / ratio) * cs[d];
      }

      // Inner loop: fine j -> coarse j/ratio
      for (int j = 0; j < ni_inner; ++j) {
        fine_ptr[fine_base + j] += coarse_ptr[coarse_base + j / ratio];
      }
    }
  }
};

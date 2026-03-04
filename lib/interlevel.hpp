/*
  interlevel.hpp
  Inter-level transfer operators for geometric multigrid.
  Restriction (fine → coarse) and Prolongation (coarse → fine).
  Separate from operators.hpp (single-level stencils) for clean
  single-responsibility includes.
*/
#pragma once
#include "grid.hpp"
#include "operators.hpp"
#include "parallel_loop.hpp"

// Restriction: fine → coarse via cell averaging.
// Each coarse cell = mean of ratio^Dims fine children.
// Preserves conservation: integral of restricted field equals
// integral of fine field.
//
// Iterates coarse cells. For each, sums fine children using a flat
// loop with modular index decomposition to find per-dimension offsets.

template<typename T, int Dims>
struct Restriction {
  static void apply(FieldAccessor<T, Dims>& fine,
                    FieldAccessor<T, Dims>& coarse,
                    const GridGeometry<T, Dims>& coarse_geom,
                    int ratio) {

    // total number of fine children per coarse cell
    int n_children = 1;
    for (int d = 0; d < Dims; ++d) {
      n_children *= ratio;
    }
    T inv_children = T{1} / static_cast<T>(n_children);

    parallel_for_interior<T, Dims>(coarse_geom,
      [&](const std::array<int, Dims>& coarse_cell) {
        T sum = T{0};

        // flat loop over all ratio^Dims fine children
        for (int flat = 0; flat < n_children; ++flat) {
          std::array<int, Dims> fine_cell;
          int remainder = flat;
          for (int d = Dims - 1; d >= 0; --d) {
            fine_cell[d] = coarse_cell[d] * ratio + (remainder % ratio);
            remainder /= ratio;
          }
          sum += fine[fine_cell];
        }

        coarse[coarse_cell] = sum * inv_children;
      });
  }
};

// Prolongation: coarse → fine via piecewise-constant injection.
// Uses += (not =): in the V-cycle the coarse grid holds a correction,
// and prolongation adds that correction to the existing fine solution.
//
// Iterates fine cells, computes parent coarse index via integer division.

template<typename T, int Dims>
struct Prolongation {
  static void apply(FieldAccessor<T, Dims>& coarse,
                    FieldAccessor<T, Dims>& fine,
                    const GridGeometry<T, Dims>& fine_geom,
                    int ratio) {

    parallel_for_interior<T, Dims>(fine_geom,
      [&](const std::array<int, Dims>& fine_cell) {
        std::array<int, Dims> parent;
        for (int d = 0; d < Dims; ++d) {
          parent[d] = fine_cell[d] / ratio;
        }
        fine[fine_cell] += coarse[parent];
      });
  }
};

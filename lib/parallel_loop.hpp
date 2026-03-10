/*
  Meadows CPP
  Robbie Alexander
  parallel_loop.hpp

  Nested-loop parallel iteration for structured grids.

  Key design decisions:
    - The OUTER dimensions (0..Dims-2) are flattened into a single
      parallel loop. OpenMP forks/joins ONCE per call, not per cell.
    - The INNERMOST dimension (Dims-1) runs as a tight sequential
      loop, contiguous in row-major memory. This enables the compiler
      to auto-vectorise (SIMD) the stencil work.
    - Unflatten (integer div/mod) is amortised: paid once per row,
      not once per cell. For 2D this means zero unflatten cost.
    - The colour loop uses stride-2 on the inner dimension with a
      computed start offset, so it visits ONLY matching cells —
      no branch, no wasted iterations.
    - A minimum-work threshold (OMP_MIN_OUTER) prevents OpenMP
      fork/join on coarse multigrid levels where the overhead
      would exceed the computation time.

  Guards all OpenMP pragmas behind #ifdef _OPENMP so the code
  compiles and runs correctly with or without -fopenmp.
*/
#pragma once
#include <array>

// Minimum outer iterations to justify OpenMP fork/join.
// Below this, the parallel region overhead dominates.
// For 2D, outer_total = number of rows. A 64×64 grid has 64 rows —
// enough work. A 4×4 coarse grid has 4 rows — not worth threading.
#ifndef OMP_MIN_OUTER
#define OMP_MIN_OUTER 32
#endif

// ============================================================================
// parallel_for_interior
// ============================================================================
// Iterates all interior cells. OpenMP parallelises outer dimensions;
// innermost dimension is a tight sequential loop for SIMD.
//
// Lambda signature: void(const std::array<int, Dims>&)

template<typename T, int Dims, typename Func>
void parallel_for_interior(const GridGeometry<T, Dims>& geom, Func&& f) {
  const auto& ni = geom._n_interior;

  // Total iterations over all dimensions except the innermost
  int outer_total = 1;
  for (int d = 0; d < Dims - 1; ++d) {
    outer_total *= ni[d];
  }

  // Only fork threads when there's enough outer work to amortise overhead
#ifdef _OPENMP
  if (outer_total >= OMP_MIN_OUTER) {
    #pragma omp parallel for schedule(static)
    for (int outer = 0; outer < outer_total; ++outer) {
      std::array<int, Dims> cell{};
      int rem = outer;
      for (int d = Dims - 2; d >= 0; --d) {
        cell[d] = rem % ni[d];
        rem /= ni[d];
      }
      for (int j = 0; j < ni[Dims - 1]; ++j) {
        cell[Dims - 1] = j;
        f(cell);
      }
    }
    return;
  }
#endif

  // Serial fallback: coarse grids or no OpenMP
  for (int outer = 0; outer < outer_total; ++outer) {
    std::array<int, Dims> cell{};
    int rem = outer;
    for (int d = Dims - 2; d >= 0; --d) {
      cell[d] = rem % ni[d];
      rem /= ni[d];
    }
    for (int j = 0; j < ni[Dims - 1]; ++j) {
      cell[Dims - 1] = j;
      f(cell);
    }
  }
}

// ============================================================================
// parallel_reduce_interior
// ============================================================================
// Iterates all interior cells, accumulating a scalar via reduction.
// Same nested structure: outer parallel, inner sequential.
//
// Lambda signature: T(const std::array<int, Dims>&)

template<typename T, int Dims, typename Func>
T parallel_reduce_interior(const GridGeometry<T, Dims>& geom, T init, Func&& f) {
  const auto& ni = geom._n_interior;

  int outer_total = 1;
  for (int d = 0; d < Dims - 1; ++d) {
    outer_total *= ni[d];
  }

  T result = init;

#ifdef _OPENMP
  if (outer_total >= OMP_MIN_OUTER) {
    #pragma omp parallel for schedule(static) reduction(+:result)
    for (int outer = 0; outer < outer_total; ++outer) {
      std::array<int, Dims> cell{};
      int rem = outer;
      for (int d = Dims - 2; d >= 0; --d) {
        cell[d] = rem % ni[d];
        rem /= ni[d];
      }
      for (int j = 0; j < ni[Dims - 1]; ++j) {
        cell[Dims - 1] = j;
        result += f(cell);
      }
    }
    return result;
  }
#endif

  // Serial fallback
  for (int outer = 0; outer < outer_total; ++outer) {
    std::array<int, Dims> cell{};
    int rem = outer;
    for (int d = Dims - 2; d >= 0; --d) {
      cell[d] = rem % ni[d];
      rem /= ni[d];
    }
    for (int j = 0; j < ni[Dims - 1]; ++j) {
      cell[Dims - 1] = j;
      result += f(cell);
    }
  }

  return result;
}

// ============================================================================
// parallel_for_interior_colour
// ============================================================================
// Red-Black iteration: visits ONLY cells where (sum of indices) % 2 == colour.
//
// Instead of iterating all cells and branching, we compute the correct
// starting offset for the innermost dimension and stride by 2:
//   outer_sum = sum of indices in dimensions 0..Dims-2
//   j_start   = colour XOR (outer_sum & 1)
// This means every iteration does useful work — no skipped cells.
//
// Lambda signature: void(const std::array<int, Dims>&)

template<typename T, int Dims, typename Func>
void parallel_for_interior_colour(const GridGeometry<T, Dims>& geom, int colour, Func&& f) {
  const auto& ni = geom._n_interior;

  int outer_total = 1;
  for (int d = 0; d < Dims - 1; ++d) {
    outer_total *= ni[d];
  }

#ifdef _OPENMP
  if (outer_total >= OMP_MIN_OUTER) {
    #pragma omp parallel for schedule(static)
    for (int outer = 0; outer < outer_total; ++outer) {
      std::array<int, Dims> cell{};
      int rem = outer;
      int outer_sum = 0;
      for (int d = Dims - 2; d >= 0; --d) {
        cell[d] = rem % ni[d];
        rem /= ni[d];
        outer_sum += cell[d];
      }
      int j_start = colour ^ (outer_sum & 1);
      for (int j = j_start; j < ni[Dims - 1]; j += 2) {
        cell[Dims - 1] = j;
        f(cell);
      }
    }
    return;
  }
#endif

  // Serial fallback
  for (int outer = 0; outer < outer_total; ++outer) {
    std::array<int, Dims> cell{};
    int rem = outer;
    int outer_sum = 0;
    for (int d = Dims - 2; d >= 0; --d) {
      cell[d] = rem % ni[d];
      rem /= ni[d];
      outer_sum += cell[d];
    }
    int j_start = colour ^ (outer_sum & 1);
    for (int j = j_start; j < ni[Dims - 1]; j += 2) {
      cell[Dims - 1] = j;
      f(cell);
    }
  }
}

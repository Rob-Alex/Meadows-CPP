/*
  parallel_loop.hpp
  Flat parallel iteration utilities for OpenMP parallelisation.
  Guards all OpenMP pragmas behind #ifdef _OPENMP so the code
  compiles and runs correctly with or without -fopenmp.
*/
#pragma once
#include <array>

// unflatten: decompose a flat index into multi-dimensional coordinates.
// Row-major order: last dimension varies fastest.
template<int Dims>
inline std::array<int, Dims> unflatten(int flat, const std::array<int, Dims>& ni) {
  std::array<int, Dims> idx;
  for (int d = Dims - 1; d >= 0; --d) {
    idx[d] = flat % ni[d];
    flat /= ni[d];
  }
  return idx;
}

// parallel_for_interior: iterate all interior cells in parallel.
// Lambda signature: void(const std::array<int, Dims>&)
template<typename T, int Dims, typename Func>
void parallel_for_interior(const GridGeometry<T, Dims>& geom, Func&& f) {
  const int N = geom.total_interior_cells();
  const auto& ni = geom._n_interior;

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int flat = 0; flat < N; ++flat) {
    auto cell = unflatten<Dims>(flat, ni);
    f(cell);
  }
}

// parallel_reduce_interior: iterate all interior cells, accumulating a value.
// Lambda signature: T(const std::array<int, Dims>&)
template<typename T, int Dims, typename Func>
T parallel_reduce_interior(const GridGeometry<T, Dims>& geom, T init, Func&& f) {
  const int N = geom.total_interior_cells();
  const auto& ni = geom._n_interior;
  T result = init;

#ifdef _OPENMP
  #pragma omp parallel for schedule(static) reduction(+:result)
#endif
  for (int flat = 0; flat < N; ++flat) {
    auto cell = unflatten<Dims>(flat, ni);
    result += f(cell);
  }

  return result;
}

// parallel_for_interior_colour: iterate only cells where (sum of indices) % 2 == colour.
// Iterates half the cells and computes colour-correct indices directly.
// Lambda signature: void(const std::array<int, Dims>&)
template<typename T, int Dims, typename Func>
void parallel_for_interior_colour(const GridGeometry<T, Dims>& geom, int colour, Func&& f) {
  const int N = geom.total_interior_cells();
  const auto& ni = geom._n_interior;

  // Count cells of this colour: ceil(N/2) or floor(N/2)
  // Simple approach: flat loop + skip non-matching cells.
  // The branch is well-predicted and the iteration cost is negligible
  // compared to the stencil work.
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int flat = 0; flat < N; ++flat) {
    auto cell = unflatten<Dims>(flat, ni);
    int c = 0;
    for (int d = 0; d < Dims; ++d) {
      c += cell[d];
    }
    if (c % 2 != colour) continue;
    f(cell);
  }
}

#pragma once
#include <cstdio>
#include <cstring>
#include "grid.hpp"
#include "bm_utils.hpp"  // Timer, do_not_optimise, print_header

namespace bm {

// BM 1: Allocation/deallocation throughput
// HostAllocator (aligned_alloc) vs new[]/delete[]
void alloc_throughput(size_t count, int iters) {
  std::printf("\n--- alloc/dealloc throughput (count=%zu, iters=%d) ---\n", count, iters);

  // HostAllocator (no tracking)
  {
    using Alloc = HostAllocator<double, no_tracking>;
    Timer t;
    for (int i = 0; i < iters; ++i) {
      double* p = Alloc::allocate(count);
      do_not_optimise(p);
      Alloc::deallocate(p);
    }
    std::printf("  HostAllocator<no_tracking>:    %.3f ms\n", t.elapsed_ms());
  }

  // plain new/delete
  {
    Timer t;
    for (int i = 0; i < iters; ++i) {
      double* p = new double[count];
      do_not_optimise(p);
      delete[] p;
    }
    std::printf("  new[]/delete[]:                %.3f ms\n", t.elapsed_ms());
  }
}

// BM 2: Tracking overhead
// no_tracking vs track_allocations on the same allocator
void tracking_overhead(size_t count, int iters) {
  std::printf("\n--- tracking overhead (count=%zu, iters=%d) ---\n", count, iters);

  // no tracking
  {
    using Alloc = HostAllocator<double, no_tracking>;
    Timer t;
    for (int i = 0; i < iters; ++i) {
      double* p = Alloc::allocate(count);
      do_not_optimise(p);
      Alloc::deallocate(p);
    }
    std::printf("  no_tracking:        %.3f ms\n", t.elapsed_ms());
  }

  // with tracking
  {
    using Alloc = HostAllocator<double, track_allocations>;
    // clear state
    track_allocations::get_allocations().clear();
    Timer t;
    for (int i = 0; i < iters; ++i) {
      double* p = Alloc::allocate(count);
      do_not_optimise(p);
      Alloc::deallocate(p);
    }
    std::printf("  track_allocations:  %.3f ms\n", t.elapsed_ms());
  }
}

// BM 3: Sequential sweep (write + read)
// Measures cache-line benefit of 64-byte alignment vs default new
void sequential_sweep(size_t count, int sweeps) {
  std::printf("\n--- sequential sweep (count=%zu, sweeps=%d) ---\n", count, sweeps);

  using Alloc = HostAllocator<double, no_tracking>;

  // aligned buffer
  {
    double* buf = Alloc::allocate(count);
    std::memset(buf, 0, count * sizeof(double));

    Timer t;
    for (int s = 0; s < sweeps; ++s) {
      for (size_t i = 0; i < count; ++i) {
        buf[i] = buf[i] * 0.5 + 1.0;
      }
      do_not_optimise(buf[count - 1]);
    }
    std::printf("  HostAllocator (64B aligned):  %.3f ms\n", t.elapsed_ms());
    Alloc::deallocate(buf);
  }

  // new[] buffer (default alignment)
  {
    double* buf = new double[count]();

    Timer t;
    for (int s = 0; s < sweeps; ++s) {
      for (size_t i = 0; i < count; ++i) {
        buf[i] = buf[i] * 0.5 + 1.0;
      }
      do_not_optimise(buf[count - 1]);
    }
    std::printf("  new[] (default aligned):      %.3f ms\n", t.elapsed_ms());
    delete[] buf;
  }
}

// BM 4: GridHierarchy component access via FieldAccessor
// Simulates a 2D stencil-style sweep over registered components
void cellgrid_access(int nx, int ny, int sweeps) {
  std::printf("\n--- GridHierarchy component access (%dx%d, sweeps=%d) ---\n", nx, ny, sweeps);

  GridGeometry<double, 2> geom(
    {0.0, 0.0},
    {1.0 / nx, 1.0 / ny},
    {nx, ny},
    0, 1
  );

  using Alloc = HostAllocator<double, no_tracking>;
  GridHierarchy<double, 2, Alloc> hier;
  int phi = hier.register_component("phi");
  int rhs = hier.register_component("rhs");
  hier.build(geom, 1);

  // initialise rhs to 1.0
  auto rhs_acc = hier.finest().accessor(rhs);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      rhs_acc(i, j) = 1.0;
    }
  }

  // sweep: phi[i,j] += 0.25 * rhs[i,j]  (simplified Jacobi-like update)
  {
    Timer t;
    for (int s = 0; s < sweeps; ++s) {
      auto phi_acc = hier.finest().accessor(phi);
      auto rhs_acc_inner = hier.finest().accessor(rhs);
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          phi_acc(i, j) += 0.25 * rhs_acc_inner(i, j);
        }
      }
      do_not_optimise(phi_acc(nx - 1, ny - 1));
    }
    std::printf("  FieldAccessor sweep:  %.3f ms\n", t.elapsed_ms());
  }

  // same sweep with raw pointer arithmetic (baseline)
  {
    auto& box = hier.finest_box();
    double* phi_raw = box.component_data(phi);
    double* rhs_raw = box.component_data(rhs);
    auto strides = box.strides();

    // reset phi
    std::memset(phi_raw, 0, box.total_cells() * sizeof(double));

    Timer t;
    for (int s = 0; s < sweeps; ++s) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          size_t idx = (i + 1) * strides[0] + (j + 1) * strides[1];
          phi_raw[idx] += 0.25 * rhs_raw[idx];
        }
      }
      do_not_optimise(phi_raw[(nx) * strides[0] + (ny) * strides[1]]);
    }
    std::printf("  raw pointer sweep:    %.3f ms\n", t.elapsed_ms());
  }
}

void run_allocator_microbenchmarks() {
  print_header("Allocator Micro-Benchmarks");

  alloc_throughput(1024, 100000);
  alloc_throughput(1024 * 1024, 1000);

  tracking_overhead(1024, 100000);

  sequential_sweep(1024 * 1024, 100);

  cellgrid_access(256, 256, 500);
  cellgrid_access(1024, 1024, 50);
}

} // namespace bm

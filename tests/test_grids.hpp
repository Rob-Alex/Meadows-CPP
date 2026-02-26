#pragma once
#include "test_harness.hpp"
#include "grid.hpp"
#include <string>
#include <cmath>

TestResults run_grids_tests() {
  TestResults results;
  std::printf("[grids]\n");

  // GridGeometry 2D 

  GridGeometry<double, 2> geom(
    {0.0, 0.0},
    {0.01, 0.02},
    {10, 20},
    0, 2
  );

  TEST_ASSERT_EQ(geom._n_interior[0], 10, "2D n_interior[0] == 10");
  TEST_ASSERT_EQ(geom._n_interior[1], 20, "2D n_interior[1] == 20");
  TEST_ASSERT_EQ(geom._nghosts, 2, "2D nghosts == 2");
  TEST_ASSERT_EQ(geom._level, 0, "2D level == 0");

  // total cells = (10+4) * (20+4) = 14 * 24 = 336
  TEST_ASSERT_EQ(geom.total_cells(), 336, "2D total_cells == 336");
  TEST_ASSERT_EQ(geom.total_interior_cells(), 200, "2D total_interior_cells == 200");

  // domain length
  TEST_ASSERT_NEAR(geom.get_domain_length(0), 0.1, 1e-12, "2D domain_length[0] == 0.1");
  TEST_ASSERT_NEAR(geom.get_domain_length(1), 0.4, 1e-12, "2D domain_length[1] == 0.4");

  // domain extents
  auto extents = geom.get_domain_extents();
  TEST_ASSERT_NEAR(extents[0], 0.1, 1e-12, "2D domain_extents[0] == 0.1");
  TEST_ASSERT_NEAR(extents[1], 0.4, 1e-12, "2D domain_extents[1] == 0.4");

  // n_with_ghosts
  auto nwg = geom.get_n_with_ghosts();
  TEST_ASSERT_EQ(nwg[0], 14, "2D n_with_ghosts[0] == 14");
  TEST_ASSERT_EQ(nwg[1], 24, "2D n_with_ghosts[1] == 24");

  // GridGeometry 3D 

  GridGeometry<double, 3> geom3d(
    {0.0, 0.0, 0.0},
    {0.01, 0.02, 0.03},
    {10, 20, 30},
    0, 1
  );

  TEST_ASSERT_EQ(geom3d._n_interior[0], 10, "3D n_interior[0] == 10");
  TEST_ASSERT_EQ(geom3d._n_interior[1], 20, "3D n_interior[1] == 20");
  TEST_ASSERT_EQ(geom3d._n_interior[2], 30, "3D n_interior[2] == 30");

  // total = (10+2)*(20+2)*(30+2) = 12*22*32 = 8448
  TEST_ASSERT_EQ(geom3d.total_cells(), 8448, "3D total_cells == 8448");
  TEST_ASSERT_EQ(geom3d.total_interior_cells(), 6000, "3D total_interior_cells == 6000");

  // GridGeometry 1D 

  GridGeometry<double, 1> geom1d({0.0}, {0.1}, {100}, 0, 2);
  TEST_ASSERT_EQ(geom1d.total_cells(), 104, "1D total_cells == 104");
  TEST_ASSERT_EQ(geom1d.total_interior_cells(), 100, "1D total_interior_cells == 100");
  TEST_ASSERT_NEAR(geom1d.get_domain_length(0), 10.0, 1e-12, "1D domain_length == 10.0");

  // CellGrid construction and topology 

  GridGeometry<double, 2> cg_geom({0.0, 0.0}, {0.1, 0.1}, {8, 8}, 0, 1);
  CellGrid<double, 2> grid(cg_geom);

  // dims include ghosts: 8+2 = 10 in each direction
  TEST_ASSERT_EQ(grid.dims()[0], 10, "CellGrid dims[0] == 10");
  TEST_ASSERT_EQ(grid.dims()[1], 10, "CellGrid dims[1] == 10");

  // total = 10*10 = 100
  TEST_ASSERT_EQ(grid.total_cells(), 100u, "CellGrid total_cells == 100");

  // strides: row-major, stride[0]=10, stride[1]=1
  TEST_ASSERT_EQ(grid.strides()[0], 10u, "CellGrid stride[0] == 10");
  TEST_ASSERT_EQ(grid.strides()[1], 1u, "CellGrid stride[1] == 1");

  // no components initially
  TEST_ASSERT_EQ(grid.num_components(), 0u, "CellGrid starts with 0 components");

  // Component registration 

  int phi_idx = grid.register_component("phi");
  TEST_ASSERT_EQ(phi_idx, 0, "First component index == 0");
  TEST_ASSERT_EQ(grid.num_components(), 1u, "1 component after registration");

  int rhs_idx = grid.register_component("rhs");
  TEST_ASSERT_EQ(rhs_idx, 1, "Second component index == 1");
  TEST_ASSERT_EQ(grid.num_components(), 2u, "2 components after registration");

  // duplicate registration returns same index
  int phi_dup = grid.register_component("phi");
  TEST_ASSERT_EQ(phi_dup, 0, "Duplicate registration returns same index");
  TEST_ASSERT_EQ(grid.num_components(), 2u, "No extra component from duplicate");

  // get_component_index
  TEST_ASSERT_EQ(grid.get_component_index("phi"), 0, "get_component_index(phi) == 0");
  TEST_ASSERT_EQ(grid.get_component_index("rhs"), 1, "get_component_index(rhs) == 1");

  // Component data is zero-initialised 

  const double* phi_data = grid.component_data(phi_idx);
  bool all_zero = true;
  for (size_t i = 0; i < grid.total_cells(); ++i) {
    if (phi_data[i] != 0.0) { all_zero = false; break; }
  }
  TEST_ASSERT(all_zero, "Component data is zero-initialised");

  // FieldAccessor read/write 

  auto phi_acc = grid.accessor(phi_idx);

  // write to interior cells
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      phi_acc(i, j) = (i + 1) * 10.0 + (j + 1);
    }
  }

  // read back
  TEST_ASSERT_NEAR(phi_acc(0, 0), 11.0, 1e-12, "accessor(0,0) == 11.0");
  TEST_ASSERT_NEAR(phi_acc(7, 7), 88.0, 1e-12, "accessor(7,7) == 88.0");
  TEST_ASSERT_NEAR(phi_acc(3, 5), 46.0, 1e-12, "accessor(3,5) == 46.0");

  // accessor by name
  auto phi_acc2 = grid.accessor("phi");
  TEST_ASSERT_NEAR(phi_acc2(0, 0), 11.0, 1e-12, "accessor by name matches");

  // Ghost cell access (negative indices) 

  // write to ghost cell at (-1, 0)
  phi_acc(-1, 0) = 99.0;
  TEST_ASSERT_NEAR(phi_acc(-1, 0), 99.0, 1e-12, "Ghost access (-1,0) write/read");

  // ghost cell at (0, -1)
  phi_acc(0, -1) = 77.0;
  TEST_ASSERT_NEAR(phi_acc(0, -1), 77.0, 1e-12, "Ghost access (0,-1) write/read");

  // Cell volume 

  TEST_ASSERT_NEAR(grid.cell_volume(), 0.01, 1e-12, "cell_volume == dx*dy = 0.01");

  // Cell centre positions 

  // cell (0,0) centre should be at origin + 0.5*spacing
  auto cc00 = grid.cell_centre(0, 0);
  TEST_ASSERT_NEAR(cc00[0], 0.05, 1e-12, "cell_centre(0,0)[0] == 0.05");
  TEST_ASSERT_NEAR(cc00[1], 0.05, 1e-12, "cell_centre(0,0)[1] == 0.05");

  // cell (7,7) centre
  auto cc77 = grid.cell_centre(7, 7);
  TEST_ASSERT_NEAR(cc77[0], 0.75, 1e-12, "cell_centre(7,7)[0] == 0.75");
  TEST_ASSERT_NEAR(cc77[1], 0.75, 1e-12, "cell_centre(7,7)[1] == 0.75");

  // refinement tests

  auto refine = grid.refine(2);
  TEST_ASSERT_EQ(refine._geom._n_interior[0], 16, "Refined n_interior[0] == 16");
  TEST_ASSERT_EQ(refine._geom._n_interior[1], 16, "Refined n_interior[1] == 16");
  TEST_ASSERT_NEAR(refine._geom._spacing[0], 0.05, 1e-12, "Refined n_interior[0] == 0.05"); 
  TEST_ASSERT_NEAR(refine._geom._spacing[1], 0.05, 1e-12, "Refined n_interior[1] == 0.05");
  TEST_ASSERT_EQ(refine._geom._level, 1, "Refined level == 1");
  TEST_ASSERT_EQ(refine._geom._nghosts, 1, "Refined nghosts preserved");

  // coarsen grid tests, this should only go to level 0 or throw a runtime error

  auto coarse = refine.coarsen(2);
  TEST_ASSERT_EQ(coarse._geom._n_interior[0], 8, "Coarsened n_interior[0] == 4");
  TEST_ASSERT_EQ(coarse._geom._n_interior[1], 8, "Coarsened n_interior[1] == 4");
  TEST_ASSERT_NEAR(coarse._geom._spacing[0], 0.1, 1e-12, "Coarsened spacing[0] == 0.2");
  TEST_ASSERT_NEAR(coarse._geom._spacing[1], 0.1, 1e-12, "Coarsened spacing[1] == 0.2");
  TEST_ASSERT_EQ(coarse._geom._level, 0, "Coarsened level == 0");
  TEST_ASSERT_EQ(coarse._geom._nghosts, 1, "Coarsened nghosts preserved");

  // coarsened grid should have correct dims: (8+2)*(8+2) = 100
  TEST_ASSERT_EQ(coarse.total_cells(), 100u, "Coarsened total_cells == 100");
  
  // Test that coarsening past level 0 throws a runtime error
  bool caught_exception = false;
  try {
    auto coarse_past_0 = coarse.coarsen(2);
  } catch (const std::runtime_error& e) {
    caught_exception = true;
  }
  TEST_ASSERT(caught_exception, "Coarsening past level 0 throws runtime_error");

  // Move semantics 

  GridGeometry<double, 2> mv_geom({0.0, 0.0}, {0.1, 0.1}, {4, 4}, 0, 1);
  CellGrid<double, 2> grid_a(mv_geom);
  grid_a.register_component("test");
  auto test_acc = grid_a.accessor(0);
  test_acc(0, 0) = 42.0;

  // move construct
  CellGrid<double, 2> grid_b(std::move(grid_a));
  TEST_ASSERT_EQ(grid_b.num_components(), 1u, "Move ctor preserves components");
  auto test_acc_b = grid_b.accessor(0);
  TEST_ASSERT_NEAR(test_acc_b(0, 0), 42.0, 1e-12, "Move ctor preserves data");

  // move assign
  CellGrid<double, 2> grid_c(mv_geom);
  grid_c = std::move(grid_b);
  TEST_ASSERT_EQ(grid_c.num_components(), 1u, "Move assign preserves components");
  auto test_acc_c = grid_c.accessor(0);
  TEST_ASSERT_NEAR(test_acc_c(0, 0), 42.0, 1e-12, "Move assign preserves data");

  // FaceGrid holds geometry 

  FaceGrid<double, 2> face_grid{geom};
  TEST_ASSERT_EQ(face_grid._geom.total_cells(), 336, "FaceGrid holds geometry");

  return results;
}

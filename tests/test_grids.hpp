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

  // Single-level GridHierarchy (replaces old CellGrid tests)

  {
    GridGeometry<double, 2> cg_geom({0.0, 0.0}, {0.1, 0.1}, {8, 8}, 0, 1);
    GridHierarchy<double, 2> hier;
    int phi_idx = hier.register_component("phi");
    int rhs_idx = hier.register_component("rhs");

    TEST_ASSERT_EQ(phi_idx, 0, "First component index == 0");
    TEST_ASSERT_EQ(rhs_idx, 1, "Second component index == 1");
    TEST_ASSERT_EQ(hier.num_components(), 2, "2 components after registration");

    // duplicate registration returns same index
    int phi_dup = hier.register_component("phi");
    TEST_ASSERT_EQ(phi_dup, 0, "Duplicate registration returns same index");
    TEST_ASSERT_EQ(hier.num_components(), 2, "No extra component from duplicate");

    // get_component_index
    TEST_ASSERT_EQ(hier.get_component_index("phi"), 0, "get_component_index(phi) == 0");
    TEST_ASSERT_EQ(hier.get_component_index("rhs"), 1, "get_component_index(rhs) == 1");

    hier.build(cg_geom, 1);

    auto& box = hier.finest_box();

    // dims include ghosts: 8+2 = 10 in each direction
    TEST_ASSERT_EQ(box.dims()[0], 10, "Box dims[0] == 10");
    TEST_ASSERT_EQ(box.dims()[1], 10, "Box dims[1] == 10");

    // total = 10*10 = 100
    TEST_ASSERT_EQ(box.total_cells(), 100u, "Box total_cells == 100");

    // strides: row-major, stride[0]=10, stride[1]=1
    TEST_ASSERT_EQ(box.strides()[0], 10u, "Box stride[0] == 10");
    TEST_ASSERT_EQ(box.strides()[1], 1u, "Box stride[1] == 1");

    // num components
    TEST_ASSERT_EQ(box.num_components(), 2, "Box has 2 components");

    // Component data is zero-initialised
    const double* phi_data = box.component_data(phi_idx);
    bool all_zero = true;
    for (size_t i = 0; i < box.total_cells(); ++i) {
      if (phi_data[i] != 0.0) { all_zero = false; break; }
    }
    TEST_ASSERT(all_zero, "Component data is zero-initialised");

    // FieldAccessor read/write
    auto phi_acc = hier.finest().accessor(phi_idx);

    for (int j = 0; j < 8; ++j) {
      for (int i = 0; i < 8; ++i) {
        phi_acc(i, j) = (i + 1) * 10.0 + (j + 1);
      }
    }

    TEST_ASSERT_NEAR(phi_acc(0, 0), 11.0, 1e-12, "accessor(0,0) == 11.0");
    TEST_ASSERT_NEAR(phi_acc(7, 7), 88.0, 1e-12, "accessor(7,7) == 88.0");
    TEST_ASSERT_NEAR(phi_acc(3, 5), 46.0, 1e-12, "accessor(3,5) == 46.0");

    // Ghost cell access (negative indices)
    phi_acc(-1, 0) = 99.0;
    TEST_ASSERT_NEAR(phi_acc(-1, 0), 99.0, 1e-12, "Ghost access (-1,0) write/read");

    phi_acc(0, -1) = 77.0;
    TEST_ASSERT_NEAR(phi_acc(0, -1), 77.0, 1e-12, "Ghost access (0,-1) write/read");

    // Cell volume
    TEST_ASSERT_NEAR(box.cell_volume(), 0.01, 1e-12, "cell_volume == dx*dy = 0.01");

    // Cell centre positions
    auto cc00 = box.cell_centre(0, 0);
    TEST_ASSERT_NEAR(cc00[0], 0.05, 1e-12, "cell_centre(0,0)[0] == 0.05");
    TEST_ASSERT_NEAR(cc00[1], 0.05, 1e-12, "cell_centre(0,0)[1] == 0.05");

    auto cc77 = box.cell_centre(7, 7);
    TEST_ASSERT_NEAR(cc77[0], 0.75, 1e-12, "cell_centre(7,7)[0] == 0.75");
    TEST_ASSERT_NEAR(cc77[1], 0.75, 1e-12, "cell_centre(7,7)[1] == 0.75");

    // Component name lookup
    TEST_ASSERT_EQ(hier.component_name(0), std::string("phi"), "component_name(0) == phi");
    TEST_ASSERT_EQ(hier.component_name(1), std::string("rhs"), "component_name(1) == rhs");
  }

  // Multi-level hierarchy (4 levels, verify geometry at each level)

  {
    GridGeometry<double, 2> finest_geom({0.0, 0.0}, {0.01, 0.01}, {64, 64}, 3, 1);
    GridHierarchy<double, 2> hier;
    int phi = hier.register_component("phi");
    [[maybe_unused]] int rhs = hier.register_component("rhs");
    [[maybe_unused]] int res = hier.register_component("residual");
    hier.build(finest_geom, 4);

    TEST_ASSERT_EQ(hier.num_levels(), 4, "4 levels in hierarchy");
    TEST_ASSERT_EQ(hier.finest_level(), 3, "finest_level == 3");
    TEST_ASSERT_EQ(hier.ref_ratio(), 2, "ref_ratio == 2");

    // Level 0 (coarsest): 8x8, spacing 0.08
    auto& lvl0_geom = hier.level(0).geometry();
    TEST_ASSERT_EQ(lvl0_geom._n_interior[0], 8, "Level 0 n_interior[0] == 8");
    TEST_ASSERT_EQ(lvl0_geom._n_interior[1], 8, "Level 0 n_interior[1] == 8");
    TEST_ASSERT_NEAR(lvl0_geom._spacing[0], 0.08, 1e-12, "Level 0 spacing[0] == 0.08");
    TEST_ASSERT_NEAR(lvl0_geom._spacing[1], 0.08, 1e-12, "Level 0 spacing[1] == 0.08");
    TEST_ASSERT_EQ(lvl0_geom._level, 0, "Level 0 _level == 0");

    // Level 1: 16x16, spacing 0.04
    auto& lvl1_geom = hier.level(1).geometry();
    TEST_ASSERT_EQ(lvl1_geom._n_interior[0], 16, "Level 1 n_interior[0] == 16");
    TEST_ASSERT_NEAR(lvl1_geom._spacing[0], 0.04, 1e-12, "Level 1 spacing[0] == 0.04");

    // Level 2: 32x32, spacing 0.02
    auto& lvl2_geom = hier.level(2).geometry();
    TEST_ASSERT_EQ(lvl2_geom._n_interior[0], 32, "Level 2 n_interior[0] == 32");
    TEST_ASSERT_NEAR(lvl2_geom._spacing[0], 0.02, 1e-12, "Level 2 spacing[0] == 0.02");

    // Level 3 (finest): 64x64, spacing 0.01
    auto& lvl3_geom = hier.level(3).geometry();
    TEST_ASSERT_EQ(lvl3_geom._n_interior[0], 64, "Level 3 n_interior[0] == 64");
    TEST_ASSERT_NEAR(lvl3_geom._spacing[0], 0.01, 1e-12, "Level 3 spacing[0] == 0.01");

    // All levels have same number of components
    for (int lvl = 0; lvl < 4; ++lvl) {
      TEST_ASSERT_EQ(hier.level(lvl).box(0).num_components(), 3,
        "All levels have 3 components");
    }

    // Component indices are consistent across levels
    auto phi_fine = hier.finest().accessor(phi);
    phi_fine(0, 0) = 42.0;
    TEST_ASSERT_NEAR(phi_fine(0, 0), 42.0, 1e-12, "Finest level accessor works");

    auto phi_coarse = hier.level(0).accessor(phi);
    phi_coarse(0, 0) = 99.0;
    TEST_ASSERT_NEAR(phi_coarse(0, 0), 99.0, 1e-12, "Coarsest level accessor works");

    // Domain extents are the same at all levels
    for (int lvl = 0; lvl < 4; ++lvl) {
      auto ext = hier.level(lvl).geometry().get_domain_extents();
      TEST_ASSERT_NEAR(ext[0], 0.64, 1e-12, "Domain extent[0] consistent across levels");
      TEST_ASSERT_NEAR(ext[1], 0.64, 1e-12, "Domain extent[1] consistent across levels");
    }
  }

  // Error: registration after build throws

  {
    GridGeometry<double, 2> g({0.0, 0.0}, {0.1, 0.1}, {4, 4}, 0, 1);
    GridHierarchy<double, 2> hier;
    hier.register_component("phi");
    hier.build(g, 1);

    bool caught = false;
    try {
      hier.register_component("new_comp");
    } catch (const std::runtime_error&) {
      caught = true;
    }
    TEST_ASSERT(caught, "Registration after build throws runtime_error");
  }

  // Error: access before build throws

  {
    GridHierarchy<double, 2> hier;
    hier.register_component("phi");

    bool caught = false;
    try {
      hier.level(0);
    } catch (const std::runtime_error&) {
      caught = true;
    }
    TEST_ASSERT(caught, "Access before build throws runtime_error");
  }

  // Error: non-divisible grid throws

  {
    GridGeometry<double, 2> g({0.0, 0.0}, {0.1, 0.1}, {5, 5}, 0, 1);
    GridHierarchy<double, 2> hier;
    hier.register_component("phi");

    bool caught = false;
    try {
      hier.build(g, 3);  // 5 not divisible by 2^2=4
    } catch (const std::runtime_error&) {
      caught = true;
    }
    TEST_ASSERT(caught, "Non-divisible grid throws runtime_error");
  }

  // Move semantics on hierarchy

  {
    GridGeometry<double, 2> mv_geom({0.0, 0.0}, {0.1, 0.1}, {4, 4}, 0, 1);
    GridHierarchy<double, 2> hier_a;
    hier_a.register_component("test");
    hier_a.build(mv_geom, 1);
    auto test_acc = hier_a.finest().accessor(0);
    test_acc(0, 0) = 42.0;

    // move construct
    GridHierarchy<double, 2> hier_b(std::move(hier_a));
    TEST_ASSERT_EQ(hier_b.num_levels(), 1, "Move ctor preserves levels");
    TEST_ASSERT_EQ(hier_b.num_components(), 1, "Move ctor preserves components");
    auto test_acc_b = hier_b.finest().accessor(0);
    TEST_ASSERT_NEAR(test_acc_b(0, 0), 42.0, 1e-12, "Move ctor preserves data");

    // move assign
    GridHierarchy<double, 2> hier_c;
    hier_c = std::move(hier_b);
    TEST_ASSERT_EQ(hier_c.num_levels(), 1, "Move assign preserves levels");
    auto test_acc_c = hier_c.finest().accessor(0);
    TEST_ASSERT_NEAR(test_acc_c(0, 0), 42.0, 1e-12, "Move assign preserves data");
  }

  // FaceGrid holds geometry

  FaceGrid<double, 2> face_grid{geom};
  TEST_ASSERT_EQ(face_grid._geom.total_cells(), 336, "FaceGrid holds geometry");

  return results;
}

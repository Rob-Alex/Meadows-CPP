#pragma once
#include "test_harness.hpp"
#include "boundary_conditions.hpp"

TestResults run_boundary_conditions_tests() {
  TestResults results;
  std::printf("[boundary_conditions]\n");

  // helpers
  // 1D geometry: 4 interior cells, nghosts=2, dx=0.1
  // interior values phi[i] = (i+1) * 10.0  →  {10, 20, 30, 40}
  //
  // 2D geometry: 4×4 interior cells, nghosts=2, dx=dy=0.1
  // interior values phi(i,j) = (i+1)*10.0 + (j+1)
  //   (0,0)=11  (1,0)=21  (2,0)=31  (3,0)=41
  //   (0,1)=12  (1,1)=22  (2,1)=32  (3,1)=42
  //   (0,2)=13  (1,2)=23  (2,2)=33  (3,2)=43
  //   (0,3)=14  (1,3)=24  (2,3)=34  (3,3)=44

  constexpr double tol = 1e-12;

  // 1D Dirichlet Tests:  

  {
    GridGeometry<double, 1> geom({0.0}, {0.1}, {4}, 0, 2);
    CellGrid<double, 1> grid(geom);
    int phi = grid.register_component("phi");
    auto acc = grid.accessor(phi);

    for (int i = 0; i < 4; ++i) acc(i) = (i + 1) * 10.0;

    BCRegistry<double, 1> bcs;
    bcs.set(0, 0, DirichletBC<double, 1>{{}, 5.0});
    bcs.set(0, 1, DirichletBC<double, 1>{{}, 5.0});
    bcs.fill_all(acc, geom);

    // low face: ghost[-(k+1)] = 2*5 - interior[k]
    TEST_ASSERT_NEAR(acc(-1), 2*5.0 - 10.0, tol, "1D Dirichlet low ghost[-1]");
    TEST_ASSERT_NEAR(acc(-2), 2*5.0 - 20.0, tol, "1D Dirichlet low ghost[-2]");

    // high face: ghost[ni+k] = 2*5 - interior[ni-1-k]
    TEST_ASSERT_NEAR(acc(4), 2*5.0 - 40.0, tol, "1D Dirichlet high ghost[ni]");
    TEST_ASSERT_NEAR(acc(5), 2*5.0 - 30.0, tol, "1D Dirichlet high ghost[ni+1]");

    // interior must be untouched
    TEST_ASSERT_NEAR(acc(0), 10.0, tol, "1D Dirichlet interior[0] untouched");
    TEST_ASSERT_NEAR(acc(3), 40.0, tol, "1D Dirichlet interior[3] untouched");
  }

  //1D Neumann zero-flux tests: 

  {
    GridGeometry<double, 1> geom({0.0}, {0.1}, {4}, 0, 2);
    CellGrid<double, 1> grid(geom);
    int phi = grid.register_component("phi");
    auto acc = grid.accessor(phi);

    for (int i = 0; i < 4; ++i) acc(i) = (i + 1) * 10.0;

    BCRegistry<double, 1> bcs;
    bcs.set(0, 0, NeumannBC<double, 1>{{}, 0.0});
    bcs.set(0, 1, NeumannBC<double, 1>{{}, 0.0});
    bcs.fill_all(acc, geom);

    // zero-flux: ghost = nearest interior cell (copy)
    TEST_ASSERT_NEAR(acc(-1), 10.0, tol, "1D Neumann zero-flux low ghost[-1]");
    TEST_ASSERT_NEAR(acc(-2), 10.0, tol, "1D Neumann zero-flux low ghost[-2]");
    TEST_ASSERT_NEAR(acc(4),  40.0, tol, "1D Neumann zero-flux high ghost[ni]");
    TEST_ASSERT_NEAR(acc(5),  40.0, tol, "1D Neumann zero-flux high ghost[ni+1]");
  }

  // 1D Neumann non-zero flux Tests:  

  {
    // flux=2.0, dx=0.1 → ghost[-(k+1)] = interior[0] + (k+1)*2.0*0.1
    GridGeometry<double, 1> geom({0.0}, {0.1}, {4}, 0, 2);
    CellGrid<double, 1> grid(geom);
    int phi = grid.register_component("phi");
    auto acc = grid.accessor(phi);

    for (int i = 0; i < 4; ++i) acc(i) = (i + 1) * 10.0;

    BCRegistry<double, 1> bcs;
    bcs.set(0, 0, NeumannBC<double, 1>{{}, 2.0});
    bcs.set(0, 1, NeumannBC<double, 1>{{}, 2.0});
    bcs.fill_all(acc, geom);

    TEST_ASSERT_NEAR(acc(-1), 10.0 + 1*2.0*0.1, tol, "1D Neumann nonzero low ghost[-1]");
    TEST_ASSERT_NEAR(acc(-2), 10.0 + 2*2.0*0.1, tol, "1D Neumann nonzero low ghost[-2]");
    TEST_ASSERT_NEAR(acc(4),  40.0 + 1*2.0*0.1, tol, "1D Neumann nonzero high ghost[ni]");
    TEST_ASSERT_NEAR(acc(5),  40.0 + 2*2.0*0.1, tol, "1D Neumann nonzero high ghost[ni+1]");
  }

  // 1D Periodic Tests:  

  {
    // low:  ghost[-1]=interior[ni-1]=40, ghost[-2]=interior[ni-2]=30
    // high: ghost[ni]=interior[0]=10,    ghost[ni+1]=interior[1]=20
    GridGeometry<double, 1> geom({0.0}, {0.1}, {4}, 0, 2);
    CellGrid<double, 1> grid(geom);
    int phi = grid.register_component("phi");
    auto acc = grid.accessor(phi);

    for (int i = 0; i < 4; ++i) acc(i) = (i + 1) * 10.0;

    BCRegistry<double, 1> bcs;
    bcs.set(0, 0, PeriodicBC<double, 1>{{}});
    bcs.set(0, 1, PeriodicBC<double, 1>{{}});
    bcs.fill_all(acc, geom);

    TEST_ASSERT_NEAR(acc(-1), 40.0, tol, "1D Periodic low ghost[-1] == interior[ni-1]");
    TEST_ASSERT_NEAR(acc(-2), 30.0, tol, "1D Periodic low ghost[-2] == interior[ni-2]");
    TEST_ASSERT_NEAR(acc(4),  10.0, tol, "1D Periodic high ghost[ni] == interior[0]");
    TEST_ASSERT_NEAR(acc(5),  20.0, tol, "1D Periodic high ghost[ni+1] == interior[1]");
  }

  // 2D Dirichlet x-faces (dim=0) Tests 
  // this tests the transverse loop iterating over dim=1 (j axis).

  {
    GridGeometry<double, 2> geom({0.0, 0.0}, {0.1, 0.1}, {4, 4}, 0, 2);
    CellGrid<double, 2> grid(geom);
    int phi = grid.register_component("phi");
    auto acc = grid.accessor(phi);

    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        acc(i, j) = (i + 1) * 10.0 + (j + 1);

    BCRegistry<double, 2> bcs;
    bcs.set(0, 0, DirichletBC<double, 2>{{}, 7.0});
    bcs.set(0, 1, DirichletBC<double, 2>{{}, 7.0});
    bcs.fill_all(acc, geom);

    // low face: ghost[-1,j] = 2*7 - phi(0,j) = 14 - (10 + j+1) = 3 - j
    TEST_ASSERT_NEAR(acc(-1, 0), 14.0 - 11.0, tol, "2D Dirichlet x-low ghost[-1,0]");
    TEST_ASSERT_NEAR(acc(-1, 1), 14.0 - 12.0, tol, "2D Dirichlet x-low ghost[-1,1]");
    TEST_ASSERT_NEAR(acc(-1, 2), 14.0 - 13.0, tol, "2D Dirichlet x-low ghost[-1,2]");
    TEST_ASSERT_NEAR(acc(-1, 3), 14.0 - 14.0, tol, "2D Dirichlet x-low ghost[-1,3]");

    // second ghost layer: ghost[-2,j] = 14 - phi(1,j) = 14 - (20 + j+1)
    TEST_ASSERT_NEAR(acc(-2, 0), 14.0 - 21.0, tol, "2D Dirichlet x-low ghost[-2,0]");
    TEST_ASSERT_NEAR(acc(-2, 1), 14.0 - 22.0, tol, "2D Dirichlet x-low ghost[-2,1]");

    // high face: ghost[4,j] = 14 - phi(3,j) = 14 - (40 + j+1)
    TEST_ASSERT_NEAR(acc(4, 0), 14.0 - 41.0, tol, "2D Dirichlet x-high ghost[4,0]");
    TEST_ASSERT_NEAR(acc(4, 1), 14.0 - 42.0, tol, "2D Dirichlet x-high ghost[4,1]");

    // interior untouched
    TEST_ASSERT_NEAR(acc(0, 0), 11.0, tol, "2D Dirichlet x interior(0,0) untouched");
    TEST_ASSERT_NEAR(acc(3, 3), 44.0, tol, "2D Dirichlet x interior(3,3) untouched");
  }

  // 2D Dirichlet y-faces (dim=1) tests 

  {
    GridGeometry<double, 2> geom({0.0, 0.0}, {0.1, 0.1}, {4, 4}, 0, 2);
    CellGrid<double, 2> grid(geom);
    int phi = grid.register_component("phi");
    auto acc = grid.accessor(phi);

    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        acc(i, j) = (i + 1) * 10.0 + (j + 1);

    BCRegistry<double, 2> bcs;
    bcs.set(1, 0, DirichletBC<double, 2>{{}, 7.0});
    bcs.set(1, 1, DirichletBC<double, 2>{{}, 7.0});
    bcs.fill_all(acc, geom);

    // low face: ghost[i,-1] = 14 - phi(i,0) = 14 - ((i+1)*10 + 1)
    TEST_ASSERT_NEAR(acc(0, -1), 14.0 - 11.0, tol, "2D Dirichlet y-low ghost[0,-1]");
    TEST_ASSERT_NEAR(acc(1, -1), 14.0 - 21.0, tol, "2D Dirichlet y-low ghost[1,-1]");
    TEST_ASSERT_NEAR(acc(2, -1), 14.0 - 31.0, tol, "2D Dirichlet y-low ghost[2,-1]");
    TEST_ASSERT_NEAR(acc(3, -1), 14.0 - 41.0, tol, "2D Dirichlet y-low ghost[3,-1]");

    // second ghost layer: ghost[i,-2] = 14 - phi(i,1)
    TEST_ASSERT_NEAR(acc(0, -2), 14.0 - 12.0, tol, "2D Dirichlet y-low ghost[0,-2]");
    TEST_ASSERT_NEAR(acc(1, -2), 14.0 - 22.0, tol, "2D Dirichlet y-low ghost[1,-2]");

    // high face: ghost[i,4] = 14 - phi(i,3) = 14 - ((i+1)*10 + 4)
    TEST_ASSERT_NEAR(acc(0, 4), 14.0 - 14.0, tol, "2D Dirichlet y-high ghost[0,4]");
    TEST_ASSERT_NEAR(acc(1, 4), 14.0 - 24.0, tol, "2D Dirichlet y-high ghost[1,4]");
    TEST_ASSERT_NEAR(acc(2, 4), 14.0 - 34.0, tol, "2D Dirichlet y-high ghost[2,4]");
    TEST_ASSERT_NEAR(acc(3, 4), 14.0 - 44.0, tol, "2D Dirichlet y-high ghost[3,4]");
  }

  // 2D Neumann zero-flux x-faces 

  {
    GridGeometry<double, 2> geom({0.0, 0.0}, {0.1, 0.1}, {4, 4}, 0, 2);
    CellGrid<double, 2> grid(geom);
    int phi = grid.register_component("phi");
    auto acc = grid.accessor(phi);

    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        acc(i, j) = (i + 1) * 10.0 + (j + 1);

    BCRegistry<double, 2> bcs;
    bcs.set(0, 0, NeumannBC<double, 2>{{}, 0.0});
    bcs.set(0, 1, NeumannBC<double, 2>{{}, 0.0});
    bcs.fill_all(acc, geom);

    // zero-flux: ghost copies nearest interior cell along x
    TEST_ASSERT_NEAR(acc(-1, 0), acc(0, 0), tol, "2D Neumann x-low ghost[-1,0] == interior[0,0]");
    TEST_ASSERT_NEAR(acc(-1, 2), acc(0, 2), tol, "2D Neumann x-low ghost[-1,2] == interior[0,2]");
    TEST_ASSERT_NEAR(acc(-2, 1), acc(0, 1), tol, "2D Neumann x-low ghost[-2,1] == interior[0,1]");
    TEST_ASSERT_NEAR(acc(4,  0), acc(3, 0), tol, "2D Neumann x-high ghost[4,0] == interior[3,0]");
    TEST_ASSERT_NEAR(acc(5,  3), acc(3, 3), tol, "2D Neumann x-high ghost[5,3] == interior[3,3]");
  }

  // 2D Periodic x-faces 

  {
    GridGeometry<double, 2> geom({0.0, 0.0}, {0.1, 0.1}, {4, 4}, 0, 2);
    CellGrid<double, 2> grid(geom);
    int phi = grid.register_component("phi");
    auto acc = grid.accessor(phi);

    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        acc(i, j) = (i + 1) * 10.0 + (j + 1);

    BCRegistry<double, 2> bcs;
    bcs.set(0, 0, PeriodicBC<double, 2>{{}});
    bcs.set(0, 1, PeriodicBC<double, 2>{{}});
    bcs.fill_all(acc, geom);

    // low ghosts wrap to high interior: ghost[-1,j] = interior[ni-1,j]
    TEST_ASSERT_NEAR(acc(-1, 0), acc(3, 0), tol, "2D Periodic x-low ghost[-1,0] == interior[3,0]");
    TEST_ASSERT_NEAR(acc(-1, 1), acc(3, 1), tol, "2D Periodic x-low ghost[-1,1] == interior[3,1]");
    TEST_ASSERT_NEAR(acc(-1, 3), acc(3, 3), tol, "2D Periodic x-low ghost[-1,3] == interior[3,3]");
    TEST_ASSERT_NEAR(acc(-2, 0), acc(2, 0), tol, "2D Periodic x-low ghost[-2,0] == interior[2,0]");

    // high ghosts wrap to low interior: ghost[ni,j] = interior[0,j]
    TEST_ASSERT_NEAR(acc(4, 0), acc(0, 0), tol, "2D Periodic x-high ghost[4,0] == interior[0,0]");
    TEST_ASSERT_NEAR(acc(4, 2), acc(0, 2), tol, "2D Periodic x-high ghost[4,2] == interior[0,2]");
    TEST_ASSERT_NEAR(acc(5, 1), acc(1, 1), tol, "2D Periodic x-high ghost[5,1] == interior[1,1]");
  }

  // 2D Periodic y-faces 

  {
    GridGeometry<double, 2> geom({0.0, 0.0}, {0.1, 0.1}, {4, 4}, 0, 2);
    CellGrid<double, 2> grid(geom);
    int phi = grid.register_component("phi");
    auto acc = grid.accessor(phi);

    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        acc(i, j) = (i + 1) * 10.0 + (j + 1);

    BCRegistry<double, 2> bcs;
    bcs.set(1, 0, PeriodicBC<double, 2>{{}});
    bcs.set(1, 1, PeriodicBC<double, 2>{{}});
    bcs.fill_all(acc, geom);

    // low ghosts wrap to high interior: ghost[i,-1] = interior[i,ni-1]
    TEST_ASSERT_NEAR(acc(0, -1), acc(0, 3), tol, "2D Periodic y-low ghost[0,-1] == interior[0,3]");
    TEST_ASSERT_NEAR(acc(1, -1), acc(1, 3), tol, "2D Periodic y-low ghost[1,-1] == interior[1,3]");
    TEST_ASSERT_NEAR(acc(0, -2), acc(0, 2), tol, "2D Periodic y-low ghost[0,-2] == interior[0,2]");

    // high ghosts wrap to low interior: ghost[i,ni] = interior[i,0]
    TEST_ASSERT_NEAR(acc(0, 4), acc(0, 0), tol, "2D Periodic y-high ghost[0,4] == interior[0,0]");
    TEST_ASSERT_NEAR(acc(2, 4), acc(2, 0), tol, "2D Periodic y-high ghost[2,4] == interior[2,0]");
    TEST_ASSERT_NEAR(acc(3, 5), acc(3, 1), tol, "2D Periodic y-high ghost[3,5] == interior[3,1]");
  }

  return results;
}

/* 
  Grids.hpp 
  Author: Robbie Alexander 
  write up in (blog www.rob-blog.co.uk),
  grid file to hold all information 
  


*/
#pragma once 
#include <array>

//contains all relevant information for grid structure where all data is
template<typename T, int Dims>
struct GridGeometry{
  std::array<T, Dims> _origin;        // physics origin of the grid [m] on each axis
  std::array<T, Dims> _spacing;       // distance between grid points [m] on each axis
  int _level;                         // refinement level for GMR/AMR etc.
                                      //
  std::array<int, Dims> _n_interior;  // interior cells (without ghosts) per axis
  int _nghosts;

  //ctor
  GridGeometry(std::array<T, Dims> origin,
    std::array<T, Dims> spacing,
    std::array<int, Dims> n_interior,
    int level, int nghosts)
  : _origin(origin), _spacing(spacing),
  _n_interior(n_interior), _level(level),  _nghosts(nghosts) {
    
  }

  // total number of cells including ghost cells for the entire domain 
  int total_cells() const {
    int total_cells = 1;
    for (const auto& n : _n_interior) {
      total_cells *= (n+(2*_nghosts));
    }
    return total_cells;
  }

  // total number of cells representing physical domain 
  int total_interior_cells() const {
    int total_cells = 1;
    for (const auto& n: _n_interior) {
      total_cells *= n;
    }
    return total_cells;
  }

  T get_domain_length(int dim) const { 
    return _spacing[dim] * _n_interior[dim]; 
  }

  std::array<int, Dims> get_domain_extents() const {
    std::array<int, Dims> extents;
    for (int i = 0; i < Dims; ++i) {
      extents[i] = get_domain_length(i);
    }
    return extents;
  }

  std::array<int, Dims> get_n_with_ghosts() const {
    std::array<int, Dims> ns; 
    for (int i = 0; i < Dims; ++i){
      ns[i] = _n_interior + (2*_nghosts); 
    }
    return ns;
  }
};

//contain cell-centred stuff 
template<typename T, int Dims>
struct CellGrid{
  GridGeometry<T, Dims> _geom;
};

//contain face-centred stuff
template<typename T, int Dims>
struct FaceGrid{

};

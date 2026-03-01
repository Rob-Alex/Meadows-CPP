/* 
  Grids.hpp 
  Author: Robbie Alexander 
  grid file to hold all information about geometry and how 
  each component (scalar or vector field) depending on if 
  cellCentre data or cell face data. 
*/
#pragma once 
#include <array>
#include <stdexcept>
#include <vector>
#include "memory.hpp"
#include "fields.hpp"
/* 
  GridGeometry: this is responsible for holding information about the physical/simulation domain exclusively, pure value type 
 */
template<typename T, int Dims>
struct GridGeometry{
  std::array<T, Dims> _origin;        // physics origin of the grid [m] on each axis
  std::array<T, Dims> _spacing;       // distance between grid points [m] on each axis
  std::array<int, Dims> _n_interior;  // interior cells (without ghosts) per axis
  int _level;                         // refinement level for GMR/AMR etc.
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

  std::array<T, Dims> get_domain_extents() const {
    std::array<T, Dims> extents;
    for (int i = 0; i < Dims; ++i) {
      extents[i] = get_domain_length(i);
    }
    return extents;
  }

  std::array<int, Dims> get_n_with_ghosts() const {
    std::array<int, Dims> ns; 
    for (int i = 0; i < Dims; ++i){
      ns[i] = _n_interior[i] + (2*_nghosts); 
    }
    return ns;
  }
};

//contain cell-centred stuff

template<typename T, int Dims, typename Alloc = HostAllocator<T, default_tracking<T>>>
class CellGrid{
public:
  GridGeometry<T, Dims> _geom;

private:
  // Component registry: maps component name to index (setup-time only)
  std::unordered_map<std::string, int> _comp_registry;

  // Per-component aligned buffers Structure of Array layout
  // one allocation per component (i.e the field)
  std::vector<T*> _comp_ptrs;

  // Cached topology (computed once at construction of the CellGrid)
  std::array<int, Dims> _dims;         // dimensions including ghosts
  std::array<size_t, Dims> _strides;   // row-major strides
  size_t _total;                        // total cells including ghosts
  Alloc _allocator;

public:
  // Ctor
  explicit CellGrid(GridGeometry<T, Dims> geom, Alloc allocator = Alloc())
    : _geom(geom), _allocator(allocator) {
    _dims = _geom.get_n_with_ghosts();
    _total = _geom.total_cells();

    // Compute row-major strides
    _strides[Dims - 1] = 1;
    for (int i = Dims - 2; i >= 0; --i) {
      _strides[i] = _strides[i + 1] * _dims[i + 1];
    }
  }

  // Move-only semantics (deleted copy, owns raw memory via Alloc)
  CellGrid(const CellGrid&) = delete;
  CellGrid& operator=(const CellGrid&) = delete;

  // Move constructor
  CellGrid(CellGrid&& other) noexcept
    : _geom(other._geom),
      _comp_registry(std::move(other._comp_registry)),
      _comp_ptrs(std::move(other._comp_ptrs)),
      _dims(other._dims),
      _strides(other._strides),
      _total(other._total),
      _allocator(std::move(other._allocator)) {
    // Clear other's pointers to prevent double-free
    other._comp_ptrs.clear();
  }

  // Move assignment
  CellGrid& operator=(CellGrid&& other) noexcept {
    if (this != &other) {
      // Deallocate existing components
      for (T* ptr : _comp_ptrs) {
        if (ptr) { 
          _allocator.deallocate(ptr); 
        };
      }

      _geom = other._geom;
      _comp_registry = std::move(other._comp_registry);
      _comp_ptrs = std::move(other._comp_ptrs);
      _dims = other._dims;
      _strides = other._strides;
      _total = other._total;
      _allocator = std::move(other._allocator);

      other._comp_ptrs.clear();
    }
    return *this;
  }

  // Destructor
  ~CellGrid() {
    for (T* ptr : _comp_ptrs) {
      if (ptr) {
        _allocator.deallocate(ptr);
      }
    }
  }

  // Register a new component, returns its index
  // Allocates aligned buffer for SoA storage
  int register_component(const std::string& name) {
    auto it = _comp_registry.find(name);
    if (it != _comp_registry.end()) {
      return it->second;  // Already registered
    }

    int idx = static_cast<int>(_comp_ptrs.size());
    _comp_registry[name] = idx;

    // Allocate aligned buffer for this component
    T* buffer = _allocator.allocate(_total);
    _comp_ptrs.push_back(buffer);

    // Initialize to zero
    std::fill(buffer, buffer + _total, T{0});

    return idx;
  }

  // Get raw pointer to component data for MPI/GPU transfer
  T* component_data(int comp) {
    return _comp_ptrs[comp];
  }

  const T* component_data(int comp) const {
    return _comp_ptrs[comp];
  }

  // Get component index by name
  int get_component_index(const std::string& name) const {
    auto it = _comp_registry.find(name);
    if (it == _comp_registry.end()) {
      throw std::runtime_error("Component not found: " + name);
    }
    return it->second;
  }

  // Get pre-offset accessor for zero-based indexing (0,0) → first interior cell
  FieldAccessor<T, Dims> accessor(int comp) {
    T* base = _comp_ptrs[comp];

    // Offset by nghosts in each direction so (0,0,...) maps to first interior cell
    size_t offset = 0;
    for (int d = 0; d < Dims; ++d) {
      offset += _geom._nghosts * _strides[d];
    }

    return FieldAccessor<T, Dims>(base + offset, _strides, _dims);
  }

  FieldAccessor<T, Dims> accessor(const std::string& name) {
    return accessor(get_component_index(name));
  }

  // Cell volume (constant for uniform grid)
  T cell_volume() const {
    T volume = _geom._spacing[0];
    for (int i = 1; i < Dims; ++i) {
      volume *= _geom._spacing[i];
    }
    return volume;
  }

  // Cell centre position given indices (0-based interior indexing)
  template<typename... Indices>
  std::array<T, Dims> cell_centre(Indices... indices) const {
    static_assert(sizeof...(indices) == Dims, "Number of indices must match Dims");

    std::array<int, Dims> idx_array = {static_cast<int>(indices)...};
    std::array<T, Dims> centre;

    for (int d = 0; d < Dims; ++d) {
      // Centre is at origin + (index + 0.5) * spacing
      centre[d] = _geom._origin[d] + ((idx_array[d] + T{0.5}) * _geom._spacing[d]);
    }

    return centre;
  }

  // Coarsen grid by given ratio, returns new CellGrid at next level above
CellGrid coarsen(int ratio) const {
    if (_geom._level == 0) {
      throw std::runtime_error("Cannot coarsen grid at level 0: would result in negative refinement level");
    }
    std::array<int, Dims> coarse_n_interior;
    for (int d = 0; d < Dims; ++d) {
      coarse_n_interior[d] = _geom._n_interior[d] / ratio;
    }

    std::array<T, Dims> coarse_spacing;
    for (int d = 0; d < Dims; ++d) {
      coarse_spacing[d] = _geom._spacing[d] * ratio;
    }

    GridGeometry<T, Dims> coarse_geom(
      _geom._origin,
      coarse_spacing,
      coarse_n_interior,
      _geom._level - 1,
      _geom._nghosts
    );

    return CellGrid(coarse_geom, _allocator);
  }

  // regine grid by a given ratio, returning a new CellGrid at level below
  CellGrid refine(int ratio) const {
    std::array<int, Dims> refined_n_interior;
    for (int d = 0; d < Dims; ++d){
      refined_n_interior[d] = _geom._n_interior[d] * ratio;
    }
    std::array<T, Dims> refined_spacing;
    for (int d = 0; d < Dims; ++d) {
      refined_spacing[d] = _geom._spacing[d] / ratio; 
    }

    GridGeometry<T, Dims> refined_geom(
      _geom._origin,
      refined_spacing,
      refined_n_interior,
      _geom._level + 1,
      _geom._nghosts
    );
    return CellGrid(refined_geom, _allocator);
  }

  // Number of registered components
  size_t num_components() const {
    return _comp_ptrs.size();
  }

  // Access topology info
  const std::array<int, Dims>& dims() const { return _dims; }
  const std::array<size_t, Dims>& strides() const { return _strides; }
  size_t total_cells() const { return _total; }
};

// why has god has forsaken me 
// to implement a face grid we first must invent the universe...
// each facegrid must store fluxes on each face of the cell, e.g. for phi_x, phi_y, phi_z
// therefore the strategy would be to implement something like cellgrid but for each direction 
// so its more like X_phi, X_E (different components stored within x-direction flux)
// x = 0, y = 1, z = 2 for the direction class for facegrid, each direction storage
// will still have multiple dimensions as it is face per cell 
template<typename T, int Dims, typename Alloc> 
struct Direction{
  using valueType = T;
  using  ptr_valueType = T*;
  //again mapping components to name, shouldnt cost too much in hot loop 
  std::unordered_map<std::string, int> _comp_registry;  
  std::vector<ptr_valueType> _comp_ptrs;
  std::array<int, Dims> _dims; 
  std::array<size_t, Dims> _strides;
  size_t _total;
  Alloc _allocator; 
  //ctors
  Direction() = default; 
  Direction(const GridGeometry<T, Dims>& geom, int direction, Alloc allocator = Alloc()) : _allocator(allocator){
    for(int d = 0; d< Dims; ++d) {
      int n = geom._n_interior[d] + (2 * geom._nghosts);
      if (d == direction) { 
        n += 1;
      }
      _dims[d] = n;
    }
    _strides[Dims - 1] = 1;
    for(int d = Dims - 2; d >= 0; --d) {
      _strides[d] = _strides[d + 1] * _dims[d + 1];
    }
    _total = 1;
    for (int d = 0; d < Dims; ++d) {
      _total *= _dims[d];
    }
  }
  ptr_valueType allocate_component() { 
    ptr_valueType buf = _allocator.allocate(_total);
    std::fill(buf, buf + _total, valueType{0}); 
    _comp_ptrs.push_back(buf);    
    return buf;
  }
  ~Direction(){
    for (ptr_valueType p : _comp_ptrs) {
      if (p) {
        _allocator.deallocate(p);
      }
    }
  }
  Direction(const Direction&) = delete;
  Direction& operator=(const Direction&) = delete; 
  Direction(Direction&& o) noexcept : _dims(o._dims), _strides(o._strides), _total(o._total), _comp_ptrs(std::move(o._comp_ptrs)), _allocator(std::move(o._allocator)){
    o._comp_ptrs.clear();
  }
  Direction& operator=(Direction&& o) noexcept { 
    if (this != &o) {
      for (ptr_valueType p : _comp_ptrs) {
        if (p) { _allocator.deallocate(p); }
      }
      _dims = o._dims; 
      _strides = o._strides; 
      _total = o._total;
      _comp_ptrs = std::move(o._comp_ptrs);
      _allocator = std::move(o._allocator);
      o._comp_ptrs.clear();
    }
    return *this;
  }
};

template<typename T, int Dims, typename Alloc = HostAllocator<T, default_tracking<T>>>
class FaceGrid{
public:
  GridGeometry<T, Dims> _geom;
private:
  std::array<Direction<T, Dims, Alloc>, Dims> _dirs;
  std::unordered_map<std::string, int> _comp_registry;
public: 
  explicit FaceGrid(GridGeometry<T, Dims> geom, Alloc allocator = Alloc()) : _geom(geom) {
    for(int d = 0; d < Dims; ++d) {
      _dirs[d] = Direction<T, Dims, Alloc>(geom, d, allocator);
    }
  }
  int register_component(const std::string& name) {
    auto it = _comp_registry.find(name);
    if (it != _comp_registry.end()) {
      return it->second;
    }

    int idx = static_cast<int>(_comp_registry.size());
    _comp_registry[name] = idx;

    for (int d = 0; d < Dims; ++d) {
      _dirs[d].allocate_component();
    }
    return idx;
  }

  // Raw pointer for direction + component (MPI/GPU transfer)
  T* component_data(int dir, int comp) {
    return _dirs[dir]._comp_ptrs[comp];
  }

  const T* component_data(int dir, int comp) const {
    return _dirs[dir]._comp_ptrs[comp];
  }

  // Accessor for a given direction and component
  // Pre-offset so (0,0) = first interior face
  FieldAccessor<T, Dims> accessor(int dir, int comp) {
    T* base = _dirs[dir]._comp_ptrs[comp];

    size_t offset = 0;
    for (int d = 0; d < Dims; ++d) {
      offset += _geom._nghosts * _dirs[dir]._strides[d];
    }

    return FieldAccessor<T, Dims>(
      base + offset, _dirs[dir]._strides, _dirs[dir]._dims);
  }

  // Face area constant for uniform grid, direction-dependent in general
  T face_area(int dir) const {
    T area = T{1};
    for (int d = 0; d < Dims; ++d) {
      if (d != dir){
        area *= _geom._spacing[d];
      }
    }
    return area;
  }

  // Move-only
  FaceGrid(const FaceGrid&) = delete;
  FaceGrid& operator=(const FaceGrid&) = delete;
  FaceGrid(FaceGrid&&) noexcept = default;
  FaceGrid& operator=(FaceGrid&&) noexcept = default;
};


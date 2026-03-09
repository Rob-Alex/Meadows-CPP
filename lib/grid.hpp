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
#include <cassert>
#include <unordered_map>
#include <string>
#include <algorithm>
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

// Box: fundamental data unit. Owns per-component SoA buffers for one rectangular region.
// If you do some digging youll see this has been refactored from CellGrid
// this use to hold its own component registry however then we lose 
// a singular source of truth as each level contains components
// new approach is box (CellGrid on each level) which there can be X many of
// Does NOT own a component registry — it receives n_comps at construction.

template<typename T, int Dims, typename Alloc = HostAllocator<T, default_tracking<T>>>
class Box {
public:
  GridGeometry<T, Dims> _geom;

private:
  std::vector<T*> _comp_ptrs;       // one buffer per component
  std::array<int, Dims> _dims;      // dims including ghosts
  std::array<size_t, Dims> _strides;
  size_t _total;
  Alloc _allocator;

public:
  // Construct with known component count (from hierarchy)
  Box(GridGeometry<T, Dims> geom, int n_comps, Alloc allocator = Alloc())
    : _geom(geom), _allocator(allocator) {
    _dims = _geom.get_n_with_ghosts();
    _total = _geom.total_cells();

    // Compute row-major strides
    _strides[Dims - 1] = 1;
    for (int i = Dims - 2; i >= 0; --i) {
      _strides[i] = _strides[i + 1] * _dims[i + 1];
    }

    // Allocate and zero-initialise all component buffers
    _comp_ptrs.reserve(n_comps);
    for (int c = 0; c < n_comps; ++c) {
      T* buffer = _allocator.allocate(_total);
      std::fill(buffer, buffer + _total, T{0});
      _comp_ptrs.push_back(buffer);
    }
  }

  // Move-only semantics
  Box(const Box&) = delete;
  Box& operator=(const Box&) = delete;

  Box(Box&& other) noexcept
    : _geom(other._geom),
      _comp_ptrs(std::move(other._comp_ptrs)),
      _dims(other._dims),
      _strides(other._strides),
      _total(other._total),
      _allocator(std::move(other._allocator)) {
    other._comp_ptrs.clear();
  }

  Box& operator=(Box&& other) noexcept {
    if (this != &other) {
      for (T* ptr : _comp_ptrs) {
        if (ptr) _allocator.deallocate(ptr);
      }

      _geom = other._geom;
      _comp_ptrs = std::move(other._comp_ptrs);
      _dims = other._dims;
      _strides = other._strides;
      _total = other._total;
      _allocator = std::move(other._allocator);

      other._comp_ptrs.clear();
    }
    return *this;
  }

  ~Box() {
    for (T* ptr : _comp_ptrs) {
      if (ptr) _allocator.deallocate(ptr);
    }
  }

  // Data access by integer component index
  T* component_data(int comp) { return _comp_ptrs[comp]; }
  const T* component_data(int comp) const { return _comp_ptrs[comp]; }

  // Get pre-offset accessor for zero-based indexing (0,0) → first interior cell
  FieldAccessor<T, Dims> accessor(int comp) {
    T* base = _comp_ptrs[comp];

    size_t offset = 0;
    for (int d = 0; d < Dims; ++d) {
      offset += _geom._nghosts * _strides[d];
    }

    return FieldAccessor<T, Dims>(base + offset, _strides, _dims);
  }

  // Topology
  const std::array<int, Dims>& dims() const { return _dims; }
  const std::array<size_t, Dims>& strides() const { return _strides; }
  size_t total_cells() const { return _total; }
  int num_components() const { return static_cast<int>(_comp_ptrs.size()); }

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
      centre[d] = _geom._origin[d] + ((idx_array[d] + T{0.5}) * _geom._spacing[d]);
    }

    return centre;
  }
};

// Level: groups boxes at the same refinement level.
// For geometric multigrid, this contains exactly 1 box 
// covering the whole domain. however 
// for AMR, contains N boxes covering tagged regions.
// that are at the same refinement level

template<typename T, int Dims, typename Alloc = HostAllocator<T, default_tracking<T>>>
class Level {
public:
  int _level_index;                      // 0 = coarsest (convention) 

private:
  GridGeometry<T, Dims> _geom;
  std::vector<Box<T, Dims, Alloc>> _boxes;

public:
  Level(GridGeometry<T, Dims> geom, int level_index, int n_comps, Alloc alloc = Alloc())
    : _level_index(level_index), _geom(geom) {
    // Create a single box covering the whole domain for GMG
    _boxes.emplace_back(geom, n_comps, alloc);
  }

  // Move-only (Box is move-only)
  Level(const Level&) = delete;
  Level& operator=(const Level&) = delete;
  Level(Level&&) noexcept = default;
  Level& operator=(Level&&) noexcept = default;

  // Box access
  Box<T, Dims, Alloc>& box(int i) { return _boxes[i]; }
  const Box<T, Dims, Alloc>& box(int i) const { return _boxes[i]; }
  int num_boxes() const { return static_cast<int>(_boxes.size()); }

  // Convenience for single-box levels (GMG) — forward to box(0)
  FieldAccessor<T, Dims> accessor(int comp) {
    assert(num_boxes() == 1);
    return _boxes[0].accessor(comp);
  }

  T* component_data(int comp) {
    assert(num_boxes() == 1);
    return _boxes[0].component_data(comp);
  }

  const T* component_data(int comp) const {
    assert(num_boxes() == 1);
    return _boxes[0].component_data(comp);
  }

  // Level geometry
  const GridGeometry<T, Dims>& geometry() const { return _geom; }
};

// GridHierarchy: this is the new GridCell class which is a top-level owner. 
// this is the new single component registry so there is one source of 
// truth. he is the truth and can handle it... 
// Register-then-build pattern to be able to allocate memory to 
// cellGrid/box structs .

template<typename T, int Dims, typename Alloc = HostAllocator<T, default_tracking<T>>>
class GridHierarchy {
private:
  // Component registry which can only be allocated at build
  std::unordered_map<std::string, int> _comp_registry;
  std::vector<std::string> _comp_names;  // index → name reverse lookup
  int _n_comps = 0;
  bool _built = false;

  // Hierarchy structure
  std::vector<Level<T, Dims, Alloc>> _levels;  // index 0 = coarsest
  int _ref_ratio = 2; //this is to ensure we dont end up with fractional 
                      //division of the domain which would blast everything
  Alloc _allocator;

public:
  GridHierarchy(Alloc alloc = Alloc()) : _allocator(alloc) {}

  // Move-only
  GridHierarchy(const GridHierarchy&) = delete;
  GridHierarchy& operator=(const GridHierarchy&) = delete;
  GridHierarchy(GridHierarchy&&) noexcept = default;
  GridHierarchy& operator=(GridHierarchy&&) noexcept = default;

  // Registration of components phase before build so no memory allocs in run
  int register_component(const std::string& name) {
    if (_built) {
      throw std::runtime_error("Cannot register components after build");
    }
    auto it = _comp_registry.find(name);
    if (it != _comp_registry.end()) {
      return it->second;  // Already registered
    }
    int idx = _n_comps++;
    _comp_registry[name] = idx;
    _comp_names.push_back(name);
    return idx;
  }

  int get_component_index(const std::string& name) const {
    auto it = _comp_registry.find(name);
    if (it == _comp_registry.end()) {
      throw std::runtime_error("Component not found: " + name);
    }
    return it->second;
  }

  int num_components() const { return _n_comps; }

  // Build function  
  // Takes the FINEST level geometry + number of coarsenings.
  // Level 0 = coarsest, level n_levels-1 = finest.
  void build(GridGeometry<T, Dims> finest_geom, int n_levels, int ref_ratio = 2) {
    if (_built) {
      throw std::runtime_error("Hierarchy already built");
    }
    if (n_levels < 1) {
      throw std::runtime_error("n_levels must be >= 1");
    }

    _ref_ratio = ref_ratio;

    // Validate: n_interior must be divisible by ref_ratio^(n_levels-1) 
    // as stated previously for the _ref_ratio
    int total_ratio = 1;
    for (int i = 0; i < n_levels - 1; ++i) total_ratio *= ref_ratio;

    for (int d = 0; d < Dims; ++d) {
      if (finest_geom._n_interior[d] % total_ratio != 0) {
        throw std::runtime_error(
          "n_interior[" + std::to_string(d) + "] = " +
          std::to_string(finest_geom._n_interior[d]) +
          " is not divisible by ref_ratio^(n_levels-1) = " +
          std::to_string(total_ratio));
      }
    }

    // Build levels from coarsest (0) to finest (n_levels-1)
    _levels.reserve(n_levels);

    for (int lvl = 0; lvl < n_levels; ++lvl) {
      // How many coarsenings from finest to this level
      int coarsenings = (n_levels - 1) - lvl;
      int ratio = 1;
      for (int i = 0; i < coarsenings; ++i) ratio *= ref_ratio;

      std::array<int, Dims> n_interior;
      std::array<T, Dims> spacing;
      std::array<T, Dims> origin;
      for (int d = 0; d < Dims; ++d) {
        n_interior[d] = finest_geom._n_interior[d] / ratio;
        spacing[d] = finest_geom._spacing[d] * ratio;
        // origin = position of first cell centre on this level
        // domain lower boundary: x_lo = finest_origin - dx_fine/2
        // coarse first cell centre: x_lo + dx_coarse/2
        origin[d] = finest_geom._origin[d]
                  - finest_geom._spacing[d] / T{2}
                  + spacing[d] / T{2};
      }

      GridGeometry<T, Dims> level_geom(
        origin,
        spacing,
        n_interior,
        lvl,
        finest_geom._nghosts
      );

      _levels.emplace_back(level_geom, lvl, _n_comps, _allocator);
    }

    _built = true;
  }

  // Access pattern for each of the components etc after init 

  Level<T, Dims, Alloc>& level(int lvl) {
    if (!_built) throw std::runtime_error("Hierarchy not built yet");
    return _levels[lvl];
  }

  const Level<T, Dims, Alloc>& level(int lvl) const {
    if (!_built) throw std::runtime_error("Hierarchy not built yet");
    return _levels[lvl];
  }

  int num_levels() const { return static_cast<int>(_levels.size()); }
  int finest_level() const { return num_levels() - 1; }
  int ref_ratio() const { return _ref_ratio; }

  // convenience: access finest level directly
  Level<T, Dims, Alloc>& finest() { return level(finest_level()); }
  const Level<T, Dims, Alloc>& finest() const { return level(finest_level()); }

  // convenience: finest level, first box (most common GMG case)
  Box<T, Dims, Alloc>& finest_box() { return finest().box(0); }
  const Box<T, Dims, Alloc>& finest_box() const { return finest().box(0); }

  // Component name lookup
  const std::string& component_name(int comp) const { return _comp_names[comp]; }

  bool is_built() const { return _built; }

};

// now this is for the fluxes, which currently are computed each time with
// no form of persistant storage, however I might change this
// why has god has forsaken me
// to implement a face grid we first must invent the universe...
// each facegrid must store fluxes on each face of the 
// cell, e.g. for phi_x, phi_y, phi_z
// therefore the strategy would be to implement something like 
// cellgrid but for each direction
// so its more like X_phi, X_E (different components stored within
// x-direction flux)
// x = 0, y = 1, z = 2 for the direction class for facegrid, 
// each direction storage
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

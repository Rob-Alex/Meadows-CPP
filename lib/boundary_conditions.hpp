/*
  Meadows-CPP
  Robbie Alexander 
  boundary_conditions.hpp
  Boundary Condition classes
*/

// CRTP based boundary condition class that can be evaulated at compile time.
// heterogeneous face storage via std::variant + std::visit (visitor pattern).
// current boundary conditions are only for the outer walls of the
// entire domain. howeverm embedded boundary conditions will be implemented through
// either masking (similar to afivo-streamer) or the cut-cell method later.
#pragma once
#include <variant>
#include "grid.hpp"

// compile-time transverse loop for assignement of the ghost cells and what not 
// recursively generates a nested loop over every dimension except FaceDim.
// At D == FaceDim the dimension is skipped (no loop body).
// At D == Dims the base case launches the kernel with the completed index array.
// for example in the case of Dims=2, FaceDim=0 this produces 
//   for (int i = 0; i < ni[1]; ++i) { idx[1] = i; f(idx); }
// — no integer division, no runtime branches, inner loop is vectorisable.

template<int Dims, int FaceDim, int D>
struct TransverseLoop {
  template<typename Func>
  static void run(const std::array<int, Dims>& ni,
                  std::array<int, Dims>& idx,
                  Func&& f) {
    if constexpr (D == FaceDim) {
      // skip the face-normal dimensionso no loop  
      TransverseLoop<Dims, FaceDim, D + 1>::run(ni, idx, f);
    } else {
      for (int i = 0; i < ni[D]; ++i) {
        idx[D] = i;
        TransverseLoop<Dims, FaceDim, D + 1>::run(ni, idx, f);
      }
    }
  }
};

// Base case: all dimensions visited
template<int Dims, int FaceDim>
struct TransverseLoop<Dims, FaceDim, Dims> {
  template<typename Func>
  static void run(const std::array<int, Dims>&,
                  std::array<int, Dims>& idx,
                  Func&& f) {
    f(idx);
  }
};

// CRTP base, this is the Non-Virtual Interface (compile time polymorphism) 
// I could have instead implemented this as a virtual function for 
// "dynamic polymorphism" however this would be slower because 
// of vtable look up and pointer indirection?
// fill_ghosts is the sole public entry point. face_dim and side arrive as
// runtime values from BCRegistry::fill_all. dispatch<FaceDim> converts them
// to compile-time template parameters, then calls fill_impl<FaceDim, Side>
// on the derived type — zero virtual dispatch, full inlining 
// which can only help performance I imagine.

template<class Derived, typename T, int Dims>
struct BoundaryConditionBase {

  // called by BCRegistry with runtime face_dim / side
  void fill_ghosts(FieldAccessor<T, Dims>& acc,
                   const std::array<int, Dims>& ni,
                   int nghosts, int face_dim, int side, T dx) const {
    dispatch<0>(acc, ni, nghosts, face_dim, side, dx);
  }

private:
  // walks FaceDim from 0 to Dims-1 until face_dim matches,
  // then calls fill_impl<FaceDim, Side> on the derived type.
  template<int FaceDim>
  void dispatch(FieldAccessor<T, Dims>& acc,
                const std::array<int, Dims>& ni,
                int nghosts, int face_dim, int side, T dx) const {
    if constexpr (FaceDim < Dims) {
      if (face_dim == FaceDim) {
        if (side == 0)
          static_cast<const Derived*>(this)
              ->template fill_impl<FaceDim, 0>(acc, ni, nghosts, dx);
        else
          static_cast<const Derived*>(this)
              ->template fill_impl<FaceDim, 1>(acc, ni, nghosts, dx);
      } else {
        dispatch<FaceDim + 1>(acc, ni, nghosts, face_dim, side, dx);
      }
    }
  }
};

// Dirichlet BC 
// phi (T) or templated value type at the face.
// ghost fill: ghost[-(k+1)] = 2*value - interior[k]  (low side)
//             ghost[ni+k]   = 2*value - interior[ni-1-k]  (high side)
// Linear antisymmetric reflection about the face preserves the face value
// to 2nd order for any cell-centred stencil that reads the ghost.

template<typename T, int Dims>
struct DirichletBC : BoundaryConditionBase<DirichletBC<T, Dims>, T, Dims> {
  T value;

  template<int FaceDim, int Side>
  void fill_impl(FieldAccessor<T, Dims>& acc,
                 const std::array<int, Dims>& ni,
                 int nghosts, T /*dx*/) const {
    for (int k = 0; k < nghosts; ++k) {
      std::array<int, Dims> idx{};
      idx[FaceDim]    = (Side == 0) ? -(k + 1)          : ni[FaceDim] + k;
      int mirror_n    = (Side == 0) ?   k                : ni[FaceDim] - 1 - k;

      TransverseLoop<Dims, FaceDim, 0>::run(ni, idx,
        [&](std::array<int, Dims>& ghost_idx) {
          std::array<int, Dims> mirror_idx = ghost_idx;
          mirror_idx[FaceDim] = mirror_n;
          acc[ghost_idx] = T{2} * value - acc[mirror_idx];
        });
    }
  }
};

// Neumann BC
// the rate of change of the value flux at the face (flux=0 for insulating walls).
// ghost fill: ghost[-(k+1)] = interior_nearest + (k+1)*flux*dx  (low side)
//             ghost[ni+k]   = interior_nearest + (k+1)*flux*dx  (high side)
// All ghost layers reference the same nearest interior cell, then extrapolate
// linearly outward at the prescribed gradient. Zero-flux collapses to a copy.

template<typename T, int Dims>
struct NeumannBC : BoundaryConditionBase<NeumannBC<T, Dims>, T, Dims> {
  T flux;

  template<int FaceDim, int Side>
  void fill_impl(FieldAccessor<T, Dims>& acc,
                 const std::array<int, Dims>& ni,
                 int nghosts, T dx) const {
    for (int k = 0; k < nghosts; ++k) {
      std::array<int, Dims> idx{};
      idx[FaceDim] = (Side == 0) ? -(k + 1)        : ni[FaceDim] + k;
      int ref_n    = (Side == 0) ?  0               : ni[FaceDim] - 1;

      TransverseLoop<Dims, FaceDim, 0>::run(ni, idx,
        [&](std::array<int, Dims>& ghost_idx) {
          std::array<int, Dims> ref_idx = ghost_idx;
          ref_idx[FaceDim] = ref_n;
          acc[ghost_idx] = acc[ref_idx] + T(k + 1) * flux * dx;
        });
    }
  }
};

// Periodic BC 
// ghost cells on one side are filled from the interior on the opposite side.
// ghost[-(k+1)] = interior[ni-1-k]  (low side wraps to high interior)
// ghost[ni+k]   = interior[k]       (high side wraps to low interior)

template<typename T, int Dims>
struct PeriodicBC : BoundaryConditionBase<PeriodicBC<T, Dims>, T, Dims> {

  template<int FaceDim, int Side>
  void fill_impl(FieldAccessor<T, Dims>& acc,
                 const std::array<int, Dims>& ni,
                 int nghosts, T /*dx*/) const {
    for (int k = 0; k < nghosts; ++k) {
      std::array<int, Dims> idx{};
      idx[FaceDim]  = (Side == 0) ? -(k + 1)            : ni[FaceDim] + k;
      int mirror_n  = (Side == 0) ?  ni[FaceDim] - 1 - k : k;

      TransverseLoop<Dims, FaceDim, 0>::run(ni, idx,
        [&](std::array<int, Dims>& ghost_idx) {
          std::array<int, Dims> mirror_idx = ghost_idx;
          mirror_idx[FaceDim] = mirror_n;
          acc[ghost_idx] = acc[mirror_idx];
        });
    }
  }
};

// BCVariant 
// apparently this is called the 
// "discriminated union" over all BC types. Stored inline — no heap allocation.
// std::visit generates a fully-inlined branch per type (no vtable).
// again similar to a visitor pattern in AST traversal

template<typename T, int Dims>
using BCVariant = std::variant<
  DirichletBC<T, Dims>,
  NeumannBC<T, Dims>,
  PeriodicBC<T, Dims>
>;

// BCRegistry 
// BCVariant per face of the domain boundary.
// index: faces[2*dim + side]  (dim: 0=X 1=Y 2=Z,  side: 0=low 1=high)
// fill_all drives the outer (dim, side) loop and dispatches via std::visit.

template<typename T, int Dims>
struct BCRegistry {
  std::array<BCVariant<T, Dims>, 2 * Dims> faces;

  void set(int dim, int side, BCVariant<T, Dims> bc) {
    faces[2 * dim + side] = std::move(bc);
  }

  void fill_all(FieldAccessor<T, Dims>& acc,
                const GridGeometry<T, Dims>& geom) const {
    for (int dim = 0; dim < Dims; ++dim) {
      for (int side = 0; side < 2; ++side) {
        std::visit([&](const auto& bc) {
          bc.fill_ghosts(acc,
                         geom._n_interior,
                         geom._nghosts,
                         dim, side,
                         geom._spacing[dim]);
        }, faces[2 * dim + side]);
      }
    }
  }

  // Create a homogeneous version of this BC registry.
  // For multigrid correction levels, the error at boundaries is zero,
  // so we need: Dirichlet value→0, Neumann flux→0, Periodic→unchanged.
  BCRegistry<T, Dims> make_homogeneous() const {
    BCRegistry<T, Dims> result;
    for (int dim = 0; dim < Dims; ++dim) {
      for (int side = 0; side < 2; ++side) {
        std::visit([&](const auto& bc) {
          using BCType = std::decay_t<decltype(bc)>;
          if constexpr (std::is_same_v<BCType, DirichletBC<T, Dims>>) {
            DirichletBC<T, Dims> h;
            h.value = T{0};
            result.set(dim, side, h);
          } else if constexpr (std::is_same_v<BCType, NeumannBC<T, Dims>>) {
            NeumannBC<T, Dims> h;
            h.flux = T{0};
            result.set(dim, side, h);
          } else if constexpr (std::is_same_v<BCType, PeriodicBC<T, Dims>>) {
            result.set(dim, side, bc);
          }
        }, faces[2 * dim + side]);
      }
    }
    return result;
  }
};

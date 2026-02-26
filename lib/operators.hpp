/* 
  Operators.hpp
  Author: Me 
  this contains all the relevant stencil operators, etc 
*/ 
#pragma once 
#include "grid.hpp"


//Laplacian
template <typename FluxPolicy, typename T, int Dims, typename Alloc = HostAllocator<T, default_tracking<T>>> 
struct laplacian{
  void apply(CellGrid<T, Dims, Alloc>& cellGrid, int phi, int rhs){ 
     
  }
};

template <typename T, int Dims, typename Alloc = HostAllocator<T, default_tracking<T>>>
struct l2_norm{

};

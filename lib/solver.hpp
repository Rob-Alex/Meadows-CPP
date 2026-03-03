/*
  Robbie Alexander 
  this uses CRTP (curiously recurring template pattern) for my own mental torture.  
 */ 
#pragma once
#include "grid.hpp"
#include "boundary_conditions.hpp"
#include "operators.hpp"

template<class Derived, typename T, int Dims> 
class SolverBase{
public:
    
  void init();
  void solve(){
    static_cast<const Derived*>(this)->solve_impl();
  }; 
protected:
  SolverBase() = default;
};

template<typename T, int Dims>
class EllipticSolverGMG : SolverBase<EllipticSolverGMG<T, Dims>, T, Dims> {
public:
  void solve_impl(){ 
  
  }
  void V_cycle(){
  }
  void full_multigrid_cycle(){
  };
};

/*
TODO: Implement hyerbolic solver
template<typename T, int Dims>
class HyperbolicSolver: SolverBase<HyperbolicSolver<T,Dims>,T, Dims> {

};
*/

/*
  Robbie Alexander 
  this uses CRTP (curiously recurring template pattern) for my own mental torture.  
 */ 
#pragma once

template<class Derived, typename T, int Dims> 
class SolverBase{
  
};

template<typename T, int Dims>
class EllipticSolver : SolverBase<EllipticSolver<T, Dims>, T, Dims> {

};

/*
TODO: Implement hyerbolic solver
template<typename T, int Dims>
class HyperbolicSolver: SolverBase<T, Dims> {

};
*/

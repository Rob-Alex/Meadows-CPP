/* 
 * Meadows-CPP
 * smoothers 
 * 
 */

template <class Derived, typename T, int Dims>
class SmootherBase {

};

template <typename T, int Dims>
class RBGS : SmootherBase<RBGS<T, Dims>, T, Dims> {

};

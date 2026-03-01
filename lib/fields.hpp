#pragma once
#include <array>
#include <cstddef>

template<typename T, int Dims>
struct FieldAccessor {
  T* _data;
  std::array<size_t, Dims> _strides;
  std::array<int, Dims> _dims;

  FieldAccessor(T* data, const std::array<size_t, Dims>& strides,
                const std::array<int, Dims>& dims)
    : _data(data), _strides(strides), _dims(dims) {}

  template<typename... Indices>
  T& operator()(Indices... indices) {
    static_assert(sizeof...(indices) == Dims, "Number of indices must match Dims");
    std::array<int, Dims> idx = {static_cast<int>(indices)...};
    size_t offset = 0;
    for (int d = 0; d < Dims; ++d) {
      offset += idx[d] * _strides[d];
    }
    return _data[offset];
  }

  template<typename... Indices>
  const T& operator()(Indices... indices) const {
    static_assert(sizeof...(indices) == Dims, "Number of indices must match Dims");
    std::array<int, Dims> idx = {static_cast<int>(indices)...};
    size_t offset = 0;
    for (int d = 0; d < Dims; ++d) {
      offset += idx[d] * _strides[d];
    }
    return _data[offset];
  }
 
  T& operator[](const std::array<int, Dims>& idx) {
    size_t offset = 0;
    for (int d = 0; d < Dims; ++d){
      offset += idx[d] * _strides[d];
    }
    return _data[offset];
  }
  
  const T& operator[](const std::array<int, Dims>& idx) const {
    size_t offset = 0;
    for (int d = 0; d < Dims; ++d){
      offset += idx[d] * _strides[d];
    }
    return _data[offset];
  }

};

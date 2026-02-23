/* 
 Memory Layer
 */ 
#pragma once

// this is able to allocate CPU/GPU memory=
template <typename T>
class Allocator{
  ~Allocator() = default;
  void virtual allocate(); 
};

template <typename T>
class HostAllocator : Allocator<T>{

};

template <typename T>
class DeviceAllocator : Allocator<T>{

};

template <typename T>
class UMVAllocator : Allocator <T>{


};

/* 
  Memory Layer
  Important for managing 
  Robbie Alexander 

│  Allocator<T>  : this is a policy based design for the class             │
│    ├── HostAllocator<T>          64-byte aligned, std::aligned_alloc     │
│    ├── DeviceAllocator<T>        cudaMalloc / hipMalloc                  │
│    ├── PinnedAllocator<T>        cudaMallocHost (async transfers)        │
│    └── UVMAllocator<T>           cudaMallocManaged (debug / prototyping) │
│                                                                          │
│  Design notes:                                                           │
│    • Per-component allocation (stable pointers, independent transfers)   │
│    • 64-byte alignment (AVX-512 / GPU cache-line ready)                  │
│    • Kokkos::View<T*, MemoryUnmanaged> wraps raw T* for dispatch         │
│    • Umpire-inspired ResourceManager pattern for multi-device tracking   │
│    • Move-only semantics on all owning types

 */ 
#pragma once
#include <new>
#include <cstddef>
#include <concepts>
#include <unordered_map>
#include <mutex>
// Layer 1: Memory Space Tags for each type of memory allocation that is done at compile time 

namespace memory_space {
  struct host {
    static constexpr const char* name = "host";
    static constexpr size_t default_alignment = 64;  // AVX-512 cache line
  };

  struct device {
    static constexpr const char* name = "device";
    static constexpr size_t default_alignment = 256;  // GPU requirement
  };

  struct pinned {
    static constexpr const char* name = "pinned";
    static constexpr size_t default_alignment = 64;
  };

  struct managed {
    static constexpr const char* name = "managed";
    static constexpr size_t default_alignment = 256;
  };
}

// Layer 2: Backend Tags 

namespace backend {
  struct cpu {
    static constexpr const char* name = "cpu";
  };

  struct cuda {
    static constexpr const char* name = "cuda";
  };
  // TODO: implement mental support
  /*
  struct metal {
    static constexpr const char* name = "metal";
  }; */
}

// 

// Allocator Concept (C++20)

template<typename A>
concept Allocator = requires {
  typename A::value_type;
  typename A::memory_space;
  typename A::backend;
  { A::alignment } -> std::convertible_to<size_t>;
  { A::is_host_accessible } -> std::convertible_to<bool>;
  { A::is_device_accessible } -> std::convertible_to<bool>;
} && requires(size_t n, typename A::value_type* ptr) {
  { A::allocate(n) } -> std::same_as<typename A::value_type*>;
  { A::deallocate(ptr) } -> std::same_as<void>;
};

// Host Allocator (CPU backend, always available)

struct no_tracking {
  template<typename T>
  static void on_allocate(T* ptr, size_t count) {}

  template<typename T>
  static void on_deallocate(T* ptr, size_t count) {}
};

struct track_allocations {
  struct allocation_info {
    size_t count;
    size_t bytes;
  };

  static std::unordered_map<void*, allocation_info>& get_allocations() {
    static std::unordered_map<void*, allocation_info> allocations;
    return allocations;
  }

  static std::mutex& get_mutex() {
    static std::mutex mtx;
    return mtx;
  }

  template<typename T>
  static void on_allocate(T* ptr, size_t count) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(get_mutex());
    get_allocations()[ptr] = {count, count * sizeof(T)};
  }

  template<typename T>
  static void on_deallocate(T* ptr, size_t count) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(get_mutex());
    get_allocations().erase(ptr);
  }

  static size_t get_active_count() {
    std::lock_guard<std::mutex> lock(get_mutex());
    return get_allocations().size();
  }

  static size_t get_total_bytes() {
    std::lock_guard<std::mutex> lock(get_mutex());
    size_t total = 0;
    for (const auto& [ptr, info] : get_allocations()) {
      total += info.bytes;
    }
    return total;
  }
};

#ifdef DEBUG_BUILD
template<typename T>
using default_tracking = track_allocations;
#else
template<typename T>
using default_tracking = no_tracking;
#endif

template<typename T, class Track = track_allocations>
struct HostAllocator {
  using value_type = T;
  using memory_space = memory_space::host;
  using backend = backend::cpu;

  static constexpr size_t alignment = 64;  // AVX-512 / cache line
  static constexpr bool is_host_accessible = true;
  static constexpr bool is_device_accessible = false;

  static T* allocate(size_t count) {
    if (count == 0) {
      return nullptr;
    }
    size_t bytes = count * sizeof(T);
    // std::aligned_alloc requires bytes to be a multiple of alignment
    size_t aligned_bytes = (bytes + alignment - 1) & ~(alignment - 1);
    void* ptr = std::aligned_alloc(alignment, aligned_bytes);

    if (!ptr) {
      throw std::bad_alloc();
    }

    T* typed_ptr = static_cast<T*>(ptr);
    Track::template on_allocate<T>(typed_ptr, count);
    return typed_ptr;
  }

  static void deallocate(T* ptr) {
    if (ptr) {
      Track::template on_deallocate<T>(ptr, 0);
      std::free(ptr);
    }
  }
};
// Default Allocator Aliases for convenience
template<typename T>
using default_host_allocator = HostAllocator<T>;

// TODO: Change this default_device_allocator to use DeviceAlloctot<T> 
template<typename T>
using default_device_allocator = HostAllocator<T>;  // No GPU yet

// TODO: Figure out how I want to work with pinned/pageable memory 
template<typename T>
using default_pinned_allocator = HostAllocator<T>;  // No GPU yet

template<typename T>
using default_umv_device_allocator = HostAllocator<T>; // no UMV yet 

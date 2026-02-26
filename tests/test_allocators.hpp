#pragma once
#include "test_harness.hpp"
#include "memory.hpp"

TestResults run_allocators_tests() {
  TestResults results;
  std::printf("[allocators]\n");

  using TrackedHostAllocator = HostAllocator<double, track_allocations>;

  // Clear any existing allocations from previous tests
  auto& allocations = track_allocations::get_allocations();
  allocations.clear();

  // Test 1: Basic allocation
  {
    double* ptr = TrackedHostAllocator::allocate(10);
    TEST_ASSERT(ptr != nullptr, "Basic allocation");
    TEST_ASSERT_EQ(track_allocations::get_active_count(), 1, "Active count after allocation");
    TEST_ASSERT_EQ(track_allocations::get_total_bytes(), 10 * sizeof(double), "Total bytes after allocation");
    TrackedHostAllocator::deallocate(ptr);
  }

  // Test 2: Deallocation removes tracking
  {
    double* ptr = TrackedHostAllocator::allocate(5);
    TrackedHostAllocator::deallocate(ptr);
    TEST_ASSERT_EQ(track_allocations::get_active_count(), 0, "Active count after deallocation");
    TEST_ASSERT_EQ(track_allocations::get_total_bytes(), 0, "Total bytes after deallocation");
  }

  // Test 3: Multiple allocations tracking
  {
    double* ptr1 = TrackedHostAllocator::allocate(10);
    double* ptr2 = TrackedHostAllocator::allocate(20);
    double* ptr3 = TrackedHostAllocator::allocate(30);

    TEST_ASSERT_EQ(track_allocations::get_active_count(), 3, "Active count with multiple allocations");
    size_t expected_bytes = (10 + 20 + 30) * sizeof(double);
    TEST_ASSERT_EQ(track_allocations::get_total_bytes(), expected_bytes, "Total bytes with multiple allocations");

    TrackedHostAllocator::deallocate(ptr2);
    TEST_ASSERT_EQ(track_allocations::get_active_count(), 2, "Active count after partial deallocation");

    TrackedHostAllocator::deallocate(ptr1);
    TrackedHostAllocator::deallocate(ptr3);
    TEST_ASSERT_EQ(track_allocations::get_active_count(), 0, "Active count after full deallocation");
  }

  // Test 4: Zero count allocation
  {
    double* ptr = TrackedHostAllocator::allocate(0);
    TEST_ASSERT(ptr == nullptr, "Zero count allocation returns nullptr");
    TEST_ASSERT_EQ(track_allocations::get_active_count(), 0, "No tracking for zero count allocation");
  }

  // Test 5: Alignment verification (64-byte aligned)
  {
    double* ptr = TrackedHostAllocator::allocate(100);
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    TEST_ASSERT(addr % 64 == 0, "Allocation is 64-byte aligned");
    TrackedHostAllocator::deallocate(ptr);
  }

  // Test 6: Different type allocations
  {
    using TrackedFloatAllocator = HostAllocator<float, track_allocations>;
    using TrackedIntAllocator = HostAllocator<int, track_allocations>;

    float* fptr = TrackedFloatAllocator::allocate(50);
    int* iptr = TrackedIntAllocator::allocate(25);

    TEST_ASSERT_EQ(track_allocations::get_active_count(), 2, "Multiple type allocations tracked");
    size_t expected = 50 * sizeof(float) + 25 * sizeof(int);
    TEST_ASSERT_EQ(track_allocations::get_total_bytes(), expected, "Mixed type allocation bytes");

    TrackedFloatAllocator::deallocate(fptr);
    TrackedIntAllocator::deallocate(iptr);
  }

  // Test 7: Nullptr deallocation is safe
  {
    TrackedHostAllocator::deallocate(nullptr);
    TEST_ASSERT(true, "Nullptr deallocation is safe");
  }

  return results;
}

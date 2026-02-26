#pragma once
#include <cstdio>
#include <cmath>

struct TestResults {
  int passed = 0;
  int failed = 0;

  void merge(const TestResults& other) {
    passed += other.passed;
    failed += other.failed;
  }
};

#define TEST_ASSERT(expr, name)                                         \
  do {                                                                  \
    if (expr) {                                                         \
      std::printf("  PASS: %s\n", name);                                \
      results.passed++;                                                 \
    } else {                                                            \
      std::printf("  FAIL: %s\n", name);                                \
      results.failed++;                                                 \
    }                                                                   \
  } while (0)

#define TEST_ASSERT_EQ(a, b, name)  TEST_ASSERT((a) == (b), name)

#define TEST_ASSERT_NEAR(a, b, tol, name)                              \
  TEST_ASSERT(std::fabs((a) - (b)) < (tol), name)

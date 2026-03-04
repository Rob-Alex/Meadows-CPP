#pragma once
#include "test_harness.hpp"

// Exporter tests require HDF5 — only compiled when HDF5 is available.
// The test binary is built without HDF5 by default.
// To test the exporter, use: make test-exporter (builds with HDF5 flags).

TestResults run_exporter_tests() {
  TestResults results;
  std::printf("[exporter]\n");

  // TODO: add exporter tests (requires HDF5 linkage)

  return results;
}

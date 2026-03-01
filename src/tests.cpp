#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

#include "test_grids.hpp"
#include "test_allocators.hpp"
#include "test_fields.hpp"
#include "test_operators.hpp"
#include "test_elliptic.hpp"
#include "test_lookup_tables.hpp"
#include "test_exporter.hpp"
#include "test_boundary_conditions.hpp"

struct TestSuite {
  const char* name;
  TestResults (*run)();
};

static const TestSuite suites[] = {
  {"grids",               run_grids_tests},
  {"allocators",          run_allocators_tests},
  {"fields",              run_fields_tests},
  {"operators",           run_operators_tests},
  {"elliptic",            run_elliptic_tests},
  {"lookup_tables",       run_lookup_tables_tests},
  {"exporter",            run_exporter_tests},
  {"boundary_conditions", run_boundary_conditions_tests},
};

static constexpr int n_suites = sizeof(suites) / sizeof(suites[0]);

static void print_usage() {
  std::printf("Usage: meadows-tests [suite ...]\n");
  std::printf("       meadows-tests all\n\n");
  std::printf("Available suites:\n");
  for (int i = 0; i < n_suites; ++i) {
    std::printf("  %s\n", suites[i].name);
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    return 1;
  }

  bool run_all = (argc == 2 && std::strcmp(argv[1], "all") == 0);

  std::vector<const TestSuite*> to_run;

  if (run_all) {
    for (int i = 0; i < n_suites; ++i) {
      to_run.push_back(&suites[i]);
    }
  } else {
    for (int a = 1; a < argc; ++a) {
      bool found = false;
      for (int i = 0; i < n_suites; ++i) {
        if (std::strcmp(argv[a], suites[i].name) == 0) {
          to_run.push_back(&suites[i]);
          found = true;
          break;
        }
      }
      if (!found) {
        std::printf("Unknown test suite: %s\n\n", argv[a]);
        print_usage();
        return 1;
      }
    }
  }

  TestResults total;
  for (const auto* suite : to_run) {
    TestResults r = suite->run();
    total.merge(r);
    std::printf("\n");
  }

  std::printf("========================================\n");
  std::printf("Total: %d passed, %d failed\n", total.passed, total.failed);
  std::printf("========================================\n");

  return total.failed > 0 ? 1 : 0;
}

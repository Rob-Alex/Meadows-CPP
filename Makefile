CXX      := clang++
CXXFLAGS := -std=c++20 -Wall -Wextra -O3

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Darwin)
  LLVM_PREFIX := $(shell brew --prefix llvm 2>/dev/null)
  LDFLAGS     := -L$(LLVM_PREFIX)/lib/c++ -Wl,-rpath,$(LLVM_PREFIX)/lib/c++
else
  LDFLAGS     :=
endif

ifdef DEBUG
  CXXFLAGS += -DDEBUG_BUILD -g -O0
endif
INCLUDES := -Ilib -Itests -Ibenchmarks

SRC_DIR   := src
BUILD_DIR := build

TARGET       := $(BUILD_DIR)/meadows
TEST_TARGET  := $(BUILD_DIR)/meadows-tests
BENCH_TARGET := $(BUILD_DIR)/meadows-bench

SUITES := grids allocators fields operators elliptic lookup_tables exporter boundary_conditions

all: $(TARGET)

# build and run all tests
test: $(TEST_TARGET)
	./$(TEST_TARGET) all

# build only
test-build: $(TEST_TARGET)

# per-suite convenience targets: make test-grids, make test-allocators, etc.
$(addprefix test-,$(SUITES)): test-%: $(TEST_TARGET)
	./$(TEST_TARGET) $*

$(TARGET): $(BUILD_DIR)/main.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(TEST_TARGET): $(BUILD_DIR)/tests.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(BENCH_TARGET): $(BUILD_DIR)/benchmarks.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

bench: $(BENCH_TARGET)
	./$(BENCH_TARGET)

bench-build: $(BENCH_TARGET)

.PHONY: all test test-build bench bench-build clean $(addprefix test-,$(SUITES))

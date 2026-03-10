CXX      := clang++
CXXFLAGS := -std=c++20 -Wall -Wextra -O3

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Darwin)
  LLVM_PREFIX  := $(shell brew --prefix llvm 2>/dev/null)
  LDFLAGS      := -L$(LLVM_PREFIX)/lib/c++ -Wl,-rpath,$(LLVM_PREFIX)/lib/c++
  HDF5_PREFIX  := $(shell brew --prefix hdf5 2>/dev/null)
  HDF5_CFLAGS  := -I$(HDF5_PREFIX)/include
  HDF5_LDFLAGS := -L$(HDF5_PREFIX)/lib -lhdf5_cpp -lhdf5
else
  LDFLAGS      :=
  HDF5_CFLAGS  := $(shell pkg-config --cflags hdf5 2>/dev/null)
  HDF5_LDFLAGS := $(shell pkg-config --libs hdf5_cpp hdf5 2>/dev/null || echo "-lhdf5_cpp -lhdf5")
endif

ifdef DEBUG
  CXXFLAGS += -DDEBUG_BUILD -g -O0
endif

# OpenMP (opt-in via OMP=1, e.g. `make all OMP=1`)
ifdef OMP
  ifeq ($(UNAME_S), Darwin)
    OMP_LLVM_PREFIX := $(shell brew --prefix llvm 2>/dev/null)
    CXX      := $(OMP_LLVM_PREFIX)/bin/clang++
    CXXFLAGS += -fopenmp
    LDFLAGS  += -L$(OMP_LLVM_PREFIX)/lib -lomp
  else
    CXXFLAGS += -fopenmp
    LDFLAGS  += -fopenmp
  endif
endif
INCLUDES := -Ilib -Itests -Ibenchmarks

SRC_DIR   := src
BUILD_DIR := build

TARGET       := $(BUILD_DIR)/meadows
BENCH_TARGET := $(BUILD_DIR)/benchmark

all: $(TARGET)

# build and run all tests
test: $(TEST_TARGET)
	./$(TEST_TARGET) all

# build only
test-build: $(TEST_TARGET)

# per-suite convenience targets: make test-grids, make test-allocators, etc.
$(addprefix test-,$(SUITES)): test-%: $(TEST_TARGET)
	./$(TEST_TARGET) $*

# main binary (links HDF5)
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(HDF5_CFLAGS) -c $< -o $@

$(TARGET): $(BUILD_DIR)/main.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(HDF5_LDFLAGS) $^ -o $@

# benchmark binary (no HDF5 dependency)
$(BUILD_DIR)/benchmark_main.o: $(SRC_DIR)/benchmark_main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BENCH_TARGET): $(BUILD_DIR)/benchmark_main.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

benchmark: $(BENCH_TARGET)
	./$(BENCH_TARGET) $(BENCH_MODE)

benchmark-build: $(BENCH_TARGET)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

run: $(TARGET)
	./$(TARGET)

.PHONY: all build clean run benchmark benchmark-build $(addprefix test-,$(SUITES))

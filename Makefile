CXX      := clang++
CXXFLAGS := -std=c++20 -Wall -Wextra -O2
INCLUDES := -Ilib

SRC_DIR   := src
BUILD_DIR := build

TARGET      := $(BUILD_DIR)/meadows
TEST_TARGET := $(BUILD_DIR)/meadows-tests

all: $(TARGET)

test: $(TEST_TARGET)

$(TARGET): $(BUILD_DIR)/main.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(TEST_TARGET): $(BUILD_DIR)/tests.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all test clean

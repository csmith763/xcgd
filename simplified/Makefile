
A2D_INCLUDE_DIR := ${HOME}/git/a2d/include

PYBIND11_CFLAGS := $(shell python3 -m pybind11 --includes)
PYTHON_EXT_SUFFIX := $(shell python3-config --extension-suffix)

MODULE := xcgd
SRC := xcgd.cpp

CXXFLAGS := -O3 -std=c++17 -fPIC $(PYBIND11_CFLAGS)

ifeq ($(shell uname),Darwin)
  LDFLAGS := -bundle -undefined dynamic_lookup -framework Accelerate
else
  LDFLAGS := -shared
endif


TARGET := $(MODULE)$(PYTHON_EXT_SUFFIX)

.PHONY: all clean rebuild

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) -I$(A2D_INCLUDE_DIR) $(CXXFLAGS) $(LDFLAGS) $< -o $@

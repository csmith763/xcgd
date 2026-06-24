
A2D_INCLUDE_DIR := ${HOME}/git/a2d/include
ALGOIM_INCLUDE_DIR := ${HOME}/git/algoim/algoim

PYBIND11_CFLAGS := $(shell python3 -m pybind11 --includes)
PYTHON_EXT_SUFFIX := $(shell python3-config --extension-suffix)

MODULE := xcgd
SRC := xcgd.cpp

CXXFLAGS := -g -std=c++17 -fPIC $(PYBIND11_CFLAGS)

ifeq ($(shell uname),Darwin)
  LDFLAGS := -bundle -undefined dynamic_lookup -framework Accelerate
else
  LDFLAGS := -shared
endif

INCLUDES := -I$(A2D_INCLUDE_DIR) -I$(ALGOIM_INCLUDE_DIR)

TARGET := $(MODULE)$(PYTHON_EXT_SUFFIX)

.PHONY: all clean rebuild

all: $(TARGET)

$(TARGET): force
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(LDFLAGS) $(SRC) -o $@

force:
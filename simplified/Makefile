
A2D_INCLUDE_DIR := ${HOME}/git/a2d/include
ALGOIM_INCLUDE_DIR := ${HOME}/git/algoim/algoim

PYBIND11_CFLAGS := $(shell python -m pybind11 --includes)
PYTHON_EXT_SUFFIX := $(shell python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

MODULE := xcgd
SRC := xcgd.cpp

CXXFLAGS_RELEASE := -O3 -std=c++17 -fPIC $(PYBIND11_CFLAGS)
CXXFLAGS_DEBUG := -g -std=c++17 -fPIC $(PYBIND11_CFLAGS)

ifeq ($(shell uname),Darwin)
  LDFLAGS := -bundle -undefined dynamic_lookup -framework Accelerate
else
  LDFLAGS := -shared
endif

INCLUDES := -I$(A2D_INCLUDE_DIR) -I$(ALGOIM_INCLUDE_DIR) -Ielement -Iphysics -Imesh -Iquadtree

TARGET := $(MODULE)$(PYTHON_EXT_SUFFIX)

.PHONY: all debug clean rebuild force

all: CXXFLAGS := $(CXXFLAGS_RELEASE)
all: $(TARGET)

debug: CXXFLAGS := $(CXXFLAGS_DEBUG)
debug: $(TARGET)

$(TARGET): force
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(SRC) $(LDFLAGS) -o $@

force:

clean:
	rm -f $(MODULE)*.so $(MODULE)*.dylib

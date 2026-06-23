#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "assembler.h"
#include "mesh_base.h"
#include "physics.h"

namespace py = pybind11;

template <typename T, class Physics>
void bind_assembler(py::module_& m, const std::string& name) {
  py::class_<xcgd::MeshAssembler<T, Physics>, xcgd::MeshAssemblerBase<T>,
             std::shared_ptr<xcgd::MeshAssembler<T, Physics>>>(m, name.c_str())
      .def(py::init<std::shared_ptr<xcgd::MeshBase<T>>, const Physics&>(),
           py::arg("mesh"), py::arg("physics"));
}

PYBIND11_MODULE(xcgd, m) {
  using T = double;

  // Base class definitions
  py::class_<xcgd::MeshBase<T>, std::shared_ptr<xcgd::MeshBase<T>>>(m,
                                                                    "MeshBase");
  py::class_<xcgd::MeshAssemblerBase<T>,
             std::shared_ptr<xcgd::MeshAssemblerBase<T>>>(m,
                                                          "MeshAssemblerBase");

  // Cartesian mesh base class
  py::class_<xcgd::CartesianMesh<T>, xcgd::MeshBase<T>,
             std::shared_ptr<xcgd::CartesianMesh<T>>>(m, "CartesianMesh")
      .def(py::init<int, int, T>(), py::arg("nx"), py::arg("ny"),
           py::arg("delta"));

  // Bind the physics classes
  py::class_<xcgd::HelmholtzPhysics>(m, "Helmholtz")
      .def(py::init<T>(), py::arg("r"));
  py::class_<xcgd::LinearElasticity2D<>>(m, "LinearElasticity2D")
      .def(py::init<T, T>(), py::arg("E"), py::arg("nu"));

  // Bind the different physics that is needed
  bind_assembler<T, xcgd::HelmholtzPhysics>(m, "HelmholtzAssembler");
  bind_assembler<T, xcgd::LinearElasticity2D<>>(m,
                                                "LinearElasticity2DAssembler");

  py::class_<xcgd::Assembler<T>, std::shared_ptr<xcgd::Assembler<T>>>(
      m, "Assembler")
      .def(py::init<std::vector<std::shared_ptr<xcgd::MeshAssemblerBase<T>>>>(),
           py::arg("assemblers"));
}

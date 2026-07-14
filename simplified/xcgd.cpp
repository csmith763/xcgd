#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "assembler.h"
#include "cartesian_mesh.h"
#include "cut_mesh.h"
#include "cut_quadtree_mesh.h"
#include "physics.h"
#include "quadtree.h"
#include "quadtree_mesh.h"

namespace py = pybind11;

template <typename T, class Physics>
void bind_assembler(py::module_& m, const std::string& name) {
  py::class_<xcgd::MeshAssembler<T, Physics>, xcgd::MeshAssemblerBase<T>,
             std::shared_ptr<xcgd::MeshAssembler<T, Physics>>>(m, name.c_str())
      .def(py::init<std::shared_ptr<xcgd::MeshBase<T>>, const Physics&>(),
           py::arg("mesh"), py::arg("physics"));
}

template <typename T>
py::array_t<T> make_vector_view(std::vector<T>& vec, py::handle base) {
  return py::array_t<T>({static_cast<py::ssize_t>(vec.size())},
                        {static_cast<py::ssize_t>(sizeof(T))}, vec.data(),
                        base);
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
           py::arg("delta"))
      .def("get_node_locations", [](xcgd::CartesianMesh<T>& self) {
        std::vector<T> X = self.get_node_locations();
        py::ssize_t num_nodes = static_cast<py::ssize_t>(X.size() / 2);
        py::array_t<T> arr({num_nodes, py::ssize_t(2)});
        auto r = arr.template mutable_unchecked<2>();
        for (py::ssize_t i = 0; i < num_nodes; i++) {
          r(i, 0) = X[2 * i];
          r(i, 1) = X[2 * i + 1];
        }
        return arr;
      });

  // Enum identifying which sub-domain of the cut mesh to query. This must be
  // registered before it is used as a default argument below.
  py::enum_<xcgd::CutDomain>(m, "CutDomain")
      .value("INTERIOR_VOLUME", xcgd::CutDomain::INTERIOR_VOLUME)
      .value("EXTERIOR_VOLUME", xcgd::CutDomain::EXTERIOR_VOLUME)
      .value("INTERFACE_BOUNDARY", xcgd::CutDomain::INTERFACE_BOUNDARY);

  py::class_<xcgd::CartesianCutMesh<T>,
             std::shared_ptr<xcgd::CartesianCutMesh<T>>>(m, "CartesianCutMesh")
      .def(py::init<std::shared_ptr<xcgd::CartesianMesh<T>>>())
      .def("update", &xcgd::CartesianCutMesh<T>::update)
      .def("update_derivatives", &xcgd::CartesianCutMesh<T>::update_derivatives)
      .def(
          "get_lsf",
          [](xcgd::CartesianCutMesh<T>& self) {
            return make_vector_view(self.get_lsf(), py::cast(&self));
          },
          py::return_value_policy::reference_internal)
      .def(
          "get_node_locations",
          [](xcgd::CartesianCutMesh<T>& self, xcgd::CutDomain domain) {
            std::vector<T> X = self.get_node_locations(domain);
            py::ssize_t num_nodes = static_cast<py::ssize_t>(X.size() / 2);
            py::array_t<T> arr({num_nodes, py::ssize_t(2)});
            auto r = arr.template mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < num_nodes; i++) {
              r(i, 0) = X[2 * i];
              r(i, 1) = X[2 * i + 1];
            }
            return arr;
          },
          py::arg("domain") = xcgd::CutDomain::INTERIOR_VOLUME)
      .def("create_interior_mesh",
           &xcgd::CartesianCutMesh<T>::create_interior_mesh)
      .def("create_exterior_mesh",
           &xcgd::CartesianCutMesh<T>::create_exterior_mesh)
      .def("create_interface_mesh",
           &xcgd::CartesianCutMesh<T>::create_interface_mesh);

  // Bind the physics classes
  py::class_<xcgd::HelmholtzPhysics>(m, "Helmholtz")
      .def(py::init<T>(), py::arg("r"));
  py::class_<xcgd::LinearElasticity2D<>>(m, "LinearElasticity2D")
      .def(py::init<T, T>(), py::arg("E"), py::arg("nu"));
  py::class_<xcgd::ElasticityMass2D>(m, "ElasticityMass2D")
      .def(py::init<T>(), py::arg("rho"));

  // Bind the different physics that is needed
  bind_assembler<T, xcgd::HelmholtzPhysics>(m, "HelmholtzAssembler");
  bind_assembler<T, xcgd::LinearElasticity2D<>>(m,
                                                "LinearElasticity2DAssembler");
  bind_assembler<T, xcgd::ElasticityMass2D>(m, "ElasticityMass2DAssembler");

  // Wrapper for the CSR matrix
  py::class_<xcgd::CSRMat<T>, std::shared_ptr<xcgd::CSRMat<T>>>(m, "CSRMat")
      .def("zero", &xcgd::CSRMat<T>::zero)
      .def_readonly("nrows", &xcgd::CSRMat<T>::nrows)
      .def_property_readonly(
          "rowp",
          [](xcgd::CSRMat<T>& self) {
            return make_vector_view(self.rowp, py::cast(&self));
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "cols",
          [](xcgd::CSRMat<T>& self) {
            return make_vector_view(self.cols, py::cast(&self));
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "data",
          [](xcgd::CSRMat<T>& self) {
            return make_vector_view(self.data, py::cast(&self));
          },
          py::return_value_policy::reference_internal);

  // Wrapper for the assembler class
  py::class_<xcgd::Assembler<T>, std::shared_ptr<xcgd::Assembler<T>>>(
      m, "Assembler")
      .def(py::init<std::vector<std::shared_ptr<xcgd::MeshAssemblerBase<T>>>>(),
           py::arg("assemblers"))
      .def("update", &xcgd::Assembler<T>::update)
      .def(
          "get_dof",
          [](xcgd::Assembler<T>& self) {
            return make_vector_view(self.get_dof(), py::cast(&self));
          },
          py::return_value_policy::reference_internal)
      .def(
          "get_residual",
          [](xcgd::Assembler<T>& self) {
            return make_vector_view(self.get_residual(), py::cast(&self));
          },
          py::return_value_policy::reference_internal)
      .def(
          "get_adjoint",
          [](xcgd::Assembler<T>& self) {
            return make_vector_view(self.get_adjoint(), py::cast(&self));
          },
          py::return_value_policy::reference_internal)
      .def(
          "get_dfdx",
          [](xcgd::Assembler<T>& self) {
            return make_vector_view(self.get_dfdx(), py::cast(&self));
          },
          py::return_value_policy::reference_internal)
      .def(
          "get_jacobian",
          [](xcgd::Assembler<T>& self) -> xcgd::CSRMat<T>& {
            return self.get_jacobian();
          },
          py::return_value_policy::reference_internal)
      .def("eval_functional", &xcgd::Assembler<T>::eval_functional)
      .def("eval_residual", &xcgd::Assembler<T>::eval_residual)
      .def("eval_jacobian", &xcgd::Assembler<T>::eval_jacobian);

  py::class_<xcgd::Quadtree, std::shared_ptr<xcgd::Quadtree>>(m, "Quadtree")
      .def(py::init<>())
      .def("size", &xcgd::Quadtree::size)
      .def("to_vtk", &xcgd::Quadtree::to_vtk, py::arg("filename"))
      .def("duplicate", &xcgd::Quadtree::duplicate)
      .def("coarsen", &xcgd::Quadtree::coarsen)
      .def("balance", &xcgd::Quadtree::balance,
           py::arg("balance_corner") = true)
      .def(
          "refine",
          [](xcgd::Quadtree& tree, py::object refinement,
             std::int32_t min_level, std::int32_t max_level) {
            if (refinement.is_none()) {
              tree.refine(nullptr, min_level, max_level);
              return;
            }

            auto arr =
                py::array_t<int, py::array::c_style | py::array::forcecast>(
                    refinement);

            if (arr.ndim() != 1) {
              throw py::value_error(
                  "refinement must be a 1D integer array/list");
            }

            if (arr.shape(0) != tree.size()) {
              throw py::value_error(
                  "refinement must have length equal to quadtree.size()");
            }

            tree.refine(arr.data(), min_level, max_level);
          },
          py::arg("refinement") = py::none(), py::arg("min_level") = 0,
          py::arg("max_level") = xcgd::Quadrant::MAX_LEVEL);

  py::class_<xcgd::QuadtreeMesh<T>, xcgd::MeshBase<T>,
             std::shared_ptr<xcgd::QuadtreeMesh<T>>>(m, "QuadtreeMesh")
      .def(py::init<std::shared_ptr<xcgd::Quadtree>, T>(), py::arg("tree"),
           py::arg("length") = 1.0)
      .def("update", &xcgd::QuadtreeMesh<T>::update)
      .def("get_node_locations", &xcgd::QuadtreeMesh<T>::get_node_locations);

  py::class_<xcgd::QuadtreeCutMesh<T>,
             std::shared_ptr<xcgd::QuadtreeCutMesh<T>>>(m, "QuadtreeCutMesh")
      .def(py::init<std::shared_ptr<xcgd::QuadtreeMesh<T>>,
                    std::shared_ptr<xcgd::QuadtreeMesh<T>>>())
      .def("update", &xcgd::QuadtreeCutMesh<T>::update)
      .def("update_derivatives", &xcgd::QuadtreeCutMesh<T>::update_derivatives)
      .def(
          "get_lsf",
          [](xcgd::QuadtreeCutMesh<T>& self) {
            return make_vector_view(self.get_lsf(), py::cast(&self));
          },
          py::return_value_policy::reference_internal)
      .def("get_interface_elements",
           &xcgd::QuadtreeCutMesh<T>::get_interface_elements);
}

#ifndef XCGD_CUT_MESH_H
#define XCGD_CUT_MESH_H

#include <array>
#include <vector>

#include "mesh_base.h"
#include "quadrature.h"
#include "vandermonde.h"

namespace xcgd {

template <typename T>
class CartesianCutMesh {
  static constexpr int spatial_dim = 2;
  static constexpr int degree = 3;
  enum class Location { INTERIOR, INTERFACE, EXTERIOR };

 public:
  CartesianCutMesh(std::shared_ptr<CartesianMesh<T>> mesh)
      : mesh(mesh),
        lsf(mesh->get_max_node_index()),
        elem_location(mesh->get_num_elements(), Location::INTERIOR),
        num_interior_nodes(0),
        interior_nodes(mesh->get_max_node_index(), -1),
        num_exterior_nodes(0),
        exterior_nodes(mesh->get_max_node_index(), -1),
        interior_edge_nodes(mesh->get_num_edges()),
        exterior_edge_nodes(mesh->get_num_edges()) {}

  std::vector<T>& get_lsf() { return lsf; }

  void update() {
    // First update the tags for whether the cell is interior, exterior or
    // an interface element
    int num_elements = mesh->get_num_elements();
    std::vector<int> nodes(4);
    int num_interior = 0, num_exterior = 0, num_interface = 0;
    for (int elem = 0; elem < num_elements; elem++) {
      mesh->get_cell_nodes(elem, nodes);

      // lsf[i] > 0 => exterior
      // lsf[i] < 0 => interior
      // lsf[i] == 0 => boundary
      bool interior = true, exterior = true;
      for (int ii = 0; ii < 4; ii++) {
        if (lsf[nodes[ii]] > T(0)) {
          // This indicates at least one exterior node, therefore the element
          // is not in the interior..
          interior = false;
        }
        if (lsf[nodes[ii]] <= T(0)) {
          // This indicates an interior node so the element cannot be exterior.
          // When the level set is identically = 0, we treat this also as an
          // interior node for tie breaking
          exterior = false;
        }
      }

      if (interior == false && exterior == false) {
        elem_location[elem] = Location::INTERFACE;
        num_interface++;
      } else if (interior) {
        elem_location[elem] = Location::INTERIOR;
        num_interior++;
      } else {
        elem_location[elem] = Location::EXTERIOR;
        num_exterior++;
      }
    }

    // Set the elements
    interior_elems.clear();
    exterior_elems.clear();
    interface_elems.clear();

    interior_elems.reserve(num_interior);
    exterior_elems.reserve(num_exterior);
    interface_elems.reserve(num_interface);

    // Fill in the element types
    for (int elem = 0; elem < num_elements; elem++) {
      if (elem_location[elem] == Location::INTERIOR) {
        interior_elems.push_back(elem);
      } else if (elem_location[elem] == Location::EXTERIOR) {
        interior_elems.push_back(elem);
      } else {  // elem_location[elem] == Location::INTERFACE
        interface_elems.push_back(elem);
      }
    }

    update_interface_quadratures();

    // First set interior nodes
    std::fill(interior_nodes.begin(), interior_nodes.end(), -1);
    num_interior_nodes = 0;
    for (int elem = 0; elem < num_elements; elem++) {
      if (elem_location[elem] == Location::INTERIOR ||
          elem_location[elem] == Location::INTERFACE) {
        mesh->get_cell_nodes(elem, nodes);

        for (int ii = 0; ii < 4; ii++) {
          if (interior_nodes[nodes[ii]] < 0) {
            interior_nodes[nodes[ii]] = num_interior_nodes;
            num_interior_nodes++;
          }
        }
      }
    }

    // Set the exterior node numbers
    std::fill(exterior_nodes.begin(), exterior_nodes.end(), -1);
    num_exterior_nodes = 0;
    for (int elem = 0; elem < num_elements; elem++) {
      if (elem_location[elem] == Location::EXTERIOR ||
          elem_location[elem] == Location::INTERFACE) {
        mesh->get_cell_nodes(elem, nodes);

        for (int ii = 0; ii < 4; ii++) {
          if (exterior_nodes[nodes[ii]] < 0) {
            exterior_nodes[nodes[ii]] = num_exterior_nodes;
            num_exterior_nodes++;
          }
        }
      }
    }
  }

 private:
  void update_interface_quadratures() {
    int num_interface = interface_elems.size();
    interior_points.resize(num_interface);
    interior_weights.resize(num_interface);

    exterior_points.resize(num_interface);
    exterior_weights.resize(num_interface);

    interface_points.resize(num_interface);
    interface_weights.resize(num_interface);
    interface_normals.resize(num_interface);

    // Local info about the level set
    std::vector<int> elem_nodes(mesh->get_max_num_nodes());
    std::vector<T> elem_lsf(mesh->get_max_num_nodes());

    for (int i = 0; i < num_interface; i++) {
      int elem = interface_elems[i];

      // Get the lsf at the nodes
      int nnodes = mesh->get_nodes(elem, elem_nodes);
      for (int j = 0; j < nnodes; j++) {
        elem_lsf[j] = lsf[elem_nodes[j]];
      }

      // Form the interpolant
      auto interp = mesh->create_interp(elem);

      // Allocate the quadrature object
      compute_level_set_quadrature<spatial_dim, degree>(
          interp, elem_lsf, interior_points[i], interior_weights[i],
          exterior_points[i], exterior_weights[i], interface_points[i],
          interface_weights[i], interface_normals[i]);
    }
  }

  // The underlying Cartesian mesh that defines the level set
  std::shared_ptr<CartesianMesh<T>> mesh;

  // The level set values
  std::vector<T> lsf;

  // Location of the elements (interior, interface or exterior)
  std::vector<Location> elem_location;

  // Interior nodes associated with the interior and cut elements
  int num_interior_nodes;
  std::vector<int> interior_nodes;

  // Exterior nodes associated with the cut and exterior elements
  int num_exterior_nodes;
  std::vector<int> exterior_nodes;

  // Interior nodes for each interior element
  std::vector<std::array<int, degree + 1>> interior_edge_nodes;

  // Exterior nodes for each exterior edge
  std::vector<std::array<int, degree + 1>> exterior_edge_nodes;

  // Store the element numbers by type
  std::vector<int> interior_elems;
  std::vector<int> exterior_elems;
  std::vector<int> interface_elems;

  // Store the quadratures for each interface element
  std::vector<std::vector<T>> interior_points;
  std::vector<std::vector<T>> interior_weights;

  std::vector<std::vector<T>> exterior_points;
  std::vector<std::vector<T>> exterior_weights;

  std::vector<std::vector<T>> interface_points;
  std::vector<std::vector<T>> interface_weights;
  std::vector<std::vector<T>> interface_normals;
};

// template <typename T>
// class InteriorCutMesh : public MeshBase<T> {
//  public:
//   InteriorCutMesh(std::shared_ptr<CartesianCutMesh<T>> cut_mesh) : mesh(mesh)
//   {}

//   void update() {}

//   int get_max_node_index() const {}
//   int get_max_num_nodes() const { return 12; }
//   int get_max_num_quadrature_points() const {}
//   int get_num_elements() const {}

//   int get_nodes(int elem, std::vector<int>& nodes) const {}
//   void get_node_points(int elem, std::vector<T>& X) const {}
//   int get_quadrature(int elem, std::vector<T>& weights, std::vector<T>&
//   points,
//                      std::vector<T>& normals) const {}

//   void eval_basis(int elem, int num_quad_points, const std::vector<T>& pts,
//                   std::vector<T>& Nd) const {}

//   std::shared_ptr<CartesianCutMesh<T>> cut_mesh;
// };

}  // namespace xcgd

#endif  // XCGD_CUT_MESH_H
#ifndef XCGD_CUT_MESH_H
#define XCGD_CUT_MESH_H

#include <array>
#include <vector>

#include "basis.h"
#include "mesh_base.h"
#include "quadrature.h"
#include "vandermonde.h"

namespace xcgd {

enum class ElementLocation { INTERIOR, EXTERIOR, INTERFACE };

enum class CutDomain { INTERIOR_VOLUME, EXTERIOR_VOLUME, INTERFACE_BOUNDARY };

// Forward declaration
template <typename T>
class CutMeshComponent;

template <typename T>
class CartesianCutMesh
    : public std::enable_shared_from_this<CartesianCutMesh<T>> {
  static constexpr int spatial_dim = 2;
  static constexpr int degree = 3;

 public:
  CartesianCutMesh(std::shared_ptr<CartesianMesh<T>> mesh)
      : mesh(mesh),
        lsf(mesh->get_max_node_index()),
        elem_location(mesh->get_num_elements(), ElementLocation::INTERIOR),
        num_interior_nodes(0),
        interior_node_map(mesh->get_max_node_index(), -1),
        num_exterior_nodes(0),
        exterior_node_map(mesh->get_max_node_index(), -1) {
    num_interior = 0;
    num_exterior = 0;
    num_interface = 0;
    max_quad_points = 0;
  }

  std::shared_ptr<MeshBase<T>> create_interior_mesh() {
    return std::make_shared<CutMeshComponent<T>>(this->shared_from_this(),
                                                 CutDomain::INTERIOR_VOLUME);
  }
  std::shared_ptr<MeshBase<T>> create_exterior_mesh() {
    return std::make_shared<CutMeshComponent<T>>(this->shared_from_this(),
                                                 CutDomain::EXTERIOR_VOLUME);
  }
  std::shared_ptr<MeshBase<T>> create_interface_mesh() {
    return std::make_shared<CutMeshComponent<T>>(this->shared_from_this(),
                                                 CutDomain::INTERFACE_BOUNDARY);
  }

  int get_num_elements(CutDomain domain) const {
    if (domain == CutDomain::INTERIOR_VOLUME) {
      return num_interior + num_interface;
    } else if (domain == CutDomain::EXTERIOR_VOLUME) {
      return num_exterior + num_interface;
    } else {
      return num_interface;
    }
  }

  int get_max_node_index(CutDomain domain) const {
    if (domain == CutDomain::INTERIOR_VOLUME) {
      return num_interior_nodes;
    } else if (domain == CutDomain::EXTERIOR_VOLUME) {
      return num_interior_nodes + num_exterior_nodes;
    } else {
      return num_interior_nodes + num_exterior_nodes;
    }
  }

  int get_max_num_nodes(CutDomain domain) const {
    if (domain == CutDomain::INTERIOR_VOLUME ||
        domain == CutDomain::EXTERIOR_VOLUME) {
      return detail::PolyBasis2D::MAX_BASIS;
    } else {
      return 2 * detail::PolyBasis2D::MAX_BASIS;
    }
  }

  int get_max_num_quadrature_points(CutDomain domain) const {
    return max_quad_points;
  }

  int get_nodes(CutDomain domain, int elem, std::vector<int>& nodes) const {
    int nnodes = 0;
    if (domain == CutDomain::INTERIOR_VOLUME) {
      if (elem < num_interior) {
        nodes.assign(interior.stencil[elem].begin(),
                     interior.stencil[elem].end());
        nnodes = static_cast<int>(interior.stencil[elem].size());
      } else {
        int k = elem - num_interior;
        nodes.assign(interface_interior.stencil[k].begin(),
                     interface_interior.stencil[k].end());
        nnodes = static_cast<int>(interface_interior.stencil[k].size());
      }
    } else if (domain == CutDomain::EXTERIOR_VOLUME) {
      if (elem < num_exterior) {
        nodes.assign(exterior.stencil[elem].begin(),
                     exterior.stencil[elem].end());
        nnodes = static_cast<int>(exterior.stencil[elem].size());
      } else {
        int k = elem - num_exterior;
        nodes.assign(interface_exterior.stencil[k].begin(),
                     interface_exterior.stencil[k].end());
        nnodes = static_cast<int>(interface_exterior.stencil[k].size());
      }
      for (int i = 0; i < nnodes; i++) {
        nodes[i] += num_interior_nodes;
      }
    } else {
      // Assign both the interior and exterior interface nodes
      nodes.clear();
      nodes.insert(nodes.end(), interface_interior.stencil[elem].begin(),
                   interface_interior.stencil[elem].end());
      nodes.insert(nodes.end(), interface_exterior.stencil[elem].begin(),
                   interface_exterior.stencil[elem].end());
      nnodes = static_cast<int>(nodes.size());
    }

    return nnodes;
  }

  void get_node_points(CutDomain domain, int elem, std::vector<T>& X) const {
    if (domain == CutDomain::INTERIOR_VOLUME) {
      if (elem < num_interior) {
        X.assign(interior.X[elem].begin(), interior.X[elem].end());
      } else {
        int k = elem - num_interior;
        X.assign(interface_interior.X[k].begin(),
                 interface_interior.X[k].end());
      }
    } else if (domain == CutDomain::EXTERIOR_VOLUME) {
      if (elem < num_exterior) {
        X.assign(exterior.X[elem].begin(), exterior.X[elem].end());
      } else {
        int k = elem - num_exterior;
        X.assign(interface_exterior.X[k].begin(),
                 interface_exterior.X[k].end());
      }
    } else {
      // Assign both the interior and exterior interface nodes
      X.clear();
      X.insert(X.end(), interface_interior.X[elem].begin(),
               interface_interior.X[elem].end());
      X.insert(X.end(), interface_exterior.X[elem].begin(),
               interface_exterior.X[elem].end());
    }
  }

  int get_quadrature(CutDomain domain, int elem, std::vector<T>& weights,
                     std::vector<T>& points, std::vector<T>& normals) const {
    if (domain == CutDomain::INTERIOR_VOLUME) {
      if (elem < num_interior) {
        int underlying = interior_elems[elem];
        return mesh->get_quadrature(underlying, weights, points, normals);
      } else {
        int k = elem - num_interior;
        weights.assign(interior_weights[k].begin(), interior_weights[k].end());
        points.assign(interior_points[k].begin(), interior_points[k].end());
        return static_cast<int>(weights.size());
      }
    } else if (domain == CutDomain::EXTERIOR_VOLUME) {
      if (elem < num_exterior) {
        int underlying = exterior_elems[elem];
        return mesh->get_quadrature(underlying, weights, points, normals);
      } else {
        int k = elem - num_exterior;
        weights.assign(exterior_weights[k].begin(), exterior_weights[k].end());
        points.assign(exterior_points[k].begin(), exterior_points[k].end());
        return static_cast<int>(weights.size());
      }
    } else {
      weights.assign(interface_weights[elem].begin(),
                     interface_weights[elem].end());
      points.assign(interface_points[elem].begin(),
                    interface_points[elem].end());
      normals.assign(interface_normals[elem].begin(),
                     interface_normals[elem].end());
      return static_cast<int>(weights.size());
    }
  }

  void eval_basis(CutDomain domain, int elem, int num_quad_points,
                  const std::vector<T>& pts, std::vector<T>& Nd) const {
    if (domain == CutDomain::INTERIOR_VOLUME ||
        domain == CutDomain::EXTERIOR_VOLUME) {
      T x0, y0, delta;
      uint32_t exclude(0);
      std::size_t npts = 0;
      const T* X;

      if (domain == CutDomain::INTERIOR_VOLUME) {
        if (elem < num_interior) {
          mesh->get_element_base_point(interior_elems[elem], x0, y0, delta);

          exclude = interior.exclude[elem];
          npts = interior.stencil[elem].size();
          X = interior.X[elem].data();
        } else {
          int k = elem - num_interior;
          mesh->get_element_base_point(interface_elems[k], x0, y0, delta);
          exclude = interface_interior.exclude[k];
          npts = interface_interior.stencil[k].size();
          X = interface_interior.X[k].data();
        }
      } else if (domain == CutDomain::EXTERIOR_VOLUME) {
        if (elem < num_exterior) {
          mesh->get_element_base_point(exterior_elems[elem], x0, y0, delta);
          exclude = exterior.exclude[elem];
          npts = exterior.stencil[elem].size();
          X = exterior.X[elem].data();
        } else {
          int k = elem - num_exterior;
          mesh->get_element_base_point(interface_elems[k], x0, y0, delta);
          exclude = interface_exterior.exclude[k];
          npts = interface_exterior.stencil[k].size();
          X = interface_exterior.X[k].data();
        }
      }

      if (exclude == uint32_t(0)) {
        int nnodes = static_cast<int>(npts);
        Vandermonde2D interp(x0, y0, delta, nnodes, X,
                             detail::RegularPolyBasis2D{},
                             detail::RegularPolyBasisDeriv2D{});
        interp.eval_basis(num_quad_points, pts.data(), Nd.data());
      } else {
        int nnodes = static_cast<int>(npts);
        Vandermonde2D interp(x0, y0, delta, nnodes, X,
                             detail::PolyBasis2D(exclude),
                             detail::PolyBasisDeriv2D(exclude));
        interp.eval_basis(num_quad_points, pts.data(), Nd.data());
      }
    } else {
      T x0, y0, delta;
      mesh->get_element_base_point(interface_elems[elem], x0, y0, delta);

      uint32_t exclude_int = interface_interior.exclude[elem];
      std::size_t npts_int = interface_interior.stencil[elem].size();
      const T* X_int = interface_interior.X[elem].data();

      uint32_t exclude_ext = interface_exterior.exclude[elem];
      std::size_t npts_ext = interface_exterior.stencil[elem].size();
      const T* X_ext = interface_exterior.X[elem].data();

      // Need to figure out how to do this..
    }
  }

  std::vector<T>& get_lsf() { return lsf; }

  void update() {
    // First update the tags for whether the cell is interior, exterior or
    // an interface element
    int num_elements = mesh->get_num_elements();
    std::vector<int> nodes(4);

    num_interior = 0;
    num_exterior = 0;
    num_interface = 0;
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
          // This indicates an interior node so the element cannot be
          // exterior. When the level set is identically = 0, we treat this
          // also as an interior node for tie breaking
          exterior = false;
        }
      }

      if (interior == false && exterior == false) {
        elem_location[elem] = ElementLocation::INTERFACE;
        num_interface++;
      } else if (interior) {
        elem_location[elem] = ElementLocation::INTERIOR;
        num_interior++;
      } else {
        elem_location[elem] = ElementLocation::EXTERIOR;
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
      if (elem_location[elem] == ElementLocation::INTERIOR) {
        interior_elems.push_back(elem);
      } else if (elem_location[elem] == ElementLocation::EXTERIOR) {
        exterior_elems.push_back(elem);
      } else {  // elem_location[elem] == ElementLocation::INTERFACE
        interface_elems.push_back(elem);
      }
    }

    update_interface_quadratures();

    // Set the interior nodes
    std::fill(interior_node_map.begin(), interior_node_map.end(), -1);
    num_interior_nodes = 0;
    for (int elem = 0; elem < num_elements; elem++) {
      if (elem_location[elem] == ElementLocation::INTERIOR ||
          elem_location[elem] == ElementLocation::INTERFACE) {
        mesh->get_cell_nodes(elem, nodes);

        for (int ii = 0; ii < 4; ii++) {
          if (interior_node_map[nodes[ii]] < 0) {
            interior_node_map[nodes[ii]] = num_interior_nodes;
            num_interior_nodes++;
          }
        }
      }
    }

    // Set the exterior node numbers
    std::fill(exterior_node_map.begin(), exterior_node_map.end(), -1);
    num_exterior_nodes = 0;
    for (int elem = 0; elem < num_elements; elem++) {
      if (elem_location[elem] == ElementLocation::EXTERIOR ||
          elem_location[elem] == ElementLocation::INTERFACE) {
        mesh->get_cell_nodes(elem, nodes);

        // Order the exterior nodes after the interior ones
        for (int ii = 0; ii < 4; ii++) {
          if (exterior_node_map[nodes[ii]] < 0) {
            exterior_node_map[nodes[ii]] =
                num_exterior_nodes + num_interior_nodes;
            num_exterior_nodes++;
          }
        }
      }
    }

    // Update the stencils for each element
    update_stencil(interior, interior_node_map, interior_elems);
    update_stencil(exterior, exterior_node_map, exterior_elems);
    update_stencil(interface_interior, interior_node_map, interface_elems);
    update_stencil(interface_exterior, exterior_node_map, interface_elems);
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

    std::size_t max_pts = 0;
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

      max_pts =
          std::max(max_pts, std::max(interior_weights[i].size(),
                                     std::max(exterior_weights[i].size(),
                                              interface_weights[i].size())));
    }

    max_quad_points = max_pts;
  }

  bool append_edge_stencil(const EdgeLine& line,
                           const std::vector<int>& node_map,
                           std::vector<int>& stencil, std::vector<T>& X) const {
    static constexpr int windows[][degree + 1] = {
        {-1, 0, 1, 2}, {0, 1, 2, 3}, {-2, -1, 0, 1}};

    int best_count = -1;
    std::array<int, degree + 1> best_global;
    std::array<int, degree + 1> best_local;

    best_global.fill(-1);
    best_local.fill(-1);

    for (int k = 0; k < 3; k++) {
      const int* w = windows[k];

      int count = 0;
      std::array<int, degree + 1> trial_global;
      std::array<int, degree + 1> trial_local;

      trial_global.fill(-1);
      trial_local.fill(-1);

      for (int a = 0; a < degree + 1; a++) {
        int ix = line.ix0 + w[a] * line.dix;
        int iy = line.iy0 + w[a] * line.diy;

        int global = mesh->get_node(ix, iy);

        if (global < 0 || global >= static_cast<int>(node_map.size())) {
          continue;
        }

        int local = node_map[global];

        if (local < 0) {
          continue;
        }

        trial_global[a] = global;
        trial_local[a] = local;
        count++;
      }

      if (count == degree + 1) {
        best_global = trial_global;
        best_local = trial_local;
        best_count = count;
        break;
      }

      if (count > best_count) {
        best_count = count;
        best_global = trial_global;
        best_local = trial_local;
      }
    }

    // Append the best stencil, skipping invalid nodes and duplicates.
    for (int a = 0; a < degree + 1; a++) {
      int global = best_global[a];
      int local = best_local[a];

      if (global < 0 || local < 0) {
        continue;
      }

      bool duplicate = false;
      for (int existing : stencil) {
        if (existing == local) {
          duplicate = true;
          break;
        }
      }

      if (duplicate) {
        continue;
      }

      T x, y;
      mesh->get_node_location(global, x, y);

      stencil.push_back(local);
      X.push_back(x);
      X.push_back(y);
    }

    return best_count == degree + 1;
  }

  struct StencilInfo {
    // Exclude info indicating the selected basis functions
    std::vector<uint32_t> exclude;

    // The mapped node numbers for the stencil
    std::vector<std::vector<int>> stencil;

    // Node locations for each element
    std::vector<std::vector<T>> X;
  };

  void update_stencil(StencilInfo& info, const std::vector<int>& node_map,
                      const std::vector<int>& elem_map) {
    info.exclude.resize(elem_map.size());
    info.stencil.resize(elem_map.size());
    info.X.resize(elem_map.size());

    constexpr EdgeSide edge_sides[4] = {EdgeSide::X_NEG, EdgeSide::X_POS,
                                        EdgeSide::Y_NEG, EdgeSide::Y_POS};

    for (std::size_t index = 0; index < elem_map.size(); index++) {
      int elem = elem_map[index];

      info.stencil[index].clear();
      info.X[index].clear();

      // Fill in the stencil for this element
      bool regular = true;
      for (int edge = 0; edge < 4; edge++) {
        EdgeLine line = mesh->get_edge_line(elem, edge_sides[edge]);

        // Populate info.stencil[index] and info.X[index]
        bool flag = append_edge_stencil(line, node_map, info.stencil[index],
                                        info.X[index]);
        regular = regular && flag;
      }

      T x0, y0, delta;
      mesh->get_element_base_point(elem, x0, y0, delta);

      // if (regular) {
      //   info.exclude[index] = uint32_t(0);
      // } else {
      const int num_points = static_cast<int>(info.stencil[index].size());

      info.exclude[index] = compute_exclude_bits_from_points(
          num_points, x0, y0, delta, info.X[index].data());
      // }
    }
  }

  // The underlying Cartesian mesh that defines the level set
  std::shared_ptr<CartesianMesh<T>> mesh;

  // The level set values
  std::vector<T> lsf;

  // Location of the elements (interior, interface or exterior)
  std::vector<ElementLocation> elem_location;

  // Interior nodes associated with the interior and cut elements
  int num_interior_nodes;
  std::vector<int> interior_node_map;

  // Exterior nodes associated with the cut and exterior elements
  int num_exterior_nodes;
  std::vector<int> exterior_node_map;

  // The number of interior exterior and interface elements
  int num_interior;
  int num_exterior;
  int num_interface;
  int max_quad_points;

  // Store the element numbers by type
  std::vector<int> interior_elems;
  std::vector<int> exterior_elems;
  std::vector<int> interface_elems;

  // Store the element stencil information
  StencilInfo interior;
  StencilInfo exterior;
  StencilInfo interface_interior;
  StencilInfo interface_exterior;

  // Store the quadratures for each interface element
  std::vector<std::vector<T>> interior_points;
  std::vector<std::vector<T>> interior_weights;

  std::vector<std::vector<T>> exterior_points;
  std::vector<std::vector<T>> exterior_weights;

  std::vector<std::vector<T>> interface_points;
  std::vector<std::vector<T>> interface_weights;
  std::vector<std::vector<T>> interface_normals;
};

template <typename T>
class CutMeshComponent final : public MeshBase<T> {
 public:
  CutMeshComponent(std::shared_ptr<CartesianCutMesh<T>> mesh, CutDomain domain)
      : mesh(std::move(mesh)), domain(domain) {}

  int get_max_node_index() const override {
    return mesh->get_max_node_index(domain);
  }

  int get_max_num_nodes() const override {
    return mesh->get_max_num_nodes(domain);
  }

  int get_max_num_quadrature_points() const override {
    return mesh->get_max_num_quadrature_points(domain);
  }

  int get_num_elements() const override {
    return mesh->get_num_elements(domain);
  }

  int get_nodes(int elem, std::vector<int>& nodes) const override {
    return mesh->get_nodes(domain, elem, nodes);
  }

  void get_node_points(int elem, std::vector<T>& X) const override {
    mesh->get_node_points(domain, elem, X);
  }

  int get_quadrature(int elem, std::vector<T>& weights, std::vector<T>& points,
                     std::vector<T>& normals) const override {
    return mesh->get_quadrature(domain, elem, weights, points, normals);
  }

  void eval_basis(int elem, int num_quad_points, const std::vector<T>& pts,
                  std::vector<T>& Nd) const override {
    mesh->eval_basis(domain, elem, num_quad_points, pts, Nd);
  }

 private:
  std::shared_ptr<CartesianCutMesh<T>> mesh;
  CutDomain domain;
};

}  // namespace xcgd

#endif  // XCGD_CUT_MESH_H

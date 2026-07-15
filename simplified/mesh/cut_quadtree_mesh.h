#ifndef XCGD_CUT_QUADTREE_MESH_H
#define XCGD_CUT_QUADTREE_MESH_H

#include "cut_mesh.h"
#include "quadtree_mesh.h"

namespace xcgd {

template <typename T>
class CutQuadMeshComponent;

template <typename T>
class QuadtreeCutMesh
    : public std::enable_shared_from_this<QuadtreeCutMesh<T>> {
 public:
  static constexpr int spatial_dim = 2;
  static constexpr int degree = 3;

  QuadtreeCutMesh(std::shared_ptr<QuadtreeMesh<T>> mesh,
                  std::shared_ptr<QuadtreeMesh<T>> lsf_mesh)
      : mesh(mesh), lsf_mesh(lsf_mesh), lsf(lsf_mesh->get_max_node_index()) {
    num_interior = 0;
    num_exterior = 0;
    num_interface = 0;
    max_quad_points = 0;
  }

  const std::vector<int>& get_interior_elements() const {
    return interior_elems;
  }
  const std::vector<int>& get_exterior_elements() const {
    return exterior_elems;
  }
  const std::vector<int>& get_interface_elements() const {
    return interface_elems;
  }

  std::shared_ptr<MeshBase<T>> create_interior_mesh() {
    return std::make_shared<CutQuadMeshComponent<T>>(
        this->shared_from_this(), CutDomain::INTERIOR_VOLUME);
  }
  std::shared_ptr<MeshBase<T>> create_exterior_mesh() {
    return std::make_shared<CutQuadMeshComponent<T>>(
        this->shared_from_this(), CutDomain::EXTERIOR_VOLUME);
  }
  std::shared_ptr<MeshBase<T>> create_interface_mesh() {
    return std::make_shared<CutQuadMeshComponent<T>>(
        this->shared_from_this(), CutDomain::INTERFACE_BOUNDARY);
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

  int get_max_element_nodes(CutDomain domain) const {
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

  void get_points(CutDomain domain, int elem, std::vector<T>& X) const {
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
          mesh->get_base_data(interior_elems[elem], x0, y0, delta);
          exclude = interior.exclude[elem];
          npts = interior.stencil[elem].size();
          X = interior.X[elem].data();
        } else {
          int k = elem - num_interior;
          mesh->get_base_data(interface_elems[k], x0, y0, delta);
          exclude = interface_interior.exclude[k];
          npts = interface_interior.stencil[k].size();
          X = interface_interior.X[k].data();
        }
      } else if (domain == CutDomain::EXTERIOR_VOLUME) {
        if (elem < num_exterior) {
          mesh->get_base_data(exterior_elems[elem], x0, y0, delta);
          exclude = exterior.exclude[elem];
          npts = exterior.stencil[elem].size();
          X = exterior.X[elem].data();
        } else {
          int k = elem - num_exterior;
          mesh->get_base_data(interface_elems[k], x0, y0, delta);
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
      mesh->get_base_data(interface_elems[elem], x0, y0, delta);

      uint32_t exclude_int = interface_interior.exclude[elem];
      std::size_t npts_int = interface_interior.stencil[elem].size();
      const T* X_int = interface_interior.X[elem].data();

      uint32_t exclude_ext = interface_exterior.exclude[elem];
      std::size_t npts_ext = interface_exterior.stencil[elem].size();
      const T* X_ext = interface_exterior.X[elem].data();

      // Need to figure out how to do this..
    }
  }

  void reverse_eval_basis(CutDomain domain, int elem, int num_quad_points,
                          const std::vector<T>& pts, const std::vector<T>& bNd,
                          std::vector<T> bpts) const {
    if (domain == CutDomain::INTERIOR_VOLUME ||
        domain == CutDomain::EXTERIOR_VOLUME) {
      T x0, y0, delta;
      uint32_t exclude(0);
      std::size_t npts = 0;
      const T* X;

      if (domain == CutDomain::INTERIOR_VOLUME) {
        if (elem < num_interior) {
          mesh->get_base_data(interior_elems[elem], x0, y0, delta);
          exclude = interior.exclude[elem];
          npts = interior.stencil[elem].size();
          X = interior.X[elem].data();
        } else {
          int k = elem - num_interior;
          mesh->get_base_data(interface_elems[k], x0, y0, delta);
          exclude = interface_interior.exclude[k];
          npts = interface_interior.stencil[k].size();
          X = interface_interior.X[k].data();
        }
      } else if (domain == CutDomain::EXTERIOR_VOLUME) {
        if (elem < num_exterior) {
          mesh->get_base_data(exterior_elems[elem], x0, y0, delta);
          exclude = exterior.exclude[elem];
          npts = exterior.stencil[elem].size();
          X = exterior.X[elem].data();
        } else {
          int k = elem - num_exterior;
          mesh->get_base_data(interface_elems[k], x0, y0, delta);
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
        interp.reverse_eval_basis(num_quad_points, pts.data(), bNd.data(),
                                  bpts.data());
      } else {
        int nnodes = static_cast<int>(npts);
        Vandermonde2D interp(x0, y0, delta, nnodes, X,
                             detail::PolyBasis2D(exclude),
                             detail::PolyBasisDeriv2D(exclude));
        interp.reverse_eval_basis(num_quad_points, pts.data(), bNd.data(),
                                  bpts.data());
      }
    } else {
      T x0, y0, delta;
      mesh->get_base_data(interface_elems[elem], x0, y0, delta);

      uint32_t exclude_int = interface_interior.exclude[elem];
      std::size_t npts_int = interface_interior.stencil[elem].size();
      const T* X_int = interface_interior.X[elem].data();

      uint32_t exclude_ext = interface_exterior.exclude[elem];
      std::size_t npts_ext = interface_exterior.stencil[elem].size();
      const T* X_ext = interface_exterior.X[elem].data();

      // Need to figure out how to do this..
    }
  }

  // The nodes are the design variables
  int get_max_design_index() const { return lsf_mesh->get_max_node_index(); }

  int get_max_element_design_vars() const {
    return lsf_mesh->get_max_element_nodes();
  }

  int get_quadrature_derivative(CutDomain domain, int elem,
                                std::vector<T>& dwdx, std::vector<T>& dpdx,
                                std::vector<T>& dndx, int& ndvs,
                                std::vector<int>& dvs) const {
    ndvs = 0;
    if (domain == CutDomain::INTERIOR_VOLUME) {
      if (elem >= num_interior) {
        // Copy the derivative of the weights/point wrt. design variables
        int k = elem - num_interior;
        dwdx.assign(interior_weights_jacobian[k].begin(),
                    interior_weights_jacobian[k].end());
        dpdx.assign(interior_points_jacobian[k].begin(),
                    interior_points_jacobian[k].end());

        // Set the desgin variable indices (nodes from the underlying mesh)
        int mesh_elem = interface_elems[k];
        int lsf_elem = elem_to_lsf[mesh_elem];
        ndvs = lsf_mesh->get_nodes(lsf_elem, dvs);

        return static_cast<int>(interior_weights[k].size());
      }
    } else if (domain == CutDomain::EXTERIOR_VOLUME) {
      if (elem >= num_exterior) {
        // Copy the derivative of the weights/point wrt. design variables
        int k = elem - num_exterior;
        dwdx.assign(exterior_weights_jacobian[k].begin(),
                    exterior_weights_jacobian[k].end());
        dpdx.assign(exterior_points_jacobian[k].begin(),
                    exterior_points_jacobian[k].end());

        // Set the desgin variable indices (nodes from the underlying mesh)
        int mesh_elem = interface_elems[k];
        int lsf_elem = elem_to_lsf[mesh_elem];
        ndvs = lsf_mesh->get_nodes(lsf_elem, dvs);

        return static_cast<int>(exterior_weights[k].size());
      }
    } else {
      dwdx.assign(interface_weights_jacobian[elem].begin(),
                  interface_weights_jacobian[elem].end());
      dpdx.assign(interface_points_jacobian[elem].begin(),
                  interface_points_jacobian[elem].end());
      dndx.assign(interface_normals_jacobian[elem].begin(),
                  interface_normals_jacobian[elem].end());

      // Set the desgin variable indices (nodes from the underlying mesh)
      int mesh_elem = interface_elems[elem];
      int lsf_elem = elem_to_lsf[mesh_elem];

      ndvs = lsf_mesh->get_nodes(lsf_elem, dvs);

      return static_cast<int>(interface_weights.size());
    }

    // Only interface elements contribute derivatives directly
    return 0;
  }

  std::vector<T>& get_lsf() { return lsf; }

  void update() {
    // First update the tags for whether the cell is interior, exterior or
    // an interface element
    int num_elements = mesh->get_num_elements();
    int num_nodes = mesh->get_max_node_index();

    elem_location.resize(num_elements, ElementLocation::INTERIOR);
    interior_node_map.resize(num_nodes, -1);
    exterior_node_map.resize(num_nodes, -1);

    // Find the lsf element that lies within the given level set
    elem_to_lsf.resize(num_elements);

    // Find the enclosing quadrants
    for (int elem = 0; elem < num_elements; elem++) {
      Quadrant q = mesh->get_quadrant(elem);
      int index = lsf_mesh->find_enclosing_index(q);

      if (index < 0) {
        throw std::runtime_error("No enclosing quadrant found");
      }
      elem_to_lsf[elem] = index;
    }

    // Nodes and points for the LSF in each cell
    std::vector<int> nodes(4);
    std::vector<T> points(8);

    // Interpolate the LSF to the nodes of the mesh from the lsf_mesh
    std::vector<T> lsf_at_nodes(mesh->get_max_node_index());

    // Values needed for the interpolation
    std::vector<int> elem_nodes(lsf_mesh->get_max_element_nodes());
    std::vector<T> elem_lsf(lsf_mesh->get_max_element_nodes());

    for (int elem = 0; elem < num_elements; elem++) {
      int lsf_elem = elem_to_lsf[elem];

      // Get the lsf at the nodes
      int nnodes = lsf_mesh->get_nodes(lsf_elem, elem_nodes);
      for (int j = 0; j < nnodes; j++) {
        elem_lsf[j] = lsf[elem_nodes[j]];
      }

      auto interp = lsf_mesh->create_interp(lsf_elem);

      mesh->get_cell_nodes(elem, nodes);
      mesh->get_cell_points(elem, points);
      for (int corner = 0; corner < 4; corner++) {
        lsf_at_nodes[nodes[corner]] =
            interp.eval(&points[2 * corner], elem_lsf.data());
      }
    }

    // Evaluate the level set function at the nodes of the mesh
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
        if (lsf_at_nodes[nodes[ii]] > T(0)) {
          // This indicates at least one exterior node, therefore the element
          // is not in the interior..
          interior = false;
        }
        if (lsf_at_nodes[nodes[ii]] <= T(0)) {
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

    // Update the stencils for each element class
    update_stencil(interior, interior_node_map, interior_elems);
    update_stencil(exterior, exterior_node_map, exterior_elems);
    update_stencil(interface_interior, interior_node_map, interface_elems);
    update_stencil(interface_exterior, exterior_node_map, interface_elems);
  }

  void update_derivatives() {
    int num_interface = interface_elems.size();
    interior_points_jacobian.resize(num_interface);
    interior_weights_jacobian.resize(num_interface);

    exterior_points_jacobian.resize(num_interface);
    exterior_weights_jacobian.resize(num_interface);

    interface_points_jacobian.resize(num_interface);
    interface_weights_jacobian.resize(num_interface);
    interface_normals_jacobian.resize(num_interface);

    // Local info about the level set
    std::vector<int> elem_nodes(lsf_mesh->get_max_element_nodes());
    std::vector<T> elem_lsf(lsf_mesh->get_max_element_nodes());

    for (int i = 0; i < num_interface; i++) {
      interior_points_jacobian[i].clear();
      interior_weights_jacobian[i].clear();
      exterior_points_jacobian[i].clear();
      exterior_weights_jacobian[i].clear();
      interface_points_jacobian[i].clear();
      interface_weights_jacobian[i].clear();
      interface_normals_jacobian[i].clear();

      int elem = interface_elems[i];
      int lsf_elem = elem_to_lsf[elem];

      // Get the lsf at the nodes
      int nnodes = lsf_mesh->get_nodes(lsf_elem, elem_nodes);
      for (int j = 0; j < nnodes; j++) {
        elem_lsf[j] = lsf[elem_nodes[j]];
      }

      // Form the interpolant
      auto interp = lsf_mesh->create_interp(lsf_elem);

      // Get the domain of integration
      T x0, y0, delta;
      mesh->get_base_data(elem, x0, y0, delta);

      // Allocate the quadrature object
      compute_level_set_quadrature_derivatives<spatial_dim, degree>(
          x0, y0, delta, interp, elem_lsf, interior_points_jacobian[i],
          interior_weights_jacobian[i], exterior_points_jacobian[i],
          exterior_weights_jacobian[i], interface_points_jacobian[i],
          interface_weights_jacobian[i], interface_normals_jacobian[i]);
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
    std::vector<int> elem_nodes(lsf_mesh->get_max_element_nodes());
    std::vector<T> elem_lsf(lsf_mesh->get_max_element_nodes());

    std::size_t max_pts = 0;
    for (int i = 0; i < num_interface; i++) {
      interior_points[i].clear();
      interior_weights[i].clear();
      exterior_points[i].clear();
      exterior_weights[i].clear();
      interface_points[i].clear();
      interface_weights[i].clear();
      interface_normals[i].clear();

      int elem = interface_elems[i];
      int lsf_elem = elem_to_lsf[elem];

      // Get the lsf at the nodes
      int nnodes = lsf_mesh->get_nodes(lsf_elem, elem_nodes);
      for (int j = 0; j < nnodes; j++) {
        elem_lsf[j] = lsf[elem_nodes[j]];
      }

      // Form the interpolant
      auto interp = lsf_mesh->create_interp(lsf_elem);

      // Get the domain of integration
      T x0, y0, delta;
      mesh->get_base_data(elem, x0, y0, delta);

      // Allocate the quadrature object
      compute_level_set_quadrature<spatial_dim, degree>(
          x0, y0, delta, interp, elem_lsf, interior_points[i],
          interior_weights[i], exterior_points[i], exterior_weights[i],
          interface_points[i], interface_weights[i], interface_normals[i]);

      max_pts =
          std::max(max_pts, std::max(interior_weights[i].size(),
                                     std::max(exterior_weights[i].size(),
                                              interface_weights[i].size())));
    }

    max_quad_points = max_pts;
  }

  struct StencilInfo {
    // Exclude info indicating the selected basis functions
    std::vector<uint32_t> exclude;

    // The mapped node numbers for the stencil
    std::vector<std::array<std::vector<int>, 4>> edge_stencil;
    std::vector<std::vector<int>> stencil;

    // Node locations for each element
    std::vector<std::vector<T>> X;
  };

  int get_mapped_node_index(const QuadrantNode& node,
                            const std::vector<int>& node_map) {
    NodeArray& nodes = *mesh->get_node_array();

    int base_index = nodes.get_index(node);
    if (base_index >= static_cast<int>(node_map.size())) {
      throw std::runtime_error("Base index out of range");
    }
    if (base_index >= 0) {
      return node_map[base_index];
    }
    return -1;
  }

  void build_hanging_edge_stencil(StencilInfo& info,
                                  const std::vector<int>& node_map,
                                  const std::vector<int>& elem_map) {
    QuadrantArray& quads = *mesh->get_quadrants();

    const int edge_index_to_adjacent[] = {1, 0, 3, 2};

    std::vector<int> inv_elem_map(quads.size(), -1);
    for (int index = 0; index < elem_map.size(); index++) {
      inv_elem_map[elem_map[index]] = index;
    }

    for (int index = 0; index < elem_map.size(); index++) {
      int elem = elem_map[index];

      // Check if the info flag is set
      if (quads[elem].info) {
        // Check if we have a dependent edge from a coarse element
        for (int edge_index = 0; edge_index < 4; edge_index++) {
          if (quads[elem].info & (1 << (4 + edge_index))) {
            std::vector<int>& edge = info.edge_stencil[index][edge_index];
            const Quadrant& quad = quads[elem];

            // Normally, this will be of length 5
            edge.clear();
            edge.reserve(5);

            // Find the hanging node index
            std::int32_t h = quad.get_size();
            std::int32_t hd = h / 2;

            QuadrantNode node;
            if (edge_index < 2) {
              node.x = quad.x + h * (edge_index % 2);
              node.y = quad.y + hd;
            } else {
              node.x = quad.x + hd;
              node.y = quad.y + h * (edge_index % 2);
            }

            // Get the node index
            int node_index = get_mapped_node_index(node, node_map);

            // This is a hanging node edge
            if (node_index >= 0) {
              edge.push_back(node_index);
            }

            // Build the remaining stencil along the edge
            add_edge_stencil(quad, edge_index, node_map, edge);
          }
        }
      }
    }

    // The remaining dependent edges can be copied from the others
    for (int index = 0; index < elem_map.size(); index++) {
      int elem = elem_map[index];

      if (quads[elem].info) {
        for (int edge_index = 0; edge_index < 4; edge_index++) {
          if (quads[elem].info & (1 << edge_index)) {
            Quadrant p = quads[elem].parent();
            p = p.edge_neighbor(edge_index);

            int element = quads.get_index(p);
            int adj_edge = edge_index_to_adjacent[edge_index];
            int mapped_element = inv_elem_map[element];

            // Copy the contents to the other edge
            if (mapped_element >= 0) {
              info.edge_stencil[index][edge_index] =
                  info.edge_stencil[mapped_element][adj_edge];
            }
          }
        }
      }
    }
  }

  void build_regular_edge_stencil(StencilInfo& info,
                                  const std::vector<int>& node_map,
                                  const std::vector<int>& elem_map) {
    QuadrantArray& quads = *mesh->get_quadrants();

    for (int index = 0; index < elem_map.size(); index++) {
      int elem = elem_map[index];
      const Quadrant& quad = quads[elem];

      // Check that no info is set for this edge
      for (int edge_index = 0; edge_index < 4; edge_index++) {
        std::vector<int>& edge = info.edge_stencil[index][edge_index];

        // If the size of the edge is zero, then nothing has been set
        // if ((quads[i].info & (1 << edge_index)) == 0 &&
        //     (quads[i].info & (1 << (4 + edge_index))) == 0) {
        if (edge.size() == 0) {
          edge.reserve(4);
          add_edge_stencil(quad, edge_index, node_map, edge);
        }
      }
    }
  }

  void find_node(const QuadrantNode& node, const QuadrantNode& dir,
                 const std::vector<int>& node_map, int& idx, int& step) {
    idx = -1;
    step = 0;

    for (int k = 0; k < 3; k++) {
      QuadrantNode n;
      n.x = node.x + (1 << k) * dir.x;
      n.y = node.y + (1 << k) * dir.y;

      int index = get_mapped_node_index(n, node_map);

      // Success
      if (index >= 0) {
        idx = index;
        step = 1 << k;
        break;
      }
    }
  }

  void add_edge_stencil(const Quadrant& quad, int edge_index,
                        const std::vector<int>& node_map,
                        std::vector<int>& edge) {
    // Half the side length to look for
    std::int64_t h = std::int64_t(1) << (Quadrant::MAX_LEVEL - quad.level);
    std::int64_t hd = h / 2;

    // The corner nodes and directions
    QuadrantNode n0, n1;
    QuadrantNode d0, d1;

    // Set the first corner node
    if (edge_index < 2) {
      n0.x = quad.x + h * (edge_index % 2);
      n0.y = quad.y;
      d0.x = 0;
      d0.y = -hd;

      n1.x = quad.x + h * (edge_index % 2);
      n1.y = quad.y + h;
      d1.x = 0;
      d1.y = hd;
    } else {
      n0.x = quad.x;
      n0.y = quad.y + h * (edge_index % 2);
      d0.x = -hd;
      d0.y = 0;

      n1.x = quad.x + h;
      n1.y = quad.y + h * (edge_index % 2);
      d1.x = hd;
      d1.y = 0;
    }

    // Add the indices from the corner nodes
    int idx0 = get_mapped_node_index(n0, node_map);
    if (idx0 >= 0) {
      edge.push_back(idx0);
    }

    int idx1 = get_mapped_node_index(n1, node_map);
    if (idx1 >= 0) {
      edge.push_back(idx1);
    }

    int s0;
    find_node(n0, d0, node_map, idx0, s0);
    if (idx0 >= 0) {
      edge.push_back(idx0);
    }

    int s1;
    find_node(n1, d1, node_map, idx1, s1);
    if (idx1 >= 0) {
      edge.push_back(idx1);
    }

    // Both steps were successful, no further action required
    if (idx0 >= 0 && idx1 >= 0) {
      return;
    }

    // One step was successful and the other one wasn't. Pick an additional
    // step along the successful direction
    if ((idx0 >= 0) || (idx1 >= 0)) {
      QuadrantNode n, d;
      int idx, s;

      if (idx0 >= 0) {
        n = n0;
        d = d0;
        s = s0;
      } else {
        n = n1;
        d = d1;
        s = s1;
      }

      n.x += s * d.x;
      n.y += s * d.y;

      find_node(n, d, node_map, idx, s);
      if (idx >= 0) {
        edge.push_back(idx);
      }
    }
  }

  void build_element_stencil(StencilInfo& info) {
    const auto& edge_stencil = info.edge_stencil;
    auto& stencil = info.stencil;

    for (std::size_t elem = 0; elem < edge_stencil.size(); ++elem) {
      std::vector<int>& elem_stencil = stencil[elem];
      elem_stencil.clear();

      // Reserve enough space to avoid repeated allocations
      std::size_t total_size = 0;
      for (const auto& edge : edge_stencil[elem]) {
        total_size += edge.size();
      }
      elem_stencil.reserve(total_size);

      // Merge all four edge stencils
      for (const auto& edge : edge_stencil[elem]) {
        for (int node : edge) {
          if (node >= 0) {
            elem_stencil.push_back(node);
          }
        }
      }

      // Sort and remove duplicate nodes shared by adjacent edges
      std::sort(elem_stencil.begin(), elem_stencil.end());
      elem_stencil.erase(std::unique(elem_stencil.begin(), elem_stencil.end()),
                         elem_stencil.end());
    }
  }

  void update_stencil(StencilInfo& info, const std::vector<int>& node_map,
                      const std::vector<int>& elem_map) {
    int num_elements = elem_map.size();
    info.exclude.resize(num_elements);
    info.edge_stencil.resize(num_elements);
    info.stencil.resize(num_elements);
    info.X.resize(num_elements);

    for (int i = 0; i < num_elements; i++) {
      for (int j = 0; j < 4; j++) {
        info.edge_stencil[i][j].clear();
      }
    }

    build_hanging_edge_stencil(info, node_map, elem_map);
    build_regular_edge_stencil(info, node_map, elem_map);

    // Based on the edge stencils, build the full stencil for an element
    build_element_stencil(info);

    // Set the node locations for each element
    const T length = mesh->get_length();
    NodeArray& node_array = *mesh->get_node_array();

    // Set up an inverse node mapping
    std::vector<int> inv_node_map(num_interior_nodes + num_exterior_nodes, -1);
    for (int i = 0; i < node_map.size(); i++) {
      if (node_map[i] >= 0) {
        inv_node_map[node_map[i]] = i;
      }
    }

    constexpr std::int64_t hmax = std::int64_t(1) << Quadrant::MAX_LEVEL;

    for (int index = 0; index < num_elements; index++) {
      info.X[index].resize(2 * info.stencil[index].size());

      for (int k = 0; k < info.stencil[index].size(); k++) {
        int node_index = info.stencil[index][k];
        int base_node = inv_node_map[node_index];
        if (base_node < 0) {
          throw std::runtime_error("Stencil failure: Use of un-mapped node");
        }

        info.X[index][2 * k] = length * node_array[base_node].x / hmax;
        info.X[index][2 * k + 1] = length * node_array[base_node].y / hmax;
      }
    }

    // Now compute the exclusion for each basis
    QuadrantArray& quads = *mesh->get_quadrants();

    info.exclude.resize(num_elements);
    for (int index = 0; index < num_elements; index++) {
      int elem = elem_map[index];

      T delta = length * quads[elem].get_size() / hmax;
      T x0 = length * quads[elem].x / hmax;
      T y0 = length * quads[elem].y / hmax;
      const T* Xelem = info.X[index].data();

      int num_points = static_cast<int>(info.stencil[index].size());

      info.exclude[index] =
          compute_exclude_bits_from_points(num_points, x0, y0, delta, Xelem);
    }
  }

  // The underlying Cartesian mesh that defines the level set
  std::shared_ptr<QuadtreeMesh<T>> mesh;
  std::shared_ptr<QuadtreeMesh<T>> lsf_mesh;

  // Mapping from the quadtree elements to the level set elements
  std::vector<int> elem_to_lsf;

  // The level set values associated with the level set function
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

  // Store the quadratures for the interface elements
  std::vector<std::vector<T>> interior_points;
  std::vector<std::vector<T>> interior_weights;
  std::vector<std::vector<T>> exterior_points;
  std::vector<std::vector<T>> exterior_weights;
  std::vector<std::vector<T>> interface_points;
  std::vector<std::vector<T>> interface_weights;
  std::vector<std::vector<T>> interface_normals;

  // Store flattened Jacobians of the interface quadratures
  std::vector<std::vector<T>> interior_points_jacobian;
  std::vector<std::vector<T>> interior_weights_jacobian;
  std::vector<std::vector<T>> exterior_points_jacobian;
  std::vector<std::vector<T>> exterior_weights_jacobian;
  std::vector<std::vector<T>> interface_points_jacobian;
  std::vector<std::vector<T>> interface_weights_jacobian;
  std::vector<std::vector<T>> interface_normals_jacobian;
};

template <typename T>
class CutQuadMeshComponent final : public MeshBase<T> {
 public:
  CutQuadMeshComponent(std::shared_ptr<QuadtreeCutMesh<T>> mesh,
                       CutDomain domain)
      : mesh(std::move(mesh)), domain(domain) {}

  int get_max_node_index() const override {
    return mesh->get_max_node_index(domain);
  }

  int get_max_element_nodes() const override {
    return mesh->get_max_element_nodes(domain);
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

  void get_points(int elem, std::vector<T>& X) const override {
    mesh->get_points(domain, elem, X);
  }

  int get_quadrature(int elem, std::vector<T>& weights, std::vector<T>& points,
                     std::vector<T>& normals) const override {
    return mesh->get_quadrature(domain, elem, weights, points, normals);
  }

  void eval_basis(int elem, int num_quad_points, const std::vector<T>& pts,
                  std::vector<T>& Nd) const override {
    mesh->eval_basis(domain, elem, num_quad_points, pts, Nd);
  }

  int get_max_design_index() const override {
    return mesh->get_max_design_index();
  }
  int get_max_element_design_vars() const override {
    return mesh->get_max_element_design_vars();
  }
  int get_quadrature_derivative(int elem, std::vector<T>& dwdx,
                                std::vector<T>& dpdx, std::vector<T>& dndx,
                                int& ndvs,
                                std::vector<int>& dvs) const override {
    return mesh->get_quadrature_derivative(domain, elem, dwdx, dpdx, dndx, ndvs,
                                           dvs);
  }
  void reverse_eval_basis(int elem, int num_quad_points,
                          const std::vector<T>& pts, const std::vector<T>& bNd,
                          std::vector<T>& bpts) const override {
    mesh->reverse_eval_basis(domain, elem, num_quad_points, pts, bNd, bpts);
  }

 private:
  std::shared_ptr<QuadtreeCutMesh<T>> mesh;
  CutDomain domain;
};

}  // namespace xcgd

#endif  // XCGD_CUT_QUADTREE_MESH_H
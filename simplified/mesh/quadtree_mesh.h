#ifndef XCGD_QUADTREE_MESH_H
#define XCGD_QUADTREE_MESH_H

#include <memory>

#include "cartesian_mesh.h"
#include "quadtree.h"
#include "vandermonde.h"

namespace xcgd {

template <typename T>
class QuadtreeMesh : public MeshBase<T> {
 public:
  static constexpr int mesh_degree = 1;
  static constexpr int degree = 3;
  static constexpr std::int64_t hmax =
      mesh_degree * (std::int64_t(1) << Quadrant::MAX_LEVEL);

  QuadtreeMesh(std::shared_ptr<Quadtree> tree, T length)
      : tree(tree), length(length) {
    // Create the connectivity on the underlying quadtree
    nodes = tree->create_nodes(mesh_degree);

    // Set the quadratures
    p3[0] = -sqrt(3.0 / 5.0);
    p3[1] = 0.0;
    p3[2] = sqrt(3.0 / 5.0);

    w3[0] = 5.0 / 9.0;
    w3[1] = 8.0 / 9.0;
    w3[0] = 5.0 / 9.0;

    p4[0] = -sqrt((3.0 + 2.0 * sqrt(6.0 / 5.0)) / 7.0);
    p4[1] = -sqrt((3.0 - 2.0 * sqrt(6.0 / 5.0)) / 7.0);
    p4[2] = sqrt((3.0 - 2.0 * sqrt(6.0 / 5.0)) / 7.0);
    p4[3] = sqrt((3.0 + 2.0 * sqrt(6.0 / 5.0)) / 7.0);

    w4[0] = (18.0 - sqrt(30.0)) / 36.0;
    w4[1] = (18.0 + sqrt(30.0)) / 36.0;
    w4[2] = (18.0 + sqrt(30.0)) / 36.0;
    w4[3] = (18.0 - sqrt(30.0)) / 36.0;

    // Set up everything
    update();
  }

  void update() {
    QuadrantArray& quads = *tree->get_quadrants();
    int num_elements = quads.size();

    // Build the stencil for each edge
    edge_stencil.resize(num_elements);
    build_hanging_edge_stencil();
    build_regular_edge_stencil();

    // Based on the edge stencils, build the full stencil for an element
    stencil.resize(num_elements);
    build_element_stencil();

    // Set the node locations for each element
    NodeArray& node_array = *nodes;
    X.resize(num_elements);

    for (int elem = 0; elem < num_elements; elem++) {
      X[elem].resize(2 * stencil[elem].size());

      for (int k = 0; k < stencil[elem].size(); k++) {
        int index = stencil[elem][k];
        X[elem][2 * k] = length * node_array[index].x / hmax;
        X[elem][2 * k + 1] = length * node_array[index].y / hmax;
      }
    }

    // Now compute the exclusion for each basis
    exclude.resize(num_elements);
    for (int elem = 0; elem < num_elements; elem++) {
      T delta = length * quads[elem].get_size() / hmax;
      T x0 = length * quads[elem].x / hmax;
      T y0 = length * quads[elem].y / hmax;
      const T* Xelem = X[elem].data();

      int num_points = static_cast<int>(stencil[elem].size());

      exclude[elem] =
          compute_exclude_bits_from_points(num_points, x0, y0, delta, Xelem);
    }
  }

  int get_max_node_index() const { return nodes->size(); }
  int get_num_elements() const { return tree->size(); }
  int get_max_element_nodes() const { return detail::PolyBasis2D::MAX_BASIS; }
  int get_max_num_quadrature_points() const { return 16; }

  // Element-wise access functions
  int get_nodes(int elem, std::vector<int>& nodes) const {
    nodes.assign(stencil[elem].begin(), stencil[elem].end());
    return stencil[elem].size();
  }

  void get_points(int elem, std::vector<T>& Xelem) const {
    Xelem.assign(X[elem].begin(), X[elem].end());
  }

  int get_quadrature(int elem, std::vector<T>& weights, std::vector<T>& points,
                     std::vector<T>& normals) const {
    // Set the delta, x and y locations
    QuadrantArray& quads = *tree->get_quadrants();
    T delta = length * quads[elem].get_size() / hmax;
    T x0 = length * quads[elem].x / hmax;
    T y0 = length * quads[elem].y / hmax;

    for (int jj = 0; jj < 4; jj++) {
      for (int ii = 0; ii < 4; ii++) {
        int k = ii + 4 * jj;
        weights[k] = T(0.25) * delta * delta * w4[ii] * w4[jj];
        points[2 * k] = x0 + delta * (p4[ii] + T(1)) / T(2);
        points[2 * k + 1] = y0 + delta * (p4[jj] + T(1)) / T(2);
        normals[2 * k] = normals[2 * k + 1] = 0.0;
      }
    }

    return 16;
  }

  void eval_basis(int elem, int num_quad_points, const std::vector<T>& pts,
                  std::vector<T>& Nd) const {
    // Set the delta, x and y locations
    QuadrantArray& quads = *tree->get_quadrants();
    T delta = length * quads[elem].get_size() / hmax;
    T x0 = length * quads[elem].x / hmax;
    T y0 = length * quads[elem].y / hmax;

    int nnodes = stencil[elem].size();
    const T* Xelem = X[elem].data();

    Vandermonde2D interp(x0, y0, delta, nnodes, Xelem,
                         detail::PolyBasis2D(exclude[elem]),
                         detail::PolyBasisDeriv2D(exclude[elem]));

    interp.eval_basis(num_quad_points, pts.data(), Nd.data());
  }

 private:
  void build_hanging_edge_stencil() {
    QuadrantArray& quads = *tree->get_quadrants();

    const int edge_index_to_adjacent[] = {1, 0, 3, 2};

    for (int i = 0; i < quads.size(); i++) {
      // Check if the info flag is set
      if (quads[i].info) {
        // Check if we have a dependent edge from a coarse element
        for (int edge_index = 0; edge_index < 4; edge_index++) {
          if (quads[i].info & (1 << (4 + edge_index))) {
            edge_stencil[i][edge_index].reserve(5);

            // Find the hanging node index
            std::int32_t h = quads[i].get_size();
            std::int32_t hd = h / 2;

            QuadrantNode node;
            if (edge_index < 2) {
              node.x = quads[i].x + h * (edge_index % 2);
              node.y = quads[i].y + hd;
            } else {
              node.x = quads[i].x + hd;
              node.y = quads[i].y + h * (edge_index % 2);
            }

            // Get the node index
            int index = nodes->get_index(node);

            // This is a hanging node edge
            if (index >= 0) {
              edge_stencil[i][edge_index].push_back(index);
            }

            // Build the remaining stencil along the edge
            add_edge_stencil(quads[i], edge_index, edge_stencil[i][edge_index]);
          }
        }
      }
    }

    // The remaining dependent edges can be copied from the others
    for (int i = 0; i < quads.size(); i++) {
      if (quads[i].info) {
        for (int edge_index = 0; edge_index < 4; edge_index++) {
          if (quads[i].info & (1 << edge_index)) {
            Quadrant p = quads[i].parent();
            p = p.edge_neighbor(edge_index);

            int element = quads.get_index(p);
            int adj_edge = edge_index_to_adjacent[edge_index];

            // Copy the contents to the other edge
            edge_stencil[i][edge_index] = edge_stencil[element][adj_edge];
          }
        }
      }
    }
  }

  void build_regular_edge_stencil() {
    QuadrantArray& quads = *tree->get_quadrants();

    for (int i = 0; i < quads.size(); i++) {
      // Check that no info is set for this edge
      for (int edge_index = 0; edge_index < 4; edge_index++) {
        if ((quads[i].info & (1 << edge_index)) == 0 &&
            (quads[i].info & (1 << (4 + edge_index))) == 0) {
          edge_stencil[i][edge_index].reserve(4);
          add_edge_stencil(quads[i], edge_index, edge_stencil[i][edge_index]);
        }
      }
    }
  }

  void find_node(const QuadrantNode& node, const QuadrantNode& dir, int& idx,
                 int& step) {
    idx = -1;
    step = 0;

    for (int k = 0; k < 3; k++) {
      QuadrantNode n;
      n.x = node.x + (1 << k) * dir.x;
      n.y = node.y + (1 << k) * dir.y;

      int index = nodes->get_index(n);

      // Success
      if (index >= 0) {
        idx = index;
        step = 1 << k;
        break;
      }
    }
  }

  void add_edge_stencil(const Quadrant& quad, int edge_index,
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
    int idx0 = nodes->get_index(n0);
    if (idx0 >= 0) {
      edge.push_back(idx0);
    }

    int idx1 = nodes->get_index(n1);
    if (idx1 >= 0) {
      edge.push_back(idx1);
    }

    int s0;
    find_node(n0, d0, idx0, s0);
    if (idx0 >= 0) {
      edge.push_back(idx0);
    }

    int s1;
    find_node(n1, d1, idx1, s1);
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

      find_node(n, d, idx, s);
      if (idx >= 0) {
        edge.push_back(idx);
      }
    }
  }

  void build_element_stencil() {
    for (std::size_t elem = 0; elem < edge_stencil.size(); ++elem) {
      auto& elem_stencil = stencil[elem];
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

  std::shared_ptr<Quadtree> tree;
  T length;  // Conversion factor for quadtree units to position units

  // Nodes for the quadtree
  std::shared_ptr<NodeArray> nodes;

  // The stencil of nodes for each element
  std::vector<std::array<std::vector<int>, 4>> edge_stencil;
  std::vector<std::vector<int>> stencil;

  // The node locations for each element
  std::vector<std::vector<T>> X;

  // The exclusion for each element
  std::vector<uint32_t> exclude;

  T p3[3], w3[3];
  T p4[4], w4[4];
};

}  // namespace xcgd

#endif  // XCGD_QUADTREE_MESH_H
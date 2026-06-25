#ifndef XCGD_CARTESIAN_MESH_H
#define XCGD_CARTESIAN_MESH_H

#include <vector>

#include "basis.h"
#include "mesh_base.h"
#include "vandermonde.h"

namespace xcgd {

enum class EdgeSide { X_NEG = 0, X_POS = 1, Y_NEG = 2, Y_POS = 3 };

struct EdgeLine {
  int ix0, iy0;
  int dix, diy;
};

template <typename T>
class CartesianMesh : public MeshBase<T> {
 public:
  static constexpr int degree = 3;

  CartesianMesh(int nx, int ny, T delta) : nx(nx), ny(ny), delta(delta) {
    if (nx < 3 || ny < 3) {
      throw std::runtime_error("Not enough elements: nx < 3 || ny < 3");
    }

    p[0] = -sqrt((3.0 + 2.0 * sqrt(6.0 / 5.0)) / 7.0);
    p[1] = -sqrt((3.0 - 2.0 * sqrt(6.0 / 5.0)) / 7.0);
    p[2] = sqrt((3.0 - 2.0 * sqrt(6.0 / 5.0)) / 7.0);
    p[3] = sqrt((3.0 + 2.0 * sqrt(6.0 / 5.0)) / 7.0);

    w[0] = (18.0 - sqrt(30.0)) / 36.0;
    w[1] = (18.0 + sqrt(30.0)) / 36.0;
    w[2] = (18.0 + sqrt(30.0)) / 36.0;
    w[3] = (18.0 - sqrt(30.0)) / 36.0;
  }

  // Get the node number
  int get_node(int i, int j) const {
    if (i < 0 || i > nx || j < 0 || j > ny) {
      return -1;
    }
    return i + j * (nx + 1);
  }

  void get_node_location(int node, T& x, T& y) const {
    int i = node % (nx + 1);
    int j = node / (nx + 1);

    x = delta * i;
    y = delta * j;
  }

  // Get an edge line - (i, j) of the node and the direction (di, dj)
  EdgeLine get_edge_line(int elem, EdgeSide edge) const {
    int i = elem % nx;
    int j = elem / nx;

    if (edge == EdgeSide::X_NEG) {
      return EdgeLine{i, j, 0, 1};
    } else if (edge == EdgeSide::X_POS) {
      return EdgeLine{i + 1, j, 0, 1};
    } else if (edge == EdgeSide::Y_NEG) {
      return EdgeLine{i, j, 1, 0};
    } else {
      return EdgeLine{i, j + 1, 1, 0};
    }
  }

  void get_cell_nodes(int elem, std::vector<int>& nodes) const {
    int i = elem % nx;
    int j = elem / nx;

    nodes[0] = i + j * (nx + 1);
    nodes[1] = i + 1 + j * (nx + 1);
    nodes[2] = i + (j + 1) * (nx + 1);
    nodes[3] = i + 1 + (j + 1) * (nx + 1);
  }

  // Overrides needed to use this as an analysis mesh directly
  int get_max_node_index() const { return (nx + 1) * (ny + 1); }
  int get_max_num_quadrature_points() const {
    return (degree + 1) * (degree + 1);
  }
  int get_max_num_nodes() const { return (degree + 1) * (degree + 1); }
  int get_num_elements() const { return nx * ny; }

  int get_nodes(int elem, std::vector<int>& nodes) const {
    return get_node_numbers(elem, nodes);
  }
  void get_node_points(int elem, std::vector<T>& X) const {
    get_point_locations(elem, X);
  }

  int get_quadrature(int elem, std::vector<T>& weights, std::vector<T>& points,
                     std::vector<T>& normals) const {
    int i = elem % nx;
    int j = elem / nx;
    T x0 = delta * i;
    T y0 = delta * j;

    for (int jj = 0; jj < 4; jj++) {
      for (int ii = 0; ii < 4; ii++) {
        int k = ii + 4 * jj;
        weights[k] = T(0.25) * delta * delta * w[ii] * w[jj];
        points[2 * k] = x0 + delta * (p[ii] + T(1)) / T(2);
        points[2 * k + 1] = y0 + delta * (p[jj] + T(1)) / T(2);
        normals[2 * k] = normals[2 * k + 1] = 0.0;
      }
    }

    return 16;
  }

  void eval_basis(int elem, int num_quad_points, const std::vector<T>& pts,
                  std::vector<T>& Nd) const {
    T X[24];  // nnodes = 12 always for this mesh
    int nnodes = get_point_locations(elem, X);

    int i = elem % nx;
    int j = elem / nx;
    T x0 = delta * i;
    T y0 = delta * j;

    Vandermonde2D interp(x0, y0, delta, nnodes, X, detail::RegularPolyBasis2D{},
                         detail::RegularPolyBasisDeriv2D{});

    interp.eval_basis(num_quad_points, pts.data(), Nd.data());
  }

  auto create_interp(int elem) const {
    T X[24];  // nnodes = 12 always for this mesh
    int nnodes = get_point_locations(elem, X);

    int i = elem % nx;
    int j = elem / nx;
    T x0 = delta * i;
    T y0 = delta * j;

    return Vandermonde2D(x0, y0, delta, nnodes, X, detail::RegularPolyBasis2D{},
                         detail::RegularPolyBasisDeriv2D{});
  }

  void get_element_base_point(int elem, T& x0, T& y0, T& delta_) const {
    int i = elem % nx;
    int j = elem / nx;
    x0 = delta * i;
    y0 = delta * j;
    delta_ = delta;
  }

 private:
  template <class ArrayType>
  int get_point_locations(int elem, ArrayType& X) const {
    int nodes[12];
    int nnodes = get_node_numbers(elem, nodes);

    for (int k = 0; k < nnodes; k++) {
      int i = nodes[k] % (nx + 1);
      int j = nodes[k] / (nx + 1);

      X[2 * k] = i * delta;
      X[2 * k + 1] = j * delta;
    }

    return nnodes;
  }

  template <class ArrayType>
  int get_node_numbers(int elem, ArrayType& array) const {
    int i = elem % nx;
    int j = elem / nx;

    int istart = i - 1;
    int iend = i + 2;
    int jstart = j - 1;
    int jend = j + 2;

    if (i == 0) {
      istart = 0;
      iend = degree;
    } else if (i == nx - 1) {
      istart = nx - 3;
      iend = nx;
    }

    if (j == 0) {
      jstart = 0;
      jend = degree;
    } else if (j == ny - 1) {
      jstart = ny - 3;
      jend = ny;
    }

    // Fill in the node numbers based on the locations
    int count = 0;
    for (int ii = istart; ii <= iend; ii++) {
      for (int jj = j; jj <= j + 1; jj++, count++) {
        array[count] = ii + jj * (nx + 1);
      }
    }

    for (int jj = jstart; jj <= jend; jj++) {
      if (jj == j || jj == j + 1) {
        continue;
      }
      for (int ii = i; ii <= i + 1; ii++, count++) {
        array[count] = ii + jj * (nx + 1);
      }
    }

    return 12;
  }

 private:
  int nx, ny;
  T delta;

  T p[4], w[4];
};

}  // namespace xcgd

#endif  // XCGD_MESH_BASE_H
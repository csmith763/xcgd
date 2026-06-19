#ifndef XCGD_MESH_BASE_H
#define XCGD_MESH_BASE_H

#include <vector>

#include "vandermonde.h"

namespace xcgd {

template <typename T>
class MeshBase {
 public:
  virtual ~MeshBase() = default;

  virtual int get_max_node_index() const = 0;
  virtual int get_max_num_nodes() const = 0;
  virtual int get_max_num_quadrature_points() const = 0;
  virtual int get_num_elements() const = 0;

  virtual int get_nodes(int elem, std::vector<int>& nodes) = 0;
  virtual void get_node_points(int elem, std::vector<T>& X) const = 0;
  virtual int get_quadrature(int elem, std::vector<T>& weights,
                             std::vector<T>& points,
                             std::vector<T>& normals) const = 0;
  virtual void eval_basis(int elem, int num_quad_points,
                          const std::vector<T>& pts, std::vector<T>& N,
                          std::vector<T>& Nx) const = 0;
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

  int get_max_node_index() const { return (nx + 1) * (ny + 1) - 1; }
  int get_max_num_quadrature_points() const {
    return (degree + 1) * (degree + 1);
  }
  int get_max_num_nodes() const { return (degree + 1) * (degree + 1); }
  int get_num_elements() const { return nx * ny; }

  int get_nodes(int elem, std::vector<int>& nodes) {}
  void get_node_points(int elem, std::vector<T>& X) const {
    get_point_locations(elem, X);
  }

  int get_quadrature(int elem, std::vector<T>& weights, std::vector<T>& points,
                     std::vector<T>& normals) const {
    for (int j = 0; j < 4; j++) {
      for (int i = 0; i < 4; i++) {
        int k = i + 4 * j;
        weights[k] = delta * delta * w[i] * w[j];
        points[2 * k] = p[i];
        points[2 * k + 1] = p[j];
        normals[2 * k] = normals[2 * k + 1] = 0.0;
      }
    }

    return 16;
  }

  void eval_basis(int elem, int num_quad_points, const std::vector<T>& pts,
                  std::vector<T>& N, std::vector<T>& Nx) const {
    T X[24];  // nnodes = 12 always for this mesh
    int nnodes = get_point_locations(elem, X);

    int i = elem % nx;
    int j = elem / nx;

    T x0 = delta * i;
    T y0 = delta * j;

    auto basis = [&](T x, T y, T p[]) {
      T xp[4], yp[4];
      xp[0] = 1.0;
      xp[1] = x;
      xp[2] = x * xp[1];
      xp[3] = x * xp[2];
      yp[0] = 1.0;
      yp[1] = y;
      yp[2] = y * yp[1];
      yp[3] = y * yp[2];

      p[0] = 1.0;
      p[1] = xp[1];
      p[2] = yp[1];
      p[3] = xp[2];
      p[4] = xp[1] * yp[1];
      p[5] = yp[2];
      p[6] = xp[3];
      p[7] = xp[2] * yp[1];
      p[8] = xp[1] * yp[2];
      p[9] = yp[3];
      p[10] = xp[3] * yp[1];
      p[11] = xp[1] * yp[3];
    };
    auto basis_deriv = [&](T x, T y, T dx[], T dy[]) {
      T xp[4], yp[4];
      xp[0] = 1.0;
      xp[1] = x;
      xp[2] = x * xp[1];
      xp[3] = x * xp[2];
      yp[0] = 1.0;
      yp[1] = y;
      yp[2] = y * yp[1];
      yp[3] = y * yp[2];

      dx[0] = 0.0;
      dx[1] = 1.0;
      dx[2] = 0.0;
      dx[3] = 2 * xp[1];
      dx[4] = yp[1];
      dx[5] = 0.0;
      dx[6] = 3 * xp[2];
      dx[7] = 2 * xp[1] * yp[1];
      dx[8] = yp[2];
      dx[9] = 0.0;
      dx[10] = 3 * xp[2] * yp[1];
      dx[11] = yp[3];

      dy[0] = 0.0;
      dy[1] = 0.0;
      dy[2] = 1.0;
      dy[3] = 0.0;
      dy[4] = xp[1];
      dy[5] = 2 * yp[1];
      dy[6] = 0.0;
      dy[7] = xp[2];
      dy[8] = 2 * xp[1] * yp[1];
      dy[9] = 3 * yp[2];
      dy[10] = xp[3];
      dy[11] = 3 * xp[1] * yp[2];
    };

    Vandermonde2D interp(x0, y0, delta, nnodes, X, basis, basis_deriv);

    interp.eval(num_quad_points, pts.data(), N.data(), Nx.data());
  }

 private:
  template <class ArrayType>
  int get_node_numbers(int elem, ArrayType array) const {
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
      istart = nx - 2;
      iend = nx + 1;
    }

    if (j == 0) {
      jstart = 0;
      jend = degree;
    } else if (j == ny - 1) {
      jstart = ny - 2;
      jend = ny + 1;
    }

    // Fill in the node numbers based on the locations
    for (int ii = istart; ii <= iend; ii++) {
    }

    return 12;
  }

  template <class ArrayType>
  int get_point_locations(int elem, ArrayType X) const {
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

 private:
  int nx, ny;
  T delta;

  T p[4], w[4];
};

}  // namespace xcgd

#endif  // XCGD_MESH_BASE_H
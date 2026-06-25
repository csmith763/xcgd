
#ifndef XCGD_BASIS_H
#define XCGD_BASIS_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace xcgd {

namespace detail {

/**
 * @brief This is a basis for a regular set of points using the GD method
 *
 */
class RegularPolyBasis2D {
 public:
  template <class T>
  void operator()(T x, T y, T p[]) const {
    T xp[4], yp[4];
    xp[0] = T(1);
    xp[1] = x;
    xp[2] = x * xp[1];
    xp[3] = x * xp[2];
    yp[0] = T(1);
    yp[1] = y;
    yp[2] = y * yp[1];
    yp[3] = y * yp[2];

    p[0] = T(1);
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
  }
};

class RegularPolyBasisDeriv2D {
 public:
  template <class T>
  void operator()(T x, T y, T p[], T dx[], T dy[]) const {
    T xp[4], yp[4];
    xp[0] = T(1);
    xp[1] = x;
    xp[2] = x * xp[1];
    xp[3] = x * xp[2];
    yp[0] = T(1);
    yp[1] = y;
    yp[2] = y * yp[1];
    yp[3] = y * yp[2];

    p[0] = T(1);
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

    dx[0] = T(0);
    dx[1] = T(1);
    dx[2] = T(0);
    dx[3] = T(2) * xp[1];
    dx[4] = yp[1];
    dx[5] = T(0);
    dx[6] = T(3) * xp[2];
    dx[7] = T(2) * xp[1] * yp[1];
    dx[8] = yp[2];
    dx[9] = T(0);
    dx[10] = T(3) * xp[2] * yp[1];
    dx[11] = yp[3];

    dy[0] = T(0);
    dy[1] = T(0);
    dy[2] = T(1);
    dy[3] = T(0);
    dy[4] = xp[1];
    dy[5] = T(2) * yp[1];
    dy[6] = T(0);
    dy[7] = xp[2];
    dy[8] = T(2) * xp[1] * yp[1];
    dy[9] = T(3) * yp[2];
    dy[10] = xp[3];
    dy[11] = T(3) * xp[1] * yp[2];
  }
};

class PolyBasis2D {
 public:
  PolyBasis2D(uint32_t exclude = uint32_t(0)) : exclude(exclude) {}

  static constexpr int MAX_BASIS = 18;
  uint32_t exclude;

  template <class T>
  int operator()(T x, T y, T poly[]) const {
    T p[MAX_BASIS];
    T xp[5], yp[5];
    xp[0] = T(1);
    for (int i = 1; i < 5; i++) {
      xp[i] = x * xp[i - 1];
    }
    yp[0] = T(1);
    for (int i = 1; i < 5; i++) {
      yp[i] = y * yp[i - 1];
    }

    // Const/linear
    p[0] = T(1);
    p[1] = xp[1];
    p[2] = yp[1];

    // Quadratic
    p[3] = xp[2];
    p[4] = yp[2];
    p[5] = xp[1] * yp[1];

    // Cubic
    p[6] = xp[3];
    p[7] = yp[3];
    p[8] = xp[2] * yp[1];
    p[9] = xp[1] * yp[2];

    // Quartic
    p[10] = xp[4];
    p[11] = yp[4];
    p[12] = xp[1] * yp[3];
    p[13] = xp[2] * yp[2];
    p[14] = xp[3] * yp[1];

    // Remaining terms in Q3
    p[15] = xp[3] * yp[2];
    p[16] = xp[2] * yp[3];
    p[17] = xp[3] * yp[3];

    // Include basis values where the exclude bit is zero
    int counter = 0;
    for (int i = 0; i < MAX_BASIS; i++) {
      if ((exclude & (uint32_t(1) << i)) == 0) {
        poly[counter] = p[i];
        counter++;
      }
    }

    return counter;
  }
};

class PolyBasisDeriv2D {
 public:
  PolyBasisDeriv2D(uint32_t exclude = uint32_t(0)) : exclude(exclude) {}

  static constexpr int MAX_BASIS = 18;
  uint32_t exclude;

  template <class T>
  int operator()(T x, T y, T poly[], T dx[], T dy[]) const {
    T p[MAX_BASIS], px[MAX_BASIS], py[MAX_BASIS];
    T xp[5], yp[5];
    T dxp[5], dyp[5];

    xp[0] = T(1);
    for (int i = 1; i < 5; i++) {
      xp[i] = x * xp[i - 1];
    }
    yp[0] = T(1);
    for (int i = 1; i < 5; i++) {
      yp[i] = y * yp[i - 1];
    }

    dxp[0] = T(0);
    for (int i = 1; i < 5; i++) {
      dxp[i] = T(i) * xp[i - 1];
    }
    dyp[0] = T(0);
    for (int i = 1; i < 5; i++) {
      dyp[i] = T(i) * yp[i - 1];
    }

    // Const/linear
    p[0] = T(1);
    p[1] = xp[1];
    p[2] = yp[1];

    px[0] = T(0);
    px[1] = T(1);
    px[2] = T(0);

    py[0] = T(0);
    py[1] = T(0);
    py[2] = T(1);

    // Quadratic
    p[3] = xp[2];
    p[4] = yp[2];
    p[5] = xp[1] * yp[1];

    px[3] = dxp[2];
    px[4] = T(0);
    px[5] = yp[1];

    py[3] = T(0);
    py[4] = dyp[2];
    py[5] = xp[1];

    // Cubic
    p[6] = xp[3];
    p[7] = yp[3];
    p[8] = xp[2] * yp[1];
    p[9] = xp[1] * yp[2];

    px[6] = dxp[3];
    px[7] = T(0);
    px[8] = dxp[2] * yp[1];
    px[9] = yp[2];

    py[6] = T(0);
    py[7] = dyp[3];
    py[8] = xp[2];
    py[9] = xp[1] * dyp[2];

    // Quartic
    p[10] = xp[4];
    p[11] = yp[4];
    p[12] = xp[1] * yp[3];
    p[13] = xp[2] * yp[2];
    p[14] = xp[3] * yp[1];

    px[10] = dxp[4];
    px[11] = T(0);
    px[12] = yp[3];
    px[13] = dxp[2] * yp[2];
    px[14] = dxp[3] * yp[1];

    py[10] = T(0);
    py[11] = dyp[4];
    py[12] = xp[1] * dyp[3];
    py[13] = xp[2] * dyp[2];
    py[14] = xp[3];

    // Remaining terms in Q3
    p[15] = xp[3] * yp[2];
    p[16] = xp[2] * yp[3];
    p[17] = xp[3] * yp[3];

    px[15] = dxp[3] * yp[2];
    px[16] = dxp[2] * yp[3];
    px[17] = dxp[3] * yp[3];

    py[15] = xp[3] * dyp[2];
    py[16] = xp[2] * dyp[3];
    py[17] = xp[3] * dyp[3];

    // Include basis values where the exclude bit is zero
    int counter = 0;
    for (int i = 0; i < MAX_BASIS; i++) {
      if ((exclude & (uint32_t(1) << i)) == 0) {
        poly[counter] = p[i];
        dx[counter] = px[i];
        dy[counter] = py[i];
        counter++;
      }
    }

    return counter;
  }
};

}  // namespace detail

template <typename T>
uint32_t compute_exclude_bits_from_points(int npts, T x0, T y0, T delta,
                                          const T* X, T rel_tol = 1e-12) {
  static constexpr int MAX_BASIS = detail::PolyBasis2D::MAX_BASIS;
  detail::PolyBasis2D full_basis;

  if (npts > MAX_BASIS) {
    throw std::runtime_error("More points than candidate basis functions");
  }

  // Build the full Vandermonde matrix with column-major storage
  //   V[a + npts * j] = phi_j(x_a, y_a)
  std::vector<T> V(npts * MAX_BASIS);
  std::vector<T> p(MAX_BASIS);
  std::vector<T> Q(npts * npts);
  std::vector<T> r(npts);

  for (int a = 0; a < npts; a++) {
    T x = (X[2 * a] - x0) / delta;
    T y = (X[2 * a + 1] - y0) / delta;

    // Copy over all of the basis functions
    int nbasis = full_basis(x, y, p.data());
    for (int j = 0; j < MAX_BASIS; j++) {
      V[a + npts * j] = p[j];
    }
  }

  // Q stores orthonormalized accepted columns.
  int rank = 0;

  // Start by excluding everything.
  uint32_t exclude = 0;
  for (int j = 0; j < MAX_BASIS; j++) {
    exclude |= (uint32_t(1) << j);
  }

  for (int j = 0; j < MAX_BASIS && rank < npts; j++) {
    // r = candidate column j
    T col_norm2 = 0.0;
    for (int a = 0; a < npts; a++) {
      r[a] = V[a + npts * j];
      col_norm2 += r[a] * r[a];
    }

    T col_norm = std::sqrt(col_norm2);
    if (col_norm == 0.0) {
      continue;
    }

    // Orthogonalize against previously accepted columns.
    for (int k = 0; k < rank; k++) {
      T dot = 0.0;
      for (int a = 0; a < npts; a++) {
        dot += Q[a + npts * k] * r[a];
      }

      for (int a = 0; a < npts; a++) {
        r[a] -= dot * Q[a + npts * k];
      }
    }

    // Reorthogonalize once for better numerical stability.
    for (int k = 0; k < rank; k++) {
      T dot = 0.0;
      for (int a = 0; a < npts; a++) {
        dot += Q[a + npts * k] * r[a];
      }

      for (int a = 0; a < npts; a++) {
        r[a] -= dot * Q[a + npts * k];
      }
    }

    T res_norm2 = 0.0;
    for (int a = 0; a < npts; a++) {
      res_norm2 += r[a] * r[a];
    }

    T res_norm = std::sqrt(res_norm2);

    // Accept this basis function if it increases rank.
    if (res_norm > rel_tol * std::max(1.0, col_norm)) {
      for (int a = 0; a < npts; a++) {
        Q[a + npts * rank] = r[a] / res_norm;
      }

      // Clear the bit, meaning "include this basis function".
      exclude &= ~(uint32_t(1) << j);

      rank++;
    }
  }

  if (rank != npts) {
    throw std::runtime_error(
        "Could not find enough independent basis functions for point cloud");
  }

  return exclude;
}

}  // namespace xcgd

#endif  // XCGD_BASIS_H
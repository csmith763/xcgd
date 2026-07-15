#ifndef XCGD_VANDERMONDE_H
#define XCGD_VANDERMONDE_H

#include <stdexcept>
#include <string>
#include <vector>

#if defined(__APPLE__)

// Use the Fortran LAPACK interface provided by Accelerate on macOS.
using lapack_int_t = int;

extern "C" {

void dgetrf_(const lapack_int_t* m, const lapack_int_t* n, double* a,
             const lapack_int_t* lda, lapack_int_t* ipiv, lapack_int_t* info);

void dgetrs_(const char* trans, const lapack_int_t* n, const lapack_int_t* nrhs,
             const double* a, const lapack_int_t* lda, const lapack_int_t* ipiv,
             double* b, const lapack_int_t* ldb, lapack_int_t* info);

void sgetrf_(const lapack_int_t* m, const lapack_int_t* n, float* a,
             const lapack_int_t* lda, lapack_int_t* ipiv, lapack_int_t* info);

void sgetrs_(const char* trans, const lapack_int_t* n, const lapack_int_t* nrhs,
             const float* a, const lapack_int_t* lda, const lapack_int_t* ipiv,
             float* b, const lapack_int_t* ldb, lapack_int_t* info);
}

#elif defined(__linux__)

// Use the LAPACKE C interface on Linux.
#include <lapacke.h>

using lapack_int_t = lapack_int;

#else
#error \
    "Unsupported platform: LAPACK interface is implemented only for macOS and Linux"
#endif

namespace xcgd {

namespace detail {

template <typename T>
struct Lapack;

template <>
struct Lapack<double> {
  using int_type = lapack_int_t;

  static void getrf(int_type n, double* A, int_type* ipiv) {
#if defined(__APPLE__)
    int_type info = 0;
    dgetrf_(&n, &n, A, &n, ipiv, &info);
#else
    const int_type info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, A, n, ipiv);
#endif

    check_getrf_info(info, "dgetrf");
  }

  static void getrs(char trans, int_type n, int_type nrhs, const double* LU,
                    const int_type* ipiv, double* B) {
#if defined(__APPLE__)
    int_type info = 0;
    dgetrs_(&trans, &n, &nrhs, LU, &n, ipiv, B, &n, &info);
#else
    const int_type info =
        LAPACKE_dgetrs(LAPACK_COL_MAJOR, trans, n, nrhs, LU, n, ipiv, B, n);
#endif

    check_getrs_info(info, "dgetrs");
  }

 private:
  static void check_getrf_info(int_type info, const char* name) {
    if (info < 0) {
      throw std::runtime_error(std::string(name) + ": illegal argument " +
                               std::to_string(-info));
    }

    if (info > 0) {
      throw std::runtime_error(
          std::string(name) + ": singular matrix; zero pivot at U(" +
          std::to_string(info) + ", " + std::to_string(info) + ")");
    }
  }

  static void check_getrs_info(int_type info, const char* name) {
    if (info < 0) {
      throw std::runtime_error(std::string(name) + ": illegal argument " +
                               std::to_string(-info));
    }

    if (info > 0) {
      throw std::runtime_error(std::string(name) +
                               ": unexpected positive info value " +
                               std::to_string(info));
    }
  }
};

template <>
struct Lapack<float> {
  using int_type = lapack_int_t;

  static void getrf(int_type n, float* A, int_type* ipiv) {
#if defined(__APPLE__)
    int_type info = 0;
    sgetrf_(&n, &n, A, &n, ipiv, &info);
#else
    const int_type info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, n, n, A, n, ipiv);
#endif

    check_getrf_info(info, "sgetrf");
  }

  static void getrs(char trans, int_type n, int_type nrhs, const float* LU,
                    const int_type* ipiv, float* B) {
#if defined(__APPLE__)
    int_type info = 0;
    sgetrs_(&trans, &n, &nrhs, LU, &n, ipiv, B, &n, &info);
#else
    const int_type info =
        LAPACKE_sgetrs(LAPACK_COL_MAJOR, trans, n, nrhs, LU, n, ipiv, B, n);
#endif

    check_getrs_info(info, "sgetrs");
  }

 private:
  static void check_getrf_info(int_type info, const char* name) {
    if (info < 0) {
      throw std::runtime_error(std::string(name) + ": illegal argument " +
                               std::to_string(-info));
    }

    if (info > 0) {
      throw std::runtime_error(
          std::string(name) + ": singular matrix; zero pivot at U(" +
          std::to_string(info) + ", " + std::to_string(info) + ")");
    }
  }

  static void check_getrs_info(int_type info, const char* name) {
    if (info < 0) {
      throw std::runtime_error(std::string(name) + ": illegal argument " +
                               std::to_string(-info));
    }

    if (info > 0) {
      throw std::runtime_error(std::string(name) +
                               ": unexpected positive info value " +
                               std::to_string(info));
    }
  }
};

}  // namespace detail

/**
 * @brief Lagrange interpolation using a Vandermonde-type matrix
 *
 * V = [ p(x1) ]     N = [ N1(x1) ]
 *     [ p(x2) ]         [ N2(x1) ]
 *     [   .   ]         [   .    ]
 *     [ p(xm) ]         [ Nm(x1) ]
 *
 * Nk(x) = p(x) * ck
 * C = [ c1 | c2 | . | ck ]
 *
 * Therefore V * C = I => C = V^{-1}
 *
 * N(x) = p(x) * C^{-1}
 * dN/dx = dp/dx * C^{-1}
 *
 */
template <typename T, class Basis, class BasisDeriv>
class Vandermonde2D {
 public:
  Vandermonde2D(T x0, T y0, T delta, int num_nodes, const T* X,
                const Basis& basis, const BasisDeriv& deriv)
      : x0(x0),
        y0(y0),
        delta(delta),
        num_nodes(num_nodes),
        basis(basis),
        deriv(deriv),
        V(num_nodes * num_nodes),
        ipiv(num_nodes) {
    // Build the Vandermonde matrix
    for (int i = 0; i < num_nodes; i++) {
      T x = (X[2 * i] - x0) / delta;
      T y = (X[2 * i + 1] - y0) / delta;

      // Evaluate the column of the Vandermonde matrix
      basis(x, y, &V[i * num_nodes]);
    }

    // Factor the matrix
    detail::Lapack<T>::getrf(num_nodes, V.data(), ipiv.data());
  }

  void get_base_data(T& x0_, T& y0_, T& delta_) const {
    x0_ = x0;
    y0_ = y0;
    delta_ = delta;
  }

  /**
   * @brief Evaluate the value of an interpolation at the given point
   *
   * @param pt The x and y location to evaluate
   * @param vals The value at the node locations
   * @return T
   */

  T eval(const T* pt, const T* vals) const {
    std::vector<T> N(num_nodes);

    T x = (pt[0] - x0) / delta;
    T y = (pt[1] - y0) / delta;

    basis(x, y, N.data());

    int nrhs = 1;
    detail::Lapack<T>::getrs('N', num_nodes, nrhs, V.data(), ipiv.data(),
                             N.data());

    T value = 0.0;
    for (int i = 0; i < num_nodes; i++) {
      value += N[i] * vals[i];
    }

    return value;
  }

  /**
   * @brief Return the basis functions and derivatives evaluated at all of the
   * specified points.
   *
   * @param num_points The number of point
   * @param pts The x and y locations of the 2D points
   * @param Nd The basis functions and their x and y derivatives at each point
   */
  void eval_basis(int num_points, const T* pts, T* Nd) const {
    const int block_size = 3 * num_nodes;
    const T inv = 1.0 / delta;
    for (int q = 0; q < num_points; q++) {
      const T x = (pts[2 * q] - x0) / delta;
      const T y = (pts[2 * q + 1] - y0) / delta;

      T* p = &Nd[block_size * q];
      T* px = &Nd[block_size * q + num_nodes];
      T* py = &Nd[block_size * q + 2 * num_nodes];

      deriv(x, y, p, px, py);

      for (int i = 0; i < num_nodes; i++) {
        px[i] *= inv;
        py[i] *= inv;
      }
    }

    // Solve to obtain the basis functions
    int nrhs = 3 * num_points;
    detail::Lapack<T>::getrs('N', num_nodes, nrhs, V.data(), ipiv.data(), Nd);
  }

 private:
  T x0, y0;
  T delta;
  int num_nodes;
  Basis basis;
  BasisDeriv deriv;

  // Storage for the factorization of V
  std::vector<T> V;
  std::vector<typename detail::Lapack<T>::int_type> ipiv;
};

}  // namespace xcgd

#endif  // XCGD_VANDERMONDE_H
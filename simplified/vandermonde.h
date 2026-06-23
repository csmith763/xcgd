#ifndef XCGD_VANDERMONDE_H
#define XCGD_VANDERMONDE_H

#include <vector>

// Fortran LAPACK interface
extern "C" {
void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv,
             int* info);

void dgetrs_(const char* trans, const int* n, const int* nrhs, const double* a,
             const int* lda, const int* ipiv, double* b, const int* ldb,
             int* info);

void sgetrf_(const int* m, const int* n, float* a, const int* lda, int* ipiv,
             int* info);

void sgetrs_(const char* trans, const int* n, const int* nrhs, const float* a,
             const int* lda, const int* ipiv, float* b, const int* ldb,
             int* info);
}

namespace xcgd {

namespace detail {

template <typename T>
struct Lapack;

template <>
struct Lapack<double> {
  static void getrf(int n, double* A, int* ipiv) {
    int info = 0;
    dgetrf_(&n, &n, A, &n, ipiv, &info);

    if (info < 0) {
      throw std::runtime_error("dgetrf: illegal argument " +
                               std::to_string(-info));
    } else if (info > 0) {
      throw std::runtime_error("dgetrf: singular matrix");
    }
  }

  static void getrs(char trans, int n, int nrhs, const double* LU,
                    const int* ipiv, double* B) {
    int info = 0;
    dgetrs_(&trans, &n, &nrhs, LU, &n, ipiv, B, &n, &info);

    if (info != 0) {
      throw std::runtime_error("dgetrs: illegal argument " +
                               std::to_string(-info));
    }
  }
};

template <>
struct Lapack<float> {
  static void getrf(int n, float* A, int* ipiv) {
    int info = 0;
    sgetrf_(&n, &n, A, &n, ipiv, &info);

    if (info < 0) {
      throw std::runtime_error("sgetrf: illegal argument " +
                               std::to_string(-info));
    } else if (info > 0) {
      throw std::runtime_error("sgetrf: singular matrix");
    }
  }

  static void getrs(char trans, int n, int nrhs, const float* LU,
                    const int* ipiv, float* B) {
    int info = 0;
    sgetrs_(&trans, &n, &nrhs, LU, &n, ipiv, B, &n, &info);

    if (info != 0) {
      throw std::runtime_error("sgetrs: illegal argument " +
                               std::to_string(-info));
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

  void eval(int num_points, const T* pts, T* Nd) const {
    const int block_size = 3 * num_nodes;

    const T inv = 1.0 / delta;
    for (int q = 0; q < num_points; q++) {
      const T x = (pts[2 * q] - x0) / delta;
      const T y = (pts[2 * q + 1] - y0) / delta;

      T* p = &Nd[block_size * q];
      T* px = &Nd[block_size * q + num_nodes];
      T* py = &Nd[block_size * q + 2 * num_nodes];

      basis(x, y, p);
      deriv(x, y, px, py);

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
  const Basis& basis;
  const BasisDeriv& deriv;

  // Storage for the factorization of V
  std::vector<T> V;
  std::vector<int> ipiv;
};

}  // namespace xcgd

#endif  // XCGD_VANDERMONDE_H
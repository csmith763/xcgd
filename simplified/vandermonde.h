#ifndef XCGD_VANDERMONDE_H
#define XCGD_VANDERMONDE_H

#include <vector>

namespace xcgd {

/**
 * @brief Lagrange interpolation using a Vandermonde-type matrix
 *
 * V = [ p(x1) ]     N = [ N1(x1) ]
 *     [ p(x2) ]         [ N2(x1) ]
 *     [   .   ]         [   .    ]
 *     [ p(xm) ]         [ Nm(x1) ]
 *
 * Nk(x) = p(x) * ak
 * A = [ a1 | a2 | . | ak ]
 *
 * Therefore V * A = I => A = V^{-1}
 *
 * N(x) = p(x) * A^{-1}
 * dN/dx = dp/dx * A^{-1}
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
        basis(basis),
        deriv(deriv),
        A(num_nodes * num_nodes) {
    // Build the interpolation matrix
    std::vector<T> V(num_nodes * num_nodes);

    for (int i = 0; i < num_nodes; i++) {
      T x = (X[2 * i] - x0) / delta;
      T y = (X[2 * i + 1] - y0) / delta;

      // Evaluate the row of the interpolation matrix
      basis(x, y, &V[i * num_nodes]);
    }

    compute_inverse(num_nodes, V, A);
  }

  void eval(int num_points, const T* pts, T* N, T* Nx) const {
    std::vector<T> p(num_nodes);
    std::vector<T> px(num_nodes);
    std::vector<T> py(num_nodes);

    for (int q = 0; q < num_points; q++) {
      const T x = (pts[2 * q + 0] - x0) / delta;
      const T y = (pts[2 * q + 1] - y0) / delta;

      basis(x, y, p.data());
      deriv(x, y, px.data(), py.data());

      T* Nq = &N[q * num_nodes];
      T* Nxq = &Nx[q * 2 * num_nodes];

      // N(q, i) = sum_a p_a(q) * A(a, i)
      for (int i = 0; i < num_nodes; i++) {
        T value = T(0);

        for (int a = 0; a < num_nodes; a++) {
          value += p[a] * A[num_nodes * a + i];
        }

        Nq[i] = value;
      }

      // dN_i/dx = (1 / delta) * sum_a dp_a/dxi * A(a, i)
      // dN_i/dy = (1 / delta) * sum_a dp_a/deta * A(a, i)
      for (int i = 0; i < num_nodes; i++) {
        T dx = T(0);
        T dy = T(0);

        for (int a = 0; a < num_nodes; a++) {
          dx += px[a] * A[num_nodes * a + i];
          dy += py[a] * A[num_nodes * a + i];
        }

        Nxq[2 * i + 0] = dx / delta;
        Nxq[2 * i + 1] = dy / delta;
      }
    }
  }

 private:
  T x0, y0;
  T delta;
  int num_nodes;
  const Basis& basis;
  const BasisDeriv& deriv;

  std::vector<T> A;

  static void compute_inverse(int n, const std::vector<T>& M,
                              std::vector<T>& Minv) {
    std::vector<T> A(n * n);
    std::vector<T> I(n * n, T(0));

    std::copy(M.begin(), M.end(), A.begin());

    for (int i = 0; i < n; i++) {
      I[n * i + i] = T(1);
    }

    for (int k = 0; k < n; k++) {
      int pivot = k;
      T max_abs = abs_value(A[n * k + k]);

      for (int i = k + 1; i < n; i++) {
        const T value = abs_value(A[n * i + k]);

        if (value > max_abs) {
          max_abs = value;
          pivot = i;
        }
      }

      if (max_abs == T(0)) {
        throw std::runtime_error(
            "Vandermonde2D interpolation matrix is singular");
      }

      if (pivot != k) {
        for (int j = 0; j < n; j++) {
          std::swap(A[n * k + j], A[n * pivot + j]);
          std::swap(I[n * k + j], I[n * pivot + j]);
        }
      }

      const T Akk = A[n * k + k];

      for (int j = 0; j < n; j++) {
        A[n * k + j] /= Akk;
        I[n * k + j] /= Akk;
      }

      for (int i = 0; i < n; i++) {
        if (i == k) {
          continue;
        }

        const T factor = A[n * i + k];

        if (factor == T(0)) {
          continue;
        }

        for (int j = 0; j < n; j++) {
          A[n * i + j] -= factor * A[n * k + j];
          I[n * i + j] -= factor * I[n * k + j];
        }
      }
    }

    Minv = std::move(I);
  }

  static T abs_value(T x) {
    using std::abs;
    return abs(x);
  }
};

}  // namespace xcgd

#endif  // XCGD_VANDERMONDE_H
#ifndef XCGD_QUADRATURE_H
#define XCGD_QUADRATURE_H

#include <stdexcept>
#include <string>
#include <vector>

#include "quadrature_multipoly.hpp"
#include "vandermonde.h"

namespace xcgd {

enum class LevelSetDerivMethod { AD, CENTRAL_FD, FORWARD_FD };

template <int spatial_dim, int degree, typename T, class Vandermonde>
void compute_level_set_quadrature(
    T x0, T y0, T delta, const Vandermonde& interp, const std::vector<T>& lsf,
    std::vector<T>& interior_points, std::vector<T>& interior_weights,
    std::vector<T>& exterior_points, std::vector<T>& exterior_weights,
    std::vector<T>& interface_points, std::vector<T>& interface_weights,
    std::vector<T>& interface_normals) {
  using algvec = algoim::uvector<T, spatial_dim>;

  T data[(degree + 1) * (degree + 1)];
  algoim::xarray<T, spatial_dim> phi(
      data, algoim::uvector<int, spatial_dim>(degree + 1, degree + 1));

  algoim::bernstein::bernsteinInterpolate<spatial_dim>(
      [&](const algvec& xi) {  // xi in [0, 1]
        T pt[2] = {x0 + delta * xi(0), y0 + delta * xi(1)};
        return interp.eval(pt, lsf.data());
      },
      phi);
  algoim::ImplicitPolyQuadrature<spatial_dim, T> ipquad(phi);

  auto vol_func = [&](const algvec& x, T w) {
    if (algoim::bernstein::evalBernsteinPoly(phi, x) <= 0.0) {
      interior_points.push_back(x0 + delta * x(0));
      interior_points.push_back(y0 + delta * x(1));
      interior_weights.push_back(delta * delta * w);
    } else {
      exterior_points.push_back(x0 + delta * x(0));
      exterior_points.push_back(y0 + delta * x(1));
      exterior_weights.push_back(delta * delta * w);
    }
  };
  ipquad.integrate(algoim::AutoMixed, degree + 1, vol_func);

  auto surf_func = [&](const algvec& x, T w, const algvec& _) {
    // Evaluate the gradient on the quadrature point
    // We assume that ipquad.phi.count() == 1 here
    algvec g =
        algoim::bernstein::evalBernsteinPolyGradient(ipquad.phi.poly(0), x);

    T nrm2 = T(0.0);
    for (int d = 0; d < spatial_dim; d++) {
      nrm2 += g(d) * g(d);
    }
    T nrm = sqrt(nrm2);

    // Normalize g
    algvec gn;
    for (int d = 0; d < spatial_dim; d++) {
      gn(d) = g(d) / nrm;
    }

    interface_points.push_back(x0 + delta * x(0));
    interface_points.push_back(y0 + delta * x(1));
    interface_normals.push_back(gn(0));
    interface_normals.push_back(gn(1));
    interface_weights.push_back(delta * w);
  };
  ipquad.integrate_surf(algoim::AutoMixed, degree + 1, surf_func);
}

template <int spatial_dim, int degree, typename T, class Vandermonde>
void compute_level_set_quadrature_derivatives(
    T x0, T y0, T delta, const Vandermonde& interp, const std::vector<T>& lsf,
    std::vector<T>& interior_points_jac, std::vector<T>& interior_weights_jac,
    std::vector<T>& exterior_points_jac, std::vector<T>& exterior_weights_jac,
    std::vector<T>& interface_points_jac, std::vector<T>& interface_weights_jac,
    std::vector<T>& interface_normals_jac,
    LevelSetDerivMethod method = LevelSetDerivMethod::FORWARD_FD,
    T dh = T(1e-8)) {
  const int n = static_cast<int>(lsf.size());

  auto eval_quad =
      [&](const std::vector<T>& lsf_eval, std::vector<T>& interior_points,
          std::vector<T>& interior_weights, std::vector<T>& exterior_points,
          std::vector<T>& exterior_weights, std::vector<T>& interface_points,
          std::vector<T>& interface_weights,
          std::vector<T>& interface_normals) {
        interior_points.clear();
        interior_weights.clear();
        exterior_points.clear();
        exterior_weights.clear();
        interface_points.clear();
        interface_weights.clear();
        interface_normals.clear();

        compute_level_set_quadrature<spatial_dim, degree, T, Vandermonde>(
            x0, y0, delta, interp, lsf_eval, interior_points, interior_weights,
            exterior_points, exterior_weights, interface_points,
            interface_weights, interface_normals);
      };

  std::vector<T> interior_points_0, interior_weights_0;
  std::vector<T> exterior_points_0, exterior_weights_0;
  std::vector<T> interface_points_0, interface_weights_0, interface_normals_0;

  eval_quad(lsf, interior_points_0, interior_weights_0, exterior_points_0,
            exterior_weights_0, interface_points_0, interface_weights_0,
            interface_normals_0);

  const int nip = static_cast<int>(interior_points_0.size());
  const int niw = static_cast<int>(interior_weights_0.size());
  const int nep = static_cast<int>(exterior_points_0.size());
  const int newt = static_cast<int>(exterior_weights_0.size());
  const int nfp = static_cast<int>(interface_points_0.size());
  const int nfw = static_cast<int>(interface_weights_0.size());
  const int nfn = static_cast<int>(interface_normals_0.size());

  interior_points_jac.assign(nip * n, T(0));
  interior_weights_jac.assign(niw * n, T(0));
  exterior_points_jac.assign(nep * n, T(0));
  exterior_weights_jac.assign(newt * n, T(0));
  interface_points_jac.assign(nfp * n, T(0));
  interface_weights_jac.assign(nfw * n, T(0));
  interface_normals_jac.assign(nfn * n, T(0));

  auto check_size = [](const std::vector<T>& v, int expected,
                       const char* name) {
    if (static_cast<int>(v.size()) != expected) {
      throw std::runtime_error(
          std::string("compute_level_set_quadrature_derivatives: output size "
                      "changed for ") +
          name +
          ". The finite-difference derivative is not well-defined at this "
          "level-set configuration.");
    }
  };

  auto add_column_forward = [](const std::vector<T>& y_plus,
                               const std::vector<T>& y0, std::vector<T>& jac,
                               int col, T inv_dh) {
    const int m = static_cast<int>(y0.size());
    for (int row = 0; row < m; row++) {
      jac[row + col * m] = (y_plus[row] - y0[row]) * inv_dh;
    }
  };

  auto add_column_central = [](const std::vector<T>& y_plus,
                               const std::vector<T>& y_minus,
                               std::vector<T>& jac, int col, T inv_2dh) {
    const int m = static_cast<int>(y_plus.size());
    for (int row = 0; row < m; row++) {
      jac[row + col * m] = (y_plus[row] - y_minus[row]) * inv_2dh;
    }
  };

  const bool use_forward = (method == LevelSetDerivMethod::FORWARD_FD);

  for (int col = 0; col < n; col++) {
    std::vector<T> lsf_plus = lsf;
    lsf_plus[col] += dh;

    std::vector<T> interior_points_p, interior_weights_p;
    std::vector<T> exterior_points_p, exterior_weights_p;
    std::vector<T> interface_points_p, interface_weights_p, interface_normals_p;

    eval_quad(lsf_plus, interior_points_p, interior_weights_p,
              exterior_points_p, exterior_weights_p, interface_points_p,
              interface_weights_p, interface_normals_p);

    check_size(interior_points_p, nip, "interior_points");
    check_size(interior_weights_p, niw, "interior_weights");
    check_size(exterior_points_p, nep, "exterior_points");
    check_size(exterior_weights_p, newt, "exterior_weights");
    check_size(interface_points_p, nfp, "interface_points");
    check_size(interface_weights_p, nfw, "interface_weights");
    check_size(interface_normals_p, nfn, "interface_normals");

    if (use_forward) {
      const T inv_dh = T(1) / dh;

      add_column_forward(interior_points_p, interior_points_0,
                         interior_points_jac, col, inv_dh);
      add_column_forward(interior_weights_p, interior_weights_0,
                         interior_weights_jac, col, inv_dh);
      add_column_forward(exterior_points_p, exterior_points_0,
                         exterior_points_jac, col, inv_dh);
      add_column_forward(exterior_weights_p, exterior_weights_0,
                         exterior_weights_jac, col, inv_dh);
      add_column_forward(interface_points_p, interface_points_0,
                         interface_points_jac, col, inv_dh);
      add_column_forward(interface_weights_p, interface_weights_0,
                         interface_weights_jac, col, inv_dh);
      add_column_forward(interface_normals_p, interface_normals_0,
                         interface_normals_jac, col, inv_dh);
    } else {
      std::vector<T> lsf_minus = lsf;
      lsf_minus[col] -= dh;

      std::vector<T> interior_points_m, interior_weights_m;
      std::vector<T> exterior_points_m, exterior_weights_m;
      std::vector<T> interface_points_m, interface_weights_m,
          interface_normals_m;

      eval_quad(lsf_minus, interior_points_m, interior_weights_m,
                exterior_points_m, exterior_weights_m, interface_points_m,
                interface_weights_m, interface_normals_m);

      check_size(interior_points_m, nip, "interior_points");
      check_size(interior_weights_m, niw, "interior_weights");
      check_size(exterior_points_m, nep, "exterior_points");
      check_size(exterior_weights_m, newt, "exterior_weights");
      check_size(interface_points_m, nfp, "interface_points");
      check_size(interface_weights_m, nfw, "interface_weights");
      check_size(interface_normals_m, nfn, "interface_normals");

      const T inv_2dh = T(0.5) / dh;

      add_column_central(interior_points_p, interior_points_m,
                         interior_points_jac, col, inv_2dh);
      add_column_central(interior_weights_p, interior_weights_m,
                         interior_weights_jac, col, inv_2dh);
      add_column_central(exterior_points_p, exterior_points_m,
                         exterior_points_jac, col, inv_2dh);
      add_column_central(exterior_weights_p, exterior_weights_m,
                         exterior_weights_jac, col, inv_2dh);
      add_column_central(interface_points_p, interface_points_m,
                         interface_points_jac, col, inv_2dh);
      add_column_central(interface_weights_p, interface_weights_m,
                         interface_weights_jac, col, inv_2dh);
      add_column_central(interface_normals_p, interface_normals_m,
                         interface_normals_jac, col, inv_2dh);
    }
  }
}

}  // namespace xcgd

#endif  // XCGD_QUADRATURE_H
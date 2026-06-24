#ifndef XCGD_QUADRATURE_H
#define XCGD_QUADRATURE_H

#include "quadrature_multipoly.hpp"
#include "vandermonde.h"

namespace xcgd {

template <int spatial_dim, int degree, typename T, class Vandermonde>
void compute_level_set_quadrature(
    Vandermonde& interp, std::vector<T>& lsf, std::vector<T>& interior_points,
    std::vector<T>& interior_weights, std::vector<T>& exterior_points,
    std::vector<T>& exterior_weights, std::vector<T>& interface_points,
    std::vector<T>& interface_weights, std::vector<T>& interface_normals) {
  using algvec = algoim::uvector<T, spatial_dim>;
  T x0, y0, delta;
  interp.get_base_data(x0, y0, delta);

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
      for (int d = 0; d < spatial_dim; d++) {
        interior_points.push_back(x(d));
      }
      interior_weights.push_back(w);
    } else {
      for (int d = 0; d < spatial_dim; d++) {
        exterior_points.push_back(x(d));
      }
      exterior_weights.push_back(w);
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

    for (int d = 0; d < spatial_dim; d++) {
      interface_points.push_back(x(d));
      interface_normals.push_back(gn(d));
    }
    interface_weights.push_back(w);
  };
  ipquad.integrate_surf(algoim::AutoMixed, degree + 1, surf_func);
}

}  // namespace xcgd

#endif  // XCGD_QUADRATURE_H
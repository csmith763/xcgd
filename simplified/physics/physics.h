#ifndef XCGD_PHYSICS_H
#define XCGD_PHYSICS_H

#include <cstddef>
#include <type_traits>

namespace xcgd {

class HelmholtzPhysics {
 public:
  HelmholtzPhysics(double r) : r(r) {}

  static constexpr int spatial_dim = 2;
  static constexpr int dof_per_node = 1;

  template <typename T>
  using location_t = A2D::Vec<T, 2>;

  template <typename T>
  using normal_t = A2D::Vec<T, 2>;

  template <typename T>
  using input_t = A2D::Vec<T, 1>;

  template <typename T>
  using gradient_t = A2D::Vec<T, 2>;

  template <typename T>
  T integrand(T& weight, location_t<T>& xloc, normal_t<T>& normal,
              input_t<T>& val, gradient_t<T>& grad) const {
    return 0.5 * weight *
           (val[0] * val[0] + r * r * (grad[0] * grad[0] + grad[1] * grad[1]));
  }

  template <typename T>
  void residual(T weight, location_t<T>& xloc, normal_t<T>& normal,
                input_t<T>& val, gradient_t<T>& grad, input_t<T>& val_res,
                gradient_t<T>& grad_res) const {
    val_res[0] = weight * val[0];
    grad_res[0] = weight * r * r * grad[0];
    grad_res[1] = weight * r * r * grad[1];
  }

 private:
  double r;
};

template <class BodyForceFunc = std::nullptr_t>
class LinearElasticity2D {
 public:
  LinearElasticity2D(double E, double nu)
      : G(0.5 * E / (1.0 + nu)),
        lambda(E * nu / (1.0 - nu * nu)),
        body_force(nullptr) {}

  LinearElasticity2D(double E, double nu, const BodyForceFunc& body_force)
      : G(0.5 * E / (1.0 + nu)),
        lambda(E * nu / (1.0 - nu * nu)),
        body_force(body_force) {}

  static constexpr int spatial_dim = 2;
  static constexpr int dof_per_node = 2;

  template <typename T>
  using location_t = A2D::Vec<T, spatial_dim>;

  template <typename T>
  using normal_t = A2D::Vec<T, spatial_dim>;

  template <typename T>
  using input_t = A2D::Vec<T, dof_per_node>;

  template <typename T>
  using gradient_t = A2D::Mat<T, dof_per_node, spatial_dim>;

  template <typename T>
  T integrand(T& weight, location_t<T>& xloc, normal_t<T>& normal,
              input_t<T>& val, gradient_t<T>& grad) const {
    T strain_energy, potential = 0.0;
    A2D::SymMat<T, spatial_dim> E, S;

    A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad, E);
    A2D::SymIsotropic(G, lambda, E, S);
    A2D::SymMatMultTrace(E, S, strain_energy);
    if constexpr (!std::is_same_v<BodyForceFunc, std::nullptr_t>) {
      A2D::Vec<T, dof_per_node> g = body_force(xloc);
      A2D::VecDot(g, val, potential);
    }
    return weight * (0.5 * strain_energy - potential);
  }

  template <typename T>
  void residual(T weight, location_t<T>& xloc, normal_t<T>& normal,
                input_t<T>& val_input, gradient_t<T>& grad_input,
                input_t<T>& val_res, gradient_t<T>& grad_res) const {
    A2D::ADObj<A2D::Vec<T, dof_per_node>&> val(val_input, val_res);
    A2D::ADObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad(grad_input,
                                                             grad_res);

    A2D::ADObj<T> strain_energy, output;
    A2D::ADObj<A2D::SymMat<T, spatial_dim>> E, S;

    if constexpr (!std::is_same_v<BodyForceFunc, std::nullptr_t>) {
      A2D::ADObj<T> potential;
      A2D::Vec<T, dof_per_node> g = body_force(xloc);
      auto stack = A2D::MakeStack(
          A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad, E),
          A2D::SymIsotropic(G, lambda, E, S),
          A2D::SymMatMultTrace(E, S, strain_energy),
          A2D::VecDot(g, val, potential),
          A2D::Eval(weight * (0.5 * strain_energy - potential), output));
      output.bvalue() = 1.0;
      stack.reverse();
    } else {
      auto stack = A2D::MakeStack(
          A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad, E),
          A2D::SymIsotropic(G, lambda, E, S),
          A2D::SymMatMultTrace(E, S, strain_energy),
          A2D::Eval(0.5 * weight * strain_energy, output));
      output.bvalue() = 1.0;
      stack.reverse();
    }
  }

 private:
  double lambda, G;
  BodyForceFunc body_force;
};

class ElasticityMass2D {
 public:
  ElasticityMass2D(double rho) : rho(rho) {}

  static constexpr int spatial_dim = 2;
  static constexpr int dof_per_node = 2;

  template <typename T>
  using location_t = A2D::Vec<T, spatial_dim>;

  template <typename T>
  using normal_t = A2D::Vec<T, spatial_dim>;

  template <typename T>
  using input_t = A2D::Vec<T, dof_per_node>;

  template <typename T>
  using gradient_t = A2D::Mat<T, dof_per_node, spatial_dim>;

  template <typename T>
  T integrand(T& weight, location_t<T>& xloc, normal_t<T>& normal,
              input_t<T>& val, gradient_t<T>& grad) const {
    return 0.5 * weight * rho * (val[0] * val[0] + val[1] * val[1]);
  }

  template <typename T>
  void residual(T weight, location_t<T>& xloc, normal_t<T>& normal,
                input_t<T>& val_input, gradient_t<T>& grad_input,
                input_t<T>& val_res, gradient_t<T>& grad_res) const {
    val_res[0] = weight * rho * val_input[0];
    val_res[1] = weight * rho * val_input[1];
  }

 private:
  double rho;
};

class Area2D {
 public:
  static constexpr int spatial_dim = 2;
  static constexpr int dof_per_node = 1;

  template <typename T>
  using location_t = A2D::Vec<T, spatial_dim>;

  template <typename T>
  using normal_t = A2D::Vec<T, spatial_dim>;

  template <typename T>
  using input_t = A2D::Vec<T, dof_per_node>;

  template <typename T>
  using gradient_t = A2D::Mat<T, dof_per_node, spatial_dim>;

  template <typename T>
  T integrand(T& weight, location_t<T>& xloc, normal_t<T>& normal,
              input_t<T>& val, gradient_t<T>& grad) const {
    return weight;
  }

  template <typename T>
  void residual(T weight, location_t<T>& xloc, normal_t<T>& normal,
                input_t<T>& val_input, gradient_t<T>& grad_input,
                input_t<T>& val_res, gradient_t<T>& grad_res) const {}
};

}  // namespace xcgd

#endif  // XCGD_PHYSICS_H
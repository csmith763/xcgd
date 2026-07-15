#ifndef XCGD_ASSEMBLER_H
#define XCGD_ASSEMBLER_H

#include <vector>

#include "mesh_assembler.h"

namespace xcgd {

template <typename T>
class Assembler {
 public:
  Assembler(std::vector<std::shared_ptr<MeshAssemblerBase<T>>> assemblers)
      : assemblers(assemblers) {
    update();
  }

  void update() {
    num_dof = 0;
    num_design_vars = 0;
    for (int i = 0; i < assemblers.size(); i++) {
      int index = assemblers[i]->get_max_dof_index();
      num_dof = std::max(index, num_dof);

      index = assemblers[i]->get_max_design_index();
      num_design_vars = std::max(index, num_design_vars);
    }

    dof.resize(num_dof);
    res.resize(num_dof);
    adjoint.resize(num_dof);
    dfdx.resize(num_design_vars);

    // Build the non-zero pattern
    pattern_builder.initialize(num_dof);
    pattern_builder.begin_count();
    for (int i = 0; i < assemblers.size(); i++) {
      assemblers[i]->add_row_counts(pattern_builder);
    }

    pattern_builder.begin_fill();
    for (int i = 0; i < assemblers.size(); i++) {
      assemblers[i]->insert_columns(pattern_builder);
    }

    // Finalize the CSR pattern
    pattern_builder.finalize(csr);
  }

  std::vector<T>& get_dof() { return dof; }
  std::vector<T>& get_residual() { return res; }
  std::vector<T>& get_adjoint() { return adjoint; }
  std::vector<T>& get_dfdx() { return dfdx; }
  CSRMat<T>& get_jacobian() { return csr; }

  T eval_functional() const {
    T toal_value = 0.0;
    for (int i = 0; i < assemblers.size(); i++) {
      toal_value += assemblers[i]->functional(dof);
    }
    return toal_value;
  }

  void eval_residual() {
    std::fill(res.begin(), res.end(), T(0));
    for (int i = 0; i < assemblers.size(); i++) {
      assemblers[i]->add_residual(dof, res);
    }
  }

  void eval_jacobian() {
    csr.zero();
    for (int i = 0; i < assemblers.size(); i++) {
      assemblers[i]->add_jacobian(dof, csr);
    }
  }

  // Functions for compute the derivatives
  void zero_derivative() { std::fill(dfdx.begin(), dfdx.end(), T(0)); }

  void add_functional_derivative() {
    for (int i = 0; i < assemblers.size(); i++) {
      assemblers[i]->add_functional_derivative(dof, dfdx);
    }
  }

  void add_adjoint_residual_product() {
    for (int i = 0; i < assemblers.size(); i++) {
      assemblers[i]->add_adjoint_residual_product(dof, adjoint, dfdx);
    }
  }

 private:
  int num_dof;
  int num_design_vars;
  std::vector<T> dof;
  std::vector<T> res;
  std::vector<T> adjoint;
  std::vector<T> dfdx;
  CSRMat<T> csr;
  std::vector<std::shared_ptr<MeshAssemblerBase<T>>> assemblers;
  CSRPatternBuilder pattern_builder;
};

}  // namespace xcgd

#endif  // XCGD_ASSEMBLER_H
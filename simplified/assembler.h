#ifndef XCGD_ASSEMBLER_H
#define XCGD_ASSEMBLER_H

#include <vector>

#include "a2dcore.h"
#include "mesh_base.h"
#include "sparse_matrix.h"

namespace xcgd {

template <typename T>
class MeshAssemblerBase {
 public:
  virtual ~MeshAssemblerBase() = default;

  virtual int get_max_dof_index() const = 0;
  virtual T energy(const std::vector<T>& dof) const = 0;
  virtual void add_residual(const std::vector<T>& dof,
                            std::vector<T>& res) const = 0;
  virtual void add_jacobian(const std::vector<T>& dof,
                            CSRMat<T>& csr) const = 0;

  virtual void add_row_counts(CSRPatternBuilder& pattern_builder) const = 0;
  virtual void insert_columns(CSRPatternBuilder& pattern_builder) const = 0;
};

template <typename T>
class Assembler {
 public:
  Assembler(std::vector<std::shared_ptr<MeshAssemblerBase<T>>> assemblers)
      : assemblers(assemblers) {}

  void update() {
    num_dof = 0;
    for (int i = 0; i < assemblers.size(); i++) {
      int index = assemblers[i]->get_max_dof_index();
      num_dof = std::max(index, num_dof);
    }

    dof.resize(num_dof);
    res.resize(num_dof);

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
  CSRMat<T>& get_jacobian() { return csr; }

  T eval_energy() const {
    T total_energy = 0.0;
    for (int i = 0; i < assemblers.size(); i++) {
      total_energy += assemblers[i]->energy(dof);
    }
    return total_energy;
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

 private:
  int num_dof;
  std::vector<T> dof;
  std::vector<T> res;
  CSRMat<T> csr;
  std::vector<std::shared_ptr<MeshAssemblerBase<T>>> assemblers;
  CSRPatternBuilder pattern_builder;
};

template <typename T, class Physics>
class MeshAssembler : public MeshAssemblerBase<T> {
 public:
  // Declarations for this type
  static constexpr int dof_per_node = Physics::dof_per_node;
  static constexpr int spatial_dim = Physics::spatial_dim;

  MeshAssembler(std::shared_ptr<MeshBase<T>> mesh, const Physics& physics)
      : mesh(mesh), physics(physics) {}
  ~MeshAssembler() {}

  int get_max_dof_index() const {
    return dof_per_node * mesh->get_max_node_index();
  }

  /**
   * @brief Compute the energy
   *
   * @param dof Input degrees of freedom vector
   * @return T Energy contribution
   */
  T energy(const std::vector<T>& dof) const {
    T total_energy = 0.0;

    // Query the max nodes and max quadrature points
    int max_nodes = mesh->get_max_num_nodes();
    int max_quad_pts = mesh->get_max_num_quadrature_points();

    // Arrays for the node numbers
    std::vector<int> nodes(max_nodes);
    std::vector<T> X(spatial_dim * max_nodes);
    std::vector<T> elem_dof(dof_per_node * max_nodes);

    // Arrays for the quadrature points, weights and normals
    std::vector<T> weights(max_quad_pts);
    std::vector<T> points(spatial_dim * max_quad_pts);
    std::vector<T> normals(spatial_dim * max_quad_pts);

    // Arrays for storing the basis functions and derivatives
    std::vector<T> Nd((1 + spatial_dim) * max_nodes * max_quad_pts);

    for (int elem = 0; elem < mesh->get_num_elements(); elem++) {
      // Get the node numbers and locations associated with the element
      int num_nodes = mesh->get_nodes(elem, nodes);

      // Get the quadrature weights and points associated with the element
      int num_quad_points =
          mesh->get_quadrature(elem, weights, points, normals);

      // Evaluate the basis at all the quadrature points
      mesh->eval_basis(elem, num_quad_points, points, Nd);

      // Get the node locations
      mesh->get_node_points(elem, X);

      // Get the variables associated with the nodes
      get_element_vars(num_nodes, nodes, dof, elem_dof);

      // Perform the quadrature
      for (int i = 0; i < num_quad_points; i++) {
        typename Physics::template location_t<T> xloc;
        typename Physics::template normal_t<T> normal;
        typename Physics::template input_t<T> vals;
        typename Physics::template gradient_t<T> grad;

        const T* Nptr = &Nd[(spatial_dim + 1) * num_nodes * i];
        const T* Nxptr = &Nd[(spatial_dim + 1) * num_nodes * i + num_nodes];

        for (int k = 0; k < spatial_dim; k++) {
          normal[k] = normals[i * spatial_dim + k];
        }

        interp_values(spatial_dim, num_nodes, Nptr, X, xloc);
        interp_values(dof_per_node, num_nodes, Nptr, elem_dof, vals);
        interp_gradient(dof_per_node, num_nodes, Nxptr, elem_dof, grad);

        total_energy += physics.energy(weights[i], xloc, normal, vals, grad);
      }
    }

    return total_energy;
  }

  /**
   * @brief Add the residual from this mesh/vector combo
   *
   * @param dof Input degrees of freedom
   * @param res Residual value
   */
  void add_residual(const std::vector<T>& dof, std::vector<T>& res) const {
    // Query the max nodes and max quadrature points
    int max_nodes = mesh->get_max_num_nodes();
    int max_quad_pts = mesh->get_max_num_quadrature_points();

    // Arrays for the node numbers
    std::vector<int> nodes(max_nodes);
    std::vector<T> X(spatial_dim * max_nodes);
    std::vector<T> elem_dof(dof_per_node * max_nodes);
    std::vector<T> elem_res(dof_per_node * max_nodes);

    // Arrays for the quadrature points, weights and normals
    std::vector<T> weights(max_quad_pts);
    std::vector<T> points(spatial_dim * max_quad_pts);
    std::vector<T> normals(spatial_dim * max_quad_pts);

    // Arrays for storing the basis functions and derivatives
    std::vector<T> Nd((1 + spatial_dim) * max_nodes * max_quad_pts);

    for (int elem = 0; elem < mesh->get_num_elements(); elem++) {
      // Get the node numbers and locations associated with the element
      int num_nodes = mesh->get_nodes(elem, nodes);

      // Get the quadrature weights and points associated with the element
      int num_quad_points =
          mesh->get_quadrature(elem, weights, points, normals);

      // Evaluate the basis at all the quadrature points
      mesh->eval_basis(elem, num_quad_points, points, Nd);

      // Get the node locations
      mesh->get_node_points(elem, X);

      // Get the variables associated with the nodes
      get_element_vars(num_nodes, nodes, dof, elem_dof);

      // Fill the element zeros in
      std::fill(elem_res.begin(), elem_res.begin() + dof_per_node * num_nodes,
                T(0));

      // Perform the quadrature
      for (int i = 0; i < num_quad_points; i++) {
        typename Physics::template location_t<T> xloc;
        typename Physics::template normal_t<T> normal;
        typename Physics::template input_t<T> vals;
        typename Physics::template gradient_t<T> grad;
        typename Physics::template input_t<T> vals_res;
        typename Physics::template gradient_t<T> grad_res;

        for (int k = 0; k < spatial_dim; k++) {
          normal[k] = normals[i * spatial_dim + k];
        }
        const T* Nptr = &Nd[(spatial_dim + 1) * num_nodes * i];
        const T* Nxptr = &Nd[(spatial_dim + 1) * num_nodes * i + num_nodes];

        interp_values(spatial_dim, num_nodes, Nptr, X, xloc);
        interp_values(dof_per_node, num_nodes, Nptr, elem_dof, vals);
        interp_gradient(dof_per_node, num_nodes, Nxptr, elem_dof, grad);

        physics.residual(weights[i], xloc, normal, vals, grad, vals_res,
                         grad_res);

        add_res_values(dof_per_node, num_nodes, Nptr, vals_res, elem_res);
        add_res_gradient(dof_per_node, num_nodes, Nxptr, grad_res, elem_res);
      }

      add_element_residual(num_nodes, nodes, elem_res, res);
    }
  }

  void add_jacobian(const std::vector<T>& dof, CSRMat<T>& csr) const {
    // Query the max nodes and max quadrature points
    int max_nodes = mesh->get_max_num_nodes();
    int max_quad_pts = mesh->get_max_num_quadrature_points();

    // Arrays for the node numbers
    std::vector<int> nodes(max_nodes);
    std::vector<T> X(spatial_dim * max_nodes);
    std::vector<T> elem_dof(dof_per_node * max_nodes);
    std::vector<T> elem_jac(dof_per_node * max_nodes * dof_per_node *
                            max_nodes);

    // Arrays for the quadrature points, weights and normals
    std::vector<T> weights(max_quad_pts);
    std::vector<T> points(spatial_dim * max_quad_pts);
    std::vector<T> normals(spatial_dim * max_quad_pts);

    // Arrays for storing the basis functions and derivatives
    std::vector<T> Nd((1 + spatial_dim) * max_nodes * max_quad_pts);

    for (int elem = 0; elem < mesh->get_num_elements(); elem++) {
      // Get the node numbers and locations associated with the element
      int num_nodes = mesh->get_nodes(elem, nodes);

      // Get the quadrature weights and points associated with the element
      int num_quad_points =
          mesh->get_quadrature(elem, weights, points, normals);

      // Evaluate the basis at all the quadrature points
      mesh->eval_basis(elem, num_quad_points, points, Nd);

      // Get the node locations
      mesh->get_node_points(elem, X);

      // Get the variables associated with the nodes
      get_element_vars(num_nodes, nodes, dof, elem_dof);

      // Fill the element zeros in
      std::fill(elem_jac.begin(), elem_jac.end(), T(0));

      // Perform the quadrature
      for (int i = 0; i < num_quad_points; i++) {
        using ad_t = A2D::ADScalar<T, dof_per_node * (1 + spatial_dim)>;

        typename Physics::template location_t<ad_t> xloc;
        typename Physics::template normal_t<ad_t> normal;
        typename Physics::template input_t<ad_t> vals;
        typename Physics::template gradient_t<ad_t> grad;
        typename Physics::template input_t<ad_t> vals_res;
        typename Physics::template gradient_t<ad_t> grad_res;

        for (int k = 0; k < spatial_dim; k++) {
          normal[k] = normals[i * spatial_dim + k];
        }
        const T* Nptr = &Nd[(spatial_dim + 1) * num_nodes * i];
        const T* Nxptr = &Nd[(spatial_dim + 1) * num_nodes * i + num_nodes];

        interp_values(spatial_dim, num_nodes, Nptr, X, xloc);
        interp_values(dof_per_node, num_nodes, Nptr, elem_dof, vals);
        interp_gradient(dof_per_node, num_nodes, Nxptr, elem_dof, grad);

        // Seed the derivatives
        for (int j = 0; j < dof_per_node; j++) {
          vals[j].deriv[j] = 1.0;
        }
        for (int j = 0; j < spatial_dim * dof_per_node; j++) {
          grad[j].deriv[j + dof_per_node] = 1.0;
        }

        // Compute the Jacobian
        physics.residual(ad_t(weights[i]), xloc, normal, vals, grad, vals_res,
                         grad_res);

        // Extract the result as a 2 x 2 block of Jacobian entries
        T jac_vv[dof_per_node * dof_per_node];
        T jac_vg[dof_per_node * dof_per_node * spatial_dim];
        T jac_gv[dof_per_node * dof_per_node * spatial_dim];
        T jac_gg[dof_per_node * dof_per_node * spatial_dim * spatial_dim];

        for (int j = 0; j < dof_per_node; j++) {
          for (int k = 0; k < dof_per_node; k++) {
            jac_vv[k + j * dof_per_node] = vals_res[j].deriv[k];
          }
        }
        for (int j = 0; j < dof_per_node; j++) {
          for (int k = 0; k < dof_per_node * spatial_dim; k++) {
            jac_vg[k + j * dof_per_node * spatial_dim] =
                vals_res[j].deriv[dof_per_node + k];
          }
        }
        for (int j = 0; j < dof_per_node * spatial_dim; j++) {
          for (int k = 0; k < dof_per_node; k++) {
            jac_gv[k + j * dof_per_node] = grad_res[j].deriv[k];
          }
        }
        for (int j = 0; j < dof_per_node * spatial_dim; j++) {
          for (int k = 0; k < dof_per_node * spatial_dim; k++) {
            jac_gg[k + j * dof_per_node * spatial_dim] =
                grad_res[j].deriv[dof_per_node + k];
          }
        }

        add_jac_values(dof_per_node, num_nodes, Nptr, jac_vv, elem_jac);
        add_jac_values_gradient(dof_per_node, num_nodes, Nptr, Nxptr, jac_vg,
                                elem_jac);
        add_jac_gradient_values(dof_per_node, num_nodes, Nptr, Nxptr, jac_gv,
                                elem_jac);
        add_jac_gradient(dof_per_node, num_nodes, Nxptr, jac_gg, elem_jac);
      }

      add_element_jacobian(num_nodes, nodes, elem_jac, csr);
    }
  }

  void add_row_counts(CSRPatternBuilder& pattern_builder) const {
    int max_nodes = mesh->get_max_num_nodes();
    std::vector<int> nodes(max_nodes);
    std::vector<int> elem_dofs(dof_per_node * max_nodes);

    // Pass 1: count approximate row insertions
    for (int elem = 0; elem < mesh->get_num_elements(); elem++) {
      int num_nodes = mesh->get_nodes(elem, nodes);

      const int num_elem_dofs = dof_per_node * num_nodes;
      fill_element_dofs(num_nodes, nodes, elem_dofs);

      pattern_builder.count_dense_block(elem_dofs.data(), num_elem_dofs,
                                        num_elem_dofs);
    }
  }

  void insert_columns(CSRPatternBuilder& pattern_builder) const {
    int max_nodes = mesh->get_max_num_nodes();
    std::vector<int> nodes(max_nodes);
    std::vector<int> elem_dofs(dof_per_node * max_nodes);

    for (int elem = 0; elem < mesh->get_num_elements(); elem++) {
      int num_nodes = mesh->get_nodes(elem, nodes);

      const int num_elem_dofs = dof_per_node * num_nodes;
      fill_element_dofs(num_nodes, nodes, elem_dofs);

      pattern_builder.add_dense_block(elem_dofs.data(), num_elem_dofs,
                                      elem_dofs.data(), num_elem_dofs);
    }
  }

 private:
  std::shared_ptr<MeshBase<T>> mesh;
  const Physics& physics;

  void fill_element_dofs(int num_nodes, const std::vector<int>& nodes,
                         std::vector<int>& elem_dofs) const {
    for (int i = 0; i < num_nodes; i++) {
      int node = nodes[i];

      for (int a = 0; a < dof_per_node; a++) {
        elem_dofs[dof_per_node * i + a] = dof_per_node * node + a;
      }
    }
  }

  void get_element_vars(const int num_nodes, const std::vector<int>& nodes,
                        const std::vector<T>& dof, std::vector<T>& vars) const {
    for (int i = 0; i < num_nodes; i++) {
      int node = nodes[i];
      for (int j = 0; j < dof_per_node; j++) {
        vars[dof_per_node * i + j] = dof[dof_per_node * node + j];
      }
    }
  }

  void add_element_residual(int num_nodes, const std::vector<int>& nodes,
                            const std::vector<T>& elem_res,
                            std::vector<T>& res) const {
    for (int i = 0; i < num_nodes; i++) {
      int node = nodes[i];
      for (int j = 0; j < dof_per_node; j++) {
        res[dof_per_node * node + j] += elem_res[dof_per_node * i + j];
      }
    }
  }

  template <class Output>
  void interp_values(const int dim, const int num_nodes, const T* N,
                     std::vector<T>& vals, Output& out) const {
    for (int k = 0; k < dim; k++) {
      out[k] = 0.0;
    }

    for (int i = 0; i < num_nodes; i++) {
      for (int k = 0; k < dim; k++) {
        out[k] += N[i] * vals[dim * i + k];
      }
    }
  }

  template <class Output>
  void interp_gradient(const int dim, const int num_nodes, const T* Nx,
                       std::vector<T>& vals, Output& out) const {
    for (int k = 0; k < dim * spatial_dim; k++) {
      out[k] = 0.0;
    }

    for (int i = 0; i < spatial_dim; i++) {
      for (int j = 0; j < dim; j++) {
        for (int k = 0; k < num_nodes; k++) {
          out[i + spatial_dim * j] += Nx[k + i * num_nodes] * vals[dim * k + j];
        }
      }
    }
  }

  template <class InputRes>
  void add_res_values(int dim, const int num_nodes, const T* N,
                      const InputRes& vals_res,
                      std::vector<T>& elem_res) const {
    for (int i = 0; i < num_nodes; i++) {
      for (int k = 0; k < dim; k++) {
        elem_res[dim * i + k] += N[i] * vals_res[k];
      }
    }
  }

  template <class GradRes>
  void add_res_gradient(int dim, int num_nodes, const T* Nx,
                        const GradRes& grad_res,
                        std::vector<T>& elem_res) const {
    for (int i = 0; i < spatial_dim; i++) {
      for (int j = 0; j < dim; j++) {
        for (int k = 0; k < num_nodes; k++) {
          elem_res[dim * k + j] +=
              Nx[k + i * num_nodes] * grad_res[i + spatial_dim * j];
        }
      }
    }
  }

  void add_element_jacobian(int num_nodes, const std::vector<int>& nodes,
                            const std::vector<T>& elem_jac,
                            CSRMat<T>& csr) const {
    const int nd = dof_per_node * num_nodes;

    for (int i = 0; i < num_nodes; i++) {
      const int inode = nodes[i];

      for (int a = 0; a < dof_per_node; a++) {
        const int local_row = dof_per_node * i + a;
        const int global_row = dof_per_node * inode + a;

        const int row_start = csr.rowp[global_row];
        const int row_end = csr.rowp[global_row + 1];

        for (int j = 0; j < num_nodes; j++) {
          const int jnode = nodes[j];

          for (int b = 0; b < dof_per_node; b++) {
            const int local_col = dof_per_node * j + b;
            const int global_col = dof_per_node * jnode + b;

            auto begin = csr.cols.begin() + row_start;
            auto end = csr.cols.begin() + row_end;

            auto it = std::lower_bound(begin, end, global_col);
            const int csr_index = static_cast<int>(it - csr.cols.begin());

            csr.data[csr_index] += elem_jac[nd * local_row + local_col];
          }
        }
      }
    }
  }

  void add_jac_values(int dim, int num_nodes, const T* N, const T* jac_vv,
                      std::vector<T>& elem_jac) const {
    const int nd = dim * num_nodes;

    for (int i = 0; i < num_nodes; i++) {
      for (int j = 0; j < num_nodes; j++) {
        const T NiNj = N[i] * N[j];

        for (int a = 0; a < dim; a++) {
          const int row = dim * i + a;

          for (int b = 0; b < dim; b++) {
            const int col = dim * j + b;

            elem_jac[nd * row + col] += NiNj * jac_vv[dim * a + b];
          }
        }
      }
    }
  }

  void add_jac_values_gradient(int dim, int num_nodes, const T* N, const T* Nx,
                               const T* jac_vg,
                               std::vector<T>& elem_jac) const {
    const int nd = dim * num_nodes;
    const int gdim = dim * spatial_dim;

    for (int i = 0; i < num_nodes; i++) {
      for (int j = 0; j < num_nodes; j++) {
        for (int a = 0; a < dim; a++) {
          const int row = dim * i + a;

          for (int b = 0; b < dim; b++) {
            const int col = dim * j + b;

            T value = T(0.0);

            for (int n = 0; n < spatial_dim; n++) {
              const int gb = spatial_dim * b + n;

              value += jac_vg[gdim * a + gb] * Nx[n * num_nodes + j];
            }

            elem_jac[nd * row + col] += N[i] * value;
          }
        }
      }
    }
  }

  void add_jac_gradient_values(int dim, int num_nodes, const T* Nx, const T* N,
                               const T* jac_gv,
                               std::vector<T>& elem_jac) const {
    const int nd = dim * num_nodes;

    for (int i = 0; i < num_nodes; i++) {
      for (int j = 0; j < num_nodes; j++) {
        for (int a = 0; a < dim; a++) {
          const int row = dim * i + a;

          for (int b = 0; b < dim; b++) {
            const int col = dim * j + b;

            T value = T(0.0);

            for (int m = 0; m < spatial_dim; m++) {
              const int ga = spatial_dim * a + m;

              value += Nx[m * num_nodes + i] * jac_gv[dim * ga + b];
            }

            elem_jac[nd * row + col] += value * N[j];
          }
        }
      }
    }
  }

  void add_jac_gradient(int dim, int num_nodes, const T* Nx, const T* jac_gg,
                        std::vector<T>& elem_jac) const {
    const int nd = dim * num_nodes;
    const int gdim = dim * spatial_dim;

    for (int i = 0; i < num_nodes; i++) {
      for (int j = 0; j < num_nodes; j++) {
        for (int a = 0; a < dim; a++) {
          const int row = dim * i + a;

          for (int b = 0; b < dim; b++) {
            const int col = dim * j + b;

            T value = T(0);

            for (int m = 0; m < spatial_dim; m++) {
              const int ga = spatial_dim * a + m;

              for (int n = 0; n < spatial_dim; n++) {
                const int gb = spatial_dim * b + n;

                value += Nx[m * num_nodes + i] * jac_gg[gdim * ga + gb] *
                         Nx[n * num_nodes + j];
              }
            }

            elem_jac[nd * row + col] += value;
          }
        }
      }
    }
  }
};

}  // namespace xcgd

#endif  // XCGD_ASSEMBLER_H
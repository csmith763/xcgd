#ifndef XCGD_MESH_BASE_H
#define XCGD_MESH_BASE_H

#include <vector>

namespace xcgd {

template <typename T>
class MeshBase {
 public:
  virtual ~MeshBase() = default;

  virtual int get_max_node_index() const = 0;
  virtual int get_num_elements() const = 0;
  virtual int get_max_element_nodes() const = 0;
  virtual int get_max_num_quadrature_points() const = 0;
  virtual int get_nodes(int elem, std::vector<int>& nodes) const = 0;
  virtual void get_points(int elem, std::vector<T>& X) const = 0;
  virtual int get_quadrature(int elem, std::vector<T>& weights,
                             std::vector<T>& points,
                             std::vector<T>& normals) const = 0;
  virtual void eval_basis(int elem, int num_quad_points,
                          const std::vector<T>& pts,
                          std::vector<T>& Nd) const = 0;

  virtual int get_max_design_index() const { return 0; }
  virtual int get_max_element_design_vars() const { return 0; }
  virtual int get_quadrature_derivative(int elem, std::vector<T>& dwdx,
                                        std::vector<T>& dpdx,
                                        std::vector<T>& dndx, int& ndvs,
                                        std::vector<int>& dvs) {
    ndvs = 0;
    return 0;
  }
};

}  // namespace xcgd

#endif  // XCGD_MESH_BASE_H
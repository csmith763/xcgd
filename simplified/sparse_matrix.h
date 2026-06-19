#ifndef XCGD_SPARSE_MATRIX_H
#define XCGD_SPARSE_MATRIX_H

#include <vector>

namespace xcgd {

template <typename T>
class CSRMat {
 public:
  void zero() { std::fill(data.begin(), data.end(), T(0)); }

  int nrows;
  std::vector<int> rowp;
  std::vector<int> cols;
  std::vector<T> data;
};

class CSRPatternBuilder {
 public:
  void initialize(int nrows) {
    this->nrows = nrows;
    rows.resize(nrows);
    row_counts.resize(nrows);
  }

  void begin_count() { std::fill(row_counts.begin(), row_counts.end(), 0); }

  void count_dense_block(const int* row_dofs, int nrow_dofs, int ncol_dofs) {
    for (int i = 0; i < nrow_dofs; i++) {
      row_counts[row_dofs[i]] += ncol_dofs;
    }
  }

  void begin_fill() {
    for (int i = 0; i < nrows; i++) {
      rows[i].clear();
      rows[i].reserve(row_counts[i]);
    }
  }

  void add_dense_block(const int* row_dofs, int nrow_dofs, const int* col_dofs,
                       int ncol_dofs) {
    for (int i = 0; i < nrow_dofs; i++) {
      auto& row = rows[row_dofs[i]];
      row.insert(row.end(), col_dofs, col_dofs + ncol_dofs);
    }
  }

  template <typename T>
  void finalize(CSRMat<T>& csr) {
    csr.nrows = nrows;
    csr.rowp.resize(nrows + 1);
    csr.rowp[0] = 0;

    for (int i = 0; i < nrows; i++) {
      auto& r = rows[i];

      std::sort(r.begin(), r.end());
      r.erase(std::unique(r.begin(), r.end()), r.end());

      csr.rowp[i + 1] = csr.rowp[i] + static_cast<int>(r.size());
    }

    csr.cols.resize(csr.rowp[nrows]);

    for (int i = 0; i < nrows; i++) {
      std::copy(csr.rows[i].begin(), csr.rows[i].end(),
                csr.cols.begin() + csr.rowp[i]);
    }

    csr.data.resize(csr.rowp[nrows]);
    std::fill(csr.data, csr.data + csr.rowp[nrows]);
  }

 private:
  int nrows = 0;
  std::vector<int> row_counts;
  std::vector<std::vector<int>> rows;
};

}  // namespace xcgd

#endif  //
#pragma once

#include <ipc/math/math.hpp>
#include <ipc/utils/eigen_ext.hpp>

#include <Eigen/Core>

#include <cassert>

namespace ipc {

/// @brief A non-owning view that presents one or two column-major matrices
/// (with the same number of columns) as a single vertically concatenated
/// matrix.
template <int ncols = 3>
class VertexMatrixView {
public:
    /// @brief Construct a view concatenating two matrices vertically.
    /// @param A The top matrix.
    /// @param B The bottom matrix.
    VertexMatrixView(
        Eigen::ConstRef<Eigen::MatrixXd> A,
        Eigen::ConstRef<Eigen::MatrixXd> B)
        : n_A_rows(A.rows()),
          n_B_rows(B.rows()),
          m_A(A.data()),
          m_B(B.data())
    {
        if (A.cols() != ncols || B.cols() != ncols) {
            log_and_throw_error("Incompatible matrix columns!");
        }
    }

    /// @brief Construct a view wrapping a single matrix (no concatenation).
    explicit VertexMatrixView(Eigen::ConstRef<Eigen::MatrixXd> A)
        : n_A_rows(A.rows()),
          n_B_rows(0),
          m_A(A.data()),
          m_B(nullptr)
    {
        if (A.cols() != ncols) {
            log_and_throw_error("Incompatible matrix columns!");
        }
    }

    /// @brief Access row i of the concatenated matrix.
    Eigen::RowVector<double, ncols> operator()(index_t i) const
    {
        assert(i < rows());
        if (i < n_A_rows) {
            return Eigen::RowVector<double, ncols>(
                m_A[i + 0 * n_A_rows],
                m_A[i + 1 * n_A_rows],
                m_A[i + 2 * n_A_rows]);
        }
        else {
            i -= n_A_rows;
            return Eigen::RowVector<double, ncols>(
                m_B[i + 0 * n_B_rows],
                m_B[i + 1 * n_B_rows],
                m_B[i + 2 * n_B_rows]);
        }
    }

    /// @brief Total number of rows (A rows + B rows).
    index_t rows() const
    {
        return n_A_rows + n_B_rows;
    }

    /// @brief Number of columns (compile-time constant).
    index_t cols() const { return ncols; }

    const index_t n_A_rows;
    const index_t n_B_rows;
    const double* const m_A;
    const double* const m_B;
};

} // namespace ipc

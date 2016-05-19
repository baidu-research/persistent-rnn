#pragma once

// Forward Declarations
namespace prnn { namespace matrix { class Matrix;           } }
namespace prnn { namespace matrix { class DynamicView;      } }
namespace prnn { namespace matrix { class ConstDynamicView; } }
namespace prnn { namespace matrix { class Dimension;        } }

namespace prnn
{
namespace matrix
{

void gemm(Matrix& result, const Matrix& left, const Matrix& right);
Matrix gemm(const Matrix& left, const Matrix& right);

void gemm(Matrix& result, const Matrix& left, bool transposeLeft,
    const Matrix& right, bool transposeRight);
Matrix gemm(const Matrix& left, bool transposeLeft, const Matrix& right, bool transposeRight);

void gemm(const DynamicView& result, double beta, const ConstDynamicView& left, bool transposeLeft,
    double alpha, const ConstDynamicView& right, bool transposeRight);

void gemm(Matrix& result, double beta,
    const Matrix& left, bool transposeLeft, double alpha,
    const Matrix& right, bool transposeRight);
Matrix gemm(const Matrix& left, bool transposeLeft, double alpha,
    const Matrix& right, bool transposeRight);

}
}

